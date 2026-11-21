"""Build-time tracer: which lines, classes and files does each test touch?

Uses sys.monitoring (Python 3.12+). A line reports once per test (or per case
of a test, see harness.py) and is then switched off until the next one starts,
so tracing costs little.
"""

from __future__ import annotations

import os
import sys

mon = sys.monitoring
TOOL = mon.COVERAGE_ID
CO_OPTIMIZED = 0x1
# After this many calls in a row that show nothing new, stop looking at who a function is called on.
CALL_SAMPLE = 3000

CONSTRUCTORS = ("__init__", "__new__", "__post_init__", "__copy__", "__deepcopy__", "__setstate__", "copy")

MAIN = None  # the part of a test outside any case


class Tracer:
    def __init__(self, root: str):
        self.root = os.path.realpath(root) + os.sep
        self.code_info: dict = {}
        self.current: str | None = None
        self.case = MAIN
        self.active = False
        self.importing = 0
        self.lines: set = set()
        self.types: set = set()
        self.opens: set = set()
        self.loader_opens: set = set()
        self.calls: dict = {}
        # results: test -> case -> set
        self.test_lines: dict[str, dict] = {}
        self.test_classes: dict[str, dict] = {}
        self.test_opens: dict[str, set] = {}
        self.import_tags: dict[tuple[str, int], set] = {}
        self.ancestors: dict[str, tuple] = {}  # class -> the sqlglot/tests classes it inherits from
        # Classes whose instances are used while a module loads: class -> the modules that do so.
        self.class_tags: dict[str, set] = {}
        self.import_stack: list[str] = []
        self._tagged: set = set()
        self._phase: list[str] = []
        self._phase_data = (set(), set(), set())

        mon.use_tool_id(TOOL, "blast")
        E = mon.events
        mon.register_callback(TOOL, E.LINE, self._line)
        mon.register_callback(TOOL, E.PY_START, self._start)
        mon.register_callback(TOOL, E.PY_RETURN, self._return)
        mon.register_callback(TOOL, E.PY_UNWIND, self._unwind)
        mon.set_events(TOOL, E.LINE | E.PY_START | E.PY_UNWIND)
        sys.addaudithook(self._audit)

    # -- classification of code objects ------------------------------------
    def _info(self, code):
        info = self.code_info.get(code)
        if info is None:
            filename = code.co_filename
            rel = None
            if filename.startswith(self.root) and filename.endswith(".py"):
                rel = filename[len(self.root):]
            elif not os.path.isabs(filename):
                real = os.path.realpath(filename)
                if real.startswith(self.root):
                    rel = real[len(self.root):]
            if rel is None or not (rel.startswith("sqlglot/") or rel.startswith("tests/")):
                info = (None, 0, None)
            elif code.co_name == "<module>":
                info = (rel, 1, None)
            elif not code.co_flags & CO_OPTIMIZED:
                info = (rel, 2, None)  # class body
            else:
                first = code.co_varnames[0] if code.co_argcount and code.co_varnames else None
                first = first if first in ("self", "cls", "mcs", "klass") else None
                # Constructors are always watched: every new instance shows its class there.
                info = (rel, 4 if first and code.co_name in CONSTRUCTORS else 3, first)
            self.code_info[code] = info
        return info

    # -- callbacks -----------------------------------------------------------
    def _line(self, code, lineno):
        rel, kind, _ = self._info(code)
        if kind < 3:
            return mon.DISABLE
        if self.importing:
            self._tag(rel, lineno)
        elif self.active:
            self.lines.add((rel, lineno))
        return mon.DISABLE

    def _tag(self, rel, lineno):
        frame = sys._getframe(2)
        owner = None
        while frame is not None:
            frel, kind, _ = self._info(frame.f_code)
            if kind == 2:
                owner = ("C", frel, frame.f_code.co_qualname)
                break
            if kind == 1:
                owner = ("M", frel, "")
                break
            frame = frame.f_back
        if owner is not None:
            self.import_tags.setdefault((rel, lineno), set()).add(owner)

    def _start(self, code, offset):
        rel, kind, first = self._info(code)
        if kind == 1:
            self.importing += 1
            self.import_stack.append(rel)
            mon.set_local_events(TOOL, code, mon.events.PY_RETURN)
            mon.restart_events()
            return None
        if kind >= 3 and first is not None:
            if self.importing:
                self._tag_class(first)
                return None
            if self.active:
                try:
                    obj = sys._getframe(1).f_locals[first]
                except Exception:  # noqa: BLE001
                    return None
                kind_of = obj if isinstance(obj, type) else type(obj)
                seen = self.calls.get(code)
                if seen is None:
                    seen = self.calls[code] = [0, set()]
                if kind_of in seen[1]:
                    seen[0] += 1
                    if seen[0] >= CALL_SAMPLE and kind == 3:
                        return mon.DISABLE
                else:
                    seen[0] = 0
                    seen[1].add(kind_of)
                    self.types.add(kind_of)
            return None
        return mon.DISABLE

    def _tag_class(self, first):
        try:
            obj = sys._getframe(2).f_locals[first]
        except Exception:  # noqa: BLE001
            return
        kind_of = obj if isinstance(obj, type) else type(obj)
        owner = self.import_stack[-1] if self.import_stack else None
        if owner is None or (kind_of, owner) in self._tagged:
            return
        self._tagged.add((kind_of, owner))
        try:
            mro = kind_of.__mro__
        except Exception:  # noqa: BLE001
            return
        for c in mro:
            module = getattr(c, "__module__", "") or ""
            if module.startswith(("sqlglot", "tests")):
                # A class takes part in its own module's loading by being defined there;
                # that says nothing about who else depends on it.
                home = module.replace(".", "/")
                if owner in (home + ".py", home + "/__init__.py"):
                    continue
                self.class_tags.setdefault(f"{module}:{c.__qualname__}", set()).add(owner)

    def _left_module(self, code):
        if self._info(code)[1] == 1 and self.importing:
            self.importing -= 1
            if self.import_stack:
                self.import_stack.pop()
            mon.restart_events()

    def _return(self, code, offset, retval):
        self._left_module(code)

    def _unwind(self, code, offset, exc):
        self._left_module(code)

    def _audit(self, event, args):
        if not self.active:
            return
        if event == "open" or event in ("os.listdir", "os.scandir", "glob.glob"):
            path = args[0] if args else None
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if isinstance(path, str):
                real = os.path.realpath(path)
                if real.startswith(self.root) and not real.endswith((".py", ".pyc")):
                    self.opens.add(real[len(self.root):])

    # -- test boundaries -----------------------------------------------------
    def begin(self, test_id):
        """Start a test, or (None) the setup phase shared by a class's tests."""
        self._flush()
        if test_id is None:
            self._close_phase()
        else:
            self._phase.append(test_id)
        self.current = test_id
        self.case = MAIN
        self.active = True
        self.calls = {}
        mon.restart_events()

    def switch(self, case):
        """Move to another case of the current test."""
        if case == self.case:
            return
        self._flush()
        self.case = case
        self.calls = {}
        mon.restart_events()

    def end(self):
        self._flush()
        self.current = None
        self.case = MAIN
        self.active = True  # between tests we are back in the class phase

    def finish(self):
        self._flush()
        self._close_phase()
        self.active = False

    def _class_keys(self):
        keys = set()
        for t in self.types:
            try:
                mro = t.__mro__
            except Exception:  # noqa: BLE001
                continue
            chain = []
            for c in mro:
                module = getattr(c, "__module__", "") or ""
                if module.startswith(("sqlglot", "tests")):
                    chain.append(f"{module}:{c.__qualname__}")
            keys.update(chain)
            if chain:
                self.ancestors[chain[0]] = tuple(chain[1:])
        return keys

    def _flush(self):
        # A fixture read through a case loader is tracked per case, not as a plain read.
        lines, classes, opens = self.lines, self._class_keys(), self.opens - self.loader_opens
        self.lines, self.types, self.opens, self.loader_opens = set(), set(), set(), set()
        if self.current is None:
            self._phase_data[0].update(lines)
            self._phase_data[1].update(classes)
            self._phase_data[2].update(opens)
        else:
            self.test_lines.setdefault(self.current, {}).setdefault(self.case, set()).update(lines)
            self.test_classes.setdefault(self.current, {}).setdefault(self.case, set()).update(classes)
            self.test_opens.setdefault(self.current, set()).update(opens)

    def _close_phase(self):
        lines, classes, opens = self._phase_data
        for test_id in self._phase:
            self.test_lines.setdefault(test_id, {}).setdefault(MAIN, set()).update(lines)
            self.test_classes.setdefault(test_id, {}).setdefault(MAIN, set()).update(classes)
            self.test_opens.setdefault(test_id, set()).update(opens)
        self._phase = []
        self._phase_data = (set(), set(), set())
