"""From a patch to the tests it could break.

Everything here reads maps made from the pinned commit (see build_map.py) and
the patch itself. The answer is a bitmask over the map's test list: bit n set
means test n might fail and must be either run or selected.
"""

from __future__ import annotations

import ast
import os
import pickle
import re

from blast import pyindex

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
CONSTANT = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
BLOCKS = ("body", "orelse", "finalbody")
COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.TryStar) + FUNC


class Maps:
    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "core.pkl"), "rb") as f:
            core = pickle.load(f)
        self.ids: list[str] = core["ids"]
        self.bad = set(core["bad"])
        self.seconds: list[float] = core["seconds"]
        self.overhead: dict[str, float] = core.get("overhead", {})
        self.classes: dict[str, int] = core["classes"]
        self.class_cases: dict[str, dict[int, int]] = core.get("class_cases", {})
        self.subclasses: dict[str, list[str]] = core.get("subclasses", {})
        self.class_tags: dict[str, list[str]] = core.get("class_tags", {})
        self._class_tag_users: dict[str, int] = {}
        # test index -> its cases, in bit order (bit 0 is the test outside any case)
        self.test_cases: dict[int, list] = core.get("test_cases", {})
        self.label_tests: dict[str, int] = {}
        for n, cases in self.test_cases.items():
            for label, _ in cases:
                self.label_tests[label] = self.label_tests.get(label, 0) | (1 << n)
        # Per commit: which cases of a test the reasons found so far point at, and
        # which tests were picked for a reason that says nothing about cases.
        self.case_bits: dict[int, int] = {}
        self.whole = 0
        # Modules whose patched version should be imported once, to see that it still loads.
        self.smoke: set[str] = set()
        # True when the patch can change state the whole process shares (what gets
        # imported, what is registered or cached). Test results then depend on what ran
        # before, so candidates are run all together in suite order or not at all.
        self.shared_state = False
        self.opens: dict[str, int] = core["opens"]
        self.imports: dict[str, list[str]] = core["imports"]
        self.core_modules = set(core.get("core_modules", ()))
        # Slow tests with a scaled-down stand-in that passes at the pinned commit: id -> seconds
        self.scaled: dict[str, float] = core.get("scaled", {})
        self.case_seconds: dict[str, dict] = core.get("case_seconds", {})
        self.everything = (1 << len(self.ids)) - 1
        self._files: dict[str, dict | None] = {}
        self._refs = None
        self._importers = None
        self._module_users: dict[str, int] = {}
        self._tests_in: dict[str, int] | None = None

    def reset(self) -> None:
        self.case_bits = {}
        self.whole = 0
        self.smoke = set()
        self.shared_state = False

    def _class(self, key: str) -> int:
        mask = self.classes.get(key, 0)
        if mask:
            for n, bits_ in self.class_cases.get(key, {}).items():
                self.case_bits[n] = self.case_bits.get(n, 0) | bits_
        if key in self.class_tags:
            # Instances are also used while some modules load: whatever those modules
            # build then depends on the class, and so does everything that needs them.
            if key not in self._class_tag_users:
                loaded = 0
                for owner in self.class_tags[key]:
                    loaded |= self.isolated(lambda: self.importer_users(owner))[0]
                self._class_tag_users[key] = loaded
            loaded = self._class_tag_users[key]
            self.whole |= loaded
            mask |= loaded
        return mask

    def file(self, rel: str) -> dict | None:
        if rel not in self._files:
            path = os.path.join(self.directory, "files", rel.replace("/", "__") + ".pkl")
            try:
                with open(path, "rb") as f:
                    self._files[rel] = pickle.load(f)
            except OSError:
                self._files[rel] = None
        return self._files[rel]

    @property
    def refs(self) -> dict:
        if self._refs is None:
            with open(os.path.join(self.directory, "refs.pkl"), "rb") as f:
                self._refs = pickle.load(f)
        return self._refs

    def tests_in(self, rel: str) -> int:
        if self._tests_in is None:
            self._tests_in = {}
            for n, test_id in enumerate(self.ids):
                path = test_id.split("::", 1)[0]
                self._tests_in[path] = self._tests_in.get(path, 0) | (1 << n)
        return self._tests_in.get(rel, 0)

    # -- who uses what ---------------------------------------------------------
    def lines(self, rel: str, first: int, last: int, _seen=None) -> int:
        """Tests that execute any function line in [first, last] of the old file."""
        data = self.file(rel)
        if not data:
            return 0
        mask = 0
        covered, tags, cases, lazy = data["lines"], data["tags"], data.get("cases", {}), data.get("lazy", ())
        for line in range(first, last + 1):
            mask |= covered.get(line, 0)
            if line in lazy:
                # Runs once and its result is kept: every later case may lean on it.
                self.whole |= covered.get(line, 0)
                self.shared_state = True
            elif line in cases:
                for n, bits_ in cases[line].items():
                    self.case_bits[n] = self.case_bits.get(n, 0) | bits_
            if line in tags:
                self.shared_state = True
                seen = _seen if _seen is not None else set()
                for kind, owner_rel, qual in tags[line]:
                    # The line ran while a module was being imported: whatever it
                    # built lives on in that module or class.
                    if (kind, owner_rel, qual) in seen:
                        continue
                    seen.add((kind, owner_rel, qual))
                    if kind == "C":
                        mask |= self.class_users(owner_rel, qual)
                    else:
                        mask |= self.importer_users(owner_rel)
        return mask

    def class_users(self, rel: str, qual: str, constant: bool = False) -> int:
        """Tests that call a method on an instance (or subclass) of the class, or of a class around it.

        With `constant`, also tests that run a line naming the class or one of its
        subclasses: a constant can be read straight off the class, `DataType.TEXT_TYPES`.
        """
        module = pyindex.module_name(rel)
        mask = 0
        parts = qual.split(".")
        for depth in range(len(parts), 0, -1):
            mask |= self._class(f"{module}:{'.'.join(parts[:depth])}")
        if not self.classes.get(f"{module}:{qual}", 0):
            # Never seen as `self` (an enum, a namespace): fall back to who names it.
            mask |= self.name_users(rel, {parts[0]})
        elif constant:
            names = {parts[0]}
            for key in self.subclasses.get(f"{module}:{qual}", ()):
                names.add(key.split(":", 1)[1].split(".")[0])
            if len(names) > 40:
                return self.everything_whole()
            for name in names:
                for owner_rel, index, line, mode in self.refs.get(name, ()):
                    if mode == pyindex.WHOLE:
                        _, _, _, start, end, _ = self.file(owner_rel)["units"][index]
                        mask |= self.lines(owner_rel, start, end)
                    elif mode != pyindex.DIRECT:
                        mask |= self.lines(owner_rel, line, line)
        return mask

    def everything_whole(self) -> int:
        self.whole = self.everything
        return self.everything

    def module_users(self, rel: str) -> int:
        if rel not in self._module_users:
            data = self.file(rel)
            mask = self.tests_in(rel)
            if data:
                for line_mask in set(data["lines"].values()):
                    mask |= line_mask
                module = pyindex.module_name(rel)
                for kind, scope, name, *_ in data["units"]:
                    if kind == "H":
                        qual = f"{scope}.{name}" if scope else name
                        mask |= self.classes.get(f"{module}:{qual}", 0)
            self._module_users[rel] = mask
        return self._module_users[rel]

    def importer_users(self, rel: str) -> int:
        """Tests that use the module or anything that imports it, directly or not."""
        self.shared_state = True
        if self._importers is None:
            self._importers = {}
            for importer, imported in self.imports.items():
                for target in imported:
                    self._importers.setdefault(target, set()).add(importer)
        mask = 0
        seen = {rel}
        stack = [rel]
        while stack:
            current = stack.pop()
            mask |= self.module_users(current)
            if mask == self.everything:
                break
            for importer in self._importers.get(current, ()):
                if importer not in seen:
                    seen.add(importer)
                    stack.append(importer)
        self.whole |= mask
        return mask

    def name_users(self, rel: str, names, removed: bool = False) -> int:
        """Tests that run code naming any of these module-level names."""
        mask = 0
        seen_refs: set = set()
        seen_names: set = set()
        work = list(names)
        while work:
            name = work.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            for ref in self.refs.get(name, ()):
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                if len(seen_refs) > 6000:
                    self.whole = self.everything
                    return self.everything
                owner_rel, index, line, mode = ref
                kind, scope, unit_name, start, end, bound = self.file(owner_rel)["units"][index]
                if mode == pyindex.WHOLE:
                    mask |= self.lines(owner_rel, start, end)
                elif mode != pyindex.DIRECT:
                    mask |= self.lines(owner_rel, line, line)
                elif scope:
                    mask |= self.classes_only(owner_rel, scope if kind == "S" else (f"{scope}.{unit_name}"))
                elif kind == "H":
                    mask |= self.classes_only(owner_rel, unit_name)
                else:
                    if removed and owner_rel != rel:
                        # `from module import name` now raises: the importer cannot load.
                        mask |= self.importer_users(owner_rel)
                    if name in bound:
                        pass  # an import or re-export of the same name: nothing new to follow
                    elif bound:
                        work.extend(bound)
                    else:
                        mask |= self.importer_users(owner_rel)
                if mask == self.everything:
                    return mask
        return mask

    def attribute_readers(self, names) -> int | None:
        """Tests that execute a line reading one of these class attributes, or something
        derived from one. None when the trail cannot be followed."""
        mask = 0
        seen_names: set = set()
        work = list(names)
        while work:
            name = work.pop()
            if name in seen_names:
                continue
            seen_names.add(name)
            # Only constants are read by their own name; methods and the like are also
            # reached through getattr with a computed name, which leaves no trail here.
            if len(seen_names) > 12 or not CONSTANT.match(name):
                return None
            found = False
            for owner_rel, index, line, mode in self.refs.get(name, ()):
                found = found or mode != pyindex.DIRECT or name not in self.file(owner_rel)["units"][index][5]
                data = self.file(owner_rel)
                kind, scope, unit_name, start, end, bound = data["units"][index]
                if mode == pyindex.WHOLE:
                    mask |= self.lines(owner_rel, start, end)
                elif mode != pyindex.DIRECT:
                    if line in data["tags"] and mode:
                        # Read while a module loads and stored under other names: follow those.
                        mask |= data["lines"].get(line, 0)
                        self.whole |= data["lines"].get(line, 0)
                        work.extend(mode)
                    else:
                        mask |= self.lines(owner_rel, line, line)
                elif name in bound:
                    continue  # the attribute's own definition, here or in a subclass
                elif bound and kind == "S":
                    work.extend(bound)
                else:
                    return None
            if not found:
                return None  # nothing reads it by name
        return mask

    def isolated(self, compute):
        """Run compute() with its own case bookkeeping: (mask, case_bits, whole)."""
        saved = self.case_bits, self.whole
        self.case_bits, self.whole = {}, 0
        try:
            mask = compute()
            return mask, self.case_bits, self.whole
        finally:
            self.case_bits, self.whole = saved

    def absorb(self, mask: int, case_bits: dict, whole: int) -> None:
        self.whole |= whole & mask
        for n, bits_ in case_bits.items():
            if mask >> n & 1:
                self.case_bits[n] = self.case_bits.get(n, 0) | bits_

    def class_attribute_users(self, rel: str, qual: str, names) -> int:
        """Tests that use the class and also read the attribute."""
        constant = all(CONSTANT.match(name) for name in names)
        users, user_bits, user_whole = self.isolated(lambda: self.class_users(rel, qual, constant))
        readers, reader_bits, reader_whole = self.isolated(lambda: self.attribute_readers(names))
        if readers is None:
            self.absorb(users, user_bits, user_whole)
            return users
        mask = users & readers
        for n in self.test_cases:
            if not mask >> n & 1:
                continue
            a = None if user_whole >> n & 1 else user_bits.get(n)
            b = None if reader_whole >> n & 1 else reader_bits.get(n)
            if a is None and b is None:
                self.whole |= 1 << n
            else:
                both = b if a is None else a if b is None else (a & b) or b
                self.case_bits[n] = self.case_bits.get(n, 0) | both
        return mask

    def classes_only(self, rel: str, qual: str) -> int:
        module = pyindex.module_name(rel)
        mask = 0
        parts = qual.split(".")
        for depth in range(len(parts), 0, -1):
            mask |= self._class(f"{module}:{'.'.join(parts[:depth])}")
        return mask


# -- comparing two versions of a unit ----------------------------------------------

def _dump(node) -> str:
    return ast.dump(node, annotate_fields=False)


def _parse_slice(lines: list[str], start: int, end: int, header_only: bool = False):
    """Parse old lines [start, end] on their own. Returns (node, line offset)."""
    chunk = lines[start - 1 : end]
    indent = len(chunk[0]) - len(chunk[0].lstrip())
    text = "\n".join(chunk)
    if header_only:
        text += "\n" + " " * (indent + 4) + "pass"
    if indent:
        tree = ast.parse("if 1:\n" + text)
        return tree.body[0].body[0], start - 2
    return ast.parse(text).body[0], start - 1


def _scope_signature(node) -> tuple:
    stores, flags = set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            stores.add(n.id)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            flags.add((type(n).__name__, tuple(n.names)))
        elif isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)):
            flags.add(type(n).__name__)
        elif isinstance(n, FUNC + (ast.ClassDef,)) and n is not node:
            stores.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            stores.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.arg):
            stores.add(n.arg)
    return frozenset(stores), frozenset(flags)


def _header(node) -> str:
    parts = []
    for field, value in ast.iter_fields(node):
        if field in BLOCKS or field == "handlers":
            continue
        parts.append(_dump(value) if isinstance(value, ast.AST) else repr(
            [_dump(v) if isinstance(v, ast.AST) else v for v in value] if isinstance(value, list) else value
        ))
    return "|".join(parts)


def _first_line(node) -> int:
    decorators = getattr(node, "decorator_list", None)
    return min([node.lineno] + [d.lineno for d in decorators]) if decorators else node.lineno


def _diff_body(old: list, new: list, ranges: list) -> bool:
    """Collect old line ranges of changed statements. False when the shapes differ too much."""
    if len(old) != len(new):
        return False
    for o, n in zip(old, new):
        if type(o) is not type(n):
            return False
        if _dump(o) == _dump(n):
            continue
        if isinstance(o, COMPOUND):
            if _header(o) != _header(n):
                body_start = o.body[0].lineno if o.body else o.end_lineno
                ranges.append((_first_line(o), max(o.lineno, body_start - 1)))
                if isinstance(o, FUNC):
                    # A nested function's signature: everyone who calls it.
                    ranges.append((_first_line(o), o.end_lineno))
            for field in BLOCKS:
                if not _diff_body(getattr(o, field, []) or [], getattr(n, field, []) or [], ranges):
                    return False
            old_handlers, new_handlers = getattr(o, "handlers", []), getattr(n, "handlers", [])
            if len(old_handlers) != len(new_handlers):
                return False
            for oh, nh in zip(old_handlers, new_handlers):
                if _dump(oh.type) if oh.type else None != (_dump(nh.type) if nh.type else None) or oh.name != nh.name:
                    ranges.append((oh.lineno, oh.lineno))
                if not _diff_body(oh.body, nh.body, ranges):
                    return False
        elif isinstance(o, (ast.Match, ast.ClassDef)):
            return False
        else:
            ranges.append((o.lineno, o.end_lineno))
    return True


def _lambda_ranges(old, new, ranges: list) -> bool:
    """True when old and new differ only inside lambda bodies; collects those bodies' lines."""
    if type(old) is not type(new):
        return False
    if isinstance(old, ast.Lambda):
        if _dump(old.args) != _dump(new.args):
            return False
        if _dump(old.body) != _dump(new.body):
            ranges.append((old.lineno, old.end_lineno))
        return True
    for (field, a), (_, b) in zip(ast.iter_fields(old), ast.iter_fields(new)):
        if isinstance(a, ast.AST):
            if not isinstance(b, ast.AST) or not _lambda_ranges(a, b, ranges):
                return False
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if isinstance(x, ast.AST):
                    if not isinstance(y, ast.AST) or not _lambda_ranges(x, y, ranges):
                        return False
                elif x != y:
                    return False
        elif a != b:
            return False
    return True


def _drop_annotations(node) -> None:
    """With `from __future__ import annotations`, annotations are never evaluated."""
    for n in ast.walk(node):
        if isinstance(n, ast.arg):
            n.annotation = None
        elif isinstance(n, FUNC):
            n.returns = None
        elif isinstance(n, ast.AnnAssign):
            n.annotation = ast.Constant(value=None)


def _is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def python_change(maps: Maps, rel: str, old_text: str | None, new_text: str | None, notes: list) -> int:
    """Tests a change to one Python file could break."""
    if old_text is None:
        # A new module changes nothing until something imports it, and that edit is analysed on its own.
        return 0
    if new_text is None:
        notes.append(f"{rel}: deleted")
        return maps.importer_users(rel)
    data = maps.file(rel)
    if data is None:
        notes.append(f"{rel}: not in the map")
        maps.whole = maps.everything
        return maps.everything
    try:
        new_tree = ast.parse(new_text)
        compile(new_tree, rel, "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        notes.append(f"{rel}: does not compile")
        return maps.importer_users(rel)

    lazy_annotations = "from __future__ import annotations" in old_text and "from __future__ import annotations" in new_text
    if lazy_annotations:
        new_text_units = pyindex.units_of(new_tree)  # spans first: the tree is edited next
        _drop_annotations(new_tree)
    else:
        new_text_units = pyindex.units_of(new_tree)
    old_lines = old_text.split("\n")
    new_lines = new_text.split("\n")
    old_units: dict[tuple, list] = {}
    for kind, scope, name, start, end, bound in data["units"]:
        old_units.setdefault((kind, scope, name), []).append((start, end, bound))
    new_units: dict[tuple, list] = {}
    for unit in new_text_units:
        new_units.setdefault(unit.key, []).append(unit)

    mask = 0

    def class_or_names(scope: str, names, removed=False) -> int:
        if scope:
            return maps.class_users(rel, scope)
        if names:
            return maps.name_users(rel, set(names), removed=removed)
        return maps.importer_users(rel)

    for key in set(old_units) | set(new_units):
        kind, scope, name = key
        olds, news = old_units.get(key, []), new_units.get(key, [])
        for start, end, bound in olds[len(news):]:
            notes.append(f"{rel}:{start} removed {kind} {scope}.{name}")
            if kind == "F":
                mask |= maps.lines(rel, start, end)
            if kind == "H":
                mask |= maps.class_users(rel, f"{scope}.{name}" if scope else name)
                mask |= maps.name_users(rel, {name}, removed=True) if not scope else 0
            else:
                mask |= class_or_names(scope, bound, removed=True)
        for unit in news[len(olds):]:
            notes.append(f"{rel}:{unit.start} added {kind} {scope}.{name}")
            if kind == "S" and _is_docstring(unit.node):
                continue
            if kind == "H":
                continue
            if not scope and rel.startswith("sqlglot/"):
                maps.smoke.add(rel)  # new module-level code: it must at least still import
            mask |= class_or_names(scope, unit.names)
        for (start, end, bound), unit in zip(olds, news):
            if old_lines[start - 1 : end] == new_lines[unit.start - 1 : unit.end]:
                continue
            try:
                old_node, offset = _parse_slice(old_lines, start, end, header_only=kind == "H")
            except (SyntaxError, ValueError, IndexError, AttributeError):
                notes.append(f"{rel}:{start} could not isolate {kind} {name}")
                mask |= maps.lines(rel, start, end) | class_or_names(scope, bound)
                continue
            new_node = unit.node
            if lazy_annotations:
                _drop_annotations(old_node)
            if kind == "H":
                old_head = [_dump(x) for x in old_node.bases + old_node.keywords + old_node.decorator_list]
                new_head = [_dump(x) for x in new_node.bases + new_node.keywords + new_node.decorator_list]
                if old_head != new_head:
                    notes.append(f"{rel}:{start} class header {name}")
                    mask |= maps.class_users(rel, f"{scope}.{name}" if scope else name)
                continue
            if _dump(old_node) == _dump(new_node):
                continue  # comments or layout only
            if kind == "F":
                whole = False
                head_old = _dump(old_node.args) + "".join(_dump(d) for d in old_node.decorator_list)
                head_new = _dump(new_node.args) + "".join(_dump(d) for d in new_node.decorator_list)
                if head_old != head_new or type(old_node) is not type(new_node):
                    notes.append(f"{rel}:{start} signature of {name}")
                    mask |= class_or_names(scope, {name})
                    whole = True
                ranges: list = []
                if not whole and _scope_signature(old_node) != _scope_signature(new_node):
                    whole = True
                if not whole and not _diff_body(old_node.body, new_node.body, ranges):
                    whole = True
                if whole:
                    notes.append(f"{rel}:{start} function {name} (whole)")
                    mask |= maps.lines(rel, start, end)
                else:
                    for first, last in ranges:
                        notes.append(f"{rel}:{first + offset}-{last + offset} in {name}")
                        mask |= maps.lines(rel, first + offset, last + offset)
            else:
                if _is_docstring(old_node) and _is_docstring(new_node):
                    continue
                ranges = []
                if _lambda_ranges(old_node, new_node, ranges) and ranges:
                    for first, last in ranges:
                        notes.append(f"{rel}:{first + offset}-{last + offset} lambda in {scope or 'module'}")
                        mask |= maps.lines(rel, first + offset, last + offset)
                else:
                    notes.append(f"{rel}:{start} statement {name} in {scope or 'module'}")
                    if not scope:
                        maps.shared_state = True
                    if not scope and rel.startswith("sqlglot/"):
                        maps.smoke.add(rel)
                    mask |= maps.lines(rel, start, end)
                    names = set(bound) | unit.names
                    plain = isinstance(old_node, (ast.Assign, ast.AnnAssign)) and isinstance(
                        new_node, (ast.Assign, ast.AnnAssign)
                    )
                    if scope and plain and names:
                        mask |= maps.class_attribute_users(rel, scope, names)
                    else:
                        mask |= class_or_names(scope, names)
    return mask


def other_change(maps: Maps, rel: str | None, new_rel: str | None) -> int:
    """Tests that read a non-Python file, or list the directory it lives in."""
    mask = 0
    for path in {rel, new_rel} - {None}:
        mask |= maps.opens.get(path, 0)
    if rel is None or new_rel is None or rel != new_rel:
        for path in {rel, new_rel} - {None}:
            mask |= maps.opens.get(os.path.dirname(path), 0)
    maps.whole |= mask
    # Tests that read the file one case at a time: only the cases the patch changes matter,
    # and those are found when the patched file is loaded.
    for path in {rel, new_rel} - {None}:
        by_case = maps.label_tests.get(path, 0)
        mask |= by_case
        n = 0
        while by_case >> n:
            if by_case >> n & 1:
                maps.case_bits.setdefault(n, 0)
            n += 1
    return mask
