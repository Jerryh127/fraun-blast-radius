"""Runs sqlglot tests in-process with plain unittest and reports what failed.

Used twice: at image build time (with tracing on) to learn which source lines,
classes and fixture files each test touches at the pinned commit, and at run
time to execute a commit's candidate tests against the patched tree.

A test id is the pytest node id: tests/x.py::Class::method.

Cases
-----
Some slow tests are loops over independent cases: one per statement of a
fixture file, or one per dialect. The loop sources are wrapped here so that
the tracer can record coverage per case, and so that a later run can replay
only the cases a patch can reach. Code outside the loop always runs.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib
import inspect
import io
import os
import pickle
import sys
import textwrap
import time
import unittest

# Fixture loaders in tests/helpers.py whose items are independent cases.
LOADERS = ("load_sql_fixture_pairs", "load_sql_fixtures")
# Module globals that a test iterates, one independent case per item.
GLOBAL_CASES = {"tests.test_generator": ("Dialects",)}
# These tests index back into the case list, so a skipped case must keep its
# slot: it is replaced by a statement the test itself ignores.
KEEP_SLOTS = {"test_execute_tpch", "test_execute_tpcds"}
PLACEHOLDER = ({}, "SELECT 1", "SELECT 1")
# A stand-in for a test too slow to run while deciding: the same test body with
# smaller numbers. (old literal, new literal) pairs are swapped in its source.
SCALED = {
    "tests/test_parser.py::TestParser::test_max_nodes": (
        (10000, 300),
        (1_000_000, 20_000),
        (50_000, 5_000),
        ("(50000)", "(5000)"),
    ),
}


class Control:
    """What the case wrappers consult: the current test, its filter, the tracer."""

    def __init__(self):
        self.tracer = None
        self.filters: dict[str, dict[str, frozenset] | None] = {}
        self.changed_fixtures: dict[str, str] = {}  # label -> fixtures directory of the pinned commit
        self.replay = False  # True at run time: filters apply and a failed test stops early
        self.test: str | None = None
        self.method: str | None = None
        self.failed = False
        self.materialised: list[tuple[str, list]] = []
        self.setup_seconds = 0.0
        # Build-time timing of each case: test -> case -> seconds (None is the rest of the test)
        self.timing = False
        self.case_seconds: dict[str, dict] = {}
        self._case = None
        self._since = 0.0
        self.changed_cases: dict[str, frozenset] = {}

    def _clock(self, case) -> None:
        now = time.perf_counter()
        if self.test is not None:
            spent = self.case_seconds.setdefault(self.test, {})
            spent[self._case] = spent.get(self._case, 0.0) + now - self._since
        self._case, self._since = case, now

    def begin(self, test_id: str | None) -> None:
        if self.timing:
            self._clock(None)
        self.test = test_id
        self.method = test_id.rsplit("::", 1)[1] if test_id else None
        self.failed = False

    def allowed(self, label: str):
        """Case indexes to replay for the current test, or None for all of them."""
        if not self.replay or self.test is None:
            return None
        spec = self.filters.get(self.test)
        if spec is None:
            return None
        return spec.get(label, frozenset()) | self.changed_cases.get(label, frozenset())

    def enter(self, label: str, index: int) -> None:
        if self.test is None:
            return
        if self.tracer is not None:
            self.tracer.switch((label, index))
        if self.timing:
            self._clock((label, index))

    def enter_key(self, key) -> None:
        if self.test is None or key is None:
            return
        if self.tracer is not None:
            self.tracer.switch(key)
        if self.timing:
            self._clock(key)

    @property
    def stop(self) -> bool:
        return self.replay and self.failed and self.test is not None


CONTROL = Control()


class CaseList(list):
    """A fixture's statements kept as a list on a test class (see TestExecutor.setUpClass)."""

    label = ""

    def __iter__(self):
        allowed = CONTROL.allowed(self.label)
        keep = CONTROL.method in KEEP_SLOTS
        for index in range(len(self)):
            if (allowed is not None and index not in allowed) or CONTROL.stop:
                if keep:
                    yield PLACEHOLDER
                continue
            CONTROL.enter(self.label, index)
            yield list.__getitem__(self, index)


class CaseProxy:
    """Stands in for a module global that tests iterate (an enum of dialects)."""

    def __init__(self, target, label):
        self._target = target
        self._label = label

    def __iter__(self):
        allowed = CONTROL.allowed(self._label)
        for index, item in enumerate(self._target):
            if (allowed is not None and index not in allowed) or CONTROL.stop:
                continue
            CONTROL.enter(self._label, index)
            yield item

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __getitem__(self, key):
        return self._target[key]

    def __len__(self):
        return len(self._target)

    def __contains__(self, item):
        return item in self._target

    def __call__(self, *args, **kwargs):
        return self._target(*args, **kwargs)


def _wrap_loader(helpers, name):
    original = getattr(helpers, name)
    if getattr(original, "_blast", False):
        return

    def changed_cases(filename, label):
        # The patch edits this fixture: the cases that differ from the pinned file must run.
        new_items = list(original(filename))
        current = helpers.FIXTURES_DIR
        helpers.FIXTURES_DIR = CONTROL.changed_fixtures[label]
        try:
            old_items = list(original(filename))
        except Exception:  # noqa: BLE001
            old_items = None
        finally:
            helpers.FIXTURES_DIR = current
        if old_items is None or len(old_items) != len(new_items):
            return frozenset(range(len(new_items)))
        return frozenset(i for i, (a, b) in enumerate(zip(old_items, new_items)) if a != b)

    def loader(filename):
        label = "tests/fixtures/" + filename
        if CONTROL.replay and label in CONTROL.changed_fixtures and label not in CONTROL.changed_cases:
            CONTROL.changed_cases[label] = changed_cases(filename, label)
        allowed = CONTROL.allowed(label)
        produced: list = []
        complete = True
        if CONTROL.tracer is not None:
            CONTROL.tracer.loader_opens.add(label)
        for index, item in enumerate(original(filename)):
            if (allowed is not None and index not in allowed) or CONTROL.stop:
                complete = False
                continue
            CONTROL.enter(label, index)
            produced.append(item)
            yield item
        if complete and CONTROL.test is None:
            CONTROL.materialised.append((label, produced))

    loader._blast = True  # type: ignore[attr-defined]
    setattr(helpers, name, loader)


def _adopt_case_lists(cls) -> None:
    """Turn lists that setUpClass filled from a fixture loader into CaseLists."""
    for name, value in list(vars(cls).items()):
        if type(value) is not list:
            continue
        for label, items in CONTROL.materialised:
            if len(items) == len(value) and all(a is b for a, b in zip(items, value)):
                wrapped = CaseList(value)
                wrapped.label = label
                setattr(cls, name, wrapped)
                break


class _Future(concurrent.futures.Future):
    case = None

    def result(self, timeout=None):
        CONTROL.enter_key(self.case)
        return super().result(timeout)


class SyncPool:
    """In-process stand-in for ProcessPoolExecutor.

    sqlglot's optimizer and executor tests farm work out to spawned worker
    processes. Spawning costs far more CPU than the work, so tasks run here
    instead. Arguments and results still go through pickle, because that round
    trip is part of what those tests exercise.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        self._initializer = initializer
        self._initargs = initargs
        self._ready = False

    def submit(self, fn, /, *args, **kwargs):
        future = _Future()
        tracer = CONTROL.tracer
        future.case = tracer.case if tracer is not None else CONTROL._case if CONTROL.timing else None
        try:
            if not self._ready:
                self._ready = True
                if self._initializer is not None:
                    self._initializer(*self._initargs)
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            result = fn(*args, **kwargs)
            future.set_result(pickle.loads(pickle.dumps(result)))
        except BaseException as error:  # noqa: BLE001 - a worker's error belongs to the future
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        return (f.result() for f in futures)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_installed = False


def _limit_duckdb_threads() -> None:
    """DuckDB starts a thread per host core for every connection; the tests need one or two."""
    try:
        import duckdb
    except Exception:  # noqa: BLE001
        return
    original = duckdb.connect
    if getattr(original, "_blast", False):
        return

    def connect(*args, **kwargs):
        config = dict(kwargs.get("config") or {})
        config.setdefault("threads", 2)
        kwargs["config"] = config
        try:
            return original(*args, **kwargs)
        except TypeError:
            kwargs.pop("config", None)
            return original(*args, **kwargs)

    connect._blast = True  # type: ignore[attr-defined]
    duckdb.connect = connect


def install_patches() -> None:
    """Call with the checkout on sys.path, before any test module is imported."""
    global _installed
    if _installed:
        return
    _installed = True
    concurrent.futures.ProcessPoolExecutor = SyncPool  # type: ignore[misc]
    import concurrent.futures.process as process

    process.ProcessPoolExecutor = SyncPool  # type: ignore[misc]
    _limit_duckdb_threads()
    try:
        helpers = importlib.import_module("tests.helpers")
    except BaseException as error:  # noqa: BLE001 - a broken helpers module fails the tests that need it
        if isinstance(error, KeyboardInterrupt):
            raise
        return
    for name in LOADERS:
        if hasattr(helpers, name):
            _wrap_loader(helpers, name)


def _prepare_module(module) -> None:
    for name in GLOBAL_CASES.get(module.__name__, ()):
        target = getattr(module, name, None)
        if target is not None and not isinstance(target, CaseProxy):
            setattr(module, name, CaseProxy(target, f"{module.__name__}.{name}"))


def _prepare_class(cls) -> None:
    if "_blast_setup" in vars(cls):
        return
    cls._blast_setup = True
    original = cls.setUpClass

    def setUpClass(klass):
        t0 = time.perf_counter()
        original()
        _adopt_case_lists(klass)
        CONTROL.setup_seconds = time.perf_counter() - t0

    cls.setUpClass = classmethod(setUpClass)


def _scale(cls, method: str, swaps) -> None:
    """Replace a slow test method by the same code with smaller literals."""
    function = getattr(cls, method)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    table = dict(swaps)

    class Swap(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) in (int, str) and node.value in table:
                return ast.copy_location(ast.Constant(table[node.value]), node)
            return node

    tree = ast.fix_missing_locations(Swap().visit(tree))
    namespace: dict = {}
    exec(compile(tree, inspect.getsourcefile(function) or "<scaled>", "exec"), function.__globals__, namespace)
    setattr(cls, method, namespace[method])


def split_id(test_id: str) -> tuple[str, str, str]:
    path, cls, method = test_id.split("::")
    return path[:-3].replace("/", "."), cls, method


class _Result(unittest.TestResult):
    """Records, per test method, whether anything went wrong."""

    def __init__(self, on_start=None, on_stop=None):
        super().__init__()
        self.bad: set[str] = set()
        self.holder_errors: list[str] = []
        self.durations: dict[str, float] = {}
        self._on_start = on_start
        self._on_stop = on_stop
        self._t0 = 0.0

    @staticmethod
    def _key(test) -> str | None:
        test = getattr(test, "test_case", test)  # subtests point at their parent
        method = getattr(test, "_testMethodName", None)
        if method is None:
            return None
        cls = type(test)
        return f"{cls.__module__}::{cls.__qualname__}::{method}"

    def _mark(self, test) -> None:
        key = self._key(test)
        if key is None:
            self.holder_errors.append(str(test))
        else:
            self.bad.add(key)
            CONTROL.failed = True

    def startTest(self, test):
        super().startTest(test)
        if self._on_start:
            self._on_start(self._key(test))
        self._t0 = time.perf_counter()

    def stopTest(self, test):
        key = self._key(test)
        self.durations[key] = time.perf_counter() - self._t0
        if self._on_stop:
            self._on_stop(key, key in self.bad)
        super().stopTest(test)

    def addError(self, test, err):
        self._mark(test)

    def addFailure(self, test, err):
        self._mark(test)

    def addUnexpectedSuccess(self, test):
        self._mark(test)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._mark(test)


def run(test_ids, report, tracer=None, started=None, class_done=None, scaled=()):
    """Run tests grouped by class.

    `report(test_id, failed, seconds)` is called as each test finishes (a test
    may be reported twice; any failed report wins). `started(test_id)` is
    called before each test, so a caller can tell which test hung.
    """
    CONTROL.tracer = tracer
    install_patches()
    groups: dict[tuple[str, str], list[str]] = {}
    for test_id in test_ids:
        module, cls, method = split_id(test_id)
        groups.setdefault((module, cls), []).append(test_id)

    for (module_name, cls_name), ids in groups.items():
        CONTROL.begin(None)
        if tracer:
            tracer.begin(None)
        try:
            module = importlib.import_module(module_name)
            _prepare_module(module)
            cls = getattr(module, cls_name)
            _prepare_class(cls)
            for i in ids:
                if i in scaled and i in SCALED:
                    _scale(cls, split_id(i)[2], SCALED[i])
        except BaseException as error:  # noqa: BLE001 - an import error fails the whole file
            if isinstance(error, KeyboardInterrupt):
                raise
            for i in ids:
                report(i, True, 0.0)
            continue

        # A test method the patch removed is not collected, so it cannot fail.
        cases = []
        for i in list(ids):
            if callable(getattr(cls, split_id(i)[2], None)):
                cases.append(cls(split_id(i)[2]))
            else:
                ids.remove(i)
                report(i, False, 0.0)
        by_key = {f"{module_name}::{cls_name}::{split_id(i)[2]}": i for i in ids}
        done: set[str] = set()

        def on_start(key):
            CONTROL.begin(by_key.get(key))
            if tracer:
                tracer.begin(by_key.get(key))
            if started and key in by_key:
                started(by_key[key])

        def on_stop(key, failed):
            if tracer:
                tracer.end()
            CONTROL.begin(None)
            if key in by_key:
                done.add(key)
                report(by_key[key], failed, result.durations.get(key, 0.0))

        result = _Result(on_start, on_stop)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        try:
            unittest.TestSuite(cases).run(result)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
        # A failing setUpClass/tearDownClass fails the tests of the class, as pytest reports it.
        for key, test_id in by_key.items():
            if result.holder_errors or key not in done:
                report(test_id, True, result.durations.get(key, 0.0))
        if tracer:
            tracer.end()
        if class_done:
            class_done(f"{module_name}::{cls_name}", CONTROL.setup_seconds)
            CONTROL.setup_seconds = 0.0


def main() -> None:
    """python harness.py <root> < ids ; prints `F|P <seconds> <id>` lines to stdout."""
    root = sys.argv[1]
    sys.path.insert(0, root)
    os.chdir(root)
    ids = [line.strip() for line in sys.stdin if line.strip()]
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    def report(test_id, failed, seconds):
        out.write(f"{'F' if failed else 'P'} {seconds:.4f} {test_id}\n")
        out.flush()

    run(ids, report)


if __name__ == "__main__":
    main()
