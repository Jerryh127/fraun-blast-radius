"""Builds the maps the selector reads, from sqlglot at the pinned commit.

    python -m blast.build_map <sqlglot checkout> <output dir> [--jobs N] [--ids FILE]

Two passes over the whole suite:

1. Plain: every test file in one process, to record how long each test takes
   here and which tests already fail at the pinned commit.
2. Traced: every test in a process of its own, forked from a parent that has
   sqlglot imported. Starting each test from the same clean state matters:
   sqlglot fills caches lazily, and a test that finds a cache already filled
   would not be seen executing the code that fills it.

Nothing here knows about any patch.
"""

from __future__ import annotations

import ast
import importlib
import os
import pickle
import pkgutil
import subprocess
import sys
import time
import unittest

from blast import pyindex

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover(root: str) -> list[str]:
    sys.path.insert(0, root)
    ids = []

    def walk(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                walk(item)
            else:
                cls = type(item)
                method = getattr(item, "_testMethodName", None)
                if method is None or cls.__module__.startswith("unittest"):
                    continue
                path = cls.__module__.replace(".", "/") + ".py"
                if path.startswith("tests/"):
                    ids.append(f"{path}::{cls.__qualname__}::{method}")

    walk(unittest.defaultTestLoader.discover(os.path.join(root, "tests"), top_level_dir=root))
    return sorted(set(ids))


def plain_worker(root: str, out: str) -> None:
    from blast import harness

    ids = [line.strip() for line in sys.stdin if line.strip()]
    sys.path.insert(0, root)
    os.chdir(root)
    sys.dont_write_bytecode = True
    status: dict[str, tuple[bool, float]] = {}
    overhead: dict[str, float] = {}

    def report(i, failed, seconds):
        status[i] = (failed or status.get(i, (False, 0))[0], seconds)

    harness.CONTROL.timing = True
    harness.run(ids, report, class_done=overhead.__setitem__)
    harness.CONTROL.timing = False
    case_seconds = {i: spent for i, spent in harness.CONTROL.case_seconds.items() if len(spent) > 1}

    # Once more, traced, now that every cache this file's tests fill is warm. A line a
    # test runs on its own but skips here is lazy: it runs once and its result is kept.
    # Only tests that loop over cases need this (see the case filters in the selector).
    warm: dict[str, set] = {}
    looping = [i for i in ids if i in case_seconds]
    if looping:
        from blast.tracer import Tracer

        tracer = Tracer(root)
        harness.run(looping, lambda *_: None, tracer)
        tracer.finish()
        for i in looping:
            warm[i] = set().union(*tracer.test_lines.get(i, {}).values())
    with open(out, "wb") as f:
        pickle.dump(
            {"status": status, "overhead": overhead, "case_seconds": case_seconds, "warm": warm}, f, pickle.HIGHEST_PROTOCOL
        )


def scaled_worker(root: str, out: str) -> None:
    """Check that the scaled-down stand-ins for the slowest tests pass at the pinned commit."""
    from blast import harness

    sys.path.insert(0, root)
    os.chdir(root)
    sys.dont_write_bytecode = True
    harness.CONTROL.replay = True
    status: dict[str, tuple[bool, float]] = {}

    def report(i, failed, seconds):
        status[i] = (failed or status.get(i, (False, 0))[0], seconds)

    harness.run(list(harness.SCALED), report, scaled=set(harness.SCALED))
    with open(out, "wb") as f:
        pickle.dump({i: seconds for i, (failed, seconds) in status.items() if not failed}, f, pickle.HIGHEST_PROTOCOL)


def trace_worker(root: str, out: str) -> None:
    """Imports sqlglot once under the tracer, then forks for each test."""
    from blast import harness
    from blast.tracer import Tracer

    ids = [line.strip() for line in sys.stdin if line.strip()]
    sys.path.insert(0, root)
    os.chdir(root)
    sys.dont_write_bytecode = True
    tracer = Tracer(root)
    harness.CONTROL.tracer = tracer

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if info.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception:  # noqa: BLE001
            pass
    harness.install_patches()
    for module in sorted({harness.split_id(i)[0] for i in ids}):
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001
            pass

    results = {}
    for n, test_id in enumerate(ids):
        part = f"{out}.{n}"
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                tracer.import_tags = {}
                tracer.class_tags = {}
                status = {}
                harness.run([test_id], lambda i, failed, seconds: status.__setitem__(i, failed or status.get(i, False)), tracer)
                tracer.finish()
                with open(part, "wb") as f:
                    pickle.dump(
                        {
                            "failed": status.get(test_id, True),
                            "lines": tracer.test_lines.get(test_id, {}),
                            "classes": tracer.test_classes.get(test_id, {}),
                            "opens": tracer.test_opens.get(test_id, set()),
                            "tags": tracer.import_tags,
                            "ancestors": tracer.ancestors,
                            "class_tags": tracer.class_tags,
                        },
                        f,
                        pickle.HIGHEST_PROTOCOL,
                    )
                code = 0
            finally:
                os._exit(code)
        _, state = os.waitpid(pid, 0)
        if state == 0 and os.path.exists(part):
            with open(part, "rb") as f:
                results[test_id] = pickle.load(f)
            os.remove(part)
        else:
            results[test_id] = None
    with open(out, "wb") as f:
        pickle.dump(
            {"tests": results, "tags": tracer.import_tags, "class_tags": tracer.class_tags}, f, pickle.HIGHEST_PROTOCOL
        )


def static_index(root: str) -> dict:
    files = {}
    for top in ("sqlglot", "tests"):
        for directory, _, names in os.walk(os.path.join(root, top)):
            for name in names:
                if name.endswith(".py"):
                    path = os.path.join(directory, name)
                    rel = os.path.relpath(path, root)
                    with open(path, encoding="utf-8") as f:
                        files[rel] = ast.parse(f.read())
    modules = {pyindex.module_name(rel): rel for rel in files}
    imports = {}
    units = {}
    parsed_units = {}
    scope_names: set[str] = set()  # names bound at module or class level anywhere
    for rel, tree in files.items():
        imports[rel] = sorted(
            {modules[m] for m in pyindex.module_imports(tree, rel) if m in modules} - {rel}
        )
        parsed_units[rel] = pyindex.units_of(tree)
        units[rel] = [(u.kind, u.scope, u.name, u.start, u.end, sorted(u.names)) for u in parsed_units[rel]]
        for u in parsed_units[rel]:
            scope_names |= u.names
    # name -> [(file, unit, line, mode)], see pyindex.references
    refs: dict[str, set] = {}
    for rel, listed in parsed_units.items():
        for i, u in enumerate(listed):
            for name, line, mode in pyindex.references(u):
                if name in scope_names:
                    refs.setdefault(name, set()).add((rel, i, line, mode))
    # Every package's __init__ is imported before its modules.
    for rel in files:
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            init = "/".join(parts[:depth]) + "/__init__.py"
            if init in files and init != rel and init not in imports[rel]:
                imports[rel].append(init)
    refs = {name: sorted(where, key=repr) for name, where in refs.items()}
    return {"imports": imports, "refs": refs, "units": units, "modules": modules}


CORE_IMPORTS = "import sqlglot, sqlglot.optimizer, sqlglot.executor, sqlglot.lineage, sqlglot.diff, sqlglot.planner, sqlglot.serde, sqlglot.transforms, sqlglot.schema"


def core_modules(root: str, env: dict) -> list[str]:
    """sqlglot modules that load without any dialect being asked for."""
    code = CORE_IMPORTS + "; import sys; print(chr(10).join(sorted(m for m in sys.modules if m.split('.')[0] == 'sqlglot')))"
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=dict(env, PYTHONPATH=root), capture_output=True, text=True)
    return result.stdout.split() if result.returncode == 0 else []


def run_workers(mode: str, shards: list[list[str]], root: str, out: str, jobs: int, env: dict) -> list[str]:
    queue = list(enumerate(shards))
    running: list[tuple[subprocess.Popen, str]] = []
    parts = []
    while queue or running:
        while queue and len(running) < jobs:
            n, shard = queue.pop(0)
            part = os.path.join(out, f"{mode}{n}.pkl")
            p = subprocess.Popen(
                [sys.executable, "-m", "blast.build_map", "--worker", mode, root, part],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, text=True,
            )
            p.stdin.write("\n".join(shard) + "\n")
            p.stdin.close()
            running.append((p, part))
        for p, part in list(running):
            if p.poll() is not None:
                running.remove((p, part))
                if p.returncode != 0 or not os.path.exists(part):
                    raise SystemExit(f"map worker failed for {part}")
                parts.append(part)
        time.sleep(0.05)
    return parts


def main() -> None:
    args = sys.argv[1:]
    if args[0] == "--worker":
        {"plain": plain_worker, "trace": trace_worker, "scaled": scaled_worker}[args[1]](args[2], args[3])
        return
    root, out = os.path.realpath(args[0]), args[1]
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else (os.cpu_count() or 2)
    if "--ids" in args:
        with open(args[args.index("--ids") + 1]) as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = discover(root)
    os.makedirs(out, exist_ok=True)
    started = time.time()
    env = dict(os.environ, SKIP_INTEGRATION="0", PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))

    # Pass 1: timings and the pinned commit's own failures, one process per test file.
    by_file: dict[str, list[str]] = {}
    for i in ids:
        by_file.setdefault(i.split("::")[0], []).append(i)
    shards = sorted(by_file.values(), key=lambda shard: -len(shard))
    plain: dict[str, tuple[bool, float]] = {}
    overhead: dict[str, float] = {}
    case_seconds: dict[str, dict] = {}
    warm: dict[str, set] = {}
    for part in run_workers("plain", shards, root, out, jobs, env):
        with open(part, "rb") as f:
            data = pickle.load(f)
        os.remove(part)
        plain.update(data["status"])
        overhead.update(data["overhead"])
        case_seconds.update(data.get("case_seconds", {}))
        warm.update(data.get("warm", {}))
    scaled: dict[str, float] = {}
    for part in run_workers("scaled", [[""]], root, out, jobs, env):
        with open(part, "rb") as f:
            scaled = pickle.load(f)
        os.remove(part)
    log(f"plain pass done in {time.time() - started:.0f}s")

    # Pass 2: coverage, one process per test; the longest tests are spread over the workers first.
    ordered = sorted(ids, key=lambda i: -plain.get(i, (True, 0.0))[1])
    shards = [ordered[n::jobs] for n in range(jobs)]
    index = {i: n for n, i in enumerate(ids)}
    lines: dict[str, dict[int, int]] = {}
    line_cases: dict[str, dict[int, dict[int, int]]] = {}
    classes: dict[str, int] = {}
    class_cases: dict[str, dict[int, int]] = {}
    opens: dict[str, int] = {}
    tags: dict[str, dict[int, set]] = {}
    test_cases: dict[int, list] = {}
    lazy: dict[str, set] = {}
    subclasses: dict[str, set] = {}
    class_tags: dict[str, set] = {}
    traced_bad = set()
    for part in run_workers("trace", [s for s in shards if s], root, out, jobs, env):
        with open(part, "rb") as f:
            data = pickle.load(f)
        os.remove(part)
        for (rel, lineno), owners in data["tags"].items():
            tags.setdefault(rel, {}).setdefault(lineno, set()).update(owners)
        for key, owners in data.get("class_tags", {}).items():
            class_tags.setdefault(key, set()).update(owners)
        for test_id, record in data["tests"].items():
            n = index[test_id]
            bit = 1 << n
            if record is None:
                traced_bad.add(test_id)
                continue
            if record["failed"]:
                traced_bad.add(test_id)
            for (rel, lineno), owners in record["tags"].items():
                tags.setdefault(rel, {}).setdefault(lineno, set()).update(owners)
            for rel in record["opens"]:
                opens[rel] = opens.get(rel, 0) | bit
            for key, owners in record.get("class_tags", {}).items():
                class_tags.setdefault(key, set()).update(owners)
            for child, parents in record.get("ancestors", {}).items():
                for parent in parents:
                    subclasses.setdefault(parent, set()).add(child)
            # Case 0 is the part of the test outside any case.
            names = [c for c in record["lines"] if c is not None]
            names += [c for c in record["classes"] if c is not None and c not in names]
            names.sort()
            has_cases = bool(names)
            if has_cases:
                test_cases[n] = names
                alone = set().union(*record["lines"].values())
                for rel, lineno in alone - warm.get(test_id, alone):
                    lazy.setdefault(rel, set()).add(lineno)
            case_bit = {None: 1}
            case_bit.update({c: 1 << (k + 1) for k, c in enumerate(names)})
            for case, covered in record["lines"].items():
                for rel, lineno in covered:
                    per = lines.setdefault(rel, {})
                    per[lineno] = per.get(lineno, 0) | bit
                    if has_cases:
                        slot = line_cases.setdefault(rel, {}).setdefault(lineno, {})
                        slot[n] = slot.get(n, 0) | case_bit[case]
            for case, used in record["classes"].items():
                for key in used:
                    classes[key] = classes.get(key, 0) | bit
                    if has_cases:
                        slot = class_cases.setdefault(key, {})
                        slot[n] = slot.get(n, 0) | case_bit[case]
    log(f"traced pass done in {time.time() - started:.0f}s")

    static = static_index(root)
    os.makedirs(os.path.join(out, "files"), exist_ok=True)
    for rel in static["units"]:
        intern: dict[int, int] = {}
        per = {ln: intern.setdefault(mask, mask) for ln, mask in lines.get(rel, {}).items()}
        with open(os.path.join(out, "files", rel.replace("/", "__") + ".pkl"), "wb") as f:
            pickle.dump(
                {
                    "lines": per,
                    "cases": line_cases.get(rel, {}),
                    "lazy": lazy.get(rel, set()),
                    "tags": tags.get(rel, {}),
                    "units": static["units"][rel],
                },
                f,
                pickle.HIGHEST_PROTOCOL,
            )
    bad = sorted(i for i in ids if plain.get(i, (True, 0))[0] or i in traced_bad)
    with open(os.path.join(out, "core.pkl"), "wb") as f:
        pickle.dump(
            {
                "ids": ids,
                "bad": bad,
                "seconds": [plain.get(i, (True, 0.0))[1] for i in ids],
                "overhead": overhead,
                "scaled": scaled,
                "case_seconds": case_seconds,
                "classes": classes,
                "class_cases": class_cases,
                "class_tags": {key: sorted(owners) for key, owners in class_tags.items()},
                "subclasses": {key: sorted(children) for key, children in subclasses.items()},
                "test_cases": test_cases,
                "opens": opens,
                "imports": static["imports"],
                "modules": static["modules"],
                "core_modules": core_modules(root, env),
            },
            f,
            pickle.HIGHEST_PROTOCOL,
        )
    with open(os.path.join(out, "refs.pkl"), "wb") as f:
        pickle.dump(static["refs"], f, pickle.HIGHEST_PROTOCOL)
    log(f"map: {len(ids)} tests, {len(bad)} failing at the pinned commit, {time.time() - started:.0f}s")
    for i in bad:
        log("  failing at the pinned commit:", i)


def log(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
