"""Blast Radius selector.

For each commit:
  1. Work out, from maps built at the pinned commit, which tests could be
     affected by the patch (blast/analyze.py).
  2. When that set is cheap enough, run it here against the patched tree and
     keep only the tests that fail. Otherwise select the whole set unrun.
  3. Split the selection over workers so the busiest one is as light as possible.

One JSON line per commit goes to stdout; everything else goes to stderr.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import sys
import time

START = time.monotonic()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from blast import analyze, patch  # noqa: E402

REPO = os.environ.get("BLAST_REPO", "/opt/sqlglot")
MAP = os.environ.get("BLAST_MAP", "/opt/blast-map")
DATA = os.environ.get("BLAST_DATA", "/data")
SCRATCH = os.environ.get("BLAST_TMP", "/tmp/blast")
VERBOSE = os.environ.get("BLAST_VERBOSE") == "1"
MAX_LEVEL = int(os.environ.get("BLAST_MAX_LEVEL", "2"))  # development: 0 turns shared imports off
FORCE_RUN = os.environ.get("BLAST_FORCE_RUN") == "1"  # development: run every candidate set

WORKER_COST_MS = 2000
MAX_WORKERS = 8
PARALLEL = 2

# Chance that a candidate set holds a failing test at all, and the share of the
# set's seconds that then has to be selected. Both only steer the run-or-select choice.
FAIL_CHANCE = 0.6


HEAVY_MS = 3000
HEAVY_SHARE = 0.8
# A failing test that loops over cases stops at its first failing case.
EARLY_EXIT = 0.3


def fail_share(count: int) -> float:
    """A handful of candidates tends to fail together; of a broad set, only a part does."""
    return 0.9 if count <= 3 else 0.5 if count <= 15 else 0.2

# CPU a run costs before its first test, by how much of sqlglot the parent already holds.
START_SECONDS = (0.2, 0.1, 0.04)
MAX_RUN_SECONDS = float(os.environ.get("BLAST_MAX_RUN", "25"))


def log(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)


# -- splitting a selection across workers ------------------------------------------

def schedule(numbers: list[int], millis: dict[int, int]) -> tuple[int, list[list[int]]]:
    """Cheapest split found: (cost in ms, workers)."""
    if not numbers:
        return 0, []
    ordered = sorted(numbers, key=lambda n: (-millis[n], n))
    total = sum(millis[n] for n in ordered)
    best: tuple[int, list[list[int]]] | None = None
    for count in range(1, min(MAX_WORKERS, len(ordered)) + 1):
        floor = max(millis[ordered[0]], -(-total // count)) + WORKER_COST_MS * count
        if best is not None and floor >= best[0]:
            continue
        loads = [0] * count
        workers: list[list[int]] = [[] for _ in range(count)]
        for n in ordered:
            lightest = loads.index(min(loads))
            workers[lightest].append(n)
            loads[lightest] += millis[n]
        cost = max(loads) + WORKER_COST_MS * count
        if best is None or cost < best[0]:
            best = (cost, workers)
    assert best is not None
    return best


# -- the patched tree --------------------------------------------------------------

def build_overlay(root: str, changed: dict[str, str | None]) -> None:
    """A copy of the checkout made of symlinks, with real files only where the patch changed them."""
    real_dirs = {""}
    for rel in changed:
        parts = rel.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:depth]))
    for directory in sorted(real_dirs, key=lambda d: d.count("/") if d else -1):
        target = os.path.join(root, directory) if directory else root
        os.makedirs(target, exist_ok=True)
        source = os.path.join(REPO, directory) if directory else REPO
        if not os.path.isdir(source):
            continue
        for entry in os.listdir(source):
            rel = f"{directory}/{entry}" if directory else entry
            if rel in real_dirs or rel in changed:
                continue
            os.symlink(os.path.join(source, entry), os.path.join(target, entry))
    for rel, text in changed.items():
        if text is not None:
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
                f.write(text)


# -- one commit ---------------------------------------------------------------------

class Commit:
    def __init__(self, commit_id: str):
        self.id = commit_id
        self.mask = 0
        self.changed: dict[str, str | None] = {}
        self.runnable = True
        self.notes: list[str] = []
        self.selected: set[int] = set()  # map indexes to select without running
        self.to_run: list[int] = []
        self.failed: set[int] = set()
        self.estimate = 0.0
        self.gain = 0.0
        self.case_bits: dict[int, int] = {}
        self.whole = 0
        self.shared_state = False
        self.level = 0  # how much of sqlglot the runner may have imported before forking
        self.filters: dict[str, dict[str, frozenset]] = {}
        self.scaled: set[str] = set()
        self.smoke: dict[str, int] = {}  # module -> tests to select if it no longer imports
        self.expected: dict[str, float] = {}  # test id -> seconds it should take to run here


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def analyse(commit: Commit, maps: analyze.Maps, text: str) -> None:
    try:
        files = patch.parse(text)
    except Exception as error:  # noqa: BLE001
        commit.notes.append(f"unreadable patch: {error!r}")
        commit.mask, commit.runnable = maps.everything, False
        return
    for fp in files:
        if fp.mode_only and not fp.hunks and fp.old == fp.new:
            continue
        old_rel, new_rel = fp.old, fp.new
        old_text = read_text(os.path.join(REPO, old_rel)) if old_rel else None
        if old_rel and old_text is None and os.path.exists(os.path.join(REPO, old_rel)):
            fp.binary = True
        name = old_rel or new_rel or ""
        is_python = name.endswith(".py") and name.startswith(("sqlglot/", "tests/"))
        if pytest_config_change(fp, name):
            # pytest's own configuration decides how every test is collected and run.
            commit.notes.append(f"{name}: pytest configuration")
            commit.mask, commit.runnable = maps.everything, False
            continue
        if fp.binary:
            commit.runnable = False
            commit.notes.append(f"{old_rel or new_rel}: binary")
            if is_python:
                commit.mask |= maps.importer_users(old_rel) if old_rel else 0
            else:
                commit.mask |= analyze.other_change(maps, old_rel, new_rel) | (
                    analyze.other_change(maps, None, old_rel) if old_rel else 0
                )
            continue
        if new_rel is None:
            new_text = None
        else:
            try:
                new_text = patch.apply(old_text or "", fp.hunks)
            except patch.PatchError as error:
                commit.notes.append(f"{old_rel}: patch does not apply ({error})")
                commit.runnable = False
                commit.mask |= maps.importer_users(old_rel) if is_python and old_rel else maps.everything
                continue
        if old_rel and old_rel != new_rel:
            commit.changed[old_rel] = None
        if new_rel:
            commit.changed[new_rel] = new_text
        if old_rel and new_rel and old_rel != new_rel:
            # A rename: the old path is gone, and the new one is a new file.
            if old_rel.endswith(".py"):
                commit.mask |= analyze.python_change(maps, old_rel, old_text, None, commit.notes)
            else:
                commit.mask |= analyze.other_change(maps, old_rel, None)
            if not new_rel.endswith(".py"):
                commit.mask |= analyze.other_change(maps, None, new_rel)
            continue
        rel = old_rel or new_rel
        if old_text is not None and new_text == old_text:
            continue
        if is_python:
            try:
                commit.mask |= analyze.python_change(maps, rel, old_text, new_text, commit.notes)
            except Exception as error:  # noqa: BLE001 - never let one odd patch sink the batch
                commit.notes.append(f"{rel}: analysis failed ({error!r})")
                commit.mask |= maps.importer_users(rel) if old_text is not None else 0
        else:
            commit.mask |= analyze.other_change(maps, old_rel, new_rel)


PYTEST_FILES = ("conftest.py", "pytest.ini", "tox.ini", ".pytest.ini")
PYTEST_SECTIONS = ("[tool.pytest", "[pytest]", "[tool:pytest]")


def pytest_config_change(fp, name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    if base in PYTEST_FILES:
        return "/" not in name or name.startswith("tests/")
    if name in ("pyproject.toml", "setup.cfg"):
        body = "\n".join(row for _, rows in fp.hunks for row in rows)
        if any(section in body for section in PYTEST_SECTIONS):
            return True
        old = read_text(os.path.join(REPO, name)) or ""
        changed = [row[1:] for _, rows in fp.hunks for row in rows if row[:1] in "+-"]
        # A changed line inside an existing pytest section.
        section = ""
        inside = set()
        for line in old.split("\n"):
            if line.startswith("["):
                section = line
            elif any(section.startswith(s) for s in PYTEST_SECTIONS):
                inside.add(line)
        return any(line in inside for line in changed if line.strip())
    return False


def bits(mask: int) -> list[int]:
    out = []
    index = 0
    while mask:
        low = mask & -mask
        index = low.bit_length() - 1
        out.append(index)
        mask ^= low
    return out


# -- running candidates -------------------------------------------------------------

class Job:
    def __init__(self, commit: Commit, ids: list[str], limit: float, index_of: dict[str, int]):
        self.commit = commit
        self.index_of = index_of
        self.ids = ids
        self.limit = limit
        self.pid = 0
        self.fd = -1
        self.buffer = b""
        self.results: dict[str, bool] = {}
        self.current: str | None = None
        self.deadline = 0.0
        self.root = os.path.join(SCRATCH, commit.id)
        self.killed = False
        self.cpu = 0.0
        self.broken: set[str] = set()
        self.expected: dict[str, float] = {}
        self.test_deadline = float("inf")
        self.attempt = 1

    def start(self) -> None:
        from blast import harness

        shutil.rmtree(self.root, ignore_errors=True)
        build_overlay(self.root, self.commit.changed)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.setsid()
                os.close(read_fd)
                devnull = os.open(os.devnull, os.O_RDWR)
                for fd in (0, 1, 2):
                    os.dup2(devnull, fd)
                # Whatever the parent imported from the pinned checkout is unchanged by this
                # patch; everything else must now come from the patched tree.
                if REPO in sys.path:
                    sys.path.remove(REPO)
                sys.path.insert(0, self.root)
                for name, module in list(sys.modules.items()):
                    if name == "sqlglot" or name.startswith("sqlglot."):
                        path = getattr(module, "__path__", None)
                        if path:
                            module.__path__ = [self.root + p[len(REPO):] if p.startswith(REPO) else p for p in path]
                os.chdir(self.root)
                sys.dont_write_bytecode = True
                control = harness.CONTROL
                control.replay = True
                control.filters = self.commit.filters
                control.changed_fixtures = {
                    rel: os.path.join(REPO, "tests", "fixtures")
                    for rel in self.commit.changed
                    if rel.startswith("tests/fixtures/")
                }

                def report(test_id, failed, seconds):
                    os.write(write_fd, f"{'F' if failed else 'P'} {test_id}\n".encode())

                def started(test_id):
                    os.write(write_fd, f"S {test_id}\n".encode())

                import importlib

                for name in self.commit.smoke:
                    try:
                        importlib.import_module(name)
                        os.write(write_fd, f"M {name} ok\n".encode())
                    except BaseException:  # noqa: BLE001
                        os.write(write_fd, f"M {name} broken\n".encode())

                harness.run(self.ids, report, started=started, scaled=self.commit.scaled)
                os.write(write_fd, b"D done\n")
            except BaseException:  # noqa: BLE001
                pass
            finally:
                os._exit(0)
        os.close(write_fd)
        self.pid, self.fd = pid, read_fd
        self.deadline = time.monotonic() + self.limit

    def feed(self) -> bool:
        """Read what the child wrote. False once the pipe is closed."""
        chunk = os.read(self.fd, 65536)
        if not chunk:
            return False
        self.buffer += chunk
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            tag, _, test_id = line.decode().partition(" ")
            if tag == "M":
                name, _, state = test_id.partition(" ")
                if state == "broken":
                    self.broken.add(name)
            elif tag == "S":
                self.current = test_id
                # A test far over its usual time is treated as hung.
                self.test_deadline = time.monotonic() + max(2.5, 8 * self.expected.get(test_id, 0.0) + 1.5)
            elif tag in "FP":
                self.results[test_id] = self.results.get(test_id, False) or tag == "F"
                if self.current == test_id:
                    self.current = None
                    self.test_deadline = float("inf")
        return True

    def stop(self, kill: bool) -> None:
        if kill:
            self.killed = True
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            _, _, usage = os.wait4(self.pid, 0)
            self.cpu = usage.ru_utime + usage.ru_stime
        except OSError:
            pass
        if not kill:
            try:  # anything a test left running
                os.killpg(self.pid, signal.SIGKILL)
            except OSError:
                pass
        ACTIVE.discard(self)
        shutil.rmtree(self.root, ignore_errors=True)
        # A test that ran and passed is dropped. One that did not run, or was still
        # running when its time ran out, stays selected.
        commit = self.commit
        for test_id in self.ids:
            if self.results.get(test_id) is False:
                commit.selected.discard(self.index_of[test_id])
        for name in self.broken:
            commit.notes.append(f"{name} no longer imports")
            commit.selected |= set(bits(commit.smoke.get(name, 0)))
        commit.notes.insert(0, f"ran in {self.cpu:.2f}s cpu")
        if self.killed or len(self.results) < len(self.ids):
            commit.notes.append(f"run cut short after {len(self.results)}/{len(self.ids)} tests")


def preload(need_heavy: bool) -> None:
    """Import what the tests import, once, before forking."""
    import concurrent.futures.process  # noqa: F401
    import csv  # noqa: F401
    import datetime  # noqa: F401
    import decimal  # noqa: F401
    import gzip  # noqa: F401
    import logging  # noqa: F401
    import multiprocessing  # noqa: F401
    import sqlite3  # noqa: F401
    import tempfile  # noqa: F401
    import tracemalloc  # noqa: F401
    import unittest.mock  # noqa: F401

    from blast import harness  # noqa: F401

    names = ["pytz", "dateutil.relativedelta"]
    if need_heavy:
        names += ["numpy", "pandas", "pandas.testing", "duckdb"]
    for name in names:
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            pass


ACTIVE: set = set()
MAX_CHILD_BYTES = 1400 * 1024 * 1024  # the container has 4 GB for everything
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def resident_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * PAGE
    except (OSError, ValueError, IndexError):
        return 0


CORE_IMPORTS = (
    "sqlglot", "sqlglot.optimizer", "sqlglot.executor", "sqlglot.lineage", "sqlglot.diff",
    "sqlglot.planner", "sqlglot.serde", "sqlglot.transforms", "sqlglot.schema",
)


def preload_sqlglot(everything: bool) -> None:
    """Import unpatched sqlglot in the parent, so children that cannot be affected skip it."""
    import importlib
    import logging

    logging.disable(logging.CRITICAL)
    if REPO not in sys.path:
        sys.path.append(REPO)
    try:
        for name in CORE_IMPORTS:
            importlib.import_module(name)
        if everything:
            import pkgutil

            import sqlglot

            for info in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
                if not info.name.endswith("__main__"):
                    importlib.import_module(info.name)
    finally:
        logging.disable(logging.NOTSET)


def run_level(commit: Commit, core_modules: set) -> int:
    """0: the patch touches sqlglot code that is always loaded. 1: only code loaded on demand.
    2: no sqlglot code at all."""
    touched = [rel for rel in commit.changed if rel.startswith("sqlglot/") and rel.endswith(".py")]
    if not touched:
        return 2
    for rel in touched:
        name = rel[:-3].replace("/", ".")
        if name.endswith(".__init__") or name in core_modules or not core_modules:
            return 0
    return 1


def run_jobs(jobs: list[Job], wall_deadline: float) -> None:
    queue = list(jobs)
    active: dict[int, Job] = {}
    while queue or active:
        now = time.monotonic()
        while queue and len(active) < PARALLEL:
            job = queue.pop(0)
            if now + min(job.limit, job.commit.estimate * 1.5 + 1.0) > wall_deadline:
                continue  # no time left: its tests stay selected unrun
            job.start()
            ACTIVE.add(job)
            job.deadline = min(job.deadline, wall_deadline)
            active[job.fd] = job
        if not active:
            break
        timeout = max(0.0, min(j.deadline for j in active.values()) - time.monotonic())
        ready, _, _ = select.select(list(active), [], [], min(timeout, 0.25))
        for fd in ready:
            job = active[fd]
            if not job.feed():
                del active[fd]
                job.stop(kill=False)
        now = time.monotonic()
        for fd, job in list(active.items()):
            if now >= job.deadline or now >= job.test_deadline or resident_bytes(job.pid) > MAX_CHILD_BYTES:
                del active[fd]
                job.stop(kill=True)
                # The test that was running stays selected; the ones after it get one more go.
                rest = [i for i in job.ids if i not in job.results and i != job.current]
                if rest and job.current and job.attempt == 1 and now < job.deadline:
                    again = Job(job.commit, rest, job.limit, job.index_of)
                    again.expected, again.attempt = job.expected, 2
                    queue.insert(0, again)


# -- main -----------------------------------------------------------------------------

def main(state: State) -> None:
    os.environ["SKIP_INTEGRATION"] = "0"
    with open(os.path.join(DATA, "tests.json"), encoding="utf-8") as f:
        listed = json.load(f)["tests"]
    number_of = {t["id"]: t["number"] for t in listed}
    millis = {t["number"]: round(t["seconds"] * 1000) for t in listed}
    file_of = {t["number"]: t["file"] for t in listed}

    commits_dir = os.path.join(DATA, "commits")
    commit_ids = sorted(name[:-6] for name in os.listdir(commits_dir) if name.endswith(".patch"))
    state.commit_ids, state.millis = commit_ids, millis

    # Wall-clock budget: 3 minutes for the practice batch, 4.5 for the larger final one.
    limit = 180.0 if len(commit_ids) <= 48 else 270.0
    wall_deadline = START + limit - 40.0
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(limit - 22.0))

    try:
        maps = analyze.Maps(MAP)
    except Exception as error:  # noqa: BLE001 - without maps, every test is selected
        log(f"no usable map ({error!r}); selecting every test")
        return

    if VERBOSE and maps.bad:
        log(f"{len(maps.bad)} tests fail in this image at the pinned commit and are never run here:")
        for test_id in sorted(maps.bad):
            log("  " + test_id)
    index_number = [number_of.get(test_id) for test_id in maps.ids]
    known = set(maps.ids)
    unmapped = [t["number"] for t in listed if t["id"] not in known]
    if unmapped:
        log(f"{len(unmapped)} listed tests are not in the map; they are selected whenever a commit selects anything")
    bad_mask = 0
    for n, test_id in enumerate(maps.ids):
        if test_id in maps.bad:
            bad_mask |= 1 << n
    listed_mask = 0
    for n, number in enumerate(index_number):
        if number is not None:
            listed_mask |= 1 << n
    class_of = [test_id.rsplit("::", 1)[0].replace("/", ".").replace(".py::", "::") for test_id in maps.ids]
    state.unmapped = unmapped

    commits: list[Commit] = state.commits
    for commit_id in commit_ids:
        commit = Commit(commit_id)
        try:
            with open(os.path.join(commits_dir, commit_id + ".patch"), encoding="utf-8", errors="surrogateescape", newline="") as f:
                text = f.read()
            maps.reset()
            analyse(commit, maps, text)
            commit.case_bits, commit.whole = maps.case_bits, maps.whole
            commit.shared_state = maps.shared_state
            for rel in maps.smoke:
                if commit.changed.get(rel) is not None:
                    name = rel[:-3].replace("/", ".")
                    name = name[: -len(".__init__")] if name.endswith(".__init__") else name
                    commit.smoke[name] = maps.isolated(lambda: maps.importer_users(rel))[0] & listed_mask
        except Exception as error:  # noqa: BLE001
            commit.notes.append(f"analysis failed: {error!r}")
            commit.mask, commit.runnable = maps.everything, False
        commit.mask &= listed_mask

        candidates = bits(commit.mask)
        commit.selected = set(candidates)
        commits.append(commit)
        state.index_number = index_number
        blind_cost, _ = schedule([index_number[i] for i in candidates] + (unmapped if candidates else []), millis)
        run_list = [i for i in candidates if not bad_mask >> i & 1]
        commit.selected = set(candidates)
        if not candidates or not commit.runnable or not run_list:
            continue
        # What each candidate would cost to run here.
        run_seconds: dict[int, float] = {}
        heavy: set[int] = set()  # long tests that loop over cases: when anything breaks, they tend to
        for i in run_list:
            seconds = maps.seconds[i]
            test_id = maps.ids[i]
            if test_id in maps.scaled:
                # Too slow to run here: a scaled-down copy of the test decides instead.
                commit.scaled.add(test_id)
                seconds = maps.scaled[test_id]
            cases = maps.test_cases.get(i)
            # Bit 0 is the test outside its cases: what runs there can set up state every
            # case relies on, so the whole test runs.
            if cases and not commit.whole >> i & 1 and i in commit.case_bits and not commit.case_bits[i] & 1:
                picked = commit.case_bits[i]
                spec: dict[str, set] = {}
                for k, (label, index) in enumerate(cases):
                    if picked >> (k + 1) & 1:
                        spec.setdefault(label, set()).add(index)
                count = sum(len(v) for v in spec.values())
                if count < len(cases):
                    commit.filters[test_id] = {label: frozenset(v) for label, v in spec.items()}
                    timed = maps.case_seconds.get(test_id)
                    if timed:
                        typical = sum(timed.values()) / max(len(timed), 1)
                        chosen = sum(timed.get((label, index), typical) for label, v in spec.items() for index in v)
                        seconds = min(seconds, timed.get(None, 0.0) + chosen + 0.01)
                    else:
                        seconds = seconds * max(0.05, min(1.0, (count + 1) / len(cases)))
            if cases and millis[index_number[i]] >= HEAVY_MS:
                heavy.add(i)
            run_seconds[i] = seconds
            commit.expected[test_id] = seconds

        # Choose which candidates to run and which to select unrun. Running the costliest
        # ones is what hurts, so try leaving out the 0, 1, 2, ... costliest and keep the
        # split with the lowest expected total.
        fixed = [i for i in candidates if bad_mask >> i & 1]
        start_seconds = START_SECONDS[min(MAX_LEVEL, run_level(commit, maps.core_modules))]
        by_cost = sorted(run_list, key=lambda i: -run_seconds[i])
        best = (blind_cost / 1000, [], 0.0)
        splits = [(by_cost, fixed)]
        if not commit.shared_state:
            splits += [(by_cost[skipped:], fixed + by_cost[:skipped]) for skipped in range(1, min(len(by_cost), 24) + 1)]
        # The other way round: run only the few slow tests that are cheap to decide here.
        bargains = sorted(
            (i for i in run_list if millis[index_number[i]] >= 1000),
            key=lambda i: -millis[index_number[i]] / max(run_seconds[i], 0.01),
        )
        for count in range(1, (0 if commit.shared_state else min(len(bargains), 8)) + 1):
            chosen = set(bargains[:count])
            splits.append((bargains[:count], fixed + [i for i in run_list if i not in chosen]))
        for running, unrun in splits:
            if not running:
                continue
            share = fail_share(len(running))
            estimate = start_seconds + sum(maps.overhead.get(c, 0.0) for c in {class_of[i] for i in running})
            failing = 0.0  # expected recorded seconds of the tests that fail, given that some do
            failing_heavy = 0.0
            for i in running:
                likely = HEAVY_SHARE if i in heavy else share
                if i in heavy:
                    failing_heavy += likely * millis[index_number[i]] / 1000
                else:
                    failing += likely * millis[index_number[i]] / 1000
                if i in heavy:
                    estimate += run_seconds[i] * (1 - FAIL_CHANCE * likely * (1 - EARLY_EXIT))
                else:
                    estimate += run_seconds[i]
            if estimate > MAX_RUN_SECONDS:
                continue
            if unrun:
                # Workers are being paid for anyway; a failing test joins them.
                after = schedule([index_number[i] for i in unrun], millis)[0] / 1000 + FAIL_CHANCE * (failing / 4 + failing_heavy)
            else:
                after = FAIL_CHANCE * (WORKER_COST_MS / 1000 + failing + failing_heavy)
            if estimate + after < best[0]:
                best = (estimate + after, running, estimate)
        if FORCE_RUN:
            best = (0.0, run_list, start_seconds + sum(run_seconds.values()))
        commit.to_run = sorted(best[1])
        commit.estimate = best[2]

    if VERBOSE:
        own = os.times()
        log(f"analysis done: cpu {own.user + own.system:.2f}s, wall {time.monotonic() - START:.2f}s")
    jobs = []
    index_of = {test_id: n for n, test_id in enumerate(maps.ids)}
    for commit in sorted((c for c in commits if c.to_run or (c.smoke and c.runnable)), key=lambda c: c.estimate):
        ids = [maps.ids[i] for i in commit.to_run]
        job = Job(commit, ids, max(6.0, commit.estimate * 4 + 4.0), index_of)
        job.expected = commit.expected
        jobs.append(job)
    if jobs:
        heavy = any("test_optimizer" in i or "test_executor" in i for job in jobs for i in job.ids)
        preload(heavy)
        import gc

        gc.collect()
        gc.freeze()
        os.makedirs(SCRATCH, exist_ok=True)
        for job in jobs:
            job.commit.level = 0 if MAX_LEVEL == 0 else min(MAX_LEVEL, run_level(job.commit, maps.core_modules))
        # Importing every dialect up front only pays off when enough runs can share it.
        share_all = sum(1 for job in jobs if job.commit.level == 2) >= 8
        for level in (0, 1, 2):
            batch = [job for job in jobs if (job.commit.level if share_all else min(job.commit.level, 1)) == level]
            if not batch:
                continue
            if level:
                try:
                    preload_sqlglot(everything=level == 2)
                except BaseException as error:  # noqa: BLE001
                    if isinstance(error, (KeyboardInterrupt, OutOfTime)):
                        raise
                    log(f"could not preload sqlglot ({error!r}); the remaining tests stay selected")
                    break
                gc.collect()
                gc.freeze()
            run_jobs(batch, wall_deadline)

    emit(state)


class State:
    """Everything needed to print an answer, kept current so a late failure can still print one."""

    def __init__(self):
        self.commit_ids: list[str] = []
        self.millis: dict[int, int] = {}
        self.commits: list[Commit] = []
        self.index_number: list = []
        self.unmapped: list[int] = []
        self.printed = False


def emit(state: State) -> None:
    if state.printed:
        return
    state.printed = True
    out = []
    total = 0
    by_id = {commit.id: commit for commit in state.commits}
    everything = sorted(state.millis)
    for commit_id in state.commit_ids:
        commit = by_id.get(commit_id)
        if commit is None or not state.index_number:
            numbers = everything  # never analysed: select every test
        else:
            numbers = sorted(state.index_number[i] for i in commit.selected)
            if numbers and state.unmapped:
                numbers = sorted(set(numbers) | set(state.unmapped))
        cost, workers = schedule(numbers, state.millis)
        total += cost
        out.append(json.dumps({"commit": commit_id, "workers": workers}, separators=(",", ":")))
        if VERBOSE and commit is not None:
            log(
                f"{commit.id}: candidates={bin(commit.mask).count('1')} ran={len(commit.to_run)} "
                f"selected={len(numbers)} cost={cost / 1000:.2f}s est_run={commit.estimate:.2f}s | " + "; ".join(commit.notes[:3])
            )
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    sys.stdout.flush()
    cpu = os.times()
    log(
        f"{len(state.commit_ids)} commits, simulated {total / 1000:.1f}s, cpu {cpu.user + cpu.system + cpu.children_user + cpu.children_system:.1f}s, "
        f"wall {time.monotonic() - START:.1f}s"
    )


class OutOfTime(Exception):
    pass


def _alarm(signum, frame):
    raise OutOfTime()


if __name__ == "__main__":
    STATE = State()
    try:
        main(STATE)
    except BaseException as error:  # noqa: BLE001 - whatever happens, every commit gets a line
        log(f"stopped early: {error!r}")
    finally:
        signal.alarm(0)
        for job in list(ACTIVE):
            try:
                os.killpg(job.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            emit(STATE)
        finally:
            shutil.rmtree(SCRATCH, ignore_errors=True)
