# Blast Radius entry

A test selector for the [Blast Radius track](https://www.joinbuilders.org/buildathon/tracks/blast-radius): given a patch to sqlglot at commit `eb11aec4`, print the tests worth running and how to split them over workers.

## How it decides

For each commit the selector does three things.

1. **Find the tests the patch could reach.** While the image builds, every sqlglot test runs once at the pinned commit under a tracer (`blast/build_map.py`). That records, per test, the function lines it executes, the classes whose instances it uses, and the fixture files it reads. At run time the patch is compared with the pinned source statement by statement (`blast/analyze.py`):
   - a change inside a function selects the tests that execute the changed statement;
   - a change to a class attribute selects the tests that use that class or a subclass;
   - a change to a module-level name selects the tests that run code naming it;
   - a change to a fixture selects the tests that read it;
   - a change that leaves the syntax tree alone (a comment, blank lines) selects nothing.
2. **Run them, when that is cheap.** If the candidates cost less CPU to run than they would cost to select unrun, they are run here against the patched tree (`blast/harness.py`), in suite order, and only the ones that fail are selected. Most single-line edits reach a handful of fast tests, and about half of them break nothing, so this is where most of the score comes from. Candidates that are too expensive to run are selected as they are. When a patch can change state the whole process shares (what gets imported or registered), a test's result depends on what ran before it, so the candidates are run all together or not at all.
3. **Split the selection.** Longest test first onto the lightest worker, for every worker count from 1 to 8, keeping the cheapest.

Two details keep step 2 cheap:

- **Cases.** The slowest tests loop over independent cases, such as the statements of a fixture file or the list of dialects. Coverage is recorded per case, so a run replays only the cases the patch can reach, and stops a test at its first failure. A fixture edit replays only the statements that changed.
- **No worker pools.** sqlglot's optimizer and executor tests hand work to spawned processes. Here that work runs in-process, still through pickle, which removes most of their cost.

One test, `test_max_nodes`, is too slow to run while deciding (about 29 s). A copy of it with smaller numbers runs instead and stands in for it.

Nothing is looked up by patch: the maps describe the pinned commit only, and every selection comes from them or from running the tests.

## Layout

| Path | What it is |
| --- | --- |
| `selector.py` | Entry point. Reads `/data`, prints one JSON line per commit. |
| `blast/analyze.py` | Patch to candidate tests. |
| `blast/harness.py` | Runs tests in-process with unittest; case replay. |
| `blast/build_map.py`, `blast/tracer.py` | Build-time coverage maps. |
| `blast/patch.py`, `blast/pyindex.py` | Unified diff reader; per-file index of functions and statements. |
| `Dockerfile` | Installs the test dependencies, fetches sqlglot at the pinned commit, builds the maps. |
| `sample/`, `score.py`, `SCORING.md` | The organisers' sample commits, local scorer and scoring rules. |

## What the build records

`blast/build_map.py` runs the suite at the pinned commit in two passes. The first runs each test file in one process, for timings and for the tests that already fail in the image (those are never run at decision time, only selected). The second runs every test in a process of its own, forked from a parent with sqlglot imported, under `sys.monitoring`. Each test starts from the same clean state because sqlglot fills caches lazily: a test that found a cache already warm would not be seen executing the code that fills it.

`make explain` prints, per commit, how many tests were candidates, how many were run and what was selected, and lists any test that fails in the image at the pinned commit.

## Run it

You need Docker, make and Python 3.9 or newer.

```sh
make sample    # build, run on the 30 sample commits under the scorer's limits, score
make explain   # the same, with one line per commit on stderr
make check     # output format only
```

The build runs sqlglot's whole test suite twice to make the maps, which takes a few minutes natively and much longer under emulation. On an Apple Silicon Mac, `make sample` builds for linux/amd64 as the scorer does; for a quick local loop, build and run natively by removing `--platform linux/amd64` from the Makefile.

## Submit

Push this folder to a public GitHub repository with the `Dockerfile` at its top, then enter the repository and commit hash on the track page.
