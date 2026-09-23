#!/usr/bin/env python3
"""Regenerate one problem folder per accuracy band from bands.json.

    python make_bands.py            # rewrite mnist-medium-*/task.yml and sutro.yaml
    python make_bands.py --check    # fail if anything is out of date

Thresholds live in bands.json and nowhere else. The generated task.yml files
are the only thing KernelBot reads, so changing a band is a one-line edit here
followed by one run of this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- timeouts
#
# KernelBot enforces test_timeout / benchmark_timeout / ranked_timeout on the
# whole `python eval.py <mode> <cases>` process. The evaluator has to fit all
# of this inside one of them:
#
#   load_pools()      both datasets, verified and box-area resized.  Measured
#                     at 0.93 s on CPU for mnist + fashion at size 9; it runs
#                     BEFORE mode_deadline() is taken, so it is invisible to
#                     the evaluator's own budget and only KernelBot's timeout
#                     covers it.  POOL_LOAD_S is ~30x the measured cost.
#   Child startup     spawn, import torch, initialise CUDA.
#   the warm-up call  bounded by warmup_max_call_ms.
#   every timed call  bounded by max_call_ms.
#   MODE_RESERVE_S    held back by mode_deadline() so the evaluator can always
#                     report a failure itself instead of being killed.
#
# If a mode's timeout is below that sum, max_call_ms is not the real per-call
# limit: a submission that respects it still dies of "ran out of its mode time
# budget", and the board silently enforces a tighter limit than it publishes.
# check_timeouts() refuses to generate a band in that state.
#
# These three mirror constants in eval.py; tests/test_eval.py asserts they are
# still equal, so the two files cannot drift apart.
POOL_LOAD_S = 30.0
CHILD_STARTUP_S = 120.0  # eval.py: Child(startup_timeout_s=120.0)
MODE_RESERVE_S = 30.0  # eval.py: MODE_RESERVE_S


def timed_calls(settings: dict, mode: str) -> int:
    """How many bounded calls a mode makes, worst case, excluding the warm-up."""
    if mode == "test":
        return 1
    if mode == "benchmark":
        calls = min(settings["draws"], settings["bench_draws"])
        return calls + (settings["bench_holdout_draws"] if settings["holdout"] else 0)
    if mode == "leaderboard":
        return settings["draws"] + (settings["holdout_draws"] if settings["holdout"] else 0)
    raise ValueError(mode)


def timeout_floor(settings: dict, mode: str) -> int:
    """Seconds a mode needs before max_call_ms stops being the binding limit."""
    seconds = (
        POOL_LOAD_S
        + CHILD_STARTUP_S
        + settings["warmup_max_call_ms"] / 1000.0
        + timed_calls(settings, mode) * settings["max_call_ms"] / 1000.0
        + MODE_RESERVE_S
    )
    return math.ceil(seconds)


DEADLINE_FORMAT = "%Y-%m-%d %H:%M"


def check_deadline(config: dict, now: dt.datetime | None = None) -> list[str]:
    """The deadline goes to KernelBot in sutro.yaml. A malformed one is fatal;
    one that has already passed is a warning, because regenerating a finished
    competition's files is legitimate."""
    problems = []
    for name, value in [("competition", config["deadline"])] + [
        (band["name"], band["deadline"]) for band in config["bands"] if "deadline" in band
    ]:
        try:
            when = dt.datetime.strptime(value, DEADLINE_FORMAT)
        except (TypeError, ValueError):
            problems.append(f"{name}: deadline {value!r} is not '{DEADLINE_FORMAT}'")
            continue
        if when < (now or dt.datetime.now()):
            problems.append(f"{name}: deadline {value} has already passed (warning)")
    return problems


TIMEOUT_FIELD = {
    "test": "test_timeout",
    "benchmark": "benchmark_timeout",
    "leaderboard": "ranked_timeout",
}


def check_timeouts(config: dict) -> list[str]:
    """Every band whose mode timeout cannot cover its own per-call limits."""
    problems = []
    for band in config["bands"]:
        settings = {**config["defaults"], **band}
        for mode, field in TIMEOUT_FIELD.items():
            floor = timeout_floor(settings, mode)
            calls = timed_calls(settings, mode)
            if settings[field] < floor:
                problems.append(
                    f"{band['name']}: {field} is {settings[field]} s but {mode} mode needs "
                    f"{floor} s ({calls} call{'' if calls == 1 else 's'} at "
                    f"{settings['max_call_ms'] / 1000:g} s, a "
                    f"{settings['warmup_max_call_ms'] / 1000:g} s warm-up, "
                    f"{CHILD_STARTUP_S:g} s of child start-up, {POOL_LOAD_S:g} s of pool load "
                    f"and {MODE_RESERVE_S:g} s held in reserve)"
                )
    return problems

TASK_TEMPLATE = """\
# name: {name}
# GENERATED by make_bands.py from bands.json -- do not edit by hand.

files:
  - {{"name": "submission.py", "source": "@SUBMISSION@"}}
  - {{"name": "task.py", "source": "../task.py"}}
  - {{"name": "utils.py", "source": "../utils.py"}}
  - {{"name": "reference.py", "source": "../reference.py"}}
  - {{"name": "mnist_data.py", "source": "../mnist_data.py"}}
  - {{"name": "eval.py", "source": "../eval.py"}}

lang: "py"

description: |
  MNIST-medium, {label} error band. Train a classifier from scratch and predict,
  on the GPU, in as little time as possible.

  Every call receives {train:,} training images ({size}x{size} float32 in [0, 1]) with
  their labels, and {test:,} unlabelled test images, all resident on the GPU:

    custom_kernel((train_x, train_y, test_x)) -> labels

    train_x  ({train}, 1, {size}, {size}) float32      test_x  ({test}, 1, {size}, {size}) float32
    train_y  ({train},) int64 in [0, 9]      return  ({test},) integer labels in [0, 9]

  The ranked number is the mean CUDA-event time of one complete
  training-and-prediction call over {draws} fresh draws. Each draw is a new random
  {train:,}/{test:,} split of the official MNIST training set, drawn so that no test
  image is ever shown with a label, and with a secret per-draw label
  permutation. One warm-up call is untimed, on a draw from a different dataset:
  compilation, autotuning and CUDA graph capture are free, training is not.

  All {draws} timed calls are also scored. The submission qualifies when
  sum(correct) >= {required:,} of {total:,}, i.e. a mean error of at most {label}, and
  no single draw may fall more than {draw_slack_pct:g} percentage points below the band.
  A leaderboard run interleaves {holdout_draws} further calls on equally shaped draws from
  a different dataset at secret positions; they are timed and ranked like the
  rest, and must be at least {holdout_pct:g}% correct in aggregate. A submission that
  memorizes MNIST instead of learning fails that check.

  A submission file is limited to {max_source_bytes:,} bytes, with no literal over
  {max_literal_bytes:,} bytes, so an entry cannot carry the dataset with it.

  Rules, the full protocol and the design notes are in the competition README.

config:
  main: "eval.py"

templates:
  Python: "submission.py"

test_timeout: {test_timeout}
benchmark_timeout: {benchmark_timeout}
ranked_timeout: {ranked_timeout}
ranking_by: "last"

tests:
  - {test_case}

benchmarks:
  - {benchmark_case}
"""

COMPETITION_TEMPLATE = """\
# GENERATED by make_bands.py from bands.json -- do not edit by hand.
name: Sutro MNIST-medium time challenge
deadline: "{deadline}"
description: "Train a classifier on 10,000 9x9 MNIST images and label 10,000 more, on one A100, in the shortest time that still meets the band's accuracy."
problems:
{problems}"""

PROBLEM_TEMPLATE = """\
  - directory: {directory}
    name: {name}
    deadline: "{deadline}"
    gpus:
      - {gpu}
"""

FOLDER_README = """\
# {name}

Generated from `../bands.json` by `../make_bands.py`. The only file here is
`task.yml`; the evaluator, the types and the template are shared one level up.

Band: mean error at most {label} ({error_bp} basis points), which is
{required:,} correct of {total:,} over {draws} ranked draws.
"""


def case_line(fields: dict) -> str:
    return "{" + ", ".join(f'"{k}": {v}' for k, v in fields.items()) + "}"


def required_correct(total: int, error_bp: int) -> int:
    return -(-(total * (10000 - error_bp)) // 10000)


def render(config: dict) -> dict[str, str]:
    """Return every generated path (relative to this directory) and its contents."""
    defaults = config["defaults"]
    files: dict[str, str] = {}
    problems = []
    for band in config["bands"]:
        settings = {**defaults, **band}
        name, label, error_bp = band["name"], band["label"], band["error_bp"]
        total = settings["draws"] * settings["test"]
        shared = {
            "size": settings["size"],
            "train": settings["train"],
            "test": settings["test"],
            "error_bp": error_bp,
            "draw_slack_bp": settings["draw_slack_bp"],
            "dispersion_x10": settings["dispersion_x10"],
            "max_call_ms": settings["max_call_ms"],
            "warmup_max_call_ms": settings["warmup_max_call_ms"],
            "max_source_bytes": settings["max_source_bytes"],
            "max_literal_bytes": settings["max_literal_bytes"],
            # The evaluator clamps every per-command deadline to what is left of
            # the mode's budget, so it needs the same timeouts KernelBot uses.
            "test_timeout": settings["test_timeout"],
            "benchmark_timeout": settings["benchmark_timeout"],
            "ranked_timeout": settings["ranked_timeout"],
        }
        test_case = case_line(
            {**shared, "draws": 1, "bench_draws": 1, "seed": settings["test_seed"], "holdout": 0}
        )
        benchmark_case = case_line(
            {
                **shared,
                "draws": settings["draws"],
                "bench_draws": settings["bench_draws"],
                "seed": settings["benchmark_seed"],
                "holdout": settings["holdout"],
                "holdout_draws": settings["holdout_draws"],
                "bench_holdout_draws": settings["bench_holdout_draws"],
                "holdout_min_bp": settings["holdout_min_bp"],
            }
        )
        files[f"{name}/task.yml"] = TASK_TEMPLATE.format(
            name=name,
            label=label,
            size=settings["size"],
            train=settings["train"],
            test=settings["test"],
            draws=settings["draws"],
            total=total,
            required=required_correct(total, error_bp),
            holdout_pct=(10000 - settings["holdout_min_bp"]) / 100,
            holdout_draws=settings["holdout_draws"],
            draw_slack_pct=settings["draw_slack_bp"] / 100,
            max_source_bytes=settings["max_source_bytes"],
            max_literal_bytes=settings["max_literal_bytes"],
            test_timeout=settings["test_timeout"],
            benchmark_timeout=settings["benchmark_timeout"],
            ranked_timeout=settings["ranked_timeout"],
            test_case=test_case,
            benchmark_case=benchmark_case,
        )
        files[f"{name}/README.md"] = FOLDER_README.format(
            name=name,
            label=label,
            error_bp=error_bp,
            required=required_correct(total, error_bp),
            total=total,
            draws=settings["draws"],
        )
        problems.append(
            PROBLEM_TEMPLATE.format(
                directory=f"sutro_mnist/{name}",
                name=name,
                deadline=settings.get("deadline", config["deadline"]),
                gpu=settings.get("gpu", config["gpu"]),
            )
        )
    files.update(band_templates(config))
    files["sutro.yaml"] = COMPETITION_TEMPLATE.format(
        deadline=config["deadline"], problems="".join(problems)
    )
    files["bands.md"] = bands_table(config) + timeout_table(config)
    files["README.md"] = readme_with_bands(config)
    return files


def band_templates(config: dict) -> dict[str, str]:
    """One starter file per band, so a downloaded template posts to its own board.

    KernelBot hands every problem the same ``templates:`` text, and a
    ``#!POPCORN leaderboard`` header in a shared file would send every
    participant's first submission to whichever band the shared file names.
    """
    body = (HERE / "submission.py").read_text().splitlines(keepends=True)
    if not body or not body[0].startswith("#!POPCORN leaderboard "):
        raise SystemExit("submission.py must start with a #!POPCORN leaderboard line")
    files = {}
    for band in config["bands"]:
        name = band["name"]
        files[f"{name}/submission.py"] = f"#!POPCORN leaderboard {name}\n" + "".join(body[1:])
    return files


README_MARKER = "<!-- GENERATED by make_bands.py from bands.json -- do not edit by hand. -->"
README_END = "<!-- END GENERATED -->"


def readme_with_bands(config: dict) -> str:
    """The README with its band table refreshed, so prose cannot go stale."""
    readme = (HERE / "README.md").read_text()
    start = readme.find(README_MARKER)
    end = readme.find(README_END)
    if start < 0 or end < start:
        raise SystemExit(
            f"README.md must contain {README_MARKER} ... {README_END} around the band table"
        )
    return readme[:start] + bands_table(config) + timeout_table(config) + "\n" + readme[end:]


def bands_table(config: dict) -> str:
    """The band table the README includes, so prose and task.yml cannot drift."""
    defaults = config["defaults"]
    lines = [
        "<!-- GENERATED by make_bands.py from bands.json -- do not edit by hand. -->",
        "",
        "| Problem | Mean error at most | Correct needed | Ranked draws |",
        "| --- | ---: | ---: | ---: |",
    ]
    for band in config["bands"]:
        settings = {**defaults, **band}
        total = settings["draws"] * settings["test"]
        lines.append(
            f"| `{band['name']}` | {band['label']} | "
            f"{required_correct(total, band['error_bp']):,} / {total:,} | {settings['draws']} |"
        )
    return "\n".join(lines) + "\n"


def timeout_table(config: dict) -> str:
    """What each mode's timeout has to cover, so the headroom is never implicit."""
    defaults = config["defaults"]
    settings = {**defaults, **config["bands"][0]}
    lines = [
        "",
        "## Mode time budgets",
        "",
        "The per-call limit is `max_call_ms`. A mode's timeout has to cover the",
        "pool load, the child's start-up, the untimed warm-up call, every timed",
        "call at that limit, and the reserve the evaluator keeps so it can report",
        "a failure itself. `make_bands.py` refuses to generate a band where it",
        "does not, because then the published per-call limit is not the real one.",
        "",
        "| Mode | Timed calls | Needs | Timeout | Headroom |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, field in TIMEOUT_FIELD.items():
        floor = timeout_floor(settings, mode)
        lines.append(
            f"| `{mode}` | {timed_calls(settings, mode)} | {floor} s | "
            f"{settings[field]} s | {settings[field] - floor} s |"
        )
    lines += [
        "",
        f"Fixed costs: {POOL_LOAD_S:g} s pool load, {CHILD_STARTUP_S:g} s child start-up, "
        f"{settings['warmup_max_call_ms'] / 1000:g} s warm-up, "
        f"{MODE_RESERVE_S:g} s reserve; {settings['max_call_ms'] / 1000:g} s per timed call.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only report drift")
    args = parser.parse_args()
    config = json.loads((HERE / "bands.json").read_text())

    # A band whose mode timeout cannot cover its own max_call_ms publishes a
    # per-call limit it does not enforce. Refuse to generate it either way:
    # --check must fail, and a plain run must not write the bad task.yml.
    fatal = False
    for line in check_deadline(config):
        print("deadline: " + line)
        fatal = fatal or not line.endswith("(warning)")
    if fatal:
        return 1

    broken = check_timeouts(config)
    if broken:
        for line in broken:
            print("timeout floor: " + line)
        return 1

    files = render(config)

    stale = []
    for relative, contents in sorted(files.items()):
        path = HERE / relative
        current = path.read_text() if path.exists() else None
        if current == contents:
            continue
        stale.append(relative)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)

    expected = {name for name in (band["name"] for band in config["bands"])}
    for folder in HERE.glob("mnist-medium-*"):
        if folder.is_dir() and folder.name not in expected:
            stale.append(f"{folder.name}/ (removed)")
            if not args.check:
                shutil.rmtree(folder)

    if args.check:
        if stale:
            print("out of date: " + ", ".join(stale))
            return 1
        print("up to date")
        return 0
    print("wrote " + ", ".join(stale) if stale else "nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
