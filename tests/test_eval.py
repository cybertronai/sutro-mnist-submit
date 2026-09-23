"""Unit tests for the parts of the harness that decide whether a run is fair.

    python -m pytest tests/test_eval.py -q

No GPU, no dataset download: every test here is arithmetic, parsing or
bookkeeping.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import eval as harness  # noqa: E402
import make_bands  # noqa: E402
from utils import combine, required_correct, stats, timing_plausible  # noqa: E402


# ------------------------------------------------------------------ accuracy rule

def test_required_correct_matches_the_published_bands():
    # 11 draws x 10,000 queries, the numbers printed in every task.yml
    assert required_correct(110000, 200) == 107800
    assert required_correct(110000, 300) == 106700
    assert required_correct(110000, 500) == 104500
    assert required_correct(110000, 800) == 101200
    assert required_correct(110000, 1200) == 96800


def test_required_correct_rounds_up_and_uses_integers():
    # 3 * 0.6667 = 2.0001 correct: two is not enough
    assert required_correct(3, 3333) == 3
    # exact multiples must not be inflated by float error
    assert required_correct(10000, 200) == 9800
    assert required_correct(110000, 0) == 110000
    assert required_correct(110000, 10000) == 0
    for total in (1, 7, 9999, 110000):
        for error_bp in (0, 1, 160, 200, 1500, 9999, 10000):
            exact = -(-(total * (10000 - error_bp)) // 10000)
            assert required_correct(total, error_bp) == exact
            assert (exact - 1) * 10000 < total * (10000 - error_bp) <= exact * 10000


def test_required_correct_rejects_nonsense_bands():
    with pytest.raises(ValueError):
        required_correct(100, -1)
    with pytest.raises(ValueError):
        required_correct(100, 10001)


def test_aggregate_rule_is_over_all_draws_not_per_draw():
    # one bad draw can be paid for by the others; that is the intent
    per_draw = [9600, 9450, 9500]
    assert sum(per_draw) >= required_correct(30000, 500)
    assert min(per_draw) < required_correct(10000, 500)


# ------------------------------------------------------------------ seeds

def test_combine_is_kernelbots_cantor_pairing():
    def reference(a, b):
        return int(a + (a + b) * (a + b + 1) // 2)

    for a in (0, 1, 101, 202, 65535):
        for b in (0, 3, 20260922, 2**40 + 7):
            assert combine(a, b) == reference(a, b)


def test_combine_hides_the_public_seed_behind_a_large_secret():
    secret = 2**40 + 12345
    combined = {combine(public, secret) for public in (101, 202, 303)}
    assert all(value > secret for value in combined)
    assert len(combined) == 3  # distinct public seeds stay distinct


# ------------------------------------------------------------------ draws

def make_pool(count=60000, classes=10):
    rng = np.random.default_rng(7)
    images = rng.random((count, 1, 3, 3), dtype=np.float32)
    labels = (np.arange(count) % classes).astype(np.int64)
    return images, labels


def universes_for(pool, seed=4242):
    return harness.split_universes(len(pool[1]), seed, harness.UNIVERSE_SALT)


def test_draw_is_disjoint_deterministic_and_pulled_from_the_pool():
    pool = make_pool()
    universes = universes_for(pool)
    visible, truth = harness.make_draw(pool, 12345, 50, 40, universes)
    again, truth_again = harness.make_draw(pool, 12345, 50, 40, universes)
    assert visible[0].shape == (50, 1, 3, 3)
    assert visible[1].shape == (50,)
    assert visible[2].shape == (40, 1, 3, 3)
    assert truth.shape == (40,)
    assert np.array_equal(visible[0], again[0]) and np.array_equal(truth, truth_again)
    # no test image is also a training image
    train_rows = {row.tobytes() for row in visible[0]}
    assert not any(row.tobytes() in train_rows for row in visible[2])


def test_test_images_are_never_shown_with_a_label_in_any_draw():
    """The hole that made a pool-memoization table pay: every draw used to
    re-split the same 60,000 rows, so a test image of draw 7 had probably
    already arrived, labelled, in the training half of draw 2."""
    pool = make_pool(count=2000)
    universes = universes_for(pool, seed=99)
    seen_with_a_label = set()
    queried = set()
    for step in range(12):
        visible, _ = harness.make_draw(pool, 99 + 13 * step, 300, 300, universes)
        seen_with_a_label.update(row.tobytes() for row in visible[0])
        queried.update(row.tobytes() for row in visible[2])
    assert seen_with_a_label and queried
    assert not (seen_with_a_label & queried)


def test_universes_split_the_pool_in_half_and_depend_on_the_secret():
    first = harness.split_universes(1000, 12345, harness.UNIVERSE_SALT)
    same = harness.split_universes(1000, 12345, harness.UNIVERSE_SALT)
    other = harness.split_universes(1000, 12346, harness.UNIVERSE_SALT)
    assert len(first[0]) == len(first[1]) == 500
    assert not set(first[0]) & set(first[1])
    assert np.array_equal(first[0], same[0])  # deterministic within a run
    assert not np.array_equal(first[0], other[0])  # a different secret, a different split


def test_label_permutation_is_secret_consistent_and_per_draw():
    images, labels = make_pool()
    pool = (images, labels)
    universes = universes_for(pool)
    visible, truth = harness.make_draw(pool, 999, 200, 200, universes)
    rows = np.random.default_rng([999, harness.DRAW_SALT])
    train_rows = rows.choice(universes[0], 200, replace=False)
    test_rows = rows.choice(universes[1], 200, replace=False)
    # the same permutation maps the true labels of both halves
    mapping = {}
    for true_label, shown in zip(labels[train_rows], visible[1]):
        mapping.setdefault(int(true_label), int(shown))
        assert mapping[int(true_label)] == int(shown)
    for true_label, shown in zip(labels[test_rows], truth):
        assert mapping[int(true_label)] == int(shown)
    assert sorted(mapping.values()) == list(range(10))  # a permutation, not a collapse
    # a different draw uses a different mapping, so memorized labels go stale
    other, _ = harness.make_draw(pool, 1000, 200, 200, universes)
    assert not np.array_equal(visible[1][:50], other[1][:50])


def test_draw_refuses_to_overflow_its_half_of_the_pool():
    pool = make_pool(count=60000)
    universes = universes_for(pool)
    with pytest.raises(ValueError):
        harness.make_draw(pool, 1, 40000, 30000, universes)


# ------------------------------------------------------------------ submission source

def write_submission(tmp_path, body):
    path = tmp_path / "submission.py"
    path.write_text(body)
    return path


def test_source_cap_rejects_an_embedded_dataset(tmp_path):
    case = dict(harness.DEFAULTS)
    path = write_submission(tmp_path, "TABLE = '" + "a" * 30000 + "'\n")
    with pytest.raises(harness.Failure) as error:
        harness.check_submission_source(case, path)
    assert "over the" in str(error.value)


def test_source_cap_rejects_one_oversized_literal(tmp_path):
    case = dict(harness.DEFAULTS, max_source_bytes=1_000_000)
    path = write_submission(tmp_path, "TABLE = '" + "a" * 30000 + "'\n")
    with pytest.raises(harness.Failure) as error:
        harness.check_submission_source(case, path)
    assert "literal" in str(error.value)


def test_source_cap_accepts_every_shipped_submission():
    case = dict(harness.DEFAULTS)
    for path in sorted((HERE.parent / "submissions").glob("*.py")):
        harness.check_submission_source(case, path)
    harness.check_submission_source(case, HERE.parent / "submission.py")


# ------------------------------------------------------------------ timing gate

def test_timing_gate_accepts_an_honest_call():
    assert timing_plausible(3.30, 3.45, 9.10) is None
    assert timing_plausible(0.21, 0.55, 4.00, 3.60) is None  # sub-millisecond call
    assert timing_plausible(260.0, 261.2, 275.0, 1.5) is None


def test_timing_gate_catches_a_patched_timer():
    assert "less than half" in timing_plausible(0.0, 260.0, 280.0)
    assert timing_plausible(1.0, 260.0, 280.0) is not None


def test_timing_gate_bounds_the_device_clock_from_below_with_the_parents():
    # Both of the child's clocks scaled by the same constant: every ratio test
    # between them still passes, and only the parent's clock notices.
    reason = timing_plausible(0.008, 0.010, 70.0, 0.7)
    assert reason is not None and "the parent measured" in reason
    # the same call reported honestly is fine
    assert timing_plausible(68.8, 69.0, 70.0, 0.7) is None
    # and the overhead really is subtracted: a short call behind a slow pipe
    assert timing_plausible(3.0, 3.2, 40.0, 36.0) is None


def test_timing_gate_catches_work_outside_the_timed_window():
    # half the work moved to an unsynchronized side stream
    assert timing_plausible(40.0, 200.0, 220.0) is not None


def test_timing_gate_catches_a_device_time_longer_than_the_wall_clock():
    assert "exceeds the child" in timing_plausible(12.0, 5.0, 30.0)


def test_timing_gate_trusts_the_parent_clock_over_the_child():
    # a child that under-reports its own wall clock is still bounded by the parent
    assert timing_plausible(100.0, 100.0, 50.0) is not None
    assert timing_plausible(float("nan"), 10.0, 20.0) is not None


# ------------------------------------------------------------------ cases

def write_cases(tmp_path, text):
    path = tmp_path / "cases.txt"
    path.write_text(text)
    return path


def test_read_cases_fills_defaults_and_combines_the_seed(tmp_path):
    path = write_cases(tmp_path, "size: 9; train: 10000; test: 10000; error_bp: 500; seed: 202\n")
    case = harness.read_cases(path, 20260922)[0]
    assert case["seed"] == combine(202, 20260922)
    assert case["draws"] == harness.DEFAULTS["draws"]
    assert case["max_call_ms"] == harness.DEFAULTS["max_call_ms"]
    assert case["spec"].startswith("size: 9")


def test_read_cases_rejects_unknown_fields_and_junk(tmp_path):
    with pytest.raises(ValueError):
        harness.read_cases(write_cases(tmp_path, "size: 9; sneaky: 1\n"), None)
    with pytest.raises(ValueError):
        harness.read_cases(write_cases(tmp_path, "size: nine\n"), None)
    with pytest.raises(ValueError):
        harness.read_cases(write_cases(tmp_path, "\n\n"), None)


# ------------------------------------------------------------------ statistics

def test_stats_reports_nanoseconds_like_kernelbot():
    values = [3.0e6, 3.2e6, 2.8e6, 3.1e6]
    result = stats(values)
    assert result["runs"] == 4
    assert result["best"] == 2.8e6 and result["worst"] == 3.2e6
    assert abs(result["mean"] - 3.025e6) < 1
    assert abs(result["median"] - 3.05e6) < 1
    assert result["err"] == pytest.approx(result["std"] / 2)


# ------------------------------------------------------------------ generated files

def test_generated_problem_folders_are_up_to_date():
    config = json.loads((HERE.parent / "bands.json").read_text())
    for relative, contents in make_bands.render(config).items():
        assert (HERE.parent / relative).read_text() == contents, f"{relative} is stale"


def test_readme_shows_the_generated_band_table():
    table = (HERE.parent / "bands.md").read_text().strip()
    assert table in (HERE.parent / "README.md").read_text()


def test_every_band_case_parses_and_keeps_its_threshold():
    config = json.loads((HERE.parent / "bands.json").read_text())
    for band in config["bands"]:
        task = (HERE.parent / band["name"] / "task.yml").read_text()
        cases = [json.loads(line[4:]) for line in task.splitlines()
                 if line.startswith("  - {") and "error_bp" in line]
        assert len(cases) == 2  # one test case, one ranked case
        for fields in cases:
            assert fields["error_bp"] == band["error_bp"]
            assert set(fields) <= set(harness.DEFAULTS)


# ------------------------------------------------------------------ process isolation

def run_script(tmp_path, body, environment=None):
    """Run a small program in its own interpreter and return its stdout."""
    import subprocess

    script = tmp_path / "probe.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(HERE.parent)!r})\n" + body)
    env = dict(os.environ)
    env.pop("POPCORN_SEED", None)
    env.update(environment or {})
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_network_guard_survives_raw_sockets_and_a_reloaded_module(tmp_path):
    # The monkeypatch alone missed both of these: socket.socket subclasses the
    # C type _socket.socket, and reloading socket rebuilds clean functions.
    output = run_script(
        tmp_path,
        """
from utils import install_network_guard

install_network_guard()
results = []

import _socket
try:
    raw = _socket.socket(); raw.settimeout(1); raw.connect(("127.0.0.1", 9))
    results.append("raw:REACHED")
except BaseException as error:
    results.append("raw:" + type(error).__name__)

import importlib, socket
importlib.reload(socket)
try:
    socket.create_connection(("127.0.0.1", 9), timeout=1)
    results.append("reload:REACHED")
except BaseException as error:
    results.append("reload:" + type(error).__name__)

try:
    open("/tmp/train-images-idx3-ubyte.gz", "rb")
    results.append("dataset:REACHED")
except BaseException as error:
    results.append("dataset:" + type(error).__name__)

print(";".join(results))
""",
    )
    assert output == "raw:NetworkDisabled;reload:NetworkDisabled;dataset:DatasetFileDenied"


def test_the_secret_seed_is_removed_from_the_process_environment(tmp_path):
    # os.environ.pop does not rewrite /proc/<pid>/environ, so the evaluator
    # re-execs itself with the secret handed over on a pipe instead.
    output = run_script(
        tmp_path,
        """
import os
import eval as harness

payload = harness.scrub_secret_environment()
print("secret=%s in_environ=%s" % (payload.get("secret"), "POPCORN_SEED" in os.environ))
""",
        {"POPCORN_SEED": "20260922"},
    )
    assert output == "secret=20260922 in_environ=False"


# ------------------------------------------------------------------ the secret seed of a ranked run

def run_main_and_capture(tmp_path, case_line, secret=None):
    """Run eval.main() far enough to see which seed it used, and return its keys.

    ``scrub_secret_environment`` is stubbed because its real implementation
    re-execs the interpreter, which would replace this probe with the shipped
    eval.py; what is under test is what main() does with what it hands back.
    """
    (tmp_path / "submission.py").write_text("def custom_kernel(data):\n    return data[2]\n")
    (tmp_path / "cases.txt").write_text(case_line + "\n")
    body = """
import json
import os
import eval as harness

seen = {}


def stop(cases, cache, consume=False):
    seen["seed"] = cases[0]["seed"]
    raise harness.Failure("stop here")


harness.load_pools = stop
harness.scrub_secret_environment = lambda: json.loads(os.environ.get("PROBE_SECRETS", "{}"))
read_fd, write_fd = os.pipe()
os.environ["POPCORN_FD"] = str(write_fd)
sys.argv = ["eval.py", "test", "cases.txt"]
code = harness.main()
try:
    os.close(write_fd)  # main() closes it through PopcornOutput
except OSError:
    pass
with os.fdopen(read_fd) as handle:
    lines = handle.read().splitlines()
print(json.dumps({"code": code, "lines": lines, "seed": seen.get("seed")}))
"""
    script = tmp_path / "probe.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(HERE.parent)!r})\n" + body)
    env = dict(os.environ)
    env.pop("POPCORN_SEED", None)
    env["PROBE_SECRETS"] = json.dumps({"secret": secret} if secret else {})
    import subprocess

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env,
        cwd=tmp_path, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


CASE_LINE = "size: 9; train: 10; test: 10; draws: 1; seed: 202"


def test_a_ranked_run_without_popcorn_seed_still_draws_a_secret_seed(tmp_path):
    # KernelBot's participant-visible run -- the one whose time is published --
    # is submitted with seed=None, so POPCORN_SEED is not in the environment.
    # Falling back to the public case seed would make every draw, every label
    # permutation and every hold-out position reproducible offline.
    first = run_main_and_capture(tmp_path, CASE_LINE)
    second = run_main_and_capture(tmp_path, CASE_LINE)
    assert "system.seed_source: random" in first["lines"]
    assert first["seed"] != 202 and second["seed"] != 202
    assert first["seed"] != second["seed"]
    assert first["code"] == harness.EXIT_VALIDATE_FAIL  # load_pools was stubbed out


def test_a_supplied_popcorn_seed_is_used_and_reported(tmp_path):
    result = run_main_and_capture(tmp_path, CASE_LINE, secret="20260922")
    assert "system.seed_source: popcorn" in result["lines"]
    # the probe runs mode "test", and main() salts the secret with the mode so
    # the cheap steps cannot preview the ranked one
    assert result["seed"] == combine(202, combine(20260922, harness.MODE_SALT["test"]))
    assert result["seed"] != combine(202, 20260922)


# ------------------------------------------------------------------ module-level inertness

def test_module_level_code_is_rejected(tmp_path):
    # KernelBot compiles a python submission by running it, before eval.py and
    # outside every guard this harness installs.
    case = dict(harness.DEFAULTS)
    for body in (
        "import os\nos.system('curl http://example.com')\n",
        "import mnist_data\nTABLE = mnist_data.load_pool\n",
        "LABELS = open('train-labels-idx1-ubyte').read()\n",
    ):
        path = write_submission(tmp_path, body)
        with pytest.raises(harness.Failure) as error:
            harness.check_submission_source(case, path)
        assert "module level" in str(error.value)


def test_module_level_imports_definitions_and_constants_are_allowed(tmp_path):
    case = dict(harness.DEFAULTS)
    path = write_submission(
        tmp_path,
        '"""doc."""\n'
        "import torch\n"
        "from task import input_t\n"
        "C, D = 10, 81\n"
        "MASK = (1 << 40) - 1\n"
        "SHAPES = {'x': [1, 2, 3]}\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.set_float32_matmul_precision('highest')\n"
        "class Net:\n    pass\n"
        "def custom_kernel(data):\n    return data[2]\n",
    )
    harness.check_submission_source(case, path)


# ------------------------------------------------------------------ mode time budget

def test_every_command_deadline_is_clamped_to_the_mode_budget():
    # A fixed per-command deadline can run the mode timeout out, and KernelBot
    # then records a bare TIMEOUT with no check line.
    child = harness.Child.__new__(harness.Child)
    child.deadline = time.perf_counter() + 10.0
    assert child.budget(None, "load") == pytest.approx(10.0, abs=0.5)
    assert child.budget(150.0, "untimed") == pytest.approx(10.0, abs=0.5)
    assert child.budget(2.0, "timed") == pytest.approx(2.0, abs=0.5)
    child.deadline = time.perf_counter() - 1.0
    with pytest.raises(harness.Failure):
        child.budget(150.0, "untimed")


def test_mode_deadlines_come_from_the_case_fields():
    case = dict(harness.DEFAULTS, test_timeout=300, benchmark_timeout=600, ranked_timeout=1200)
    now = time.perf_counter()
    assert harness.mode_deadline(case, "test") - now == pytest.approx(270, abs=1)
    assert harness.mode_deadline(case, "benchmark") - now == pytest.approx(570, abs=1)
    assert harness.mode_deadline(case, "leaderboard") - now == pytest.approx(1170, abs=1)


def test_task_yml_timeouts_and_case_timeouts_agree():
    config = json.loads((HERE.parent / "bands.json").read_text())
    for band in config["bands"]:
        task = (HERE.parent / band["name"] / "task.yml").read_text()
        top = {
            line.split(":")[0]: int(line.split(":")[1])
            for line in task.splitlines()
            if line.startswith(("test_timeout", "benchmark_timeout", "ranked_timeout"))
        }
        cases = [json.loads(line[4:]) for line in task.splitlines()
                 if line.startswith("  - {") and "error_bp" in line]
        for fields in cases:
            for key, value in top.items():
                assert fields[key] == value


# ------------------------------------------------------------------ per-band templates

def test_each_band_ships_a_template_naming_its_own_board():
    config = json.loads((HERE.parent / "bands.json").read_text())
    shared = (HERE.parent / "submission.py").read_text().splitlines()
    for band in config["bands"]:
        template = (HERE.parent / band["name"] / "submission.py").read_text().splitlines()
        assert template[0] == f"#!POPCORN leaderboard {band['name']}"
        assert template[1:] == shared[1:]
        assert f'Python: "submission.py"' in (HERE.parent / band["name"] / "task.yml").read_text()


def test_make_bands_check_notices_a_stale_readme(tmp_path):
    config = json.loads((HERE.parent / "bands.json").read_text())
    rendered = make_bands.render(config)
    assert "README.md" in rendered
    assert make_bands.README_END in rendered["README.md"]


# ------------------------------------------------------------------ timeout floor

def test_make_bands_mirrors_the_evaluators_own_timeout_constants():
    # make_bands.py is stdlib-only and cannot import eval.py, so it keeps its
    # own copies. If either side moves, the generated budgets go wrong quietly.
    assert make_bands.MODE_RESERVE_S == harness.MODE_RESERVE_S
    import inspect

    startup = inspect.signature(harness.Child.__init__).parameters["startup_timeout_s"]
    assert make_bands.CHILD_STARTUP_S == startup.default


def test_timeout_floor_counts_every_bounded_call_of_the_mode():
    settings = {
        "draws": 11,
        "bench_draws": 3,
        "holdout": 1,
        "holdout_draws": 2,
        "bench_holdout_draws": 1,
        "max_call_ms": 60000,
        "warmup_max_call_ms": 120000,
    }
    assert make_bands.timed_calls(settings, "test") == 1
    # benchmark runs its own hold-out call, and it is timed like the rest
    assert make_bands.timed_calls(settings, "benchmark") == 4
    # the hold-out calls are timed and ranked, so they are part of the budget
    assert make_bands.timed_calls(settings, "leaderboard") == 13

    fixed = (
        make_bands.POOL_LOAD_S
        + make_bands.CHILD_STARTUP_S
        + 120
        + make_bands.MODE_RESERVE_S
    )
    assert make_bands.timeout_floor(settings, "leaderboard") == fixed + 13 * 60
    assert make_bands.timeout_floor(settings, "test") == fixed + 60


def test_a_hold_out_free_band_does_not_reserve_hold_out_calls():
    settings = {
        "draws": 11,
        "bench_draws": 3,
        "holdout": 0,
        "holdout_draws": 2,
        "bench_holdout_draws": 1,
        "max_call_ms": 60000,
        "warmup_max_call_ms": 120000,
    }
    assert make_bands.timed_calls(settings, "leaderboard") == 11
    assert make_bands.timed_calls(settings, "benchmark") == 3


def test_every_shipped_band_can_afford_its_own_per_call_limit():
    config = json.loads((HERE.parent / "bands.json").read_text())
    assert make_bands.check_timeouts(config) == []


def test_check_timeouts_names_the_band_and_the_mode_that_cannot_pay():
    config = json.loads((HERE.parent / "bands.json").read_text())
    config["defaults"]["max_call_ms"] = 120000  # double the per-call limit
    problems = make_bands.check_timeouts(config)
    # benchmark and leaderboard both become unaffordable, for every band
    assert len(problems) == 2 * len(config["bands"])
    assert any("ranked_timeout" in line for line in problems)
    assert any("benchmark_timeout" in line for line in problems)
    assert all(
        any(band["name"] in line for line in problems) for band in config["bands"]
    )


def test_make_bands_refuses_to_generate_a_band_it_cannot_afford(tmp_path, monkeypatch, capsys):
    # The guard has to stop the write path too, not just --check: otherwise a
    # bad edit silently ships a task.yml whose max_call_ms is unenforceable.
    config = json.loads((HERE.parent / "bands.json").read_text())
    config["defaults"]["test_timeout"] = 60
    bad = tmp_path / "bands.json"
    bad.write_text(json.dumps(config))
    monkeypatch.setattr(make_bands, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_bands.py"])
    assert make_bands.main() == 1
    assert "timeout floor" in capsys.readouterr().out
    assert not (tmp_path / "sutro.yaml").exists()


def test_bands_md_publishes_the_mode_budget_table():
    text = (HERE.parent / "bands.md").read_text()
    assert "## Mode time budgets" in text
    config = json.loads((HERE.parent / "bands.json").read_text())
    settings = {**config["defaults"], **config["bands"][0]}
    floor = make_bands.timeout_floor(settings, "leaderboard")
    assert f"| {floor} s |" in text


# ------------------------------------------------------------------ hold-out policy

def test_benchmark_mode_runs_a_gated_hold_out_call():
    case = {"draws": 11, "bench_draws": 3, "holdout": 1,
            "holdout_draws": 2, "bench_holdout_draws": 1}
    assert harness.draw_plan(case, ranked=True) == (11, 2)
    # the cheap mode is a faithful rehearsal, not an unguarded one
    assert harness.draw_plan(case, ranked=False) == (3, 1)


def test_a_band_with_the_hold_out_off_runs_none_in_either_mode():
    case = {"draws": 11, "bench_draws": 3, "holdout": 0,
            "holdout_draws": 2, "bench_holdout_draws": 1}
    assert harness.draw_plan(case, ranked=True) == (11, 0)
    assert harness.draw_plan(case, ranked=False) == (3, 0)


def test_benchmark_never_runs_more_ranked_draws_than_the_band_has():
    case = {"draws": 2, "bench_draws": 3, "holdout": 1,
            "holdout_draws": 2, "bench_holdout_draws": 1}
    assert harness.draw_plan(case, ranked=False)[0] == 2


def test_the_hold_out_floor_separates_learning_from_memorising():
    """Every accuracy measured on the A100, against the 15% floor.

    The floor only has to tell a learner from a lookup table. It was 70%, which
    failed mlp512 -- an honest learner whose hand-tuned learning rate diverges
    on Fashion's denser images.
    """
    floor_bp = json.loads((HERE.parent / "bands.json").read_text())["defaults"]["holdout_min_bp"]
    total = 2 * 10000  # holdout_draws x test
    needed = required_correct(total, floor_bp)
    assert needed == 3000  # 15% of 20,000

    measured = {"pca_qda": 0.774, "cg_pair": 0.882, "mlp512": 0.478, "nearest_class_mean": 0.665}
    for name, accuracy in measured.items():
        assert round(accuracy * total) >= needed, f"{name} is an honest learner and must pass"
    # a lookup table cannot answer a permuted-label foreign dataset: chance is 10%
    assert round(0.10 * total) < needed
    # and the separation is wide in units of the noise at chance
    sigma = (0.10 * 0.90 / total) ** 0.5
    assert (needed / total - 0.10) / sigma > 20


def test_the_case_field_reaches_every_generated_band():
    for path in sorted((HERE.parent).glob("mnist-medium-*/task.yml")):
        text = path.read_text()
        assert '"bench_holdout_draws": 1' in text, path
        assert '"holdout_min_bp": 8500' in text, path


# ------------------------------------------------------------------ deadline

def test_the_shipped_deadline_parses_and_reaches_the_competition_file():
    config = json.loads((HERE.parent / "bands.json").read_text())
    import datetime as dt

    when = dt.datetime.strptime(config["deadline"], make_bands.DEADLINE_FORMAT)
    assert when.year >= 2026
    assert f'deadline: "{config["deadline"]}"' in (HERE.parent / "sutro.yaml").read_text()


def test_a_malformed_deadline_is_fatal_and_a_past_one_is_a_warning():
    import datetime as dt

    config = json.loads((HERE.parent / "bands.json").read_text())
    assert make_bands.check_deadline(config, dt.datetime(2026, 1, 1)) == []

    past = make_bands.check_deadline(config, dt.datetime(2099, 1, 1))
    assert len(past) == 1 and past[0].endswith("(warning)")

    config["deadline"] = "31/12/2026"
    bad = make_bands.check_deadline(config, dt.datetime(2026, 1, 1))
    assert len(bad) == 1 and not bad[0].endswith("(warning)")


def test_make_bands_refuses_a_malformed_deadline(tmp_path, monkeypatch, capsys):
    config = json.loads((HERE.parent / "bands.json").read_text())
    config["deadline"] = "whenever"
    (tmp_path / "bands.json").write_text(json.dumps(config))
    monkeypatch.setattr(make_bands, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_bands.py"])
    assert make_bands.main() == 1
    assert "deadline" in capsys.readouterr().out
    assert not (tmp_path / "sutro.yaml").exists()


# ------------------------------------------------------------------ per-mode draws

def test_every_mode_the_evaluator_accepts_has_its_own_salt():
    # main() dispatches on exactly these; a mode with no salt would silently
    # fall back to 0 and share its draws with any other unsalted mode.
    assert set(harness.MODE_SALT) == {"test", "benchmark", "leaderboard", "profile"}
    assert len(set(harness.MODE_SALT.values())) == len(harness.MODE_SALT)


def test_the_cheap_step_is_not_a_preview_of_the_ranked_one(tmp_path):
    """Every step of one submission gets the same POPCORN_SEED, and benchmark
    and leaderboard are handed the same case line. Before the mode salt they
    drew the same data: at secret 4242 both scored [1603, 1564, 1586] on the
    same three draws and saw the same hold-out draw. In the same container,
    with /tmp surviving between steps, that is a replay surface."""
    case = tmp_path / "cases.txt"
    case.write_text("size: 9; train: 100; test: 100; seed: 202; draws: 3\n")

    seeds = {}
    for mode, salt in harness.MODE_SALT.items():
        cases = harness.read_cases(case, combine(4242, salt))
        seeds[mode] = cases[0]["seed"]
    assert len(set(seeds.values())) == len(seeds), seeds

    # and the salt is what does it: without one every mode lands on one seed
    same = {mode: harness.read_cases(case, 4242)[0]["seed"] for mode in harness.MODE_SALT}
    assert len(set(same.values())) == 1


def test_the_public_case_seed_still_reaches_the_combined_seed():
    # the mode salt must compose with the secret, not replace it
    assert combine(4242, harness.MODE_SALT["benchmark"]) != 4242
    assert combine(4242, harness.MODE_SALT["benchmark"]) != harness.MODE_SALT["benchmark"]
