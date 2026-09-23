"""Tests for the submission website's budget ledger, validation, worker deadline and routes.

    uv run --python 3.12 --with modal --with "fastapi[standard]" --with pytest --with pyyaml \
        python -m pytest tests/test_web.py -q   # from the harness directory

No Modal account and no GPU: Modal is replaced by in-memory stores, a fake
spawner and a fake settler. The deadline and pipe tests spawn real (tiny)
subprocesses.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web"))
sys.path.insert(0, str(HERE.parent))

import app as site_app  # noqa: E402
import run_modal  # noqa: E402
from app import (  # noqa: E402
    MemoryLog,
    MemoryStore,
    Site,
    deadline_seconds,
    evaluate_job,
    ledger,
    load_task,
    reservation_usd,
    run_with_deadline,
    summarize,
    validate_source,
    worst_case_seconds,
)

TEMPLATE = (HERE.parent / "submission.py").read_text()
CPU_RESULT = HERE.parent / "results" / "cpu-pca-qda-5pct.json"


class FakeModal:
    """Spawned calls sit in ``pending`` until the test finishes or breaks them."""

    def __init__(self):
        self.jobs = {}
        self.outcomes = {}
        self.cancelled = []
        self.spawn_error = None
        self.transport_error = None

    def spawner(self, job):
        if self.spawn_error:
            raise self.spawn_error
        call_id = f"fc-{len(self.jobs) + 1}"
        self.jobs[call_id] = job
        return call_id

    def settler(self, call_id):
        if self.transport_error:
            raise self.transport_error
        return self.outcomes.get(call_id, ("pending", None))

    def canceller(self, call_id):
        self.cancelled.append(call_id)

    def finish(self, call_id, passed=True, billable_s=200.0, runs=None):
        self.outcomes[call_id] = ("done", {"runs": runs or {}, "passed": passed, "gpu": "A100",
                                           "billable_s": billable_s})

    def crash(self, call_id):
        self.outcomes[call_id] = ("error", "worker failed")


class Clock:
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        return self.now


@pytest.fixture
def fake():
    return FakeModal()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def site(fake, clock):
    return Site(MemoryStore(), MemoryStore(), MemoryLog(), fake.spawner, fake.settler, fake.canceller,
                default_budget=50.0, clock=clock)


# ------------------------------------------------------------------ validation

def test_template_is_a_valid_kernel():
    assert validate_source(TEMPLATE) is None


def test_validation_rejects_empty_syntax_errors_and_missing_kernel():
    assert "empty" in validate_source("   \n")
    assert "syntax error" in validate_source("def custom_kernel(data:\n    return 1")
    assert "custom_kernel" in validate_source("import torch\n\ndef other(x):\n    return x\n")
    assert "larger than" in validate_source("x = 1\n" + "#" * site_app.MAX_SOURCE_BYTES)


def test_validation_accepts_aliased_kernels():
    assert validate_source("from mylib import kernel as custom_kernel\n") is None
    assert validate_source("def f(d):\n    return d[1]\ncustom_kernel = f\n") is None


# ------------------------------------------------------------------ budget arithmetic

def test_reservations_follow_the_task_timeouts_and_the_worker_deadline():
    task = load_task("mnist-medium-5pct")
    # The timeouts themselves live in bands.json; read them rather than repeat
    # them, so raising one is a one-line edit there and not a test failure here.
    bands = json.loads((Path(site_app.__file__).resolve().parent.parent / "bands.json").read_text())
    defaults = bands["defaults"]
    assert task["test_timeout"] == defaults["test_timeout"]
    assert task["benchmark_timeout"] == defaults["benchmark_timeout"]
    assert task["ranked_timeout"] == defaults["ranked_timeout"]
    assert deadline_seconds(task, "test") == task["test_timeout"] + site_app.POOL_PAD_S
    assert deadline_seconds(task, "benchmark") == task["benchmark_timeout"] + site_app.POOL_PAD_S
    assert deadline_seconds(task, "leaderboard") == (
        task["test_timeout"] + task["benchmark_timeout"] + task["ranked_timeout"]
        + site_app.POOL_PAD_S
    )
    # the deadline the worker enforces always fits under Modal's own container timeout
    for mode in site_app.MODES:
        assert deadline_seconds(task, mode) + site_app.STARTUP_PAD_S <= site_app.GPU_TIMEOUT_S
        assert worst_case_seconds(task, mode) == deadline_seconds(task, mode) + site_app.STARTUP_PAD_S
        assert reservation_usd(task, mode) == round(worst_case_seconds(task, mode) * site_app.RATE_USD_PER_S, 4)
    assert 1.5 < reservation_usd(task, "leaderboard") < 2.5
    assert reservation_usd(task, "test") < reservation_usd(task, "benchmark") < reservation_usd(task, "leaderboard")


def test_ledger_counts_reservations_and_charges_with_a_durable_floor():
    items = [
        {"status": "queued", "reserved_usd": 1.5, "charged_usd": None},
        {"status": "passed", "reserved_usd": 1.5, "charged_usd": 0.25},
        {"status": "error", "reserved_usd": 0.4, "charged_usd": 0.4},
    ]
    book = ledger(items, 50.0)
    assert book["charged_usd"] == 0.65 and book["reserved_usd"] == 1.5
    assert book["remaining_usd"] == pytest.approx(50 - 2.15)
    assert book["inflight"] == 1
    # records that expired from the Dict are still counted through the charge log
    assert ledger(items, 50.0, charged_floor=3.0)["charged_usd"] == 3.0
    assert ledger(items, 50.0, charged_floor=0.1)["charged_usd"] == 0.65


# ------------------------------------------------------------------ submission lifecycle

def test_submit_reserves_persists_before_spawning_then_charges_measured_time(site, fake):
    record, rejection = site.submit("ann", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    assert rejection is None and record["status"] == "queued"
    reserved = record["reserved_usd"]
    assert site.ledger()["reserved_usd"] == reserved
    job = fake.jobs[record["call_id"]]
    assert job["source"] == TEMPLATE and job["seed"] == record["seed"]
    assert job["deadline_s"] == record["deadline_s"] == deadline_seconds(load_task("mnist-medium-5pct"), "leaderboard")
    assert "source" not in record and site.source(record["id"]) == TEMPLATE

    site.settle()  # still pending
    assert site.get(record["id"])["status"] == "queued"

    fake.finish(record["call_id"], passed=True, billable_s=300.0, runs={"test": {"passed": True, "exit_code": 0,
                                                                                "duration_s": 1, "result": {}}})
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "passed"
    assert done["charged_usd"] == pytest.approx(300 * site_app.RATE_USD_PER_S, abs=1e-4)
    assert done["overrun_usd"] == 0 and done["charged_usd"] < reserved
    assert site.runs(record["id"])["test"]["passed"] is True
    book = site.ledger()
    assert book["reserved_usd"] == 0 and book["charged_usd"] == done["charged_usd"]
    assert site.charges.total() == pytest.approx(done["charged_usd"])


def test_overruns_are_charged_in_full_and_flagged(site, fake):
    record, _ = site.submit("ann", "mnist-medium-5pct", "test", TEMPLATE)
    fake.finish(record["call_id"], billable_s=10_000.0)
    site.settle()
    done = site.get(record["id"])
    assert done["charged_usd"] == pytest.approx(10_000 * site_app.RATE_USD_PER_S, abs=1e-3)
    assert done["charged_usd"] > record["reserved_usd"]
    assert done["overrun_usd"] == pytest.approx(done["charged_usd"] - record["reserved_usd"], abs=1e-3)


def test_crashed_worker_is_charged_its_reservation_and_cancelled(site, fake):
    record, _ = site.submit("bob", "mnist-medium-3pct", "test", TEMPLATE)
    fake.crash(record["call_id"])
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "error" and done["charged_usd"] == record["reserved_usd"]
    assert "worker failed" in done["error"]
    assert fake.cancelled == [record["call_id"]]


def test_container_limit_timeout_is_charged_the_whole_container_life(site, fake):
    record, _ = site.submit("bob", "mnist-medium-3pct", "test", TEMPLATE)
    fake.outcomes[record["call_id"]] = ("timeout", "the run hit Modal's 2400 s container limit")
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "error"
    assert done["charged_usd"] == site_app.BACKSTOP_USD > record["reserved_usd"]
    assert done["overrun_usd"] == pytest.approx(site_app.BACKSTOP_USD - record["reserved_usd"], abs=1e-4)
    assert fake.cancelled == [record["call_id"]]
    assert site.charges.total() == pytest.approx(site_app.BACKSTOP_USD)


def test_failed_evaluation_still_charges_measured_time(site, fake):
    record, _ = site.submit("bob", "mnist-medium-2pct", "benchmark", TEMPLATE)
    fake.finish(record["call_id"], passed=False, billable_s=120.0)
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "failed"
    assert done["charged_usd"] == pytest.approx(120 * site_app.RATE_USD_PER_S, abs=1e-4)


def test_missing_billable_time_is_charged_at_the_worst_case(site, fake):
    record, _ = site.submit("bob", "mnist-medium-2pct", "test", TEMPLATE)
    fake.outcomes[record["call_id"]] = ("done", {"runs": {}, "passed": False})
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "failed" and done["charged_usd"] == record["reserved_usd"]


def test_spawn_failure_releases_the_reservation_without_a_gpu(site, fake):
    fake.spawn_error = RuntimeError("modal is down")
    record, rejection = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    assert record is None and "could not start" in rejection
    stored = site.submissions()[0]
    assert stored["status"] == "error" and stored["charged_usd"] == 0 and stored["call_id"] is None
    book = site.ledger()
    assert book["reserved_usd"] == 0 and book["charged_usd"] == 0


def test_lost_call_id_is_written_off_after_the_grace_period(site, fake, clock):
    record, _ = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    record["call_id"] = None  # the spawn happened but the id was never persisted
    site.save(record)
    site.settle()
    assert site.get(record["id"])["status"] == "queued"
    clock.now += site_app.LOST_CALL_GRACE_S + 1
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "error" and done["charged_usd"] == record["reserved_usd"]


def test_transport_errors_are_retried_then_written_off(site, fake):
    record, _ = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    fake.transport_error = ConnectionError("gRPC unavailable")
    for attempt in range(1, site_app.MAX_SETTLE_FAILURES):
        site.settle()
        current = site.get(record["id"])
        assert current["status"] == "queued" and current["settle_failures"] == attempt
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "error" and done["charged_usd"] == record["reserved_usd"]
    assert fake.cancelled == [record["call_id"]]
    # a successful poll resets the counter
    other, _ = site.submit("b", "mnist-medium-5pct", "test", TEMPLATE)
    site.settle()
    fake.transport_error = None
    site.settle()
    assert site.get(other["id"])["settle_failures"] == 0


def test_malformed_payload_becomes_an_error_record_not_a_500(site, fake):
    record, _ = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    fake.outcomes[record["call_id"]] = ("done", "not a dict")
    site.settle()
    done = site.get(record["id"])
    assert done["status"] == "error" and done["charged_usd"] == record["reserved_usd"]
    assert "could not be interpreted" in done["error"]
    other, _ = site.submit("b", "mnist-medium-5pct", "test", TEMPLATE)
    fake.finish(other["call_id"], runs={"leaderboard": {"passed": True, "exit_code": 0, "duration_s": 1,
                                                        "result": {"benchmark.0.mean": "garbage",
                                                                   "benchmark.0.per_draw": "{not json"}}})
    site.settle()
    done = site.get(other["id"])
    assert done["status"] == "passed"
    assert done["summary"]["mean_ms"] is None and done["summary"]["per_draw"] is None


def test_budget_cap_counts_reservations_of_unfinished_runs(fake):
    task = load_task("mnist-medium-5pct")
    one_run = reservation_usd(task, "leaderboard")
    site = Site(MemoryStore(), MemoryStore(), MemoryLog(), fake.spawner, fake.settler,
                default_budget=2.5 * one_run)
    first, _ = site.submit("a", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    second, _ = site.submit("b", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    third, rejection = site.submit("c", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    assert first and second and third is None
    assert "budget" in rejection
    # a cheaper mode may still fit in what is left
    cheap, rejection = site.submit("c", "mnist-medium-5pct", "test", TEMPLATE)
    assert cheap is not None, rejection
    # once the first run settles cheaply, room opens up again
    fake.finish(first["call_id"], billable_s=100.0)
    fourth, rejection = site.submit("d", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    assert fourth is not None, rejection


def test_charge_log_floor_survives_losing_the_records(fake):
    store = MemoryStore()
    log = MemoryLog()
    site = Site(store, MemoryStore(), log, fake.spawner, fake.settler, default_budget=1.0)
    record, _ = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    fake.finish(record["call_id"], billable_s=1000.0)
    site.settle()
    charged = site.get(record["id"])["charged_usd"]
    store.data = {k: v for k, v in store.data.items() if not k.startswith("sub:")}  # the Dict expired
    assert site.ledger()["charged_usd"] == pytest.approx(charged)
    assert site.submit("b", "mnist-medium-5pct", "test", TEMPLATE)[0] is None


def test_refresh_rewrites_every_key(site, fake):
    site.token()
    site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    written = []
    original = site.store.put
    site.store.put = lambda key, value: (written.append(key), original(key, value))
    count = site.refresh()
    assert count == len(site.store.items()) + len(site.blobs.items())
    assert "config:token" in written and any(k.startswith("sub:") for k in written)


def test_refresh_migrates_first_deployment_records(site):
    legacy = {"id": "20200101-000000-abcdef", "created_at": "2020-01-01T00:00:00Z", "name": "x",
              "band": "mnist-medium-5pct", "mode": "test", "status": "passed", "reserved_usd": 0.4,
              "charged_usd": 0.05, "source": "def custom_kernel(d): return d[1]", "runs": {"test": {}}}
    site.save(legacy)
    site.refresh()
    migrated = site.get(legacy["id"])
    assert "source" not in migrated and "runs" not in migrated
    assert site.source(legacy["id"]) == "def custom_kernel(d): return d[1]"
    assert site.runs(legacy["id"]) == {"test": {}}


def test_cancel_writes_off_a_pending_run(site, fake):
    record, _ = site.submit("a", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    cancelled = site.cancel(record["id"])
    assert cancelled["status"] == "error" and cancelled["charged_usd"] == record["reserved_usd"]
    assert fake.cancelled == [record["call_id"]]
    assert site.cancel(record["id"])["status"] == "error"  # idempotent


def test_budget_can_be_changed_at_runtime(site, fake):
    site.set_budget(0.01)
    record, rejection = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    assert record is None and "budget" in rejection
    site.set_budget(50)
    record, rejection = site.submit("a", "mnist-medium-5pct", "test", TEMPLATE)
    assert record is not None


def test_queue_length_is_capped(site, fake):
    for _ in range(site_app.MAX_INFLIGHT):
        record, rejection = site.submit("a", "mnist-medium-12pct", "test", TEMPLATE)
        assert record is not None, rejection
    record, rejection = site.submit("a", "mnist-medium-12pct", "test", TEMPLATE)
    assert record is None and "queued" in rejection


def test_submit_rejects_bad_inputs_without_spawning(site, fake):
    assert site.submit("a", "mnist-medium-1pct", "test", TEMPLATE)[0] is None
    assert site.submit("a", "mnist-medium-5pct", "profile", TEMPLATE)[0] is None
    assert site.submit("a", "mnist-medium-5pct", "test", "print(1)")[0] is None
    assert fake.jobs == {} and site.submissions() == []


# ------------------------------------------------------------------ result summaries

@pytest.mark.skipif(not CPU_RESULT.exists(), reason="archived CPU result not present")
def test_summarize_reads_the_ranked_numbers_from_a_real_payload():
    payload = json.loads(CPU_RESULT.read_text())
    ranked = payload["runs"]["leaderboard"]["result"]
    summary = summarize(payload)
    assert summary["verdict"] == "pass" and summary["ranked_step"] == "leaderboard"
    assert summary["mean_ms"] == pytest.approx(float(ranked["benchmark.0.mean"]) / 1e6)
    assert summary["accuracy_pct"] == pytest.approx(100 * float(ranked["benchmark.0.accuracy"]))
    assert summary["correct"] == int(ranked["benchmark.0.correct"])
    assert summary["total"] == int(ranked["benchmark.0.total"])
    assert summary["required"] == int(ranked["benchmark.0.required"])
    assert summary["per_draw"] == json.loads(ranked["benchmark.0.per_draw"])
    assert sum(summary["per_draw"]) == summary["correct"]
    assert summary["holdout_pct"] == pytest.approx(100 * float(ranked["benchmark.0.holdout_accuracy"]))
    assert [s[0] for s in summary["steps"]] == ["test", "benchmark", "leaderboard"]
    assert summary["system"]["torch"] == ranked["system.torch"]


def test_summarize_test_mode_failures_and_statless_leaderboard():
    payload = {"passed": True, "gpu": "A100", "runs": {"test": {
        "passed": True, "exit_code": 0, "duration_s": 3.0, "stderr": "",
        "result": {"test.0.message": "9000/10000 correct (90.00%), needs 8500 at the test-mode slack "
                                     "and 9500 on the leaderboard; 1.234 ms per call", "check": "pass"}}}}
    summary = summarize(payload)
    assert summary["mean_ms"] == 1.234 and summary["accuracy_pct"] == 90.0
    assert "ranked_step" not in summary

    failed = {"passed": False, "runs": {"test": {
        "passed": False, "exit_code": 112, "duration_s": 3.0, "stderr": "",
        "result": {"test.0.status": "fail", "test.0.error": "accuracy below the test-mode threshold",
                   "check": "fail"}}}}
    assert summarize(failed)["error"] == "accuracy below the test-mode threshold"

    crashed = {"passed": False, "runs": {"test": {
        "passed": False, "exit_code": 1, "duration_s": 1.0, "result": {},
        "stderr": "Traceback\nModuleNotFoundError: No module named 'triton'"}}}
    assert "triton" in summarize(crashed)["error"]

    deadline = {"passed": False, "runs": {}, "error": "the whole evaluation exceeded its 480 s deadline"}
    assert "deadline" in summarize(deadline)["error"] and summarize(deadline)["last_step"] is None

    stats = {"benchmark.0.mean": "2e6", "benchmark.0.std": "1e5", "benchmark.0.best": "1.9e6",
             "benchmark.0.median": "2e6", "benchmark.0.runs": "3", "benchmark.0.accuracy": "0.95",
             "benchmark.0.correct": "28500", "benchmark.0.total": "30000", "benchmark.0.required": "28500",
             "benchmark.0.per_draw": "[9500, 9500, 9500]", "system.torch": "x"}
    statless = {"passed": False, "runs": {
        "test": {"passed": True, "exit_code": 0, "duration_s": 1, "result": {"check": "pass"}},
        "benchmark": {"passed": True, "exit_code": 0, "duration_s": 1, "result": stats},
        "leaderboard": {"passed": False, "exit_code": -1, "duration_s": 1200, "result": {}, "stderr": "killed"}}}
    summary = summarize(statless)
    assert summary["ranked_step"] == "benchmark" and summary["mean_ms"] == 2.0 and summary["draws"] == 3
    assert summary["error"] == "killed" and summary["system"]["torch"] == "x"


# ------------------------------------------------------------------ worker deadline

SLEEPER = """
import os, subprocess, sys, time
# a grandchild that outlives us and would keep any inherited pipe open
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print("started", child.pid, flush=True)
time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_run_with_deadline_kills_the_whole_process_group():
    started = time.perf_counter()
    code, log, timed_out = run_with_deadline([sys.executable, "-c", SLEEPER], deadline_s=1.5)
    assert timed_out and code is None
    assert time.perf_counter() - started < 10
    grandchild = int(log.split()[1])
    time.sleep(0.2)
    assert not _alive(grandchild)


def test_run_with_deadline_returns_normally():
    code, log, timed_out = run_with_deadline([sys.executable, "-c", "print('hi')"], deadline_s=5)
    assert (code, timed_out) == (0, False) and "hi" in log


def test_evaluate_job_reports_a_deadline_overrun(tmp_path):
    harness = tmp_path / "harness"
    (harness / "web").mkdir(parents=True)
    (harness / "web" / "runner.py").write_text("import time\ntime.sleep(60)\n")
    job = {"id": "x", "band": "mnist-medium-5pct", "mode": "test", "source": "", "seed": 1, "deadline_s": 1.0}
    payload = evaluate_job(job, harness, tmp_path / "work")
    assert payload["passed"] is False and "deadline" in payload["error"] and payload["runs"] == {}


def test_evaluate_job_reads_the_runner_payload(tmp_path):
    harness = tmp_path / "harness"
    (harness / "web").mkdir(parents=True)
    (harness / "web" / "runner.py").write_text(
        "import json, sys\njson.dump({'runs': {}, 'passed': True, 'marker': 7}, open(sys.argv[2], 'w'))\n")
    job = {"id": "x", "band": "mnist-medium-5pct", "mode": "test", "source": "", "seed": 1, "deadline_s": 10}
    payload = evaluate_job(job, harness, tmp_path / "work")
    assert payload["passed"] is True and payload["marker"] == 7


def test_runner_builds_the_same_sources_as_run_modal(tmp_path):
    sys.path.insert(0, str(HERE.parent / "web"))
    import runner

    job = {"id": "x", "band": "mnist-medium-3pct", "mode": "test", "source": "SUB", "seed": 1}
    task, sources = runner.sources_for(job, HERE.parent / "mnist-medium-3pct")
    submission = tmp_path / "s.py"
    submission.write_text("SUB")
    expected = run_modal.collect_sources(HERE.parent / "mnist-medium-3pct", task, submission)
    assert sources == expected and set(sources) >= {"submission.py", "eval.py", "task.py", "utils.py"}


def test_runner_lifts_the_baked_pool_into_memory_and_deletes_it(tmp_path):
    import runner

    assert runner.lift_baked_pool(tmp_path / "missing") is None
    baked = tmp_path / "pool"
    baked.mkdir()
    (baked / "a.gz").write_bytes(b"AAA")
    (baked / "b.gz").write_bytes(b"BB")
    (baked / "notes.txt").write_text("ignored")
    pool = runner.lift_baked_pool(baked)
    assert pool == {"a.gz": b"AAA", "b.gz": b"BB"}
    assert not baked.exists()


HOLDING_EVAL = """
import os, subprocess, sys, time
fd = int(os.environ["POPCORN_FD"])
os.write(fd, b"check: pass\\n")
# a grandchild inherits the result pipe and never lets go of it
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], pass_fds=[fd])
time.sleep(60)
"""


def test_run_one_cannot_be_held_past_its_timeout_by_an_orphan(tmp_path):
    (tmp_path / "eval.py").write_text(HOLDING_EVAL)
    started = time.perf_counter()
    run = run_modal.run_one(tmp_path, "test", "size: 9\n", timeout=2, seed=1, env_extra={})
    assert time.perf_counter() - started < 15
    assert run["exit_code"] == -1 and run["passed"] is True  # the lines it did write are kept
    assert run["result"] == {"check": "pass"}


def test_run_one_normal_path_matches_the_old_contract(tmp_path):
    (tmp_path / "eval.py").write_text(
        "import os, sys\nfd = int(os.environ['POPCORN_FD'])\n"
        "os.write(fd, ('seed: ' + os.environ['POPCORN_SEED'] + '\\nmode: ' + sys.argv[1] + '\\ncheck: pass\\n').encode())\n"
        "print('out'); print('err', file=sys.stderr)\n")
    run = run_modal.run_one(tmp_path, "benchmark", "size: 9\n", timeout=10, seed=42, env_extra={})
    assert run["passed"] and run["exit_code"] == 0
    assert run["result"] == {"seed": "42", "mode": "benchmark", "check": "pass"}
    assert run["stdout"].strip() == "out" and run["stderr"].strip() == "err"


# ------------------------------------------------------------------ http routes

@pytest.fixture
def client(site):
    from fastapi.testclient import TestClient

    return TestClient(site_app.build_api(site, TEMPLATE))


def test_wrong_or_odd_tokens_are_404s(client, site):
    assert client.get("/").status_code == 404
    assert client.get("/not-the-token").status_code == 404
    assert client.get("/caf%C3%A9").status_code == 404
    assert client.get("/caf%C3%A9/api/submissions").status_code == 404
    assert client.get("/not-the-token/api/submissions").status_code == 404
    assert client.post("/not-the-token/submit", data={"band": "mnist-medium-5pct", "mode": "test",
                                                     "source": TEMPLATE}).status_code == 404


def test_index_submit_and_detail_pages(client, site, fake):
    token = site.token()
    index = client.get(f"/{token}")
    assert index.status_code == 200 and "Shared A100 budget" in index.text and "custom_kernel" in index.text

    response = client.post(f"/{token}/submit", data={"name": "ann", "band": "mnist-medium-3pct",
                                                     "mode": "test", "source": TEMPLATE},
                           follow_redirects=False)
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith(f"/{token}/s/")
    detail = client.get(location)
    assert detail.status_code == 200 and "running" in detail.text and "http-equiv=\"refresh\"" in detail.text

    sid = location.rsplit("/", 1)[-1]
    fake.finish(site.get(sid)["call_id"], billable_s=90.0, runs={"test": {
        "passed": True, "exit_code": 0, "duration_s": 2.0, "stderr": "warn <b>",
        "result": {"test.0.message": "9000/10000 correct (90.00%); 1.5 ms per call", "check": "pass"}}})
    detail = client.get(location)
    assert "PASS" in detail.text and "refresh" not in detail.text
    assert "test output" in detail.text and "&lt;b&gt;" in detail.text
    listing = client.get(f"/{token}/api/submissions").json()
    assert listing["submissions"][0]["status"] == "passed"
    assert "source" not in listing["submissions"][0] and "seed" not in listing["submissions"][0]
    assert listing["ledger"]["charged_usd"] == pytest.approx(90 * site_app.RATE_USD_PER_S, abs=1e-4)
    assert client.get(f"{location}/source").text == TEMPLATE
    assert client.get(f"/{token}/api/s/{sid}").json()["charged_usd"] == listing["ledger"]["charged_usd"]


def test_benchmark_mode_is_not_labelled_as_the_ranked_number(client, site, fake):
    token = site.token()
    record, _ = site.submit("a", "mnist-medium-5pct", "benchmark", TEMPLATE)
    stats = {"benchmark.0.mean": "2e6", "benchmark.0.std": "0", "benchmark.0.best": "2e6",
             "benchmark.0.median": "2e6", "benchmark.0.runs": "3", "benchmark.0.accuracy": "0.95",
             "benchmark.0.correct": "28500", "benchmark.0.total": "30000", "benchmark.0.required": "28500",
             "benchmark.0.per_draw": "[9500, 9500, 9500]"}
    fake.finish(record["call_id"], runs={"benchmark": {"passed": True, "exit_code": 0, "duration_s": 1, "result": stats}})
    page = client.get(f"/{token}/s/{record['id']}").text
    assert "not the ranked number" in page and "3 draws" in page


def test_error_and_pending_pages_render_for_sparse_records(client, site, fake):
    token = site.token()
    record, _ = site.submit("a", "mnist-medium-5pct", "leaderboard", TEMPLATE)
    fake.crash(record["call_id"])
    page = client.get(f"/{token}/s/{record['id']}").text
    assert "Infrastructure error" in page and "worker failed" in page
    # a record missing optional fields (older schema) must not 500
    sparse = {"id": "20200101-000000-abcdef", "created_at": "2020-01-01T00:00:00Z", "name": "x",
              "band": "mnist-medium-5pct", "mode": "test", "status": "passed", "reserved_usd": 0.4}
    site.save(sparse)
    assert client.get(f"/{token}/s/{sparse['id']}").status_code == 200
    assert client.get(f"/{token}").status_code == 200


def test_file_upload_replaces_textarea(client, site, fake):
    token = site.token()
    response = client.post(f"/{token}/submit", data={"band": "mnist-medium-5pct", "mode": "benchmark",
                                                     "source": ""},
                           files={"file": ("k.py", TEMPLATE.encode(), "text/x-python")},
                           follow_redirects=False)
    assert response.status_code == 303
    sid = response.headers["location"].rsplit("/", 1)[-1]
    assert site.source(sid) == TEMPLATE


def test_bad_upload_keeps_the_typed_kernel(client, site):
    token = site.token()
    response = client.post(f"/{token}/submit", data={"band": "mnist-medium-5pct", "mode": "test",
                                                     "source": "def custom_kernel(d):\n    return d[1]\n"},
                           files={"file": ("k.py", b"\xff\xfe\x00bad", "text/x-python")})
    assert response.status_code == 400
    assert "not UTF-8" in response.text and "def custom_kernel(d)" in response.text
    big = client.post(f"/{token}/submit", data={"band": "mnist-medium-5pct", "mode": "test", "source": ""},
                      files={"file": ("k.py", b"#" * (site_app.MAX_SOURCE_BYTES + 1), "text/x-python")})
    assert big.status_code == 400 and "larger than" in big.text
    assert site.submissions() == []


def test_oversized_requests_are_refused_up_front(client, site):
    token = site.token()
    response = client.post(f"/{token}/submit", headers={"content-length": str(site_app.MAX_REQUEST_BYTES + 1)},
                           content=b"")
    assert response.status_code == 413


def test_rejection_shows_the_reason_and_keeps_the_kernel(client, site):
    token = site.token()
    response = client.post(f"/{token}/submit", data={"band": "mnist-medium-5pct", "mode": "test",
                                                     "source": "def nothing():\n    pass\n"})
    assert response.status_code == 400
    assert "Not submitted" in response.text and "custom_kernel" in response.text
    assert "def nothing" in response.text
    assert site.submissions() == []


def test_the_container_backstop_never_truncates_the_evaluator():
    """deadline_seconds() clamps to Modal's container timeout. That clamp must
    never bind: if it does, the worker kills a run that is still inside the
    per-step budgets its own task.yml publishes, and the submitter sees a bare
    timeout instead of a verdict."""
    root = Path(site_app.__file__).resolve().parent.parent
    for band in json.loads((root / "bands.json").read_text())["bands"]:
        task = load_task(band["name"])
        for mode in site_app.MODES:
            uncapped = sum(
                {"test": task["test_timeout"],
                 "benchmark": task["benchmark_timeout"],
                 "leaderboard": task["ranked_timeout"]}[step]
                for step in site_app.STEP_SEQUENCE[mode]
            ) + site_app.POOL_PAD_S
            assert deadline_seconds(task, mode) == uncapped, (
                f"{band['name']}/{mode}: the container backstop is truncating "
                f"{uncapped:.0f} s of step budget to {deadline_seconds(task, mode):.0f} s"
            )
            assert worst_case_seconds(task, mode) <= site_app.GPU_TIMEOUT_S


# ------------------------------------------------------------------ board variants

def test_board_label_reduces_a_device_name_to_its_capacity():
    assert site_app.board_label("NVIDIA A100-SXM4-40GB") == "40GB"
    assert site_app.board_label("NVIDIA A100-SXM4-80GB") == "80GB"
    assert site_app.board_label("NVIDIA A100 80GB PCIe") == "80GB"
    assert site_app.board_label(None) is None
    assert site_app.board_label("   ") is None
    # an unrecognised name still identifies the board rather than vanishing
    assert site_app.board_label("NVIDIA H100") == "NVIDIA H100"


def test_only_runs_with_a_ranked_time_count_towards_the_board_set():
    items = [
        {"gpu": "NVIDIA A100-SXM4-40GB", "summary": {"mean_ms": 4.5}},
        {"gpu": "NVIDIA A100-SXM4-80GB", "summary": {"mean_ms": None}},  # no time yet
        {"gpu": None, "summary": {"mean_ms": 9.0}},
    ]
    assert site_app.boards_in(items) == ["40GB"]


def test_a_single_board_gets_no_warning_and_two_boards_do():
    one = [{"gpu": "NVIDIA A100-SXM4-40GB", "summary": {"mean_ms": 4.5}}]
    assert site_app.board_warning(one) == ""
    two = one + [{"gpu": "NVIDIA A100-SXM4-80GB", "summary": {"mean_ms": 4.3}}]
    warning = site_app.board_warning(two)
    assert "Mixed hardware" in warning and "40GB" in warning and "80GB" in warning


def test_the_index_shows_the_board_column_and_warns_on_mixed_hardware(client, site, fake):
    for index, gpu in enumerate(("NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB")):
        site.save({
            "id": f"bd{index}", "status": "passed", "name": "ann", "band": "mnist-medium-5pct",
            "mode": "leaderboard", "created_at": site_app.utcnow(), "gpu": gpu,
            "reserved_usd": 2.0, "charged_usd": 0.07,
            "summary": {"mean_ms": 4.5 - 0.2 * index, "accuracy_pct": 95.3, "verdict": "pass"},
        })
    body = client.get(f"/{site.token()}").text
    assert "<th>board</th>" in body
    assert ">40GB<" in body and ">80GB<" in body
    assert "Mixed hardware" in body
