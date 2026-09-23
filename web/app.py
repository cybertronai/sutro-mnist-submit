"""Submission website for the MNIST-medium time challenge, hosted on Modal.

One link, no login. Anyone holding the link can paste a ``submission.py``,
pick a band (2%, 3%, 5%, 8% or 12% mean error) and a mode, and the kernel is
evaluated on one A100 in the owner's Modal account by the same ``eval.py`` the
leaderboard uses. The page then shows the ranked time, the accuracy and what
the run cost.

Budget
  A global cap (``$50`` by default) is enforced by a ledger. Every submission
  reserves its worst case before it is queued: the evaluator's per-mode
  timeouts plus a dataset pad, which the worker enforces as a hard deadline on
  the whole evaluation (process group killed when it passes), plus a
  start-up pad, priced at the A100 80 GB rate with a 20% overhead for CPU,
  memory and rounding. Once the result is in, the run is charged its measured
  container time, uncapped, so an overrun shows up in the ledger instead of
  being absorbed. A run whose container dies, times out or is cancelled is
  charged its reservation. Nothing is queued once charged + reserved would
  exceed the cap. Records live in a ``modal.Dict`` (refreshed by a cron so the
  7-day idle expiry never fires while deployed); every charge is also written
  to a ``modal.Volume`` as a durable floor for the ledger. Modal's billing
  dashboard remains the source of truth.

Security
  Security through obscurity, as requested: the only protection is the random
  token in the URL. The A100 container runs the submitter's code with
  ``restrict_modal_access=True`` so it cannot use the container's Modal
  credentials to spawn more GPU work, is single-use so nothing survives into
  the next submitter's run, has no network egress (the datasets are baked into
  the image), and ``eval.py`` runs the kernel in a separate process behind an
  audit hook. That is a guard rail, not a sandbox.

Deploy and get the link (run from the harness directory: the repository root
of sutro-mnist-submit, or gpumode/ inside cybertronai/sutro-problems):

    uvx modal deploy web/app.py
    uvx modal run web/app.py::link

Other owner commands:

    uvx modal run web/app.py::status                 # ledger + submissions
    uvx modal run web/app.py::set_budget --usd 75    # raise or lower the cap
    uvx modal run web/app.py::cancel --sid <id>      # kill a running submission
    uvx modal run web/app.py::rotate_token           # invalidate the old link
    uvx modal run web/app.py::export --output subs.json
"""

import ast
import datetime as dt
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

import modal

APP_NAME = "sutro-mnist-submit"
RECORDS_NAME = f"{APP_NAME}-records"  # small records, token, budget
BLOBS_NAME = f"{RECORDS_NAME}-blobs"  # kernel text and evaluator output
LEDGER_VOLUME_NAME = f"{APP_NAME}-ledger"  # one file per charge
LEDGER_MOUNT = "/ledger"
REMOTE_HARNESS = "/root/harness"
LOCAL_HARNESS = Path(__file__).resolve().parent.parent
HARNESS = LOCAL_HARNESS if modal.is_local() else Path(REMOTE_HARNESS)
sys.path.insert(0, str(HARNESS))

import run_modal  # noqa: E402  (stdlib-only at import time)

# ------------------------------------------------------------------ pricing

# modal.com/pricing on 2026-09-22: A100 40 GB $0.000583/s ($2.10/h), A100 80 GB
# $0.000694/s ($2.50/h). KernelBot's gpu string "A100" is fulfilled by either
# part (smoke runs landed on both), so every run is priced at the 80 GB rate.
# CPU ($0.047/core/h) and memory ($0.008/GiB/h) for the 2-core, 8 GiB container
# add under 8%; the 20% overhead also covers image pulls and per-second rounding.
A100_USD_PER_S = 0.000694
OVERHEAD = 1.20
RATE_USD_PER_S = A100_USD_PER_S * OVERHEAD
STARTUP_PAD_S = 60.0  # image pull, container boot and teardown around the deadline
POOL_PAD_S = 180.0  # dataset download and box-area resize inside eval.py
GPU_TIMEOUT_S = 3000  # Modal's backstop on the container; the worker's own deadline is shorter.
# It has to stay above sum(step timeouts) + POOL_PAD_S + STARTUP_PAD_S, or the
# min() in deadline_seconds() starts binding and the worker kills a legitimate
# run before the evaluator has spent the budget its own task.yml promises.
# tests/test_web.py::test_the_container_backstop_never_truncates_the_evaluator
# holds that line for every band and mode.
DEFAULT_BUDGET_USD = float(os.environ.get("SUTRO_BUDGET_USD", "50"))
MAX_INFLIGHT = 8
MAX_SOURCE_BYTES = 256 * 1024
MAX_REQUEST_BYTES = MAX_SOURCE_BYTES + 64 * 1024
LOST_CALL_GRACE_S = 180.0  # a queued record with no call id after this long is written off
MAX_SETTLE_FAILURES = 5  # consecutive transport failures before a run is written off

BANDS = ["mnist-medium-2pct", "mnist-medium-3pct", "mnist-medium-5pct",
         "mnist-medium-8pct", "mnist-medium-12pct"]
MODES = {
    "test": "test: one draw, format check and a loose accuracy gate (cheapest)",
    "benchmark": "benchmark: 3 timed draws and the accuracy gate, no hold-out (a dress rehearsal)",
    "leaderboard": "leaderboard: test + benchmark + 11 timed draws with the hold-out check (the ranked protocol)",
}
STEP_SEQUENCE = {
    "test": ["test"],
    "benchmark": ["benchmark"],
    "leaderboard": ["test", "benchmark", "leaderboard"],
}
PENDING = ("queued",)
FINAL = ("passed", "failed", "error")

# ------------------------------------------------------------------ modal objects


def _ignore(path: Path) -> bool:
    parts = set(path.parts)
    if parts & {"__pycache__", ".pytest_cache", "results", "redteam", ".venv"}:
        return True
    return path.suffix in {".pyc", ".json"} and path.name != "bands.json"


web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]>=0.115", "PyYAML>=6.0")
    .add_local_dir(LOCAL_HARNESS, REMOTE_HARNESS, ignore=_ignore)
)

# The worker runs on run_modal.py's own image: KernelBot's CUDA base and pins,
# the MNIST and Fashion-MNIST files baked in at build time (the container has
# no egress), and the harness mounted at /root/harness. Reusing it keeps the
# site and `python run_modal.py` on identical software.
gpu_image = run_modal.image

app = modal.App(APP_NAME)
records = modal.Dict.from_name(RECORDS_NAME, create_if_missing=True)
ledger_volume = modal.Volume.from_name(LEDGER_VOLUME_NAME, create_if_missing=True)

# ------------------------------------------------------------------ pure logic


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(stamp: str) -> float:
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def new_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def load_task(band: str) -> dict:
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}")
    return run_modal.load_task(HARNESS / band)


def deadline_seconds(task: dict, mode: str) -> float:
    """The hard deadline the worker enforces on the whole evaluation."""
    timeouts = {
        "test": task.get("test_timeout", 180),
        "benchmark": task.get("benchmark_timeout", 180),
        "leaderboard": task.get("ranked_timeout", 180),
    }
    inside = sum(timeouts[step] for step in STEP_SEQUENCE[mode]) + POOL_PAD_S
    return min(inside, GPU_TIMEOUT_S - STARTUP_PAD_S)


def worst_case_seconds(task: dict, mode: str) -> float:
    """Upper bound on the A100 container lifetime for one submission."""
    return deadline_seconds(task, mode) + STARTUP_PAD_S


def reservation_usd(task: dict, mode: str) -> float:
    return round(worst_case_seconds(task, mode) * RATE_USD_PER_S, 4)


def validate_source(source: str) -> str | None:
    """Return a rejection reason, or None when the kernel is worth a GPU."""
    if not source.strip():
        return "the kernel is empty"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return f"the kernel is larger than {MAX_SOURCE_BYTES // 1024} KB"
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return f"syntax error: line {error.lineno}: {error.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "custom_kernel":
            return None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "custom_kernel" for target in node.targets
        ):
            return None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.asname or alias.name for alias in node.names]
            if "custom_kernel" in names:
                return None
    return "no `custom_kernel` is defined; see the template"


def ledger(items: list[dict], budget: float, charged_floor: float = 0.0) -> dict:
    """Totals over the records, with the durable charge log as a floor."""
    charged = sum(float(r.get("charged_usd") or 0) for r in items if r["status"] in FINAL)
    charged = max(charged, float(charged_floor or 0))
    reserved = sum(float(r.get("reserved_usd") or 0) for r in items if r["status"] in PENDING)
    committed = charged + reserved
    return {
        "budget_usd": budget,
        "charged_usd": round(charged, 4),
        "reserved_usd": round(reserved, 4),
        "committed_usd": round(committed, 4),
        "remaining_usd": round(budget - committed, 4),
        "inflight": sum(1 for r in items if r["status"] in PENDING),
    }


def _first_error(result: dict) -> str | None:
    for key, value in result.items():
        if key == "error" or str(key).endswith(".error"):
            return str(value)
    return None


def _num(result: dict, key: str, cast=float):
    try:
        return cast(result[key])
    except (KeyError, TypeError, ValueError):
        return None


def _scaled(result: dict, key: str, divisor: float = 1.0):
    value = _num(result, key)
    return None if value is None else value / divisor


def summarize(payload: dict) -> dict:
    """Pull the numbers a submitter cares about out of the worker's payload.

    Tolerates anything the evaluator (or a submission writing to the result
    pipe) may have put in the result lines: a bad field becomes None, never an
    exception.
    """
    runs = payload.get("runs") or {}
    if not isinstance(runs, dict):
        runs = {}
    summary: dict = {
        "verdict": "pass" if payload.get("passed") else "fail",
        "steps": [],
        "gpu": payload.get("gpu"),
        "error": payload.get("error"),
        "system": {},
    }
    for step, run in runs.items():
        if not isinstance(run, dict):
            continue
        summary["steps"].append((step, bool(run.get("passed")), run.get("exit_code"), run.get("duration_s")))
        result = run.get("result") or {}
        if isinstance(result, dict):
            summary["system"].update({str(k)[7:]: str(v) for k, v in result.items() if str(k).startswith("system.")})
    last_step = next(reversed(runs), None) if runs else None
    summary["last_step"] = last_step
    if last_step is not None and isinstance(runs[last_step], dict):
        last = runs[last_step]
        result = last.get("result") if isinstance(last.get("result"), dict) else {}
        summary["error"] = summary["error"] or _first_error(result)
        if not last.get("passed") and not summary["error"]:
            tail = str(last.get("stderr") or "").strip().splitlines()
            summary["error"] = tail[-1] if tail else "evaluator produced no result"
    ranked = next(
        (step for step in ("leaderboard", "benchmark")
         if isinstance(runs.get(step), dict) and isinstance(runs[step].get("result"), dict)
         and "benchmark.0.mean" in runs[step]["result"]),
        None,
    )
    if ranked:
        result = runs[ranked]["result"]
        summary.update(
            ranked_step=ranked,
            mean_ms=_scaled(result, "benchmark.0.mean", 1e6),
            std_ms=_scaled(result, "benchmark.0.std", 1e6),
            best_ms=_scaled(result, "benchmark.0.best", 1e6),
            median_ms=_scaled(result, "benchmark.0.median", 1e6),
            draws=_num(result, "benchmark.0.runs", int),
            accuracy_pct=_scaled(result, "benchmark.0.accuracy", 0.01),
            correct=_num(result, "benchmark.0.correct", int),
            total=_num(result, "benchmark.0.total", int),
            required=_num(result, "benchmark.0.required", int),
        )
        try:
            per_draw = json.loads(str(result.get("benchmark.0.per_draw", "null")))
            summary["per_draw"] = per_draw if isinstance(per_draw, list) else None
        except (TypeError, ValueError):
            summary["per_draw"] = None
        if "benchmark.0.holdout_accuracy" in result:
            summary.update(
                holdout_pct=_scaled(result, "benchmark.0.holdout_accuracy", 0.01),
                holdout_correct=_num(result, "benchmark.0.holdout_correct", int),
                holdout_required=_num(result, "benchmark.0.holdout_required", int),
                holdout_ms=_num(result, "benchmark.0.holdout_ms"),
            )
    elif isinstance(runs.get("test"), dict) and isinstance(runs["test"].get("result"), dict):
        message = str(runs["test"]["result"].get("test.0.message", ""))
        summary["message"] = message
        matched = re.search(r"([0-9.]+) ms per call", message)
        if matched:
            summary["mean_ms"] = _num({"v": matched[1]}, "v")
        matched = re.search(r"(\d+)/(\d+) correct", message)
        if matched and int(matched[2]) > 0:
            summary["correct"], summary["total"] = int(matched[1]), int(matched[2])
            summary["accuracy_pct"] = 100 * summary["correct"] / summary["total"]
    return summary


def finish(record: dict, payload: dict) -> dict:
    """Settle a record from the worker's return value; the charge is uncapped."""
    if not isinstance(payload, dict):
        raise TypeError(f"worker returned {type(payload).__name__}, expected a dict")
    billable = payload.get("billable_s")
    if not isinstance(billable, (int, float)) or isinstance(billable, bool) or billable < 0:
        billable = worst_case_seconds(load_task(record["band"]), record["mode"])
    charged = round(float(billable) * RATE_USD_PER_S, 4)
    record.update(
        status="passed" if payload.get("passed") else "failed",
        finished_at=utcnow(),
        billable_s=round(float(billable), 1),
        charged_usd=charged,
        overrun_usd=round(max(0.0, charged - float(record["reserved_usd"])), 4),
        summary=summarize(payload),
        gpu=payload.get("gpu"),
    )
    return record


BACKSTOP_USD = round((GPU_TIMEOUT_S + STARTUP_PAD_S) * RATE_USD_PER_S, 4)


def fail(record: dict, message: str, charged: float | None = None) -> dict:
    """The container died, timed out or was cancelled: charge the reservation.

    ``charged`` overrides that when more is known: a run that hit Modal's
    container limit is charged the whole container life (``BACKSTOP_USD``).
    """
    charged = record["reserved_usd"] if charged is None else max(float(charged), float(record["reserved_usd"]))
    record.update(
        status="error",
        finished_at=utcnow(),
        charged_usd=round(charged, 4),
        overrun_usd=round(max(0.0, charged - float(record["reserved_usd"])), 4),
        error=str(message)[:2000],
    )
    return record


# ------------------------------------------------------------------ storage


class MemoryStore:
    """The subset of modal.Dict the site uses; for tests and dry runs."""

    def __init__(self):
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key, value):
        self.data[key] = value

    def items(self):
        return list(self.data.items())


class MemoryLog:
    """Durable charge log stand-in: id -> usd."""

    def __init__(self):
        self.charges: dict = {}

    def record(self, sid: str, usd: float, extra: dict | None = None) -> None:
        self.charges.setdefault(sid, float(usd))

    def total(self) -> float:
        return sum(self.charges.values())


class VolumeLog:
    """One immutable JSON file per settled run on a modal.Volume.

    Files are written once, so concurrent settlers (the web container and the
    cron) can never conflict; the floor for the ledger is the sum over files.
    """

    def __init__(self, volume, root: str = LEDGER_MOUNT):
        self.volume = volume
        self.root = Path(root) / "charges"

    def record(self, sid: str, usd: float, extra: dict | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{sid}.json"
        if path.exists():
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"id": sid, "charged_usd": float(usd), "at": utcnow(), **(extra or {})}))
        temporary.replace(path)
        self.volume.commit()

    def total(self) -> float:
        try:
            self.volume.reload()
        except Exception:  # noqa: BLE001 - a stale view only makes the floor lower, never negative
            pass
        total = 0.0
        if self.root.exists():
            for path in self.root.glob("*.json"):
                try:
                    total += float(json.loads(path.read_text()).get("charged_usd") or 0)
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
        return total


class Site:
    """Everything the routes need, with Modal behind a few callables."""

    def __init__(self, store, blobs, charges, spawner, settler, canceller=None,
                 default_budget: float = DEFAULT_BUDGET_USD, clock=time.time):
        self.store = store  # small records and config
        self.blobs = blobs  # source text and full evaluator output, keyed by submission id
        self.charges = charges  # durable charge log (floor for the ledger)
        self.spawner = spawner  # job dict -> call id
        self.settler = settler  # call id -> ("pending"|"done"|"timeout"|"error", payload); raises on transport trouble
        self.canceller = canceller or (lambda call_id: None)
        self.default_budget = default_budget
        self.clock = clock
        self.lock = threading.Lock()

    # config
    def token(self) -> str:
        token = self.store.get("config:token")
        if not token:
            token = secrets.token_urlsafe(18)
            self.store.put("config:token", token)
        return token

    def session_key(self) -> str:
        """Key for the sign-in cookie and the OAuth state. Rotating the token
        deliberately does not rotate this: a link rotation should not sign
        everyone out, and a leaked cookie is bounded by SESSION_MAX_AGE_S."""
        key = self.store.get("config:session_key")
        if not key:
            key = secrets.token_urlsafe(32)
            self.store.put("config:session_key", key)
        return key

    def rotate_token(self) -> str:
        token = secrets.token_urlsafe(18)
        self.store.put("config:token", token)
        return token

    def budget(self) -> float:
        value = self.store.get("config:budget_usd")
        return float(value) if value is not None else self.default_budget

    def set_budget(self, usd: float) -> None:
        self.store.put("config:budget_usd", float(usd))

    # records
    def submissions(self) -> list[dict]:
        items = [value for key, value in self.store.items()
                 if str(key).startswith("sub:") and isinstance(value, dict)]
        return sorted(items, key=lambda r: r["id"], reverse=True)

    def get(self, sid: str) -> dict | None:
        return self.store.get(f"sub:{sid}")

    def save(self, record: dict) -> None:
        self.store.put(f"sub:{record['id']}", record)

    def source(self, sid: str) -> str | None:
        return self.blobs.get(f"src:{sid}")

    def runs(self, sid: str) -> dict | None:
        return self.blobs.get(f"runs:{sid}")

    def refresh(self) -> int:
        """Rewrite every key so modal.Dict's 7-day idle expiry never fires.

        Also migrates records written by the first deployment, which kept the
        kernel text and the evaluator output inside the record itself.
        """
        count = 0
        for key, value in list(self.store.items()):
            if str(key).startswith("sub:") and isinstance(value, dict) and ("source" in value or "runs" in value):
                if "source" in value:
                    self.blobs.put(f"src:{value['id']}", value.pop("source"))
                if "runs" in value:
                    self.blobs.put(f"runs:{value['id']}", value.pop("runs"))
            self.store.put(key, value)
            count += 1
        for key, value in list(self.blobs.items()):
            self.blobs.put(key, value)
            count += 1
        return count

    def _write_off(self, record: dict, message: str, reason: str, charged: float | None = None) -> None:
        if record.get("call_id"):
            self.canceller(record["call_id"])
        self.save(fail(record, message, charged))
        self.charges.record(record["id"], record["charged_usd"], {"reason": reason})

    def _settle_one(self, record: dict) -> None:
        if not record.get("call_id"):
            age = self.clock() - parse_time(record["created_at"])
            if age > LOST_CALL_GRACE_S:
                self._write_off(record, "the run was queued but its call id was never recorded; "
                                        "charged the reservation to be safe", "lost call id")
            return
        try:
            state, payload = self.settler(record["call_id"])
        except Exception as error:  # noqa: BLE001 - transport trouble: retry a few times, then write off
            failures = int(record.get("settle_failures") or 0) + 1
            record["settle_failures"] = failures
            record["last_settle_error"] = repr(error)[:500]
            if failures >= MAX_SETTLE_FAILURES:
                self._write_off(record, f"could not read the result after {failures} attempts: {error!r}",
                                "unreachable")
            else:
                self.save(record)
            return
        if state == "pending":
            if record.get("settle_failures"):
                record["settle_failures"] = 0
                self.save(record)
            return
        if state == "done":
            try:
                finish(record, payload)
                self.blobs.put(f"runs:{record['id']}", (payload or {}).get("runs") or {})
            except Exception:  # noqa: BLE001 - never let one payload poison every page
                fail(record, "the worker's result could not be interpreted:\n" + traceback.format_exc()[-1500:])
            self.save(record)
            self.charges.record(record["id"], record["charged_usd"], {"status": record["status"]})
        elif state == "timeout":
            self._write_off(record, str(payload), "container limit", BACKSTOP_USD)
        else:
            self._write_off(record, str(payload), "worker error")

    def settle(self) -> list[dict]:
        """Poll every in-flight call once; settle the ones that finished."""
        items = self.submissions()
        for record in items:
            if record["status"] not in PENDING:
                continue
            try:
                self._settle_one(record)
            except Exception:  # noqa: BLE001 - keep the page up; the next settle retries
                print(f"settle failed for {record['id']}:\n{traceback.format_exc()}", file=sys.stderr)
        return items

    def ledger_for(self, items: list[dict]) -> dict:
        return ledger(items, self.budget(), self.charges.total())

    def ledger(self) -> dict:
        return self.ledger_for(self.submissions())

    def cancel(self, sid: str, reason: str = "cancelled by the owner") -> dict | None:
        record = self.get(sid)
        if record is None or record["status"] not in PENDING:
            return record
        self._write_off(record, reason, "cancelled")
        return record

    def submit(self, name: str, band: str, mode: str, source: str,
               github: str | None = None) -> tuple[dict | None, str | None]:
        """Validate, reserve budget, persist, spawn. Returns (record, rejection)."""
        if band not in BANDS:
            return None, f"unknown band {band!r}"
        if mode not in MODES:
            return None, f"unknown mode {mode!r}"
        reason = validate_source(source)
        if reason:
            return None, reason
        task = load_task(band)
        reserved = reservation_usd(task, mode)
        with self.lock:
            items = self.settle()
            book = self.ledger_for(items)
            if book["inflight"] >= MAX_INFLIGHT:
                return None, f"{MAX_INFLIGHT} submissions are already queued; try again later"
            if book["remaining_usd"] < reserved:
                return None, (
                    f"budget: this {mode} run could cost up to ${reserved:.2f} but only "
                    f"${book['remaining_usd']:.2f} of the ${book['budget_usd']:.0f} cap is left"
                )
            record = {
                "id": new_id(),
                "created_at": utcnow(),
                "name": (name or "anonymous").strip()[:80],
                "band": band,
                "mode": mode,
                "seed": secrets.randbelow(2**31),
                "status": "queued",
                "reserved_usd": reserved,
                "deadline_s": deadline_seconds(task, mode),
                "charged_usd": None,
                "call_id": None,
                "github": github,
            }
            # Reserve first: the record (and its reservation) exists before any GPU is asked for.
            self.blobs.put(f"src:{record['id']}", source)
            self.save(record)
            job = {"id": record["id"], "band": band, "mode": mode, "source": source,
                   "seed": record["seed"], "deadline_s": record["deadline_s"]}
            try:
                record["call_id"] = self.spawner(job)
            except Exception as error:  # noqa: BLE001 - nothing ran: release the reservation
                record.update(status="error", finished_at=utcnow(), charged_usd=0.0, overrun_usd=0.0,
                              error=f"could not start the run: {error!r}"[:2000])
                self.save(record)
                return None, f"could not start the run on Modal: {error!r}"[:500]
            self.save(record)
        return record, None


# ------------------------------------------------------------------ worker


# ------------------------------------------------------------------ identity
#
# The secret link is the only gate by default, and that is all the challenge
# needs while it is shared by hand. What the link cannot do is say who spent
# the budget: anyone holding it can burn the whole cap anonymously, and a link
# that leaks cannot be attributed, only rotated.
#
# So GitHub login is optional and OFF unless a Modal secret named by
# GITHUB_SECRET_NAME provides GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET. With
# no secret every route behaves exactly as before. With one, a submitter signs
# in before spending anything and the record carries their login.
#
# GITHUB_ALLOWED_LOGINS, if set, is a comma-separated allowlist; empty means
# any GitHub account.
GITHUB_SECRET_NAME = "sutro-mnist-github-oauth"
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
SESSION_COOKIE = "sutro_id"
SESSION_MAX_AGE_S = 7 * 24 * 3600
STATE_MAX_AGE_S = 600
LOGIN_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")

# Attaching a Modal secret that does not exist fails the deploy, so sign-in is
# opted into explicitly at deploy time:
#     modal secret create sutro-mnist-github-oauth \
#         GITHUB_CLIENT_ID=... GITHUB_CLIENT_SECRET=...
#     SUTRO_GITHUB_OAUTH=1 uvx modal deploy web/app.py
# Without it the deploy is byte-for-byte what it was and the link is the only gate.
GITHUB_OAUTH_ENABLED = os.environ.get("SUTRO_GITHUB_OAUTH", "").lower() not in ("", "0", "false", "no")
GITHUB_SECRETS = [modal.Secret.from_name(GITHUB_SECRET_NAME)] if GITHUB_OAUTH_ENABLED else []


def sign(key: str, message: str) -> str:
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def mint(key: str, payload: str, issued: float) -> str:
    """A tamper-evident 'payload.issued.signature' blob for a cookie or a state."""
    body = f"{payload}.{int(issued)}"
    return f"{body}.{sign(key, body)}"


def read_signed(key: str, blob: object, max_age_s: float, now: float) -> str | None:
    """The payload of a blob this key signed and that has not expired, else None."""
    if not isinstance(blob, str) or not blob.isascii():
        return None  # compare_digest raises TypeError on non-ASCII str
    parts = blob.split(".")
    if len(parts) != 3:
        return None
    payload, issued, signature = parts
    if not hmac.compare_digest(signature, sign(key, f"{payload}.{issued}")):
        return None
    try:
        age = now - int(issued)
    except ValueError:
        return None
    if age < -60 or age > max_age_s:  # tolerate a little clock skew
        return None
    return payload


class Identity:
    """GitHub sign-in, or a disabled stand-in when no OAuth secret is present."""

    def __init__(self, client_id: str = "", client_secret: str = "", signing_key: str = "",
                 allowed: str = "", exchange=None):
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.signing_key = signing_key
        self.allowed = {name.strip().lower() for name in (allowed or "").split(",") if name.strip()}
        self._exchange = exchange or self._github_exchange

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret and self.signing_key)

    def permits(self, login: str) -> bool:
        return not self.allowed or login.lower() in self.allowed

    def start(self, redirect_uri: str, now: float) -> tuple[str, str]:
        """The GitHub URL to send the browser to, and the state that guards it."""
        state = mint(self.signing_key, secrets.token_urlsafe(9), now)
        query = urllib.parse.urlencode({
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": "read:user",
            "state": state,
        })
        return f"{GITHUB_AUTHORIZE_URL}?{query}", state

    def finish(self, code: object, state: object, redirect_uri: str, now: float) -> str:
        """The GitHub login behind a callback, or raise ValueError with a reason."""
        if read_signed(self.signing_key, state, STATE_MAX_AGE_S, now) is None:
            raise ValueError("the sign-in link expired or did not come from this site")
        if not isinstance(code, str) or not code.isascii() or not code.strip():
            raise ValueError("GitHub did not return an authorisation code")
        login = self._exchange(code.strip(), redirect_uri)
        if not isinstance(login, str) or not LOGIN_PATTERN.match(login):
            raise ValueError("GitHub returned no usable login")
        if not self.permits(login):
            raise ValueError(f"{login} is not on this site's allowlist")
        return login

    def session(self, login: str, now: float) -> str:
        return mint(self.signing_key, login, now)

    def viewer(self, cookie: object, now: float) -> str | None:
        login = read_signed(self.signing_key, cookie, SESSION_MAX_AGE_S, now)
        return login if login and LOGIN_PATTERN.match(login) and self.permits(login) else None

    # The real round trip. Injectable, so every test above runs without network.
    def _github_exchange(self, code: str, redirect_uri: str) -> str:  # pragma: no cover
        import urllib.request

        body = urllib.parse.urlencode({
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }).encode()
        request = urllib.request.Request(
            GITHUB_TOKEN_URL, data=body,
            headers={"Accept": "application/json", "User-Agent": APP_NAME},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        access = payload.get("access_token")
        if not access:
            raise ValueError(f"GitHub refused the code: {payload.get('error', 'no access_token')}")
        request = urllib.request.Request(
            GITHUB_USER_URL,
            headers={"Accept": "application/vnd.github+json",
                     "Authorization": f"Bearer {access}", "User-Agent": APP_NAME},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return str(json.loads(response.read().decode("utf-8")).get("login") or "")


def run_with_deadline(command: list[str], deadline_s: float, cwd: str | None = None) -> tuple[int | None, str, bool]:
    """Run a command in its own session; kill the whole group when the deadline passes.

    Returns (exit code or None when killed, combined output tail, timed_out).
    """
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=deadline_s)
        return process.returncode, (output or "")[-8000:], False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
        return None, (output or "")[-8000:], True


def evaluate_job(job: dict, harness: Path, work: Path, python: str = sys.executable) -> dict:
    """Evaluate one job under its deadline; always returns a payload dict."""
    work.mkdir(parents=True, exist_ok=True)
    job_path, out_path = work / "job.json", work / "payload.json"
    job_path.write_text(json.dumps(job))
    deadline = float(job.get("deadline_s") or GPU_TIMEOUT_S - STARTUP_PAD_S)
    code, log, timed_out = run_with_deadline(
        [python, str(harness / "web" / "runner.py"), str(job_path), str(out_path), str(work)],
        deadline, cwd=str(work),
    )
    if timed_out:
        return {"runs": {}, "passed": False,
                "error": f"the whole evaluation exceeded its {deadline:.0f} s deadline for {job['mode']} mode "
                         f"and every process it started was killed",
                "log": log}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text())
            if isinstance(payload, dict):
                return payload
        except ValueError:
            pass
    return {"runs": {}, "passed": False, "error": f"the runner exited with code {code} and no result", "log": log}


_BOOT = time.time()


@app.function(
    image=gpu_image,
    gpu=run_modal.GPU,
    cpu=2,
    memory=8192,
    timeout=GPU_TIMEOUT_S,
    max_containers=1,
    scaledown_window=5,
    retries=0,
    restrict_modal_access=True,
    single_use_containers=True,
    block_network=run_modal.BLOCK_NETWORK,
)
def evaluate_submission(job: dict) -> dict:
    """Run one submission through eval.py exactly as run_modal.py does, under a deadline."""
    started = time.time()
    payload = evaluate_job(job, Path(REMOTE_HARNESS), Path("/tmp/sutro-job"))
    try:
        import torch

        payload["gpu"] = torch.cuda.get_device_name(0)
    except Exception as error:  # pragma: no cover
        payload["gpu"] = f"unknown: {error!r}"
    payload["billable_s"] = (time.time() - started) + (started - _BOOT) + STARTUP_PAD_S
    payload["job_id"] = job["id"]
    return payload


TRANSPORT_ERRORS = (
    modal.exception.ConnectionError,
    modal.exception.InternalError,
    modal.exception.InternalFailure,
    modal.exception.ServiceError,
    modal.exception.ClientClosed,
    modal.exception.AuthError,
)


def modal_spawner(job: dict) -> str:
    return evaluate_submission.spawn(job).object_id


def modal_settler(call_id: str):
    """Map FunctionCall.get(timeout=0) onto (state, payload); transport errors raise."""
    call = modal.FunctionCall.from_id(call_id)
    try:
        return "done", call.get(timeout=0)
    except TimeoutError:  # the builtin: not finished yet
        return "pending", None
    except modal.exception.FunctionTimeoutError as error:
        return "timeout", f"the run hit Modal's {GPU_TIMEOUT_S} s container limit: {error}"
    except modal.exception.OutputExpiredError:
        return "error", "the result expired before it was collected"
    except TRANSPORT_ERRORS:
        raise
    except Exception as error:  # noqa: BLE001 - the worker itself raised: charge the reservation
        return "error", f"worker failed: {error!r}"


def modal_canceller(call_id: str) -> None:
    try:
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    except Exception as error:  # noqa: BLE001
        print(f"cancel {call_id} failed: {error!r}", file=sys.stderr)


def modal_site() -> Site:
    blobs = modal.Dict.from_name(BLOBS_NAME, create_if_missing=True)
    return Site(records, blobs, VolumeLog(ledger_volume), modal_spawner, modal_settler, modal_canceller)


def modal_identity(site: Site) -> Identity:
    """GitHub sign-in if the OAuth secret is attached, a disabled stand-in if not."""
    client_id = os.environ.get("GITHUB_CLIENT_ID", "")
    client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
    if not (client_id and client_secret):
        return Identity()
    return Identity(client_id, client_secret, site.session_key(),
                    os.environ.get("GITHUB_ALLOWED_LOGINS", ""))


# ------------------------------------------------------------------ html

STYLE = """
body{font:15px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1c1c1c}
h1{font-size:1.5rem} h2{font-size:1.15rem;margin-top:2rem}
table{border-collapse:collapse;width:100%} th,td{text-align:left;padding:.3rem .5rem;border-bottom:1px solid #ddd;vertical-align:top}
th{font-weight:600;background:#f4f4f4} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
pre{background:#f6f8fa;padding:.75rem;overflow:auto;font-size:13px;border-radius:4px}
textarea{width:100%;font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;min-height:22rem}
label{display:block;margin:.6rem 0 .2rem;font-weight:600} input[type=text],select{font-size:15px;padding:.3rem}
.budget{background:#eef5ff;padding:.75rem 1rem;border-radius:6px;margin:1rem 0}
.bar{height:10px;background:#ddd;border-radius:5px;overflow:hidden;margin-top:.4rem}
.bar span{display:block;height:100%;background:#3b7ddd}
.ok{color:#137333;font-weight:600} .bad{color:#c5221f;font-weight:600} .wait{color:#8a6d00;font-weight:600}
.err{background:#fdecea;border:1px solid #f5c6c2;padding:.75rem 1rem;border-radius:6px}
.warn{background:#fff8e1;border:1px solid #f0d58c;padding:.75rem 1rem;border-radius:6px;margin:1rem 0}
button{font-size:15px;padding:.45rem 1rem;margin-top:1rem} small,.muted{color:#666}
"""


def page(title: str, body: str, refresh: int | None = None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>{meta}"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def money(value) -> str:
    try:
        return "—" if value is None else f"${float(value):.2f}"
    except (TypeError, ValueError):
        return esc(value)


def fmt(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return esc(value)


def count(value) -> str:
    try:
        return "—" if value is None else f"{int(value):,}"
    except (TypeError, ValueError):
        return esc(value)


def status_cell(record: dict, position: int | None = None) -> str:
    status = record["status"]
    if status == "queued":
        where = "running" if position == 0 else f"queued (#{position} in line)" if position else "queued"
        return f"<span class='wait'>{where}</span>"
    if status == "passed":
        return "<span class='ok'>pass</span>"
    if status == "failed":
        return "<span class='bad'>fail</span>"
    return "<span class='bad'>error</span>"


def budget_block(book: dict) -> str:
    used = 0 if book["budget_usd"] <= 0 else min(100, 100 * book["committed_usd"] / book["budget_usd"])
    return (
        "<div class='budget'><b>Shared A100 budget</b>: "
        f"{money(book['charged_usd'])} spent + {money(book['reserved_usd'])} reserved by "
        f"{book['inflight']} in-flight run(s) = {money(book['committed_usd'])} of {money(book['budget_usd'])}; "
        f"<b>{money(book['remaining_usd'])} left</b>."
        f"<div class='bar'><span style='width:{used:.1f}%'></span></div>"
        "<small>Each run reserves its worst case up front and is charged its measured container time "
        f"at ${RATE_USD_PER_S * 3600:.2f}/h. Runs are rejected once the cap would be exceeded.</small></div>"
    )


# Modal's gpu="A100" is fulfilled with either a 40 GB or an 80 GB SXM4 board.
# The same computation measured 4.311 ms on the 80 GB card against 4.504-4.547 ms
# on five separate 40 GB containers: 4.5% apart, about 20x the within-run spread
# and 5x the cross-container spread. Two entries that landed on different boards
# therefore cannot be ranked against each other, so the board is shown on every
# row and the page says so out loud as soon as more than one appears.
BOARD_PATTERN = re.compile(r"(\d+)\s*GB", re.IGNORECASE)


def board_label(gpu: object) -> str | None:
    """Short name for the board a run landed on, e.g. '40GB'."""
    if not isinstance(gpu, str) or not gpu.strip():
        return None
    matched = BOARD_PATTERN.search(gpu)
    return f"{matched[1]}GB" if matched else gpu.strip()[:24]


def boards_in(items: list[dict]) -> list[str]:
    """Every distinct board among runs that produced a ranked time, sorted."""
    seen = set()
    for record in items:
        summary = record.get("summary") or {}
        if summary.get("mean_ms") is None:
            continue
        label = board_label(record.get("gpu"))
        if label:
            seen.add(label)
    return sorted(seen)


def board_warning(items: list[dict]) -> str:
    boards = boards_in(items)
    if len(boards) < 2:
        return ""
    return (
        "<div class='warn'><b>Mixed hardware.</b> These runs landed on more than one "
        f"A100 board ({', '.join(esc(b) for b in boards)}). The same computation is about "
        "4.5% faster on the 80 GB card than on the 40 GB card, which is roughly 20x the "
        "run-to-run spread, so times are comparable only within one board. Compare the "
        "<b>board</b> column before reading anything into a ranking.</div>"
    )


def queue_positions(items: list[dict]) -> dict:
    queue = [r["id"] for r in sorted(items, key=lambda r: r["id"]) if r["status"] in PENDING]
    return {sid: index for index, sid in enumerate(queue)}


def github_cell(record: dict) -> str:
    login = record.get("github")
    return f"<br><small>@{esc(login)}</small>" if isinstance(login, str) and login else ""


def submissions_table(items: list[dict], base: str) -> str:
    if not items:
        return "<p class='muted'>No submissions yet.</p>"
    positions = queue_positions(items)
    rows = []
    for record in items:
        summary = record.get("summary") or {}
        final = record["status"] in FINAL
        cost = money(record.get("charged_usd")) if final else money(record.get("reserved_usd"))
        rows.append(
            f"<tr><td><a href='{base}/s/{esc(record['id'])}'>{esc(record['id'])}</a></td>"
            f"<td>{esc(str(record.get('created_at', ''))[:16].replace('T', ' '))}</td>"
            f"<td>{esc(record.get('name'))}{github_cell(record)}</td>"
            f"<td>{esc(str(record.get('band', '')).replace('mnist-medium-', ''))}</td><td>{esc(record.get('mode'))}</td>"
            f"<td>{status_cell(record, positions.get(record['id']))}</td>"
            f"<td class='num'>{fmt(summary.get('mean_ms'))}</td>"
            f"<td>{esc(board_label(record.get('gpu')) or '')}</td>"
            f"<td class='num'>{fmt(summary.get('accuracy_pct'), 2, '%')}</td>"
            f"<td class='num'>{cost}{'' if final else '<br><small>reserved</small>'}</td></tr>"
        )
    return (
        "<table><tr><th>id</th><th>submitted (UTC)</th><th>name</th><th>band</th><th>mode</th><th>status</th>"
        "<th class='num'>mean ms</th><th>board</th><th class='num'>accuracy</th><th class='num'>cost</th></tr>"
        + "".join(rows) + "</table>"
    )


def sign_in_block(base: str, signed_in: str | None, required: bool) -> str:
    """Who you are, or how to become someone. Empty when sign-in is off."""
    if not required:
        return ""
    if signed_in:
        return (f"<div class='budget'>Signed in as <b>{esc(signed_in)}</b>. "
                f"<a href='{base}/logout'>Sign out</a>. Runs you start are recorded "
                "against this account.</div>")
    return (f"<div class='warn'><b>Sign in to submit.</b> This site spends a shared A100 "
            f"budget, so a run has to be attributable. <a href='{base}/login'>Sign in with "
            "GitHub</a> — the only thing read is your login name.</div>")


def index_html(site: Site, base: str, template: str, error: str | None = None, prefill: dict | None = None,
               items: list[dict] | None = None, signed_in: str | None = None,
               sign_in_required: bool = False) -> str:
    prefill = prefill or {}
    if items is None:
        items = site.settle()
    book = site.ledger_for(items)
    reference = load_task("mnist-medium-5pct")
    options = "".join(
        f"<option value='{band}' {'selected' if band == prefill.get('band', 'mnist-medium-5pct') else ''}>"
        f"{band} — at most {band.split('-')[-1][:-3]}% mean error</option>"
        for band in BANDS
    )
    modes = "".join(
        f"<option value='{mode}' {'selected' if mode == prefill.get('mode', 'leaderboard') else ''}>"
        f"{esc(text)} — up to {money(reservation_usd(reference, mode))}</option>"
        for mode, text in MODES.items()
    )
    error_block = f"<div class='err'><b>Not submitted:</b> {esc(error)}</div>" if error else ""
    body = f"""
<h1>Sutro MNIST-medium: submit a kernel</h1>
<p>Train on 10,000 9×9 MNIST digits and label 10,000 more on one A100, as fast as you can while
staying inside the band's error budget. Your <code>custom_kernel((train_x, train_y, test_x))</code>
is timed with CUDA events over fresh secret draws; the accuracy on those same draws gates the result.
Rules and the interface are in the template below and in the
<a href="https://github.com/cybertronai/sutro-problems/tree/main/mnist">MNIST problem page</a>.</p>
{budget_block(book)}
{sign_in_block(base, signed_in, sign_in_required)}
{board_warning(items)}
{error_block}
<form method="post" action="{base}/submit" enctype="multipart/form-data">
<label for="name">Your name or handle</label>
<input type="text" id="name" name="name" maxlength="80" value="{esc(prefill.get('name', ''))}" placeholder="shown on the results table">
<label for="band">Problem (accuracy band)</label>
<select id="band" name="band">{options}</select>
<label for="mode">Mode</label>
<select id="mode" name="mode">{modes}</select>
<label for="source">Kernel (<code>submission.py</code>)</label>
<textarea id="source" name="source" spellcheck="false">{esc(prefill.get('source', template))}</textarea>
<label for="file">…or upload a file instead</label>
<input type="file" id="file" name="file" accept=".py,text/x-python,text/plain">
<br><button type="submit">Run on an A100</button>
<p><small>Runs are queued one at a time on a fresh container. A leaderboard run takes a few minutes for a
fast kernel; slow kernels are cut off by the evaluator's timeouts (5 min test, 10 min benchmark,
20 min leaderboard) and by a hard deadline on the whole run.</small></p>
</form>
<h2>Submissions</h2>
{submissions_table(items, base)}
"""
    return page("Sutro MNIST-medium submissions", body)


def detail_html(record: dict, base: str, position: int | None, source: str | None, runs: dict | None) -> str:
    summary = record.get("summary") or {}
    pending = record["status"] in PENDING
    head = (
        f"<p><a href='{base}'>← all submissions</a></p><h1>Submission {esc(record['id'])}</h1>"
        f"<p><b>{esc(record.get('name'))}</b> · {esc(record.get('band'))} · mode <b>{esc(record.get('mode'))}</b> · "
        f"submitted {esc(record.get('created_at'))} · status {status_cell(record, position)}</p>"
    )
    parts = [head]
    if pending:
        parts.append("<p class='wait'>This page refreshes every 15 seconds until the run finishes.</p>")
        parts.append(f"<p>Reserved {money(record.get('reserved_usd'))} of budget for this run; "
                     f"hard deadline {fmt(record.get('deadline_s'), 0, ' s')} on the whole evaluation.</p>")
    elif record["status"] == "error":
        parts.append(f"<div class='err'><b>Infrastructure error</b> (charged the {money(record.get('charged_usd'))} "
                     f"reservation):<pre>{esc(record.get('error'))}</pre></div>")
    else:
        verdict = "<span class='ok'>PASS</span>" if summary.get("verdict") == "pass" else "<span class='bad'>FAIL</span>"
        rows = [f"<tr><th>verdict</th><td>{verdict}</td></tr>"]
        if summary.get("mean_ms") is not None:
            if summary.get("ranked_step") == "leaderboard":
                label = "mean time per call (the ranked number)"
            elif summary.get("ranked_step") == "benchmark":
                label = f"mean time per call ({count(summary.get('draws'))} draws; benchmark mode, not the ranked number)"
            else:
                label = "time per call (test mode, 1 draw)"
            rows.append(f"<tr><th>{esc(label)}</th><td><b>{fmt(summary['mean_ms'])} ms</b></td></tr>")
        if summary.get("std_ms") is not None:
            rows.append(f"<tr><th>std / best / median</th><td>{fmt(summary.get('std_ms'))} / {fmt(summary.get('best_ms'))} / "
                        f"{fmt(summary.get('median_ms'))} ms over {count(summary.get('draws'))} draws</td></tr>")
        if summary.get("accuracy_pct") is not None:
            need = f" (needs {count(summary['required'])})" if summary.get("required") else ""
            rows.append(f"<tr><th>accuracy</th><td><b>{fmt(summary['accuracy_pct'], 2)}%</b> = "
                        f"{count(summary.get('correct'))} / {count(summary.get('total'))} correct{need}</td></tr>")
        if summary.get("per_draw"):
            rows.append(f"<tr><th>correct per draw</th><td>{esc(summary['per_draw'])}</td></tr>")
        if summary.get("holdout_pct") is not None:
            rows.append(f"<tr><th>Fashion-MNIST hold-out</th><td>{fmt(summary['holdout_pct'], 2)}% "
                        f"({count(summary.get('holdout_correct'))} correct, needs {count(summary.get('holdout_required'))}) "
                        f"in {fmt(summary.get('holdout_ms'))} ms</td></tr>")
        if summary.get("message"):
            rows.append(f"<tr><th>evaluator</th><td>{esc(summary['message'])}</td></tr>")
        if summary.get("error"):
            rows.append(f"<tr><th>failure</th><td class='bad'>{esc(summary['error'])}</td></tr>")
        system = summary.get("system") or {}
        rows.append(f"<tr><th>hardware</th><td>{esc(record.get('gpu'))} · torch {esc(system.get('torch'))} · "
                    f"CUDA {esc(system.get('cuda'))} · {esc(system.get('harness'))}</td></tr>")
        overrun = record.get("overrun_usd") or 0
        rows.append(f"<tr><th>cost</th><td>{money(record.get('charged_usd'))} for {fmt(record.get('billable_s'), 0)} s "
                    f"of container time (reserved {money(record.get('reserved_usd'))}"
                    f"{'; <b>overran the reservation by ' + money(overrun) + '</b>' if overrun > 0 else ''})</td></tr>")
        parts.append("<h2>Result</h2><table>" + "".join(rows) + "</table>")
        steps = summary.get("steps") or []
        if steps:
            parts.append("<h2>Steps</h2><table><tr><th>step</th><th>result</th><th>exit</th><th class='num'>seconds</th></tr>"
                         + "".join(f"<tr><td>{esc(s)}</td><td>{'pass' if ok else 'fail'}</td><td>{esc(code)}</td>"
                                   f"<td class='num'>{esc(sec)}</td></tr>" for s, ok, code, sec in steps) + "</table>")
        for step, run in (runs or {}).items():
            if not isinstance(run, dict):
                continue
            result = run.get("result") if isinstance(run.get("result"), dict) else {}
            lines = "\n".join(f"{k}: {v}" for k, v in result.items())
            parts.append(f"<h2>{esc(step)} output</h2><pre>{esc(lines) or '(no result lines)'}</pre>")
            if str(run.get("stderr") or "").strip():
                parts.append(f"<details><summary>{esc(step)} stderr</summary><pre>{esc(str(run['stderr'])[-6000:])}</pre></details>")
    parts.append(f"<h2>Kernel</h2><p><a href='{base}/s/{esc(record['id'])}/source'>raw</a></p>"
                 f"<pre>{esc(source) if source is not None else '(source no longer available)'}</pre>")
    return page(f"Submission {record['id']}", "".join(parts), refresh=15 if pending else None)


# ------------------------------------------------------------------ fastapi


def build_api(site: Site, template: str, identity: Identity | None = None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
    from starlette.concurrency import run_in_threadpool

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    identity = identity or Identity()

    def viewer(request) -> str | None:
        """The signed-in GitHub login, or None. Always None when sign-in is off."""
        if not identity.enabled:
            return None
        return identity.viewer(request.cookies.get(SESSION_COOKIE), time.time())

    def callback_uri(request) -> str:
        # Token-free on purpose: the secret link must not travel to GitHub or
        # sit in its logs. The browser comes back here and is sent on to the
        # one link this site serves.
        return str(request.base_url).rstrip("/") + "/auth/callback"

    def check(token: str) -> str:
        if not token.isascii() or not secrets.compare_digest(token, site.token()):
            raise HTTPException(status_code=404, detail="not found")
        return f"/{token}"

    def public(record: dict) -> dict:
        return {k: v for k, v in record.items() if k not in ("seed", "call_id", "last_settle_error")}

    @api.get("/", response_class=PlainTextResponse)
    def root():
        return PlainTextResponse("nothing here", status_code=404)

    @api.get("/{token}", response_class=HTMLResponse)
    def index(token: str, request: Request):
        base = check(token)
        return HTMLResponse(index_html(site, base, template, signed_in=viewer(request),
                                       sign_in_required=identity.enabled))

    @api.get("/{token}/login")
    def login(token: str, request: Request):
        base = check(token)
        if not identity.enabled:
            return RedirectResponse(base, status_code=303)
        url, _ = identity.start(callback_uri(request), time.time())
        return RedirectResponse(url, status_code=303)

    @api.get("/{token}/logout")
    def logout(token: str):
        base = check(token)
        response = RedirectResponse(base, status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @api.get("/auth/callback", response_class=HTMLResponse)
    async def auth_callback(request: Request):
        """Token-free by design; a cookie here is worthless without the link."""
        if not identity.enabled:
            raise HTTPException(status_code=404, detail="not found")
        try:
            login_name = await run_in_threadpool(
                identity.finish, request.query_params.get("code"),
                request.query_params.get("state"), callback_uri(request), time.time())
        except ValueError as error:
            return HTMLResponse(page("Sign-in failed",
                                     f"<h1>Sign-in failed</h1><div class='err'>{esc(str(error))}</div>"),
                                status_code=400)
        base = "/" + await run_in_threadpool(site.token)
        response = RedirectResponse(base, status_code=303)
        response.set_cookie(
            SESSION_COOKIE, identity.session(login_name, time.time()),
            max_age=SESSION_MAX_AGE_S, httponly=True, samesite="lax",
            secure=request.url.scheme == "https", path="/")
        return response

    @api.post("/{token}/submit")
    async def submit(token: str, request: Request):
        base = await run_in_threadpool(check, token)  # check() reads the Dict: keep it off the event loop
        who = await run_in_threadpool(viewer, request)
        if identity.enabled and not who:
            return HTMLResponse(await run_in_threadpool(
                index_html, site, base, template,
                "sign in with GitHub before spending the shared budget", None, None,
                None, True), status_code=403)
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared > MAX_REQUEST_BYTES:
            return HTMLResponse(await run_in_threadpool(
                index_html, site, base, template, f"the request is larger than {MAX_REQUEST_BYTES // 1024} KB"),
                status_code=413)
        form = await request.form()
        name = str(form.get("name") or "")
        band = str(form.get("band") or "")
        mode = str(form.get("mode") or "")
        typed = form.get("source")
        typed = typed if isinstance(typed, str) else ""
        source = typed
        upload = form.get("file")
        if upload is not None and getattr(upload, "filename", ""):
            raw = await upload.read(MAX_SOURCE_BYTES + 1)
            prefill = {"name": name, "band": band, "mode": mode, "source": typed}
            if len(raw) > MAX_SOURCE_BYTES:
                return HTMLResponse(await run_in_threadpool(
                    index_html, site, base, template,
                    f"the uploaded file is larger than {MAX_SOURCE_BYTES // 1024} KB", prefill), status_code=400)
            try:
                source = raw.decode("utf-8")
            except UnicodeDecodeError:
                return HTMLResponse(await run_in_threadpool(
                    index_html, site, base, template, "the uploaded file is not UTF-8 text", prefill),
                    status_code=400)
        source = source.replace("\r\n", "\n")
        record, rejection = await run_in_threadpool(site.submit, name, band, mode, source, who)
        if rejection:
            return HTMLResponse(await run_in_threadpool(
                index_html, site, base, template, rejection,
                {"name": name, "band": band, "mode": mode, "source": source}), status_code=400)
        return RedirectResponse(f"{base}/s/{record['id']}", status_code=303)

    @api.get("/{token}/s/{sid}", response_class=HTMLResponse)
    def detail(token: str, sid: str):
        base = check(token)
        items = site.settle()
        record = site.get(sid)
        if record is None:
            raise HTTPException(status_code=404, detail="no such submission")
        position = queue_positions(items).get(sid)
        runs = site.runs(sid) if record["status"] in FINAL else None
        return HTMLResponse(detail_html(record, base, position, site.source(sid), runs))

    @api.get("/{token}/s/{sid}/source", response_class=PlainTextResponse)
    def source(token: str, sid: str):
        check(token)
        text = site.source(sid)
        if text is None:
            raise HTTPException(status_code=404, detail="no such submission")
        return PlainTextResponse(text)

    @api.get("/{token}/api/submissions")
    def api_list(token: str):
        check(token)
        items = site.settle()
        return JSONResponse({"ledger": site.ledger_for(items), "submissions": [public(r) for r in items]})

    @api.get("/{token}/api/s/{sid}")
    def api_detail(token: str, sid: str):
        check(token)
        site.settle()
        record = site.get(sid)
        if record is None:
            raise HTTPException(status_code=404, detail="no such submission")
        return JSONResponse(public(record))

    return api


@app.function(image=web_image, max_containers=1, scaledown_window=300, min_containers=0,
              volumes={LEDGER_MOUNT: ledger_volume}, secrets=GITHUB_SECRETS,
              env={"SUTRO_BUDGET_USD": str(DEFAULT_BUDGET_USD)})
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    template = (Path(REMOTE_HARNESS) / "submission.py").read_text()
    site = modal_site()
    return build_api(site, template, modal_identity(site))


@app.function(image=web_image, schedule=modal.Period(minutes=10),
              volumes={LEDGER_MOUNT: ledger_volume},
              env={"SUTRO_BUDGET_USD": str(DEFAULT_BUDGET_USD)})
def settle_cron():
    """Settle finished runs and refresh every key, even when nobody is looking."""
    site = modal_site()
    site.settle()
    refreshed = site.refresh()
    print(json.dumps({"refreshed_keys": refreshed, **site.ledger()}))


# ------------------------------------------------------------------ owner commands


def _deployed_url() -> str:
    try:
        url = modal.Function.from_name(APP_NAME, "web").get_web_url()
    except Exception as error:  # noqa: BLE001
        here = Path(__file__).resolve()
        try:
            shown = here.relative_to(Path.cwd())
        except ValueError:
            shown = here
        raise SystemExit(f"deploy first: uvx modal deploy {shown} ({error})")
    return url or ""


class RemoteVolumeLog(VolumeLog):
    """Read the charge log through the SDK (no mount) for local owner commands."""

    def total(self) -> float:
        total = 0.0
        try:
            for entry in self.volume.listdir("charges"):
                if not entry.path.endswith(".json"):
                    continue
                data = b"".join(self.volume.read_file(entry.path))
                total += float(json.loads(data).get("charged_usd") or 0)
        except Exception as error:  # noqa: BLE001
            print(f"(charge log unreadable: {error!r})", file=sys.stderr)
        return total

    def record(self, sid, usd, extra=None):
        print(f"(charge for {sid} is logged by the site on its next settle)", file=sys.stderr)


def _owner_site() -> Site:
    blobs = modal.Dict.from_name(BLOBS_NAME, create_if_missing=True)
    return Site(records, blobs, RemoteVolumeLog(ledger_volume), modal_spawner, modal_settler, modal_canceller)


@app.local_entrypoint()
def link():
    """Print the secret submission link."""
    print(f"{_deployed_url().rstrip('/')}/{_owner_site().token()}")


@app.local_entrypoint()
def rotate_token():
    print(f"{_deployed_url().rstrip('/')}/{_owner_site().rotate_token()}")


@app.local_entrypoint()
def set_budget(usd: float):
    site = _owner_site()
    site.set_budget(usd)
    print(json.dumps(site.ledger(), indent=2))


@app.local_entrypoint()
def cancel(sid: str):
    site = _owner_site()
    record = site.cancel(sid)
    print(json.dumps(record and {k: record.get(k) for k in ("id", "status", "charged_usd", "error")}, indent=2))


@app.local_entrypoint()
def status():
    site = _owner_site()
    items = site.submissions()  # read-only: settlement is the site's job
    print(json.dumps(site.ledger_for(items), indent=2))
    for record in items:
        summary = record.get("summary") or {}
        mean = f"{summary['mean_ms']:.3f} ms" if summary.get("mean_ms") is not None else "—"
        acc = f"{summary['accuracy_pct']:.2f}%" if summary.get("accuracy_pct") is not None else "—"
        cost = record.get("charged_usd") if record["status"] in FINAL else f"{record['reserved_usd']} reserved"
        print(f"{record['id']}  {record['status']:7} {record['band']:18} {record['mode']:11} "
              f"{mean:>12} {acc:>8}  ${cost}  {record['name']}")


@app.local_entrypoint()
def export(output: str = "submissions-export.json"):
    site = _owner_site()
    items = site.submissions()
    for record in items:
        record["source"] = site.source(record["id"])
        record["runs"] = site.runs(record["id"])
    Path(output).write_text(json.dumps(items, indent=2) + "\n")
    print(f"wrote {output}")
