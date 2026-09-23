"""KernelBot evaluator for the MNIST-medium *time* leaderboards.

The ranked number is the mean GPU time of one complete training-and-prediction
call, measured with CUDA events over several fresh, secret draws. Every timed
call is also an accuracy sample: the same draws feed the ranking and the
accuracy gate, so a submission cannot be fast on the calls that are timed and
accurate on the calls that are scored.

Modes (KernelBot runs each as a separate process):
  test         one draw; output format plus a loose accuracy check
  benchmark    the ranked protocol with the case's ``draws``, no hold-out
  leaderboard  the ranked protocol plus the Fashion-MNIST hold-out draws
  profile      one call under torch.profiler

Trust boundary
  This process holds the image pool, the test labels, the draw seeds and each
  draw's secret label permutation. It never imports the submission. A spawned
  child process imports the submission on command, receives only training
  images, training labels and test images, and returns predicted labels.
  Scoring, and every clock that decides a rank, happens here.

Case fields (integers, set in task.yml): size, train, test, error_bp, draws,
bench_draws, seed, holdout, holdout_draws, holdout_min_bp, max_call_ms,
warmup_max_call_ms, draw_slack_bp, dispersion_x10, max_source_bytes,
max_literal_bytes.

Exit codes follow KernelBot: 0 success, 112 validation failure, 111 no result
pipe, 113 unreadable case file.
"""

from __future__ import annotations

import ast
import base64
import json
import multiprocessing
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import mnist_data
from utils import (
    HARNESS_VERSION,
    L2_FLUSH_BYTES,
    combine,
    install_network_guard,
    required_correct,
    set_seed,
    stats,
    system_info,
    timing_plausible,
)

EXIT_SUCCESS = 0
EXIT_NO_FD = 111
EXIT_VALIDATE_FAIL = 112
EXIT_BAD_CASES = 113

TEST_SLACK_BP = 1000  # test mode only: a looser gate, so smoke runs are informative

DEFAULTS = {
    "size": 9,
    "train": 10000,
    "test": 10000,
    "error_bp": 500,
    "draws": 11,
    "bench_draws": 3,
    "seed": 0,
    "holdout": 0,
    "holdout_draws": 2,
    "holdout_min_bp": 3000,
    "max_call_ms": 60000,
    "warmup_max_call_ms": 120000,
    "draw_slack_bp": 150,
    "dispersion_x10": 20,
    "max_source_bytes": 20480,
    "max_literal_bytes": 4096,
}

# Seed offsets, so warm-up, timed and hold-out draws never coincide.
WARMUP_OFFSET = 7
TIMED_STRIDE = 13
HOLDOUT_OFFSET = 9001
HOLDOUT_STRIDE = 101
LABEL_SALT = 0x5EED
UNIVERSE_SALT = 0x5711
DRAW_SALT = 0xD4A7

# Calibration of the inter-process overhead: round trips measured before the
# submission is imported, so nothing it does can inflate the number.
CALIBRATION_ROUNDS = 5
# A staged draw that costs more than this multiple of the calibrated baseline
# means work has been hijacked into the untimed staging step. The parent-side
# factor is the one that decides a run; the child-side one only gives a clearer
# message when the child's own clock is honest.
STAGE_FACTOR = 3.0
STAGE_SLACK_MS = 5.0
CHILD_STAGE_FACTOR = 10.0
CHILD_STAGE_SLACK_MS = 2.0

SECRET_FD_ENV = "MNIST_EVAL_SECRET_FD"

# Removed from the submission process's sys.modules before it can import them.
HIDDEN_MODULES = ("eval", "__mp_main__", "mnist_data", "utils")


def debug(*parts):
    print(*parts, file=sys.stderr, flush=True)


class Failure(Exception):
    """A submission-visible failure; the message is logged and the run fails."""


class PopcornOutput:
    """KernelBot's result channel: plain ``key: value`` lines on POPCORN_FD."""

    def __init__(self, fd: int):
        self.file = os.fdopen(fd, "w")
        os.set_inheritable(fd, False)

    def log(self, key, value):
        print(f"{key}: {value}", file=self.file, flush=True)

    def close(self):
        self.file.close()


# ------------------------------------------------------------------ cases

CASE_PATTERN = re.compile(r"\s*([a-zA-Z_]\w*):\s*([+-]?[0-9]+)\s*")


def read_cases(path, secret):
    """Parse KernelBot's ``k: v; k: v`` lines, combining each seed with the secret."""
    cases = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        case = dict(DEFAULTS)
        for part in line.split(";"):
            matched = CASE_PATTERN.fullmatch(part)
            if not matched:
                raise ValueError(f"invalid case {line!r}: {part!r}")
            key, value = matched[1], int(matched[2])
            if key not in DEFAULTS:
                raise ValueError(f"unknown case field {key!r} in {line!r}")
            case[key] = value
        if secret is not None:
            case["seed"] = combine(case["seed"], secret)
        case["spec"] = line.strip()
        cases.append(case)
    if not cases:
        raise ValueError("no cases")
    return cases


# ------------------------------------------------------------------ submission source

def check_submission_source(case, path="submission.py"):
    """Reject a submission that carries the dataset with it.

    Rule 2 (no memorized constants) used to be an honour rule. 60,000 MNIST
    labels do not compress below about 25 KB, so a source-size cap makes exact
    memorization of the public pool information-theoretically impossible, and
    the literal cap stops the same table arriving as one base64 blob. Every
    legitimate entry ported into ``submissions/`` is under 6 KB.
    """
    source = Path(path).read_bytes()
    limit = case["max_source_bytes"]
    if len(source) > limit:
        raise Failure(
            f"submission.py is {len(source)} bytes, over the {limit}-byte limit; "
            "entries may not carry embedded data (rule 2)"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise Failure(f"submission.py does not parse: {error}") from None
    literal_limit = case["max_literal_bytes"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            if len(node.value) > literal_limit:
                raise Failure(
                    f"submission.py contains a {len(node.value)}-byte literal, over the "
                    f"{literal_limit}-byte limit; entries may not carry embedded data (rule 2)"
                )


def prepare_submission_dir():
    """A private directory holding only what the submission is allowed to see.

    The harness files (eval.py, utils.py, mnist_data.py) and the cases file stay
    in the evaluator's own directory: a submission that could import them would
    read the draw constants, the public case seed and the pool loader.
    """
    directory = Path(tempfile.mkdtemp(prefix="sutro-submission-"))
    os.chmod(directory, 0o700)
    shutil.copy("submission.py", directory / "submission.py")
    for optional in ("task.py",):
        if Path(optional).exists():
            shutil.copy(optional, directory / optional)
    return directory


# ------------------------------------------------------------------ data

def load_pools(cases, cache=None, consume=False):
    """Read the pools into RAM, then remove the raw files before any child starts.

    ``cache`` (from ``MNIST_POOL_CACHE``) keeps verified downloads for offline
    dry runs on a trusted machine. A hosted run either downloads into a
    temporary directory that is deleted here, or -- when the container has no
    egress -- points at the copy baked into the image and sets
    ``MNIST_POOL_CONSUME=1`` (``consume``), which deletes it after loading. The
    public labels must not be readable on disk while the submission is alive.
    """
    sizes = sorted({case["size"] for case in cases})
    raw_dir = Path(cache) if cache else Path(tempfile.mkdtemp(prefix="mnist-raw-"))
    try:
        pools = {}
        for size in sizes:
            pools[("mnist", size)] = mnist_data.load_pool(raw_dir, "mnist", size)
            # Fashion is loaded in every mode: it is both the hold-out and the
            # warm-up pool, so nothing fitted before the timed loop is about the
            # dataset the timed calls use.
            pools[("fashion", size)] = mnist_data.load_pool(raw_dir, "fashion", size)
        return pools
    finally:
        if not cache or consume:
            shutil.rmtree(raw_dir, ignore_errors=True)


def split_universes(count, seed, salt):
    """Split a pool in half, once per evaluation, from the secret seed.

    Training halves are drawn from one half and test halves from the other, so
    no test image is ever shown with a label earlier in the same run. Without
    this, every draw re-splits the same 60,000 rows and a submission can build a
    hash table of the images it has already been taught.
    """
    order = np.random.default_rng([int(seed), salt]).permutation(count)
    middle = count // 2
    return order[:middle], order[middle:]


def make_draw(pool, seed, n_train, n_test, universes):
    """One draw: train and test rows from disjoint universes, plus a secret relabel.

    The label permutation is applied to both halves, so labels only mean
    something relative to the training set they arrive with, and it changes on
    every draw.
    """
    images, labels = pool
    train_universe, test_universe = universes
    if n_train > len(train_universe) or n_test > len(test_universe):
        raise ValueError("train or test exceeds its half of the pool")
    rng = np.random.default_rng([int(seed), DRAW_SALT])
    train_rows = rng.choice(train_universe, n_train, replace=False)
    test_rows = rng.choice(test_universe, n_test, replace=False)
    relabel = np.random.default_rng([int(seed), LABEL_SALT]).permutation(10)
    visible = (
        images[train_rows],
        relabel[labels[train_rows]].astype(np.int64),
        images[test_rows],
    )
    return visible, relabel[labels[test_rows]].astype(np.int64)


# ------------------------------------------------------------------ child process

def child_main(connection, device, work_dir):
    """Runs in a spawned process: imports the submission and times its calls.

    Never sees test labels, draw seeds or the label permutation. Everything the
    harness needs inside this process -- the clock, the CUDA event class, the
    synchronize barrier, the tensor methods used to stage inputs and read
    outputs -- is captured into locals here, *before* the submission is
    imported, and only those captured references are used afterwards. Rebinding
    ``time.perf_counter``, ``torch.cuda.Event``, ``torch.cuda.synchronize`` or
    ``torch.Tensor.copy_`` therefore changes nothing the harness measures.
    """
    harness_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        os.setsid()  # so the parent can kill anything the submission spawns
    except (AttributeError, OSError):  # pragma: no cover - not POSIX
        pass
    import time as _time

    perf = _time.perf_counter
    try:
        work_dir = str(work_dir)
        os.chdir(work_dir)
        # The submission may import only what is in its own directory.
        sys.path[:] = [
            entry
            for entry in sys.path
            if entry not in ("", ".", harness_dir, work_dir) and os.path.abspath(entry) != harness_dir
        ]
        sys.path.insert(0, work_dir)
        # Spawn bootstraps this child by importing eval.py, which leaves the
        # harness modules in sys.modules where a submission could pick them up
        # and read the draw constants, ``combine`` and the pool loader. Hide
        # them, keeping a reference so nothing they hold is collected.
        hidden = [
            module
            for module in (sys.modules.pop(name, None) for name in HIDDEN_MODULES)
            if module is not None
        ]
        install_network_guard()
        import torch
    except BaseException as error:  # pragma: no cover - reported to the parent
        connection.send(("error", f"child start-up failed: {error!r}"))
        return

    captured_sync = torch.cuda.synchronize
    captured_event = torch.cuda.Event
    captured_empty = torch.empty
    captured_from_numpy = torch.from_numpy
    captured_ascontiguous = np.ascontiguousarray
    captured_copy = torch.Tensor.copy_
    captured_fill = torch.Tensor.fill_
    captured_zero = torch.Tensor.zero_
    captured_min = torch.Tensor.min
    captured_max = torch.Tensor.max
    captured_item = torch.Tensor.item
    captured_detach = torch.Tensor.detach
    captured_to = torch.Tensor.to
    captured_cpu = torch.Tensor.cpu
    captured_numpy = torch.Tensor.numpy
    captured_int64 = torch.int64
    captured_float32 = torch.float32
    integer_dtypes = (torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8)
    tensor_type = torch.Tensor

    state = {"inputs": None, "kernel": None, "stage_ms": 0.0, "flush": None}

    def sync():
        if device == "cuda":
            captured_sync()

    def flush_l2():
        """Evict the L2 so every timed call starts from the same cache state."""
        if device != "cuda":
            return
        buffer = state["flush"]
        if buffer is None:
            buffer = captured_empty(
                L2_FLUSH_BYTES // 4, dtype=captured_float32, device="cuda"
            )
            state["flush"] = buffer
        captured_fill(buffer, 0.0)

    def ensure_inputs(train_shape, test_shape):
        inputs = state["inputs"]
        if inputs is None:
            inputs = (
                captured_empty(train_shape, dtype=captured_float32, device=device),
                captured_empty(train_shape[0], dtype=captured_int64, device=device),
                captured_empty(test_shape, dtype=captured_float32, device=device),
            )
            state["inputs"] = inputs
        elif tuple(inputs[0].shape) != tuple(train_shape) or tuple(inputs[2].shape) != tuple(
            test_shape
        ):
            raise ValueError("every draw in one evaluation must share its shapes")
        return inputs

    def stage(draw):
        """Copy a draw into the fixed input tensors, and time the copy.

        Staging is untimed by design, which makes it a place to hide work
        (patch ``Tensor.copy_``, train while the third input lands, return a
        pre-computed answer from a zero-cost call). The parent compares this
        number against a baseline measured before the submission existed.
        """
        started = perf()
        inputs = ensure_inputs(draw[0].shape, draw[2].shape)
        for tensor, array in zip(inputs, draw):
            captured_copy(tensor, captured_from_numpy(captured_ascontiguous(array)))
        sync()
        state["stage_ms"] = (perf() - started) * 1e3
        return state["stage_ms"]

    def validate(output, count):
        # Exactly torch.Tensor, not a subclass: a lazy or proxy tensor could
        # defer its real work into this untimed check (reference-kernels#161).
        if type(output) is not tensor_type:
            raise TypeError(
                f"custom_kernel returned {type(output).__name__}, expected a plain torch.Tensor"
            )
        if output.device.type != device:
            raise TypeError(
                f"custom_kernel returned a {output.device.type} tensor, expected {device}"
            )
        if tuple(output.shape) != (count,):
            raise ValueError(
                f"custom_kernel returned shape {tuple(output.shape)}, expected ({count},)"
            )
        if output.dtype not in integer_dtypes:
            raise TypeError(f"custom_kernel returned dtype {output.dtype}, expected an integer type")
        low = int(captured_item(captured_min(output)))
        high = int(captured_item(captured_max(output)))
        if low < 0 or high > 9:
            raise ValueError(f"predicted labels must lie in [0, 9], saw [{low}, {high}]")

    def to_host(output):
        return captured_numpy(captured_cpu(captured_to(captured_detach(output), captured_int64))).copy()

    def staged_inputs():
        inputs = state["inputs"]
        if inputs is None:
            raise RuntimeError("no draw has been staged")
        return inputs, int(inputs[2].shape[0])

    def timed_call():
        """One timed call on the draw the parent has already staged.

        The child's wall clock deliberately also covers validating the output
        and copying it to the host: work deferred past ``end_event`` into that
        window shows up as a device time far below the wall time.
        """
        inputs, count = staged_inputs()
        kernel = state["kernel"]
        if kernel is None:
            raise RuntimeError("no submission has been loaded")
        if device == "cuda":
            captured_sync()
            flush_l2()
            # Drain the flush before the clocks start, so the device clock and
            # the two wall clocks bracket the same work and can be compared.
            captured_sync()
            start_event = captured_event(enable_timing=True)
            end_event = captured_event(enable_timing=True)
            wall_start = perf()
            start_event.record()
            output = kernel(inputs)
            end_event.record()
            captured_sync()
            device_ms = float(start_event.elapsed_time(end_event))
        else:
            wall_start = perf()
            output = kernel(inputs)
            device_ms = (perf() - wall_start) * 1e3
        validate(output, count)
        predictions = to_host(output)
        del output
        wall_ms = (perf() - wall_start) * 1e3
        return predictions, device_ms, wall_ms, state["stage_ms"]

    def probe_call():
        """A timed call with no submission in it: the cost of the protocol itself.

        The CUDA preamble is part of that cost and must be measured here too:
        the two barriers and the 256 MB L2 flush sit inside the parent's
        round trip of a timed call but outside the child's wall clock, and the
        flush buffer's first allocation is several milliseconds on its own. A
        probe that skipped them under-measured the protocol by ~10 ms on an
        A100 and the parent-clock bound then rejected honest millisecond
        kernels (GPU run 1, 2026-09-22).
        """
        inputs, count = staged_inputs()
        if device == "cuda":
            captured_sync()
            flush_l2()
            captured_sync()
        wall_start = perf()
        output = captured_empty(count, dtype=captured_int64, device=device)
        captured_zero(output)
        sync()
        validate(output, count)
        predictions = to_host(output)
        del output
        wall_ms = (perf() - wall_start) * 1e3
        return predictions, 0.0, wall_ms, state["stage_ms"]

    def load_submission():
        from submission import custom_kernel

        state["kernel"] = custom_kernel
        return True

    def untimed_call(draw):
        stage(draw)
        inputs, count = staged_inputs()
        kernel = state["kernel"]
        if kernel is None:
            raise RuntimeError("no submission has been loaded")
        output = kernel(inputs)
        sync()
        validate(output, count)
        predictions = to_host(output)
        del output
        return predictions

    def profile_call():
        from torch.profiler import ProfilerActivity, profile

        inputs, _ = staged_inputs()
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities) as prof:
            state["kernel"](inputs)
            sync()
        key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
        return prof.key_averages().table(sort_by=key, row_limit=20)

    connection.send(("ready", os.getpid()))
    while True:
        try:
            command, args = connection.recv()
        except EOFError:  # pragma: no cover - parent went away
            return
        if command == "stop":
            connection.send(("ok", None))
            return
        try:
            if command == "stage":
                connection.send(("ok", stage(*args)))
            elif command == "timed":
                connection.send(("ok", timed_call()))
            elif command == "probe":
                connection.send(("ok", probe_call()))
            elif command == "load":
                connection.send(("ok", load_submission()))
            elif command == "untimed":
                connection.send(("ok", untimed_call(*args)))
            elif command == "profile":
                connection.send(("ok", profile_call()))
            elif command == "sysinfo":
                connection.send(("ok", system_info()))
            else:
                connection.send(("error", f"unknown command {command!r}"))
        except BaseException as error:
            connection.send(("error", repr(error)))


class Child:
    """The submission's process, plus the parent-side clock around every call."""

    def __init__(self, device, work_dir, startup_timeout_s=120.0):
        context = multiprocessing.get_context("spawn")
        self.connection, remote = context.Pipe()
        self.process = context.Process(
            target=child_main, args=(remote, device, str(work_dir)), daemon=True
        )
        self.process.start()
        remote.close()  # otherwise a dead child never produces EOF and recv() hangs
        if not self.connection.poll(startup_timeout_s):
            self.kill()
            raise Failure(f"the submission process did not start within {startup_timeout_s:.0f} s")
        status, value = self.connection.recv()
        if status != "ready":
            raise RuntimeError(value)

    def call(self, command, *args, timeout_s=None):
        """Send one command and wait for its reply, bounded by ``timeout_s``.

        The deadline is enforced by a watchdog that kills the process group,
        not by ``poll``: a submission that scribbles on low file descriptors
        corrupts the pipe, and then ``poll`` reports data while ``recv`` blocks
        forever on a message that will never arrive. Killing the child closes
        its end and turns the wedge into a failed run instead of a mode timeout
        with no result at all.
        """
        import threading

        started = time.perf_counter()
        self.connection.send((command, args))
        expired = []
        watchdog = None
        if timeout_s is not None:
            def fire():
                expired.append(True)
                self.kill()

            watchdog = threading.Timer(timeout_s, fire)
            watchdog.daemon = True
            watchdog.start()
        try:
            status, value = self.connection.recv()
        except Exception as error:
            if expired:
                raise Failure(
                    f"the submission did not return from {command!r} within "
                    f"{timeout_s:.0f} s"
                ) from None
            raise Failure(
                f"the submission process died during {command!r}: {error!r}"
            ) from None
        finally:
            if watchdog is not None:
                watchdog.cancel()
        wall_ms = (time.perf_counter() - started) * 1e3
        if status != "ok":
            raise RuntimeError(f"submission {command} failed: {value}")
        return value, wall_ms

    def kill(self):
        """Kill the submission and anything it spawned (it runs in its own session)."""
        pid = self.process.pid
        if pid:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError, OSError):
                pass
        if self.process.is_alive():
            self.process.kill()
        self.process.join(5)

    def stop(self):
        if self.process.is_alive():
            try:
                self.connection.send(("stop", ()))
                if self.connection.poll(15):
                    self.connection.recv()
            except (EOFError, OSError, BrokenPipeError):
                pass
            self.process.join(5)
        self.kill()


# ------------------------------------------------------------------ protocol

def calibrate(child, warm_visible):
    """Measure the protocol's own cost before the submission is imported.

    Three numbers, each the cheapest of ``CALIBRATION_ROUNDS`` observations:

    ``ipc_ms``          the parent-measured round trip of a *probe* call, which
                        does everything a timed call does except run the
                        submission: allocate the output, validate it, copy it to
                        the host and send it back. Subtracting this from the
                        parent's measurement of a real timed call leaves the
                        call itself, so the parent's clock bounds the device
                        clock from below even for a short kernel.
    ``stage_round_ms``  the parent-measured round trip of staging one draw. The
                        6.5 MB of input travel here, outside the timed window
                        and outside the bound above, so staging gets its own
                        ceiling instead of loosening the timing gate.
    ``stage_child_ms``  what the child reports staging cost it.

    All three are measured with no submission in the process, so nothing it does
    can poison them to buy itself slack. (Calibrating after the import is not
    safe: a patched ``torch.zeros`` that busy-waits only for the probe's exact
    signature inflates the overhead and makes the budget negative.)
    """
    probes, stage_rounds, stage_costs = [], [], []
    for _ in range(CALIBRATION_ROUNDS):
        _, stage_round_ms = child.call("stage", warm_visible, timeout_s=120)
        (_, _, _, stage_ms), probe_ms = child.call("probe", timeout_s=120)
        probes.append(probe_ms)
        stage_rounds.append(stage_round_ms)
        stage_costs.append(stage_ms)
    # The staging ceiling uses the *worst* calibration round: the transfer is
    # 6.5 MB and its cost is noisy, and this bound is a coarse tripwire, not
    # the number that ranks anybody.
    return min(probes), max(stage_rounds), min(stage_costs)


def run_case(child, pools, case, *, timed_draws, holdout, floor_bp):
    """Warm up once on foreign data, then run the timed calls on fresh draws.

    Hold-out calls (leaderboard mode) are drawn from Fashion-MNIST, land at
    positions chosen by the secret seed, and are timed and ranked exactly like
    the MNIST calls -- they are not a free compute slot, and their position
    cannot be predicted. Their accuracy is scored separately, against the
    hold-out floor.
    """
    pool = pools[("mnist", case["size"])]
    fashion = pools[("fashion", case["size"])]
    seed = case["seed"]
    n_train, n_test = case["train"], case["test"]
    universes = {
        "mnist": split_universes(len(pool[1]), seed, UNIVERSE_SALT),
        "fashion": split_universes(len(fashion[1]), seed, UNIVERSE_SALT + 1),
    }
    call_timeout = case["max_call_ms"] / 1000.0 + 30.0
    warmup_timeout = case["warmup_max_call_ms"] / 1000.0 + 30.0

    # The warm-up draw is foreign data of identical shape: compilation,
    # autotuning and CUDA-graph capture are still free, but a model fitted here
    # is about the wrong dataset, so training cannot be moved out of the timed
    # window.
    warm_visible, _ = make_draw(
        fashion, seed + WARMUP_OFFSET, n_train, n_test, universes["fashion"]
    )
    ipc_ms, stage_round_base_ms, stage_base_ms = calibrate(child, warm_visible)
    child.call("load", timeout_s=warmup_timeout)
    child.call("untimed", warm_visible, timeout_s=warmup_timeout)

    holdout_draws = case["holdout_draws"] if holdout else 0
    total_calls = timed_draws + holdout_draws
    holdout_positions = set()
    if holdout_draws:
        rng = np.random.default_rng([seed, HOLDOUT_OFFSET])
        holdout_positions = {
            int(value) for value in rng.choice(total_calls, holdout_draws, replace=False)
        }

    draw_floor = required_correct(n_test, min(10000, floor_bp))
    durations_ns, parent_ms_list, child_ms_list, stage_ms_list = [], [], [], []
    parent_stage_ms_list = []
    per_draw, holdout_per_draw, holdout_ms_list = [], [], []
    loop_started = time.perf_counter()
    for index in range(total_calls):
        is_holdout = index in holdout_positions
        if is_holdout:
            draw_seed = seed + HOLDOUT_OFFSET + HOLDOUT_STRIDE * (len(holdout_per_draw) + 1)
            visible, truth = make_draw(
                fashion, draw_seed, n_train, n_test, universes["fashion"]
            )
        else:
            draw_seed = seed + TIMED_STRIDE * (len(per_draw) + 1)
            visible, truth = make_draw(pool, draw_seed, n_train, n_test, universes["mnist"])

        # Staging and the call are separate round trips, each with its own
        # parent-side clock: the input transfer does not loosen the bound on the
        # call, and work hijacked into staging does not hide behind it.
        _, parent_stage_ms = child.call("stage", visible, timeout_s=call_timeout)
        (predictions, device_ms, child_wall_ms, stage_ms), parent_span_ms = child.call(
            "timed", timeout_s=call_timeout
        )

        if device_ms > case["max_call_ms"]:
            raise Failure(
                f"one call took {device_ms:.0f} ms, over the {case['max_call_ms']} ms limit"
            )
        if parent_stage_ms > STAGE_FACTOR * stage_round_base_ms + STAGE_SLACK_MS:
            raise Failure(
                f"staging draw {index} took {parent_stage_ms:.3f} ms against a "
                f"{stage_round_base_ms:.3f} ms baseline measured before the submission was "
                "imported; work is running before the timed window"
            )
        if stage_ms > CHILD_STAGE_FACTOR * stage_base_ms + CHILD_STAGE_SLACK_MS:
            raise Failure(
                f"staging draw {index} took {stage_ms:.3f} ms against a "
                f"{stage_base_ms:.3f} ms baseline; work is running before the timed window"
            )
        reason = timing_plausible(
            device_ms, child_wall_ms, parent_span_ms, min(ipc_ms, 0.5 * parent_span_ms)
        )
        if reason is not None:
            raise Failure(f"timing implausible on call {index}: {reason}")
        if predictions.shape != truth.shape:
            raise Failure(f"predictions shape {predictions.shape}, expected {truth.shape}")

        correct = int((predictions == truth).sum())
        durations_ns.append(device_ms * 1e6)
        parent_ms_list.append(parent_span_ms)
        parent_stage_ms_list.append(parent_stage_ms)
        child_ms_list.append(child_wall_ms)
        stage_ms_list.append(stage_ms)
        if is_holdout:
            holdout_per_draw.append(correct)
            holdout_ms_list.append(device_ms)
        else:
            if correct < draw_floor:
                raise Failure(
                    f"draw {len(per_draw)} scored {correct} of {n_test}, below the per-draw "
                    f"floor {draw_floor}; every ranked call must do the same work"
                )
            per_draw.append(correct)

    report = {
        "per_draw": per_draw,
        "correct": sum(per_draw),
        "total": n_test * timed_draws,
        "draw_floor": draw_floor,
        "stats": stats(durations_ns),
        "durations_ns": [round(value) for value in durations_ns],
        "wall_mean": sum(parent_ms_list) / len(parent_ms_list),
        "child_wall_mean": sum(child_ms_list) / len(child_ms_list),
        "stage_mean": sum(parent_stage_ms_list) / len(parent_stage_ms_list),
        "child_stage_mean": sum(stage_ms_list) / len(stage_ms_list),
        "ipc_calibration": ipc_ms,
        "stage_calibration": stage_round_base_ms,
        "loop_wall_ms": (time.perf_counter() - loop_started) * 1e3,
    }
    report["required"] = required_correct(report["total"], case["error_bp"])
    report["accuracy"] = report["correct"] / report["total"]

    # An honest learner does the same work on every equally shaped draw. A
    # submission that trains once and reuses the model shows up as one call far
    # above the median; so does one that keeps a cheap path for draws it
    # recognizes.
    dispersion = case["dispersion_x10"] / 10.0
    median_ns = report["stats"]["median"]
    worst_ns = report["stats"]["worst"]
    if len(durations_ns) >= 3 and worst_ns > dispersion * median_ns + 2e6:
        raise Failure(
            f"call durations differ by {worst_ns / max(median_ns, 1.0):.1f}x "
            f"(worst {worst_ns / 1e6:.3f} ms, median {median_ns / 1e6:.3f} ms); "
            "the ranked calls are not doing the same work"
        )

    if holdout_draws:
        report["holdout_per_draw"] = holdout_per_draw
        report["holdout_correct"] = sum(holdout_per_draw)
        report["holdout_total"] = n_test * holdout_draws
        report["holdout_accuracy"] = report["holdout_correct"] / report["holdout_total"]
        report["holdout_required"] = required_correct(
            report["holdout_total"], case["holdout_min_bp"]
        )
        report["holdout_ms"] = sum(holdout_ms_list) / len(holdout_ms_list)
        report["holdout_ratio"] = report["holdout_ms"] * 1e6 / max(median_ns, 1.0)
    return report


# ------------------------------------------------------------------ modes

def log_system(out, child):
    try:
        info, _ = child.call("sysinfo", timeout_s=120)
    except Exception as error:  # pragma: no cover
        out.log("system.error", repr(error))
        return
    for key, value in info.items():
        out.log(f"system.{key}", value)


def run_test(out, pools, cases, device):
    out.log("test-count", len(cases))
    passed = True
    for index, case in enumerate(cases):
        out.log(f"test.{index}.spec", case["spec"])
        child = None
        work_dir = prepare_submission_dir()
        loose_bp = min(10000, case["error_bp"] + TEST_SLACK_BP)
        try:
            child = Child(device, work_dir)
            if index == 0:
                log_system(out, child)
            report = run_case(
                child,
                pools,
                case,
                timed_draws=case["draws"],
                holdout=False,
                floor_bp=min(10000, loose_bp + case["draw_slack_bp"]),
            )
        except Failure as error:
            out.log(f"test.{index}.status", "fail")
            out.log(f"test.{index}.error", str(error))
            passed = False
            continue
        except Exception as error:
            out.log(f"test.{index}.status", "fail")
            out.log(f"test.{index}.error", repr(error)[:2000])
            passed = False
            continue
        finally:
            if child is not None:
                child.stop()
            shutil.rmtree(work_dir, ignore_errors=True)
        loose = required_correct(report["total"], loose_bp)
        good = report["correct"] >= loose
        out.log(f"test.{index}.status", "pass" if good else "fail")
        message = (
            f"{report['correct']}/{report['total']} correct "
            f"({100 * report['accuracy']:.2f}%), needs {loose} at the test-mode slack "
            f"and {report['required']} on the leaderboard; "
            f"{report['stats']['mean'] / 1e6:.3f} ms per call"
        )
        out.log(f"test.{index}.message", message)
        if not good:
            out.log(f"test.{index}.error", "accuracy below the test-mode threshold")
        passed = passed and good
    out.log("check", "pass" if passed else "fail")
    return EXIT_SUCCESS if passed else EXIT_VALIDATE_FAIL


def run_ranked(out, pools, cases, device, holdout):
    out.log("benchmark-count", len(cases))
    passed = True
    for index, case in enumerate(cases):
        out.log(f"benchmark.{index}.spec", case["spec"])
        child = None
        work_dir = prepare_submission_dir()
        try:
            child = Child(device, work_dir)
            if index == 0:
                log_system(out, child)
            use_holdout = bool(holdout and case["holdout"])
            # benchmark mode reuses the ranked case line but runs fewer draws;
            # KernelBot hands both modes the same benchmarks entry.
            timed_draws = case["draws"] if holdout else min(case["draws"], case["bench_draws"])
            report = run_case(
                child,
                pools,
                case,
                timed_draws=timed_draws,
                holdout=use_holdout,
                floor_bp=min(10000, case["error_bp"] + case["draw_slack_bp"]),
            )
        except Failure as error:
            out.log(f"benchmark.{index}.status", "fail")
            out.log(f"benchmark.{index}.error", str(error))
            passed = False
            break
        except Exception as error:
            out.log(f"benchmark.{index}.status", "fail")
            out.log(f"benchmark.{index}.error", repr(error)[:2000])
            passed = False
            break
        finally:
            if child is not None:
                child.stop()
            shutil.rmtree(work_dir, ignore_errors=True)

        for key, value in report["stats"].items():
            out.log(f"benchmark.{index}.{key}", value)
        out.log(f"benchmark.{index}.accuracy", f"{report['accuracy']:.6f}")
        out.log(f"benchmark.{index}.correct", report["correct"])
        out.log(f"benchmark.{index}.total", report["total"])
        out.log(f"benchmark.{index}.required", report["required"])
        out.log(f"benchmark.{index}.draw_floor", report["draw_floor"])
        out.log(f"benchmark.{index}.per_draw", json.dumps(report["per_draw"]))
        out.log(f"benchmark.{index}.durations", json.dumps(report["durations_ns"]))
        out.log(f"benchmark.{index}.wall_mean", f"{report['wall_mean'] * 1e6:.1f}")
        out.log(f"benchmark.{index}.child_wall_mean", f"{report['child_wall_mean'] * 1e6:.1f}")
        out.log(f"benchmark.{index}.stage_mean", f"{report['stage_mean'] * 1e6:.1f}")
        out.log(f"benchmark.{index}.ipc_calibration", f"{report['ipc_calibration'] * 1e6:.1f}")
        out.log(
            f"benchmark.{index}.stage_calibration", f"{report['stage_calibration'] * 1e6:.1f}"
        )
        out.log(f"benchmark.{index}.loop_wall", f"{report['loop_wall_ms'] * 1e6:.1f}")
        good = report["correct"] >= report["required"]
        if not good:
            out.log(f"benchmark.{index}.status", "fail")
            out.log(
                f"benchmark.{index}.error",
                f"accuracy gate: {report['correct']} correct of {report['total']}, "
                f"needs {report['required']} for the {case['error_bp'] / 100:g}% band",
            )
        if "holdout_accuracy" in report:
            out.log(f"benchmark.{index}.holdout_accuracy", f"{report['holdout_accuracy']:.6f}")
            out.log(f"benchmark.{index}.holdout_correct", report["holdout_correct"])
            out.log(f"benchmark.{index}.holdout_total", report["holdout_total"])
            out.log(f"benchmark.{index}.holdout_required", report["holdout_required"])
            out.log(f"benchmark.{index}.holdout_per_draw", json.dumps(report["holdout_per_draw"]))
            out.log(f"benchmark.{index}.holdout_ms", f"{report['holdout_ms']:.3f}")
            out.log(f"benchmark.{index}.holdout_ratio", f"{report['holdout_ratio']:.2f}")
            if report["holdout_correct"] < report["holdout_required"]:
                good = False
                out.log(f"benchmark.{index}.status", "fail")
                out.log(
                    f"benchmark.{index}.error",
                    f"hold-out check: {report['holdout_correct']} correct of "
                    f"{report['holdout_total']} Fashion-MNIST queries, needs "
                    f"{report['holdout_required']}; this submission does not appear to "
                    "learn from the data it is given",
                )
        passed = passed and good
        if not good:
            break
    out.log("check", "pass" if passed else "fail")
    return EXIT_SUCCESS if passed else EXIT_VALIDATE_FAIL


def run_profile(out, pools, cases, device):
    out.log("benchmark-count", len(cases))
    for index, case in enumerate(cases):
        out.log(f"benchmark.{index}.spec", case["spec"])
        child = None
        work_dir = prepare_submission_dir()
        try:
            child = Child(device, work_dir)
            pool = pools[("mnist", case["size"])]
            universes = split_universes(len(pool[1]), case["seed"], UNIVERSE_SALT)
            visible, _ = make_draw(
                pool, case["seed"] + WARMUP_OFFSET, case["train"], case["test"], universes
            )
            timeout = case["warmup_max_call_ms"] / 1000.0 + 30.0
            child.call("load", timeout_s=timeout)
            child.call("untimed", visible, timeout_s=timeout)
            child.call("stage", visible, timeout_s=timeout)
            report, _ = child.call("profile", timeout_s=timeout)
        finally:
            if child is not None:
                child.stop()
            shutil.rmtree(work_dir, ignore_errors=True)
        out.log(
            f"benchmark.{index}.report",
            base64.b64encode(report.encode("utf-8"), b"+*").decode("utf-8"),
        )
    out.log("check", "pass")
    return EXIT_SUCCESS


# ------------------------------------------------------------------ entry point

def resolve_device():
    requested = os.environ.get("MNIST_EVAL_DEVICE")
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def scrub_secret_environment():
    """Re-exec once with the secret out of the environment, and return it.

    ``os.environ.pop`` does not rewrite the region ``/proc/<pid>/environ``
    exposes, so a child process can read the parent's original environment and
    recover POPCORN_SEED -- with which every draw, permutation and hold-out of
    the run is reconstructible. Passing the secret through an inherited pipe
    and re-exec'ing removes it from the process image entirely.
    """
    handoff = os.environ.pop(SECRET_FD_ENV, None)
    if handoff:
        try:
            with os.fdopen(int(handoff), "r") as handle:
                return json.loads(handle.read() or "{}")
        except (OSError, ValueError):  # pragma: no cover
            return {}
    secret = os.environ.get("POPCORN_SEED")
    cache = os.environ.get("MNIST_POOL_CACHE")
    consume = os.environ.get("MNIST_POOL_CONSUME")
    if secret is None and cache is None:
        return {}
    payload = {"secret": secret, "cache": cache, "consume": consume}
    read_fd, write_fd = os.pipe()
    os.write(write_fd, json.dumps(payload).encode("utf-8"))
    os.close(write_fd)
    os.set_inheritable(read_fd, True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("POPCORN_SEED", "MNIST_POOL_CACHE", "MNIST_POOL_CONSUME")
    }
    environment[SECRET_FD_ENV] = str(read_fd)
    result_fd = os.environ.get("POPCORN_FD")
    if result_fd:
        try:
            os.set_inheritable(int(result_fd), True)
        except (OSError, ValueError):  # pragma: no cover
            pass
    try:
        os.execve(
            sys.executable,
            [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]],
            environment,
        )
    except OSError as error:  # pragma: no cover - fall back, but say so
        os.close(read_fd)
        debug(f"could not re-exec to scrub the environment: {error!r}")
        os.environ.pop("POPCORN_SEED", None)
        os.environ.pop("MNIST_POOL_CACHE", None)
        os.environ.pop("MNIST_POOL_CONSUME", None)
        return payload


def main():
    secrets = scrub_secret_environment()
    fd = os.getenv("POPCORN_FD")
    if not fd:
        return EXIT_NO_FD
    if len(sys.argv) < 3:
        return 2
    mode = sys.argv[1]
    secret = secrets.get("secret")
    try:
        cases = read_cases(sys.argv[2], int(secret) if secret else None)
    except Exception as error:
        print(f"could not read cases: {error}", file=sys.stderr)
        return EXIT_BAD_CASES
    set_seed(int(secret) % (2**31) if secret else 42)
    device = resolve_device()
    cache = secrets.get("cache")
    out = PopcornOutput(int(fd))
    os.environ.pop("POPCORN_FD", None)  # the fd is already non-inheritable
    started = time.perf_counter()
    try:
        check_submission_source(cases[0])
        pools = load_pools(cases, cache, consume=bool(secrets.get("consume")))
        if mode == "test":
            return run_test(out, pools, cases, device)
        if mode == "benchmark":
            return run_ranked(out, pools, cases, device, holdout=False)
        if mode == "leaderboard":
            return run_ranked(out, pools, cases, device, holdout=True)
        if mode == "profile":
            return run_profile(out, pools, cases, device)
        return 2
    except Failure as error:
        out.log("check", "fail")
        out.log("error", str(error)[:2000])
        debug(f"{mode} failed: {error}")
        return EXIT_VALIDATE_FAIL
    except Exception as error:
        out.log("check", "fail")
        out.log("error", repr(error)[:2000])
        debug(f"{mode} failed: {error!r}")
        return EXIT_VALIDATE_FAIL
    finally:
        debug(
            f"{HARNESS_VERSION} {mode} on {device} finished in "
            f"{time.perf_counter() - started:.1f} s"
        )
        out.close()


if __name__ == "__main__":
    sys.exit(main())
