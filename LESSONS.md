# Lessons learned

What building, reviewing and running this site taught, written so the next
deployment does not rediscover it. Sources: the build session on 2026-09-22,
a 64-agent adversarial review of the first version (4 reviewers with
different lenses, 3 skeptics per finding; 19 findings confirmed, 1 refuted),
and six live runs on Modal.

## 1. Budget and ledger design

**Reserve the real worst case, then charge the measured time, uncapped.**
The first version reserved the evaluator's per-mode timeouts but the
container could live 2400 s (Modal's function timeout), and it clamped every
charge to the reservation. Net effect: a `test` run could cost 4.5× what the
ledger booked, and 111 such runs fit under a $50 cap while costing ~$200.
The fix has three parts, and all three are needed: (a) the worker enforces
the reserved duration as a hard deadline (kills the process group), so the
reservation *is* the ceiling; (b) the charge is the measured container time
with no cap, and an `overrun_usd` field makes any excess visible; (c) Modal's
own 2400 s timeout stays as a backstop; if it ever fires, the deadline
logic has failed, and the run is charged the whole container life (2400 s
plus the pad, about $2.05) rather than its reservation, so even that path
cannot under-count.

**The deadline must cover everything the container does, not just the
kernel.** The evaluator's dataset load (verifying the baked files and
box-area resizing 60,000 images, twice for the hold-out) happens inside
`eval.py`, so it is already inside the per-step timeouts the deadline sums;
the 180 s pad on top of them is slack for the runner's own start-up, the
baked-pool lift into RAM and general margin, applied once per run rather
than per step. Container boot and teardown are outside the deadline; 60 s is
the pad for them, and it is added to every charge as well as every
reservation.

**Price at the dearest part Modal may hand you.** `gpu="A100"` was fulfilled
with a 40 GB card on some runs and an 80 GB card on others. The 80 GB rate
is 19% higher. Reserve and charge at the 80 GB rate, plus 20% for the 2
cores, 8 GiB and per-second rounding. Observed cost per run on the site is
$0.05–0.08 for short kernels; the reservation is $0.45 (test), $0.70
(benchmark), $1.95 (leaderboard).

**Persist the reservation before asking for a GPU.** Spawning first and
saving the record second leaves a window where a GPU job exists with no
ledger entry (the save can fail: request too large, transient RPC error,
container restart). Now the record is saved with `call_id=None`, the spawn
happens, and the id is saved after. A record with no call id for 3 minutes
is written off at its reservation; a spawn that raises releases the
reservation because nothing ran.

**The ledger needs a floor that cannot expire.** `modal.Dict` entries
expire after 7 days without a read or write. If the records vanished, so
would the memory of what was spent, and the cap would silently reset. Two
defences: the 10-minute cron rewrites every key (a write refreshes the TTL),
and every charge is also written as an immutable file on a `modal.Volume`;
the ledger takes the larger of the two sums. Deleting the Volume is the only
way to forget spend, and the runbook says so.

**One bad payload must not take the site down.** Settlement runs on every
page load. If parsing one worker result raised, every route 500'd and the
reservation stayed pinned forever. Now each record settles inside its own
try/except, an unparseable result becomes an `error` record charged its
reservation, and every numeric field in the summary is parsed defensively
(a bad value becomes `null`, never an exception).

**Transport errors are not verdicts.** `FunctionCall.get(timeout=0)` raises
the builtin `TimeoutError` when the run is simply not finished; Modal's own
`FunctionTimeoutError` (the container hit its limit) and `OutputExpiredError`
are different classes and mean the run is over. Connection and internal
errors mean *we* could not ask; those are retried on later settles and only
written off (and the call cancelled) after 5 consecutive failures.

**Cancel what you write off.** When a run is written off as an error, the
call is cancelled with `terminate_containers=True` so the GPU stops burning.
The owner's `::cancel` command uses the same path.

**Fresh container per run.** With `single_use_containers=True` nothing a
kernel leaves behind (a stray process, GPU memory, a modified environment)
can affect the next submitter's timings or hold the container alive. The
extra cold start is inside the 60 s pad.

**Modal may re-run an input after an infrastructure failure even with
`retries=0`.** It happened once during testing (a preemption). The second
container's time is what gets charged; the first attempt's time is lost from
the ledger. Accepted and documented; the billing page is the source of
truth.

## 2. Modal platform behaviour

- **`uvx modal …`** is the reliable way to run the CLI on a machine whose
  default `python3` is old or has no `modal` installed. Pin
  `uv run --python 3.12` for anything importing `modal` or `fastapi`.
- **`modal run app.py::entrypoint` creates a throwaway app** with its own
  `-dev.modal.run` URL and prints it. To find the *deployed* site's URL from
  an entrypoint, look it up with
  `modal.Function.from_name(APP_NAME, "web").get_web_url()`, not from the
  function object in the file.
- **Deploys are fast once images are cached** (2–3 s); the first build of the
  CUDA image with `torch==2.12.0` and the baked datasets takes minutes. The
  torch pin resolved against `nvidia/cuda:13.3.0-devel-ubuntu24.04` on
  2026-09-22 (`torch 2.12.0+cu130`, Python 3.13.0).
- **`restrict_modal_access=True`** on the worker stops a kernel from using
  the container's Modal credentials to spawn work outside the ledger. The
  worker can still return its result. It cannot write to a Dict or Volume,
  which is why the worker only *returns* the payload and the web/cron side
  does all bookkeeping.
- **`block_network=True`** on the worker requires the datasets to be in the
  image. `run_modal.py` fetches the four files at build time; the runner
  lifts them into RAM and deletes them before any kernel process exists.
  Reuse `run_modal.image` rather than re-declaring the image in `app.py`, so
  the site and the standalone runner cannot drift.
- **`Image.add_local_dir` must be the last step** of an image chain and its
  `ignore` callable sees paths relative to the directory. Exclude
  `__pycache__` and results so image hashes stay stable.
- **`modal.Dict.items()` transfers every value.** Keep records small and put
  kernel text and evaluator output under separate keys (here, a separate
  Dict) so the index page and the ledger do not download every kernel.
- **Two `modal.App` objects can coexist in one process.** `run_modal.py`
  declares its own app at import time; deploying `web/app.py` only deploys
  the app defined in that file. Harmless, but do not name them the same.
- **Preemption is real.** One web-container request returned 500 after 28 s
  with no application traceback, exactly when Modal logged "Container
  terminated due to preemption". Clients should retry; the smoke script does.
- **Volumes need `commit()` after writes and `reload()` before reads from
  another container.** Writing one immutable file per charge avoids
  conflicts between the web container and the cron.

## 3. Web and Python pitfalls

- **`from __future__ import annotations` breaks FastAPI when the route
  functions live inside a factory** and import `Request`/`UploadFile`
  locally: the string annotations cannot be resolved and the routes 422 on
  every request. Either import those names at module level or, as here,
  parse the form manually and drop the future import.
- **Do blocking Modal calls in a threadpool** from `async def` handlers
  (`starlette.concurrency.run_in_threadpool`); otherwise the event loop
  stalls for the duration of every gRPC round trip, and Modal prints an
  `AsyncUsageWarning`.
- **`secrets.compare_digest` on `str` raises `TypeError` for non-ASCII
  input.** A URL like `/café` produced a 500 instead of a 404. Check
  `token.isascii()` first.
- **Guard request size before parsing multipart.** Starlette spools the
  whole body before you can look at a field; check `Content-Length` first
  and read uploads with a bound.
- **Keep the user's typed kernel on every error path.** One branch
  (non-UTF-8 upload) dropped it and re-rendered the template, losing work.
- **Label benchmark-mode numbers honestly.** `benchmark` runs 3 draws and no
  hold-out; only `leaderboard` produces the ranked number.
- **When a later step fails without stats, fall back to the last step that
  has them** (a killed leaderboard step still leaves the benchmark stats).
- **Render defensively.** Records written by older versions of the app lack
  fields; every formatter accepts `None`.

## 4. Harness integration

- **Reuse the harness's own building blocks** (`load_task`, `evaluate`,
  `run_one`, `image`) rather than copying them. The site then follows every
  change to timeouts, case lines, image pins and pool handling by
  redeploying.
- **Run the evaluation in its own process group under the deadline.** The
  worker starts `runner.py` with `start_new_session=True` and `killpg`s it
  when the deadline passes. `run_one` does the same per evaluator step and
  drains the result pipe from a thread from the start, so a process that
  outlives `eval.py` can neither hold the GPU nor block the pipe read past
  the timeout. Tests cover both with real subprocesses.
- **Timings depend on the harness version.** The same baseline kernel
  measured 0.69 ms on harness 1.0.0, 5.0 ms on an intermediate 1.1.0 build
  and 0.71 ms on the final 1.1.0. Every result carries `system.harness`;
  compare only within a version.
- **The evaluator's tests belong to the evaluator.** While the harness was
  being edited in parallel, `tests/test_eval.py` failed for reasons
  unrelated to the site. Pull the harness and its tests together.
- **The evaluator's result channel is plain text lines.** Treat every
  `key: value` as untrusted: parse numbers defensively, and remember that a
  kernel with access to the process's file descriptors could in principle
  write lines of its own. Harness version 1.1.0 moved the secret seed out of
  the environment and confines the kernel to a private directory; forged
  result lines remain a harness-level concern the site does not detect.

## 5. Process

- **Smoke test the cheapest path first, on the real platform.** The
  test-mode run of the baseline kernel costs about $0.06 and caught the
  40 GB/80 GB pricing difference, the harness version drift and the
  preemption behaviour, none of which unit tests could.
- **Adversarial review before trusting a budget mechanism.** Independent
  reviewers with distinct lenses (budget, Modal API, harness fidelity,
  web), each finding challenged by three skeptics, turned up the reservation
  gap, the pipe hang, the Dict expiry and the spawn-before-save race. The
  refuted finding was a README wording nit.
- **Keep the ledger pessimistic and say so on the page.** The budget bar
  explains the reservation model in one sentence; the billing dashboard is
  named as the source of truth everywhere.
- **Expect the harness to move while you work.** Files changed under the
  site three times in one afternoon. Reusing the harness's code instead of
  copying it turned that from a hazard into a redeploy.
- **An optional Modal secret has to be opted into at deploy time.**
  `modal.Secret.from_name` on a secret that does not exist fails the whole
  deploy, so a feature that *may* need credentials cannot just reference them
  and hope. `SUTRO_GITHUB_OAUTH=1` decides whether the secret is attached at
  all, which keeps a workspace that never created one deploying exactly as
  before.
- **Keep the secret link out of any OAuth `redirect_uri`.** GitHub logs the
  redirect it was given. A callback under `/<token>/...` would hand the link
  to a third party's logs; the callback here carries no token and the session
  it grants is worthless without the link. The `state` is HMAC-signed and
  expires, so the callback only accepts a round trip the site started.
- **Rotating the link and signing people out are different operations.**
  Sharing one key for both means every link rotation logs everyone out, and
  every logout invalidates the link. Two keys, stored separately.
- **Never put the token in code, docs or commit messages.** It lives only in
  the records Dict and is printed by `::link`; rotate it if it leaks.

## 6. Numbers observed (harness 1.1.0, 2026-09-22/23)

| Run | Mode | Result | Billable s | Charge at the current rate |
| --- | --- | --- | ---: | ---: |
| ncm_baseline, 12% band | test | pass, 79.6–79.9%, 0.7 ms | 71 | $0.06 |
| pca_qda, 5% band | leaderboard | pass, 95.62%, 4.53 ms mean over 11 draws, hold-out 77.9% | 83 | $0.07 |
| kernel returning float labels, 8% band | test | fail: `expected an integer type` | 72 | $0.06 |
| ncm_baseline after the baked-dataset rebuild, 12% band | test | pass, 79.6%, 0.7 ms | 71 | $0.06 |

Billable seconds include the 60 s start-up pad; the evaluator itself ran for
7–22 s in these runs. The current rate is $0.000694 × 1.2 = $0.000833 per
second; the first three runs were booked at the earlier 40 GB rate, so the
site's ledger shows a cent less for them.
