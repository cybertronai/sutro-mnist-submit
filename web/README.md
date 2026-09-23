# Submission website

A one-link, no-login web page where anyone can paste a `submission.py` for the
MNIST-medium time challenge and have it evaluated on an A100 in the owner's
Modal account. It runs the same [`eval.py`](../eval.py) that the KernelBot
leaderboard runs, by way of [`run_modal.py`](../run_modal.py)'s `evaluate()`,
and reports the ranked time, the accuracy and what the run cost.

[`app.py`](app.py) holds the FastAPI site, the A100 worker, the ledger and the
owner commands; [`runner.py`](runner.py) is the stdlib-only script the worker
runs under a deadline.

## Deploy

Run these from the harness directory: the repository root of
`sutro-mnist-submit`, or `gpumode/` inside `cybertronai/sutro-problems`.

```bash
uvx modal deploy web/app.py      # builds both images, deploys web + worker + cron
uvx modal run web/app.py::link   # prints the secret link to share
```

The first deploy builds the CUDA image (a few minutes). Redeploying after an
edit to `app.py` or the harness is the same command; the link does not change.

Owner commands:

| Command | Effect |
| --- | --- |
| `uvx modal run web/app.py::status` | ledger totals and one line per submission |
| `uvx modal run web/app.py::set_budget --usd 75` | change the cap without redeploying |
| `uvx modal run web/app.py::cancel --sid <id>` | kill a queued or running submission (charged its reservation) |
| `uvx modal run web/app.py::rotate_token` | invalidate the old link, print a new one |
| `uvx modal run web/app.py::export --output subs.json` | dump every record, kernels and evaluator output included |

State lives in three Modal objects that survive redeploys: the Dict
`sutro-mnist-submit-records` (small records, the token, the budget), the Dict
`sutro-mnist-submit-records-blobs` (kernel text and full evaluator output) and
the Volume `sutro-mnist-submit-ledger` (one JSON file per charge). Delete all
three from the Modal dashboard to start over.

## What a submitter sees

- A form: name, band (`mnist-medium-2pct` … `12pct`), mode (`test`,
  `benchmark`, `leaderboard`), and the kernel as a textarea prefilled with
  [`submission.py`](../submission.py) or as an uploaded file.
- After submitting, a page that refreshes until the run finishes, then shows
  the verdict, mean/std/best time per call, accuracy with the required count,
  correct-per-draw, the Fashion-MNIST hold-out result, the raw `key: value`
  output of every evaluator step and its stderr, plus the cost. Benchmark-mode
  times are labelled as the 3-draw dress rehearsal, not the ranked number.
- A table of all submissions with status, mean ms, accuracy and cost, and a
  budget bar for the shared cap.

JSON is at `<link>/api/submissions` and `<link>/api/s/<id>`; the kernel text at
`<link>/s/<id>/source`.

## Budget enforcement

The cap (default $50, `SUTRO_BUDGET_USD` at deploy time or `set_budget` later)
is enforced before any GPU is touched:

1. Every submission **reserves its worst case** when it is queued: the
   evaluator timeouts for its mode (test 300 s; benchmark 600 s; leaderboard
   300 + 600 + 1200 s) plus 180 s of slack for the runner's own start-up and
   the baked-dataset lift (the evaluator's dataset load is already inside
   its per-step timeouts); that sum is also the **hard deadline** the worker
   enforces on the whole evaluation. Add 60 s for container start-up and
   teardown. That container lifetime is priced at
   the A100 80 GB rate ($0.000694/s on 2026-09-22; Modal fulfils `gpu="A100"`
   with either the 40 GB or the 80 GB part, so the dearer one is assumed)
   times a 1.2 overhead for CPU, memory and rounding: about $0.45 for
   `test`, $0.70 for `benchmark` and $1.95 for `leaderboard`.
2. The record and its reservation are **persisted before the GPU job is
   spawned**. A submission is **rejected** if `charged + reserved + its
   reservation` would exceed the cap, or if 8 runs are already in flight. A
   spawn that fails releases the reservation; a spawn whose call id is lost is
   written off at the reservation after 3 minutes.
3. The worker runs `runner.py` in its own process group and **kills the
   whole group** when the mode's deadline passes, so no kernel can keep the
   container alive past what was reserved. `run_one` in the harness does the
   same per evaluator step and drains the result pipe from a thread, so an
   orphaned submission process cannot hold it past its timeout. Each run gets
   a **fresh container** (`single_use_containers=True`), so nothing a kernel
   leaves behind can touch the next submitter's timings.
4. When a run finishes, it is **charged its measured container time,
   uncapped** (the worker's own clock plus its boot time plus the 60 s pad).
   If that ever exceeds the reservation the record says so and the ledger
   counts the full amount. A worker that crashes or is cancelled is charged
   its reservation; one that hits Modal's 2400 s backstop is charged the
   whole container life (2400 s plus the pad). A result that cannot
   be parsed becomes an `error` record charged its reservation, never a broken
   page. Transport errors while polling are retried five times before a run is
   written off and cancelled.
5. Every charge is also written as an immutable file on the ledger Volume;
   the ledger uses that sum as a floor, so spend is remembered even if the
   Dict records are lost. Settlement happens whenever a page is loaded and
   every 10 minutes from a scheduled function, which also rewrites every Dict
   key so Modal's 7-day idle expiry never fires while the app is deployed.

The A100 worker has `max_containers=1`, so runs execute one at a time and
never burn more than one GPU. Modal may re-run an input once on an
infrastructure failure; that second container's time is what gets charged.
The ledger is intentionally pessimistic; check Modal's billing page for the
true number. The tiny CPU cost of the web container and the cron function is
not counted.

## Security model

Security through obscurity, by design: the random URL token is the only
gate. Beyond it:

- The worker runs with `restrict_modal_access=True`, so a kernel cannot use
  the container's Modal credentials to spawn GPU work outside the ledger, and
  each container is single-use.
- The worker container has no network egress (`block_network=True`, from
  `run_modal.BLOCK_NETWORK`). The MNIST and Fashion-MNIST files are baked
  into `run_modal.image` at build time; `web/runner.py` lifts them into RAM
  and deletes them from disk before any kernel process exists.
- `eval.py` imports the kernel in a separate process behind an audit hook,
  in a private directory, and never exposes the test labels, the draw seeds
  or the label permutation to it. It removes `POPCORN_FD` from the
  environment and marks the result pipe non-inheritable before the kernel
  starts; the kernel still runs as the same user in the same container, so
  the result channel is not beyond reach of a determined kernel. The site
  parses every result line defensively, but a forged verdict is a
  harness-level concern the site does not detect.
- Kernels are stored with the record and shown on the site; do not submit
  anything private.
- Requests over 320 KB are refused before parsing; no other rate limiting
  beyond the queue cap and the budget.

Adding GitHub login later means wrapping the `check()` dependency in
`build_api()`.

## Tests

```bash
uv run --python 3.12 --with modal --with "fastapi[standard]" --with pytest --with pyyaml \
    python -m pytest tests/test_web.py -q
```

The tests exercise the ledger, the validation, the worker deadline (with real
subprocesses), the bounded `run_one`, and the HTTP routes with in-memory
stores and fake Modal calls; no account or GPU is needed.
