# Deployment runbook

Written for an agent (or a person) deploying this site with no prior context.
Every command is meant to be run from the repository root. Where a step can
fail, the failure and its fix are in [Troubleshooting](#troubleshooting).

## 0. What you are deploying

One Modal app named `sutro-mnist-submit` with three functions and three
persistent objects:

| Function | Runs on | Purpose |
| --- | --- | --- |
| `web` | CPU, one container, scales to zero | the FastAPI site behind `https://<workspace>--sutro-mnist-submit-web.modal.run/<token>` |
| `evaluate_submission` | one A100, single-use container, no egress | runs one kernel through the evaluator under a hard deadline |
| `settle_cron` | CPU, every 10 minutes | settles finished runs and keeps state alive |

| Object | Kind | Holds |
| --- | --- | --- |
| `sutro-mnist-submit-records` | `modal.Dict` | small submission records, the secret token, the budget cap |
| `sutro-mnist-submit-records-blobs` | `modal.Dict` | kernel text and full evaluator output per submission |
| `sutro-mnist-submit-ledger` | `modal.Volume` | one immutable JSON file per charge: the durable floor of the ledger |

All three objects are created on first use and survive redeploys. Deploying
into a workspace that already has the app **replaces the code and keeps the
state** (records, link, budget).

## 1. Get the code

```bash
git clone https://github.com/cybertronai/sutro-mnist-submit.git
cd sutro-mnist-submit
```

The repository is private; you need read access to the `cybertronai`
organization (or a token that has it). Deploy from `main` unless told
otherwise, and note the commit you deployed (`git rev-parse --short HEAD`).

## 2. Prerequisites

- A Modal account and workspace with GPU access and a payment method. The
  A100 runs are billed to that workspace.
- `uv` (recommended) or Python 3.12+ with `pip install modal`. All commands
  below use `uvx modal …`, which installs the CLI into a throwaway
  environment; if `modal` is on your PATH you can drop `uvx`.
- Network access to `modal.com`, PyPI, Docker Hub (`nvidia/cuda`), the MNIST
  mirror on S3 and GitHub (Fashion-MNIST) for the image build.

Log in once. On a machine with a browser:

```bash
uvx modal setup            # opens a browser, writes ~/.modal.toml
```

Headless (an agent, a CI host): the owner creates a token in the Modal
dashboard under **Settings → API Tokens** and hands over its id and secret;
then either

```bash
uvx modal token set --token-id <id> --token-secret <secret>
# or, without writing a file:
export MODAL_TOKEN_ID=<id> MODAL_TOKEN_SECRET=<secret>
```

Then confirm which workspace will pay:

```bash
uvx modal profile current  # prints the workspace name; it is also the first part of the site's URL
```

That workspace is billed for every A100 run. Deploying into a workspace
that already hosts this app (the owner's is `yaroslavvb`) **replaces the
live site in place** and keeps its records, link and budget; use a different
workspace, or change `APP_NAME` (section 4), to get a separate site.

## 3. Preflight checks (2 minutes, no cost)

```bash
uv run --python 3.12 --with modal --with "fastapi[standard]" --with pytest \
    --with pyyaml --with numpy python -m pytest tests -q
```

Expect every test to pass (41 in `tests/test_web.py`, 28 in
`tests/test_eval.py` at snapshot time). These need no Modal login and no GPU.

Optional: prove the evaluator works on this machine's CPU with the reference
kernel (downloads ~30 MB of MNIST on first run, a minute or two):

```bash
uv run --python 3.11 --with "torch==2.2.2" --with "numpy<2" --with pyyaml \
    python run_modal.py --band mnist-medium-5pct --submission submissions/pca_qda.py \
    --mode test --local --case draws=1 --case train=3000 --case test=3000
```

## 4. Configure (optional)

| Setting | Where | Default | Notes |
| --- | --- | --- | --- |
| Budget cap | `SUTRO_BUDGET_USD=75 uvx modal deploy web/app.py`, or `::set_budget --usd 75` after deploying | `50` | dollars; the cap on `charged + reserved` across all submissions. A value stored by `::set_budget` wins over the env var from then on; the env var only seeds a workspace where the cap was never set. Confirm with `::status`. |
| App name | `APP_NAME` in `web/app.py` | `sutro-mnist-submit` | change it to run a second, independent site in the same workspace; the two Dict names and the Volume name are derived from it |
| GPU rate | `A100_USD_PER_S`, `OVERHEAD` in `web/app.py` | `0.000694`, `1.2` | Modal's A100 80 GB price on 2026-09-22 with 20% headroom; check [modal.com/pricing](https://modal.com/pricing) and update if it changed |
| Queue cap | `MAX_INFLIGHT` | `8` | submissions queued at once |
| Kernel size cap | `MAX_SOURCE_BYTES` | 256 KB | the evaluator enforces its own, smaller cap (20,480 bytes) inside the run |
| Egress | `SUTRO_ALLOW_NETWORK=1` at deploy | denied | leave denied; the datasets are baked into the image |

## 5. Deploy

```bash
uvx modal deploy web/app.py
```

First deploy: 3–10 minutes. It builds the web image (Debian + FastAPI) and the
A100 image (`nvidia/cuda:13.3.0` + Python 3.13 + `torch==2.12.0` + the four
dataset files fetched at build time), then registers the three functions.
Later deploys reuse the images and take a few seconds. The output ends with:

```
├── 🔨 Created web function web => https://<workspace>--sutro-mnist-submit-web.modal.run
✓ App deployed
```

Then print the link:

```bash
uvx modal run web/app.py::link
```

This prints two URLs. **The one that ends in `/<token>` is the link to
share.** The other, ending in `-web-dev.modal.run`, belongs to the
throwaway app that `modal run` creates for a few seconds; ignore it. The
token is minted on first use and stored in the records Dict, so the link is
stable across redeploys until you run `::rotate_token`.

Visiting the bare `https://…modal.run/` (no token) returns a plain 404. That
is expected.

## 6. Smoke test (about $0.06 and two minutes)

```bash
uv run --python 3.12 python scripts/smoke.py "<link>" --mode test
```

The script is standard-library only (flags: `--kernel`, `--band`, `--mode`,
`--name`, `--timeout`, `--expect`; exit code 0 on the expected verdict). It posts `submissions/ncm_baseline.py` in test mode on the 12% band
and polls until the verdict. Expected output ends with `"status": "passed"`,
`"accuracy_pct"` near 80, `"mean_ms"` under a few ms, `"harness":
"sutro-mnist-medium-time/1.1.0"` (or newer), and `"charged_usd"` between
0.05 and 0.10. The first run after a deploy includes a cold container start
and can take 60–90 seconds longer than later ones.

To exercise the full ranked protocol (test + benchmark + 11 timed draws with
the hold-out), about $0.06 for this kernel:

```bash
uv run --python 3.12 python scripts/smoke.py "<link>" --kernel submissions/pca_qda.py \
    --band mnist-medium-5pct --mode leaderboard
```

Expected: pass, `mean_ms` around 3–5, `accuracy_pct` around 95.5,
`holdout_pct` around 78.

To see a failure rendered (still about $0.06): the baseline kernel is about
80% accurate, which passes test mode's loose gate but fails every band's
real accuracy gate in benchmark mode.

```bash
uv run --python 3.12 python scripts/smoke.py "<link>" --kernel submissions/ncm_baseline.py \
    --band mnist-medium-12pct --mode benchmark --expect fail
```

Then open the link in a browser: the budget bar should show the charges and
the table should list the runs.

## 7. Operate

All owner commands are local entrypoints of `web/app.py`; each spins up a
throwaway Modal app for a few seconds to reach the deployed state.

| Command | Effect |
| --- | --- |
| `uvx modal run web/app.py::status` | ledger totals and one line per submission (read-only) |
| `uvx modal run web/app.py::set_budget --usd 75` | change the cap without redeploying |
| `uvx modal run web/app.py::cancel --sid <id>` | cancel a queued or running submission; it is charged its reservation |
| `uvx modal run web/app.py::rotate_token` | invalidate the old link and print a new one |
| `uvx modal run web/app.py::export --output subs.json` | dump every record with kernel text and evaluator output |

Where to look on modal.com: **Apps → sutro-mnist-submit** for logs of all
three functions (the web function logs every request with its status code;
the worker logs the evaluator's stderr), **Storage** for the two Dicts and
the Volume, **Settings → Billing** for the real spend. The ledger is
pessimistic by design, so the billing page should read lower than the site.

What "done" looks like for a submission: status `passed` or `failed` with a
charge, or `error` with the reservation charged and an explanation. Once a
run starts it finishes within its deadline (8 min for `test`, 13 min for
`benchmark`, 38 min for `leaderboard`), and the cron settles it within 10
minutes even if nobody loads the page. Time spent *queued* is not bounded:
runs execute one at a time and up to 8 can wait in line, so the last of
eight leaderboard runs can wait several hours. `::status` shows the queue;
`::cancel` removes a run from it.

## 8. Update

- Changed `web/app.py`, `web/runner.py` or anything in the harness: run
  `uvx modal deploy web/app.py` again. State is untouched.
- Changed a band's timeouts in `bands.json`: run
  `uv run --python 3.12 python make_bands.py` first (it regenerates every `task.yml`), then deploy. Reservations follow
  the new timeouts automatically.
- Changed `run_modal.py`'s image (CUDA tag, torch pin, baked files): the
  next deploy rebuilds the A100 image (minutes).
- Pulled a newer harness from `cybertronai/sutro-problems/gpumode`: copy the
  harness files over, run the tests, deploy, and rerun the smoke test.
  Timings are only comparable within one harness version (the `harness`
  field in every result).

## 9. Tear down or reset

```bash
uvx modal run web/app.py::export --output before-stop.json   # keep the records and kernels
uvx modal run web/app.py::link                                # keep the token
uvx modal app stop -y sutro-mnist-submit                      # stops the site, the worker and the cron
```

Stopping the app also stops the cron that keeps the Dicts alive: Modal
expires a Dict entry after 7 days without a read or write, so a site left
stopped for more than a week comes back with no records and a **new token**
(the old link stops working). The Volume does not expire, so the spend
history survives and the cap still counts it. Export before a long stop.

To start over with an empty ledger, also delete the two Dicts and the Volume
(the commands prompt unless `-y` is given):

```bash
uvx modal dict delete -y sutro-mnist-submit-records
uvx modal dict delete -y sutro-mnist-submit-records-blobs
uvx modal volume delete -y sutro-mnist-submit-ledger
```

Deleting them **forgets all spend**; the cap starts again from zero.

## 10. Done checklist

- [ ] `uvx modal profile current` prints the intended workspace
- [ ] unit tests pass
- [ ] `uvx modal deploy web/app.py` ends in `App deployed`
- [ ] `::link` prints a `…modal.run/<token>` URL and it renders the form
- [ ] `scripts/smoke.py … --mode test` ends in `passed`
- [ ] `::status` shows the charge, the remaining budget, and the cap you intended
- [ ] the link is shared only with the intended people

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `modal: command not found` | the CLI is not installed globally | use `uvx modal …` or `pip install modal` |
| `import modal` fails under `python3`, or `TypeError: 'type' object is not subscriptable` from a script | the default `python3` may be old (it was 3.7 on the original machine) | always pin: `uv run --python 3.12 [--with modal] python …` |
| `::link` prints a `-dev.modal.run` URL as well | `modal run` starts a temporary app | use the URL that ends in `/<token>` |
| `deploy first: …` from `::link` | the app is not deployed in this workspace | run `uvx modal deploy web/app.py` |
| Image build fails resolving `torch==2.12.0` | the pin drifted from what PyPI offers for CUDA 13 | change `TORCH_PIN` in `run_modal.py`; that is the single place |
| Image build fails downloading a dataset file | S3 or GitHub unreachable from the builder | retry; the URLs and MD5s are in `run_modal.py` and `mnist_data.py` |
| A submission sits in `queued` | the A100 pool is at capacity, or the previous run is still going (one at a time) | wait; the deadline caps every run; `::cancel` if needed |
| A page returns 500 once, then works | Modal preempted or restarted the web container mid-request | nothing to do; there is no application traceback in that case |
| A run ends `failed` with "the whole evaluation exceeded its N s deadline" | the kernel exceeded the mode's time budget and the worker killed it | measured container time is charged (it may exceed the reservation; the record shows the overrun); the kernel is at fault |
| A run ends `error` with "hit Modal's 2400 s container limit" | the worker's own deadline failed to stop the run | the whole container life (about $2.05) is charged; report it, this should not happen |
| Charge looks high for a short run | every run pays a 60 s start-up pad and is priced at the A100 80 GB rate ×1.2 | expected; see LESSONS.md |
| `tests/test_eval.py` fails after updating the harness | the evaluator and its tests were edited together upstream | pull both; those tests are the harness's, not the site's |
| Submission rejected with "request is larger than" | the multipart body exceeded 320 KB | the kernel must be under 256 KB (the evaluator only accepts 20 KB anyway) |
| Submission rejected with "budget: …" | the cap would be exceeded by this run's reservation | raise the cap with `::set_budget`, or wait for in-flight runs to settle |
| Old submissions show "(source no longer available)" | records predate the blob layout or the blobs Dict was deleted | harmless; the cron migrates old records on its next tick |
