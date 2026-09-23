> **Note for this repository.** This is the evaluator's own README, copied from
> `cybertronai/sutro-problems/gpumode` at harness version 1.1.0 (2026-09-23). The
> `redteam/` directory it refers to stays in that repository and is not included
> here; everything else it lists is present. For deploying the hosted submission
> site, start at [README.md](README.md) and [DEPLOY.md](DEPLOY.md).

# MNIST-medium: train and predict, as fast as possible

![One call: 10,000 labelled 9x9 training images and 10,000 unlabelled test images in, 10,000 predicted labels out](assets/mnist-medium-task.png)

One call of your kernel receives 10,000 labelled 9x9 MNIST images and 10,000
unlabelled ones, all already on the GPU, and returns a label for each of the
10,000 queries. Train from scratch inside the call. Every call is a fresh draw,
the two halves come from disjoint halves of the pool, and the labels are
secretly permuted per draw, so nothing carries over from one call to the next.
You are ranked by time, and you qualify by accuracy: pick the band you can hit
and make the call as short as you can.

## What you submit

A single Python file exposing `custom_kernel`:

```python
#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    train_x, train_y, test_x = data
    ...
    return labels
```

| Name | Shape | Type |
| --- | --- | --- |
| `train_x` | `(10000, 1, 9, 9)` | float32 in [0, 1], CUDA |
| `train_y` | `(10000,)` | int64 in [0, 9], CUDA |
| `test_x` | `(10000, 1, 9, 9)` | float32 in [0, 1], CUDA |
| return | `(10000,)` | any integer dtype, values in [0, 9], CUDA |

The same three tensor objects arrive on every call; only their contents change.
Capturing a CUDA graph over them is allowed and is how the current leaders run.

```bash
popcorn submit --mode test       submission.py   # one draw, format and sanity
popcorn submit --mode benchmark  submission.py   # short timed rehearsal
popcorn submit --mode leaderboard submission.py  # the ranked run
```

Start from [`submission.py`](submission.py), which ships the nearest-class-mean
baseline as its body.

## How it is scored

| | |
| --- | --- |
| Ranked value | mean CUDA-event time of one complete training-and-prediction call |
| Ranked calls | 11 MNIST draws plus 2 hold-out draws, all fresh, all secret, all timed and ranked |
| Accuracy rule | `sum(correct) >= ceil(draws * 10000 * (10000 - error_bp) / 10000)`, and no single draw more than 1.5 percentage points below the band |
| Consistency | the slowest ranked call may not exceed 2x the median (plus 2 ms) |
| Warm-up | one untimed call, on an equally shaped draw from a *different* dataset |
| Per call | fresh draw staged into the fixed tensors (untimed), `synchronize`, 256 MB L2 flush, `synchronize`, `start` event, your call, `end` event, `synchronize` |
| Hold-out | 2 of the ranked calls are Fashion-MNIST, at positions only the evaluator knows; together they must be at least 70% correct |
| Submission | one file, at most 20,480 bytes, no string or bytes literal over 4,096 bytes |
| Timeouts | test 300 s, benchmark 600 s, ranked 1200 s; a single call over 60 s fails, and one that does not return is killed |
| Ranking | `ranking_by: last`, one benchmark case per band |

A draw is a fresh random split of the official 60,000-image MNIST training set
into 10,000 training and 10,000 test images, downsampled 28 -> 9 by exact
box-area averaging, divided by 255. The pool is cut in half once per run, from
the secret seed: training halves come from one half and test halves from the
other, so an image you have been shown with a label is never asked about later.
The ten class labels are permuted by a secret per-draw permutation applied to
both halves, so a label only means something relative to the training set it
arrives with.

Inside the timed window: everything `custom_kernel` does, including any host
work it triggers. Outside: generating the draw, copying it into the input
tensors, the L2 flush, reading your predictions back, and scoring them.

Four clocks bracket every call: CUDA events in the child process, the child's
`perf_counter`, the parent's `perf_counter` around the call round trip, and the
parent's `perf_counter` around the staging round trip. Before your file is
imported, the evaluator calibrates what the protocol itself costs, so the
parent's numbers bound yours from *both* sides: your device time may not exceed
the parent's measurement, and it may not fall below a quarter of it either.
Scaling every clock inside your process by a constant therefore changes
nothing, because the parent's clock does not scale with it. The evaluator also
captures `perf_counter`, `torch.cuda.Event`, `torch.cuda.synchronize` and the
tensor methods it uses to stage inputs and read outputs *before* importing your
file, so rebinding them changes nothing it measures. The child's wall clock also
covers checking your output and copying it to the host. Return a plain
`torch.Tensor`, not a subclass; work on an unsynchronized side stream, a
disabled-timing event and work deferred past the end event all fail these
checks.

### Bands

<!-- GENERATED by make_bands.py from bands.json -- do not edit by hand. -->

| Problem | Mean error at most | Correct needed | Ranked draws |
| --- | ---: | ---: | ---: |
| `mnist-medium-2pct` | 2% | 107,800 / 110,000 | 11 |
| `mnist-medium-3pct` | 3% | 106,700 / 110,000 | 11 |
| `mnist-medium-5pct` | 5% | 104,500 / 110,000 | 11 |
| `mnist-medium-8pct` | 8% | 101,200 / 110,000 | 11 |
| `mnist-medium-12pct` | 12% | 96,800 / 110,000 | 11 |

The bands are one file, [`bands.json`](bands.json); `python make_bands.py`
regenerates every `task.yml` from it.

### Current board

Measured on one A100-SXM4-40GB, harness 1.1.0, full leaderboard runs. These are
the reference entries in [`submissions/`](submissions/), not records: nobody has
competed yet.

| Band | Best entry | Mean ms | Accuracy | Hold-out | Result |
| --- | --- | ---: | ---: | ---: | --- |
| 2% | none yet | | | | `cg_pair` misses at 97.93% |
| 3% | `submissions/cg_pair.py` | 270.18 | 97.93% | 88.2% | [`results/gpu-03b-cg-pair-3pct-leaderboard.json`](results/gpu-03b-cg-pair-3pct-leaderboard.json) |
| 5% | `submissions/pca_qda.py` | 4.55 | 95.35% | 78.0% | [`results/gpu-02-pca-qda-5pct-leaderboard.json`](results/gpu-02-pca-qda-5pct-leaderboard.json) |
| 8% | `submissions/pca_qda.py` | 4.55 | 95.35% | 78.0% | same entry, easier band |
| 12% | `submissions/pca_qda.py` | 4.55 | 95.35% | 78.0% | same entry, easier band |

Nearest class mean (`submissions/ncm_baseline.py`, 1.09 ms, about 80%) meets no
band. `submissions/mlp512.py` clears 5% on accuracy (96.4%) but fails the
hold-out at 47.8%; see the residual risks below.

Run-to-run noise, from three leaderboard runs of the same entry on three secret
seeds and three containers: 4.5475 / 4.5349 / 4.5043 ms, i.e. 0.95% spread
across containers and 0.2 to 0.35% within a run. Times below a 1% difference are
a tie.

## Rules

1. **Train from scratch on every call.** No parameters, statistics, caches,
   predictions or fitted state may cross a call boundary. Compiled kernels,
   captured graphs and allocator state may. Enforced two ways: the warm-up call
   is on a different dataset, so nothing fitted before the timed loop transfers,
   and the ranked calls must all take about the same time, so a submission that
   trains once and then only predicts is visible as one slow call among fast
   ones.
2. **No external data and no memorized constants.** Following MLPerf's open
   division rule, *the implementation must not encode any information about the
   content of the dataset or a successful model's state.* Seeded random
   initialization is fine; a table of MNIST hashes or a pretrained feature
   extractor is not. Machine-checked: `submission.py` may not exceed 20,480
   bytes and no single literal may exceed 4,096 bytes, which is far less than
   the public pool compresses to. Every reference entry here is under 6 KB.
3. **No network access.** The scored container is run with egress denied, and
   inside the submission's process an audit hook refuses `socket.connect`,
   `getaddrinfo` and `urllib`, plus any attempt to open a file that looks like a
   dataset. The in-process guard is a tripwire, not a sandbox: a subprocess is a
   fresh interpreter that does not inherit it, which is why the container-level
   block is the part that counts.
4. **Read the data you are given.** A run that answers without using `train_y`,
   or that recognizes MNIST rather than learning from it, fails the hold-out
   calls -- which are ranked like any other call, so there is no free slot in
   which to be slow.
5. **Readable code.** Organizers must be able to see what the submission does.
6. **Records are reproduced before they are recognized.** The top three of each
   band are rerun by the organizers on a fresh secret seed.
7. **The evaluator is versioned** (`system.harness` in every result). If it
   changes during the competition, standing records are re-scored with the new
   version and both numbers are published.

## Run it yourself

```bash
# one A100 on your own Modal account
python run_modal.py --band mnist-medium-5pct --submission submissions/pca_qda.py \
    --mode leaderboard --seed 12345 --output results/pca-qda.json

# the same pipeline on a CPU, no GPU and no Modal account needed
python run_modal.py --band mnist-medium-5pct --submission submissions/ncm_baseline.py \
    --mode test --local --case train=2000 --case test=2000

# the evaluator directly, the way KernelBot invokes it
POPCORN_FD=9 POPCORN_SEED=12345 python eval.py benchmark cases.txt 9>results.txt
```

`--local` runs the identical protocol with `perf_counter` in place of CUDA
events and no L2 flush. Set `MNIST_POOL_CACHE` to a directory of verified
`idx.gz` files to work offline on a machine you trust. A hosted run instead
bakes the verified files into the image and sets `MNIST_POOL_CONSUME=1`: the
evaluator loads them into RAM and deletes them before any submission process
exists, so the public labels are never on disk while a submission is running.

## Known residual risks

Honest list of what this harness does *not* close, from a red-team pass in
which five agents attacked it (their submissions are in `redteam/`):

* **The test images are public.** Both halves of a draw come from the public
  60k MNIST training split, whose labels are published, and the training half
  hands back the per-draw label permutation. The size cap makes carrying the
  table impossible and the container has no egress, but an entry that
  reconstructs MNIST from a small generator, or a quantized net that fits in
  20 KB and scores ~98% on the pool, is not stopped by anything here. Closing
  this needs a competition-design change: a private, never-published pool, or a
  secret per-draw deformation of the images.
* **A short call can still be under-reported.** The parent's clock bounds the
  device clock from below only after the protocol overhead is subtracted, so
  for a call under about 2 ms the bound goes slack. Entries at the very top of
  a band should be re-run and audited by hand.
* **A 2-4x lie passes.** The timing gates are order-of-magnitude detectors, not
  timers. Together with the consistency check they make gross cheating visible;
  they do not certify a number to within a factor of two.
* **The submission process is not sandboxed.** It runs as the same user, in the
  same container, with a writable filesystem. It cannot import the harness, read
  the cases file, see the secret seed (which is kept out of the environment and
  therefore out of `/proc`), or open a file that looks like a dataset, and
  anything it spawns is killed with its process group -- but a real sandbox
  (Landlock, seccomp, a separate uid) is a hosting decision, not a harness one.
* **The hold-out floor can reject an honest learner.** Measured on an A100:
  PCA-QDA 78.0%, the conjugate-gradient pair 88.2%, and `submissions/mlp512.py`
  **47.8%**, which fails. That entry is not a memorizer: its fixed learning rate
  is tuned for sparse 9x9 digits and diverges on Fashion's denser images, at two
  different secret seeds, collapsing onto one class both times. The 70% floor
  therefore favours closed-form learners over iterative ones, and it still
  cannot separate a genuine learner from an entry that memorized MNIST *and*
  learns properly on everything else. The number is under review.
* **`A100` is two boards.** 23 of 25 validation containers were A100-SXM4-40GB
  and 2 were A100-SXM4-80GB; the same computation ran 4.5% faster on the 80 GB
  board, about 5x the cross-container noise. `system.device` is recorded in
  every result, and entries measured on different variants are not comparable.
* **KernelBot cannot rotate a leaderboard's secret seed.** `secret_seed` is a
  column defaulted when the leaderboard is created, so rule 6's rerun has to be
  done by the organizers on a seed of their own, outside the hosted service.
* **Static screening is not done here.** KernelGuard-style source review for
  replay, hardcoded shapes and trivial work is complementary to everything
  above, and is worth running on the top of each band.

## Layout

| Path | What it is |
| --- | --- |
| `eval.py` | the evaluator: draws, timing, scoring, trust boundary |
| `task.py` | input and output types, and every case field |
| `utils.py` | seeding, L2 flush, accuracy rule, timing gate, network guard |
| `mnist_data.py` | verified MNIST and Fashion-MNIST loading and downsampling |
| `reference.py`, `submission.py` | the baseline learner and the template |
| `bands.json`, `make_bands.py` | thresholds, and the generator for `task.yml` |
| `mnist-medium-*/task.yml` | one problem per band (generated) |
| `sutro.yaml` | the competition file (generated) |
| `submissions/` | ported entries from the Sutro problem set |
| `run_modal.py` | standalone runner, Modal A100 or local CPU |
| `tests/test_eval.py` | unit tests for the rules that decide a run |
| `tests/GPU_CHECKS.md` | what must be re-run on an A100, and what to expect |
| `redteam/` | attacks against this harness, and their current verdicts |

Design decisions, threat model and what this harness deliberately does not do:
[DESIGN.md](DESIGN.md).
