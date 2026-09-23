# Ported Sutro entries

Each file is self-contained, exposes `custom_kernel`, takes its device from the
input tensors, and runs on a CPU as well as a GPU. They are reference points and
smoke tests for the harness, not an endorsement of any particular approach.

| File | Source | Band it meets | Accuracy observed on CPU | A100 time reported upstream |
| --- | --- | --- | ---: | ---: |
| `ncm_baseline.py` | `reference.py` | none | 79.0% - 79.5% | under 1 ms |
| `pca_qda.py` | `mnist/submissions/medium-pca-qda-20260915` | 5% | 95.65% over 3 draws | ~3.3 ms |
| `mlp512.py` | `mnist/submissions/medium-affine-20260911` | 5% | 96.71% on one full draw | not measured |
| `cg_pair.py` | `mnist/submissions/medium-cg-pair-20260916` | 2% | 98.20% on one full draw | ~260 ms |

CPU numbers come from `run_modal.py --local` on this harness with the 10,000 /
10,000 draw, seed 20260922; they are accuracy checks, not timings. A CPU call
takes about 0.6 ms (`ncm_baseline`), 70 ms (`pca_qda`), 23 s (`mlp512`) and
10 s (`cg_pair`) on an Intel MacBook Pro.

`mlp512.py` is a plain-PyTorch transcription of the upstream architecture,
hyperparameters and update rule rather than a copy of its ordered-FP32 NumPy
arithmetic; see its docstring. The other two carry their upstream learners
essentially unchanged.
