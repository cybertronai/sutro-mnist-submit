"""Baseline learner: nearest class mean. About 80% on MNIST-medium, so it
misses every band, including 12%.

KernelBot problems usually put ``generate_input`` and ``check_implementation``
here and let ``eval.py`` call them inside the submission's process. This problem
cannot: the draw contains the test labels, which decide the score, so drawing
and scoring live in the parent evaluator (``eval.py``) and never enter the
process that imports the submission. This file keeps the familiar name and
exposes the baseline learner, which ``submission.py`` ships as its template
body and which doubles as the harness's own smoke test.
"""

import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    train_x, train_y, test_x = data
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    distances = (means * means).sum(1) - 2 * q @ means.T
    return distances.argmin(1)
