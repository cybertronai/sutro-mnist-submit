#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""Template. Replace the body of custom_kernel with your learner.

make_bands.py copies this file into every band folder with the matching
#!POPCORN leaderboard line, and those copies are what KernelBot serves.

Interface
    custom_kernel((train_x, train_y, test_x)) -> predicted labels

    train_x  (N, 1, 9, 9) float32 in [0, 1], on the GPU
    train_y  (N,)         int64 in [0, 9],   on the GPU
    test_x   (Q, 1, 9, 9) float32 in [0, 1], on the GPU
    return   (Q,)         integer labels in [0, 9], on the GPU

Rules that decide whether your entry counts
  * Train from scratch on every call. No model state may carry over between
    calls; the class labels are secretly permuted on every draw, so anything
    carried over is worse than useless.
  * The same three tensor objects arrive on every call, only their contents
    change. Capturing a CUDA graph over them is allowed and encouraged: the
    first call is a warm-up and is not timed.
  * No network access, no external data, and no constants that encode anything
    about MNIST's contents or a trained model's state. Seeded random
    initialization is fine.

The body below is the nearest-class-mean baseline. It runs in well under a
millisecond and is about 80% accurate, so it fails every band; it is here to
show the shapes and the calling convention.

Submit with:
    popcorn submit --mode test submission.py
    popcorn submit --mode benchmark submission.py
    popcorn submit --mode leaderboard submission.py
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
