#!POPCORN leaderboard mnist-medium-5pct
#!POPCORN gpu A100

"""512-unit MLP, ported from mnist/submissions/medium-affine-20260911.

That submission is a NumPy learner whose arithmetic is pinned to a spatial-grid
IR (ordered FP32 reductions, einsum with optimize=False). This file is a plain
PyTorch transcription of the same architecture, the same update rule and the
same constants, not a bit-for-bit copy of its reduction order; the original is
documented in that directory's README.md and config.json:

  input transform   x * 4 - 0.5 on the flattened 81 pixels
  architecture      81 -> 512 ReLU -> 10, biases start at zero
  initialization    numpy PCG64(101), uniform(-1/sqrt(81), 1/sqrt(81)) for the
                    first layer and uniform(-1/sqrt(512), 1/sqrt(512)) for the
                    second
  loss              squared error against one-hot targets, gradient scaled by
                    the learning rate divided by the minibatch size
  optimizer         plain SGD, learning rate 0.1
  schedule          200 epochs of contiguous minibatches of 30, in data order

The original reported 96% on the 6,000/6,000 medium tier. The upstream learner
asserts that the training count divides 30; with 10,000 examples the last
minibatch of each epoch holds 10, which changes nothing else.

The initial parameters are a seeded constant. They are cloned at the start of
every call, so each call trains from scratch.
"""

import torch

WIDTH = 512
EPOCHS = 200
LEARNING_RATE = 0.1
BATCH = 30
SEED = 101
FEATURES = 81

_initial = {}


def initial_parameters(device):
    """numpy PCG64(101) uniform initialization, materialized once per device."""
    key = str(device)
    if key not in _initial:
        import math

        import numpy as np

        rng = np.random.Generator(np.random.PCG64(SEED))
        w1 = rng.uniform(
            -1 / math.sqrt(FEATURES), 1 / math.sqrt(FEATURES), (FEATURES, WIDTH)
        ).astype("float32")
        w2 = rng.uniform(-1 / math.sqrt(WIDTH), 1 / math.sqrt(WIDTH), (WIDTH, 10)).astype(
            "float32"
        )
        _initial[key] = (
            torch.as_tensor(w1, device=device),
            torch.zeros(WIDTH, device=device),
            torch.as_tensor(w2, device=device),
            torch.zeros(10, device=device),
        )
    return _initial[key]


@torch.no_grad()
def custom_kernel(data):
    train_x, train_y, test_x = data
    device = train_x.device
    x = train_x.reshape(train_x.shape[0], FEATURES) * 4.0 - 0.5
    q = test_x.reshape(test_x.shape[0], FEATURES) * 4.0 - 0.5
    target = torch.nn.functional.one_hot(train_y, 10).to(torch.float32)

    w1, b1, w2, b2 = (tensor.clone() for tensor in initial_parameters(device))

    count = x.shape[0]
    for _ in range(EPOCHS):
        for start in range(0, count, BATCH):
            xb = x[start : start + BATCH]
            tb = target[start : start + BATCH]
            z = xb @ w1 + b1
            h = torch.relu(z)
            scores = h @ w2 + b2
            d2 = scores - tb
            d1 = torch.where(z > 0, d2 @ w2.T, torch.zeros((), device=device))
            step = LEARNING_RATE / xb.shape[0]
            w2 -= step * (h.T @ d2)
            b2 -= step * d2.sum(0)
            w1 -= step * (xb.T @ d1)
            b1 -= step * d1.sum(0)

    return (torch.relu(q @ w1 + b1) @ w2 + b2).argmax(1)
