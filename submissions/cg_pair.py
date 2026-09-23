#!POPCORN leaderboard mnist-medium-2pct
#!POPCORN gpu A100

"""512-filter CG pair, ported from mnist/submissions/medium-cg-pair-20260916 (@jurajselep).

Two ridge systems fitted by 300 Jacobi-preconditioned conjugate-gradient
iterations and fused by z-scores:

  1. random convolutional features - 512 frozen 3x3 filters from PCG64 seed 0,
     ReLU, 3x3 mean pooling, mean-centred, ridge lambda 1e-3 x mean diagonal
  2. an RBF kernel on the arcsine-transformed pixels, gamma 0.3, lambda 1e-2

Targets are one-versus-rest +1/-1. Reported at 98.12% on MNIST-medium (the 2%
band) and about 260 ms per call on an A100. Every constant below comes from the
submission's config.json; the filters are a seeded random initialization, not a
trained model.

The port is a literal transcription of that submission's learner.py, with the
device taken from the input tensors and the config inlined so this file stands
alone. It is slow on a CPU (minutes per call) but runs there for dry runs.
"""

import torch
import torch.nn.functional as F

FILTERS = 512
FILTER_SEED = 0
CONV_BATCH_SIZE = 128
RIDGE_LAMBDA = 1e-3
RBF_GAMMA = 0.3
RBF_LAMBDA = 1e-2
ITERATIONS = 300

torch.backends.cuda.matmul.allow_tf32 = False

_filters = {}


def frozen_filters(device):
    """numpy PCG64(0): standard_normal((512, 9)) / 3, then standard_normal(512) * 0.1."""
    key = str(device)
    if key not in _filters:
        import numpy as np

        generator = np.random.Generator(np.random.PCG64(FILTER_SEED))
        weights = (generator.standard_normal((FILTERS, 9)) / 3).astype("float32")
        biases = (generator.standard_normal(FILTERS) * 0.1).astype("float32")
        _filters[key] = (
            torch.as_tensor(weights, device=device).view(FILTERS, 1, 3, 3),
            torch.as_tensor(biases, device=device),
        )
    return _filters[key]


def cg(matrix, rhs):
    """The source recurrence, fixed iteration count, no convergence guards."""
    inverse_diagonal = 1.0 / matrix.diagonal()
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual * inverse_diagonal[:, None]
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum(0)
    for _ in range(ITERATIONS):
        product = matrix @ direction
        alpha = rz / (direction * product).sum(0)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        preconditioned = residual * inverse_diagonal[:, None]
        next_rz = (residual * preconditioned).sum(0)
        direction = preconditioned + (next_rz / rz) * direction
        rz = next_rz
    return solution


def features(pixels, weights, biases):
    chunks = [
        F.avg_pool2d(
            F.relu(F.conv2d(chunk.reshape(-1, 1, 9, 9), weights, biases, padding=1)), 3
        ).flatten(1)
        for chunk in pixels.split(CONV_BATCH_SIZE)
    ]
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def zscore(scores):
    """PyTorch's default sample standard deviation, as in the source."""
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True)


@torch.no_grad()
def custom_kernel(data):
    train_x, train_y, test_x = data
    device = train_x.device
    weights, biases = frozen_filters(device)

    u = torch.asin(torch.sqrt(train_x.reshape(train_x.shape[0], 81).clamp(0, 1)))
    uq = torch.asin(torch.sqrt(test_x.reshape(test_x.shape[0], 81).clamp(0, 1)))

    targets = torch.full((train_y.shape[0], 10), -1.0, device=device, dtype=torch.float32)
    targets[torch.arange(train_y.shape[0], device=device), train_y] = 1.0

    p, pq = features(u, weights, biases), features(uq, weights, biases)
    mean = p.mean(0)
    p = p - mean
    pq = pq - mean

    gram = p.T @ p
    gram.diagonal().add_(RIDGE_LAMBDA * gram.diagonal().mean())
    rhs = p.T @ targets

    kernel = torch.exp(-RBF_GAMMA * torch.cdist(u, u) ** 2)
    kernel.diagonal().add_(RBF_LAMBDA)
    query_kernel = torch.exp(-RBF_GAMMA * torch.cdist(uq, u) ** 2)

    ridge_scores = pq @ cg(gram, rhs)
    rbf_scores = query_kernel @ cg(kernel, targets)
    return (zscore(ridge_scores) + zscore(rbf_scores)).argmax(1)
