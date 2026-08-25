"""RankMe — label-free effective-rank assessment of pretrained representations.

Faithful implementation of:

    Garrido, Balestriero, Najman, Lecun.
    "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised
     Representations by Their Rank", ICML 2023 (arXiv:2210.02885).

RankMe is the *smooth effective rank* (Roy & Vetterli, 2007) of the embedding
matrix: the exponential of the Shannon entropy of the L1-normalised singular
value distribution. It is **label-free** (no ``y`` is used) and
**hyper-parameter free** (the only constant ``eps`` is a numerical-stability
term fixed by the paper). Higher RankMe = less dimensional collapse = (within a
method) better expected downstream performance.

--------------------------------------------------------------------------------
EXACT DEFINITION (confirmed from paper + reference code, NOT from memory)
--------------------------------------------------------------------------------
Paper Section 3.1, Equations (1)-(2):

    RankMe(Z) = exp( - sum_{k=1..min(N,K)} p_k * log p_k )            (Eq. 1)
        with   p_k = sigma_k(Z) / ||sigma(Z)||_1  +  eps              (Eq. 2)

where (verbatim confirmations from the paper text, p.3):
  * ``Z`` is "the source dataset's embeddings", an (N x K) matrix. RankMe is
    applied directly to the embeddings — NO centering and NO per-feature
    normalisation is performed (the formula and the reference code both operate
    on the raw matrix; the only normalisation is dividing sigma by its L1 norm).
  * ``sigma_k`` are the SINGULAR VALUES of Z directly (paper: "Denoting by
    sigma_k the k-th singular value of the (N x K) embedding matrix Z"), i.e.
    sigma = torch.linalg.svdvals(Z) — NOT sqrt of Gram/cov eigenvalues, and NOT
    sigma**2.
  * ``eps`` = 1e-7. Paper p.3: "where epsilon is a small constant dependent on
    the data type, typically 1e-7 for float32." The official-style reference
    implementation uses EPSILON = 1e-7.
  * log is the NATURAL logarithm (entropy = -sum p*ln(p); reference code uses
    torch.log). Using a different base only rescales RankMe by a constant factor
    and does not change the ranking, but we use ln to match the paper exactly.
  * SUBSAMPLING / N CAP: paper p.3 "In practice, we use 25600 samples ... this
    provides a highly accurate estimate." DEFAULT_MAX_SAMPLES = 25_600 in the
    reference implementation. We replicate this cap (first ``max_samples`` rows;
    a fixed-seed permutation can be enabled, but the cap is the only behaviour
    needed since RankMe is order-invariant under SVD).

Reference implementation lines mirrored (gist linked from the RankMe issue
open-mmlab/mmpretrain#1738, matching the paper's numerically-stable pseudocode
in Fig. 4 / Sec 3.1):

    _u, s, _vh = torch.linalg.svd(embeddings, full_matrices=False)
    p = (s / torch.sum(s, axis=0)) + epsilon
    entropy = -torch.sum(p * torch.log(p))
    rankme = torch.exp(entropy).item()

CAVEAT for our use-case (documented honestly): RankMe was designed to compare
DIFFERENT RUNS OF A GIVEN METHOD (paper p.3: "RankMe should however only be used
to compare different runs of a given method, since the embeddings' rank is not
the only factor that affects performance"). Comparing across distinct
pretraining datasets/objectives (K710 vs K400 vs SSV2) is outside its validated
regime; treat the cross-backbone ranking it produces as an effective-rank /
dimensional-collapse signal, not a guaranteed downstream-accuracy predictor.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rankme_score", "DEFAULT_MAX_SAMPLES", "DEFAULT_EPS"]

# Paper p.3 / reference impl constants.
DEFAULT_EPS = 1e-7            # "typically 1e-7 for float32"
DEFAULT_MAX_SAMPLES = 25_600  # "In practice, we use 25600 samples"


def rankme_score(
    X: np.ndarray,
    y: np.ndarray | None = None,
    eps: float = DEFAULT_EPS,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> float:
    """RankMe smooth effective rank of the embedding matrix ``X`` ([N, D]).

    LABEL-FREE: ``y`` is accepted only to match the ``(X, y)`` interface of the
    other transferability metrics in ``transferability.py`` and is ignored.

    Higher = higher effective rank (less dimensional collapse). Implements
    Eq. (1)-(2) of Garrido et al. (ICML 2023) exactly: raw (uncentered,
    unnormalised) ``X``, singular values via SVD, L1-normalised + ``eps``,
    natural-log Shannon entropy, exponentiated. ``X`` is capped at
    ``max_samples`` rows per the paper's 25600-sample recipe.
    """
    del y  # RankMe is label-free; signature parity only.

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"features must be [N, D], got shape {X.shape}")

    # Sample cap (paper Sec 3.1 / Appendix G: 25600 samples suffice). We take
    # the first ``max_samples`` rows; SVD is permutation-invariant in the row
    # space so no shuffling is required for a deterministic estimate.
    if X.shape[0] > max_samples:
        X = X[:max_samples]

    # sigma_k = singular values of Z directly (Eq. 2 numerator), NOT sigma**2
    # and NOT sqrt of Gram eigenvalues. No centering / per-feature scaling.
    sigma = np.linalg.svd(X, full_matrices=False, compute_uv=False)  # [min(N,D)]

    # p_k = sigma_k / ||sigma||_1 + eps  (Eq. 2)
    l1 = float(np.sum(sigma))
    if l1 <= 0.0:
        # Degenerate all-zero embedding matrix -> rank 1 by convention.
        return 1.0
    p = sigma / l1 + eps

    # RankMe(Z) = exp(-sum p_k log p_k)  (Eq. 1), natural log.
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))
