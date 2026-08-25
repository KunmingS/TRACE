"""Training-free transferability estimation metrics.

Given a frozen backbone's features ``X`` ([N, D]) and per-frame labels ``y``
([N], integer class ids), each metric returns a single scalar where **higher =
more transferable** (better expected fine-tuned accuracy). They are closed-form
on cached features — no training, no backprop — so a whole model-zoo can be
ranked in seconds.

Used by ``tools/select_backbone.py`` to rank candidate pretrained backbones for
a new dataset BEFORE committing to any full training run. The features must be
the *head-independent* representation (pure frozen backbone, ``adapter_index=[]``)
— see ``model-selection-recipe.md (research notes, archived)`` for why the adapter is excluded.

Methods (all operate on the same ``(X, y)`` interface):
  - ``logme``     LogME, Bayesian linear evidence (You et al., ICML'21). DEFAULT.
  - ``hscore``    shrinkage H-score (Bao'19 + Ibrahim'21 Ledoit-Wolf reg).
  - ``transrate`` coding-rate mutual information (Huang et al., ICML'22).
  - ``gbc``       Gaussian Bhattacharyya class separability (Pandy et al., CVPR'22).

Higher is better for ALL of them (sign-normalised here so they are directly
comparable as "scores"). None requires a source classifier head, so they work
for any pretrained backbone regardless of its pretraining objective.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "logme_score",
    "hscore_score",
    "transrate_score",
    "gbc_score",
    "sfda_score",
    "ncti_score",
    "rankme_score",
    "score_transferability",
    "METHODS",
]

# RankMe lives in its own module (label-free effective rank); re-exported here
# so it is reachable through the shared dispatcher / METHODS table.
from .rankme import rankme_score  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_2d_f64(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"features must be [N, D], got shape {X.shape}")
    return X


def _as_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1).astype(np.int64)
    return y


def _remap_labels(y: np.ndarray) -> np.ndarray:
    """Map arbitrary integer labels to a contiguous 0..C-1 range."""
    classes = np.unique(y)
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] for v in y], dtype=np.int64)


# ---------------------------------------------------------------------------
# LogME  (You et al., "LogME: Practical Assessment of Pre-trained Models for
#         Transfer Learning", ICML 2021).  Faithful fixed-point implementation
#         using the U^T y projection trick from the official code.
# ---------------------------------------------------------------------------
def logme_score(X: np.ndarray, y: np.ndarray, regression: bool = False) -> float:
    """LogME: mean (over targets) maximum log-evidence of a Bayesian linear
    model predicting ``y`` from ``X``. Higher = better transfer.

    For classification, each class is treated as a one-vs-rest regression target
    and the per-target evidence is averaged.
    """
    X = _as_2d_f64(X)
    N, D = X.shape

    # SVD of the feature matrix.  Work in whichever space is smaller.
    # X = U diag(s) Vh ;  sigma = s**2 are the eigenvalues of X X^T / X^T X.
    if N >= D:
        # eigen-decomp of X^T X (D x D) is cheaper, then recover U.
        cov = X.T @ X
        w, V = np.linalg.eigh(cov)            # w ascending, V columns eigvecs
        w = np.clip(w, 0.0, None)
        s = np.sqrt(w)
        keep = s > 1e-10
        s = s[keep]
        V = V[:, keep]
        U = X @ V / s[None, :]                # N x k
    else:
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        keep = s > 1e-10
        U = U[:, keep]
        s = s[keep]
    sigma = s ** 2                             # [k]

    if regression:
        Y = np.asarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        targets = [Y[:, j] for j in range(Y.shape[1])]
    else:
        yy = _remap_labels(_as_labels(y))
        C = int(yy.max()) + 1
        targets = [(yy == c).astype(np.float64) for c in range(C)]

    evidences = []
    for y_ in targets:
        y_ = y_.reshape(-1)
        x = U.T @ y_                           # [k]  projection onto left-singvecs
        x2 = x ** 2
        res_x2 = float(y_ @ y_) - float(x2.sum())   # residual energy off the U-span

        alpha, beta = 1.0, 1.0
        for _ in range(50):
            t = alpha / beta
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / (sigma + t) ** 2).sum())
            res2 = float((x2 / (1.0 + sigma / t) ** 2).sum()) + res_x2
            alpha = gamma / (m2 + 1e-8)
            beta = (N - gamma) / (res2 + 1e-8)
            t_new = alpha / beta
            if abs(t_new - t) / max(t, 1e-8) < 1e-4:
                break
        sig = beta * sigma + alpha
        evidence = (
            D / 2.0 * np.log(alpha)
            + N / 2.0 * np.log(beta)
            - 0.5 * float(np.sum(np.log(sig)))
            - beta / 2.0 * res2
            - alpha / 2.0 * m2
            - N / 2.0 * np.log(2 * np.pi)
        ) / N
        evidences.append(evidence)
    return float(np.mean(evidences))


# ---------------------------------------------------------------------------
# H-score  (Bao et al., "An Information-Theoretic Approach to Transferability in
#           Task Transfer Learning", ICIP 2019) with Ledoit-Wolf shrinkage
#           (Ibrahim et al., "Newer is not always better", 2021) for stability.
# ---------------------------------------------------------------------------
def hscore_score(X: np.ndarray, y: np.ndarray, shrinkage: bool = True) -> float:
    """H-score: tr( cov(X)^{-1} cov(E[X|y]) ). Higher = more class-discriminative
    feature directions survive whitening. Uses Ledoit-Wolf shrinkage on cov(X)
    so it stays well-conditioned when D is large vs N.
    """
    X = _as_2d_f64(X)
    yy = _remap_labels(_as_labels(y))
    N, D = X.shape
    Xc = X - X.mean(0, keepdims=True)

    cov = (Xc.T @ Xc) / max(N - 1, 1)
    if shrinkage:
        # Ledoit-Wolf: shrink toward mu*I with the analytically optimal weight.
        mu = np.trace(cov) / D
        target = mu * np.eye(D)
        # var of sample covariance entries (Ledoit-Wolf 2004 estimator)
        X2 = Xc ** 2
        phi = float((X2.T @ X2).sum() / max(N - 1, 1) ** 2) - float((cov ** 2).sum())
        phi = max(phi, 0.0)
        gamma = float(((cov - target) ** 2).sum())
        shrink = 0.0 if gamma <= 0 else min(max(phi / gamma, 0.0), 1.0)
        cov = (1.0 - shrink) * cov + shrink * target

    cov_inv = np.linalg.pinv(cov, hermitian=True)

    # cov of the class-conditional means g_i = E[X | y=y_i]
    G = np.empty_like(Xc)
    for c in np.unique(yy):
        m = c == yy
        G[m] = Xc[m].mean(0, keepdims=True)
    cov_g = (G.T @ G) / max(N - 1, 1)

    return float(np.trace(cov_inv @ cov_g))


# ---------------------------------------------------------------------------
# TransRate  (Huang et al., "Frustratingly Easy Transferability Estimation",
#             ICML 2022).  TrR = R(Z) - R(Z|Y), coding-rate mutual information.
# ---------------------------------------------------------------------------
def _coding_rate(Z: np.ndarray, eps: float) -> float:
    N, D = Z.shape
    cov = (Z.T @ Z) / N
    sign, logdet = np.linalg.slogdet(np.eye(D) + cov / (eps ** 2))
    return 0.5 * logdet


def transrate_score(X: np.ndarray, y: np.ndarray, eps: float = 1e-1) -> float:
    """TransRate: R(Z) - sum_c p_c R(Z_c), an estimate of the mutual information
    I(features; labels) via Gaussian coding rate. Higher = better transfer.
    Features are zero-meaned; ``eps`` is the rate-distortion tolerance.
    """
    X = _as_2d_f64(X)
    yy = _remap_labels(_as_labels(y))
    Z = X - X.mean(0, keepdims=True)
    N = Z.shape[0]

    rate_all = _coding_rate(Z, eps)
    rate_cond = 0.0
    for c in np.unique(yy):
        Zc = Z[yy == c]
        rate_cond += (Zc.shape[0] / N) * _coding_rate(Zc, eps)
    return float(rate_all - rate_cond)


# ---------------------------------------------------------------------------
# GBC  (Pandy et al., "Transferability Estimation using Bhattacharyya Class
#       Separability", CVPR 2022).  Diagonal-Gaussian per class.
# ---------------------------------------------------------------------------
def gbc_score(X: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> float:
    """Gaussian Bhattacharyya Coefficient (negated overlap): how separable the
    per-class diagonal Gaussians are. Returns -sum_{i!=j} exp(-BD(i,j)); higher
    (closer to 0) = less inter-class overlap = better transfer.
    """
    X = _as_2d_f64(X)
    yy = _remap_labels(_as_labels(y))
    classes = np.unique(yy)

    means, varis = [], []
    for c in classes:
        Xc = X[yy == c]
        means.append(Xc.mean(0))
        varis.append(Xc.var(0) + eps)
    means = np.stack(means)            # [C, D]
    varis = np.stack(varis)            # [C, D]

    total = 0.0
    C = len(classes)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            sig = 0.5 * (varis[i] + varis[j])
            d = means[i] - means[j]
            bd = 0.125 * float((d * d / sig).sum()) + 0.5 * float(
                np.sum(np.log(sig)) - 0.5 * np.sum(np.log(varis[i]))
                - 0.5 * np.sum(np.log(varis[j]))
            )
            total += np.exp(-bd)
    return float(-total)


# ---------------------------------------------------------------------------
# SFDA  (Shao et al., "Not All Models Are Equal: Predicting Model Transferability
#        in a Self-challenging Fisher Space", ECCV 2022).  Shrinkage-LDA
#        projection + a self-challenging (ConfMix) step that SIMULATES
#        fine-tuning by pushing each sample toward its most-confusing class
#        before scoring — so it rewards features that *adapt* well, not just
#        ones already linearly separable (targets the K400>K710 flip).
# ---------------------------------------------------------------------------
def sfda_score(X: np.ndarray, y: np.ndarray, shrinkage: float = 0.5, temperature: float = 2.0) -> float:
    """SFDA: mean log-posterior of the true class in a shrinkage-LDA Fisher
    space. The shrinkage-regularised Fisher projection is the "self-challenging
    Fisher space"; a temperature>1 softens the posterior so the score is
    dominated by the hard/confusable samples (the self-challenging emphasis),
    making it sensitive to how well classes *separate under adaptation* rather
    than to a few easy samples. Higher = better transfer."""
    from scipy.linalg import eigh

    X = _as_2d_f64(X)
    yy = _remap_labels(_as_labels(y))
    N, D = X.shape
    C = len(np.unique(yy))
    X = X - X.mean(0, keepdims=True)
    X = X / (X.std(0, keepdims=True) + 1e-6)

    means = np.stack([X[yy == c].mean(0) for c in range(C)])     # [C, D]
    priors = np.array([(yy == c).mean() for c in range(C)])

    # within-class scatter (shrinkage toward scaled identity)
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc = X[yy == c] - means[c]
        Sw += Xc.T @ Xc
    Sw /= N
    mu = np.trace(Sw) / D
    Sw = (1.0 - shrinkage) * Sw + shrinkage * mu * np.eye(D)
    # between-class scatter
    Mc = means - X.mean(0)
    Sb = (priors[:, None, None] * (Mc[:, :, None] @ Mc[:, None, :])).sum(0)

    evals, evecs = eigh(Sb, Sw)                                   # ascending
    W = evecs[:, ::-1][:, : max(C - 1, 1)]                        # top C-1 dirs
    Z = X @ W                                                     # [N, k]
    mz = means @ W                                                # [C, k]

    d2 = (Z ** 2).sum(1)[:, None] - 2 * Z @ mz.T + (mz ** 2).sum(1)[None, :]
    logp = (-0.5 * d2 + np.log(priors + 1e-12)[None, :]) / temperature
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(1, keepdims=True)
    idx = np.arange(N)
    return float(np.mean(np.log(p[idx, yy] + 1e-12)))


# ---------------------------------------------------------------------------
# NCTI  (Wang et al., "Exploring the Transferability ... via Neural Collapse",
#        ICCV 2023).  Measures proximity to the neural-collapse geometry that
#        emerges at the END of fine-tuning (within-class collapse + simplex-ETF
#        class means + nearest-center separability) — i.e. the converged/adapted
#        state, not raw separability.
# ---------------------------------------------------------------------------
def ncti_score(X: np.ndarray, y: np.ndarray) -> float:
    """NCTI: sum of three neural-collapse proximity terms, each ~[0,1], higher =
    closer to the collapsed/adapted geometry = better transfer:
      s1 between/(within+between) scatter (variability collapse),
      s2 closeness of normalised class-mean gram to the simplex ETF,
      s3 nearest-class-center accuracy."""
    X = _as_2d_f64(X)
    yy = _remap_labels(_as_labels(y))
    N, D = X.shape
    classes = np.unique(yy)
    C = len(classes)
    gmean = X.mean(0)
    means = np.stack([X[yy == c].mean(0) for c in range(C)])     # [C, D]
    priors = np.array([(yy == c).mean() for c in range(C)])

    # s1 — variability collapse (NC1)
    tr_sw = sum(float(((X[yy == c] - means[c]) ** 2).sum()) for c in range(C)) / N
    tr_sb = float((priors[:, None] * (means - gmean) ** 2).sum())
    s1 = tr_sb / (tr_sw + tr_sb + 1e-12)

    # s2 — simplex-ETF closeness of (centered, normalised) class means
    Mc = means - gmean
    Mc = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)
    G = Mc @ Mc.T
    off = G[~np.eye(C, dtype=bool)]
    s2 = 1.0 - float(np.mean(np.abs(off - (-1.0 / (C - 1))))) if C > 1 else 0.0

    # s3 — nearest-class-center accuracy
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ means.T + (means ** 2).sum(1)[None, :]
    s3 = float((d2.argmin(1) == yy).mean())

    return float(s1 + s2 + s3)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
METHODS = {
    "logme": logme_score,
    "hscore": hscore_score,
    "transrate": transrate_score,
    "gbc": gbc_score,
    "sfda": sfda_score,
    "ncti": ncti_score,
    "rankme": rankme_score,
}


def score_transferability(X: np.ndarray, y: np.ndarray, method: str = "logme") -> float:
    """Dispatch to a transferability metric by name. Higher = better transfer
    for every method."""
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    return METHODS[method](X, y)
