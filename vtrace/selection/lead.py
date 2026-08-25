"""LEAD transferability estimation (Hu et al., CVPR 2024).

LEAD: "Exploring Logit Space Evolution for Model Selection", Zixuan Hu et al.,
CVPR 2024, pp. 28664-28673.  Paper PDF:
  https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_LEAD_Exploring_Logit_Space_Evolution_for_Model_Selection_CVPR_2024_paper.pdf
  arXiv mirror: https://arxiv.org/abs/2507.14559 (+ supplementary material).

Unlike feature-space metrics (LogME / H-score / TransRate / GBC), LEAD models
the *logit-space evolution* a network undergoes DURING fine-tuning, then scores
the predicted FINAL logit state. This is exactly why it is a candidate to
recover a "training-emergent" ranking that frozen-feature metrics miss.

----------------------------------------------------------------------------
EXACT ALGORITHM (reproduced from the paper + supplementary Algorithm 1).
Quotes/equation numbers refer to the CVPR-2024 camera-ready unless noted "supp".

Setup (paper Sec. 4.1 "Implementation Details", supp Algorithm 1):
  - Append a *randomly initialized* MLP head h (two hidden layers, widths 1024
    and 2048 -- supp B "Calculation of NTK") after the frozen backbone f, and
    combine them as F := h o f to produce K-class logits.
  - Obtain the INITIAL logit state ``log_init`` by feeding the frozen features
    X and labels y into a classification algorithm C, "a robust and efficient
    machine learning classifier, the multi-class SVM" (One-vs-Rest, ref [29]),
    "predictions ... normalized to obtain the predicted probabilities (i.e.,
    logits) for K classes" (supp B). So log_init in R^{N x K} is the SVM's
    normalised per-class scores.   Algorithm 1, line 3:  log_init = C(Z, Y).

Dynamical equation (Eq. 5, proved in supp Conclusion 1):
        dF_t/dt = -eta * Phi * dL_t/dF_t ,    F_0 = log_init
  with eta the learning rate and Phi the NTK matrix.

Closed-form solution under MSE loss (Eq. 6 / supp Eq. 16, "Conclusion 2"):
        E(F_t(X)) = (I - e^{-eta * Phi * t}) Y  +  e^{-eta * Phi * t} log_init
  i.e. the final logits are an interpolation between the one-hot labels Y and
  the initial SVM logits log_init, with the NTK setting the convergence rate.

Class-aware decomposition (Eq. 7, supp Algorithm 1 lines 6-13, supp C.2):
  Computing Phi at finite width is hard, so they replace the matrix exponential
  by a *scalar* rate per class, using the eigenvalues of the per-class initial
  NTK matrix.  "we separately compute NTK for each class ... we extract n samples
  of each class to calculate their respective NTK matrices and perform
  decomposition to obtain eigenvalues. The average eigenvalue of each class
  serves as the unified parameter."   For a sample x in class k:
        E(F_t(x)) = (1 - e^{-eta * lambda_bar_k * t}) y  +  e^{-eta * lambda_bar_k * t} log_init(x)   (Eq. 7)
  where y is x's one-hot label and lambda_bar_k is the mean eigenvalue of the
  class-k NTK matrix.  (supp C.2 shows class-aware NTK >> mixed/constant.)

NTK computation (supp Eq. 17, fast empirical-NTK approx. of ref [20] =
  Mohamadi & Sutherland, "A fast, well-founded approximation to the empirical
  NTK", ICML 2022):
        Phi_{u,v}(X_k, X_k) = [ grad_theta sum_{j=1}^K F^{(j)}(x_{k_u}, theta) ]
                              [ grad_theta sum_{j=1}^K F^{(j)}(x_{k_v}, theta) ]^T
  "the gradient ... is only utilized on the MLP for efficiency" -- i.e. the
  Jacobian is taken w.r.t. the random MLP HEAD parameters only (backbone frozen),
  of the output SUMMED over the K logit dims, giving a single gradient vector
  per sample; Phi is the S x S Gram of those per-sample gradient vectors.

Transferability score (paper Sec. 3.3 end, supp Algorithm 1 line 15):
  "we feed predictions into the Cross-Entropy loss to obtain the transferability
  score for model ranking."  Lower CE between the EVOLVED logits and the true
  labels => better => we return the NEGATIVE mean CE so that, like every other
  metric in ``transferability.py``, HIGHER = more transferable.

Hyper-parameters (paper Fig. 4 / supp Algorithm 1 inputs):
  - t (time coefficient): default 1.
  - S (samples per class for the NTK): default 64.
  - MLP head: 2 hidden layers, widths 1024 and 2048.
  - eta only ever appears multiplied by lambda_bar and t; we fold eta=1 and
    *normalise* the per-class eigenvalues to a stable range (see note below),
    matching the paper's reliance on the NTK *eigenvalue range* rather than its
    raw scale (supp B: "[20] provides ... the eigenvalue range obtained through
    this approximation is close to that of the original method").
----------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

__all__ = ["lead_score"]


# ---------------------------------------------------------------------------
# helpers (mirror transferability.py conventions)
# ---------------------------------------------------------------------------
def _as_2d_f64(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"features must be [N, D], got shape {X.shape}")
    return X


def _as_labels(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).reshape(-1).astype(np.int64)


def _remap_labels(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] for v in y], dtype=np.int64)


def _init_logits_svm(X: np.ndarray, y: np.ndarray, C: int) -> np.ndarray:
    """``log_init`` via multi-class (One-vs-Rest) SVM, normalised to per-class
    probabilities -- supp B "Classification Algorithm" + Algorithm 1 line 3.

    Returns log_init in R^{N x C} (rows sum to 1).  Falls back gracefully if a
    class is degenerate.  The original paper uses an OvR linear SVM (ref [29]);
    we use a linear-kernel SVC with Platt probabilities, which is the standard
    OvR multiclass SVM with normalised scores.
    """
    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
    except Exception as e:  # pragma: no cover
        raise ImportError("LEAD's log_init needs scikit-learn (LinearSVC + calibration)") from e

    Xs = StandardScaler().fit_transform(X)
    if C <= 1:
        return np.ones((X.shape[0], max(C, 1)), dtype=np.float64)

    # One-vs-Rest linear SVM, Platt-scaled to probabilities then row-normalised.
    base = LinearSVC(C=1.0, max_iter=5000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xs, y)
    proba = clf.predict_proba(Xs)  # [N, n_classes_present]

    # Re-expand to the full 0..C-1 class set (a class could be absent in a fold).
    out = np.full((X.shape[0], C), 1.0 / C, dtype=np.float64)
    present = clf.classes_.astype(int)
    out[:, present] = proba
    out = np.clip(out, 1e-12, None)
    out /= out.sum(1, keepdims=True)
    return out


def _random_mlp_params(D: int, C: int, widths=(1024, 2048), seed: int = 0):
    """Randomly initialise the classification head h (supp B: two hidden layers
    of widths 1024 and 2048). Returns the weight matrices; the empirical NTK is
    taken w.r.t. THESE parameters (backbone frozen). Kaiming-style init."""
    rng = np.random.default_rng(seed)
    h1, h2 = widths
    W1 = rng.standard_normal((D, h1)) * np.sqrt(2.0 / D)
    W2 = rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
    W3 = rng.standard_normal((h2, C)) * np.sqrt(2.0 / h2)
    return W1, W2, W3


def _head_grad_sum_logits(Xk: np.ndarray, params) -> np.ndarray:
    """Per-sample gradient of  g(x) = sum_{j=1}^K F^{(j)}(x)  w.r.t. the MLP head
    parameters (supp Eq. 17). Backbone is frozen, so x = f(image) is the input.

    F = W3^T relu(W2^T relu(W1^T x)) ; we differentiate sum-over-classes of F
    w.r.t. (W1, W2, W3), flatten and concatenate into one gradient vector per
    sample.  Returns G in R^{S x P} (P = total head params); Phi = G G^T (S x S).
    """
    W1, W2, W3 = params
    # forward (ReLU MLP)
    z1 = Xk @ W1                      # [S, h1]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ W2                      # [S, h2]
    a2 = np.maximum(z2, 0.0)
    # output F = a2 @ W3 ; g = sum_j F_j  =>  dg/dF_j = 1 for all j
    s = W3.sum(1)                     # [h2]  (sum over the K output columns of W3)

    # backprop the all-ones output-grad through the two ReLU layers.
    # dg/da2 = s ; dg/dz2 = s * 1[z2>0]
    d2 = s[None, :] * (z2 > 0.0)      # [S, h2]
    # dg/da1 = d2 @ W2^T ; dg/dz1 = that * 1[z1>0]
    d1 = (d2 @ W2.T) * (z1 > 0.0)     # [S, h1]

    # parameter grads per sample (outer products), flattened
    # dg/dW1 = x outer d1  -> [S, D*h1]
    gW1 = (Xk[:, :, None] * d1[:, None, :]).reshape(Xk.shape[0], -1)
    # dg/dW2 = a1 outer d2 -> [S, h1*h2]
    gW2 = (a1[:, :, None] * d2[:, None, :]).reshape(Xk.shape[0], -1)
    # dg/dW3 = a2 outer ones(K) -> [S, h2*K]; each output column gets +a2
    gW3 = np.repeat(a2, W3.shape[1], axis=1)  # [S, h2*K]
    return np.concatenate([gW1, gW2, gW3], axis=1)


def _class_ntk_mean_eig(Xk: np.ndarray, params) -> float:
    """Mean eigenvalue of the class-k NTK matrix Phi(X_k, X_k) (supp Eq. 17,
    Algorithm 1 lines 7-11).  Phi = G G^T where G are the per-sample summed-logit
    head gradients.  mean eigenvalue of an S x S Gram = trace(G G^T)/S =
    ||G||_F^2 / S, computed without forming the (large) Gram explicitly."""
    G = _head_grad_sum_logits(Xk, params)        # [S, P]
    S = G.shape[0]
    # trace(Phi) = sum of eigenvalues; mean eig = trace/S = sum of squared entries / S
    return float((G * G).sum() / max(S, 1))


# ---------------------------------------------------------------------------
# LEAD score
# ---------------------------------------------------------------------------
def lead_score(
    X: np.ndarray,
    y: np.ndarray,
    t: float = 1.0,
    n_samples: int = 64,
    widths: tuple = (1024, 2048),
    seed: int = 0,
) -> float:
    """LEAD transferability score (Hu et al., CVPR 2024). Higher = better
    expected fine-tuned performance.

    Args:
      X: [N, D] frozen-backbone features.
      y: [N] integer class labels.
      t: time coefficient (paper default 1; Fig. 4).
      n_samples: S, samples per class for the NTK (paper default 64).
      widths: MLP head hidden widths (supp B: 1024, 2048).
      seed: RNG seed for the random MLP head init (NTK is init-dependent;
            the paper appends a *randomly* initialised head).

    Pipeline (== supp Algorithm 1):
      1. log_init = SVM(X, y)                         (line 3)
      2. random MLP head h, F = h o f                 (line 4)
      3. per class k: lambda_bar_k = mean eig Phi(X_k) (lines 6-12, Eq. 17)
      4. evolve logits  E(F_t(x)) per Eq. 7           (line 13)
      5. score = -mean CE(softmax(F_t), y)            (line 15)
    """
    X = _as_2d_f64(X)
    yy = _remap_labels(_as_labels(y))
    N, D = X.shape
    C = int(yy.max()) + 1

    # ---- step 1: initial logits via multi-class SVM (Algorithm 1, line 3) ----
    log_init = _init_logits_svm(X, yy, C)            # [N, C], rows sum to 1

    # ---- step 2: randomly-initialised MLP head F = h o f (line 4) ----
    params = _random_mlp_params(D, C, widths=widths, seed=seed)

    # ---- step 3: per-class NTK mean eigenvalue (lines 6-12, Eq. 17) ----
    rng = np.random.default_rng(seed + 1)
    lam = np.zeros(C, dtype=np.float64)
    for k in range(C):
        idx = np.where(yy == k)[0]
        if idx.size == 0:
            continue
        sel = idx if idx.size <= n_samples else rng.choice(idx, n_samples, replace=False)
        lam[k] = _class_ntk_mean_eig(X[sel], params)

    # eta only ever multiplies lambda_bar * t. The paper relies on the NTK
    # *eigenvalue range* (supp B), not its raw scale, so we normalise the
    # per-class eigenvalues by their mean -> a dimensionless convergence rate
    # ~O(1) per class, then fold eta := 1. This keeps e^{-eta*lam*t} in a
    # well-behaved (0,1) interpolation range while PRESERVING the relative,
    # class-aware differences that Eq. 7 / supp C.2 exploit.
    lam_mean = lam[lam > 0].mean() if np.any(lam > 0) else 1.0
    rate = lam / (lam_mean + 1e-12)                  # dimensionless per-class rate

    # ---- step 4: evolve logits toward the final state (Eq. 7, line 13) ----
    decay = np.exp(-rate * t)                        # [C]  e^{-eta*lambda_bar_k*t}
    decay_n = decay[yy]                              # per-sample (by true class)
    onehot = np.zeros((N, C), dtype=np.float64)
    onehot[np.arange(N), yy] = 1.0
    F_t = (1.0 - decay_n)[:, None] * onehot + decay_n[:, None] * log_init   # [N, C]

    # ---- step 5: cross-entropy of evolved logits vs labels (line 15) ----
    # F_t is already a (label, prob)-interpolated distribution in [0,1] with rows
    # summing to 1; treat it as the predicted probability and take CE directly.
    p = np.clip(F_t, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    ce = -np.mean(np.log(p[np.arange(N), yy]))
    return float(-ce)                                # higher = better
