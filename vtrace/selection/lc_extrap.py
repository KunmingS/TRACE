"""Learning-curve extrapolation for cheap backbone selection.

We run a cheap backbone proxy = truncated adapter+head training (~6 epochs) per
candidate backbone, producing a per-epoch val-mAP learning curve. We want to
predict the ~20-epoch *final* mAP from the first few epochs so we can rank
backbones without paying for full training. The KEY requirement is that the
extrapolated **ranking** is correct (preserves order), not that the absolute
value is exact.

This module provides TWO things:

(a) ``LCGODEExtrapolator`` -- a faithful *interface stub* for LC-GODE, the AAAI
    2025 method "Architecture-Aware Learning Curve Extrapolation via Graph
    Ordinary Differential Equation" (arXiv:2412.15554, Ding et al., AAAI 39(15)
    pp.16289-16297). Official code: https://github.com/dingyanna/LC-GODE
    LC-GODE needs a *pretrained predictor fitted on a corpus of historical
    fully-trained learning curves* (NAS-Bench-201 / OpenML-tabular in the paper)
    PLUS a graph representation of each architecture. We have neither a curve
    corpus for our backbones nor a clean architecture-graph encoding, so the
    full method is **not directly usable** here. The stub documents exactly what
    it would require and raises with that explanation. See the module-level
    docstring of ``LCGODEExtrapolator`` for the faithful method description.

(b) ``PowerLawExtrapolator`` -- a lightweight, *immediately runnable* fallback.
    It fits a saturating parametric curve family (default ``pow4``:
    ``y(t) = a - b * t**(-c)``) to the partial curve by least squares and
    evaluates it at the target epoch. This is exactly the classic LC-extrapolation
    baseline family (Domhan et al. 2015; the "LCE" baselines LC-GODE compares
    against). It needs no historical corpus -- it fits per-curve -- so it is what
    we recommend for ranking our 6-epoch proxy curves now.

Quick use::

    from vtrace.selection.lc_extrap import extrapolate_final
    pred = extrapolate_final([0.40, 0.55, 0.63, 0.68, 0.71, 0.73], target_epoch=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Parametric saturating curve families.
#
# All are "increasing toward an asymptote a" (we extrapolate val-mAP, which
# saturates upward). Each takes (t, *params) with t the 1-based epoch index.
# --------------------------------------------------------------------------- #
def _pow4(t, a, b, c):
    # a - b * t**(-c),  c > 0, b > 0  -> rises to a as t -> inf
    return a - b * np.power(t, -c)


def _exp3(t, a, b, tau):
    # a - b * exp(-t / tau)  -> rises to a as t -> inf
    return a - b * np.exp(-t / tau)


def _log2(t, a, b):
    # a + b * log(t)  (non-saturating but a common LC baseline; clipped later)
    return a + b * np.log(t)


def _vapor(t, a, b, c):
    # vapor-pressure / "ilog2"-ish:  a - b / log(t + c)
    return a - b / np.log(t + c)


@dataclass(frozen=True)
class _Family:
    name: str
    fn: Callable
    n_params: int
    # produce a reasonable initial guess and bounds from the observed curve
    init_and_bounds: Callable


def _pow4_init(t, y):
    y_last, y_first = float(y[-1]), float(y[0])
    span = max(y_last - y_first, 1e-3)
    # a slightly above the last point; b,c positive
    a0 = y_last + 0.5 * span
    p0 = (a0, max(span, 1e-2), 0.5)
    lo = (y_last - 1e-6, 1e-6, 1e-3)
    hi = (1.0 if y_last <= 1.0 else 10.0 * a0, 10.0, 5.0)
    return p0, (lo, hi)


def _exp3_init(t, y):
    y_last, y_first = float(y[-1]), float(y[0])
    span = max(y_last - y_first, 1e-3)
    a0 = y_last + 0.5 * span
    p0 = (a0, max(span, 1e-2), max(float(t[-1]) / 2.0, 1.0))
    lo = (y_last - 1e-6, 1e-6, 1e-2)
    hi = (1.0 if y_last <= 1.0 else 10.0 * a0, 10.0, 1e3)
    return p0, (lo, hi)


def _log2_init(t, y):
    p0 = (float(y[0]), max((float(y[-1]) - float(y[0])) / max(np.log(t[-1]), 1e-6), 1e-3))
    lo = (-10.0, 0.0)
    hi = (10.0, 10.0)
    return p0, (lo, hi)


def _vapor_init(t, y):
    y_last = float(y[-1])
    span = max(y_last - float(y[0]), 1e-3)
    p0 = (y_last + 0.5 * span, span, 1.0)
    lo = (y_last - 1e-6, 1e-6, 1e-3)
    hi = (1.0 if y_last <= 1.0 else 10.0, 50.0, 50.0)
    return p0, (lo, hi)


FAMILIES: Dict[str, _Family] = {
    "pow4": _Family("pow4", _pow4, 3, _pow4_init),
    "exp3": _Family("exp3", _exp3, 3, _exp3_init),
    "log2": _Family("log2", _log2, 2, _log2_init),
    "vapor": _Family("vapor", _vapor, 3, _vapor_init),
}


# --------------------------------------------------------------------------- #
# (b) Practical fallback: per-curve parametric fit.
# --------------------------------------------------------------------------- #
@dataclass
class PowerLawExtrapolator:
    """Fit a saturating parametric curve to a partial learning curve and
    extrapolate to a target epoch.

    This is the recommended, immediately-usable extrapolator. It fits *each*
    curve independently with least squares -- **no historical corpus needed**.

    Parameters
    ----------
    families : list of family names to try (default all). The best-fitting one
        (lowest fit residual on the observed points) is selected per curve.
    clip : optional (lo, hi) to clamp the prediction (e.g. (0, 1) for mAP).
    """

    families: Sequence[str] = ("pow4", "exp3", "vapor")
    clip: Optional[Tuple[float, float]] = (0.0, 1.0)
    _fits: Dict = field(default_factory=dict, repr=False)

    def _fit_one(self, t: np.ndarray, y: np.ndarray):
        from scipy.optimize import curve_fit

        best = None  # (residual, name, params)
        for fname in self.families:
            fam = FAMILIES[fname]
            if len(y) < fam.n_params + 1:
                continue
            p0, bounds = fam.init_and_bounds(t, y)
            try:
                popt, _ = curve_fit(
                    fam.fn, t, y, p0=p0, bounds=bounds, maxfev=20000
                )
            except Exception:
                continue
            resid = float(np.mean((fam.fn(t, *popt) - y) ** 2))
            if best is None or resid < best[0]:
                best = (resid, fname, popt)
        if best is None:
            # degenerate (too few points): fall back to last value held flat
            return ("flat", None, float(y[-1]))
        return (best[1], best[2], best[0])

    def predict(
        self,
        curve: Sequence[float],
        target_epoch: int,
        epochs: Optional[Sequence[float]] = None,
    ) -> float:
        """Extrapolate ``curve`` (early per-epoch metric values) to
        ``target_epoch`` (1-based). ``epochs`` overrides the assumed epoch index
        (default 1..len(curve))."""
        y = np.asarray(curve, dtype=float)
        t = (np.arange(1, len(y) + 1, dtype=float)
             if epochs is None else np.asarray(epochs, dtype=float))
        fname, popt, _ = self._fit_one(t, y)
        if fname == "flat":
            pred = float(y[-1])
        else:
            pred = float(FAMILIES[fname].fn(np.array([float(target_epoch)]), *popt)[0])
        self._fits["last"] = (fname, popt)
        if self.clip is not None:
            pred = float(np.clip(pred, self.clip[0], self.clip[1]))
        return pred

    def rank(
        self,
        curves: Dict[str, Sequence[float]],
        target_epoch: int,
    ) -> List[Tuple[str, float]]:
        """Extrapolate a dict of {name: partial_curve} and return
        [(name, predicted_final)] sorted descending (best first)."""
        preds = {k: self.predict(v, target_epoch) for k, v in curves.items()}
        return sorted(preds.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# (a) Faithful LC-GODE interface stub.
# --------------------------------------------------------------------------- #
class LCGODEExtrapolator:
    """Faithful interface stub for **LC-GODE** (Ding et al., AAAI 2025).

    Paper: "Architecture-Aware Learning Curve Extrapolation via Graph Ordinary
    Differential Equation", arXiv:2412.15554, Proc. AAAI 39(15):16289-16297.
    Official code: https://github.com/dingyanna/LC-GODE

    WHAT IT IS (faithful summary, from paper + code ``latent_de_graph.py``):

    Inputs (per curve):
        * a *partial* learning curve: the first ``cond_len`` epochs as
          ``{(t_i, y_i)}`` (paper uses ``cond_len = 10`` epochs);
        * the **architecture graph** of the network: node features ``X`` and
          adjacency ``A`` (a ``torch_geometric.Data`` object; CNN cells from
          NAS-Bench-201, MLP configs from JSON).

    Model (latent neural ODE, three learned components):
        1. Sequence encoder ``q(z_{n+1} | {y_i,t_i})`` -- a self-attention /
           GRU encoder producing variational posterior params
           ``(mu, sigma)`` of the initial latent state ``z``.
        2. Architecture encoder -- a GCN + DiffPool producing a graph-level
           embedding ``z_G`` from ``(X, A)``.
        3. Latent ODE ``dz/dt = f_theta([z || z_G])`` integrated (RK4 / torchsde)
           forward to all future timestamps; a decoder maps ``z_t -> y_hat_t``.
        Trained by maximizing an ELBO:
           ``loss = 5000*err_ic - log p(y|z) + beta * KL`` (KL annealed).

    Output:
        the full extrapolated curve ``{y_hat_t}`` over the remaining epochs ->
        the final-epoch metric.

    >>> WHY IT IS NOT DIRECTLY USABLE FOR US <<<
    LC-GODE is a *learned predictor*. To deploy it you MUST first TRAIN it on a
    **corpus of historical, fully-trained learning curves** for the same kind of
    models (the paper trains per source dataset: 550 trials x 200 epochs for
    MLP/OpenML, ~5000 NAS-Bench-201 architectures x 200 epochs for CNN). It also
    requires every candidate to be expressed as an **architecture graph**.

    We have:
        * NO corpus of historical full-length proxy curves for our backbones
          (we have a handful of candidate backbones, not thousands of curves);
        * NO clean architecture-graph encoding of heterogeneous frozen
          backbones (V-JEPA2 ViT-L vs MAE ViT-B etc.) that matches its
          NAS-Bench-201 graph schema;
        * a need to extrapolate from ~6 (not 10) early epochs.

    => Training LC-GODE from scratch on our setting is impractical, and there is
    no released pretrained checkpoint. Use ``PowerLawExtrapolator`` instead.

    This class therefore documents the interface and *refuses* to silently
    pretend; ``predict`` raises ``NotImplementedError`` with the corpus
    requirement unless a pretrained predictor is supplied.
    """

    REQUIRES = (
        "pretrained LC-GODE predictor trained on a corpus of historical "
        "fully-trained learning curves (e.g. NAS-Bench-201) + an "
        "architecture-graph encoding of each candidate"
    )

    def __init__(self, pretrained_predictor=None, arch_graph_fn=None, cond_len: int = 10):
        self.predictor = pretrained_predictor
        self.arch_graph_fn = arch_graph_fn
        self.cond_len = cond_len

    def predict(self, curve, target_epoch, arch=None):
        if self.predictor is None:
            raise NotImplementedError(
                "LC-GODE requires a pretrained predictor. Missing: "
                f"{self.REQUIRES}. Train via the official repo "
                "(https://github.com/dingyanna/LC-GODE, latent_de_graph.py) on a "
                "curve corpus, then pass it as `pretrained_predictor`, or use "
                "PowerLawExtrapolator for a corpus-free per-curve fit."
            )
        if self.arch_graph_fn is None or arch is None:
            raise NotImplementedError(
                "LC-GODE is architecture-aware: supply `arch_graph_fn` and an "
                "`arch` descriptor to build the torch_geometric graph input."
            )
        graph = self.arch_graph_fn(arch)
        return self.predictor(curve=curve, graph=graph, target_epoch=target_epoch)


# --------------------------------------------------------------------------- #
# Convenience entrypoint.
# --------------------------------------------------------------------------- #
def extrapolate_final(
    curve: Sequence[float],
    target_epoch: int = 20,
    *,
    method: str = "powerlaw",
    **kwargs,
) -> float:
    """Predict the metric at ``target_epoch`` from a partial ``curve``.

    ``method='powerlaw'`` uses the corpus-free parametric fit (recommended).
    ``method='lcgode'`` routes to the faithful stub (needs a pretrained
    predictor; will raise otherwise).
    """
    if method == "powerlaw":
        return PowerLawExtrapolator(**kwargs).predict(curve, target_epoch)
    if method == "lcgode":
        return LCGODEExtrapolator(**kwargs).predict(curve, target_epoch)
    raise ValueError(f"unknown method {method!r}")
