"""Calibration core (BUILD_PLAN B0) — the load-bearing metric.

VENDORED — verbatim copy of ``bre-sim/engine/calibration.py`` (the BRE-1 research
monorepo). This repo can't ``pip install`` that research repo, and the module is
numpy-only, so it is vendored here per docs/BRE_INTEGRATION_PLAN.md Phase 1. Keep
it in sync with the source; do not edit the math here — fix it upstream and re-copy.
Source: csj571/bre-sim :: engine/calibration.py

Hand-rolled in NumPy only (no sklearn/scipy), so it imports torch-free and runs
anywhere the engine does. Everything here scores a set of probabilistic
predictions against resolved binary outcomes:

    probs    : array of predicted P(outcome = 1), each in [0, 1]
    outcomes : array of realized outcomes in {0, 1}

Provides Brier score + Murphy decomposition, ECE/MCE, reliability-curve data,
two recalibrators (Platt logistic, isotonic via PAV), Cohen's kappa for the
inter-rater accuracy ceiling, and the ECE gate constant the CI test (B3) imports.

Per the spec: raw verbalized confidence is *never* scored unrecalibrated — fit a
recalibrator on held-out data and score the transformed probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Gate constants (BUILD_PLAN acceptance gates; B3's CI test imports these) ----
DEFAULT_N_BINS = 10
DEFAULT_MAX_ECE = 0.10          # target: ECE < 0.1 on recalibrated confidence
_EPS = 1e-12


def _as_pairs(probs, outcomes):
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs and outcomes must match: {p.shape} vs {y.shape}")
    return p, y


def _bin_index(probs, n_bins):
    """Equal-width bin index in [0, n_bins-1]; p==1 lands in the last bin."""
    idx = (np.asarray(probs, float) * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


# ---------------- Brier score + Murphy decomposition ----------------

def brier_score(probs, outcomes) -> float:
    """Mean squared error between predicted prob and binary outcome (lower better;
    0.0 perfect, 0.25 = always predicting 0.5)."""
    p, y = _as_pairs(probs, outcomes)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class BrierDecomposition:
    reliability: float   # calibration error (lower better)
    resolution: float    # ability to separate outcomes (higher better)
    uncertainty: float   # base-rate variance (data property)
    brier: float         # reliability - resolution + uncertainty


def brier_decomposition(probs, outcomes, n_bins: int = DEFAULT_N_BINS) -> BrierDecomposition:
    """Murphy (1973) 3-term decomposition. Exact (brier == rel - res + unc) when
    each bin holds a single distinct forecast value; otherwise within-bin variance
    makes it approximate."""
    p, y = _as_pairs(probs, outcomes)
    N = p.size
    if N == 0:
        return BrierDecomposition(0.0, 0.0, 0.0, 0.0)
    base = float(y.mean())
    uncertainty = base * (1.0 - base)
    idx = _bin_index(p, n_bins)
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        f_k = float(p[mask].mean())     # mean forecast in bin
        o_k = float(y[mask].mean())     # observed frequency in bin
        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - base) ** 2
    reliability /= N
    resolution /= N
    return BrierDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        brier=reliability - resolution + uncertainty,
    )


# ---------------- ECE / MCE ----------------

def ece(probs, outcomes, n_bins: int = DEFAULT_N_BINS) -> float:
    """Expected Calibration Error: sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|."""
    p, y = _as_pairs(probs, outcomes)
    N = p.size
    if N == 0:
        return 0.0
    idx = _bin_index(p, n_bins)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        total += (cnt / N) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def mce(probs, outcomes, n_bins: int = DEFAULT_N_BINS) -> float:
    """Maximum Calibration Error: the worst per-bin |acc - conf|."""
    p, y = _as_pairs(probs, outcomes)
    if p.size == 0:
        return 0.0
    idx = _bin_index(p, n_bins)
    worst = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        worst = max(worst, abs(float(y[mask].mean()) - float(p[mask].mean())))
    return float(worst)


# ---------------- Reliability curve (for the reliability diagram) ----------------

@dataclass(frozen=True)
class ReliabilityCurve:
    bin_confidence: np.ndarray   # mean predicted prob per non-empty bin
    bin_accuracy: np.ndarray     # observed frequency per non-empty bin
    bin_count: np.ndarray        # samples per non-empty bin
    bin_center: np.ndarray       # nominal bin center


def reliability_curve(probs, outcomes, n_bins: int = DEFAULT_N_BINS) -> ReliabilityCurve:
    """Per-bin (confidence, accuracy, count) — the data a reliability diagram plots.
    A perfectly calibrated model has bin_accuracy == bin_confidence (the diagonal)."""
    p, y = _as_pairs(probs, outcomes)
    idx = _bin_index(p, n_bins)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    conf, acc, cnt, cen = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        conf.append(float(p[mask].mean()))
        acc.append(float(y[mask].mean()))
        cnt.append(c)
        cen.append(float(centers[b]))
    return ReliabilityCurve(
        bin_confidence=np.array(conf),
        bin_accuracy=np.array(acc),
        bin_count=np.array(cnt, dtype=int),
        bin_center=np.array(cen),
    )


# ---------------- Recalibrators ----------------

def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class PlattScaler:
    """Platt scaling: fit sigmoid(a * logit(p) + b) by Newton/IRLS on log-loss,
    with Platt's target smoothing to avoid overfitting on small sets."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, probs, outcomes, iters: int = 100, ridge: float = 1e-6) -> "PlattScaler":
        z, y = _as_pairs(_logit(probs), outcomes)
        n = z.size
        if n == 0:
            return self
        # Platt target smoothing
        n_pos = float(y.sum())
        n_neg = float(n - n_pos)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        t = np.where(y > 0.5, t_pos, t_neg)
        X = np.column_stack([z, np.ones_like(z)])  # features [logit, 1]
        w = np.array([1.0, 0.0])
        for _ in range(iters):
            p = _sigmoid(X @ w)
            grad = X.T @ (p - t)
            W = p * (1.0 - p)
            H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.a, self.b = float(w[0]), float(w[1])
        return self

    def transform(self, probs) -> np.ndarray:
        return _sigmoid(self.a * _logit(probs) + self.b)


class IsotonicCalibrator:
    """Isotonic recalibration via Pool-Adjacent-Violators — a non-decreasing,
    non-parametric map prob -> calibrated prob. More flexible than Platt; needs
    more data."""

    def __init__(self):
        self._x = None
        self._y = None

    @staticmethod
    def _pav(y):
        blocks = []  # [value, count]
        for v in y:
            cur_v, cur_c = float(v), 1
            while blocks and blocks[-1][0] >= cur_v:
                pv, pc = blocks.pop()
                cur_v = (cur_v * cur_c + pv * pc) / (cur_c + pc)
                cur_c += pc
            blocks.append([cur_v, cur_c])
        out = np.empty(len(y))
        i = 0
        for v, c in blocks:
            out[i:i + c] = v
            i += c
        return out

    def fit(self, probs, outcomes) -> "IsotonicCalibrator":
        p, y = _as_pairs(probs, outcomes)
        if p.size == 0:
            self._x = np.array([0.0, 1.0])
            self._y = np.array([0.0, 1.0])
            return self
        order = np.argsort(p, kind="mergesort")
        self._x = p[order]
        self._y = self._pav(y[order])
        return self

    def transform(self, probs) -> np.ndarray:
        p = np.asarray(probs, float)
        if self._x is None:
            return p
        return np.interp(p, self._x, self._y)


# ---------------- Inter-rater agreement (accuracy ceiling) ----------------

def cohens_kappa(rater_a, rater_b) -> float:
    """Cohen's kappa between two raters' labels. Used as the ceiling: never claim
    calibration/accuracy better than the agreement of the ground-truth labels."""
    a = np.asarray(rater_a).ravel()
    b = np.asarray(rater_b).ravel()
    if a.size == 0:
        return 0.0
    po = float(np.mean(a == b))
    cats = np.unique(np.concatenate([a, b]))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ---------------- Gate ----------------

def passes_ece_gate(ece_value: float, max_ece: float = DEFAULT_MAX_ECE) -> bool:
    """Launch-blocking ECE gate (BUILD_PLAN). True if calibration is good enough."""
    return bool(ece_value <= max_ece)
