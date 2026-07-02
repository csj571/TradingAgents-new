"""Phase 1 calibration report — score 5-tier ratings against realized returns.

Offline calibration harness (BRE integration plan, Phase 1 / the
``CALIBRATION_ON_TAURIC`` realization). It takes the dated, resolved
``(rating, realized-return)`` records produced by the Phase 0 replay resolver,
maps each rating to a P(up) probability, binarizes the realized return into an
up/down outcome, and scores the calibration of those probabilities with the
vendored numpy calibration core (``tradingagents/backtest/calibration.py``).

Fully offline: no LLM, no network, no price source, no pandas/torch. Imports are
limited to the stdlib, numpy, the vendored ``calibration`` module, the Phase 0
``replay`` resolver, and the canonical ``RATINGS_5_TIER`` scale.

What it reports (see :class:`CalibrationReport`):
  * headline (raw) descriptive metrics over the WHOLE directional set — Brier +
    Murphy decomposition, ECE/MCE, reliability curve. These use the raw mapped
    probabilities and are labeled "raw" — a descriptive measurement of
    miscalibration, which is exactly what it claims to be;
  * a strict temporal-split recalibration block — fit Platt + isotonic on the
    earliest ``split_fraction`` of signals, report raw / Platt / isotonic ECE on
    the held-out later slice (never fit a recalibrator on data it is scored on);
  * a NON-FATAL ECE-gate line (informational PASS/WARN, never a launch gate).

``Hold`` is directional abstention: excluded from scoring, counted separately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from tradingagents.agents.utils.rating import RATINGS_5_TIER
from tradingagents.backtest import calibration as cal
from tradingagents.backtest.replay import ResolvedSignal, resolve_signals

# ---- Rating -> probability map (LOCKED; monotone P(up) over the 5-tier scale) ----
RATING_PROBABILITY: dict[str, float] = {
    "Buy": 0.75,
    "Overweight": 0.6,
    "Hold": 0.5,
    "Underweight": 0.4,
    "Sell": 0.25,
}

# The map must never drift from the canonical scale it scores.
assert set(RATING_PROBABILITY) == set(RATINGS_5_TIER), (
    "RATING_PROBABILITY keys must match RATINGS_5_TIER exactly"
)

# Ratings that express a direction (everything except the abstention tier).
HOLD_RATING = "Hold"

# ---- Report constants (LOCKED) ----
DEFAULT_MIN_SAMPLES = 50   # below this: raw Brier only, no curve / no gate verdict
MIN_RECAL_SLICE = 10       # each temporal slice needs this many signals to recalibrate
DEFAULT_SPLIT_FRACTION = 0.5


def rating_to_prob(rating: str) -> float | None:
    """Map a 5-tier rating to its P(instrument return > 0). ``None`` if unknown."""
    return RATING_PROBABILITY.get(rating)


def _metric_return(sig: ResolvedSignal, metric: str) -> float | None:
    """Return the realized return under ``metric`` (``"raw"`` or ``"alpha"``).

    ``"alpha"`` returns ``None`` when the signal has no alpha return (the caller
    drops and counts those); ``"raw"`` is always present on a ``ResolvedSignal``.
    """
    if metric == "raw":
        return sig.raw_return
    if metric == "alpha":
        return sig.alpha_return
    raise ValueError(f"metric must be 'raw' or 'alpha', got {metric!r}")


def signals_to_pairs(
    signals: list[ResolvedSignal], metric: str = "raw"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Turn resolved signals into ``(probs, outcomes)`` arrays for the scorer.

    Probability semantics is P(up): every non-``Hold`` rating maps to a
    probability that the instrument return over the horizon is ``> 0``, scored
    against the realized up/down outcome. ``Hold`` is directional abstention and
    is excluded (and counted). With ``metric="alpha"``, signals whose
    ``alpha_return is None`` are dropped (and counted).

    Returns ``(probs, outcomes, counts)`` where ``counts`` carries
    ``directional`` (scored), ``hold`` (abstentions excluded), and
    ``dropped_no_metric`` (dropped for a missing metric value). ``outcome`` is
    ``1`` if the metric return is ``> 0`` else ``0``.
    """
    probs: list[float] = []
    outcomes: list[int] = []
    n_hold = 0
    n_dropped = 0

    for sig in signals:
        prob = rating_to_prob(sig.rating)
        if prob is None or sig.rating == HOLD_RATING:
            # Unknown rating or an explicit Hold: not a directional call.
            if sig.rating == HOLD_RATING:
                n_hold += 1
            continue

        ret = _metric_return(sig, metric)
        if ret is None:
            n_dropped += 1
            continue

        probs.append(prob)
        outcomes.append(1 if ret > 0 else 0)

    counts = {
        "directional": len(probs),
        "hold": n_hold,
        "dropped_no_metric": n_dropped,
    }
    return (
        np.asarray(probs, dtype=float),
        np.asarray(outcomes, dtype=float),
        counts,
    )


@dataclass
class RecalibrationResult:
    """Temporal-split recalibration outcome (fit on TRAIN, scored on TEST).

    When the split cannot be run (either slice below :data:`MIN_RECAL_SLICE`),
    ``skipped_reason`` is set and the ECE fields are ``None``.
    """

    n_train: int
    n_test: int
    raw_test_ece: float | None = None
    platt_ece: float | None = None
    isotonic_ece: float | None = None
    skipped_reason: str | None = None


@dataclass
class CalibrationReport:
    """Everything needed to render the Phase 1 calibration report.

    The descriptive block (``brier``/``reliability``/``resolution``/
    ``uncertainty``/``ece``/``mce`` + the reliability-curve arrays) is computed
    over the WHOLE directional set using the RAW mapped probabilities. The
    reliability-curve arrays and ``ece_gate_pass`` are ``None`` when
    ``sample_too_small`` (directional N < ``min_samples``).
    """

    metric: str
    n_bins: int
    # Counts.
    n_directional: int
    n_hold: int
    n_dropped: int
    # Descriptive (raw) metrics over the full directional set.
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    mce: float
    # Reliability-curve arrays (None when sample_too_small).
    bin_confidence: np.ndarray | None
    bin_accuracy: np.ndarray | None
    bin_count: np.ndarray | None
    bin_center: np.ndarray | None
    # Temporal-split recalibration.
    recalibration: RecalibrationResult
    # Non-fatal ECE gate verdict (None when sample_too_small).
    ece_gate_pass: bool | None
    sample_too_small: bool
    min_samples: int


def _temporal_split(
    signals: list[ResolvedSignal], metric: str, split_fraction: float
) -> RecalibrationResult:
    """Strict, look-ahead-free temporal split recalibration.

    Sorts the directional signals ascending by ``(date, ticker)``, takes the
    earliest ``split_fraction`` as TRAIN and the remainder as TEST, fits Platt +
    isotonic on TRAIN pairs only, and scores raw / Platt / isotonic ECE on TEST.
    Skips (with a reason) unless BOTH slices have >= :data:`MIN_RECAL_SLICE`.
    """
    # Keep only directional signals with a usable metric value, dated-sorted.
    usable = [
        s for s in signals
        if rating_to_prob(s.rating) is not None
        and s.rating != HOLD_RATING
        and _metric_return(s, metric) is not None
    ]
    usable.sort(key=lambda s: (s.date, s.ticker))

    n = len(usable)
    n_train = int(n * split_fraction)
    train = usable[:n_train]
    test = usable[n_train:]

    if len(train) < MIN_RECAL_SLICE or len(test) < MIN_RECAL_SLICE:
        return RecalibrationResult(
            n_train=len(train),
            n_test=len(test),
            skipped_reason=(
                f"need >= {MIN_RECAL_SLICE} signals in each temporal slice "
                f"(train={len(train)}, test={len(test)})"
            ),
        )

    tr_p, tr_y, _ = signals_to_pairs(train, metric)
    te_p, te_y, _ = signals_to_pairs(test, metric)

    platt = cal.PlattScaler().fit(tr_p, tr_y)
    iso = cal.IsotonicCalibrator().fit(tr_p, tr_y)

    return RecalibrationResult(
        n_train=len(train),
        n_test=len(test),
        raw_test_ece=cal.ece(te_p, te_y),
        platt_ece=cal.ece(platt.transform(te_p), te_y),
        isotonic_ece=cal.ece(iso.transform(te_p), te_y),
    )


def build_report(
    signals: list[ResolvedSignal] | None = None,
    *,
    log_path=None,
    metric: str = "raw",
    n_bins: int = cal.DEFAULT_N_BINS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    split_fraction: float = DEFAULT_SPLIT_FRACTION,
) -> CalibrationReport:
    """Build a :class:`CalibrationReport` from resolved signals (or a log path).

    If ``signals is None``, resolves them offline via
    :func:`replay.resolve_signals(log_path) <tradingagents.backtest.replay.resolve_signals>`.
    """
    if signals is None:
        signals, _ = resolve_signals(log_path)

    probs, outcomes, counts = signals_to_pairs(signals, metric)
    n_directional = counts["directional"]
    sample_too_small = n_directional < min_samples

    brier = cal.brier_score(probs, outcomes)
    decomp = cal.brier_decomposition(probs, outcomes, n_bins=n_bins)
    ece_val = cal.ece(probs, outcomes, n_bins=n_bins)
    mce_val = cal.mce(probs, outcomes, n_bins=n_bins)

    if sample_too_small:
        # Raw Brier still computed above, but no curve and no gate verdict.
        bin_confidence = bin_accuracy = bin_count = bin_center = None
        ece_gate_pass = None
    else:
        curve = cal.reliability_curve(probs, outcomes, n_bins=n_bins)
        bin_confidence = curve.bin_confidence
        bin_accuracy = curve.bin_accuracy
        bin_count = curve.bin_count
        bin_center = curve.bin_center
        ece_gate_pass = cal.passes_ece_gate(ece_val)

    recalibration = _temporal_split(signals, metric, split_fraction)

    return CalibrationReport(
        metric=metric,
        n_bins=n_bins,
        n_directional=n_directional,
        n_hold=counts["hold"],
        n_dropped=counts["dropped_no_metric"],
        brier=brier,
        reliability=decomp.reliability,
        resolution=decomp.resolution,
        uncertainty=decomp.uncertainty,
        ece=ece_val,
        mce=mce_val,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
        bin_center=bin_center,
        recalibration=recalibration,
        ece_gate_pass=ece_gate_pass,
        sample_too_small=sample_too_small,
        min_samples=min_samples,
    )


def _fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def format_report(report: CalibrationReport) -> str:
    """Render a :class:`CalibrationReport` as a compact, readable text table."""
    r = report
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("BRE-1 Calibration Report (Phase 1) — 5-tier ratings vs returns")
    lines.append("=" * 60)
    lines.append(f"metric               : {r.metric} return  |  bins: {r.n_bins}")
    lines.append(
        f"signals              : directional={r.n_directional}  "
        f"hold(excluded)={r.n_hold}  dropped(no-{r.metric})={r.n_dropped}"
    )
    lines.append("")
    lines.append("-- Descriptive (RAW mapped probabilities, whole directional set) --")
    lines.append(f"  Brier score        : {_fmt(r.brier)}   (0=perfect, 0.25=always-0.5)")
    lines.append(f"  Murphy reliability : {_fmt(r.reliability)}   (lower better)")
    lines.append(f"  Murphy resolution  : {_fmt(r.resolution)}   (higher better)")
    lines.append(f"  Murphy uncertainty : {_fmt(r.uncertainty)}   (base-rate variance)")
    lines.append(f"  ECE                : {_fmt(r.ece)}")
    lines.append(f"  MCE                : {_fmt(r.mce)}")
    lines.append("")

    if r.sample_too_small:
        lines.append(
            f"  reliability curve  : (sample too small, N<{r.min_samples}) — not drawn"
        )
    else:
        # One-line reliability-curve summary: per non-empty bin conf->acc (n).
        parts = [
            f"{c:.2f}->{a:.2f}(n={n})"
            for c, a, n in zip(r.bin_confidence, r.bin_accuracy, r.bin_count, strict=True)
        ]
        lines.append("  reliability curve  : " + "  ".join(parts))
    lines.append("")

    lines.append("-- Recalibration (temporal split, fit on TRAIN, scored on TEST) --")
    rc = r.recalibration
    if rc.skipped_reason is not None:
        lines.append(f"  skipped            : {rc.skipped_reason}")
    else:
        lines.append(f"  train / test sizes : {rc.n_train} / {rc.n_test}")
        lines.append(f"  raw  test ECE      : {_fmt(rc.raw_test_ece)}")
        lines.append(f"  Platt    test ECE  : {_fmt(rc.platt_ece)}")
        lines.append(f"  isotonic test ECE  : {_fmt(rc.isotonic_ece)}")
    lines.append("")

    lines.append("-- ECE gate (informational; NON-FATAL) --")
    if r.ece_gate_pass is None:
        lines.append(
            f"  gate               : n/a (sample too small, N<{r.min_samples})"
        )
    else:
        verdict = "PASS" if r.ece_gate_pass else "WARN"
        lines.append(
            f"  gate               : {verdict}  "
            f"(ECE {_fmt(r.ece)} vs max {cal.DEFAULT_MAX_ECE})"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Builds the report, prints it, and returns 0 on success."""
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.backtest.calibration_report",
        description=(
            "Offline calibration report scoring 5-tier ratings against realized "
            "returns from a replayed decision log. No LLM, no network."
        ),
    )
    parser.add_argument("--log", required=True, help="path to the decision-log markdown file")
    parser.add_argument("--metric", choices=["raw", "alpha"], default="raw",
                        help="which realized return to score (default: raw)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"directional signals required for a curve/gate (default: {DEFAULT_MIN_SAMPLES})")
    parser.add_argument("--split", type=float, default=DEFAULT_SPLIT_FRACTION,
                        help="earliest fraction used as the recalibration TRAIN slice (default: 0.5)")
    parser.add_argument("--bins", type=int, default=cal.DEFAULT_N_BINS,
                        help=f"number of calibration bins (default: {cal.DEFAULT_N_BINS})")
    args = parser.parse_args(argv)

    report = build_report(
        log_path=args.log,
        metric=args.metric,
        n_bins=args.bins,
        min_samples=args.min_samples,
        split_fraction=args.split,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
