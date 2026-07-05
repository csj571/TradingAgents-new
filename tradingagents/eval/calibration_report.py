"""Offline rating-calibration report over the TradingAgents decision log.

Scores the 5-tier ratings recorded in the persistent memory log against their
realized outcomes: each resolved entry's rating maps to a prior probability of
positive alpha (``RATING_PRIOR_PROB``), and the outcome is the sign of the
logged alpha (excess return vs the benchmark over the entry's recorded holding
window). Emits Brier score + Murphy decomposition, ECE/MCE raw *and*
recalibrated (Platt or isotonic, fit on a temporal split — never score raw
confidence as if it were calibrated), and reliability-curve data.

Fully offline: reads the markdown log, no LLM, no network, no API keys.

Scope discipline: this measures *calibration of the ratings* — whether stated
confidence matches hit rate — nothing more. It does not claim the pipeline is
good, and its numbers apply only to this log's tickers and date range.

Usage:
    python -m tradingagents.eval.calibration_report
    python -m tradingagents.eval.calibration_report --log path/to/trading_memory.md
    python -m tradingagents.eval.calibration_report --min-n 50 --split 0.5 \
        --recalibrator platt --json calibration.json

Caveats baked into the numbers:
- The log stores returns as strings rounded to 0.1% (e.g. "+4.2%"), so alphas
  within +/-0.05% of zero may be mis-signed; exact 0.0% counts as a miss.
- The holding window varies per entry (the log records actual trading days
  available, not a fixed horizon); the report shows the distribution.
- If ``memory_log_max_entries`` is configured, rotation permanently prunes the
  oldest resolved entries — the calibration sample is capped at that number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict

import numpy as np

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATING_PRIOR_PROB
from tradingagents.eval.calibration import (
    DEFAULT_MAX_ECE,
    DEFAULT_N_BINS,
    IsotonicCalibrator,
    PlattScaler,
    brier_decomposition,
    brier_score,
    ece,
    mce,
    passes_ece_gate,
    reliability_curve,
)

DEFAULT_MIN_N = 50
DEFAULT_SPLIT = 0.5

_RECALIBRATORS = {"platt": PlattScaler, "isotonic": IsotonicCalibrator}


# ---------------------------------------------------------------------------
# Log entry -> (prob, outcome) pairs
# ---------------------------------------------------------------------------

def _parse_pct(text: str | None) -> float | None:
    """Parse the log's display-string percentage ("+4.2%", "-0.5%") to a float."""
    if not text or not text.endswith("%"):
        return None
    try:
        return float(text[:-1]) / 100.0
    except ValueError:
        return None


def _parse_holding(text: str | None) -> int | None:
    """Parse the log's holding field ("5d") to an int of trading days."""
    if not text or not text.endswith("d"):
        return None
    try:
        return int(text[:-1])
    except ValueError:
        return None


def collect_pairs(entries: list[dict]) -> dict:
    """Turn parsed memory-log entries into scoring pairs plus exclusion counts.

    Returns a dict with:
        signals   : list of {date, ticker, rating, prob, alpha, outcome, holding}
                    for resolved directional (non-Hold) entries, sorted by date
        n_pending, n_hold, n_skipped, n_zero_alpha : exclusion/caveat counts
    """
    signals: list[dict] = []
    n_pending = n_hold = n_skipped = n_zero_alpha = 0

    for e in entries:
        if e.get("pending"):
            n_pending += 1
            continue
        rating = e.get("rating")
        alpha = _parse_pct(e.get("alpha"))
        if rating not in RATING_PRIOR_PROB or alpha is None:
            n_skipped += 1
            continue
        if rating == "Hold":
            n_hold += 1
            continue
        if alpha == 0.0:
            n_zero_alpha += 1  # rounding ambiguity; scored as a miss below
        signals.append(
            {
                "date": e.get("date", ""),
                "ticker": e.get("ticker", ""),
                "rating": rating,
                "prob": RATING_PRIOR_PROB[rating],
                "alpha": alpha,
                "outcome": 1 if alpha > 0.0 else 0,
                "holding": _parse_holding(e.get("holding")),
            }
        )

    # Temporal order is load-bearing for the leakage-free split.
    signals.sort(key=lambda s: s["date"])
    return {
        "signals": signals,
        "n_pending": n_pending,
        "n_hold": n_hold,
        "n_skipped": n_skipped,
        "n_zero_alpha": n_zero_alpha,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _curve_dict(curve) -> dict:
    return {
        "bin_center": curve.bin_center.tolist(),
        "bin_confidence": curve.bin_confidence.tolist(),
        "bin_accuracy": curve.bin_accuracy.tolist(),
        "bin_count": curve.bin_count.tolist(),
    }


def build_report(
    entries: list[dict],
    *,
    n_bins: int = DEFAULT_N_BINS,
    min_n: int = DEFAULT_MIN_N,
    split: float = DEFAULT_SPLIT,
    recalibrator: str = "platt",
    max_ece: float = DEFAULT_MAX_ECE,
) -> dict:
    """Build the full calibration report as a JSON-serializable dict."""
    if recalibrator not in _RECALIBRATORS:
        raise ValueError(f"unknown recalibrator {recalibrator!r}; choose from {sorted(_RECALIBRATORS)}")
    if not 0.0 < split < 1.0:
        raise ValueError(f"--split must be in (0, 1), got {split}")

    collected = collect_pairs(entries)
    signals = collected["signals"]
    n = len(signals)

    report: dict = {
        "counts": {
            "directional": n,
            "hold_excluded": collected["n_hold"],
            "pending_excluded": collected["n_pending"],
            "skipped_unparseable": collected["n_skipped"],
            "zero_alpha_scored_as_miss": collected["n_zero_alpha"],
        },
        "rating_prior_prob": dict(RATING_PRIOR_PROB),
        "params": {
            "n_bins": n_bins,
            "min_n": min_n,
            "split": split,
            "recalibrator": recalibrator,
            "max_ece_gate": max_ece,
        },
        "caveats": [
            "Alphas are parsed from strings rounded to 0.1%; near-zero alphas may be mis-signed, and exact 0.0% scores as a miss.",
            "Holding windows vary per entry (actual trading days available), so calibration is over a mixed horizon.",
            "If memory_log_max_entries is set, rotation has pruned the oldest resolved entries; the sample is capped.",
            "These numbers validate these ratings on this log's tickers and dates only — nothing transfers beyond that.",
        ],
    }

    if n == 0:
        report["sample_ok"] = False
        report["note"] = "No resolved directional signals in the log; nothing to score."
        return report

    probs = np.array([s["prob"] for s in signals])
    outcomes = np.array([s["outcome"] for s in signals], dtype=float)

    holding_days = [s["holding"] for s in signals if s["holding"] is not None]
    report["holding_days_distribution"] = dict(sorted(Counter(holding_days).items()))
    report["date_range"] = [signals[0]["date"], signals[-1]["date"]]
    report["base_rate"] = float(outcomes.mean())
    report["rating_counts"] = dict(Counter(s["rating"] for s in signals))
    report["brier_raw"] = brier_score(probs, outcomes)

    if n < min_n:
        report["sample_ok"] = False
        report["note"] = (
            f"Sample too small ({n} directional signals < min-n {min_n}): raw Brier only. "
            "Treat as a smoke test, not validation."
        )
        return report

    report["sample_ok"] = True
    report["brier_decomposition"] = asdict(brier_decomposition(probs, outcomes, n_bins))
    report["ece_raw"] = ece(probs, outcomes, n_bins)
    report["mce_raw"] = mce(probs, outcomes, n_bins)
    report["reliability_raw"] = _curve_dict(reliability_curve(probs, outcomes, n_bins))

    # Leakage-free recalibration: fit on the earlier slice, score the later one.
    n_fit = int(n * split)
    if n_fit < 2 or n - n_fit < 2:
        report["recalibration"] = {
            "ok": False,
            "note": f"Temporal split {split} leaves too few samples to fit and score ({n_fit}/{n - n_fit}).",
        }
        return report

    scaler = _RECALIBRATORS[recalibrator]().fit(probs[:n_fit], outcomes[:n_fit])
    heldout_probs = probs[n_fit:]
    heldout_outcomes = outcomes[n_fit:]
    recal_probs = np.clip(scaler.transform(heldout_probs), 0.0, 1.0)
    ece_recal = ece(recal_probs, heldout_outcomes, n_bins)

    report["recalibration"] = {
        "ok": True,
        "method": recalibrator,
        "fit_n": n_fit,
        "heldout_n": n - n_fit,
        "split_date": signals[n_fit]["date"],
        # Held-out raw metrics, so raw-vs-recalibrated compares the same slice.
        "ece_heldout_raw": ece(heldout_probs, heldout_outcomes, n_bins),
        "mce_heldout_raw": mce(heldout_probs, heldout_outcomes, n_bins),
        "brier_heldout_raw": brier_score(heldout_probs, heldout_outcomes),
        "ece_heldout_recalibrated": ece_recal,
        "mce_heldout_recalibrated": mce(recal_probs, heldout_outcomes, n_bins),
        "brier_heldout_recalibrated": brier_score(recal_probs, heldout_outcomes),
        "reliability_heldout_recalibrated": _curve_dict(
            reliability_curve(recal_probs, heldout_outcomes, n_bins)
        ),
        "passes_ece_gate": passes_ece_gate(ece_recal, max_ece),
    }
    return report


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _format_curve(curve: dict) -> list[str]:
    lines = ["    bin    conf    freq      n"]
    for c, p, a, n in zip(
        curve["bin_center"],
        curve["bin_confidence"],
        curve["bin_accuracy"],
        curve["bin_count"],
        strict=True,
    ):
        lines.append(f"    {c:.2f}   {p:.3f}   {a:.3f}   {n:4d}")
    return lines


def format_report(report: dict) -> str:
    """Render the report dict as the human-readable text summary."""
    counts = report["counts"]
    lines = [
        "Rating calibration report (offline, decision log)",
        "=" * 50,
        f"Directional signals scored : {counts['directional']}",
        f"Hold excluded              : {counts['hold_excluded']}",
        f"Pending excluded           : {counts['pending_excluded']}",
        f"Skipped (unparseable)      : {counts['skipped_unparseable']}",
    ]
    if counts["zero_alpha_scored_as_miss"]:
        lines.append(f"Zero-alpha (scored as miss): {counts['zero_alpha_scored_as_miss']}")

    if counts["directional"] > 0:
        lines += [
            f"Date range                 : {report['date_range'][0]} .. {report['date_range'][1]}",
            f"Rating counts              : {report['rating_counts']}",
            f"Holding days (days: n)     : {report['holding_days_distribution']}",
            f"Base rate P(alpha > 0)     : {report['base_rate']:.3f}",
            f"Brier (raw, full sample)   : {report['brier_raw']:.4f}",
        ]

    if not report.get("sample_ok"):
        lines += ["", f"NOTE: {report['note']}"]
        return "\n".join(lines)

    dec = report["brier_decomposition"]
    lines += [
        f"  reliability={dec['reliability']:.4f}  resolution={dec['resolution']:.4f}  uncertainty={dec['uncertainty']:.4f}",
        f"ECE (raw, full sample)     : {report['ece_raw']:.4f}",
        f"MCE (raw, full sample)     : {report['mce_raw']:.4f}",
        "",
        "Reliability curve (raw, full sample):",
        *_format_curve(report["reliability_raw"]),
    ]

    recal = report["recalibration"]
    if not recal["ok"]:
        lines += ["", f"Recalibration skipped: {recal['note']}"]
        return "\n".join(lines)

    lines += [
        "",
        f"Recalibration ({recal['method']}, temporal split at {recal['split_date']}; "
        f"fit n={recal['fit_n']}, held-out n={recal['heldout_n']}):",
        f"  held-out raw          : Brier={recal['brier_heldout_raw']:.4f}  "
        f"ECE={recal['ece_heldout_raw']:.4f}  MCE={recal['mce_heldout_raw']:.4f}",
        f"  held-out recalibrated : Brier={recal['brier_heldout_recalibrated']:.4f}  "
        f"ECE={recal['ece_heldout_recalibrated']:.4f}  MCE={recal['mce_heldout_recalibrated']:.4f}",
        "",
        "Reliability curve (held-out, recalibrated):",
        *_format_curve(recal["reliability_heldout_recalibrated"]),
    ]

    max_ece = report["params"]["max_ece_gate"]
    if recal["passes_ece_gate"]:
        lines.append(f"\nECE gate: PASS (recalibrated ECE <= {max_ece}). Research signal, not a launch gate.")
    else:
        lines.append(
            f"\nWARNING — ECE gate: recalibrated ECE {recal['ece_heldout_recalibrated']:.4f} > {max_ece}. "
            "Stated rating confidence does not match hit rate on held-out data. (Non-fatal; research signal.)"
        )

    lines += ["", "Caveats:"] + [f"  - {c}" for c in report["caveats"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Path to the memory log (default: the configured memory_log_path).",
    )
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                        help=f"Minimum directional signals for the full report (default {DEFAULT_MIN_N}).")
    parser.add_argument("--bins", type=int, default=DEFAULT_N_BINS,
                        help=f"Reliability/ECE bin count (default {DEFAULT_N_BINS}).")
    parser.add_argument("--split", type=float, default=DEFAULT_SPLIT,
                        help=f"Temporal fraction used to fit the recalibrator (default {DEFAULT_SPLIT}).")
    parser.add_argument("--recalibrator", choices=sorted(_RECALIBRATORS), default="platt",
                        help="Recalibration method (default platt).")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Also write the full report (incl. curve arrays) as JSON.")
    args = parser.parse_args(argv)

    log_path = args.log
    if log_path is None:
        from tradingagents.default_config import DEFAULT_CONFIG

        log_path = DEFAULT_CONFIG.get("memory_log_path")
    if not log_path:
        print("No log path: pass --log or configure memory_log_path.", file=sys.stderr)
        return 2

    log = TradingMemoryLog({"memory_log_path": str(log_path)})
    if log._log_path is None or not log._log_path.exists():
        print(f"Memory log not found: {log_path}", file=sys.stderr)
        return 2

    report = build_report(
        log.load_entries(),
        n_bins=args.bins,
        min_n=args.min_n,
        split=args.split,
        recalibrator=args.recalibrator,
    )
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
