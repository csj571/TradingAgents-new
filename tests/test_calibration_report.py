"""Tests for the Phase 1 calibration report (tradingagents/backtest/calibration_report.py).

Fully offline and deterministic: no API keys, no network, no price source, no RNG.
Synthetic ResolvedSignal batches are constructed by hand; the CLI test writes a
real decision log via TradingMemoryLog (store_decision + update_with_outcome).
"""

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATINGS_5_TIER
from tradingagents.backtest.calibration_report import (
    MIN_RECAL_SLICE,
    RATING_PROBABILITY,
    build_report,
    format_report,
    main,
    rating_to_prob,
    signals_to_pairs,
)
from tradingagents.backtest.replay import ResolvedSignal

DECISION_BUY = "Rating: Buy\nEnter at $189-192."
DECISION_SELL = "Rating: Sell\nExit position immediately."


def _sig(date, ticker, rating, up, *, metric_alpha=None, alpha_missing=False):
    """Build a ResolvedSignal whose raw_return sign encodes ``up`` (True -> +)."""
    raw = 0.02 if up else -0.02
    if alpha_missing:
        alpha = None
    elif metric_alpha is not None:
        alpha = metric_alpha
    else:
        alpha = raw
    return ResolvedSignal(date=date, ticker=ticker, rating=rating,
                          raw_return=raw, alpha_return=alpha, holding_days=5)


def _well_calibrated_signals():
    """Deterministic ~80-signal set where each rating's up-rate ~= its mapped prob."""
    # rating -> (n_up, n_total) with n_up/n_total == RATING_PROBABILITY[rating].
    plan = {
        "Buy": (15, 20),          # 0.75
        "Overweight": (12, 20),   # 0.60
        "Underweight": (8, 20),   # 0.40
        "Sell": (5, 20),          # 0.25
    }
    signals = []
    day = 1
    for rating, (n_up, n_total) in plan.items():
        for i in range(n_total):
            date = f"2026-{(day // 28) + 1:02d}-{(day % 28) + 1:02d}"
            signals.append(_sig(date, "TICK", rating, up=(i < n_up)))
            day += 1
    return signals


# ---------------------------------------------------------------------------
# Rating -> probability map
# ---------------------------------------------------------------------------

class TestRatingMap:

    def test_keys_match_canonical_scale(self):
        assert set(RATING_PROBABILITY) == set(RATINGS_5_TIER)

    def test_monotone_ordering(self):
        assert (RATING_PROBABILITY["Buy"] > RATING_PROBABILITY["Overweight"]
                > RATING_PROBABILITY["Hold"] > RATING_PROBABILITY["Underweight"]
                > RATING_PROBABILITY["Sell"])

    def test_rating_to_prob_values(self):
        assert rating_to_prob("Buy") == 0.75
        assert rating_to_prob("Hold") == 0.5
        assert rating_to_prob("Sell") == 0.25

    def test_rating_to_prob_unknown_is_none(self):
        assert rating_to_prob("StrongBuy") is None
        assert rating_to_prob("") is None


# ---------------------------------------------------------------------------
# signals_to_pairs
# ---------------------------------------------------------------------------

class TestSignalsToPairs:

    def test_hold_excluded_and_counted(self):
        signals = [
            _sig("2026-01-01", "A", "Buy", up=True),
            _sig("2026-01-02", "B", "Hold", up=True),
            _sig("2026-01-03", "C", "Sell", up=False),
        ]
        probs, outcomes, counts = signals_to_pairs(signals)
        assert counts["directional"] == 2
        assert counts["hold"] == 1
        assert counts["dropped_no_metric"] == 0
        assert len(probs) == len(outcomes) == 2

    def test_outcome_sign_correct(self):
        signals = [
            _sig("2026-01-01", "A", "Buy", up=True),    # +0.02 -> 1
            _sig("2026-01-02", "B", "Buy", up=False),   # -0.02 -> 0
        ]
        probs, outcomes, _ = signals_to_pairs(signals)
        assert list(probs) == [0.75, 0.75]
        assert list(outcomes) == [1.0, 0.0]

    def test_zero_return_is_down(self):
        sig = ResolvedSignal("2026-01-01", "A", "Buy", raw_return=0.0)
        _, outcomes, _ = signals_to_pairs([sig])
        assert list(outcomes) == [0.0]  # not > 0

    def test_alpha_metric_drops_missing_and_counts(self):
        signals = [
            _sig("2026-01-01", "A", "Buy", up=True, metric_alpha=0.01),
            _sig("2026-01-02", "B", "Buy", up=True, alpha_missing=True),
            _sig("2026-01-03", "C", "Sell", up=False, metric_alpha=-0.01),
        ]
        probs, outcomes, counts = signals_to_pairs(signals, metric="alpha")
        assert counts["directional"] == 2
        assert counts["dropped_no_metric"] == 1
        assert list(outcomes) == [1.0, 0.0]  # +0.01 -> 1, -0.01 -> 0

    def test_unknown_rating_excluded_not_counted_as_hold(self):
        signals = [
            _sig("2026-01-01", "A", "Buy", up=True),
            _sig("2026-01-02", "B", "StrongBuy", up=True),  # unknown -> dropped silently
        ]
        _, _, counts = signals_to_pairs(signals)
        assert counts["directional"] == 1
        assert counts["hold"] == 0


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:

    def test_well_calibrated_large_set(self):
        report = build_report(_well_calibrated_signals())
        assert report.sample_too_small is False
        assert report.n_directional == 80
        assert report.n_hold == 0
        # Brier finite and small-ish (well-calibrated -> ~mean p(1-p) < 0.25).
        assert report.brier == pytest.approx(report.brier)  # finite
        assert 0.0 < report.brier < 0.25
        # Reliability curve present.
        assert report.bin_confidence is not None
        assert report.bin_accuracy is not None
        assert len(report.bin_confidence) > 0
        # Well-calibrated -> low ECE and a boolean gate verdict.
        assert isinstance(report.ece_gate_pass, bool)
        assert report.ece < 0.10
        assert report.ece_gate_pass is True

    def test_small_set_flags_sample_too_small(self):
        signals = [_sig(f"2026-01-{i+1:02d}", "A", "Buy", up=(i % 2 == 0))
                   for i in range(6)]
        report = build_report(signals)
        assert report.sample_too_small is True
        assert report.bin_confidence is None
        assert report.bin_accuracy is None
        assert report.ece_gate_pass is None
        # Raw Brier still computed.
        assert report.brier > 0.0

    def test_hold_counted_in_report(self):
        signals = _well_calibrated_signals()
        signals += [_sig(f"2026-06-{i+1:02d}", "H", "Hold", up=True) for i in range(3)]
        report = build_report(signals)
        assert report.n_hold == 3
        assert report.n_directional == 80

    def test_format_report_contains_sections(self):
        report = build_report(_well_calibrated_signals())
        text = format_report(report)
        assert "Brier" in text
        assert "reliability curve" in text
        assert "Recalibration" in text
        assert "gate" in text


# ---------------------------------------------------------------------------
# Temporal split (no look-ahead)
# ---------------------------------------------------------------------------

class TestTemporalSplit:

    def test_split_at_half_train_earliest(self):
        # 20 Buy signals, earliest 10 all up, latest 10 all down.
        # Sorted by (date, ticker), the 0.5 split must take the earliest 10 as TRAIN.
        early = [_sig(f"2026-01-{i+1:02d}", "A", "Buy", up=True) for i in range(10)]
        late = [_sig(f"2026-02-{i+1:02d}", "A", "Buy", up=False) for i in range(10)]
        report = build_report(early + late, min_samples=1, split_fraction=0.5)
        rc = report.recalibration
        assert rc.skipped_reason is None
        assert rc.n_train == 10
        assert rc.n_test == 10
        # TRAIN was the all-up slice; TEST is the all-down slice. Raw prob 0.75
        # against all-down outcomes -> large raw test ECE (~0.75).
        assert rc.raw_test_ece == pytest.approx(0.75, abs=1e-6)

    def test_split_ignores_input_order(self):
        # Same as above but shuffled input order -> split must still be by date.
        early = [_sig(f"2026-01-{i+1:02d}", "A", "Buy", up=True) for i in range(10)]
        late = [_sig(f"2026-02-{i+1:02d}", "A", "Buy", up=False) for i in range(10)]
        interleaved = [s for pair in zip(early, late, strict=True) for s in pair]
        report = build_report(interleaved, min_samples=1, split_fraction=0.5)
        rc = report.recalibration
        assert rc.n_train == 10 and rc.n_test == 10
        assert rc.raw_test_ece == pytest.approx(0.75, abs=1e-6)

    def test_recalibration_skipped_when_tiny(self):
        signals = [_sig(f"2026-01-{i+1:02d}", "A", "Buy", up=(i % 2 == 0))
                   for i in range(5)]
        report = build_report(signals, min_samples=1, split_fraction=0.5)
        rc = report.recalibration
        assert rc.skipped_reason is not None
        assert str(MIN_RECAL_SLICE) in rc.skipped_reason
        assert rc.raw_test_ece is None
        assert rc.platt_ece is None
        assert rc.isotonic_ece is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:

    def test_main_on_real_log_returns_zero(self, tmp_path, capsys):
        path = tmp_path / "trading_memory.md"
        log = TradingMemoryLog({"memory_log_path": str(path)})
        # A few resolved entries (below min_samples -> "sample too small" note,
        # but raw Brier is still printed).
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.03, 0.01, 5, "Correct.")
        log.store_decision("AAPL", "2026-01-06", DECISION_SELL)
        log.update_with_outcome("AAPL", "2026-01-06", -0.02, -0.01, 5, "Correct exit.")

        rc = main(["--log", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Brier" in out
        assert "sample too small" in out.lower()

    def test_main_metric_alpha(self, tmp_path, capsys):
        path = tmp_path / "trading_memory.md"
        log = TradingMemoryLog({"memory_log_path": str(path)})
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.03, 0.01, 5, "Correct.")
        rc = main(["--log", str(path), "--metric", "alpha"])
        assert rc == 0
        assert "alpha return" in capsys.readouterr().out
