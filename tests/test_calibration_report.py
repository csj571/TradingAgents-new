"""Tests for the offline rating-calibration report (tradingagents.eval)."""

import json
from datetime import date, timedelta

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.eval.calibration import brier_score, ece
from tradingagents.eval.calibration_report import (
    build_report,
    collect_pairs,
    format_report,
    main,
)

_SEP = TradingMemoryLog._SEPARATOR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolved_block(day, ticker, rating, alpha, raw="+1.0%", holding="5d"):
    return (
        f"[{day} | {ticker} | {rating} | {raw} | {alpha} | {holding}]\n\n"
        f"DECISION:\nRating: {rating}\n\n"
        f"REFLECTION:\nReflection text."
    )


def _pending_block(day, ticker, rating):
    return f"[{day} | {ticker} | {rating} | pending]\n\nDECISION:\nRating: {rating}"


def _write_log(tmp_path, blocks, filename="trading_memory.md"):
    path = tmp_path / filename
    path.write_text(_SEP.join(blocks) + _SEP, encoding="utf-8")
    return path


def _entries(path):
    return TradingMemoryLog({"memory_log_path": str(path)}).load_entries()


def _day(i):
    """Sequential ISO dates so temporal ordering matches index order."""
    return (date(2026, 1, 1) + timedelta(days=i)).isoformat()


def _calibrated_half(start_idx):
    """40 signals whose hit rates exactly match the rating priors:
    20 Buys (p=0.75) with 15 wins, 20 Sells (p=0.25) with 5 wins."""
    blocks = []
    i = start_idx
    for k in range(20):
        alpha = "+0.5%" if k < 15 else "-0.5%"
        blocks.append(_resolved_block(_day(i), "AAA", "Buy", alpha))
        i += 1
    for k in range(20):
        alpha = "+0.5%" if k < 5 else "-0.5%"
        blocks.append(_resolved_block(_day(i), "BBB", "Sell", alpha))
        i += 1
    return blocks


# ---------------------------------------------------------------------------
# Vendored-module smoke (drift guard)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vendored_calibration_golden_values():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    # Perfectly calibrated: forecast 0.75 with 3/4 hit rate.
    assert ece([0.75] * 4, [1, 1, 1, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pair collection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_collect_pairs_selection_and_parsing(tmp_path):
    blocks = [
        _resolved_block(_day(0), "NVDA", "Buy", "+4.2%"),
        _resolved_block(_day(1), "NVDA", "Sell", "-0.5%"),
        _resolved_block(_day(2), "MSFT", "Hold", "+1.0%"),
        _pending_block(_day(3), "AAPL", "Overweight"),
        _resolved_block(_day(4), "TSLA", "Maybe", "+1.0%"),   # unknown rating
        _resolved_block(_day(5), "AMD", "Underweight", "bad"),  # unparseable alpha
    ]
    result = collect_pairs(_entries(_write_log(tmp_path, blocks)))

    assert result["n_hold"] == 1
    assert result["n_pending"] == 1
    assert result["n_skipped"] == 2
    signals = result["signals"]
    assert len(signals) == 2
    buy, sell = signals
    assert (buy["rating"], buy["prob"], buy["alpha"], buy["outcome"]) == ("Buy", 0.75, pytest.approx(0.042), 1)
    assert (sell["rating"], sell["prob"], sell["outcome"]) == ("Sell", 0.25, 0)
    assert buy["holding"] == 5


@pytest.mark.unit
def test_collect_pairs_sorts_by_date_and_counts_zero_alpha(tmp_path):
    blocks = [
        _resolved_block(_day(5), "ZZZ", "Buy", "+1.0%"),
        _resolved_block(_day(1), "AAA", "Buy", "+0.0%"),  # zero alpha -> miss
    ]
    result = collect_pairs(_entries(_write_log(tmp_path, blocks)))
    signals = result["signals"]
    assert [s["ticker"] for s in signals] == ["AAA", "ZZZ"]  # date order, not file order
    assert result["n_zero_alpha"] == 1
    assert signals[0]["outcome"] == 0


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_min_n_guard_gives_raw_brier_only(tmp_path):
    blocks = [_resolved_block(_day(i), "NVDA", "Buy", "+1.0%") for i in range(10)]
    report = build_report(_entries(_write_log(tmp_path, blocks)))

    assert report["sample_ok"] is False
    assert "smoke test" in report["note"]
    assert "brier_raw" in report
    assert "ece_raw" not in report and "recalibration" not in report
    assert "NOTE:" in format_report(report)


@pytest.mark.unit
def test_empty_log_reports_nothing_to_score(tmp_path):
    report = build_report(_entries(_write_log(tmp_path, [_pending_block(_day(0), "NVDA", "Buy")])))
    assert report["sample_ok"] is False
    assert report["counts"]["directional"] == 0
    format_report(report)  # must not crash


@pytest.mark.unit
def test_perfectly_calibrated_log_scores_near_zero_ece(tmp_path):
    # Both temporal halves have hit rates matching the priors exactly, so the
    # full-sample ECE is 0 and isotonic recalibration is the identity map.
    blocks = _calibrated_half(0) + _calibrated_half(40)
    report = build_report(_entries(_write_log(tmp_path, blocks)), recalibrator="isotonic")

    assert report["sample_ok"] is True
    assert report["counts"]["directional"] == 80
    assert report["base_rate"] == pytest.approx(0.5)
    assert report["ece_raw"] == pytest.approx(0.0, abs=1e-9)
    assert report["brier_decomposition"]["reliability"] == pytest.approx(0.0, abs=1e-9)

    recal = report["recalibration"]
    assert recal["ok"] is True
    assert recal["fit_n"] == 40 and recal["heldout_n"] == 40
    assert recal["ece_heldout_raw"] == pytest.approx(0.0, abs=1e-9)
    assert recal["ece_heldout_recalibrated"] == pytest.approx(0.0, abs=1e-9)
    assert recal["passes_ece_gate"] is True
    assert "ECE gate: PASS" in format_report(report)


@pytest.mark.unit
def test_regime_change_fails_ece_gate(tmp_path):
    # Fit half: ratings are perfect (Buys always win, Sells always lose), so the
    # recalibrator learns to sharpen. Held-out half inverts (regime change), so
    # recalibrated confidence is badly wrong and the non-fatal gate must fire.
    fit_half = (
        [_resolved_block(_day(i), "AAA", "Buy", "+0.5%") for i in range(20)]
        + [_resolved_block(_day(20 + i), "BBB", "Sell", "-0.5%") for i in range(20)]
    )
    heldout_half = (
        [_resolved_block(_day(40 + i), "AAA", "Buy", "-0.5%") for i in range(20)]
        + [_resolved_block(_day(60 + i), "BBB", "Sell", "+0.5%") for i in range(20)]
    )
    report = build_report(_entries(_write_log(tmp_path, fit_half + heldout_half)))

    recal = report["recalibration"]
    assert recal["ok"] is True
    assert recal["split_date"] == _day(40)
    assert recal["ece_heldout_recalibrated"] > 0.5
    assert recal["passes_ece_gate"] is False
    assert "WARNING — ECE gate" in format_report(report)


@pytest.mark.unit
def test_build_report_rejects_bad_params(tmp_path):
    entries = _entries(_write_log(tmp_path, [_resolved_block(_day(0), "NVDA", "Buy", "+1.0%")]))
    with pytest.raises(ValueError):
        build_report(entries, recalibrator="temperature")
    with pytest.raises(ValueError):
        build_report(entries, split=1.5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cli_writes_json_report(tmp_path, capsys):
    path = _write_log(tmp_path, _calibrated_half(0) + _calibrated_half(40))
    out = tmp_path / "calibration.json"

    rc = main(["--log", str(path), "--json", str(out), "--recalibrator", "isotonic"])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "Rating calibration report" in stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["counts"]["directional"] == 80
    assert report["recalibration"]["passes_ece_gate"] is True
    # Curve arrays must be JSON-native lists for a frontend to consume.
    assert isinstance(report["reliability_raw"]["bin_confidence"], list)


@pytest.mark.unit
def test_cli_missing_log_exits_2(tmp_path, capsys):
    rc = main(["--log", str(tmp_path / "missing.md")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
