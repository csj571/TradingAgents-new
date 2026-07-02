"""Tests for the Phase 0 offline replay resolver (tradingagents/backtest/replay.py).

Fully offline: no API keys, no network, no price source. The synthetic log is
written authentically via TradingMemoryLog.store_decision + update_with_outcome so
these tests also guard against decision-log format drift. Malformed / missing-field
entries are hand-written because the write API always emits well-formed tags.
"""

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.backtest import ResolvedSignal, ResolveStats, resolve_signals
from tradingagents.backtest.replay import _parse_holding, _parse_pct

_SEP = TradingMemoryLog._SEPARATOR

DECISION_BUY = "Rating: Buy\nEnter at $189-192."
DECISION_SELL = "Rating: Sell\nExit position immediately."


def make_log(tmp_path, filename="trading_memory.md"):
    return TradingMemoryLog({"memory_log_path": str(tmp_path / filename)})


def _append_raw(tmp_path, tag_line, filename="trading_memory.md"):
    """Append a hand-written entry (tag + minimal DECISION) to the log file."""
    entry = f"{tag_line}\n\nDECISION:\nHand-written entry.{_SEP}"
    with open(tmp_path / filename, "a", encoding="utf-8") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# _parse_pct / _parse_holding helpers
# ---------------------------------------------------------------------------

class TestParseHelpers:

    @pytest.mark.parametrize("text,expected", [
        ("+3.2%", 0.032),
        ("-1.0%", -0.01),
        ("+0.0%", 0.0),
        ("  +5.5% ", 0.055),
        ("12%", 0.12),
        ("-10", -0.10),  # tolerates a missing trailing %
    ])
    def test_parse_pct_valid(self, text, expected):
        assert _parse_pct(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", [None, "", "n/a", "N/A", "pending", "garbage", "++1%"])
    def test_parse_pct_invalid_returns_none(self, text):
        assert _parse_pct(text) is None

    @pytest.mark.parametrize("text,expected", [("5d", 5), ("12d", 12), (" 3d ", 3), ("7", 7)])
    def test_parse_holding_valid(self, text, expected):
        assert _parse_holding(text) == expected

    @pytest.mark.parametrize("text", [None, "", "n/a", "abc", "5.5d"])
    def test_parse_holding_invalid_returns_none(self, text):
        assert _parse_holding(text) is None


# ---------------------------------------------------------------------------
# resolve_signals — core behaviour
# ---------------------------------------------------------------------------

class TestResolveSignals:

    def test_empty_missing_log(self, tmp_path):
        """Missing file -> empty list + zeroed stats, no crash."""
        signals, stats = resolve_signals(str(tmp_path / "does_not_exist.md"))
        assert signals == []
        assert stats == ResolveStats(0, 0, 0, 0)

    def test_no_source_is_empty(self):
        """No path/log/entries -> empty (noop TradingMemoryLog)."""
        signals, stats = resolve_signals()
        assert signals == []
        assert stats.total_entries == 0

    def test_resolves_and_parses_returns(self, tmp_path):
        """A resolved entry yields a ResolvedSignal with correctly parsed floats."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.032, 0.011, 5, "Momentum confirmed.")

        signals, stats = resolve_signals(str(tmp_path / "trading_memory.md"))

        assert stats == ResolveStats(total_entries=1, resolved=1,
                                     skipped_pending=0, skipped_malformed=0)
        assert len(signals) == 1
        sig = signals[0]
        assert isinstance(sig, ResolvedSignal)
        assert sig.date == "2026-01-05"
        assert sig.ticker == "NVDA"
        assert sig.rating == "Buy"
        # Tags round-trip through :+.1% formatting -> "+3.2%" / "+1.1%".
        assert sig.raw_return == pytest.approx(0.032, abs=1e-9)
        assert sig.alpha_return == pytest.approx(0.011, abs=1e-9)
        assert sig.holding_days == 5

    def test_negative_returns_parsed(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-12", DECISION_SELL)
        log.update_with_outcome("NVDA", "2026-01-12", -0.03, -0.01, 5, "Correct exit.")
        signals, _ = resolve_signals(str(tmp_path / "trading_memory.md"))
        assert signals[0].raw_return == pytest.approx(-0.03, abs=1e-9)
        assert signals[0].alpha_return == pytest.approx(-0.01, abs=1e-9)

    def test_pending_excluded_and_counted(self, tmp_path):
        """Pending entries are skipped and counted; resolved ones still returned."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.02, 0.01, 5, "Done.")
        log.store_decision("AAPL", "2026-02-01", DECISION_BUY)  # left pending
        log.store_decision("MSFT", "2026-02-02", DECISION_SELL)  # left pending

        signals, stats = resolve_signals(str(tmp_path / "trading_memory.md"))

        assert stats.total_entries == 3
        assert stats.resolved == 1
        assert stats.skipped_pending == 2
        assert stats.skipped_malformed == 0
        assert [s.ticker for s in signals] == ["NVDA"]

    def test_malformed_resolved_skipped_and_counted(self, tmp_path):
        """A resolved entry whose raw won't parse is skipped as malformed, not raised."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.02, 0.01, 5, "Good.")
        # Hand-written resolved entry with an unparseable raw field.
        _append_raw(tmp_path, "[2026-01-20 | BAD | Sell | garbage | n/a | 5d]")

        signals, stats = resolve_signals(str(tmp_path / "trading_memory.md"))

        assert stats.total_entries == 2
        assert stats.resolved == 1
        assert stats.skipped_malformed == 1
        assert stats.skipped_pending == 0
        assert [s.ticker for s in signals] == ["NVDA"]

    def test_missing_alpha_is_none(self, tmp_path):
        """A resolved entry lacking alpha/holding fields -> raw kept, alpha/days None."""
        _append_raw(tmp_path, "[2026-01-21 | XYZ | Buy | +2.0%]")
        signals, stats = resolve_signals(str(tmp_path / "trading_memory.md"))
        assert stats.resolved == 1
        assert signals[0].raw_return == pytest.approx(0.02, abs=1e-9)
        assert signals[0].alpha_return is None
        assert signals[0].holding_days is None

    def test_accepts_prebuilt_memory_log(self, tmp_path):
        """resolve_signals accepts a pre-built TradingMemoryLog."""
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", DECISION_BUY)
        log.update_with_outcome("NVDA", "2026-01-05", 0.04, 0.02, 5, "Done.")
        signals, stats = resolve_signals(memory_log=log)
        assert stats.resolved == 1
        assert signals[0].raw_return == pytest.approx(0.04, abs=1e-9)

    def test_accepts_prebuilt_entries(self):
        """resolve_signals accepts a pre-parsed entries list (no file needed)."""
        entries = [
            {"date": "2026-01-05", "ticker": "NVDA", "rating": "Buy", "pending": False,
             "raw": "+3.2%", "alpha": "+1.1%", "holding": "5d"},
            {"date": "2026-02-01", "ticker": "AAPL", "rating": "Buy", "pending": True,
             "raw": None, "alpha": None, "holding": None},
            {"date": "2026-01-20", "ticker": "BAD", "rating": "Sell", "pending": False,
             "raw": "garbage", "alpha": None, "holding": None},
        ]
        signals, stats = resolve_signals(entries=entries)
        assert stats == ResolveStats(total_entries=3, resolved=1,
                                     skipped_pending=1, skipped_malformed=1)
        assert signals[0].ticker == "NVDA"
        assert signals[0].holding_days == 5

    def test_logs_skip_counts(self, tmp_path, caplog):
        """Skip counts are logged (no print, no silent truncation)."""
        entries = [
            {"date": "d", "ticker": "T", "rating": "Buy", "pending": True, "raw": None},
            {"date": "d", "ticker": "T", "rating": "Buy", "pending": False, "raw": "junk"},
        ]
        with caplog.at_level("INFO", logger="tradingagents.backtest.replay"):
            resolve_signals(entries=entries)
        assert "skipped pending" in caplog.text
        assert "skipped malformed" in caplog.text
