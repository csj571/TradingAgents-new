"""Tests for the Phase 2 signal-envelope serializer (tradingagents/signal_envelope.py).

Fully offline and deterministic: no API keys, no network, no price source. History
for the calibration path is seeded authentically through TradingMemoryLog
(store_decision + update_with_outcome) so these tests also guard against
decision-log format drift feeding the Phase 0 resolver.
"""

import json

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.signal_envelope import (
    DEFAULT_MIN_HISTORY,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    _parse_sentiment,
    build_envelope,
    write_envelope,
)

# --- Synthetic run fixtures -------------------------------------------------

PM_DECISION_BUY = "Final Decision\n\nRating: Buy\n\nEnter a starter position at $189-192."

SENTIMENT_REPORT = "\n".join([
    "**Overall Sentiment:** **Bullish** (Score: 6.5/10)",
    "**Confidence:** High",
    "",
    "News flow skews positive on strong guidance.",
])


def make_final_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-06-25",
        "final_trade_decision": PM_DECISION_BUY,
        "investment_plan": "Research manager plan: accumulate on strength.",
        "investment_debate_state": {
            "bull_history": "Bull: datacenter demand accelerating, margins expanding.",
            "bear_history": "Bear: valuation stretched, competition rising.",
            "judge_decision": "Judge: the bull case is stronger; lean long with sizing discipline.",
        },
        "sentiment_report": SENTIMENT_REPORT,
    }


def make_log(tmp_path, filename="trading_memory.md"):
    return TradingMemoryLog({"memory_log_path": str(tmp_path / filename)})


def _seed_resolved(log, ticker, n, start_day=1):
    """Append n resolved directional entries (alternating Buy/Sell, mixed outcomes)."""
    for i in range(n):
        day = start_day + i
        date = f"2026-01-{day:02d}"
        if i % 2 == 0:
            decision = "Rating: Buy\nEnter now."
            raw = 0.03 if i % 3 else -0.02  # mix wins/losses
        else:
            decision = "Rating: Sell\nExit now."
            raw = -0.03 if i % 3 else 0.02
        log.store_decision(ticker, date, decision)
        log.update_with_outcome(ticker, date, raw, raw - 0.005, 5, "reflection.")


# --- Envelope shape on a full synthetic run ---------------------------------

class TestBuildEnvelopeFull:

    def test_schema_and_top_level(self):
        env = build_envelope(make_final_state())
        assert env["schema"] == SCHEMA_NAME == "bre-signal-envelope"
        assert env["schema_version"] == SCHEMA_VERSION == "0.1.0"
        assert env["ticker"] == "NVDA"
        assert env["asof"] == "2026-06-25"
        assert env["timeframe"] == "1D"
        assert env["regime"] is None

    def test_rating_confidence_estimate(self):
        env = build_envelope(make_final_state())
        assert env["evidence"]["thesis"]["rating"] == "Buy"
        assert env["confidence_raw"] == 0.75
        assert env["estimate"] == 0.25

    def test_thesis_channels(self):
        thesis = build_envelope(make_final_state())["evidence"]["thesis"]
        assert "datacenter" in thesis["bull"]
        assert "valuation" in thesis["bear"]
        assert thesis["research_manager"].startswith("Judge:")
        assert thesis["final_decision"] == PM_DECISION_BUY
        assert thesis["source"] == "tradingagents"

    def test_sentiment_parsed(self):
        sentiment = build_envelope(make_final_state())["evidence"]["thesis"]["sentiment"]
        assert sentiment == {"band": "Bullish", "score": 6.5, "confidence": "High"}

    def test_downstream_fields_none(self):
        env = build_envelope(make_final_state())
        assert env["evidence"]["validation"] is None
        assert env["action"]["decision"] is None
        assert env["action"]["gate"] is None
        assert env["uncertainty"]["aleatoric"] is None
        assert env["uncertainty"]["epistemic"] is None
        assert env["uncertainty"]["note"]

    def test_timeframe_param(self):
        env = build_envelope(make_final_state(), timeframe="1H")
        assert env["timeframe"] == "1H"

    def test_provenance_session_and_seed(self):
        env = build_envelope(make_final_state(), config={"seed": 123456})
        assert env["provenance"]["session"] == "NVDA:2026-06-25"
        assert env["provenance"]["seed"] == 123456
        assert env["provenance"]["parents"] == []


# --- Empty / degenerate input ----------------------------------------------

class TestBuildEnvelopeEmpty:

    def test_empty_final_state_no_raise(self):
        env = build_envelope({})
        assert env["schema"] == SCHEMA_NAME
        assert env["ticker"] is None
        assert env["asof"] is None
        assert env["evidence"]["thesis"]["rating"] is None
        assert env["confidence_raw"] is None
        assert env["estimate"] is None
        assert env["evidence"]["thesis"]["sentiment"] is None
        assert env["confidence_calibrated"] is None
        assert env["confidence_calibration_note"] == "no rating to calibrate"

    def test_none_final_state_no_raise(self):
        env = build_envelope(None)
        assert env["ticker"] is None
        assert env["confidence_raw"] is None


# --- _parse_sentiment -------------------------------------------------------

class TestParseSentiment:

    def test_valid_header(self):
        assert _parse_sentiment(SENTIMENT_REPORT) == {
            "band": "Bullish",
            "score": 6.5,
            "confidence": "High",
        }

    def test_multiword_band(self):
        md = "**Overall Sentiment:** **Mildly Bearish** (Score: 3.5/10)\n**Confidence:** Low"
        assert _parse_sentiment(md) == {
            "band": "Mildly Bearish",
            "score": 3.5,
            "confidence": "Low",
        }

    def test_missing_confidence_line_returns_none(self):
        md = "**Overall Sentiment:** **Bullish** (Score: 6.5/10)"
        assert _parse_sentiment(md) is None

    def test_garbage_returns_none(self):
        assert _parse_sentiment("not a sentiment report at all") is None

    def test_none_and_empty_return_none(self):
        assert _parse_sentiment(None) is None
        assert _parse_sentiment("") is None


# --- confidence_calibrated path --------------------------------------------

class TestCalibration:

    def test_platt_recalibrated_with_history(self, tmp_path):
        log = make_log(tmp_path)
        _seed_resolved(log, "AAPL", 12)  # 12 resolved directional signals
        env = build_envelope(make_final_state(), memory_log=log, min_history=5)
        cal = env["confidence_calibrated"]
        assert isinstance(cal, float)
        assert 0.0 <= cal <= 1.0
        assert "Platt-recalibrated" in env["confidence_calibration_note"]

    def test_insufficient_history_returns_none(self, tmp_path):
        log = make_log(tmp_path)
        _seed_resolved(log, "AAPL", 3)  # below min_history
        env = build_envelope(make_final_state(), memory_log=log, min_history=5)
        assert env["confidence_calibrated"] is None
        assert "insufficient history" in env["confidence_calibration_note"]

    def test_default_min_history_constant(self):
        assert DEFAULT_MIN_HISTORY == 50


# --- provenance.parents -----------------------------------------------------

class TestProvenanceParents:

    def test_prior_same_ticker_entries(self, tmp_path):
        log = make_log(tmp_path)
        log.store_decision("NVDA", "2026-06-01", "Rating: Buy\nEnter.")
        log.update_with_outcome("NVDA", "2026-06-01", 0.03, 0.02, 5, "ok.")
        log.store_decision("NVDA", "2026-06-10", "Rating: Sell\nExit.")
        log.update_with_outcome("NVDA", "2026-06-10", -0.01, -0.02, 5, "ok.")
        # A different ticker must not appear.
        log.store_decision("AAPL", "2026-06-05", "Rating: Buy\nEnter.")
        log.update_with_outcome("AAPL", "2026-06-05", 0.01, 0.0, 5, "ok.")

        env = build_envelope(make_final_state(), memory_log=log)
        parents = env["provenance"]["parents"]
        assert parents == [
            "2026-06-10|NVDA|Sell",
            "2026-06-01|NVDA|Buy",
        ]

    def test_current_run_excluded_and_capped(self, tmp_path):
        log = make_log(tmp_path)
        # Seven prior NVDA entries plus one matching the current run's (date,ticker).
        for i in range(7):
            date = f"2026-05-{i + 1:02d}"
            log.store_decision("NVDA", date, "Rating: Buy\nEnter.")
            log.update_with_outcome("NVDA", date, 0.02, 0.01, 5, "ok.")
        # Current pending run entry (same date/ticker as make_final_state()).
        log.store_decision("NVDA", "2026-06-25", "Rating: Buy\nEnter.")

        env = build_envelope(make_final_state(), memory_log=log)
        parents = env["provenance"]["parents"]
        assert len(parents) == 5  # capped
        assert all("2026-06-25" not in p for p in parents)  # current excluded


# --- write_envelope ---------------------------------------------------------

class TestWriteEnvelope:

    def test_roundtrip(self, tmp_path):
        env = build_envelope(make_final_state(), config={"seed": 7})
        path = write_envelope(env, tmp_path / "out")
        assert path.name == "signal_envelope.json"
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == env

    def test_creates_parent_dirs(self, tmp_path):
        env = build_envelope({})
        path = write_envelope(env, tmp_path / "a" / "b" / "c")
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA_NAME
