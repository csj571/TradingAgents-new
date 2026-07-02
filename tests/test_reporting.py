"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

import json
from types import SimpleNamespace

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "investment_debate_state": {"judge_decision": "RM PLAN"},
        "trader_investment_plan": "TRADE",
        "risk_debate_state": {"judge_decision": "PM DECISION"},
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert (tmp_path / "3_trading" / "trader.md").read_text() == "TRADE"
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = TradingAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = TradingAgentsGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")


@pytest.mark.unit
def test_save_reports_emits_signal_envelope(tmp_path):
    # On a real instance, save_reports also drops a valid signal envelope next
    # to the report tree (BRE integration plan, Phase 2). memory_log=None is fine.
    mock_self = SimpleNamespace(config={"emit_signal_envelope": True}, memory_log=None)
    # A real run's final_state carries the ticker/date the envelope describes.
    state = {**_state(), "company_of_interest": "AAPL", "trade_date": "2026-01-05"}
    TradingAgentsGraph.save_reports(mock_self, state, "AAPL", save_path=tmp_path)
    env_path = tmp_path / "signal_envelope.json"
    assert env_path.exists()
    env = json.loads(env_path.read_text())
    assert env["schema"] == "bre-signal-envelope"
    assert env["ticker"] == "AAPL"
    assert env["asof"] == "2026-01-05"
    assert env["evidence"]["validation"] is None  # downstream stage, not emitted here


@pytest.mark.unit
def test_save_reports_envelope_opt_out(tmp_path):
    mock_self = SimpleNamespace(config={"emit_signal_envelope": False}, memory_log=None)
    TradingAgentsGraph.save_reports(mock_self, _state(), "AAPL", save_path=tmp_path)
    assert not (tmp_path / "signal_envelope.json").exists()
    assert (tmp_path / "complete_report.md").exists()  # report tree still written


@pytest.mark.unit
def test_save_reports_unbound_skips_envelope(tmp_path):
    # Unbound (self=None) explicit-path call: report tree written, envelope skipped
    # (the serializer needs config/memory_log), and no crash.
    out = TradingAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert out == tmp_path / "complete_report.md"
    assert not (tmp_path / "signal_envelope.json").exists()
