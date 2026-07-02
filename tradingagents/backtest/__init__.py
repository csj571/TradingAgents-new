"""Offline backtest surface for TradingAgents.

Phase 0 of the BRE integration plan (see ``docs/BRE_INTEGRATION_PLAN.md``): an
offline resolver that turns the accumulated decision log into dated, resolved
``(rating, realized-return)`` records for downstream calibration.

No LLM, no network, no price fetch — it stops at clean resolved records. It does
NOT binarize up/down and does NOT apply a rating->probability map (Phase 1).
"""

from tradingagents.backtest.replay import (
    ResolvedSignal,
    ResolveStats,
    resolve_signals,
)

__all__ = ["ResolvedSignal", "ResolveStats", "resolve_signals"]
