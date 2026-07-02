"""Phase 2 signal-envelope serializer — TradingAgents' slice of the BRE contract.

Turns a completed run's ``final_state`` dict into the versioned **BRE signal
envelope** (see ``docs/BRE_INTEGRATION_PLAN.md`` §3 and ``UNIFIED-SYSTEM-DESIGN.md``
§2). This repo is the *Sense + thesis* stage: it owns the thesis text, the tier
map's directional edge, and the calibrated confidence (reusing the Phase 0/1
resolver + Platt scaler). Everything the downstream BRE-1 decision spine owns —
the epistemic/aleatoric split, the regime label, the ACT/DEFER/QUERY gate — is
emitted as ``None`` with a note, never invented here.

Fully offline: no LLM, no network, no price source, no torch/pandas/yfinance. It
reads only ``final_state`` (a plain dict), an optional ``TradingMemoryLog`` (for
calibration history + provenance parents), and the repo's numpy-only calibration
core.

Design rule (from the unified design): keep the qualitative thesis and the
quantitative validation as **separate evidence channels** — do not pre-average
them upstream of the gate. So ``evidence.thesis`` carries the debate text
verbatim and ``evidence.validation`` stays ``None`` (a separate, absent stage).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.backtest.calibration import PlattScaler
from tradingagents.backtest.calibration_report import rating_to_prob, signals_to_pairs
from tradingagents.backtest.replay import resolve_signals

SCHEMA_NAME = "bre-signal-envelope"
SCHEMA_VERSION = "0.1.0"

# Minimum resolved-history size before Platt recalibration is trusted (mirrors
# the calibration report's DEFAULT_MIN_SAMPLES honesty check).
DEFAULT_MIN_HISTORY = 50

# Cap on provenance parents (prior same-ticker log entries, most-recent-first).
_MAX_PARENTS = 5

_UNCERTAINTY_NOTE = (
    "epistemic/aleatoric split assigned downstream by BRE-1; the bull/bear debate "
    "text is carried in evidence.thesis for it to quantify"
)
_ACTION_RATIONALE = (
    "assigned downstream by the BRE-1 decision gate; this stage emits thesis + "
    "confidence only"
)

# Sentiment header regexes, matched against the render_sentiment_report shape:
#   **Overall Sentiment:** **Bullish** (Score: 6.5/10)
#   **Confidence:** High
_SENTIMENT_RE = re.compile(
    r"\*\*Overall Sentiment:\*\*\s*\*\*(?P<band>.+?)\*\*\s*"
    r"\(Score:\s*(?P<score>[-+]?[0-9]*\.?[0-9]+)\s*/\s*10\)"
)
_CONFIDENCE_RE = re.compile(r"\*\*Confidence:\*\*\s*(?P<confidence>\w+)")


def _parse_sentiment(md: str | None) -> dict | None:
    """Best-effort parse of the sentiment-report header into a structured dict.

    Returns ``{"band", "score", "confidence"}`` (score as ``float``) when both
    header lines are present and parseable, else ``None``. Never raises: any
    missing or malformed piece yields ``None``.
    """
    if not md:
        return None
    m = _SENTIMENT_RE.search(md)
    c = _CONFIDENCE_RE.search(md)
    if m is None or c is None:
        return None
    try:
        score = float(m.group("score"))
    except (TypeError, ValueError):
        return None
    return {
        "band": m.group("band").strip(),
        "score": score,
        "confidence": c.group("confidence").strip(),
    }


def _calibrate(
    p_up: float | None,
    rating: str | None,
    *,
    memory_log: TradingMemoryLog | None,
    config: dict | None,
    min_history: int,
) -> tuple[float | None, str]:
    """Platt-recalibrate ``p_up`` on resolved history; return (calibrated, note).

    History source (no look-ahead — ``resolve_signals`` already excludes pending):
    a provided ``memory_log``, else a ``config["memory_log_path"]``, else none.
    Returns ``None`` (with a human-readable reason) when there is no rating to
    calibrate or insufficient resolved history.
    """
    if rating is None or p_up is None:
        return None, "no rating to calibrate"

    if memory_log is not None:
        history, _ = resolve_signals(memory_log=memory_log)
    elif config and config.get("memory_log_path"):
        history, _ = resolve_signals(log_path=config["memory_log_path"])
    else:
        history = []

    probs, outcomes, _ = signals_to_pairs(history)
    n = len(probs)
    if n < min_history:
        return None, f"insufficient history: {n}<{min_history}"

    scaler = PlattScaler().fit(probs, outcomes)
    result = scaler.transform(np.asarray([p_up], dtype=float))
    return round(float(result[0]), 4), f"Platt-recalibrated on {n} resolved signals"


def _parents(
    memory_log: TradingMemoryLog | None, ticker: str | None, asof: str | None
) -> list[str]:
    """Prior same-ticker log entries as ``"date|ticker|rating"``, most-recent-first.

    Excludes the current run's own ``(date, ticker)`` entry (the pending one) and
    caps the list at :data:`_MAX_PARENTS`. Returns ``[]`` when no memory log.
    """
    if memory_log is None or ticker is None:
        return []
    parents: list[str] = []
    for e in reversed(memory_log.load_entries()):
        if e.get("ticker") != ticker:
            continue
        if e.get("date") == asof and e.get("ticker") == ticker:
            continue  # current pending run — not its own parent
        parents.append(f"{e.get('date')}|{e.get('ticker')}|{e.get('rating')}")
        if len(parents) >= _MAX_PARENTS:
            break
    return parents


def build_envelope(
    final_state: dict,
    *,
    config: dict | None = None,
    memory_log: TradingMemoryLog | None = None,
    timeframe: str = "1D",
    min_history: int = DEFAULT_MIN_HISTORY,
) -> dict:
    """Serialize a completed run's ``final_state`` into the BRE signal envelope.

    Pure and offline. Fills only the fields this (Sense + thesis) stage honestly
    owns — thesis text, tier-map directional edge (``estimate``), raw and
    Platt-calibrated confidence — and leaves the downstream BRE-1 fields (regime,
    uncertainty split, action gate, validation channel) as ``None`` with notes.

    Defensive throughout: a completely empty ``final_state`` returns a valid
    envelope of mostly ``None`` values without raising.
    """
    fs = final_state or {}

    ticker = fs.get("company_of_interest") or None
    asof = fs.get("trade_date") or None

    decision_md = fs.get("final_trade_decision")
    rating = parse_rating(decision_md) if decision_md else None
    p_up = rating_to_prob(rating) if rating is not None else None
    estimate = round(p_up - 0.5, 4) if p_up is not None else None

    debate = fs.get("investment_debate_state") or {}
    bull = debate.get("bull_history") or None
    bear = debate.get("bear_history") or None
    research_manager = debate.get("judge_decision") or fs.get("investment_plan") or None

    sentiment = _parse_sentiment(fs.get("sentiment_report"))

    confidence_calibrated, calibration_note = _calibrate(
        p_up,
        rating,
        memory_log=memory_log,
        config=config,
        min_history=min_history,
    )

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "ticker": ticker,
        "asof": asof,
        "timeframe": timeframe,
        "regime": None,
        "estimate": estimate,
        "uncertainty": {
            "aleatoric": None,
            "epistemic": None,
            "note": _UNCERTAINTY_NOTE,
        },
        "evidence": {
            "thesis": {
                "rating": rating,
                "bull": bull,
                "bear": bear,
                "research_manager": research_manager,
                "final_decision": decision_md or None,
                "sentiment": sentiment,
                "source": "tradingagents",
            },
            "validation": None,
        },
        "confidence_raw": p_up,
        "confidence_calibrated": confidence_calibrated,
        "confidence_calibration_note": calibration_note,
        "action": {
            "decision": None,
            "gate": None,
            "rationale": _ACTION_RATIONALE,
        },
        "provenance": {
            "session": f"{ticker}:{asof}",
            "seed": config.get("seed") if config else None,
            "parents": _parents(memory_log, ticker, asof),
        },
    }


def write_envelope(envelope: dict, save_path) -> Path:
    """Write ``envelope`` to ``signal_envelope.json`` under ``save_path``.

    Creates ``save_path`` (parents included) and returns the written file path.
    Coerces the two numeric fields to plain ``float`` in case a numpy scalar
    sneaks in, so the JSON stays clean/portable.
    """
    out = dict(envelope)
    if out.get("estimate") is not None:
        out["estimate"] = float(out["estimate"])
    if out.get("confidence_calibrated") is not None:
        out["confidence_calibrated"] = float(out["confidence_calibrated"])

    directory = Path(save_path)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / "signal_envelope.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return file_path
