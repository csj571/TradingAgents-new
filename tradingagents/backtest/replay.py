"""Offline replay resolver over the ``TradingMemoryLog``.

Phase 0 of the BRE integration plan. Reads the append-only decision log (already
keyed ``[date | ticker | rating | ...]`` with realized ``raw``/``alpha`` returns
on resolved entries) and yields dated, resolved ``(rating, realized-return)``
records — the replay artifact the calibration harness (Phase 1) will consume.

Scope discipline: this produces clean resolved records only. It deliberately does
NOT binarize up/down and does NOT apply a rating->probability map; those belong to
Phase 1. Fully offline: no LLM, no network, no price source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tradingagents.agents.utils.memory import TradingMemoryLog

logger = logging.getLogger(__name__)

# Values in the percentage fields that mean "no data" rather than a number.
_MISSING_TOKENS = {"", "n/a", "na", "none", "pending"}


@dataclass
class ResolvedSignal:
    """A single dated, resolved signal from the decision log.

    ``raw_return`` is required (a resolved entry whose raw field will not parse is
    treated as malformed and skipped). ``alpha_return`` and ``holding_days`` may be
    ``None`` when those fields are absent or unparseable.
    """

    date: str
    ticker: str
    rating: str
    raw_return: float
    alpha_return: float | None = None
    holding_days: int | None = None


@dataclass
class ResolveStats:
    """Counts describing what the resolver did with a batch of log entries."""

    total_entries: int = 0
    resolved: int = 0
    skipped_pending: int = 0
    skipped_malformed: int = 0


def _parse_pct(s: str | None) -> float | None:
    """Parse a percentage string like ``"+3.2%"`` -> ``0.032`` / ``"-1.0%"`` -> ``-0.01``.

    Tolerates a leading ``+``/``-`` sign, surrounding whitespace, and a trailing
    ``%``. Returns ``None`` for ``None``, empty/``"n/a"``-style tokens, or anything
    that will not parse as a float — never raises.
    """
    if s is None:
        return None
    text = s.strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    text = text.rstrip("%").strip()
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def _parse_holding(s: str | None) -> int | None:
    """Parse a holding-period string like ``"5d"`` -> ``5``. Returns ``None`` on failure."""
    if s is None:
        return None
    text = s.strip().lower()
    if text in _MISSING_TOKENS:
        return None
    text = text.rstrip("d").strip()
    try:
        return int(text)
    except ValueError:
        return None


def resolve_signals(
    log_path=None,
    *,
    memory_log: TradingMemoryLog | None = None,
    entries: list[dict] | None = None,
) -> tuple[list[ResolvedSignal], ResolveStats]:
    """Resolve a decision log into dated ``(rating, realized-return)`` records.

    Accepts exactly one source, in priority order:
      * ``entries`` — a pre-parsed list of entry dicts (``TradingMemoryLog`` shape);
      * ``memory_log`` — a pre-built ``TradingMemoryLog`` to ``load_entries()`` from;
      * ``log_path`` — a path to a decision-log markdown file (the default path).

    Pending entries are skipped and counted. Resolved entries whose ``raw`` field
    will not parse are skipped and counted as malformed (never raised on). Returns
    the resolved records plus a :class:`ResolveStats` summary. Fully offline.
    """
    if entries is None:
        if memory_log is None:
            memory_log = TradingMemoryLog({"memory_log_path": str(log_path)} if log_path else None)
        entries = memory_log.load_entries()

    stats = ResolveStats(total_entries=len(entries))
    resolved: list[ResolvedSignal] = []

    for entry in entries:
        if entry.get("pending"):
            stats.skipped_pending += 1
            continue

        raw_return = _parse_pct(entry.get("raw"))
        if raw_return is None:
            stats.skipped_malformed += 1
            continue

        resolved.append(
            ResolvedSignal(
                date=entry.get("date", ""),
                ticker=entry.get("ticker", ""),
                rating=entry.get("rating", ""),
                raw_return=raw_return,
                alpha_return=_parse_pct(entry.get("alpha")),
                holding_days=_parse_holding(entry.get("holding")),
            )
        )

    stats.resolved = len(resolved)

    if stats.skipped_pending or stats.skipped_malformed:
        logger.info(
            "resolve_signals: %d resolved, %d skipped pending, %d skipped malformed "
            "(of %d entries)",
            stats.resolved,
            stats.skipped_pending,
            stats.skipped_malformed,
            stats.total_entries,
        )

    return resolved, stats
