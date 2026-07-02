# BRE Integration Plan — TradingAgents (v0.3.0) as a stage in the unified system

**Status:** plan / roadmap. Nothing here is wired up yet — this document is the
buildable path, not a description of existing behaviour.

**Reads with:**
- [`UNIFIED-SYSTEM-DESIGN.md`](../UNIFIED-SYSTEM-DESIGN.md) — the cross-repo north
  star (Sense → Belief → Infer → Decide across three repos). Read it for *why*.
- `bre-sim/products/markets/CALIBRATION_ON_TAURIC.md` — the upstream proposal to
  score this repo's ratings with `bre-sim`'s `engine/calibration.py`. This plan is
  that proposal **reconciled to the code that actually exists in this repo**, plus
  the missing prerequisite it assumed.

---

## 1. This repo's role in the unified system

In the unified design, three repos compose into one decision system:

| Stage | Repo | Emits |
|---|---|---|
| **Sense + thesis** | **TradingAgents (this repo)** | a directional **thesis + prior**: a 5-tier rating plus the bull/bear debate spread standing in for epistemic uncertainty |
| **Belief + Decide** | `bre-sim` / BRE-1 | GP / BOCPD / entropy / acquisition + the ACT / DEFER / QUERY gate |
| **Validation** | `pinescript-optimization_engine` | walk-forward + Monte Carlo + a calibrated GO/NO-GO |

**TradingAgents is the Sense + thesis stage.** Its job in the union is to emit a
well-formed, calibrated signal that the BRE-1 decision spine can consume. The
first brick of that — the one with the highest leverage and the cheapest truth —
is **calibration of the ratings** (unified design §3.2, "one honesty ledger").
Markets is the one domain where the thesis's isomorphism is cheaply validatable,
because price resolves truth.

**Scope discipline (carried verbatim from the upstream proposal):** this measures
*calibration of the ratings*, nothing more. It does not upgrade *operational*
isomorphism into *validated* isomorphism, does not touch the agent prompts, and
does not claim the pipeline is "good" — only whether stated confidence matches hit
rate. Keep that framing in every phase below.

---

## 2. Reconciliation — what the upstream plan assumed vs. what exists here

`CALIBRATION_ON_TAURIC.md` was written against an assumed `backend/tradingagents/backtest/{signals,engine,metrics}.py`
layout. That layout **does not exist in this repo.** The real seams:

| Upstream plan assumed | Reality in this repo (v0.3.0) | Consequence |
|---|---|---|
| `backend/…/backtest/` package | **No `backtest/` module at all** — no `Backtester`, no `ReplaySignalSource`, no `metrics.py` | **Phase 0** below builds the minimal replacement |
| `RATINGS_5_TIER`, `parse_rating` | ✅ `tradingagents/agents/utils/rating.py` | reuse as-is |
| `Signal(date, ticker, rating)` replay source | Partially — `graph/signal_processing.py` `SignalProcessor.process_signal(text)` extracts the rating; **the dated replay artifact is the decision log, not a `Signal` class** | resolve from the decision log (below) |
| a structured decision carrying confidence | `agents/schemas.py` `PortfolioDecision` carries `rating`, `executive_summary`, `investment_thesis`, `price_target`, `time_horizon` — **no numeric confidence field** (only the Sentiment Analyst report has low/med/high) | forces the **fixed monotone tier→prob map**, option (a); option (b) "use PM confidence" is not available |
| realized returns from a backtester | ✅ `graph/trading_graph.py` already computes `(raw_return, alpha_return, …)` and `graph/reflection.py` reflects on `alpha_return` vs SPY — **per decision, online** | outcomes exist; they just aren't batched offline yet |
| `internal_labs.db` decision store | ✅ `agents/utils/memory.py` `TradingMemoryLog` — an append-only markdown log keyed `[date | ticker | rating | status]` with `DECISION:` / `REFLECTION:` blocks and `alpha_return` | **this is the replay artifact** the calibration harness reads |

**The load-bearing finding:** the decision log already accumulates
`(date, ticker, rating, alpha_return)` per resolved entry. So the calibration
harness does not need a full backtester — it needs an **offline resolver** that
turns those logged entries (plus a rating→probability map) into `(prob, outcome)`
pairs. That is Phase 0, and it is much smaller than the upstream plan feared.

---

## 3. The signal envelope, mapped to this repo

The unified design's keystone is one **signal envelope** (JSON) every stage
reads/writes. Here is how *this repo's* structured output maps onto it — the
mapping this repo owns:

| Envelope field | Source in this repo |
|---|---|
| `ticker`, `asof`, `timeframe` | run inputs (daily ticker decision → `timeframe: "1D"`) |
| `regime` | `dataflows` market context (unify later — unified §3.4) |
| `estimate` | forward-return proxy from the tier map (Phase 1) |
| `uncertainty.epistemic` | **bull/bear debate spread** (Research Manager) — the qualitative epistemic proxy |
| `uncertainty.aleatoric` | not yet emitted — TBD (BRE-1's job downstream) |
| `evidence.thesis` | `PortfolioDecision.investment_thesis` + bull/bear (`source: "tradingagents"`) |
| `confidence_raw` | tier→prob map applied to `PortfolioDecision.rating` |
| `confidence_calibrated` | output of the honesty ledger (Phase 1) |
| `action.decision` | BRE-1's gate downstream — **not this repo's call** |
| `provenance` | decision-log entry key `(date, ticker, rating)` + run seed |

**Design rule (from the unified design):** keep the qualitative thesis and the
quantitative validation as **separate evidence channels** feeding one posterior.
Do not pre-average them upstream of the gate.

---

## 4. Phased path

Aligned to the unified design's four phases (§6), made concrete for this repo.
Phase 0 is the prerequisite the upstream plan assumed away.

### Phase 0 — Offline replay resolver *(prerequisite — the missing backtest surface)*
The one thing that must exist before calibration can score anything.

- **Deliverable:** an offline module (proposed `tradingagents/backtest/` or
  `tradingagents/calibration/replay.py`) that reads the `TradingMemoryLog`
  (already `[date | ticker | rating]` + `alpha_return`) and yields dated
  `(rating, realized_forward_return)` records — **no LLM, no network.**
- Reuse `graph/trading_graph.py`'s existing realized-return / alpha computation as
  the outcome source so numbers match what the reflection loop already sees; do not
  introduce a second price source.
- Where log entries are still `pending` (outcome date not yet passed), skip them —
  and **log how many were skipped** (no silent truncation).
- **Done when:** a resolver produces a clean batch of dated `(rating, forward_return)`
  records from a replayed/accumulated decision log, fully offline, with a unit test
  on a synthetic log (mirror the existing offline test style; no keys/network).

### Phase 1 — Calibration ledger *(the `CALIBRATION_ON_TAURIC` realization; unified §3.2)*
- **Vendor** `bre-sim/engine/calibration.py` → `tradingagents/backtest/calibration.py`
  (numpy-only, torch-free; keep a docstring credit line pointing back to bre-sim so
  drift is visible). Add an offline unit test.
- **Rating → probability map:** a small explicit monotone table next to
  `RATINGS_5_TIER`, e.g. `Buy=0.75, Overweight=0.6, Hold=0.5, Underweight=0.4,
  Sell=0.25`. Only needs to be *monotone* — Platt/isotonic learn the real mapping.
- **Outcome:** binary per signal — `1` if forward return over a fixed horizon `h`
  is `> 0`, else `0`. **Hold** = directional abstention → **exclude from the
  up/down curve**, report its count separately.
- **Score & report:** assemble `(prob, outcome)` arrays → Brier + Murphy
  decomposition, ECE/MCE, reliability curve. Fit `PlattScaler` / `IsotonicCalibrator`
  on an **earlier temporal slice**, score a later one, and report *recalibrated* ECE
  too — never score raw confidence as if calibrated.
- **Surface:** a standalone offline report
  (`python -m tradingagents.backtest.calibration_report --log <path>`) printing a
  compact table; add the reliability-curve arrays to structured output for a later
  frontend diagram.
- **Gate:** expose `passes_ece_gate(ece, max_ece=0.10)` as a **non-fatal warning**,
  not a launch gate — this is a research signal in a trading context.
- **Done when:** the report prints Brier, ECE (raw + recalibrated), and a reliability
  curve from a replayed log, fully offline, with ≥ 50 directional signals required
  before a curve is drawn (below that: raw Brier + "sample too small" note).

### Phase 2 — Emit the signal envelope *(unified §2, §3.5)*
- Add an adapter that serializes each `PortfolioDecision` (+ bull/bear spread + the
  calibrated confidence from Phase 1) into the signal-envelope JSON of §3.
- Keep it a **thin adapter** around the graph output — this repo is an upstream
  fork (Tauric), so integrate around `TradingAgentsGraph.propagate()`, not a deep
  rewrite, so upstream updates keep flowing.
- **Done when:** a completed run emits a valid, versioned signal envelope alongside
  its existing report tree.

### Phase 3 — Shared services *(unified §3.3, §3.4, §3.6)*
- **Regime:** unify `dataflows` market context with BRE-1 BOCPD into one regime
  primitive; a change-point resets the calibration reference.
- **Acquisition:** turn "research depth" into an information-gain scheduler driven
  by BRE-1's acquisition (`EI/UCB/BALD/MES`).
- **Memory:** fold `TradingMemoryLog` into the event-sourced provenance store keyed
  `(ticker, date, regime)`.

### Phase 4 — Cockpit *(unified §3.5)*
- Repoint the BRE-1 web app at these live services; render debate transcripts,
  reliability curves, and the decision log in the HumanityAGSI dark/parchment brand.

---

## 5. Decisions — locked and open

**Locked by this plan:**
- Missing backtest surface is a real prerequisite → **Phase 0**, built as an offline
  resolver over the existing decision log (not a full backtester).
- Confidence source is the **fixed monotone tier map** (option a) — `PortfolioDecision`
  carries no numeric confidence field, so option (b) is unavailable today.
- **Vendor** `calibration.py` (default) rather than cross-repo import — this repo
  can't `pip install` the research monorepo, and the module is numpy-only.

**Open (answer before coding the relevant phase):**
- Horizon `h`, and whether it ties to the run's rebalance cadence.
- Whether Phase 0 lands as `tradingagents/backtest/` or `tradingagents/calibration/`.
- Whether a numeric `confidence` field should later be *added* to `PortfolioDecision`
  (would upgrade the tier map to option b) — a prompt/schema change, out of scope now.

---

## 6. Honesty checks (apply to every phase)

- **Enough samples.** ECE over a handful of trades is noise — require ≥ 50 directional
  signals before drawing a curve.
- **No look-ahead.** The horizon-`h` outcome uses only post-signal data; fit
  recalibrators on an earlier slice, score a later one (temporal split, never random).
- **Don't overclaim transfer.** This validates *these ratings on this price history* —
  nothing about a different date range, and nothing about the minds/HRV domain. Markets
  is the cheap-truth domain; keep the claim scoped to it.
- **Two evidence channels, one posterior.** Thesis and validation measure different
  things; never collapse them into one averaged number upstream of the gate.
</content>
</invoke>
