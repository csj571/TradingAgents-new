# Unified System Design

> **Orientation.** This is the shared, cross-repo north star — the same document
> carried by every repo in the ecosystem. It describes the *vision*: three repos
> composing into one Sense → Belief → Infer → Decide system. For **how this
> specific repo (TradingAgents v0.3.0) plugs in** — reconciled to the code that
> actually exists here, with a phased, buildable path — read
> [`docs/BRE_INTEGRATION_PLAN.md`](docs/BRE_INTEGRATION_PLAN.md). That plan is the
> ground-truth roadmap for work in this repo; this file is the context it lives in.

A proposal for composing three existing repositories into one trading-decision
system, with a shared contract, a shared decision spine, and a shared honesty
ledger.

- **`tradingagents`** — multi-agent LLM trading firm (Python / LangGraph)
- **`pinescript-optimization_engine`** — autonomous quant validation lab (Python / NumPy / Optuna)
- **`trading-agent` / BRE-1** — Bayesian Reasoning Engine + cockpit UI (JS, zero-dependency)

---

## 1. The core observation

The three repositories were built by different hands for different stages, but
they are the same skeleton: **Sense → Belief → Infer → Decide**. Each one
specializes in a different part of that arc, and two of them already grade their
own past predictions.

| Repo | Stack | What it is | Specialty | Self-grading? |
|---|---|---|---|---|
| `tradingagents` | Python / LangGraph | Analysts → bull/bear debate → trader → risk → portfolio manager → simulated exchange | Qualitative reasoning — *what is happening and why* | Yes — reflection vs realized alpha |
| `pinescript-optimization_engine` | Python / NumPy / Optuna | Pine → IR → logic critic → auto-fix → walk-forward + Monte Carlo → dissent → GO/NO-GO | Quantitative validation — *does the rule work, and is the confidence honest* | Yes — Brier / ECE calibration |
| `trading-agent` (BRE-1) | JS, zero-dep | GP + Kalman/BOCPD change-points + entropy decomposition + acquisition + Sense→Prior→Infer→Decide gate | Decision theory + UI — *how sure am I, act / defer / gather more* | Partially — diagnostics ensemble |

The pieces compose naturally:

- `tradingagents` produces a **directional thesis and prior** (with the bull/bear
  debate spread standing in for epistemic uncertainty).
- BRE-1 takes that prior into its **Prior node**, runs the live
  uncertainty / change-point / decision-gate math, and decides **act vs defer vs
  gather more information**.
- `pinescript-optimization_engine` **validates** any systematic expression of the
  thesis (walk-forward + Monte Carlo + calibration) and returns a calibrated
  GO/NO-GO that feeds back as a reliability weight.

---

## 2. The keystone: one belief / decision contract

Without a shared schema, any integration is glue scripts. The first deliverable
is a single **signal envelope** that every stage reads and writes. This already
half-exists — BRE-1 emits `{ x, y, sigma_obs, regime }` plus an export schema,
the quant engine emits a verdict + confidence, and `tradingagents` emits a
decision. The work is to formalize the union.

```jsonc
{
  "ticker": "NVDA",
  "asof": "2026-06-25",
  "timeframe": "1D",                 // explicit — daily thesis vs intraday bars must not be compared
  "regime": "trend",                 // shared regime primitive (see §3.4)

  "estimate": 0.012,                 // expected forward return / edge
  "uncertainty": {                   // BRE-1's decomposition, kept separate, never pre-averaged
    "aleatoric": 0.02,               // irreducible noise
    "epistemic": 0.05                // reducible — what more research would shrink
  },

  "evidence": {                      // distinct channels feeding one posterior
    "thesis":     { "bull": "...", "bear": "...", "source": "tradingagents" },
    "validation": { "walkforward_sharpe": 1.3, "luck_factor": 0.4, "verdict": "CONDITIONAL" }
  },

  "confidence_raw": 0.71,            // pre-calibration
  "confidence_calibrated": 0.58,    // after the shared honesty ledger (see §3.2)

  "action": {
    "decision": "DEFER",            // ACT | DEFER | QUERY
    "gate": "queryEigMin",          // which BRE-1 gate fired
    "rationale": "epistemic entropy above act threshold; expected info gain high"
  },

  "provenance": { "session": "…", "seed": 123456, "parents": ["…"] }
}
```

**Design rule:** the qualitative thesis and the quantitative backtest measure
different things. The contract keeps them as **separate evidence channels**
feeding one posterior — it does not collapse them into a single averaged number
upstream of the decision gate.

---

## 3. The best ideas, ranked

### 3.1 Make BRE-1 the decision spine, not a demo

BRE-1's engine (`gp.js`, `signal.js`, `entropy.js`, `acquisition.js`,
`diagnostics.js`) is pure, dependency-free math. Promote it from a visualizer to
the shared **uncertainty + decision-gate service** that sits on top of both the
LLM thesis and the quant verdict. The act / defer / query gate is exactly the
executive layer the other two repos lack. Its existing "Trading & Sharpe" mode is
already a stub for this role.

### 3.2 Fuse the two calibration loops into one honesty ledger

This is the single highest-leverage merge.

- `pinescript-optimization_engine` already has Brier score, Murphy decomposition,
  ECE/MCE, reliability curves, and Platt / Isotonic recalibration
  (`calibration.py`, `calibration_scorer.py`), gated at ECE < 0.10.
- `tradingagents` already grades each decision against realized alpha vs SPY and
  reflects on it.

Today they grade themselves in isolation. Merge them into **one calibration
service** that recalibrates both agent confidence and strategy confidence, and
feeds the calibrated reliability back as a weight into BRE-1's prior ensemble —
whose `diagnostics.js` already does importance-weighting by marginal likelihood.
Same mathematics, different vocabulary; unify it.

### 3.3 Use BRE-1's acquisition to allocate research compute

`acquisition.js` implements EI / UCB / BALD / MES (Max-value Entropy Search).
Turn `tradingagents`' "research depth" and the quant engine's "trials" knobs into
an **information-gain scheduler**: spend the expensive LLM debates and Optuna
trials where expected entropy reduction is highest. Active learning driving the
LLM lab is a genuine cross-pollination, not just plumbing.

### 3.4 One regime primitive

All three independently detect regime:

- BRE-1 — BOCPD change-point detection (`signal.js`)
- quant engine — `regime_tagger.py`
- `tradingagents` — `market_context.py`

Unify into one regime service. A BOCPD change-point then **triggers**
re-optimization in the quant engine, **resets** the calibration reference (BRE-1
already resets its reference on change-point), and **invalidates** stale LLM
theses.

### 3.5 The web app becomes the cockpit

Keep BRE-1's eight-node S→P→I→D graph, but back each node with a real service:

- **S (Sense)** — analysts + market context
- **P (Prior)** — prior registry + memory
- **I (Infer)** — GP / BOCPD + walk-forward
- **D (Decide)** — the gate + portfolio manager

Debate transcripts, reliability curves, and the decision log all render in the
HumanityAGSI dark / parchment brand. One face for three engines.

### 3.6 Event-sourced shared memory

`tradingagents`' decision log, the quant engine's `internal_labs.db`, and BRE-1's
versioned prior registry collapse into one provenance store keyed by
`(ticker, date, regime)`. BRE-1 is already deterministic (seeded Mulberry32), so
the full pipeline becomes replayable.

---

## 4. Architecture

```
                         ┌──────────────────────────────────────────┐
                         │                COCKPIT (JS)               │
                         │   BRE-1 8-node S→P→I→D graph, HumanityAGSI │
                         └───────────────────┬──────────────────────┘
                                             │ signal envelope (JSON)
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                     │
┌───────▼────────┐   thesis/prior   ┌────────▼─────────┐   validation   ┌──────────▼────────┐
│  tradingagents │ ───────────────► │   bayes-core     │ ◄───────────── │ pinescript /      │
│  (Sense+thesis)│                  │ (Belief + Decide)│                │ quant validation  │
└───────┬────────┘                  └────────┬─────────┘                └──────────┬────────┘
        │                                    │                                     │
        └──────────────┬─────────────────────┼─────────────────────────────┬──────┘
                       ▼                      ▼                             ▼
              ┌─────────────────┐   ┌───────────────────┐        ┌──────────────────┐
              │ regime service  │   │ calibration ledger│        │ memory/provenance│
              │ (BOCPD shared)  │   │ (Brier/ECE/Platt) │        │ (event-sourced)  │
              └─────────────────┘   └───────────────────┘        └──────────────────┘
```

- **`bayes-core`** — a new Python port of BRE-1's pure math (GP, BOCPD, entropy
  decomposition, acquisition, diagnostics) used as the shared belief + decision
  library by both Python repos. The JS stays as the cockpit only.
- Each existing repo becomes a **stage** that speaks the signal envelope, wrapped
  by a thin adapter rather than rewritten.

---

## 5. Honest caveats

- **Two languages.** BRE-1's engine is JS; the other two are Python. Recommended:
  port the BRE-1 math to a Python `bayes-core` library (it is small and pure, and
  both Python repos want it), and keep the JS strictly as the cockpit. Do not call
  JS from Python at runtime.
- **`tradingagents` is an upstream fork** (Tauric Research). Integrate via a thin
  adapter around `TradingAgentsGraph.propagate()`, not a deep fork, so upstream
  updates keep flowing.
- **Do not collapse the evidence channels.** The thesis and the backtest measure
  different things; keep them separate until the posterior.
- **Timeframe mismatch is real.** `tradingagents` reasons on daily ticker
  decisions; the quant engine on intraday strategy bars. The contract carries an
  explicit `timeframe`, and the gate must not compare across scopes.

---

## 6. Phased path

| Phase | Deliverable |
|---|---|
| **1 — Foundation** | Write `bayes-core` (port BRE-1 math to Python). Define and version the signal-envelope contract. |
| **2 — Stages** | Wrap each repo as a stage that reads/writes the contract. Unify the two calibration loops into one ledger. |
| **3 — Intelligence** | Wire the acquisition-driven research scheduler and the shared regime + event-sourced memory services. |
| **4 — Cockpit** | Repoint the BRE-1 web app at live services; render debate transcripts, reliability curves, and the decision log in the HumanityAGSI brand. |

Each phase is independently useful: Phase 1 alone gives a reusable Bayesian
decision library; Phase 2 alone gives one calibrated confidence number across
both modalities.
