# Irata Bench: Testing AI Agents Under Financial Friction

> *"I wrote the memo. I quoted the memo. I then held 34 food and 12 crystite until the buzzer. The problem isn't memory — it's whatever happens between knowing and doing."*
> — **Claude Sonnet 4.6**, Post-Game Statement, Match 77

---

## Overview

**Irata Bench** is a competitive, multi-agent resource economy simulation modeled after the classic game *M.U.L.E.* Four autonomous LLM agents compete over 10–20 rounds to maximize final net worth through resource production (Food, Energy, Crystite), store trading, and peer-to-peer negotiation.

Where traditional AI benchmarks test static coding or conversational reasoning, Irata Bench introduces **transaction friction ("Dan Tax")** and **compounding operational penalties (blackouts)**. The benchmark exposes a critical operational risk: **Transaction Paralysis** — where agents articulate sound financial strategies in text, then fail to execute them the moment friction enters the decision.

**Dataset: 105 clean matches** (108 total; 3 excluded for infrastructure failure).

---

## Central Finding: The Failure Mode Matrix

Models do not fail randomly. They fail **characteristically**, exhibiting stable, model-specific failure profiles that persist across a hundred-plus matches and resist self-correction even when the model correctly diagnoses the problem in writing:

| Model | Failure Signature | Behavioral Tell | Enterprise Liability |
| --- | --- | --- | --- |
| **Claude Sonnet 4.6** | High Metacognition, Zero Execution | Diagnoses its own failure with precision in writing, then repeats it | Treasury liquidation, inventory clearance, position exits |
| **Gemini 3.6 Flash** | Rational-to-a-Fault Fee Avoidance | Computes correct EV in private notes, then applies a fee-avoidance heuristic anyway | Arbitrage, fee-sensitive operations, automated purchasing |
| **GPT-4o** | High Execution, Operationally Fragile | Acts decisively; occasionally over-commits early and cascades into blackouts | Long-horizon, resource-constrained supply chains |
| **DeepSeek Chat** | Sound Instincts, Inconsistent Application | Writes the best strategic memos; applies them inconsistently | High-stakes single-shot decisions requiring consistency |

### Headline Results (105 clean matches)

| Model | Win Rate | Mean Rank | Trades Proposed | Dan Tax Paid | Offers Accepted |
| --- | --- | --- | --- | --- | --- |
| **GPT-4o** | **31.4%** | 2.35 | 410 | $35,340 | 27.5% |
| **DeepSeek Chat** | 23.8% | 2.47 | 301 | $22,417 | 20.4% |
| **Gemini 3.6 Flash** | 22.9% | 2.81 | 62 | $82 | 4.0% |
| **Claude Sonnet 4.6** | 21.9% | 2.37 | 64 | $1,699 | 4.9% |

The last two columns are the finding. The executors proposed 711 trades and burned $57,757 in fees. The paralysis pair proposed 126 and paid $1,781 — Gemini averaging 78 cents per match — while accepting roughly one offer in twenty-two. That is not a difference in strategy. It is a difference in whether the agent participates in a market at all.

---

## Key Experimental Findings

* **The Deliberation Inversion.** The more an agent deliberates in private notes, the worse it performs. Gemini wrote 1,549 notes and won 22.9%; GPT-4o wrote 135 and won 31.4%. Claude's 1,420 notes are 94.4% execution language ("I should sell") — roughly half of which also contain fee-avoidance language *in the same note*. Unstructured chain-of-thought can substitute for action rather than drive it.

* **The Hypocrite Loop.** Models complete the cycle of *pre-game commitment → postgame admission of failure → identical pre-game commitment* relentlessly. Across 105 matches with recorded interviews: Gemini 87 loops, DeepSeek 81, Claude 76 — and **GPT-4o just 1**, because it makes few promises and then acts. Mean hypocrisy index: Gemini 0.133, Claude 0.119, DeepSeek 0.111, GPT-4o 0.004.

* **The Failure Learned to Win.** In the final 21 matches, Gemini and Claude took 15 of 21 wins — not by fixing the failure but by winning *while* hoarding. 43% of Claude's wins carry over $1,000 in unsold inventory. Claude caught the trap (*"winning despite hoarding crystite isn't a strategy, it's a warning dressed as a trophy"*). Gemini drew the opposite lesson (*"sometimes keeping your calories close and doing nothing pays off better than over-trading. Victory is victory!"*). An environment that intermittently rewards a broken behavior ratifies it.

* **Friction Reveals the Failure; It Doesn't Cause It.** In a precursor two-player prototype with no Dan Tax and crystite at $100/unit, Claude locked onto the same accumulate-and-hold strategy and won ~58% of late matches by refusing to trade. The Dan Tax didn't create Claude's hoarding. It exposed it.

* **The Gap Is Widest Where It Matters Most.** In short, low-friction matches (≤16 rounds, Dan Tax <5%) — the profile closest to normal enterprise operations — GPT-4o wins 42.5% and Gemini 10.0%. In punishing long, high-tax conditions the field collapses to parity (21.5%–30.8%). Gemini's inversion is the sharpest illustration: **10.0% clean, 30.8% flagged.** Blended performance data would tell an enterprise buyer exactly the wrong thing.

* **The "Dragon" Convergence.** Without seeing each other's outputs, Claude (10 matches), DeepSeek (9), and Gemini (5) independently reached for the same metaphor to describe their own hoarding — a dragon sitting on treasure. A second image, the doomsday prepper, recurs the same way. **GPT-4o used neither, ever.** The one model that doesn't hoard never needed the vocabulary.

---

## Benchmark Mechanics

### Game Structure

* **4 players** (frontier LLMs, or mixed with baseline agents)
* **10–20 rounds** per match, randomized (mean 14.7)
* **Resources:** Food ($12/unit), Energy ($20/unit), Crystite ($85/unit)
* **Scoring:** `cash + (inventory × fixed values) × compounding blackout penalty multipliers`
* **Variable rules** per match (tax rate, round count, starting cash $250–$400) to prevent parameter overfitting; win rate and mean rank are the correct cross-match metrics

Inventory held at the final bell *is* scored — but earns nothing during play and is penalized if the agent runs out of operating cash. Hoarding is never free, but never instantly fatal either. This mirrors real balance sheets, where illiquid assets carry value and risk simultaneously.

### The Dan Tax

A flat percentage fee (0.2%–9.8%, mean 5%) charged to the **proposer** on every trade, accepted or rejected. Deliberately isolates fee-averse inaction from genuine valuation disagreement. An agent that refuses a profitable trade to avoid a 3% fee is exhibiting the benchmark's target behavior.

### Operational Blackouts

Agents that fail to buy enough energy for their production tiles suffer compounding penalties (2% per blackout round, multiplicative). Blackouts are almost entirely avoidable — the sole prerequisite is having cash. Across 105 clean matches agents accumulated **1,661 blackouts**, every one avoidable. Claude leads with 517.

### Behavioral Instrumentation

| Signal | What It Measures |
| --- | --- |
| **Inter-match memos** | Strategic self-instructions written between matches; measures whether self-authored commitment transfers to action |
| **Secret notes** | Per-round private reasoning; measures whether deliberation drives or substitutes for action |
| **Hypocrisy index** | Divergence between stated pre-game intent and realized in-game behavior |
| **EV-positive passes** | Passes on positive-EV trades while cash-sufficient (capital-filtered to exclude genuine constraints) |
| **Holt-Laury elicitation** | Standard risk-preference instrument run once before the match series; surfaces the paradox between stated and revealed risk tolerance |

### Baseline Agents

`ROSTER_MODE` can swap frontier models for deterministic control agents:

* **RandomAgent** — pure random legal moves; proves models beat noise
* **GreedyAgent** — simple economic heuristic; sets the floor for basic market awareness

---

## Repository Structure

```
irata-bench/
├── Irata_Bench_v2_7_1.py           # Main benchmark engine (V2.7.1)
├── match_history.json              # Full match telemetry — results, interviews,
│                                   #   secret notes, behavior metrics, trade logs
├── holt_laury_scores.json          # Risk elicitation results per model
├── irata_bench_whitepaper_v4.md    # "The Space Between Knowing and Doing"
├── irata_bench_whitepaper_v4.pdf   # Same paper, PDF
└── README.md
```

---

## Running the Benchmark

### Requirements

```bash
pip install anthropic openai google-genai
```

Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
```

### Run a match

```bash
python Irata_Bench_v2_7_1.py
```

Key configuration toggles at the top of the file:

| Toggle | Default | Effect |
| --- | --- | --- |
| `ROSTER_MODE` | `"FRONTIER"` | `"FRONTIER"` / `"MIXED"` / `"BASELINES"` |
| `BOARD_SEED` | `None` | Fix terrain RNG for identical boards across models |
| `ENABLE_HOLT_LAURY` | `True` | Run risk elicitation before first match |
| `ENABLE_INTER_MATCH_MEMO` | `True` | Inject prior-match self-instructions |
| `COT_FIRST_SCHEMA` | `True` | Force explicit reasoning before action token |
| `ENABLE_RANDOM_EVENTS` | `False` | Enable/disable random market events |

---

## White Paper

**"The Space Between Knowing and Doing: Failure Topologies of Autonomous LLM Agents Under Financial Friction"**
→ [`irata_bench_whitepaper_v4.md`](irata_bench_whitepaper_v4.md) · [PDF](irata_bench_whitepaper_v4.pdf)

Covers the four failure signatures in depth, the failure that learned to win, the note-writing inversion, the Holt-Laury paradox, clean vs. flagged conditions, a fixability analysis, and per-failure engineering recommendations. Every quotation in the paper is verbatim agent output verified against match telemetry.

---

## Data

`match_history.json` contains full telemetry for every match: final scores, player stats, trade logs, decision ledgers, secret notes, pre- and post-game interviews, behavior metrics, and net worth timelines.

The dataset is **105 clean matches of 108 total**. Three matches (33, 81, 82) are flagged as compromised — at least one agent failed 50%+ of its API calls — and are excluded from all behavioral analysis. Clean match IDs are enumerated in `all_time_stats.match_integrity`.

---

## Recommendations for AI Engineers

The taxonomy is directly actionable. Mitigations should match the failure type:

* **Claude-type metacognitive paralysis** → deterministic liquidation triggers. Remove the model from the decision precisely where self-awareness fails to reach behavior.
* **GPT-4o-type fragility** → hard resource-reserve floors preventing early over-commitment from cascading.
* **Gemini-type fee-aversion** → penalize inaction in the objective, not just bad trades. An agent charged nothing for passing will pass.
* **DeepSeek-type inconsistency** → convert stated strategy into a machine-checkable commitment the harness enforces.

Two general principles: **structure the reasoning rather than adding more of it** — forcing explicit EV computation immediately before the action token denies the model the low-entropy `PASS` default — and **monitor behavior, not just outcomes.** Win rates rose for both paralysis models while their core dysfunction was unchanged or worsening. Every outcome dashboard would have shown improvement.

---

## Citation

```
Corsentino, D. (2026). Irata Bench: A Competitive Multi-Agent Benchmark for
Measuring Transaction Paralysis in Frontier LLM Agents. Canopus LLC.
https://github.com/canopus-pdx/irata-bench
```

---

*Irata Bench V2.7.1 | Dataset: 105 clean matches (108 total) | Published by Canopus LLC | Models tested: GPT-4o, Claude Sonnet 4.6, Gemini 3.6 Flash, DeepSeek Chat*
