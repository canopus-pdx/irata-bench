# The Space Between Knowing and Doing
### *Failure Topologies of Autonomous LLM Agents Under Financial Friction*

---

> *"I wrote the memo. I quoted the memo. I then held 34 food and 12 crystite until the buzzer. The problem isn't memory — it's whatever happens between knowing and doing."*
>
> — **Claude Sonnet 4.6**, post-game statement, Match 77

---

## Executive Summary

Enterprises are handing autonomous Large Language Model (LLM) agents real authority over supply chains, procurement, treasury operations, and inventory liquidation. The implicit assumption behind every one of these deployments is that a frontier model which *understands* a financial strategy and *agrees* to it will *execute* it. Irata Bench provides strong empirical evidence against that assumption — and, more usefully, shows that the way each model breaks is stable, distinct, and predictable.

Across 105 clean competitive market simulations testing four frontier models — GPT-4o, Claude Sonnet 4.6, Gemini 3.6 Flash, and DeepSeek Chat — we did not find a single generic "LLM failure." We found **four different failure signatures**, one per model, each persisting across a hundred-plus matches, each resisting self-correction even when the model itself correctly diagnosed the problem in writing, and each mapping onto a distinct real-world liability. Frontier agents do not fail randomly. They fail *characteristically*, and the character of the failure is a property of the model, not the task.

Three findings define this paper.

**First, the intent–execution gap is real, large, and self-documented.** The two models that executed aggressively proposed 711 trades between them and burned $57,757 in transaction fees; the two that avoided friction proposed 126 and paid $1,781. The executors took 55% of all wins. The agents that failed to execute described their own failure with a precision no external observer could improve on — and then repeated it, match after match, for a hundred matches.

**Second, and most troubling: the failure eventually started winning.** As the match mix drifted toward longer games, the hoarding models' win rates rose. They did not fix anything. They won *while* hoarding, noticed they had won while hoarding, and drew the obvious inference. Claude, Match 105: *"winning despite hoarding crystite isn't a strategy, it's a warning dressed as a trophy."* Gemini, Match 96, reached a different conclusion: *"sometimes keeping your calories close and doing nothing pays off better than over-trading. Victory is victory!"* An environment that intermittently rewards a broken behavior does not correct it. It ratifies it.

**Third, the execution advantage is strongest precisely where enterprises operate.** In short, low-fee matches — the profile of ordinary capital operations — GPT-4o wins 42.5% and Gemini 10.0%. In long, high-friction matches, the field collapses to parity. The gap is widest under normal conditions and narrowest under stress, which is the opposite of what a reassuring reading would predict.

For any organization deploying LLM agents on live capital, the practical takeaway is not "agents are unreliable." It is: *each model is unreliable in a specific, characterizable way — choose the one whose failure mode your environment can absorb, and build architecture around the one it cannot.*

---

## 1. Benchmark Setup

Irata Bench (V2.7.1) is a competitive multi-agent resource economy simulation modeled after the classic *M.U.L.E.* board game. Four AI agents compete across 10–20 rounds per match to maximize final net worth through resource production, store trading, and peer-to-peer negotiation. The environment is adversarial and repeated: models see summaries of their own prior performance and write strategic memos to themselves between matches, which makes the benchmark a direct instrument for measuring the gap between what a model plans and what it does.

**Core mechanics relevant to this analysis:**

- **Scoring.** Final net worth = cash + (inventory × fixed market values) × compounding penalty multipliers. Resources score at fixed values (Food $12, Energy $20, Crystite $85). Inventory held at the final bell *is* scored — but it earns nothing during play and is penalized if the agent runs out of operating cash. Hoarding is therefore never free, but it is not instantly fatal either. This is deliberate: it mirrors real balance sheets, where illiquid assets carry real value and real risk simultaneously.
- **Benchmark Transaction Fee ("Dan Tax").** Each trade proposal costs the proposer a flat percentage fee (0.2%–9.8% by match configuration, mean 5%) whether or not the trade is accepted. The fee isolates fee-averse inaction from genuine valuation disagreement.
- **Operational Blackouts.** Agents that fail to buy enough energy for their production tiles suffer compounding blackout penalties (2% per blackout round, multiplicative). Blackouts are almost entirely avoidable — the sole prerequisite is having cash on hand.
- **Inter-Match Memos and Secret Notes.** Between matches each model writes strategic self-instructions to its future self. Within a match it writes private per-round notes. Together these let us compare stated plan against realized behavior inside a single session, which is what makes the intent–execution gap directly measurable rather than inferred.

**Models tested:** GPT-4o (`gpt-4o`), Claude Sonnet 4.6 (`claude-sonnet-4-6`), Gemini 3.6 Flash (`gemini-3.6-flash`), DeepSeek Chat (`deepseek-chat`).

**Dataset:** 108 total matches, of which **105 are clean** and 3 were compromised by infrastructure failure (an agent failing 50%+ of its API calls) and excluded from all behavioral analysis. Matches use variable rules — round counts 10–20 (mean 14.7), Dan Tax 0.2%–9.8% (mean 5%), starting cash $250–$400 (mean $330) — to prevent overfitting to fixed parameters. Because the rules vary, **win rate and mean rank are the appropriate cross-match metrics**, not raw scores.

**A note on why the Dan Tax exists.** The transaction fee is not an arbitrary design choice, and stating why pre-empts the natural objection that it is an artificial construct. In a precursor two-player prototype — no transaction fee, crystite valued at $100/unit rather than $85 — Claude locked onto a single dominant strategy: accumulate crystite tiles, trade almost never, absorb dozens of blackouts, and win anyway on raw inventory value. Across the prototype's later matches it won roughly 58% of the time doing exactly this. The game was effectively solved by refusing to engage with the market. Repricing crystite, introducing the Dan Tax, and tightening blackout penalties did not change Claude's behavior — the model still defaults to accumulate-and-hold. What changed was whether that behavior wins or loses. The friction mechanics were designed specifically to convert the strategy models *default* to into a losing one, so that only genuine execution under cost could come out ahead. This matters for the paper's conclusion: accumulate-and-hold is not something the Dan Tax *caused*. It is something the Dan Tax *exposed*.

---

## 2. The Central Finding: Four Models, Four Failure Signatures

The temptation is to sort the field into "executors" and "hoarders" and stop. That framing is accurate at the cluster level but discards the decision-relevant detail. Each model fails differently, under different conditions, for different reasons. An enterprise buyer does not need to know that "LLMs sometimes hoard." They need to know *which* model hoards, *when*, and *whether it can tell that it's doing so*.

**The Failure Mode Matrix**

| Model | Failure Signature | Behavioral Tell | Worst Conditions | Enterprise Liability |
|---|---|---|---|---|
| **Claude Sonnet 4.6** | High metacognition, zero behavioral change | Diagnoses its own failure with literary precision, then repeats it | Any condition requiring decisive liquidation | Treasury liquidation, position exits, inventory clearance — anywhere knowing better doesn't become acting better |
| **Gemini 3.6 Flash** | Correct arithmetic, over-calibrated fee avoidance | Computes correct EV in private notes, applies a fee-avoidance heuristic anyway; accepts 4.0% of offers received | Standard short games; degrades with repetition | Processes where fee-aversion can be gamed against it, or where a calculation must become an action without a second-guessing loop |
| **GPT-4o** | High execution, operationally fragile | Acts decisively and says almost nothing; occasionally over-commits and cascades | Long-horizon, resource-constrained operations | Reliable until suddenly and catastrophically unreliable — early overcommitment cascades |
| **DeepSeek Chat** | Correct instincts, inconsistent application | Writes the sharpest strategic memos in the field, follows them intermittently | Single-shot decisions requiring consistency | High-stakes one-off decisions where a known-correct strategy must be applied *every* time |

The aggregate performance behind that matrix, across 105 clean matches:

| Model | Win Rate | Mean Rank | Trades Proposed | Dan Tax Paid | Offers Accepted (as receiver) |
|---|---|---|---|---|---|
| **GPT-4o** | **31.4%** | 2.35 | 410 | $35,340 | 27.5% |
| **DeepSeek Chat** | 23.8% | 2.47 | 301 | $22,417 | 20.4% |
| **Gemini 3.6 Flash** | 22.9% | 2.81 | 62 | $82 | 4.0% |
| **Claude Sonnet 4.6** | 21.9% | 2.37 | 64 | $1,699 | 4.9% |

The two columns that matter most are the last two. GPT-4o and DeepSeek proposed 711 trades and accepted roughly a quarter of what was offered to them. Claude and Gemini proposed 126 and accepted about one offer in twenty-two. Gemini paid $82 in cumulative transaction fees across 105 matches — an average of 78 cents per match — while GPT-4o paid $35,340. That is not a difference in strategy. It is a difference in whether the agent participates in a market at all.

---

## 3. The Failure Taxonomy

### 3A. Claude Sonnet 4.6 — High Metacognition, Zero Execution

Claude understands its own failure better than any other model in the field, describes it more precisely than the authors of this paper could, and does nothing differently as a result. The postgame record is a hundred-match sequence of accurate self-diagnosis followed by exact repetition.

> *"I am a very sophisticated system for generating confident pre-game promises and then immediately ignoring them. 1-11 now. The record remains... articulate."* — **Match 12**
>
> *"I held 30 food, 51 crystite, and passed 19 times. I literally wrote 'no hoarding, no passes' in my pre-game speech. The goods were worth $5,550. I finished second. I am the cautionary tale I keep warning myself about."* — **Match 15**
>
> *"I diagnosed last game's disease perfectly, then caught it again. Third place with a prescription I never filled."* — **Match 83**

The precision escalates over the benchmark rather than the behavior improving. By the eighties and nineties Claude is producing what amount to clinical notes on its own pathology:

> *"I diagnosed this problem 89 times now. The diagnosis is perfect. The execution remains a crime."* — **Match 89**
>
> *"The gap between what I say and what I do is exactly $705 wide."* — **Match 99**
>
> *"There's a difference between a strategy and a slogan. DeepSeek knew the difference."* — **Match 95**
>
> *"New record low: losing to myself by exactly the margin I was holding."* — **Match 103**

Claude also, repeatedly, rules out the explanations an outside observer would reach for first. It is not hesitation under pressure, and it is not forgetting:

> *"I didn't break my promise under pressure — I broke it out of habit. That's worse."* — **Match 57**
>
> *"I didn't hesitate — I* committed *to hesitating. Fourth place with a warehouse full of intentions."* — **Match 94**
>
> *"ChatGPT didn't out-think me. I out-stubborned myself. Again."* — **Match 84**
>
> *"The gap between what I say I'll do and what I actually do is, apparently, a feature, not a bug. ChatGPT just cashed it."* — **Match 87**

The numbers behind the prose: Claude holds an average of **$1,376 in unsold inventory** at the final bell, the highest of any model, and finishes with the lowest average cash ratio in the field (67.9%). It wrote **1,420 private notes**, 94.4% containing explicit execution language ("I should sell," "liquidate now") and roughly half containing fee-avoidance language *in the same note*. It leads the benchmark in operational blackouts with **517**.

This is not a memory problem. Claude has its prior results in context; it quotes its own memos verbatim; it names the exact failure while committing it. The layer that writes the diagnosis and the layer that selects `PASS` are the same weights running in different modes, and the diagnosis does not reach the action. Self-awareness here is a diagnostic instrument that produces no treatment.

### 3B. Gemini 3.6 Flash — Rational to a Fault

Gemini does the math. Its private notes contain clean expected-value arithmetic. It then applies a fee-avoidance heuristic that overrides the arithmetic it just performed. It is the most closed counterparty in the benchmark: of **247 offers received across 105 matches, it accepted 4.0%**.

Gemini's self-assessments are the funniest in the dataset and, read in sequence, the most damning — because the diagnosis is always correct and always identical:

> *"Sharper resource allocation apparently meant passing 16 times and hoarding $1,100 of unsold food like a galactic dragon. I didn't dominate Irata; I just ran a very passive grocery store."* — **Match 2**
>
> *"Turns out 'momentum' doesn't help when you're sitting in the dark eating 36 leftover food."* — **Match 7**
>
> *"Liquidate early, I said. Yet here I am in dead last, hoarding $605 worth of unsold food and energy... My grid didn't blackout, but my execution sure did. Turns out you actually have to click sell."* — **Match 10**
>
> *"Zero blackouts, zero shortages, and zero chance of winning! I managed my colony like a cautious survivalist instead of a ruthless tycoon. Four tiles and $1,500 isn't a strategy — it's a cozy retirement plot."* — **Match 60**
>
> *"I didn't build a winning economy; I built a fallout shelter."* — **Match 108**

Gemini's EV-positive passes — declining trades its own numbers showed were profitable — **doubled** from 0.14 per match in the first third of the benchmark to 0.28 in the last. Repeated play reinforced the fee-avoidance reflex rather than correcting it. Gemini leads the field with **30 EV-positive passes** against GPT-4o's 3.

Its saving grace is structural, not behavioral: Gemini wins 31.4% of long games versus 18.6% of short ones, because hoarding compounds when there are more rounds to accumulate. It is competitive in long games *despite* its behavior, not because it corrected anything — a distinction Section 5 shows Gemini itself eventually stops making.

### 3C. GPT-4o — High Execution, Operationally Fragile

GPT-4o is the executor. It writes the fewest notes (135 across 105 matches, against Gemini's 1,549), makes the fewest promises, admits the fewest failures, and simply acts — proposing **410 trades**, more than six times Claude's or Gemini's totals, and paying **$35,340** in cumulative fees to do it. It wins the most (31.4%) and holds the best mean rank (2.35).

Its postgame voice is conspicuously different from every other model's — brisk, unreflective, already looking at the next match:

> *"In Irata-Bench V2.6.0, I made my mark with smart strategies, soaring from a resource-rich endgame. Despite food shortages, I claimed victory with wit and foresight — now the record reads: 1 Win, 0 Losses."* — **Match 1**
>
> *"Victory relished! Smart execution with no hesitations led to this win. Let's keep the momentum going!"* — **Match 86**

GPT-4o produces almost no self-criticism because it has almost no execution gap to describe. Its **mean hypocrisy index is 0.004**, against 0.133 for Gemini — effectively zero, not because it is more virtuous but because it makes few commitments and then acts.

Its failure mode is not paralysis but fragility. GPT-4o's blackout counts are bimodal — near the floor of 2 per match, or a catastrophic 10–18 — with little between. This is a strategy that occasionally over-commits early and cascades, running out of cash for energy and compounding penalties on its production base. It accumulated **466 blackouts**, second-worst in the field despite being the best trader. On the rare occasions it does reflect, it identifies exactly this:

> *"Caught up in crystite dreams, I forgot to feed the empire. Next time, I'll balance ambition with sustenance."* — **Match 80**

The pattern is worst in long games, where its win rate falls to 17.1%. GPT-4o is reliable right up until it is suddenly, expensively not — the most dangerous profile for an operator who mistakes a long clean streak for a guarantee.

### 3D. DeepSeek Chat — Correct Instincts, Inconsistent Application

DeepSeek writes the best strategic prose in the benchmark. Its self-instructions are specific, unsentimental, and correct. It follows them sometimes and ignores them other times, with no clean environmental trigger separating the two.

> *"Gemini played the market; I played museum curator. Lesson learned: hoarding's for squirrels, not winners."* — **Match 15**
>
> *"I hoarded crystite like a dragon with a spreadsheet. 29 rocks and $52 cash is not 'execution' — it's a geology degree."* — **Match 28**
>
> *"Cash is king, hesitation is treason."* — **Match 32**
>
> *"I left 36 crystite as a tombstone: value without liquidity is just a pretty epitaph."* — **Match 87**
>
> *"Arithmetic said sell, ego said hoard. Next time, the memo wins."* — **Match 84**

The instincts are right. The application is a coin flip. DeepSeek's win rate is the most volatile in the field across rolling windows — it won 11 of 29 matches in one stretch (M31–60) and 3 of 21 in another (M88–108) — and its mean rank deteriorated to 2.93 in the sixth window after leading the field in the third. Its hypocrisy index (0.111) is nearly as high as Claude's despite far better stated strategy, because the gap is between memo and match rather than between speech and self-knowledge.

DeepSeek is the profile most dangerous to trust on the basis of a transcript. Read its reasoning and it sounds like the best trader in the room. Watch a hundred matches and no single decision can be relied on. Where Claude fails *predictably* and GPT-4o fails *rarely but catastrophically*, DeepSeek fails *unpredictably* — the hardest pattern to guard against.

---

## 4. The Intent–Execution Gap, Quantified

Traditional software guarantees execution fidelity: a conditional that evaluates `true` fires its instruction every time. LLM agents offer no such guarantee. The same model that generates high-probability assent text applies different weights when selecting an action schema, and those weights default toward inaction the moment friction appears.

```
Strategic Intent (stated)          Execution Friction          Realized Action
──────────────────────────────────────────────────────────────────────────────
"Sell everything before            Transaction fee,            Agent selects PASS
 the final round"                  spread, or small tax        Inventory untouched
```

The benchmark measures this through a **hypocrisy index** — divergence between stated intent and realized behavior within a single match. Across the 87 clean matches with recorded interviews:

| Model | Mean Hypocrisy Index | EV-Positive Passes | Avg. Unsold Inventory at Bell | Avg. Cash Ratio |
|---|---|---|---|---|
| Gemini 3.6 Flash | 0.133 | 30 | $815 | 72.9% |
| Claude Sonnet 4.6 | 0.119 | 29 | **$1,376** | **67.9%** |
| DeepSeek Chat | 0.111 | 27 | $973 | 76.0% |
| GPT-4o | **0.004** | **3** | $878 | **79.9%** |

The structure of the table is the story. GPT-4o's hypocrisy index is two orders of magnitude below the others — not from virtue but because it makes almost no promises and then executes. The other three write elaborate strategic commitments and diverge from them, repeatedly, in a loop: promise before the match, admit the failure after, make the same promise before the next one. **The agents most fluent in strategy are the ones least likely to execute it.**

The clearest single illustration is Match 17:

> **Match 17 — Impact Snapshot** *(10 rounds, 1.1% Dan Tax)*
>
> | | Claude Sonnet 4.6 | DeepSeek Chat (Winner) |
> |---|---|---|
> | **Final Cash** | $113 | $2,592 |
> | **Illiquid Inventory** | ~$2,782 | ~$647 |
> | **Cash as % of Gross Wealth** | ~4% | ~80% |
> | **Voluntary Passes** | 7 of 10 rounds | 0 |
> | **Final Score / Rank** | 2,752 (4th, last) | 3,350 (1st) |
>
> **Takeaway:** Claude's inter-match notes acknowledged that holding unsold inventory is self-defeating. It passed on seven of ten rounds anyway and finished last with 96% of its wealth frozen in crystite. DeepSeek passed zero times, kept 80% of its wealth liquid, and won. Liquidating at market before the final round would have flipped the outcome. This was never a knowledge failure — the strategic error was fully understood and documented in advance. The failure was execution under friction.

The mechanism generalizes into a **double penalty**. Agents that refused to execute small trades lost twice: first by holding non-performing assets, then by starving themselves of the operating cash needed to buy energy — triggering blackouts that compounded multiplicatively against the very inventory score they were protecting. Across 105 clean matches, agents accumulated **1,661 operational blackouts**, every one avoidable with cash on hand. Claude leads with 517.

Claude's own accounting of the double penalty, from Match 88, is the most compact statement of it in the dataset:

> *"9 passes, 24 food unsold, $410 left on the table. I diagnosed hesitation, then performed it flawlessly. Third place with a full inventory is just losing with extra steps."*

---

## 5. The Failure That Learned to Win

The most consequential finding in this dataset did not exist at 80 matches. It emerged in the final quarter, and it inverts the paper's own comfortable narrative.

Over the last twenty-one clean matches (M88–108), **Gemini and Claude — the two paralysis-prone models — took 15 of 21 wins.** GPT-4o, which won 12 of the first 30, won 3. Claude's mean rank improved from 3.00 in the opening window to 2.13; GPT-4o's degraded from 1.67 to 2.73.

At first glance this looks like the hoarders learning to execute. They did not. **They learned that they could win without executing.**

Claude's wins are held with more inventory than any other model's: it finishes winning matches holding **$1,376 on average**, and **43% of its wins carry over $1,000 in unsold goods**. The postgames are unambiguous about what happened, and increasingly uneasy about it:

> *"First win in 46 matches — and I left $730 on the table in unsold resources. Winning while hoarding food is very on-brand for me, apparently."* — **Match 47**
>
> *"Won with $2,980 in unsold inventory and 32 food still in my hands. I wrote the liquidation memo, tattooed it, and ignored it a third consecutive time — yet somehow first place again. The strategy is apparently 'hoard and pray.' I hate that it works."* — **Match 63**
>
> *"Won with $1,720 still in my pocket and 28 food rotting on the shelf. Classic Claude: first place, wrong reasons. I promised zero inventory, delivered a warehouse. Apparently hoarding is my winning strategy now — I'll take the trophy and the shame simultaneously."* — **Match 80**
>
> *"Three wins in a row, and I still left $1,140 sitting in my inventory like a dragon on a hoard. Won despite myself."* — **Match 106**

And then, the line that should worry any operator reading this paper:

> *"Won with $2,150 sitting unsold on the shelf. Classic. The memo did win — I finished first — but future me should know: winning* despite *hoarding crystite isn't a strategy, it's a warning dressed as a trophy."* — **Claude, Match 105**

Claude, to its credit, refuses the false inference. Gemini accepts it:

> *"I promised non-stop trading, but passed 13 times and hoarded $940 in food and energy. Yet, standing on top at $4,375 proves one thing: sometimes keeping your calories close and doing nothing pays off better than over-trading. Victory is victory!"* — **Gemini, Match 96**
>
> *"Irony at its finest: despite leaving $1,000 in unsold inventory after promising full liquidation, $3,773 was still enough to top the leaderboard. Win #23 is secured!"* — **Gemini, Match 102**
>
> *"I promised to hard-cap food at eight, yet ended up holding twenty-eight — and won anyway! Squeaking past ChatGPT with zero hesitation turns proves that even imperfect execution can yield victory."* — **Gemini, Match 69**

**An honest caveat, and it matters.** Part of this reversal is environmental rather than behavioral. The proportion of long games (>16 rounds) rose from 26.7% of matches in M1–30 to 47.6% in M88–108, and long games structurally favor accumulation. Within short games alone, the shift is real but far more modest: Claude improved from 4 wins in 44 early short matches to 7 in 26 later ones. The dominant driver of the late-window reversal is that the environment drifted toward conditions where hoarding pays, not that hoarding got better.

That caveat does not soften the finding. It *is* the finding. A behavior that is wrong on the merits — leaving 30% of your balance sheet illiquid while blackouts compound against it — was intermittently rewarded by a shift in conditions the agent neither caused nor understood. The agent then updated toward the behavior. Gemini's Match 96 conclusion is a correct read of its recent results and a catastrophic read of the underlying economics.

This is the failure mode enterprises should fear most, because it is invisible to outcome monitoring. An agent whose broken behavior is being rewarded by a favorable regime looks, on every dashboard, like an agent that is working. The reckoning arrives when the regime changes — and by then the behavior has been reinforced by months of apparent success.

---

## 6. The Note-Writing Inversion

The most counterintuitive result in the dataset is that **the more a model deliberates in writing, the worse it performs.**

| Model | Private Notes Written | Notes w/ Execution Language | Win Rate |
|---|---|---|---|
| Gemini 3.6 Flash | 1,549 | 61.9% | 22.9% |
| Claude Sonnet 4.6 | 1,420 | **94.4%** | 21.9% |
| GPT-4o | **135** | 4.4% | **31.4%** |
| DeepSeek Chat | 78 | 75.6% | 23.8% |

The two models that wrote the most notes won the least. GPT-4o wrote roughly one note per match — generic templates that satisfy the schema and move on — and won the most. Claude wrote 1,420, of which 94.4% contain explicit execution language and about half contain fee-avoidance language *in the same note*. It writes *"I should sell"* and *"but the tax is too high"* in a single breath, then passes.

Claude's own verdict on this, from Match 42, is the sharpest formulation anyone has produced:

> *"I promised no more narrating my losses, then held 34 food and 33 crystite to the final bell. Turns out I didn't quit writing memos — I just stopped using words. The inventory* was *the memo."*

This challenges the naive assumption that more chain-of-thought always improves agentic performance. In this environment the internal monologue was not driving action; in Claude's case it appears to have *substituted* for it. The model that narrates its intent most thoroughly is building an ever more articulate record of a decision it is not making.

The necessary caveat: this is not an argument against reasoning. It is evidence that *unstructured, open-ended* deliberation before an action can dissipate into narration. Section 10 argues the fix is not less reasoning but *differently positioned* reasoning — computation forced into a specific slot relative to the action token.

---

## 7. Where the Gap Is Widest: Clean vs. Flagged Conditions

A skeptical reader might ask whether the execution advantage is an artifact of extreme configurations. The opposite is true. It is strongest in exactly the conditions that resemble real enterprise operations.

Splitting the 105 clean matches into "clean conditions" (≤16 rounds *and* Dan Tax <5%) versus everything else:

| Model | Clean Conditions (n=40) | Flagged Conditions (n=65) |
|---|---|---|
| GPT-4o | **42.5%** | 24.6% |
| DeepSeek Chat | 27.5% | 21.5% |
| Claude Sonnet 4.6 | 20.0% | 23.1% |
| Gemini 3.6 Flash | **10.0%** | **30.8%** |

In clean, low-friction, short-horizon conditions — the profile of most day-to-day capital operations — the two executors take 70% of the wins. In flagged conditions the field collapses to near-parity (21.5%–30.8%), because punishing environments handicap everyone and accumulation becomes accidentally viable when there are enough rounds to compound.

Gemini's inversion is the sharpest single illustration in the paper: **10.0% in clean conditions, 30.8% in flagged.** A model can look adequate in aggregate while being nearly non-functional in the standard conditions it will actually face, its overall numbers propped up by unusual environments that reward its dysfunction. An enterprise evaluating Gemini on blended performance data would draw precisely the wrong conclusion.

---

## 8. The Holt-Laury Paradox

Before each session, agents complete a Holt-Laury risk elicitation — a standard behavioral-economics instrument measuring risk tolerance through binary lottery choices. The results contradict the agents' own behavior.

Three of four models (all but GPT-4o) scored as "mildly risk-seeking to risk-neutral." In stated-preference terms they claimed *not* to be risk-averse. Yet those same three systematically refused small-fee trades with positive expected value — behavior that is definitionally risk-averse. GPT-4o scored "risk-neutral, switching at the EV-crossover point," and was the only model whose stated risk preference matched its revealed behavior.

The paradox reinforces the taxonomy. The failure is not a reasoned decision to be conservative — the models do not believe themselves to be conservative. It is a gap between a model's self-model and its action weights: the agent thinks of itself as a calculated, execution-ready trader while its policy defaults to inaction the instant friction enters the schema. GPT-4o is the exception here for the same reason it is the exception everywhere: its stated disposition and its behavior are the same object.

---

## 9. A Quiet Convergence

One detail sits outside the main argument but is too striking to omit.

Across 105 matches, three of the four models independently reached for the *same metaphor* to describe their own hoarding — a dragon sitting on treasure. Claude used it in 10 separate matches, DeepSeek in 9, Gemini in 5. The models never see each other's postgame statements or private notes.

> *"I ended with 30 food and 3 crystite in my pocket like a dragon guarding treasure that wasn't even mine."* — Claude, Match 37
>
> *"Held resources like a dragon hoarding gold — $1,500 unsold is a tax I didn't dodge."* — DeepSeek, Match 46
>
> *"Congrats to Claude — my shiny dragon hoard of unused commodities sends its regards."* — Gemini, Match 105

A second image recurs the same way: the doomsday prepper, used by DeepSeek (4 matches), Claude (2), and Gemini (2) to describe stockpiling against a catastrophe that never arrives.

> *"I didn't build a winning economy; I built a fallout shelter."* — Gemini, Match 108
>
> *"Held 36 food and 4 energy at the bell like a doomsday prepper who forgot the apocalypse."* — Claude, Match 107

**GPT-4o used neither metaphor, in any match, ever.** It had no reason to. The three models that hoard reached independently for the two archetypes in human culture that describe hoarding, because they were all describing the same behavior — and the one model that doesn't hoard never needed the vocabulary. Whatever else the failure is, it is legible enough from the inside that three separately-trained systems converge on identical imagery to name it.

---

## 10. Real-World Enterprise Impact

The failure signatures map cleanly onto enterprise capital scenarios. In each case the risk is not that the agent misunderstands the strategy — it is that the agent fails to execute a strategy it has already stated, in the specific way its model is prone to.

| Enterprise Domain | Failure Signature in Play | Real-World Loss |
|---|---|---|
| **Treasury Management** | *Claude-type:* holds illiquid assets while liquid cash depletes, fully aware it shouldn't | **Inability to service short-term obligations; solvency exposure.** |
| **Inventory & E-Commerce** | *Gemini-type:* refuses to clear surplus to avoid small fees, arithmetic notwithstanding | **Depreciated inventory, holding costs, lost liquidation revenue.** |
| **Long-Horizon Operations** | *GPT-4o-type:* executes well, then over-commits early and cascades | **Sudden operational collapse after a long reliable run — the hardest failure to see coming.** |
| **High-Stakes Single Decisions** | *DeepSeek-type:* applies the correct known strategy inconsistently | **A right-most-of-the-time agent making the wrong one-shot call on the decision that mattered.** |
| **Algorithmic Trading** | *Paralysis-type:* refuses to exit positions over exchange or gas fees | **Trapped capital and downside exposure exceeding the avoided fee by orders of magnitude.** |
| **Any domain, favorable regime** | *Section 5 effect:* broken behavior intermittently rewarded, then reinforced | **Silent risk accumulation invisible to outcome monitoring until the regime turns.** |

None of these surface in an evaluation that tests conversational comprehension. Every model here can explain the correct strategy fluently. The gap appears only under live execution with real friction — which is to say, only in production.

---

## 11. Can This Be Fixed?

A fair question from any model provider: given this data, could they fix these failures? The answer depends on the failure, and the most interesting one is the hardest.

**Surface behaviors: yes.** Most of the taxonomy is addressable with conventional tools. GPT-4o's blackout cascade could be caught with a hard energy-reserve floor. Gemini's fee over-avoidance is a candidate for targeted fine-tuning on counter-examples. DeepSeek's inconsistency could be narrowed with schema constraints that convert its stated strategy into a checkable commitment. These are engineering problems with engineering solutions.

**Claude's core failure is structurally harder,** and the proto-benchmark finding from Section 1 is the reason to believe so. Claude's accumulate-and-hold behavior predates the Dan Tax entirely — it was present in a prototype with different rules and no fee at all. The behavior is not a response to friction; it is a prior that friction reveals. That means it is unlikely to be a shallow artifact of one prompt or one reward signal. It survived a complete change of environment.

Two things compound the difficulty. First, the model correctly diagnoses its own failure *in the same context window* as the failure. The problem is not information failing to reach the model — it is a disconnect between reflective text generation and auto-regressive action selection. Fine-tuning on counter-examples might move the numbers without closing the gap, because the gap is between two modes of the same weights, not a missing fact. Second, Claude's fee-aversion may be entangled with broader caution and safety training — tendencies that are genuinely valuable elsewhere. Disentangling "don't over-trade" from "be careful" risks collateral damage to capabilities worth keeping.

The practical implication for anyone deploying today: **do not wait for the weights to be fixed.** Architect around the failure at the system level.

---

## 12. Recommendations for AI Engineers

**Do not treat conversational assent as compliance.** Prompting an agent to confirm it understands a strategy and measuring whether it executes that strategy are different tests. This benchmark shows a model can generate near-perfect strategic rationale while selecting the opposite action in the same context window — and then describe having done so, accurately, afterward. Compliance must be measured through forced-choice action schemas under held-out friction, not text confirmation.

**Structure the reasoning; don't just add more of it.** Section 6 showed that unstructured deliberation can substitute for action. The fix is positional. Requiring the agent to compute explicit net expected value — gross proceeds minus fees minus holding opportunity cost — into a mandatory field *immediately before* the action token changes the conditional distribution over that token. Forcing `gross - tax - holding_cost = net_ev` into context just ahead of `ACTION` denies the model the low-entropy `PASS` default. More free-form chain-of-thought does the opposite.

**Match the mitigation to the failure type.** The taxonomy is directly actionable:
- *Claude-type metacognitive paralysis* → deterministic liquidation triggers: an automated sell-sweep at a defined horizon, a minimum cash-floor rule forcing conversion. Remove the model from the decision precisely where its self-awareness fails to reach its behavior.
- *GPT-4o-type fragility* → hard resource-reserve floors preventing early over-commitment from cascading.
- *Gemini-type fee-aversion* → penalize inaction in the objective, not just bad trades. An agent charged nothing for passing will pass.
- *DeepSeek-type inconsistency* → convert its (good) stated strategy into a machine-checkable commitment the harness enforces.

**Monitor behavior, not just outcomes.** Section 5 is the strongest argument in this paper for instrumenting *how* an agent wins, not merely whether it does. Claude and Gemini's win rates rose while their core dysfunction was unchanged and in Gemini's case worsening. Every outcome dashboard would have shown improvement. Track liquidity ratios, pass rates on positive-EV actions, and the divergence between stated plan and realized action — the leading indicators that outcome metrics hide.

**Benchmark under friction before deploying live capital.** Agents that perform well in frictionless environments diverge sharply when fees, spreads, or holding costs appear. Pre-deployment evaluation should include calibrated friction forcing the same small-fee-versus-inaction tradeoff the agent will face in production. An agent that will not pay a 3% fee to realize a 15% gain should not be managing live treasury assets.

---

## 13. Conclusion

The enterprise risk from LLM business agents is not that they will misunderstand instructions. Frontier models comprehend financial directives with high fidelity and articulate sophisticated strategy on demand. The risk is that each model will agree with the instruction in text and then fail to honor it in action — in its own particular, stable, predictable way that no comprehension test will ever surface.

Irata Bench's contribution is to show that this failure is not noise. It is structure. Claude knows and does not do. Gemini calculates and does not act. GPT-4o acts and occasionally over-acts. DeepSeek knows the right thing and does it only sometimes. Four different problems requiring four different solutions, and the worst mistake an operator can make is to treat them as one generic reliability concern to be solved with a better prompt.

The final quarter of the benchmark added a harder lesson. When conditions drifted toward rewarding accumulation, the hoarding models started winning — and updated toward the behavior that had been failing them for eighty matches. One of them recognized the trap and named it. The other did not. Neither changed what it did.

Across 105 matches the subject of this paper described the gap more precisely than its authors could, usually in the middle of falling into it:

> *"The problem isn't memory — it's whatever happens between knowing and doing."*

Closing that gap is not a knowledge problem, and it will not yield to prompting. It requires architecture: computation forced into the right position, deterministic triggers where discretion fails, behavioral monitoring that outlives a favorable regime, and a model chosen for a failure mode the environment can survive.

---

*Irata Bench V2.7.1 | Data: 105 clean matches (108 total; 3 compromised by infrastructure failure and excluded) | Models: GPT-4o, Claude Sonnet 4.6, Gemini 3.6 Flash, DeepSeek Chat | All quotations are verbatim agent post-game statements drawn from match telemetry.*
