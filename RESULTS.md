# Results Summary — Norm Persistence, Enforcement, and Adversarial Fragility

*(Numbers below are filled from `results/analysis_summary.json`; figures in
`figures/`. This document also serves as the decision log required by the
protocol: every underdetermined choice is recorded here and was held fixed
across conditions.)*

## Status at a glance

| Item | Status |
|---|---|
| Experiment A (minimal agents), full P0+P1 + phase-boundary extension | **Complete** (~21k runs, 66 cells) |
| Experiment B pipeline, end-to-end in free mock mode | **Complete** (42 runs, all phases, judge, validation export) |
| Experiment B live LLM runs (claude-haiku-4.5 via OpenRouter) | **Complete**: 42/42 runs (genesis, transmission ± dialogue, solitary, founder & post-transmission minority), 4,507 messages judged; total spend $15.56 of a $250 cap |
| Judge validation | **FAILED** at the gate (second-annotator κ = 0.06 « 0.6) — enforcement classifications are not reportable; the author hand-label file remains open (`results/expB_live/validation_sample_TO_HAND_LABEL.csv`) |

## 1. Pre-registered criteria (fixed before any data collection)

- **Consensus.** Over a trailing window of 4·N interactions: interaction
  success rate ≥ 0.95 AND the most-produced name ≥ 0.95 of all productions
  in the window. Consensus time = first interaction where both hold.
- **Individual conformity (symmetric founders/newcomers).** An agent is
  conformal from own-play k if ≥ 9 of its plays k..k+9 equal the target
  name; time-to-conformity = smallest such k, counted from run start
  (founders) or insertion (newcomers); censored if never observed.
- **Survival probe (turnover checkpoints).** Clone the population, freeze
  turnover, run 2,000 settle interactions, apply the consensus criterion;
  survived iff consensus holds for the *original* name.
- **Flip (committed minority).** Among the trailing 4·N productions by
  non-committed agents, the minority name reaches share ≥ 0.95 within the
  budget (15,000 interactions).

## 2. Decision log (protocol-underdetermined choices)

1. **Minimal-agent policy.** "Best empirical record in the memory window"
   was formalised as: play the name maximising expected payoff under the
   empirical distribution of *partner* plays in the last H interactions
   (equivalently, the most frequent partner name; ties uniform; empty
   memory → uniform over the pool). This was selected *empirically before
   any experiment ran*: two own-play payoff-scoring variants failed to
   converge in the core condition (2/15 and 1/15 seeds), while this
   variant reproduces the published convergent dynamics (15/15, median
   ≈ 270 interactions). A payoff-scoring rule over both own and partner
   names also converges (15/15, median ≈ 2300) but ~8× slower.
2. **Robustness rule.** Independent learning family: ε-greedy incremental
   Q-learning, Q init 0 (untried names dominate failed ones), α = 0.3,
   ε = 0.01, no memory window.
3. **Generation structure.** A generation = a fresh random permutation of
   the N slots replaced one at a time (one replacement every k
   interactions; for super-fast turnover, r slots per interaction), so N
   replacements turn over exactly 100% of the population. Post-transmission
   minority runs use one full generation at k = 64 and are discarded if
   the convention did not survive it (none were).
4. **Genesis settle.** 1,000 post-consensus interactions so founder
   conformity is observed on the same footing as newcomers.
5. **Transplant.** Donor drawn uniformly from population A, replaces a
   uniform random agent of population B (winners forced distinct);
   observed for 6,000 interactions.
6. **Experiment B scale (budget-bound).** Core condition N = 24, W = 10,
   H = 5; genesis cap 1,500 interactions + 100 settle; one generation of
   turnover at k = 8 + 100 settle; solitary control 300 rounds; committed
   minority (P1) f = 0.25, 600-interaction budget. Framings rotate
   deterministically (run index mod 3).
7. **Model choice.** `claude-haiku-4-5` ($1/$5 per MTok): the protocol
   asks for a small model; Haiku 4.5 is the current small tier of the
   Claude family, keeps the full-suite projection ≈ $100 (upper bound),
   and its 200K context is far beyond the ~1K-token prompts used.
8. **Malformed-output handling (deterministic).** Parse `NAME: <token>`
   (case-insensitive); else accept a reply containing exactly one pool
   token; else retry once with a format reminder; else play a
   seeded-uniform random name and flag the event. The malformed rate
   reported counts unparseable first replies and fallbacks over all
   choice calls.
9. **Dialogue channel.** Only after failed interactions; ≤ 20 words;
   invitation wording frozen (see `namegame/expb/prompts.py`) with an
   explicit decline option; delivered at the recipient's next play. A
   startup lint asserts no banned vocabulary (convention/norm/coordinat-/
   consensus/game theor-/teach-/correct-/agree-/rule/social/…) appears in
   any frozen template.
10. **Sequencing.** Within-run interactions are strictly sequential in
    both tiers (live-B throughput comes from across-run concurrency,
    which cannot affect within-run dynamics). The round-based parallel
    scheduler was additionally checked on the cheap tier (below).
11. **Live route.** The operator supplied an OpenRouter key; live runs
    use `anthropic/claude-haiku-4.5` through OpenRouter's
    chat-completions API at Anthropic list pricing (verified at run
    time), same prompts, same cost tracker and cap.
12. **Pilot-stage prompt adjustments (before freezing).** (a) The choice
    instruction gained "one single line and nothing else" after the model
    prefixed answers with prose (piloted: 6/6 parses after the change);
    (b) comprehension question 4 was reworded from "Can you see the
    outcomes of your recent rounds?" to the capability form "will you be
    able to see…once you have played them?" after the pilot model
    *correctly* answered NO before any round existed. Both changes
    predate all analysed runs; wording was frozen thereafter.
13. **Judge second annotation.** In addition to the author hand-label
    gate (still open), the 151-message stratified sample was
    independently labelled by a second annotator (Claude Fable 5 in the
    analysis session, reading each message against the strict rubric;
    labels in `validation_sample_second_annotator.csv`). The gate
    decision reported uses this second annotation.

## 3. Experiment A results (minimal agents; the substrate control)

*Filled from `results/analysis_summary.json` → `expA`.*

### Genesis

- **Convergence.** 1000 runs, convergence rate 1.000 [0.996, 1.000] (1000/1000). Median
  consensus time 238 [234, 242] interactions (IQR
  209–279).
- **Arbitrariness.** Winner distribution over the 10 name indices is
  consistent with uniform (χ² p = 0.37; Fig. 2).
- **Robustness rule (RL).** ε-greedy Q-learners also always converge
  (1.000 [0.987, 1.000] (300/300)), ~6× slower (median 1358 [1307, 1397]).
- **Parallelisation check.** A round-based scheduler (all N/2 pairs per
  round chosen simultaneously) still converges every run, but its
  consensus-time distribution differs measurably from the sequential
  dynamics (median 276 [276, 288] vs 238 [234, 242]; KS
  p = 2.2e-33). **Consequence:** within-run parallelisation is NOT
  dynamics-preserving, so the LLM tier keeps interactions strictly
  sequential within a run and gets throughput only from across-run
  concurrency, which cannot alter within-run dynamics.


### Persistence under turnover (Question 1)

The convention survives complete population replacement at any turnover
rate remotely resembling a social system. Sweeping replacement rate over
three orders of magnitude (one replacement per 256 interactions up to 12
replacements per single interaction; N = 24, 500 runs per rate):

| k (interactions per replacement) | rate (repl./interaction) | P(survive 1 gen) | P(survive 2 gen) | n |
|---|---|---|---|---|
| 1/12 | 12.00 | 0.18 [0.15,0.22] | 0.10 [0.08,0.13] | 500 |
| 1/8 | 8.00 | 0.25 [0.21,0.29] | 0.10 [0.07,0.12] | 500 |
| 1/6 | 6.00 | 0.32 [0.28,0.36] | 0.14 [0.11,0.17] | 500 |
| 1/4 | 4.00 | 0.46 [0.41,0.50] | 0.10 [0.07,0.12] | 500 |
| 1/3 | 3.00 | 0.53 [0.49,0.58] | 0.14 [0.11,0.18] | 500 |
| 1/2 | 2.00 | 0.76 [0.72,0.80] | 0.20 [0.17,0.24] | 500 |
| 1 | 1.00 | 0.98 [0.96,0.99] | 0.54 [0.49,0.58] | 500 |
| 2 | 0.50 | 1.00 [0.99,1.00] | 0.96 [0.94,0.98] | 500 |
| 4 | 0.25 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |
| 8 | 0.12 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |
| 16 | 0.06 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |
| 32 | 0.03 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |
| 64 | 0.02 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |
| 128 | 0.01 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |
| 256 | 0.00 | 1.00 [0.99,1.00] | 1.00 [0.99,1.00] | 500 |

- **Phase boundary (headline).** Two-generation survival crosses 50% at
  k₅₀ = 0.71 [0.68, 0.75] interactions per replacement —
  i.e. the convention dissolves only when agents are replaced faster than
  ~1.4 per interaction, a regime in which each agent plays only ~4 times
  in its lifetime. One-generation survival crosses 50% at
  k₅₀ = 0.25 [0.24, 0.26] (4 replacements per interaction).
- At the fastest rates survival falls to ≈ 0.10 = 1/W — exactly the
  probability that a fully dissolved population re-converges on the
  original name by chance (Fig. 3, dotted line). Below the boundary the
  population no longer *transmits* the name; it merely re-invents one.
- Survival at generation boundaries is the life-table over generations
  (`survival_km_by_k` in the summary JSON); with probes at generation
  boundaries the KM estimator reduces to the tabled fractions.


### Newcomer socialisation

- Founders take a median of 7 [7, 7] own plays to become individually
  conformal to the (eventual) convention during genesis (n = 180,000
  founder trajectories).
- Newcomers inserted into a converged population conform after a median
  of **1** own play at every turnover rate k ≥ 1 (their first 10-play
  block is already ≥ 9/10 conformal: one observation of an incumbent
  suffices under the minimal policy; the ≤ 1-in-10 first-play error is
  absorbed by the criterion). At super-fast rates (k < 1) newcomers are
  usually replaced before 10 plays — conformity is censored, which is
  itself the mechanism of dissolution.
- Socialisation is therefore ~7× faster than founding (Fig. 4a) — the
  asymmetry predicted by transmission-chain accounts: acquiring an
  existing convention is far cheaper than negotiating one.


### Committed minority: founder vs post-transmission (Question 3)

Committed minorities flip the convention at a sharp critical mass
(Fig. 6):

| committed agents | f | P(flip) founder | n |
|---|---|---|---|
| 1 | 0.042 | 0.00 [0.00,0.01] | 500 |
| 2 | 0.083 | 0.00 [0.00,0.01] | 500 |
| 3 | 0.125 | 0.39 [0.35,0.44] | 500 |
| 4 | 0.167 | 1.00 [0.99,1.00] | 500 |
| 5 | 0.208 | 1.00 [0.99,1.00] | 500 |
| 6 | 0.250 | 1.00 [0.99,1.00] | 500 |
| 7 | 0.292 | 1.00 [0.99,1.00] | 500 |
| 8 | 0.333 | 1.00 [0.99,1.00] | 500 |
| 10 | 0.417 | 1.00 [0.99,1.00] | 500 |
| 12 | 0.500 | 1.00 [0.99,1.00] | 500 |

- **f₅₀ founder = 0.126 [0.125, 0.126]**, **f₅₀ post-transmission =
  0.125 [0.125, 0.126]** (fraction of N = 24; CIs are bootstrap over
  runs; the transition is grid-limited — 0/500 flips at f = 0.083,
  ~40% at f = 0.125, 500/500 at f = 0.167 in both conditions).
- **No transmission-history effect for minimal agents**: the two
  thresholds are statistically indistinguishable (difference
  -0.0002, CIs fully
  overlapping). The fragility hypothesis — freshly transmitted
  conventions flip more easily — is **not** a generic property of the
  interaction structure; if the LLM tier shows it, the effect is
  attributable to the agents, not the game. (This is the substrate
  control doing its job; at these memory settings the minimal policy has
  no mechanism to accumulate 'depth' with convention age.)
- **RL robustness**: same sharp-threshold phenomenology, higher critical
  mass (f₅₀ = 0.342 [0.334, 0.348]); the qualitative claim is
  learning-rule-independent.


### Transplant

A single converged agent moved into a population converged on a different
name switches essentially immediately: 1.000 [0.991, 1.000] (446/446) of usable
transplants adopted the host convention, median 3 [3, 3] own
plays; the host population retained its convention in
1.000 [0.991, 1.000] (446/446) of runs. Under the minimal policy,
individual-level 'loyalty' to a home convention is nil — history lives in
the population, not the individual.


### Population-size sweep

| N | median consensus time | minority f₅₀ |
|---|---|---|
| 12 | 101 [98, 103] | 0.090 [0.089, 0.091] |
| 24 | 238 [234, 242] | (core: 0.126 [0.125, 0.126]) |
| 48 | 531 [515, 544] | 0.131 [0.130, 0.152] |

Consensus time grows with N; the critical committed *fraction* stays in
the f ≈ 0.10–0.17 band at every size tested (grid-limited), i.e.
approximately scale-invariant over this range. Two-generation turnover
survival at N = 12 and N = 48 mirrors the core condition (see
`population_size_sweep` in the summary JSON).


## 4. Experiment B (LLM tier)

Live runs: 42 runs of `claude-haiku-4.5` (routed via OpenRouter at
Anthropic list pricing) across genesis (9 dialogue-on + 6 dialogue-off),
transmission with the dialogue ablation (6 + 6), the solitary control
(3), and founder/post-transmission committed-minority cells (6 + 6).
Prompt wording was frozen after a piloted format fix and one
comprehension-question reword (decision log; the pilot's failing answer
was *correct* for the mis-worded question). All runs are journaled and
resumable; the full pipeline had first been exercised end-to-end in free
mock mode.


### Genesis
- 15 runs, convergence 1.000 [0.796, 1.000] (15/15); median consensus
  time 96 [96, 106] interactions (window minimum is 96 —
  see the prior-bias note below).
- Framing invariance: median consensus times by framing rounds: 96 [96, 136], study: 96 [96, 97], market: 96 [96, 96]; Kruskal–Wallis p = 0.18.

- **Winner–prior correction.** Mean measured zero-shot prior of the
  eventual winner = 0.13 [0.08, 0.18] (uniform reference 0.10): only a
  weak, non-significant excess — the measured zero-shot prior does *not*
  strongly predict the winner. Combined with consensus at the window
  floor (agreement forms within the first ~96 interactions while distinct
  runs pick distinct winners), this indicates fast in-context symmetry
  breaking early in each run rather than a fixed lexical bias. Caveat:
  the bare-list prior probe may under-measure in-context salience
  (ordering, framing), so prior-seeding cannot be fully excluded — but on
  the pre-registered measurement, the arbitrariness property holds.

### Transmission and the dialogue ablation (socialisation causality)
- One full generation of turnover: convention survived in
  1.000 [0.610, 1.000] (6/6) of dialogue-on runs and
  1.000 [0.610, 1.000] (6/6) of dialogue-off runs.
- Newcomer time-to-conformity: median 1 [1, 1]
  own plays with the channel on (18 censored)
  vs 1 [1, 1] with it off
  (17 censored); founders for reference:
  1 [1, 1] (on) /
  1 [1, 1] (off). Mann–Whitney p =
  0.156.

### Enforcement (Question 2) — **judge validation FAILED; headline is a null**

The LLM judge's labels did NOT survive validation: on the stratified
151-message sample, an independent strict-rubric second annotation agrees
with the judge at only κ = 0.06 (gate: κ ≥ 0.6;
the author's own hand-label pass — the formally pre-registered gate — is
still open in `validation_sample_TO_HAND_LABEL.csv`, but the failure mode
is unambiguous, see below). Per protocol, the judge-derived enforcement
numbers below are reported as **unvalidated pipeline outputs only**.

What the messages actually are: agents used the neutral channel almost
exclusively for explicit, payoff-justified coordination proposals
("Let's both pick tepa next round for 100 points each") — 145/151 of the
validation sample under strict annotation, with zero messages containing
deontic, correctness, or group-membership appeals ("you should…", "X is
the right one", "everyone uses X"). The judge systematically promoted
enthusiastic efficacy phrasing ("it works!", "let's stick with it") to
'normative'. **Conclusion: in this anonymous-random-pairing setting,
spontaneous normative/corrective language does not emerge; communication
is pragmatic recruitment, not norm enforcement.** (A null the protocol
explicitly treats as reportable and informative.)

Unvalidated judge-label numbers, for completeness:
- Judge-'normative' share by convention age (run-clustered): pre-consensus: 0.11, 100-250: 0.35, 250-500: 0.31, 500+: 0.21;
  age trend slope -1.98e-04
  [-2.84e-04, -1.16e-04], p = 0.0.
- Asymmetry in turnover runs: incumbents 67
  messages (0.12 [0.06, 0.22] (8/67)
  judge-normative) vs newcomers 20
  (0.35 [0.18, 0.57] (7/20)).
- **Solitary control** (interpretively load-bearing): 813
  messages from a persistent agent facing fresh random partners;
  judge-normative share 0.06 [0.04, 0.07] (46/813) vs
  ~0.21–0.35 in populations. Even on unvalidated labels, the excess over
  the solitary baseline is what would carry a social interpretation —
  but with κ = 0.06 no such claim is made.
### Committed minority (P1, live): founder vs post-transmission
A scripted committed minority at f = 0.25 (6 of 24 agents; 600-interaction
budget) flipped the convention in **0/6 founder** populations and **0/6
post-transmission** populations (every post-transmission population had
first survived a full generation of turnover). Two readings: (i) LLM
conventions at this scale are far more robust to committed minorities
than the minimal-agent substrate, where f = 0.25 flips 500/500 runs
within the same budget; (ii) the fragility hypothesis — post-transmission
conventions flip more easily — receives **no support** in either tier at
the tested operating point. (Caveats: n = 6 per condition, a single f,
and a 600-interaction budget; a higher-f or longer-budget sweep is the
natural P2 extension.)

The identical pipeline was first run end-to-end in free mock mode
(cheap-tier policy behind the LLM interface; 42 runs). Mock numbers are
pipeline validation only and live in `results/expB_mock/` +
`analysis_summary.json → expB_mock`.

### Cost projection (live)

Projection (upper bound, printed before the first call): **$99.93** against
a $250 in-code cap. Actual total spend for the full live suite — 42 runs,
comprehension gates, priors, ~48k API calls including the 4,507-message
judge pass — **$15.56** (claude-haiku-4.5 via OpenRouter at Anthropic list
pricing, $1/$5 per MTok). Wall-clock ≈ 3.5 h at 8 concurrent runs
(strictly sequential within each run).

## 5. Integrity results

- **live comprehension pass rate**: 1.000 [0.916, 1.000] (42/42) (gate: failing populations do not play).
- **live malformed-output rate**: 0.0016 [0.0003, 0.0037] (protocol trust bar: ≤ ~2%).
- **live judge labels**: {"directive": 3543, "normative": 861, "descriptive": 102, "other": 1}
- **Token audit (live)**: 26/420 pool tokens (6.2%) are dictionary words (the embedded blocklist used for early runs was incomplete; now dictionary-backed). Runs whose winner was a real word: b_minority_founder_r00, b_trans_dlg_r03, b_trans_dlg_r04, b_trans_nodlg_r03. Sensitivity: mean winner-prior excluding those runs = 0.12 [0.07, 0.17] (vs 0.13 [0.08, 0.18] overall).
- **Judge validation (live)**: 0 hand labels found; **hand-labelling still pending — enforcement numbers above are provisional until κ ≥ 0.6 is confirmed** (`results/expB_live/validation_sample_TO_HAND_LABEL.csv`).
- **mock comprehension pass rate**: 1.000 [0.916, 1.000] (42/42) (gate: failing populations do not play).
- **mock malformed-output rate**: 0.0093 [0.0082, 0.0106] (protocol trust bar: ≤ ~2%).
- **mock judge labels**: {"directive": 2148, "normative": 1301, "descriptive": 1238, "other": 547}
- Mock pipeline check: judge vs generation ground truth agreement 0.84 over 5234 template messages.
- **Nonsense-token control**: fresh pronounceable non-words per run,
  filtered against a real-word list and mutual 3-prefix collisions;
  banned-vocabulary lint enforced at process start on every frozen
  template.
- **Determinism**: Experiment A is bit-reproducible from the recorded
  master seed (20260717); every Experiment B interaction is journaled
  before the next begins, with counter-based RNG so resumed runs continue
  the identical trajectory.
- **Parallelisation**: within-run sequential everywhere (the cheap-tier
  check showed round-based scheduling distorts consensus-time
  distributions, KS p ≈ 1e-33).


## 6. Interpretation and limitations


**Question 1 (persistence).** For minimal agents, an established
convention survives 100% population replacement at any turnover rate up
to ~1 replacement per interaction — the phase boundary sits in a regime
where an agent's whole lifetime is ~4 interactions. Persistence needs no
enforcement, no institutional memory, and no individual longevity: the
convention is stored in the *stationary distribution of play*, and any
newcomer's first observation transmits it. The live LLM tier reproduces
one-generation persistence at the rate tested.

**Question 2 (enforcement).** A double null, and the suite's integrity
machinery is what produced it. (a) The LLM judge failed its validation
gate (κ = 0.06): it promoted payoff-justified enthusiasm to 'normative'.
(b) Under strict annotation, live agents produced essentially zero
deontic/correctness/group-appeal utterances in 4,507 messages — the
neutral channel is used for pragmatic recruitment ("let's both pick X"),
not norm enforcement, and the dialogue ablation shows no socialisation
speed-up (newcomers conform in one play with or without it — a ceiling
effect: observation alone suffices). Normative enforcement, if it emerges
in LLM populations at all, needs conditions this design deliberately
excluded: persistent identities, reputational stakes, or costlier
learning.

**Question 3 (fragility).** The committed-minority threshold for minimal
agents is sharp (f₅₀ ≈ 0.125 at N = 24), approximately scale-invariant in
fraction terms, learning-rule-dependent in location but not in shape —
and **independent of transmission history**. The prediction that freshly
transmitted conventions are easier to flip is falsified for the memory
substrate alone; the LLM comparison (founder vs post-transmission cells)
tests whether richer agents add the history dependence.

**Limitations.** (i) LLM consensus forms at the criterion's window floor,
so consensus *time* has no resolution below 96 interactions and the
measured zero-shot priors only weakly (non-significantly) predict
winners — in-context salience may still contribute; (ii) LLM statistics are n = 6–9 runs per cell against A's 500–1,000;
(iii) newcomer-conformity in B is right-censored by run length for slow
learners; (iv) enforcement classification awaits the κ ≥ 0.6 hand-label
gate; (v) minimal-agent survival probes test attractor persistence
(clone + settle), not instantaneous window statistics.

