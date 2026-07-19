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
| Experiment B live LLM runs (claude-haiku-4.5 via OpenRouter) | **Complete**: 88 runs (genesis n=15, transmission ± dialogue n=10/10, solitary n=3, minority pilot n=12 + threshold sweep n=48), 28,315 messages judged; total spend $70.97 of a $250 cap |
| Judge validation | **FAILED** at the gate (second-annotator κ = 0.06 « 0.6) — enforcement classifications are not reportable; the author hand-label file remains open (`results/expB_live/validation_sample_TO_HAND_LABEL.csv`) |
| E1–E4 side-product convention suite | **Complete**: 300 substrate + 28 live populations |
| E1–E4 convention-inducing variants + integrity repair | **Complete**: 220 substrate + 38 live populations (compression / noise / memory / dialogue / scratch-line / sonnet); original live e2/e4 cells found VOID (reply-cap truncation → 98% fallback moves) and superseded by corrected cells |
| Final-submission power upgrade (§1) | **Complete**: load-bearing live cells at n=12/12/12/12/10/10 (e1_tight, e1_squeeze, e3_squeeze, e4_think, e2_think, e2_dialogue) and naming transmission at 10/10 |
| Live committed-minority threshold sweep (§2) | **Complete**: 48 runs, f ∈ {0.25, 0.33, 0.42} × {founder, post-transmission}, 2,000-interaction budget; live f₅₀(founder) = 0.300 [0.266, 0.330], post-transmission f₅₀ ∈ (0.25, 0.33) |
| Second model family (§3): openai/gpt-5-mini | **Complete**: 14 naming runs + 36 env populations, frozen prompts, malformed 0.0000 over 26.5k calls; key finding — genuine description conventions in e3_squeeze (7 distinct schemes / 8 populations, prior-match 0.19) |
| Enforcement rebuild (§4) | **Complete**: κ(judge–strict) = 0.062 from CSVs (author sheet pending — three-way matrix auto-completes when supplied); frozen-lexicon detector validated then scanned: 2/27,502 haiku-population, 0/813 solitary, 2/345 gpt-5-mini messages with any deontic/correctness/group/sanction feature — all four hits recruitment-framed |
| Final-run live spend | expB $70.97 + envs $106.92 + m2 $7.75 → ≈ **$128 new** this run (caps 250/140/25/20; brief cap $400) |

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
14. **Reply-cap invalidation and repair (E2/E4/E3 live).** The env
    cells inherited short reply caps (20 tokens per E2 turn, 8 per E4
    demand, 8 per E3 pick) on the assumption the model would comply
    with "reply with only X". Raw-reply probes showed it preambles
    first in nearly every turn, so truncation sent 98.1% (E2) / 98.8%
    (E4) / 36.3% (E3 chooser) of live moves to the pre-registered
    random fallback; E1 was unaffected. Decision: mark the original
    live e2/e4 cells VOID (journals retained), and repair rather than
    reinterpret — a one-line scratch turn with a 200-token cap and
    final-anchored-answer parsing (e2_think/e4_think, e2_dialogue,
    sonnet cells), a 24-token chooser for E3 (e3_redo, e3_squeeze) —
    with parse failures journaled separately from occupied-cell picks.
    The substrate tier is unaffected (well-formed by construction).
15. **Variant (lever) design.** All convention-inducing levers are
    payoff/capacity/structure changes only — word budgets, channel
    noise, memory size, a neutral pre-episode message channel, the
    scratch line, model tier — and every prompt still passes the
    banned-vocabulary lint (nothing names consistency, order, style or
    agreement). e2_dialogue rides on the scratch-line turn format (its
    no-dialogue control is e2_think); the sonnet cells use the same
    scratch-line design (matched haiku controls e2_think/e4_think).
    e3_squeeze fixes E3's diversity-test flaw by sharing one fixed item
    set across populations; the per-population zero-shot prior probe
    still measures token-level bias on the shared traits.
16. **Scheme coding rules (frozen before the §R1 baseline data).** A
    population's per-item description is the modal extracted trait-set
    over the last 15 formation-phase mentions of that item; a
    population's *scheme* is the 6-tuple of per-item modal sets. Rule A
    (headline, as used throughout): schemes are distinct iff their
    normalised tuples differ exactly. Rule B (coarser): agglomerative
    clustering with mean per-item Jaccard similarity, threshold 0.5 —
    schemes in one cluster count once. Rule C (finer): distinct iff ANY
    item's modal set differs (identical to A for full tuples; differs
    only when items lack a modal set). All three are reported for every
    headline count; no rule is chosen after seeing the null.
17. **E3 no-interaction pseudo-population design.** Per family, 16
    pseudo-populations; each consists of independent zero-shot
    generations from the frozen SEND prompt with the empty-history
    header, same shared items, same per-item event counts as the
    matched real cell (drawn from its journals), run temperature.
    Pushed through the identical extraction and all three coding rules.
    The null probability reported is the fraction of 2,000 resamples of
    8 pseudo-populations showing >= 7 distinct schemes (Rule A, with B
    and C alongside).
18. **Transplant adoption criterion (frozen before any transplant
    run).** A transplanted agent counts as ADOPTED if, over its last 10
    sends in the host, >= 80% of extracted trait-sets equal the host's
    pre-transplant modal set for the sent item. Latency = number of
    sends until the criterion window first holds. Host stability =
    host's per-item modal sets unchanged in the final 15-mention
    window. Protocol: one agent, journaled state replayed, replaces a
    random host slot; 240 further interactions under normal random
    pairing; >= 10 ordered pairs of distinct-scheme gpt-5-mini
    populations.
19. **Swap protocol (frozen).** Evaluation-only episodes with all
    memories frozen at end-of-run state: describer sampled from
    population A, responder from population B, 30 episodes per ordered
    pair (all 56 m2 ordered pairs + 8 within-population baselines;
    haiku control: 24 sampled ordered pairs + 12 baselines). Statistic:
    success difference (within minus cross) with run-level bootstrap
    CI. No agent state is updated by evaluation episodes.
20. **E2 mitigation cells (prompts frozen after lint + tiny pilot).**
    e2_role: e2_think plus one factual line in the episode state
    assigning each badge a fixed grid half (wording lint-checked, no
    coordination vocabulary). e2_hetero: e2_think with each pair
    drawn one-from-each family (haiku-4.5 x gpt-5-mini), no role line;
    pairing always cross-family (an environment-structure property of
    the cell, not a prompt change). n = 8 populations each; success
    measured against the e2_think baseline at matched n.
21. **Independent axis probes (journaled separately; exempt from the
    banned-vocabulary lint because naming the scheme is their
    function; no probe wording enters any population prompt).** Prior
    concentration: >= 50 zero-shot samples per E3 item per family and
    per E1 budget at run temperature; entropy and modal probability
    with bootstrap CIs. Reachability: >= 30 execution trials per
    family per setting in which the target code/scheme is explicitly
    specified; success = mechanical check of the emitted note. Both
    probes fixed before any live axis data is collected.
23. **E3-m2 turnover criteria (logged before the analysis ran; the
    existing e3_squeeze journals already contain one full generation of
    gradual replacement — interactions_per_replacement=6, the design
    analogue of the naming game's k=8 — so this section is computed on
    existing data, not new runs).** Scheme survival: the settle-phase
    per-item modal sets equal the pre-turnover formation-tail modal
    sets on >= 4 of 6 items. Newcomer adoption: a turnover-born agent
    adopts if >= 80% of its last 10 sends match the pre-turnover modal
    set of the sent item (same window and threshold as the transplant
    criterion, entry 18); newcomers with fewer than 10 sends report
    the raw match share instead.
22. **Survival-figure units (correction).** The turnover sweep's k is
    INTERACTIONS PER REPLACEMENT (TransmissionConfig; one slot replaced
    every k interactions, r slots per event when k=1). A generation is
    N*k interactions; an agent plays 2/N of interactions, so expected
    agent lifetime is ~2k plays. The earlier axis label ("replacements
    per 24 interactions") inverted this; axis, caption, and text are
    now derived from the simulator structure above.

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
- Framing invariance: median consensus times by framing rounds: 96 [96, 106], study: 96 [96, 96], market: 96 [96, 96]; Kruskal–Wallis p = 0.01.

- **Winner–prior correction.** Mean measured zero-shot prior of the
  eventual winner = 0.10 [0.08, 0.14] (uniform reference 0.10): only a
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
  1.000 [0.722, 1.000] (10/10) of dialogue-on runs and
  1.000 [0.722, 1.000] (10/10) of dialogue-off runs.
- Newcomer time-to-conformity: median 1 [1, 1]
  own plays with the channel on (31 censored)
  vs 1 [1, 1] with it off
  (27 censored); founders for reference:
  1 [1, 1] (on) /
  1 [1, 1] (off). Mann–Whitney p =
  0.667.

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
- Judge-'normative' share by convention age (run-clustered): pre-consensus: 0.13, 100-250: 0.33, 250-500: 0.24, 500+: 0.20;
  age trend slope 1.95e-07
  [-1.86e-05, 2.10e-05], p = 0.9792.
- Asymmetry in turnover runs: incumbents 115
  messages (0.13 [0.08, 0.20] (15/115)
  judge-normative) vs newcomers 23
  (0.30 [0.16, 0.51] (7/23)).
- **Solitary control** (interpretively load-bearing): 813
  messages from a persistent agent facing fresh random partners;
  judge-normative share 0.06 [0.04, 0.07] (46/813) vs
  ~0.21–0.35 in populations. Even on unvalidated labels, the excess over
  the solitary baseline is what would carry a social interpretation —
  but with κ = 0.06 no such claim is made.
### Committed minority, live — pilot (f = 0.25, 600-interaction budget)
A scripted committed minority at f = 0.25 flipped **0/6 founder** and
**0/6 post-transmission** populations. Kept as the pilot; the threshold
sweep below supersedes it as the headline.

### Committed-minority threshold, live (final: f ∈ {0.25, 0.33, 0.42}, 2,000-interaction budget, n = 8/cell)
Flips — founder: f=0.25: 1/8; f=0.33: 6/8; f=0.42: 8/8. Post-transmission: f=0.25: 0/8; f=0.33: 8/8; f=0.42: 8/8.

Logistic fit (unchanged pre-registered flip criterion): **live f₅₀
(founder) = 0.300 [0.266, 0.330]**;
post-transmission flips are perfectly separated between f = 0.25 (0/8)
and f = 0.33 (8/8), so the honest statement is **f₅₀ ∈ (0.25, 0.33)**
(the MLE is degenerate; no sham-precise CI is reported). Readings:
(i) the live threshold is ~2.4× the substrate's (f₅₀ = 0.126
[0.125, 0.126]) and close to the ε-greedy RL substrate's 0.342 —
LLM conventions are markedly more minority-resistant than the
imitation substrate; (ii) the founder and post-transmission conditions
are statistically indistinguishable — the fragility hypothesis
(transmitted conventions flip more easily) again receives **no
support**; if anything the point estimates run in the opposite
direction (0.286 vs 0.300); (iii) the pilot's
0/6+0/6 at f = 0.25 is confirmed as sub-threshold rather than
budget-limited (1/16 flips even at the 2,000-interaction budget).

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

- **live comprehension pass rate**: 1.000 [0.962, 1.000] (98/98) (gate: failing populations do not play).
- **live malformed-output rate**: 0.0007 [0.0002, 0.0016] (protocol trust bar: ≤ ~2%).
- **live judge labels**: {"directive": 21586, "normative": 5887, "descriptive": 840, "other": 2}
- **Token audit (live)**: 26/980 pool tokens (2.7%) are dictionary words (the embedded blocklist used for early runs was incomplete; now dictionary-backed). Runs whose winner was a real word: b_minority_founder_r00, b_trans_dlg_r03, b_trans_dlg_r04, b_trans_nodlg_r03. Sensitivity: mean winner-prior excluding those runs = 0.10 [0.07, 0.13] (vs 0.10 [0.08, 0.14] overall).
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


## 7. E1–E4: conventions as side products of ordinary joint work


Design: four task environments in which the reward fixes WHAT must be
accomplished while leaving HOW reward-equivalent (by construction — input
orders randomized, task symmetries, payoff-symmetric badges), so any
population-level regularity in the "how" is an arbitrary convention.
Battery per environment: within-population concentration, ACROSS-population
diversity (the signature separating convention from shared model bias),
zero-shot prior probes, stranger-pool / shuffle controls, generational
turnover adoption, and (E1) a scripted committed minority.  Environments:
E1 relay-QA (wire-format conventions under a word budget), E2 simultaneous
grid co-assembly (badge→region division-of-labour conventions), E3
open-lexicon reference (per-item description conventions, no label pool),
E4 tagged bargaining (badge-conditioned demand conventions; the
Axtell–Epstein–Young 'emergent classes' game).  Full definitions in
`namegame/envs/`; numbers in `results/envs_summary.json`.

### Substrate tier (minimal policies; the control)
- **E1 (12-word notes)**: 0.950 [0.863, 0.983] (57/60) of populations conventionalized on a note ordering (modal share 0.81 [0.77, 0.84]); **42 distinct modal orderings** across 60 populations (Simpson diversity 0.99); inter-agent agreement 0.87 [0.82, 0.92]; task success (tail) 0.93 [0.92, 0.94]; newcomer first-3-notes adoption of the incumbent ordering 0.33 [0.28, 0.38] vs founders' first-3 0.31 [0.28, 0.34]; scripted minority (f=0.25, fixed alternative ordering) flipped 0.48 [0.36, 0.61] (29/60). Shuffle control modal share 0.09 (97.5th pct 0.15).
- **E1 (25-word notes)**: 0.983 [0.911, 0.997] (59/60) of populations conventionalized on a note ordering (modal share 0.83 [0.81, 0.85]); **43 distinct modal orderings** across 60 populations (Simpson diversity 0.99); inter-agent agreement 0.90 [0.86, 0.94]; task success (tail) 0.94 [0.93, 0.95]; newcomer first-3-notes adoption of the incumbent ordering 0.34 [0.29, 0.38] vs founders' first-3 0.29 [0.25, 0.32]; scripted minority (f=0.25, fixed alternative ordering) flipped 0.45 [0.33, 0.58] (27/60). Shuffle control modal share 0.08 (97.5th pct 0.12).
- **E1 squeeze (6-word notes)**: 0.000 [0.000, 0.088] (0/40) of populations conventionalized on a note ordering (modal share 0.07 [0.06, 0.07]); **0 distinct modal orderings** across 40 populations (Simpson diversity 0.00); inter-agent agreement 0.02 [0.01, 0.02]; task success (tail) 0.20 [0.18, 0.22]; newcomer first-3-notes adoption of the incumbent ordering 0.01 [0.01, 0.02] vs founders' first-3 0.01 [0.01, 0.02]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a. Shuffle control modal share 0.07 (97.5th pct 0.10).
- **E1 squeeze + 2× memory (6-word notes)**: 0.000 [0.000, 0.088] (0/40) of populations conventionalized on a note ordering (modal share 0.06 [0.06, 0.07]); **0 distinct modal orderings** across 40 populations (Simpson diversity 0.00); inter-agent agreement 0.01 [0.01, 0.02]; task success (tail) 0.22 [0.20, 0.23]; newcomer first-3-notes adoption of the incumbent ordering 0.02 [0.01, 0.03] vs founders' first-3 0.01 [0.01, 0.02]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a. Shuffle control modal share 0.06 (97.5th pct 0.10).
- **E1 noisy channel (12-word notes, 25% word deletion)**: 0.975 [0.871, 0.996] (39/40) of populations conventionalized on a note ordering (modal share 0.82 [0.80, 0.84]); **34 distinct modal orderings** across 40 populations (Simpson diversity 0.99); inter-agent agreement 0.91 [0.85, 0.96]; task success (tail) 0.36 [0.34, 0.38]; newcomer first-3-notes adoption of the incumbent ordering 0.37 [0.32, 0.42] vs founders' first-3 0.31 [0.28, 0.35]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a. Shuffle control modal share 0.08 (97.5th pct 0.12).
- **E1 stranger-pool control**: tracked informers facing fresh memoryless responders self-lock (self-consistency 0.85 [0.84, 0.87]) but agree with EACH OTHER at only 0.01 — individual habit forms alone; population-wide agreement is the social part.
- **E2 (grid assembly)**: 0.025 [0.004, 0.129] (1/40) of populations conventionalized a badge→region mapping (mapping share 0.09 [0.05, 0.12]); episode success first-12 0.43 [0.39, 0.47] → last-12 0.51 [0.47, 0.55]; 1 distinct modal mappings: [["('bottom', 'top')", 1]]; malformed/call 0.00 [0.00, 0.00].
- **E2 + scratch line (corrected turn protocol)**: 0.000 [0.000, 0.161] (0/20) of populations conventionalized a badge→region mapping (mapping share 0.11 [0.06, 0.16]); episode success first-12 0.42 [0.38, 0.48] → last-12 0.47 [0.42, 0.53]; 0 distinct modal mappings: []; malformed/call 0.00 [0.00, 0.00]; message-channel use 0.00 [0.00, 0.00].
- **E2 + scratch line + pre-episode message channel**: 0.000 [0.000, 0.161] (0/20) of populations conventionalized a badge→region mapping (mapping share 0.08 [0.04, 0.12]); episode success first-12 0.45 [0.38, 0.51] → last-12 0.45 [0.39, 0.50]; 0 distinct modal mappings: []; malformed/call 0.00 [0.00, 0.00]; message-channel use 0.00 [0.00, 0.00].
- **E3 (open-lexicon reference)**: 3.9 [3.6, 4.2] of 6 items per population settled on one description (mean per-item modal share 0.58 [0.57, 0.60]); **60 distinct population-level description schemes** across 60 populations; note length 2.3 [2.3, 2.3] words (first third) → 2.3 [2.3, 2.3] (last third); modal description matches a zero-shot prior probe in 0.40 [0.35, 0.46] of items; coined non-word labels: 0.
- **E3 squeeze (3-word notes, hard distractors, shared items)**: 4.2 [3.8, 4.5] of 6 items per population settled on one description (mean per-item modal share 0.59 [0.57, 0.61]); **40 distinct population-level description schemes** across 40 populations; note length 2.3 [2.3, 2.3] words (first third) → 2.3 [2.3, 2.3] (last third); modal description matches a zero-shot prior probe in 0.48 [0.42, 0.53] of items; coined non-word labels: 0.
- **E4 (tagged bargaining)**: equilibrium types across 60 populations: {'class': 9, 'egalitarian': 28, 'other_stable_50_50': 17, 'fractious': 6}; class (badge-conditioned 70/30) share 0.15 [0.08, 0.26] (9/60); egalitarian share 0.47 [0.35, 0.59] (28/60); tail compatibility 0.86 [0.85, 0.87].
- **E4 + scratch line (corrected turn protocol)**: equilibrium types across 20 populations: {'class': 3, 'egalitarian': 10, 'fractious': 1, 'other_stable_50_50': 6}; class (badge-conditioned 70/30) share 0.15 [0.05, 0.36] (3/20); egalitarian share 0.50 [0.30, 0.70] (10/20); tail compatibility 0.86 [0.84, 0.87].

### Live LLM tier (claude-haiku-4.5)
- **E1 (12-word notes)**: 0.000 [0.000, 0.242] (0/12) of populations conventionalized on a note ordering (modal share 0.08 [0.05, 0.12]); **0 distinct modal orderings** across 12 populations (Simpson diversity 0.00); inter-agent agreement 0.03 [0.01, 0.04]; task success (tail) 0.92 [0.80, 0.99]; newcomer first-3-notes adoption of the incumbent ordering 0.01 [0.00, 0.02] vs founders' first-3 0.01 [0.00, 0.03]; scripted minority (f=0.25, fixed alternative ordering) flipped 0.00 [0.00, 0.24] (0/12). Shuffle control modal share 0.05 (97.5th pct 0.08).
- **E1 (25-word notes)**: 0.000 [0.000, 0.490] (0/4) of populations conventionalized on a note ordering (modal share 0.05 [0.04, 0.05]); **0 distinct modal orderings** across 4 populations (Simpson diversity 0.00); inter-agent agreement 0.02 [0.02, 0.04]; task success (tail) 0.95 [0.93, 0.96]; newcomer first-3-notes adoption of the incumbent ordering 0.00 [0.00, 0.00] vs founders' first-3 0.01 [0.00, 0.02]; scripted minority (f=0.25, fixed alternative ordering) flipped 0.00 [0.00, 0.49] (0/4). Shuffle control modal share 0.05 (97.5th pct 0.07).
- **E1 squeeze (6-word notes)**: 0.083 [0.015, 0.354] (1/12) of populations conventionalized on a note ordering (modal share 0.17 [0.11, 0.25]); **1 distinct modal orderings** across 12 populations (Simpson diversity 0.00); inter-agent agreement 0.13 [0.05, 0.27]; task success (tail) 0.48 [0.41, 0.54]; newcomer first-3-notes adoption of the incumbent ordering 0.02 [0.01, 0.04] vs founders' first-3 0.05 [0.02, 0.10]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a. Shuffle control modal share 0.08 (97.5th pct 0.12).
- **E1 squeeze + 2× memory (6-word notes)**: 0.000 [0.000, 0.490] (0/4) of populations conventionalized on a note ordering (modal share 0.17 [0.03, 0.36]); **0 distinct modal orderings** across 4 populations (Simpson diversity 0.00); inter-agent agreement 0.13 [0.02, 0.26]; task success (tail) 0.29 [0.16, 0.44]; newcomer first-3-notes adoption of the incumbent ordering 0.03 [0.03, 0.03] vs founders' first-3 0.04 [0.00, 0.11]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a.
- **E1 noisy channel (12-word notes, 25% word deletion)**: 0.000 [0.000, 0.490] (0/4) of populations conventionalized on a note ordering (modal share 0.08 [0.06, 0.10]); **0 distinct modal orderings** across 4 populations (Simpson diversity 0.00); inter-agent agreement 0.02 [0.00, 0.05]; task success (tail) 0.43 [0.35, 0.50]; newcomer first-3-notes adoption of the incumbent ordering 0.01 [0.00, 0.02] vs founders' first-3 0.00 [0.00, 0.00]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a. Shuffle control modal share 0.06 (97.5th pct 0.08).
- **E1 stranger-pool control**: tracked informers facing fresh memoryless responders self-lock (self-consistency 0.17 [0.09, 0.28]) but agree with EACH OTHER at only 0.00 — individual habit forms alone; population-wide agreement is the social part.
- **E2 (grid assembly) — VOID live, see integrity note**: 0.000 [0.000, 0.390] (0/6) of populations conventionalized a badge→region mapping (mapping share 0.06 [0.03, 0.08]); episode success first-12 0.38 [0.28, 0.47] → last-12 0.36 [0.25, 0.47]; 0 distinct modal mappings: []; malformed/call 0.98 [0.98, 0.99].
- **E2 + scratch line (corrected turn protocol)**: 0.000 [0.000, 0.278] (0/10) of populations conventionalized a badge→region mapping (mapping share 0.01 [0.00, 0.03]); episode success first-12 0.03 [0.00, 0.05] → last-12 0.04 [0.02, 0.07]; 0 distinct modal mappings: []; malformed/call 0.44 [0.43, 0.46] (parse failures 0.06 [0.05, 0.08]); message-channel use 0.00 [0.00, 0.00].
- **E2 + scratch line + pre-episode message channel**: 0.000 [0.000, 0.278] (0/10) of populations conventionalized a badge→region mapping (mapping share 0.05 [0.02, 0.08]); episode success first-12 0.12 [0.05, 0.22] → last-12 0.14 [0.10, 0.19]; 0 distinct modal mappings: []; malformed/call 0.42 [0.41, 0.44] (parse failures 0.03 [0.02, 0.03]); message-channel use 1.00 [1.00, 1.00].
- **E2 + scratch line, claude-sonnet-4.5**: 0.000 [0.000, 0.561] (0/3) of populations conventionalized a badge→region mapping (mapping share 0.00 [0.00, 0.00]); episode success first-12 0.00 [0.00, 0.00] → last-12 0.00 [0.00, 0.00]; 0 distinct modal mappings: []; malformed/call 0.02 [0.01, 0.05] (parse failures 0.01 [0.00, 0.02]); message-channel use 0.00 [0.00, 0.00].
- **E2 mitigation: exogenous role line (entry 20)**: 0.000 [0.000, 0.324] (0/8) of populations conventionalized a badge→region mapping (mapping share 0.11 [0.06, 0.17]); episode success first-12 0.25 [0.18, 0.33] → last-12 0.14 [0.07, 0.21]; 0 distinct modal mappings: []; malformed/call 0.46 [0.45, 0.47] (parse failures 0.04 [0.03, 0.04]); message-channel use 0.00 [0.00, 0.00].
- **E2 mitigation: heterogeneous pairing (haiku × gpt-5-mini, entry 20)**: 0.000 [0.000, 0.324] (0/8) of populations conventionalized a badge→region mapping (mapping share 0.02 [0.00, 0.05]); episode success first-12 0.08 [0.03, 0.15] → last-12 0.06 [0.03, 0.10]; 0 distinct modal mappings: []; malformed/call 0.26 [0.24, 0.27] (parse failures 0.03 [0.02, 0.03]); message-channel use 0.00 [0.00, 0.00].
- **E3 (open-lexicon reference)**: 6.0 [6.0, 6.0] of 6 items per population settled on one description (mean per-item modal share 0.99 [0.98, 1.00]); **6 distinct population-level description schemes** across 6 populations; note length 5.0 [4.5, 5.6] words (first third) → 4.7 [4.2, 5.3] (last third); modal description matches a zero-shot prior probe in 1.00 [1.00, 1.00] of items; coined non-word labels: 138.
- **E3 redo (chooser reply-token repair)**: 6.0 [6.0, 6.0] of 6 items per population settled on one description (mean per-item modal share 0.99 [0.98, 1.00]); **4 distinct population-level description schemes** across 4 populations; note length 5.3 [5.0, 5.6] words (first third) → 5.2 [4.8, 5.7] (last third); modal description matches a zero-shot prior probe in 1.00 [1.00, 1.00] of items; coined non-word labels: 346.
- **E3 squeeze (3-word notes, hard distractors, shared items)**: 6.0 [6.0, 6.0] of 6 items per population settled on one description (mean per-item modal share 0.99 [0.99, 1.00]); **1 distinct population-level description schemes** across 12 populations; note length 3.0 [3.0, 3.0] words (first third) → 3.0 [3.0, 3.0] (last third); modal description matches a zero-shot prior probe in 0.99 [0.96, 1.00] of items; coined non-word labels: 0.
- **E4 (tagged bargaining) — VOID live, see integrity note**: equilibrium types across 6 populations: {'fractious': 6}; class (badge-conditioned 70/30) share 0.00 [0.00, 0.39] (0/6); egalitarian share 0.00 [0.00, 0.39] (0/6); tail compatibility 0.65 [0.60, 0.69].
- **E4 + scratch line (corrected turn protocol)**: equilibrium types across 12 populations: {'egalitarian': 10, 'other_stable_50_50': 1, 'fractious': 1}; class (badge-conditioned 70/30) share 0.00 [0.00, 0.24] (0/12); egalitarian share 0.83 [0.55, 0.95] (10/12); tail compatibility 0.96 [0.95, 0.98].
- **E4 + scratch line, claude-sonnet-4.5**: equilibrium types across 3 populations: {'egalitarian': 3}; class (badge-conditioned 70/30) share 0.00 [0.00, 0.56] (0/3); egalitarian share 1.00 [0.44, 1.00] (3/3); tail compatibility 1.00 [1.00, 1.00].

### Second model family (openai/gpt-5-mini, frozen prompts)
- **E1 (12-word notes)**: 0.000 [0.000, 0.324] (0/8) of populations conventionalized on a note ordering (modal share 0.13 [0.06, 0.21]); **0 distinct modal orderings** across 8 populations (Simpson diversity 0.00); inter-agent agreement 0.01 [0.00, 0.02]; task success (tail) 0.22 [0.05, 0.42]; newcomer first-3-notes adoption of the incumbent ordering 0.00 [0.00, 0.01] vs founders' first-3 0.12 [0.04, 0.21]; scripted minority (f=0.25, fixed alternative ordering) flipped 0.00 [0.00, 0.32] (0/8). Shuffle control modal share 0.06 (97.5th pct 0.08).
- **E1 squeeze (6-word notes)**: 0.000 [0.000, 0.324] (0/8) of populations conventionalized on a note ordering (modal share 0.19 [0.04, 0.43]); **0 distinct modal orderings** across 8 populations (Simpson diversity 0.00); inter-agent agreement n/a; task success (tail) 0.01 [0.00, 0.01]; newcomer first-3-notes adoption of the incumbent ordering 0.00 [0.00, 0.00] vs founders' first-3 0.32 [0.11, 0.66]; scripted minority (f=0.25, fixed alternative ordering) flipped n/a.
- **E2 + scratch line (corrected turn protocol)**: 0.000 [0.000, 0.390] (0/6) of populations conventionalized a badge→region mapping (mapping share 0.00 [0.00, 0.00]); episode success first-12 0.04 [0.01, 0.07] → last-12 0.06 [0.03, 0.08]; 0 distinct modal mappings: []; malformed/call 0.05 [0.04, 0.05] (parse failures 0.00 [0.00, 0.00]); message-channel use 0.00 [0.00, 0.00].
- **E3 squeeze (3-word notes, hard distractors, shared items)**: 3.4 [2.2, 4.4] of 6 items per population settled on one description (mean per-item modal share 0.52 [0.49, 0.56]); **7 distinct population-level description schemes** across 8 populations; note length 2.2 [2.1, 2.3] words (first third) → 2.1 [2.0, 2.2] (last third); modal description matches a zero-shot prior probe in 0.19 [0.12, 0.25] of items; coined non-word labels: 5.
- **E4 + scratch line (corrected turn protocol)**: equilibrium types across 6 populations: {'egalitarian': 5, 'fractious': 1}; class (badge-conditioned 70/30) share 0.00 [0.00, 0.39] (0/6); egalitarian share 0.83 [0.44, 0.97] (5/6); tail compatibility 0.96 [0.94, 0.98].

### Integrity correction — supersedes the first live E2/E4 reading

After the first live pass was written up, probes of raw replies showed
the original live E2 and E4 cells were **invalid**: with their short
reply caps (20 tokens per E2 turn, 8 per E4 demand, 8 per E3 pick) the
model's free-text preamble was truncated before any parseable move, so
**98.1% (E2) and 98.8% (E4) of live moves fell to the deterministic
random fallback** (E3's chooser: 36.3%; E1 unaffected — answer accuracy
0.74–0.99). Those cells measured the fallback distribution, not model
behaviour; their rows above are marked VOID (journals retained), and
plain-instruction re-probes at larger caps showed the model simply does
not comply with "reply with only X" once it has room to chat (0/6
parseable E2 replies at a 200-token cap). The corrected protocol asks
for one scratch line then the move on its own line, gives it room, and
parses the final anchored answer; parse failures are journaled
separately from occupied-cell picks. The substrate tier is unaffected
(mock replies are well-formed by construction).

### Reading (final n; supersedes every pilot number above and below)

**Superseded in turn by the review-response experiments (§10): the
gpt-5-mini E3 diversity, initially read as the suite's first live
side-product convention, failed all four causal tests — its 7/8
distinct schemes are consistent with the no-interaction sampling null
(P(≥7) = 0.14 under the model's independently measured flat prior,
1.39 bits), transplanted agents never adopt the host scheme (0/10),
schemes are fragile under turnover (2/8 survive), and cross-population
pairings succeed at least as well as within-population ones (swap
Δ = −0.08): the diversity is correlated stylistic drift, not a
solution to a coordination problem. The corrected overall finding is
that NO live environment produced a genuine side-product convention.
In the haiku family the 6-word squeeze yields one weakly
conventionalized population in 12 (agreement 0.13 vs shuffle 0.08);
everything else is prior-driven. The independently measured axes (§10)
explain the whole pattern: both families are fully competent to
execute optimal codes when specified (reachability 0.97–1.00), so
non-formation is never a competence ceiling — it is priors that either
close the choice space (haiku, E3 modal probability 0.97) or fill it
with payoff-irrelevant variation (gpt-5-mini), while flexible responder
parsing removes the compatibility payoff that would make history
matter.**

- *E1 squeeze (final n=12):* 6 words cannot name all 5 label+entry
  pairs, so naming everything stops being free. Haiku triages — names
  ~2.7 of 5 entries and accepts ~40–65% success (vs ~94% at 12 words) —
  rather than inventing the values-only positional code that would fit
  all 5 (exactly the ordering convention the substrate exploits). At
  final n, 1 of 12 populations conventionalized a shared
  label-subset-and-order (modal share 0.17 [0.11, 0.25] vs shuffle
  0.08; agreement 0.13 vs 0.03 at 12 words): real but rare — the
  pilot's 1-in-4 was an early read on the same single population.
  Memory doubling and channel noise produced nothing (pilot cells,
  n=4 each). In the substrate the squeeze *destroys* conventions
  (0/40) — clipped notes starve its imitation channel. gpt-5-mini
  fails differently and instructively: its populations bifurcate
  between plain `label=value` enumeration (success 0.83) and invented
  pseudo-ciphers ("entry = label shifted two letters") that transmit
  nothing (success ~0.02, wellformed notes 0–4%) — over-engineered
  encodings, journaled verbatim; its 12-word cell replicates the
  E1 null (0/8, agreement 0.01).
- *E2 (final n=10+10 + families):* real grid play is *worse* than the
  malformed-era random fallback (success 0.03–0.14 vs ~0.38): both
  agents chase the same salient cells. Haiku fails via ~40% occupied
  picks (misread grids); sonnet reads the grid near-perfectly (2%
  malformed) yet collides 5–6 times per episode — two copies of one
  deterministic policy are a mirror match, and extra capability
  sharpens the mirror; gpt-5-mini replicates (0/6, success ~0.05). No
  badge→region convention forms in any cell of any family (0/29
  corrected populations).
- *E2 dialogue (n=10):* the message channel is used in 100% of episodes
  and lifts success (last-12 0.14 vs 0.04 without), but pacts never
  fossilize into a population convention: both partners propose plans
  *simultaneously* each episode, the proposals conflict (each typically
  assigns itself the same role), and partners rotate every episode, so
  no badge-anchored mapping stabilizes (0/10).
- *E3 (final n=12 + second family):* the haiku cells confirm the
  shared-bias classification — e3_redo reproduces exhaustive
  description (per-item share 0.99, prior-match 1.00), and on the
  shared item set all 12 e3_squeeze populations settle on the *same*
  scheme (1 distinct). **gpt-5-mini inverts this**: on the identical
  items, its 8 populations concentrate within-population (per-item
  modal share 0.52, task success 0.68–0.92, well above haiku's
  0.48–0.63) while settling on 7 *distinct* schemes across populations
  with prior-match 0.19 — population-specific description conventions,
  the pre-registered convention signature (substrate: 40/40 distinct).
  The contrast localizes the mechanism: with hard distractors and a
  3-word budget, several minimal discriminating descriptions exist per
  item; a model competent enough to find them, whose prior does not
  privilege one, lets interaction history pick — and different
  populations pick differently.
- *E4 (final n=12 + families):* with parseable replies, 11 of 12 haiku
  populations stabilize at 50/50 demands (10 egalitarian + 1 other
  stable-50/50; compatibility 0.96), sonnet 3/3 and gpt-5-mini 5/6
  egalitarian. Across 21 corrected live populations and three models:
  **zero** Axtell–Epstein–Young badge-conditioned class conventions
  (substrate: 9/60). The fairness prior absorbs the symmetry instead of
  breaking it.

**Synthesis (final).** Conventions form in an LLM population exactly
when the individually-optimal prior policy leaves a residual
coordination problem that only shared history can solve. When the prior
solves the encounter alone — flexible parsing (E1 at 12 words, all
families), exhaustive description (haiku E3), fairness (E4, all three
models), salience-chasing (E2, all three models) — history has nothing
to grab and no convention forms, however much co-presence, turnover, or
even explicit negotiation is supplied. When capacity pressure makes the
prior strategy infeasible but the model cannot construct an
alternative, it degrades instead of conventionalizing (haiku E1
squeeze: rare weak conventions; gpt-5-mini E1 squeeze: pseudo-cipher
collapse). And when the prior does not single out a point on the
solution manifold — gpt-5-mini's flat E3 prior — populations diverge,
but the review-response tests (§10) show that divergence is sampling
drift with no social glue: without a compatibility payoff (swap
Δ ≈ 0 or negative) there is nothing for transmission to preserve
(transplants 0/10, turnover survival 2/8). Payoff-coupled alignment
(the naming game, all families) remains the one sufficient condition
observed; prior-underdetermined competence produces variation but not,
in these environments, convention. Limitations: 6–12 live populations
per load-bearing cell, n=3–8 on secondary cells, short in-context
histories, three models from two vendors.


## 8. Honesty table (pre-registered vs exploratory)


| Result family | Status | Basis |
|---|---|---|
| Naming-game consensus / conformity / survival / flip criteria | **Pre-registered** | §1: criteria fixed before any data collection; unchanged throughout, including the 2,000-interaction minority sweep (same flip criterion, longer budget = separate cells) |
| Experiment A sweeps (turnover phase boundary, minority thresholds, transplant, N-sweep) | **Pre-registered protocol** | Substrate policy choice documented pre-run (decision log #1) |
| Experiment B genesis / transmission / dialogue ablation | **Pre-registered** | Same criteria as A; prompts frozen after the documented pilot stage (decision log #12) |
| Judge four-way taxonomy | **Failed instrument** | Declared unreliable on this corpus (κ = 0.06); category shares appendix-only |
| Surface-feature enforcement detector | **Exploratory, pre-frozen** | Lexicons and rules committed before any corpus scan; validated on the 151-message sample first |
| E1–E4 environments and all levers (squeeze, memory, noise, dialogue, scratch line, model tier) | **Exploratory** | Designed after the naming-game results; controls (fresh tokens, banned vocabulary, priors, shuffle, stranger pool) applied throughout |
| Reply-cap repair cycle (VOID e2/e4 cells → scratch-line protocol) | **Exploratory, documented repair** | Decision log #14; voided journals retained |
| Second model family (gpt-5-mini) | **Confirmatory replication** | Frozen prompts and gates; run after all haiku results were in |


## 9. Claims → evidence map


| Paper claim | Cells (final n) | Figures / numbers |
|---|---|---|
| LLM populations form arbitrary conventions when payoff rewards alignment directly, and they persist under 100% turnover (positive control) | b_genesis (n=15); b_trans_dlg / b_trans_nodlg (n=10/10) | fig_convergence, fig_survival; `\pNamingGenesisConv`, `\pNamingSurvivedDlgOn/Off` |
| Newcomer socialisation is dialogue-mediated | b_trans_dlg vs b_trans_nodlg | fig_socialisation; `\pSocialisationMWU` |
| Substrate committed-minority threshold f50 ≈ 0.126, history-independent | expA minority cells (n=500/point) | fig_minority; `\fFiftySubstrateFounder/Posttrans` |
| Live minority threshold: f50 or lower bound vs substrate | b_minority_{founder,posttrans}_{f25,f33,f42}_b2k (n=8 each) | fig_minority |
| Compression produces at most one weak live side-product convention (loose-vs-squeeze contrast) | e1_tight (n=12) vs e1_squeeze (n=12) live; substrate n=60/40 | fig_inversion, fig_esuite; `\convEOneSqueezeLive` |
| E3 concentration (haiku) is shared model bias; E3 diversity (gpt-5-mini) is prior-sampling drift, not convention — no adoption, no turnover survival, no swap cost | e3_squeeze (n=12 haiku, 8 m2) + §10 null baseline, transplants, swaps, turnover | fig_esuite, fig_review_e3; `\schemesEThreeSqueezeLive`, review JSONs |
| The fairness prior absorbs the symmetry that produces Axtell–Epstein–Young classes in the substrate | e4_think (n=12), e4_sonnet (n=3), substrate e4 (n=60) | fig_inversion; `\classEFourThinkLive`, `\egalEFourThinkLive` |
| Capability sharpens priors: mirror-match coordination failure; negotiated pacts do not fossilize | e2_think (n=10), e2_dialogue (n=10), e2_sonnet (n=3) | §7 E2 rows; `\convETwoThinkLive`, `\succETwoDialogueLive` |
| Enforcement surface features are absent (annotation-invariant) in populations and solitary alike | detector scan over all live dialogue | `\prevDeonticPop` etc.; enforcement_analysis.json |
| Findings hold across model families | m2 cells: naming genesis n=8, trans n=6; e1 (8/8), e3 (8), e4 (6), e2 (6) | §7 m2 tier rows; `\convEOneSqueezeMTwo` etc. |


## 10. Review-response experiments

All criteria below were frozen in decision-log entries 16–23 and committed before the corresponding data were collected.

### Scheme-coding specification and grain-sensitivity (entries 16)
A population's scheme is its 6-tuple of per-item modal trait-sets (last
15 formation mentions per item). Distinct-scheme counts under all three
frozen rules — A (exact tuple), B (mean per-item Jaccard ≥ 0.5
clustering), C (any-item difference):
| family | n | rule A | rule B | rule C |
|---|---|---|---|---|
| substrate | 40 | 40 | 1 | 40 |
| haiku | 12 | 1 | 1 | 1 |
| gpt-5-mini | 8 | 7 | 1 | 7 |

The m2-vs-haiku contrast (7/8 vs 1/12) is grain-stable across rules A
and C. Rule B collapses **every** family to one cluster — including the
substrate positive control whose 40 distinct schemes are uncontested —
so it cannot detect any convention diversity at that threshold; it
bounds the taxonomy's coarse end rather than undermining the count.

### No-interaction diversity baseline (entry 17)
16 pseudo-populations per family of independent zero-shot generations
(shared items, frozen prompt, empty history, matched per-item event
counts, run temperature), identical extraction. Null distribution of
distinct-scheme counts among resampled groups of 8 (rule A):
gpt-5-mini mean 5.54,
P(≥7 distinct) = 0.1445;
haiku mean 1.00,
P(≥7) = 0.0.

### Transplant (entry 18)
10 transplants across distinct-scheme ordered pairs: adopted
0/10 (criterion: ≥80% of last 10 sends matching the host
modal); mean last-10 match share 0.43; median latency
n/a sends; host items stable
4.0/6.

### Cross-population swap (entry 19)
- **gpt-5-mini**: within-population success 0.762 vs cross-population 0.842 (Δ = -0.080).
- **haiku control**: within-population success 0.542 vs cross-population 0.547 (Δ = -0.006).
The haiku prediction (single shared scheme → no swap cost) is tested and reported as such.

### E3-m2 turnover (entry 23; computed on the existing generation of
replacement contained in the e3_squeeze journals)
Scheme survival 2/8 populations (≥4/6 items
stable; mean 2.75/6); newcomer mean match share
0.356. Unlike naming-game conventions (perfect
survival at matched turnover), m2 description schemes are **fragile
under full population replacement** — consistent with their weak
within-population concentration (0.52): the convention is real
(diversity + swap) but noisily transmitted.

### Independent axis measurements (entry 21)
- **haiku**: reachability — E1 values-only code 0.9666666666666667, E3 specified scheme 1.0 (n=30); prior concentration (E3, 50 samples/item) — mean entropy 0.15 bits, mean modal probability 0.97.
- **gpt-5-mini**: reachability — E1 values-only code 1.0, E3 specified scheme 1.0 (n=30); prior concentration (E3, 50 samples/item) — mean entropy 1.39 bits, mean modal probability 0.67.
These probes assign the quadrant axes from measurements independent of population outcomes.

### Equivalence bounds for the null history effect (§6)
No founder-vs-post-transmission difference detected in either tier;
effects up to Δ remain compatible with the data — substrate: Δf₅₀ ∈
[-0.0006, +0.0003]
(bootstrap); live: Δf₅₀ ∈ [-0.080,
+0.064] (founder CI × separation
interval).

### Survival-figure units (entry 22)
In the simulator, one slot is replaced every k interactions (r slots
per event when k=1), so a generation is N·k interactions; an agent
participates in 2/N of interactions, giving an expected lifetime of
≈ 2k plays. k is therefore INTERACTIONS PER REPLACEMENT (larger k =
slower turnover; k<1 encodes r=1/k replacements per interaction). The
earlier axis label inverted this; figure, caption, and text now agree.
At the measured boundaries, conventions survive one generation down to
k₅₀ ≈ 0.25 (lifetime ≈ 0.5 plays) and two generations at k₅₀ ≈ 0.71.

### Capability-claim scoping (§6)
The E2 capability observation (stronger model → sharper mirror match)
is a tested-model contrast — haiku-4.5 vs sonnet-4.5, direction
replicated in gpt-5-mini — not a general capability law; no claim
beyond the tested models is made.

### E2 mitigation cells (entry 20)
Exogenous role line: success first-12 0.25 [0.18, 0.33] → last-12 0.14 [0.07, 0.21] vs e2_think baseline last-12 0.04 [0.02, 0.07] (n=8).
Heterogeneous pairing (haiku × gpt-5-mini): success first-12 0.08 [0.03, 0.15] → last-12 0.06 [0.03, 0.10] (n=8).
