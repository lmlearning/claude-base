# Results Summary — Norm Persistence, Enforcement, and Adversarial Fragility

*(Numbers below are filled from `results/analysis_summary.json`; figures in
`figures/`. This document also serves as the decision log required by the
protocol: every underdetermined choice is recorded here and was held fixed
across conditions.)*

## Status at a glance

| Item | Status |
|---|---|
| Experiment A (minimal agents), full P0+P1 | **Complete** (~20k runs) |
| Experiment B pipeline, end-to-end in free mock mode | **Complete** (42 runs, all phases, judge, validation export) |
| Experiment B live LLM runs | **Blocked: no `ANTHROPIC_API_KEY` in this environment.** Cost projection ≈ $100 (upper bound) against a $250 in-code cap; one command to launch (`python -m namegame expb --mode live`). |
| Judge hand-label validation | Awaiting live data (mock pipeline check passed; see Integrity) |

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

## 3. Experiment A results (minimal agents; the substrate control)

*Filled from `results/analysis_summary.json` → `expA`.*

### Genesis
{{A_GENESIS}}

### Persistence under turnover (Question 1)
{{A_TRANSMISSION}}

### Newcomer socialisation
{{A_SOCIALISATION}}

### Committed minority: founder vs post-transmission (Question 3)
{{A_MINORITY}}

### Transplant
{{A_TRANSPLANT}}

### Population-size sweep
{{A_NSWEEP}}

## 4. Experiment B (LLM tier)

Live runs require an API key not present in this environment; the entire
pipeline was exercised end-to-end in mock mode (cheap-tier policy behind
the LLM interface — prompts built, replies parsed, journals written,
judge run, validation sample exported). Mock results validate the
*pipeline*, not LLM behaviour, and are labelled as such throughout.

{{B_RESULTS}}

### Cost projection (live)
{{B_COST}}

## 5. Integrity results
{{INTEGRITY}}

## 6. Interpretation and limitations

{{INTERPRETATION}}
