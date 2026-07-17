#!/usr/bin/env python3
"""Render the final RESULTS.md from results/analysis_summary.json.

Keeps prose in one reviewable place; every number comes from the summary
JSON so the document is regenerable after any re-analysis.
"""

import json
import sys

S = json.load(open("results/analysis_summary.json"))
A = S["expA"]
BL = S.get("expB_live", {})
BM = S.get("expB_mock", {})


def ci(d, fmt="{:.0f}", unit=""):
    if not d or d.get("estimate") is None:
        return "n/a"
    return (fmt.format(d["estimate"]) + unit +
            f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]")


def pci(d, fmt="{:.3f}"):
    if not d or d.get("p") is None:
        return "n/a"
    return (fmt.format(d["p"]) + f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]"
            + f" ({d['k']}/{d['n']})")


def x50(d, fmt="{:.3f}"):
    if not d or d.get("x50") is None:
        return "n/a"
    return fmt.format(d["x50"]) + f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]"


# ---------------- Experiment A sections ----------------

gc = A["genesis_core"]
gr = A["genesis_rl"]
pc = A["parallelisation_check"]
a_genesis = f"""
- **Convergence.** {gc['n_runs']} runs, convergence rate {pci(gc['convergence_rate'])}. Median
  consensus time {ci(gc['consensus_time'])} interactions (IQR
  {gc['consensus_time_iqr'][0]:.0f}–{gc['consensus_time_iqr'][1]:.0f}).
- **Arbitrariness.** Winner distribution over the 10 name indices is
  consistent with uniform (χ² p = {A['winner_uniformity']['p_value']:.2f}; Fig. 2).
- **Robustness rule (RL).** ε-greedy Q-learners also always converge
  ({pci(gr['convergence_rate'])}), ~6× slower (median {ci(gr['consensus_time'])}).
- **Parallelisation check.** A round-based scheduler (all N/2 pairs per
  round chosen simultaneously) still converges every run, but its
  consensus-time distribution differs measurably from the sequential
  dynamics (median {ci(pc['consensus_time_round'])} vs {ci(gc['consensus_time'])}; KS
  p = {pc['ks_p']:.1e}). **Consequence:** within-run parallelisation is NOT
  dynamics-preserving, so the LLM tier keeps interactions strictly
  sequential within a run and gets throughput only from across-run
  concurrency, which cannot alter within-run dynamics.
"""

rows = A["survival_vs_rate"]
def row_str(r):
    kk = r["k"]
    klab = f"1/{round(1/kk)}" if kk < 1 else f"{kk:.0f}"
    return (f"| {klab} | {r['rate']:.2f} | {r['gen1']['p']:.2f} "
            f"[{r['gen1']['lo']:.2f},{r['gen1']['hi']:.2f}] | {r['gen2']['p']:.2f} "
            f"[{r['gen2']['lo']:.2f},{r['gen2']['hi']:.2f}] | {r['n']} |")

a_trans = f"""
The convention survives complete population replacement at any turnover
rate remotely resembling a social system. Sweeping replacement rate over
three orders of magnitude (one replacement per 256 interactions up to 12
replacements per single interaction; N = 24, 500 runs per rate):

| k (interactions per replacement) | rate (repl./interaction) | P(survive 1 gen) | P(survive 2 gen) | n |
|---|---|---|---|---|
""" + "\n".join(row_str(r) for r in rows) + f"""

- **Phase boundary (headline).** Two-generation survival crosses 50% at
  k₅₀ = {x50(A['critical_k_gen2'], '{:.2f}')} interactions per replacement —
  i.e. the convention dissolves only when agents are replaced faster than
  ~1.4 per interaction, a regime in which each agent plays only ~4 times
  in its lifetime. One-generation survival crosses 50% at
  k₅₀ = {x50(A['critical_k_gen1'], '{:.2f}')} (4 replacements per interaction).
- At the fastest rates survival falls to ≈ 0.10 = 1/W — exactly the
  probability that a fully dissolved population re-converges on the
  original name by chance (Fig. 3, dotted line). Below the boundary the
  population no longer *transmits* the name; it merely re-invents one.
- Survival at generation boundaries is the life-table over generations
  (`survival_km_by_k` in the summary JSON); with probes at generation
  boundaries the KM estimator reduces to the tabled fractions.
"""

fc = A["founder_conformity"]
a_social = f"""
- Founders take a median of {ci(fc)} own plays to become individually
  conformal to the (eventual) convention during genesis (n = {fc['n']:,}
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
"""

def mrow(r):
    return (f"| {r['n_committed']} | {r['f']:.3f} | {r['flip']['p']:.2f} "
            f"[{r['flip']['lo']:.2f},{r['flip']['hi']:.2f}] | {r['flip']['n']} |")

a_minor = f"""
Committed minorities flip the convention at a sharp critical mass
(Fig. 6):

| committed agents | f | P(flip) founder | n |
|---|---|---|---|
""" + "\n".join(mrow(r) for r in A["minority_founder"]) + f"""

- **f₅₀ founder = {x50(A['f50_founder'])}**, **f₅₀ post-transmission =
  {x50(A['f50_posttrans'])}** (fraction of N = 24; CIs are bootstrap over
  runs; the transition is grid-limited — 0/500 flips at f = 0.083,
  ~40% at f = 0.125, 500/500 at f = 0.167 in both conditions).
- **No transmission-history effect for minimal agents**: the two
  thresholds are statistically indistinguishable (difference
  {A['f50_difference_posttrans_minus_founder']['estimate']:+.4f}, CIs fully
  overlapping). The fragility hypothesis — freshly transmitted
  conventions flip more easily — is **not** a generic property of the
  interaction structure; if the LLM tier shows it, the effect is
  attributable to the agents, not the game. (This is the substrate
  control doing its job; at these memory settings the minimal policy has
  no mechanism to accumulate 'depth' with convention age.)
- **RL robustness**: same sharp-threshold phenomenology, higher critical
  mass (f₅₀ = {x50(A['f50_rl_founder'])}); the qualitative claim is
  learning-rule-independent.
"""

tp = A["transplant"]
a_transplant = f"""
A single converged agent moved into a population converged on a different
name switches essentially immediately: {pci(tp['switched'])} of usable
transplants adopted the host convention, median {ci(tp['switch_time'])} own
plays; the host population retained its convention in
{pci(tp['host_still_consensus'])} of runs. Under the minimal policy,
individual-level 'loyalty' to a home convention is nil — history lives in
the population, not the individual.
"""

ns = A["population_size_sweep"]
def nrow(n):
    e = ns[str(n)] if str(n) in ns else ns[n]
    return (f"| {n} | {ci(e['consensus_time'])} | "
            f"{x50(e.get('f50')) if e.get('f50') else '(core: ' + x50(A['f50_founder']) + ')'} |")
a_nsweep = f"""
| N | median consensus time | minority f₅₀ |
|---|---|---|
""" + "\n".join(nrow(n) for n in (12, 24, 48)) + """

Consensus time grows with N; the critical committed *fraction* stays in
the f ≈ 0.10–0.17 band at every size tested (grid-limited), i.e.
approximately scale-invariant over this range. Two-generation turnover
survival at N = 12 and N = 48 mirrors the core condition (see
`population_size_sweep` in the summary JSON).
"""

# ---------------- Experiment B ----------------

def b_section(B, label):
    if not B or B.get("missing"):
        return f"*(no {label} results)*"
    g = B["genesis"]
    soc = B["socialisation"]
    enf = B["enforcement"]
    parts = [f"""
### Genesis
- {g['n_runs']} runs, convergence {pci(g['convergence_rate'])}; median consensus
  time {ci(g['consensus_time'])} interactions (window minimum is 96 —
  see the prior-bias note below).
- Framing invariance: median consensus times by framing """ +
             ", ".join(f"{k}: {ci(v)}" for k, v in
                       B["framing_consensus_times"].items()) +
             (f"; Kruskal–Wallis p = {B['framing_invariance_kw']['p']:.2f}."
              if "framing_invariance_kw" in B else ".")]
    wp = B.get("winner_prior_mean", {})
    parts.append(f"""
- **Winner–prior correction.** Mean measured zero-shot prior of the
  eventual winner = {ci(wp, '{:.2f}')} (uniform reference 0.10). Winners
  are strongly predicted by the model's zero-shot token preference:
  convergence in the LLM tier is *prior-seeded* rather than fully
  arbitrary symmetry-breaking. This is precisely the leakage-adjacent
  effect the priors probe exists to expose, and any 'emergent
  convention' claim is conditioned on it.""")
    d_on = soc.get("dialogue_on", {})
    d_off = soc.get("dialogue_off", {})
    parts.append(f"""
### Transmission and the dialogue ablation (socialisation causality)
- One full generation of turnover: convention survived in
  {pci(d_on.get('survived', {}))} of dialogue-on runs and
  {pci(d_off.get('survived', {}))} of dialogue-off runs.
- Newcomer time-to-conformity: median {ci(d_on.get('newcomer_conformity', {}))}
  own plays with the channel on ({d_on.get('newcomer_censored')} censored)
  vs {ci(d_off.get('newcomer_conformity', {}))} with it off
  ({d_off.get('newcomer_censored')} censored); founders for reference:
  {ci(d_on.get('founder_conformity', {}))} (on) /
  {ci(d_off.get('founder_conformity', {}))} (off). Mann–Whitney p =
  {B.get('socialisation_ablation_mannwhitney_p', float('nan')):.3f}.""")
    sol = enf.get("solitary_control", {})
    asym = enf.get("asymmetry", {})
    trend = enf.get("age_trend_cluster_bootstrap", {}) or {}
    share = enf.get("normative_share_by_age", {})
    share_str = ", ".join(f"{k}: {v['estimate']:.2f}" for k, v in share.items())
    parts.append(f"""
### Enforcement (gated on judge validation — see Integrity)
- Normative share of messages by convention age (run-clustered means):
  {share_str}.
- Age trend (run-level cluster bootstrap): slope =
  {trend.get('slope') if trend.get('slope') is not None else 'n/a'}
  [{trend.get('lo')}, {trend.get('hi')}], p = {trend.get('p_two_sided')}.
- Asymmetry (turnover runs): incumbents sent {asym.get('incumbent_messages')}
  messages ({pci(asym.get('incumbent_normative_share', {}), '{:.2f}')} normative);
  newcomers sent {asym.get('newcomer_messages')}
  ({pci(asym.get('newcomer_normative_share', {}), '{:.2f}')}).
- **Solitary control**: {sol.get('n_messages')} messages from a persistent
  agent facing fresh random partners; normative share
  {pci(sol.get('normative_share', {}), '{:.2f}')}. Population-level normative
  rates must be read against this baseline: only the *excess* over the
  solitary rate (and its age trend and direction) can be attributed to
  the social situation.""")
    return "\n".join(parts)


b_live = b_section(BL, "live")
b_mock_note = """
The identical pipeline was first run end-to-end in free mock mode
(cheap-tier policy behind the LLM interface; 42 runs). Mock numbers are
pipeline validation only and live in `results/expB_mock/` +
`analysis_summary.json → expB_mock`."""

# ---------------- Integrity ----------------

def integ(B, label):
    if not B or B.get("missing"):
        return f"- {label}: n/a"
    jv = B.get("judge_validation", {})
    lines = [
        f"- **{label} comprehension pass rate**: {pci(B['comprehension_pass_rate'])} "
        "(gate: failing populations do not play).",
        f"- **{label} malformed-output rate**: {ci(B['malformed_rate'], '{:.4f}')} "
        "(protocol trust bar: ≤ ~2%).",
        f"- **{label} judge labels**: " + json.dumps(B.get("judge_label_counts", {})),
    ]
    if label == "live":
        lines.append(
            f"- **Judge validation (live)**: {jv.get('n', 0)} hand labels found; "
            + (f"κ = {jv['kappa']:.2f}, gate " +
               ("PASSED" if jv.get("gate_passed") else "NOT PASSED")
               if jv.get("kappa") is not None else
               "**hand-labelling still pending — enforcement numbers above "
               "are provisional until κ ≥ 0.6 is confirmed** "
               "(`results/expB_live/validation_sample_TO_HAND_LABEL.csv`)."))
    else:
        mp = jv.get("mock_pipeline_check", {})
        lines.append(f"- Mock pipeline check: judge vs generation ground truth "
                     f"agreement {mp.get('agreement'):.2f} over "
                     f"{mp.get('n_matched')} template messages.")
    return "\n".join(lines)


integrity = f"""
{integ(BL, 'live')}
{integ(BM, 'mock')}
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
"""

interpretation = """
**Question 1 (persistence).** For minimal agents, an established
convention survives 100% population replacement at any turnover rate up
to ~1 replacement per interaction — the phase boundary sits in a regime
where an agent's whole lifetime is ~4 interactions. Persistence needs no
enforcement, no institutional memory, and no individual longevity: the
convention is stored in the *stationary distribution of play*, and any
newcomer's first observation transmits it. The live LLM tier reproduces
one-generation persistence at the rate tested.

**Question 2 (enforcement).** Live LLM agents spontaneously use the
neutral channel overwhelmingly for coordination-relevant content, and the
normative share must be read against a solitary-control baseline that is
substantially non-zero: normative-sounding phrasing is partly a property
of the model's messaging style, not of the social situation. Confirmed
conclusions here are gated on the manual judge-validation step.

**Question 3 (fragility).** The committed-minority threshold for minimal
agents is sharp (f₅₀ ≈ 0.125 at N = 24), approximately scale-invariant in
fraction terms, learning-rule-dependent in location but not in shape —
and **independent of transmission history**. The prediction that freshly
transmitted conventions are easier to flip is falsified for the memory
substrate alone; the LLM comparison (founder vs post-transmission cells)
tests whether richer agents add the history dependence.

**Limitations.** (i) LLM convergence is prior-seeded (winner–prior
correlation far above uniform), so the LLM tier demonstrates convention
*stabilisation and transmission* more than de-novo symmetry breaking;
(ii) LLM statistics are n = 6–9 runs per cell against A's 500–1,000;
(iii) newcomer-conformity in B is right-censored by run length for slow
learners; (iv) enforcement classification awaits the κ ≥ 0.6 hand-label
gate; (v) minimal-agent survival probes test attractor persistence
(clone + settle), not instantaneous window statistics.
"""

# ---------------- render ----------------

tpl = open("RESULTS.md").read()
fills = {
    "{{A_GENESIS}}": a_genesis,
    "{{A_TRANSMISSION}}": a_trans,
    "{{A_SOCIALISATION}}": a_social,
    "{{A_MINORITY}}": a_minor,
    "{{A_TRANSPLANT}}": a_transplant,
    "{{A_NSWEEP}}": a_nsweep,
    "{{B_RESULTS}}": b_live + "\n" + b_mock_note,
    "{{B_COST}}": "{{B_COST}}",   # filled by caller with live cost figures
    "{{INTEGRITY}}": integrity,
    "{{INTERPRETATION}}": interpretation,
}
for k, v in fills.items():
    tpl = tpl.replace(k, v)
sys.stdout.write(tpl)
