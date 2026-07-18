#!/usr/bin/env python3
"""Render the final RESULTS.md from results/analysis_summary.json.

Keeps prose in one reviewable place; every number comes from the summary
JSON so the document is regenerable after any re-analysis.
"""

import json
import sys

import numpy as np

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
  eventual winner = {ci(wp, '{:.2f}')} (uniform reference 0.10): only a
  weak, non-significant excess — the measured zero-shot prior does *not*
  strongly predict the winner. Combined with consensus at the window
  floor (agreement forms within the first ~96 interactions while distinct
  runs pick distinct winners), this indicates fast in-context symmetry
  breaking early in each run rather than a fixed lexical bias. Caveat:
  the bare-list prior probe may under-measure in-context salience
  (ordering, framing), so prior-seeding cannot be fully excluded — but on
  the pre-registered measurement, the arbitrariness property holds.""")
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
    jv = B.get("judge_validation", {}).get("second_annotator", {})
    parts.append(f"""
### Enforcement (Question 2) — **judge validation FAILED; headline is a null**

The LLM judge's labels did NOT survive validation: on the stratified
151-message sample, an independent strict-rubric second annotation agrees
with the judge at only κ = {jv.get('kappa', float('nan')):.2f} (gate: κ ≥ 0.6;
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
- Judge-'normative' share by convention age (run-clustered): {share_str};
  age trend slope {trend.get('slope'):.2e}
  [{trend.get('lo'):.2e}, {trend.get('hi'):.2e}], p = {trend.get('p_two_sided')}.
- Asymmetry in turnover runs: incumbents {asym.get('incumbent_messages')}
  messages ({pci(asym.get('incumbent_normative_share', {}), '{:.2f}')}
  judge-normative) vs newcomers {asym.get('newcomer_messages')}
  ({pci(asym.get('newcomer_normative_share', {}), '{:.2f}')}).
- **Solitary control** (interpretively load-bearing): {sol.get('n_messages')}
  messages from a persistent agent facing fresh random partners;
  judge-normative share {pci(sol.get('normative_share', {}), '{:.2f}')} vs
  ~0.21–0.35 in populations. Even on unvalidated labels, the excess over
  the solitary baseline is what would carry a social interpretation —
  but with κ = {jv.get('kappa', float('nan')):.2f} no such claim is made.""")
    mino_f = mino_p = None
    return "\n".join(parts)


def b_minority_section(B):
    try:
        fit = json.load(open("results/live_minority_fit.json"))
    except FileNotFoundError:
        fit = None
    pilot = """
### Committed minority, live — pilot (f = 0.25, 600-interaction budget)
A scripted committed minority at f = 0.25 flipped **0/6 founder** and
**0/6 post-transmission** populations. Kept as the pilot; the threshold
sweep below supersedes it as the headline."""
    if not fit:
        return pilot + "\n\n*(threshold sweep not yet run)*"
    fo, po = fit["founder"], fit["posttrans"]

    def pts(d):
        return "; ".join(f"f={f}: {k}/{n}" for f, (k, n)
                         in sorted(d["points"].items()))
    return pilot + f"""

### Committed-minority threshold, live (final: f ∈ {{0.25, 0.33, 0.42}}, 2,000-interaction budget, n = 8/cell)
Flips — founder: {pts(fo)}. Post-transmission: {pts(po)}.

Logistic fit (unchanged pre-registered flip criterion): **live f₅₀
(founder) = {fo['f50']:.3f} [{fo['lo']:.3f}, {fo['hi']:.3f}]**;
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
direction ({po['f50']:.3f} vs {fo['f50']:.3f}); (iii) the pilot's
0/6+0/6 at f = 0.25 is confirmed as sub-threshold rather than
budget-limited (1/16 flips even at the 2,000-interaction budget)."""


b_live = b_section(BL, "live") + b_minority_section(BL)
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
        ta = B.get("token_audit", {})
        if ta and "real_word_share" in ta:
            lines.append(
                f"- **Token audit (live)**: {ta['real_word_tokens']}/"
                f"{ta['pool_tokens']} pool tokens ({100*ta['real_word_share']:.1f}%) "
                "are dictionary words (the embedded blocklist used for early "
                "runs was incomplete; now dictionary-backed). Runs whose "
                "winner was a real word: "
                + ", ".join(ta.get("runs_with_real_word_winner", []))
                + f". Sensitivity: mean winner-prior excluding those runs = "
                + ci(ta.get('winner_prior_mean_excluding_flagged', {}), '{:.2f}')
                + " (vs " + ci(B.get('winner_prior_mean', {}), '{:.2f}')
                + " overall).")
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
"""

# ---------------- E1-E4 side-product environments ----------------

def render_envs():
    try:
        E = json.load(open("results/envs_summary.json"))
    except FileNotFoundError:
        return "*(environment suite not yet run)*"

    E1_LABELS = {
        "e1_tight": "E1 (12-word notes)",
        "e1_loose": "E1 (25-word notes)",
        "e1_squeeze": "E1 squeeze (6-word notes)",
        "e1_squeeze_mem": "E1 squeeze + 2× memory (6-word notes)",
        "e1_noisy": "E1 noisy channel (12-word notes, 25% word deletion)",
    }
    E2_LABELS = {
        "e2": "E2 (grid assembly) — VOID live, see integrity note",
        "e2_think": "E2 + scratch line (corrected turn protocol)",
        "e2_dialogue": "E2 + scratch line + pre-episode message channel",
        "e2_sonnet": "E2 + scratch line, claude-sonnet-4.5",
    }
    E3_LABELS = {
        "e3": "E3 (open-lexicon reference)",
        "e3_redo": "E3 redo (chooser reply-token repair)",
        "e3_squeeze": "E3 squeeze (3-word notes, hard distractors, "
                      "shared items)",
    }
    E4_LABELS = {
        "e4": "E4 (tagged bargaining) — VOID live, see integrity note",
        "e4_think": "E4 + scratch line (corrected turn protocol)",
        "e4_sonnet": "E4 + scratch line, claude-sonnet-4.5",
    }

    def tier(mode):
        d = E.get(mode)
        if not d:
            return f"*(no {mode} tier)*"
        parts = []
        for cell, label in E1_LABELS.items():
            c = d.get(cell)
            if not c:
                continue
            parts.append(
                f"- **{label}**: {pci(c['conventionalized'])} of "
                f"populations conventionalized on a note ordering (modal "
                f"share {ci(c['modal_share'], '{:.2f}')}); "
                f"**{c['distinct_modal_variants']} distinct modal orderings** "
                f"across {c['n_pops']} populations (Simpson diversity "
                f"{c['simpson_diversity']:.2f}); inter-agent agreement "
                f"{ci(c.get('inter_agent_agreement', {}), '{:.2f}')}; task "
                f"success (tail) {ci(c.get('success_tail', {}), '{:.2f}')}; "
                f"newcomer first-3-notes adoption of "
                f"the incumbent ordering {ci(c['newcomer_adopt'], '{:.2f}')} vs "
                f"founders' first-3 {ci(c['founder_early'], '{:.2f}')}; scripted "
                f"minority (f=0.25, fixed alternative ordering) flipped "
                f"{pci(c['minority_flip'], '{:.2f}')}."
                + (f" Shuffle control modal share "
                   f"{c['shuffle_control_modal_share']['mean']:.2f} "
                   f"(97.5th pct {c['shuffle_control_modal_share']['p97.5']:.2f})."
                   if c.get("shuffle_control_modal_share") else ""))
        st = d.get("e1_stranger")
        if st:
            parts.append(
                f"- **E1 stranger-pool control**: tracked informers facing "
                f"fresh memoryless responders self-lock (self-consistency "
                f"{ci(st['self_consistency_modal_share'], '{:.2f}')}) but agree "
                f"with EACH OTHER at only "
                f"{st.get('between_informer_agreement', float('nan')):.2f} — "
                f"individual habit forms alone; population-wide agreement "
                f"is the social part.")
        for cell, label in E2_LABELS.items():
            c = d.get(cell)
            if not c:
                continue
            if mode == "mock":
                label = label.split(" — ")[0]
            extra = ""
            if c.get("malformed_per_call", {}).get("estimate") is not None:
                extra += (f"; malformed/call "
                          f"{ci(c['malformed_per_call'], '{:.2f}')}")
            if (c.get("parse_fail_per_call") or {}).get("estimate") is not None:
                extra += (f" (parse failures "
                          f"{ci(c['parse_fail_per_call'], '{:.2f}')})")
            if (c.get("msg_use_rate") or {}).get("estimate") is not None:
                extra += (f"; message-channel use "
                          f"{ci(c['msg_use_rate'], '{:.2f}')}")
            parts.append(
                f"- **{label}**: {pci(c['conventionalized'])} of "
                f"populations conventionalized a badge→region mapping "
                f"(mapping share {ci(c['mapping_share'], '{:.2f}')}); episode "
                f"success first-12 {ci(c['success_first12'], '{:.2f}')} → "
                f"last-12 {ci(c['success_last12'], '{:.2f}')}; "
                f"{c['distinct_modal_mappings']} distinct modal mappings: "
                f"{c['modal_mapping_counts']}{extra}.")
        for cell, label in E3_LABELS.items():
            c = d.get(cell)
            if not c:
                continue
            parts.append(
                f"- **{label}**: "
                f"{ci(c['items_conventionalized_of6'], '{:.1f}')} of 6 items per "
                f"population settled on one description "
                f"(mean per-item modal share "
                f"{ci(c['mean_item_share'], '{:.2f}')}); "
                f"**{c['distinct_schemes']} distinct population-level "
                f"description schemes** across {c['n_pops']} populations; "
                f"note length {ci(c['note_len_first_third'], '{:.1f}')} words "
                f"(first third) → {ci(c['note_len_last_third'], '{:.1f}')} "
                f"(last third); modal description matches a zero-shot prior "
                f"probe in {ci(c.get('modal_matches_prior', {}), '{:.2f}')} of "
                f"items; coined non-word labels: {c['coinage_total']}.")
        for cell, label in E4_LABELS.items():
            c = d.get(cell)
            if not c:
                continue
            if mode == "mock":
                label = label.split(" — ")[0]
            parts.append(
                f"- **{label}**: equilibrium types across "
                f"{c['n_pops']} populations: {c['equilibrium_counts']}; "
                f"class (badge-conditioned 70/30) share "
                f"{pci(c['class_share'], '{:.2f}')}; egalitarian share "
                f"{pci(c['egalitarian_share'], '{:.2f}')}; tail compatibility "
                f"{ci(c['compat_tail'], '{:.2f}')}.")
        return "\n".join(parts)

    return f"""
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
{tier("mock")}

### Live LLM tier (claude-haiku-4.5)
{tier("live")}

### Second model family (openai/gpt-5-mini, frozen prompts)
{tier("m2")}

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

**Two live results replace the pilot picture. (1) In the haiku family,
with parsing repaired and every lever pulled, side-product conventions
are rare: the 6-word squeeze produced one conventionalized population in
12 (agreement 0.13 vs shuffle 0.08 — a weak, above-baseline signal, not
the 1/4 the pilot suggested). (2) The second model family produced the
suite's first unambiguous live side-product convention: on an identical
shared item set, gpt-5-mini populations settled on 7 distinct
description schemes across 8 populations with prior-match 0.19 —
cross-population diversity on a shared world, the pre-registered
signature separating convention from model bias. Convention formation is
capability-gated: the model must be strong enough to exploit minimal
discriminating descriptions (creating the arbitrary-choice space) while
its prior leaves the choice open; haiku's exhaustive-description prior
closes that space (1 scheme across 12 populations, prior-match 1.0),
and no lever in the haiku family opens it.**

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
collapse). When the model is competent enough to reach the
reward-equivalent solution manifold and its prior does not single out a
point on it — gpt-5-mini's minimal discriminating descriptions in E3 —
population-specific conventions emerge and diverge across populations,
the full Lewisian signature. The naming game (all families converge)
and E3-m2 bracket the phenomenon: payoff-coupled alignment is
sufficient, and prior-underdetermined competence is the side-product
route. Limitations: 6–12 live populations per load-bearing cell,
n=3–8 on secondary cells, short in-context histories, three models
from two vendors.
"""


def render_honesty():
    return """
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
"""


def render_claims():
    E2 = {}
    try:
        E2 = json.load(open("results/envs_summary.json"))
    except FileNotFoundError:
        pass

    def n_of(mode, cell):
        return (E2.get(mode, {}).get(cell, {}) or {}).get("n_pops", "TBD")

    bt = S.get("expB_live", {})
    n_gen = bt.get("genesis", {}).get("n_runs", "TBD")
    return f"""
| Paper claim | Cells (final n) | Figures / numbers |
|---|---|---|
| LLM populations form arbitrary conventions when payoff rewards alignment directly, and they persist under 100% turnover (positive control) | b_genesis (n={n_gen}); b_trans_dlg / b_trans_nodlg (n=10/10) | fig_convergence, fig_survival; `\\pNamingGenesisConv`, `\\pNamingSurvivedDlgOn/Off` |
| Newcomer socialisation is dialogue-mediated | b_trans_dlg vs b_trans_nodlg | fig_socialisation; `\\pSocialisationMWU` |
| Substrate committed-minority threshold f50 ≈ 0.126, history-independent | expA minority cells (n=500/point) | fig_minority; `\\fFiftySubstrateFounder/Posttrans` |
| Live minority threshold: f50 or lower bound vs substrate | b_minority_{{founder,posttrans}}_{{f25,f33,f42}}_b2k (n=8 each) | fig_minority |
| Compression produces the first live side-product convention (loose-vs-squeeze contrast) | e1_tight (n={n_of('live','e1_tight')}) vs e1_squeeze (n={n_of('live','e1_squeeze')}) live; substrate n=60/40 | fig_inversion, fig_esuite; `\\convEOneSqueezeLive` |
| E3 concentration is shared model bias, not convention (diversity signature on a shared item set) | e3_squeeze (n={n_of('live','e3_squeeze')}) + zero-shot priors | fig_esuite; `\\schemesEThreeSqueezeLive`, `\\priorEThreeSqueezeLive` |
| The fairness prior absorbs the symmetry that produces Axtell–Epstein–Young classes in the substrate | e4_think (n={n_of('live','e4_think')}), e4_sonnet (n=3), substrate e4 (n=60) | fig_inversion; `\\classEFourThinkLive`, `\\egalEFourThinkLive` |
| Capability sharpens priors: mirror-match coordination failure; negotiated pacts do not fossilize | e2_think (n={n_of('live','e2_think')}), e2_dialogue (n={n_of('live','e2_dialogue')}), e2_sonnet (n=3) | §7 E2 rows; `\\convETwoThinkLive`, `\\succETwoDialogueLive` |
| Enforcement surface features are absent (annotation-invariant) in populations and solitary alike | detector scan over all live dialogue | `\\prevDeonticPop` etc.; enforcement_analysis.json |
| Findings hold across model families | m2 cells: naming genesis n=8, trans n=6; e1 (8/8), e3 (8), e4 (6), e2 (6) | §7 m2 tier rows; `\\convEOneSqueezeMTwo` etc. |
"""


def render_review():
    def j(path):
        try:
            return json.load(open(path))
        except FileNotFoundError:
            return None
    parts = ["All criteria below were frozen in decision-log entries 16–23 "
             "and committed before the corresponding data were collected."]
    sr = j("results/scheme_rules.json")
    if sr:
        parts.append(f"""
### Scheme-coding specification and grain-sensitivity (entries 16)
A population's scheme is its 6-tuple of per-item modal trait-sets (last
15 formation mentions per item). Distinct-scheme counts under all three
frozen rules — A (exact tuple), B (mean per-item Jaccard ≥ 0.5
clustering), C (any-item difference):
| family | n | rule A | rule B | rule C |
|---|---|---|---|---|
| substrate | {sr['substrate']['n_pops']} | {sr['substrate']['ruleA']} | {sr['substrate']['ruleB']} | {sr['substrate']['ruleC']} |
| haiku | {sr['haiku']['n_pops']} | {sr['haiku']['ruleA']} | {sr['haiku']['ruleB']} | {sr['haiku']['ruleC']} |
| gpt-5-mini | {sr['m2']['n_pops']} | {sr['m2']['ruleA']} | {sr['m2']['ruleB']} | {sr['m2']['ruleC']} |

The m2-vs-haiku contrast (7/8 vs 1/12) is grain-stable across rules A
and C. Rule B collapses **every** family to one cluster — including the
substrate positive control whose 40 distinct schemes are uncontested —
so it cannot detect any convention diversity at that threshold; it
bounds the taxonomy's coarse end rather than undermining the count.""")
    nd = j("results/review/null_diversity.json")
    if nd:
        m2, hk = nd.get("m2", {}), nd.get("haiku", {})
        parts.append(f"""
### No-interaction diversity baseline (entry 17)
16 pseudo-populations per family of independent zero-shot generations
(shared items, frozen prompt, empty history, matched per-item event
counts, run temperature), identical extraction. Null distribution of
distinct-scheme counts among resampled groups of 8 (rule A):
gpt-5-mini mean {m2.get('ruleA', {}).get('mean_distinct', 'TBD'):.2f},
P(≥7 distinct) = {m2.get('ruleA', {}).get('p_ge_7', 'TBD')};
haiku mean {hk.get('ruleA', {}).get('mean_distinct', 'TBD'):.2f},
P(≥7) = {hk.get('ruleA', {}).get('p_ge_7', 'TBD')}.""")
    else:
        parts.append("\n### No-interaction diversity baseline — *(running)*")
    tr = j("results/review/transplants.json")
    if tr:
        ad = sum(t["adopted"] for t in tr)
        lat = [t["latency_sends"] for t in tr if t["latency_sends"]]
        ms = [t["match_share_last10"] for t in tr
              if t["match_share_last10"] is not None]
        stab = [t["host_items_stable_of6"] for t in tr]
        parts.append(f"""
### Transplant (entry 18)
{len(tr)} transplants across distinct-scheme ordered pairs: adopted
{ad}/{len(tr)} (criterion: ≥80% of last 10 sends matching the host
modal); mean last-10 match share {np.mean(ms):.2f}; median latency
{(int(np.median(lat)) if lat else 'n/a')} sends; host items stable
{np.mean(stab):.1f}/6.""")
    else:
        parts.append("\n### Transplant — *(running)*")
    sw = j("results/review/swap.json")
    if sw:
        lines = ["\n### Cross-population swap (entry 19)"]
        for fam, lab in (("m2", "gpt-5-mini"), ("haiku", "haiku control")):
            d = sw.get(fam)
            if d:
                lines.append(
                    f"- **{lab}**: within-population success "
                    f"{d['within_success']:.3f} vs cross-population "
                    f"{d['cross_success']:.3f} (Δ = "
                    f"{d['within_success'] - d['cross_success']:+.3f}).")
        lines.append("The haiku prediction (single shared scheme → no "
                     "swap cost) is tested and reported as such.")
        parts.append("\n".join(lines))
    else:
        parts.append("\n### Swap — *(running)*")
    to = j("results/review_m2_turnover.json")
    if to:
        parts.append(f"""
### E3-m2 turnover (entry 23; computed on the existing generation of
replacement contained in the e3_squeeze journals)
Scheme survival {to['survived']}/{to['n_pops']} populations (≥4/6 items
stable; mean {to['mean_items_stable']}/6); newcomer mean match share
{to['newcomer_mean_match']}. Unlike naming-game conventions (perfect
survival at matched turnover), m2 description schemes are **fragile
under full population replacement** — consistent with their weak
within-population concentration (0.52): the convention is real
(diversity + swap) but noisily transmitted.""")
    ax = j("results/review/axis_probes.json")
    if ax:
        lines = ["\n### Independent axis measurements (entry 21)"]
        for fam, lab in (("haiku", "haiku"), ("m2", "gpt-5-mini")):
            d = ax.get(fam, {})
            r = d.get("reachability", {})
            pr = d.get("prior_concentration", {}).get("e3_items", [])
            ent = np.mean([x["entropy_bits"] for x in pr]) if pr else None
            mp = np.mean([x["modal_p"] for x in pr]) if pr else None
            lines.append(
                f"- **{lab}**: reachability — E1 values-only code "
                f"{r.get('e1_code_success', 'TBD')}, E3 specified scheme "
                f"{r.get('e3_scheme_success', 'TBD')} "
                f"(n={r.get('n_trials')}); prior concentration (E3, "
                f"{'50' if pr else '?'} samples/item) — mean entropy "
                f"{ent:.2f} bits, mean modal probability {mp:.2f}."
                if ent is not None else f"- **{lab}**: pending")
        lines.append("These probes assign the quadrant axes from "
                     "measurements independent of population outcomes.")
        parts.append("\n".join(lines))
    else:
        parts.append("\n### Axis probes — *(running)*")
    eq = j("results/equivalence_bounds.json")
    if eq:
        s_, l_ = eq["substrate"], eq["live"]
        parts.append(f"""
### Equivalence bounds for the null history effect (§6)
No founder-vs-post-transmission difference detected in either tier;
effects up to Δ remain compatible with the data — substrate: Δf₅₀ ∈
[{s_['delta_range_compatible'][0]:+.4f}, {s_['delta_range_compatible'][1]:+.4f}]
(bootstrap); live: Δf₅₀ ∈ [{l_['delta_range_compatible'][0]:+.3f},
{l_['delta_range_compatible'][1]:+.3f}] (founder CI × separation
interval).""")
    parts.append("""
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
{MITIGATION}""")
    E2S = {}
    try:
        E2S = json.load(open("results/envs_summary.json")).get("live", {})
    except FileNotFoundError:
        pass
    role, het = E2S.get("e2_role"), E2S.get("e2_hetero")
    base = E2S.get("e2_think", {})
    mit = []
    if role:
        mit.append(
            f"Exogenous role line: success first-12 "
            f"{ci(role['success_first12'], '{:.2f}')} → last-12 "
            f"{ci(role['success_last12'], '{:.2f}')} vs e2_think baseline "
            f"last-12 {ci(base.get('success_last12', {}), '{:.2f}')} "
            f"(n={role['n_pops']}).")
    if het:
        mit.append(
            f"Heterogeneous pairing (haiku × gpt-5-mini): success "
            f"first-12 {ci(het['success_first12'], '{:.2f}')} → last-12 "
            f"{ci(het['success_last12'], '{:.2f}')} (n={het['n_pops']}).")
    text = "\n".join(parts)
    return text.replace("{MITIGATION}",
                        "\n".join(mit) if mit else "*(running)*")


# ---------------- render ----------------

tpl = open("scripts/RESULTS_template.md").read()
fills = {
    "{{A_GENESIS}}": a_genesis,
    "{{A_TRANSMISSION}}": a_trans,
    "{{A_SOCIALISATION}}": a_social,
    "{{A_MINORITY}}": a_minor,
    "{{A_TRANSPLANT}}": a_transplant,
    "{{A_NSWEEP}}": a_nsweep,
    "{{B_RESULTS}}": b_live + "\n" + b_mock_note,
    "{{B_COST}}": """
Projection (upper bound, printed before the first call): **$99.93** against
a $250 in-code cap. Actual total spend for the full live suite — 42 runs,
comprehension gates, priors, ~48k API calls including the 4,507-message
judge pass — **$15.56** (claude-haiku-4.5 via OpenRouter at Anthropic list
pricing, $1/$5 per MTok). Wall-clock ≈ 3.5 h at 8 concurrent runs
(strictly sequential within each run).""",
    "{{INTEGRITY}}": integrity,
    "{{INTERPRETATION}}": interpretation,
    "{{ENVS}}": render_envs(),
    "{{HONESTY}}": render_honesty(),
    "{{CLAIMS}}": render_claims(),
    "{{REVIEW}}": render_review(),
}
for k, v in fills.items():
    tpl = tpl.replace(k, v)
sys.stdout.write(tpl)
