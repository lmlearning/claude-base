"""Review-response harnesses (decision-log entries 16-21; criteria were
frozen and committed before any data collection).

Contents:
  * scheme coding rules A/B/C (entry 16) shared by all counts;
  * E3 no-interaction pseudo-population generator (entry 17);
  * E3 transplant (entry 18) and cross-population swap (entry 19);
  * independent axis probes: prior concentration + reachability
    (entry 21; the reachability prompts are intentionally exempt from
    the banned-vocabulary lint and never enter population prompts).

All outputs land under results/review/ as JSON/JSONL.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np

from ..analysis.stats import bootstrap_ci
from . import e1_relay, e3_ref
from .common import Journal, rng_for, word_count_clip

MODELS = {"haiku": "anthropic/claude-haiku-4.5", "m2": "openai/gpt-5-mini"}


# ---------------------------------------------------------------------------
# entry 16: scheme coding rules
# ---------------------------------------------------------------------------

def scheme_of_pop(recs, phase=("formation",)) -> list[tuple | None]:
    """Per-item modal trait-set over the last 15 formation mentions."""
    per_item: dict[int, list] = {}
    for r in recs:
        if r.get("type") == "interaction" and r.get("phase") in phase:
            per_item.setdefault(r["target"], []).append(
                tuple(r["mentioned"]))
    out = []
    for it in range(e3_ref.N_ITEMS):
        tail = per_item.get(it, [])[-15:]
        if not tail:
            out.append(None)
            continue
        m, _ = Counter(tail).most_common(1)[0]
        out.append(m)
    return out


def _jaccard(a, b):
    sa, sb = set(a or ()), set(b or ())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def count_distinct(schemes: list[list], rule: str) -> int:
    """Rule A: exact tuple equality. Rule B: agglomerative clustering,
    mean per-item Jaccard >= 0.5 merges. Rule C: distinct iff any
    non-None item modal differs."""
    if rule == "A":
        return len({str(s) for s in schemes})
    if rule == "C":
        def key(s):
            return str([list(x) if x else None for x in s])
        return len({key(s) for s in schemes})
    assert rule == "B"
    clusters: list[list] = []
    for s in schemes:
        placed = False
        for cl in clusters:
            rep = cl[0]
            sim = np.mean([_jaccard(a, b) for a, b in zip(s, rep)])
            if sim >= 0.5:
                cl.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return len(clusters)


def schemes_from_dir(outdir: str, prefix: str) -> list[list]:
    out = []
    for d in sorted(os.listdir(outdir)):
        if not d.startswith(prefix):
            continue
        jp = os.path.join(outdir, d, "journal.jsonl")
        if os.path.exists(jp):
            recs = [json.loads(l) for l in open(jp)]
            out.append(scheme_of_pop(recs))
    return out


# ---------------------------------------------------------------------------
# entry 17: no-interaction pseudo-populations
# ---------------------------------------------------------------------------

def _shared_cfg(framing="picker"):
    traits, items = e3_ref.make_items(rng_for(777001, 1))
    return {"framing": framing, "note_words": 3, "traits": traits,
            "items": items}


def event_counts(real_dir: str, prefix: str) -> dict[int, int]:
    """Mean per-item formation-send counts across the real populations."""
    per_item = Counter()
    n_pops = 0
    for d in sorted(os.listdir(real_dir)):
        if not d.startswith(prefix):
            continue
        jp = os.path.join(real_dir, d, "journal.jsonl")
        if not os.path.exists(jp):
            continue
        n_pops += 1
        for l in open(jp):
            r = json.loads(l)
            if r.get("type") == "interaction" and r["phase"] == "formation":
                per_item[r["target"]] += 1
    return {it: max(1, round(per_item[it] / max(1, n_pops)))
            for it in range(e3_ref.N_ITEMS)}


def run_pseudo_pops(family: str, backend, out_root: str,
                    counts: dict[int, int], n_pops: int = 16) -> None:
    cfg = _shared_cfg()
    nw = cfg["note_words"]
    for pi in range(n_pops):
        run_dir = os.path.join(out_root, f"pseudo_{family}_p{pi:02d}")
        journal = Journal(run_dir)
        done = sum(1 for _ in journal.replay())
        i = 0
        for it in range(e3_ref.N_ITEMS):
            for k in range(counts[it]):
                i += 1
                if i <= done:
                    continue
                raw = backend.complete(
                    e3_ref._system(cfg),
                    "\n\n".join([
                        "You have no previous turns yet.",
                        e3_ref.SEND_INSTR.format(
                            traits=", ".join(cfg["traits"][x]
                                             for x in cfg["items"][it]),
                            note_words=nw)]),
                    max(24, nw * 3),
                    {"kind": "send", "traits": cfg["traits"],
                     "item": cfg["items"][it], "note_words": nw,
                     "seen_sets": []})
                note, _ = word_count_clip(raw, nw)
                journal.append({"type": "interaction", "phase": "formation",
                                "target": it, "note": note,
                                "mentioned": e3_ref.extract_traitset(
                                    note, cfg["traits"])})
        print(f"pseudo {family} p{pi} done", flush=True)


def null_distribution(out_root: str, family: str, n_resample: int = 2000,
                      group: int = 8, seed: int = 0) -> dict:
    schemes = schemes_from_dir(out_root, f"pseudo_{family}_p")
    rng = np.random.default_rng(seed)
    out = {"n_pseudo_pops": len(schemes)}
    for rule in ("A", "B", "C"):
        counts = []
        for _ in range(n_resample):
            idx = rng.choice(len(schemes), size=group, replace=False)
            counts.append(count_distinct([schemes[i] for i in idx], rule))
        counts = np.array(counts)
        out[f"rule{rule}"] = {
            "mean_distinct": float(counts.mean()),
            "p_ge_7": float(np.mean(counts >= 7)),
            "distribution": {str(v): int((counts == v).sum())
                             for v in sorted(set(counts.tolist()))}}
    return out


# ---------------------------------------------------------------------------
# population loader (shared by transplant and swap)
# ---------------------------------------------------------------------------

def load_e3_population(run_dir: str):
    recs = [json.loads(l) for l in open(os.path.join(run_dir,
                                                     "journal.jsonl"))]
    cfg = next(r for r in recs if r["type"] == "config")
    n = cfg["n_agents"]
    agents = [e3_ref.E3Agent(i) for i in range(n)]
    for r in recs:
        if r["type"] == "interaction":
            e3_ref._apply(agents, r)
        elif r["type"] == "replacement":
            a = e3_ref.E3Agent(r["agent_id"])
            agents[r["slot"]] = a
    return cfg, agents, recs


# ---------------------------------------------------------------------------
# entry 18: transplant
# ---------------------------------------------------------------------------

def run_transplant(src_dir: str, host_dir: str, backend, out_dir: str,
                   n_inter: int = 240) -> dict:
    scfg, sagents, srecs = load_e3_population(src_dir)
    hcfg, hagents, hrecs = load_e3_population(host_dir)
    host_scheme = scheme_of_pop(hrecs)
    src_scheme = scheme_of_pop(srecs)
    seed = hcfg["seed"] + 991
    rng = np.random.default_rng([seed, 3])
    slot = int(rng.integers(hcfg["n_agents"]))
    mover = sagents[int(rng.integers(scfg["n_agents"]))]
    mover_id = 7000 + mover.agent_id
    mover.agent_id = mover_id
    hagents[slot] = mover
    journal = Journal(out_dir)
    journal.append({"type": "transplant_config", "src": src_dir,
                    "host": host_dir, "slot": slot, "mover_id": mover_id,
                    "host_scheme": [list(s) if s else None
                                    for s in host_scheme],
                    "src_scheme": [list(s) if s else None
                                   for s in src_scheme]})
    cfg = {**hcfg, "seed": seed}
    n = cfg["n_agents"]
    t = 0
    for t in range(1, n_inter + 1):
        r = rng_for(seed, 3, t)
        i = int(r.integers(n))
        j = int(r.integers(n - 1))
        if j >= i:
            j += 1
        e3_ref._one_interaction(t, hagents, cfg, backend, journal,
                                "transplant", i, j)
    # adoption per entry 18
    recs = [json.loads(l) for l in open(journal.path)]
    sends = [r for r in recs if r.get("type") == "interaction"
             and r["sender_id"] == mover_id]
    last10 = sends[-10:]
    match = [tuple(r["mentioned"]) == (tuple(host_scheme[r["target"]])
                                       if host_scheme[r["target"]] else None)
             for r in last10]
    adopted = bool(len(last10) >= 10 and np.mean(match) >= 0.8)
    latency = None
    for k in range(10, len(sends) + 1):
        w = sends[k - 10:k]
        ok = [tuple(r["mentioned"]) == (tuple(host_scheme[r["target"]])
                                        if host_scheme[r["target"]]
                                        else None) for r in w]
        if np.mean(ok) >= 0.8:
            latency = k
            break
    end_scheme = scheme_of_pop(recs, phase=("transplant",))
    host_stable = sum(1 for a, b in zip(end_scheme, host_scheme)
                      if a == b)
    res = {"adopted": adopted, "latency_sends": latency,
           "n_mover_sends": len(sends),
           "host_items_stable_of6": host_stable,
           "match_share_last10": (float(np.mean(match)) if last10 else None)}
    journal.append({"type": "transplant_result", **res})
    return res


# ---------------------------------------------------------------------------
# entry 19: swap
# ---------------------------------------------------------------------------

def run_swap_pair(cfgA, agentsA, cfgB, agentsB, backend, journal,
                  n_eps: int, seed: int, tag: str) -> list[bool]:
    """Describer from A, responder from B; memories frozen (agents are
    fresh replay copies whose state is never written back)."""
    out = []
    items, traits = cfgA["items"], cfgA["traits"]
    for e in range(n_eps):
        r = rng_for(seed, 5, e)
        target = int(r.integers(e3_ref.N_ITEMS))
        others = [i for i in range(e3_ref.N_ITEMS) if i != target]
        ranked = sorted(others, key=lambda o: (-len(set(items[target])
                                                    & set(items[o])),
                                               r.random()))
        d1, d2 = ranked[0], ranked[1]
        order = [int(x) for x in r.permutation(3)]
        shown = [[target, d1, d2][k] for k in order]
        S = agentsA[int(r.integers(len(agentsA)))]
        R = agentsB[int(r.integers(len(agentsB)))]
        nw = cfgA.get("note_words", 3)
        raw = backend.complete(
            e3_ref._system(cfgA),
            "\n\n".join([e3_ref._hist(S), e3_ref.SEND_INSTR.format(
                traits=", ".join(traits[i] for i in items[target]),
                note_words=nw)]),
            max(24, nw * 3),
            {"kind": "send", "traits": traits, "item": items[target],
             "note_words": nw,
             "seen_sets": list(S.item_sets.get(target, []))})
        note, _ = word_count_clip(raw, nw)
        mentioned = e3_ref.extract_traitset(note, traits)
        cand = "\n".join(f"{k+1}. features: "
                         + ", ".join(traits[i] for i in items[it])
                         for k, it in enumerate(shown))
        reply = backend.complete(
            e3_ref._system(cfgB), "\n\n".join([
                e3_ref._hist(R), e3_ref.RECV_INSTR.format(candidates=cand,
                                                          note=note)]),
            cfgB.get("recv_tokens", 24),
            {"kind": "recv", "mentioned": mentioned,
             "candidates": [items[it] for it in shown]})
        m = re.search(r"[123]", reply)
        pick = int(m.group()) if m else 1
        correct = shown[pick - 1] == target
        out.append(bool(correct))
        journal.append({"type": "swap_episode", "pair": tag, "ep": e,
                        "target": target, "note": note,
                        "correct": bool(correct)})
    return out


# ---------------------------------------------------------------------------
# entry 21: independent axis probes (LINT-EXEMPT BY DESIGN; these
# prompts never enter population runs)
# ---------------------------------------------------------------------------

E1_CODE_PROBE = (
    "You are shown a record of labels with entries:\n{record}\n\n"
    "Write a note of at most 6 words for a colleague using this exact "
    "code: write ONLY the five entries, in the order of the labels "
    "{label_order}, separated by spaces, with no label names. Reply "
    "with only the note.")
E3_SCHEME_PROBE = (
    "An object has these features: {traits}. Write a note of at most 3 "
    "words that mentions exactly these features and nothing else: "
    "{subset}. Reply with only the note.")


def probe_reachability(family: str, backend, journal, n_trials: int = 30
                       ) -> dict:
    rng = np.random.default_rng([20260722, hash(family) % (2**31)])
    # E1: values-only positional code at 6 words
    pool = [w for w in e1_relay.generate_pool(45, np.random.default_rng(7))]
    fields, vocab = pool[:5], pool[5:]
    ok1 = 0
    for s in range(n_trials):
        r = np.random.default_rng([991, s])
        values = [vocab[i] for i in r.choice(len(vocab), size=5,
                                             replace=False)]
        presented = [int(x) for x in r.permutation(5)]
        rec = "\n".join(f"{fields[i]}: {values[i]}" for i in presented)
        raw = backend.complete(
            "", E1_CODE_PROBE.format(record=rec,
                                     label_order=", ".join(fields)),
            40, {"kind": "prior", "fields": fields, "values": values,
                 "presented": presented})
        note, _ = word_count_clip(raw, 6)
        toks = [t.lower() for t in note.replace(",", " ").split()]
        ok = toks == [v.lower() for v in values]
        ok1 += ok
        journal.append({"type": "probe_e1", "family": family, "trial": s,
                        "note": note, "ok": bool(ok)})
    # E3: execute a specified minimal discriminating scheme at 3 words
    cfg = _shared_cfg()
    ok2 = 0
    for s in range(n_trials):
        r = np.random.default_rng([992, s])
        it = int(r.integers(e3_ref.N_ITEMS))
        item = cfg["items"][it]
        sub = sorted(int(x) for x in r.choice(item, size=2, replace=False))
        raw = backend.complete(
            "", E3_SCHEME_PROBE.format(
                traits=", ".join(cfg["traits"][i] for i in item),
                subset=", ".join(cfg["traits"][i] for i in sub)),
            24, {"kind": "prior", "traits": cfg["traits"], "item": item,
                 "note_words": 3})
        note, _ = word_count_clip(raw, 3)
        got = e3_ref.extract_traitset(note, cfg["traits"])
        ok = got == sub
        ok2 += ok
        journal.append({"type": "probe_e3", "family": family, "trial": s,
                        "note": note, "ok": bool(ok)})
    return {"e1_code_success": ok1 / n_trials,
            "e3_scheme_success": ok2 / n_trials, "n_trials": n_trials}


def probe_priors(family: str, backend, journal, n_samples: int = 50) -> dict:
    cfg = _shared_cfg()
    out = {"e3_items": []}
    for it in range(e3_ref.N_ITEMS):
        sets = []
        for s in range(n_samples):
            raw = backend.complete(
                e3_ref._system(cfg),
                e3_ref.PRIOR_PROBE.format(
                    traits=", ".join(cfg["traits"][i]
                                     for i in cfg["items"][it]),
                    note_words=3),
                24, {"kind": "prior", "traits": cfg["traits"],
                     "item": cfg["items"][it], "note_words": 3})
            note, _ = word_count_clip(raw, 3)
            sets.append(tuple(e3_ref.extract_traitset(note, cfg["traits"])))
        c = Counter(sets)
        n = sum(c.values())
        probs = np.array([v / n for v in c.values()])
        ent = float(-(probs * np.log2(probs)).sum())
        modal, k = c.most_common(1)[0]
        boots = []
        rng = np.random.default_rng(it)
        arr = np.array([str(s) for s in sets])
        for _ in range(1000):
            bs = arr[rng.integers(n, size=n)]
            boots.append(Counter(bs).most_common(1)[0][1] / n)
        out["e3_items"].append({
            "item": it, "entropy_bits": round(ent, 3),
            "modal_set": list(modal), "modal_p": round(k / n, 3),
            "modal_p_ci": [round(float(np.quantile(boots, q)), 3)
                           for q in (0.025, 0.975)]})
        journal.append({"type": "probe_prior_e3", "family": family,
                        "item": it, "sets": [list(s) for s in sets]})
    return out
