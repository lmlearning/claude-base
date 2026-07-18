#!/usr/bin/env python3
"""Driver for the review-response harnesses (RESULTS decision log 16-22).

Usage:
  python scripts/run_review.py --mode mock|live [--cap 60]
        [--only pseudo transplant swap probes]

All live calls share one CostTracker (results/review/cost_state.json).
Every stage journals before proceeding and is idempotent/resumable.
Outputs: results/review/*.json(l) + per-stage run dirs.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from namegame.envs import e1_relay, e3_ref, review
from namegame.envs.common import EnvMockBackend, Journal
from namegame.expb.backend import CostTracker, OpenRouterBackend

OUT = "results/review"


def backends(mode, cap):
    if mode == "mock":
        def route(context, rng):
            fn = (e1_relay.mock_reply if "fields" in context
                  else e3_ref.mock_reply)
            return fn(context, rng)
        mk = lambda fam: EnvMockBackend(route, seed=hash(fam) % 999)
        return {"haiku": mk("haiku"), "m2": mk("m2")}, None
    cost = CostTracker(cap, os.path.join(OUT, "cost_state.json"))
    return {f: OpenRouterBackend(m, cost)
            for f, m in review.MODELS.items()}, cost


def stage_pseudo(be, mode):
    real = {"m2": ("results/envs_live_m2", "e3_squeeze_p"),
            "haiku": ("results/envs_live", "e3_squeeze_p")}
    res = {}
    for fam, (rdir, pref) in real.items():
        counts = (review.event_counts(rdir, pref) if os.path.isdir(rdir)
                  else {i: 60 for i in range(6)})
        if mode == "mock":
            counts = {i: 8 for i in range(6)}
        review.run_pseudo_pops(fam, be[fam], os.path.join(OUT, "pseudo"),
                               counts, n_pops=16)
        res[fam] = review.null_distribution(os.path.join(OUT, "pseudo"), fam)
        res[fam]["event_counts"] = counts
    json.dump(res, open(os.path.join(OUT, "null_diversity.json"), "w"),
              indent=1)
    print("pseudo done:", {f: r["ruleA"]["p_ge_7"] for f, r in res.items()})


def _m2_pops():
    d = "results/envs_live_m2"
    return [os.path.join(d, x) for x in sorted(os.listdir(d))
            if x.startswith("e3_squeeze_p")]


def stage_transplant(be, mode):
    pops = _m2_pops() if mode == "live" else _m2_pops()[:4]
    schemes = [review.scheme_of_pop([json.loads(l) for l in
                                     open(p + "/journal.jsonl")])
               for p in pops]
    pairs = [(i, j) for i, j in itertools.permutations(range(len(pops)), 2)
             if schemes[i] != schemes[j]]
    rng = np.random.default_rng(20260723)
    rng.shuffle(pairs)
    pairs = pairs[:(10 if mode == "live" else 2)]
    out = []
    for pi, (i, j) in enumerate(pairs):
        d = os.path.join(OUT, f"transplant_{pi:02d}")
        done = os.path.join(d, "result.json")
        if os.path.exists(done):
            out.append(json.load(open(done)))
            continue
        n_inter = 240 if mode == "live" else 30
        r = review.run_transplant(pops[i], pops[j], be["m2"], d,
                                  n_inter=n_inter)
        r["pair"] = [os.path.basename(pops[i]), os.path.basename(pops[j])]
        json.dump(r, open(done, "w"), indent=1)
        out.append(r)
        print(f"transplant {pi} {r['adopted']=} {r['latency_sends']=}",
              flush=True)
    json.dump(out, open(os.path.join(OUT, "transplants.json"), "w"),
              indent=1)


def stage_swap(be, mode):
    res = {}
    for fam, (rdir, pref, n_pairs) in {
            "m2": ("results/envs_live_m2", "e3_squeeze_p", None),
            "haiku": ("results/envs_live", "e3_squeeze_p", 24)}.items():
        pops = [os.path.join(rdir, x) for x in sorted(os.listdir(rdir))
                if x.startswith(pref)]
        if mode == "mock":
            pops = pops[:3]
        loaded = [review.load_e3_population(p) for p in pops]
        journal = Journal(os.path.join(OUT, f"swap_{fam}"))
        n_eps = 30 if mode == "live" else 5
        pairs = list(itertools.permutations(range(len(pops)), 2))
        if n_pairs and mode == "live":
            rng = np.random.default_rng(7)
            rng.shuffle(pairs)
            pairs = pairs[:n_pairs]
        rows = {}
        for (i, j) in pairs + [(i, i) for i in range(len(pops))]:
            tag = f"{i}->{j}"
            ok = review.run_swap_pair(loaded[i][0], loaded[i][1],
                                      loaded[j][0], loaded[j][1],
                                      be[fam], journal, n_eps,
                                      seed=100000 + 97 * i + j, tag=tag)
            rows[tag] = [int(sum(ok)), len(ok)]
            print(f"swap {fam} {tag}: {sum(ok)}/{len(ok)}", flush=True)
        within = [rows[f"{i}->{i}"] for i in range(len(pops))]
        cross = [v for k, v in rows.items()
                 if k.split("->")[0] != k.split("->")[1]]
        res[fam] = {
            "pairs": rows,
            "within_success": sum(k for k, _ in within) / max(
                1, sum(n for _, n in within)),
            "cross_success": sum(k for k, _ in cross) / max(
                1, sum(n for _, n in cross))}
    json.dump(res, open(os.path.join(OUT, "swap.json"), "w"), indent=1)


def stage_probes(be, mode):
    res = {}
    journal = Journal(os.path.join(OUT, "probes"))
    n = 30 if mode == "live" else 5
    ns = 50 if mode == "live" else 5
    for fam in ("haiku", "m2"):
        res[fam] = {"reachability": review.probe_reachability(
            fam, be[fam], journal, n_trials=n)}
        res[fam]["prior_concentration"] = review.probe_priors(
            fam, be[fam], journal, n_samples=ns)
        print(fam, res[fam]["reachability"], flush=True)
    json.dump(res, open(os.path.join(OUT, "axis_probes.json"), "w"),
              indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--cap", type=float, default=60.0)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    be, cost = backends(a.mode, a.cap)
    stages = a.only or ["pseudo", "transplant", "swap", "probes"]
    for s in stages:
        {"pseudo": stage_pseudo, "transplant": stage_transplant,
         "swap": stage_swap, "probes": stage_probes}[s](be, a.mode)
    if cost:
        print(f"review spend ${cost.cost_usd:.2f} of ${a.cap:.2f}")
    print("REVIEW STAGES DONE:", stages)


if __name__ == "__main__":
    main()
