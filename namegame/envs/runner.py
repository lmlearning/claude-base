"""Runner for the E1-E4 environment suite.

Mock mode = substrate control (cheap policies behind the same prompts and
parsers, larger population counts).  Live mode = claude-haiku-4.5 via the
configured key, shared CostTracker with a hard cap, populations run
concurrently (strictly sequential within each population).
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..expb.backend import PRICES, OpenRouterBackend, SpendCapExceeded
from ..expb.runner import _live_backend_factory
from . import e1_relay, e2_grid, e3_ref, e4_bargain
from .common import EnvMockBackend

MASTER_SEED_E = 20260719
ENVS = {"e1": e1_relay, "e2": e2_grid, "e3": e3_ref, "e4": e4_bargain}


def define_cells(mode: str) -> dict[str, dict]:
    """Cell -> config template. n_pops differs between tiers: the mock
    (substrate) tier carries the statistics, the live tier the LLM claim."""
    big = mode == "mock"
    cells = {}
    # E1: budget sweep is the functional-pressure lever.
    # Final-submission power upgrade: the 12-word cell (the paper's
    # "loose" arm of the loose-vs-squeeze compression contrast) runs 12
    # live populations; the 25-word cell stays at pilot n=4.
    for tag, budget, n_live in (("tight", 12, 12), ("loose", 25, 4)):
        cells[f"e1_{tag}"] = dict(
            env="e1", n_agents=12, budget_words=budget,
            formation_interactions=360, interactions_per_replacement=6,
            settle_interactions=60, minority_fraction=0.25,
            minority_interactions=200,
            n_pops=(60 if big else n_live),
            framings=["activity", "office"])
    cells["e1_stranger"] = dict(
        env="e1", n_agents=4, budget_words=12, stranger_pool=True,
        formation_interactions=120, interactions_per_replacement=0,
        settle_interactions=0, n_pops=(20 if big else 2),
        framings=["activity"])
    cells["e2"] = dict(
        env="e2", n_agents=8, formation_episodes=48,
        episodes_per_replacement=2, settle_episodes=8,
        n_pops=(40 if big else 6), framings=["site", "workshop"])
    cells["e3"] = dict(
        env="e3", n_agents=12, formation_interactions=360,
        interactions_per_replacement=6, settle_interactions=60,
        n_pops=(60 if big else 6), framings=["picker", "warehouse"])
    cells["e4"] = dict(
        env="e4", n_agents=12, formation_interactions=500,
        interactions_per_replacement=6, settle_interactions=60,
        n_pops=(60 if big else 6), framings=["bonus", "market"])

    # --- convention-inducing variants: every lever is a payoff, capacity
    # or structure change; no prompt ever mentions consistency ---
    # E1: crush the word budget (5 labels + 5 entries cannot all be named
    # in 6 words -> naming everything stops being free; position must
    # carry meaning), optionally with doubled memory; and a lossy channel.
    for tag, extra in (("squeeze", dict(budget_words=6)),
                       ("squeeze_mem", dict(budget_words=6,
                                            memory_events=12)),
                       ("noisy", dict(budget_words=12, channel_noise=0.25))):
        cells[f"e1_{tag}"] = dict(
            env="e1", n_agents=12, formation_interactions=360,
            interactions_per_replacement=6, settle_interactions=60,
            n_pops=(40 if big else (12 if tag == "squeeze" else 4)),
            framings=["activity", "office"], **extra)
    # E3: 3-word budget + hardest distractors + a SHARED item set across
    # populations (makes the cross-population diversity test direct).
    cells["e3_squeeze"] = dict(
        env="e3", n_agents=12, note_words=3, hard_distractors=True,
        shared_items=True, recv_tokens=24, formation_interactions=360,
        interactions_per_replacement=6, settle_interactions=60,
        n_pops=(40 if big else 12), framings=["picker", "warehouse"])
    # E2: a neutral pre-episode message channel (does negotiated division
    # of labour fossilize across partners and survive turnover?), and a
    # one-line scratch turn (does explicit deliberation find the
    # symmetry-breaking cue?).
    # dialogue rides on the scratch-line turn format (live probes showed
    # the plain 'reply with only CELL' instruction is near-universally
    # ignored by the model once it has room to chat, so the think-format
    # e2_think cell is the matched no-dialogue control)
    cells["e2_dialogue"] = dict(
        env="e2", n_agents=8, dialogue=True, think=True,
        formation_episodes=48, episodes_per_replacement=2,
        settle_episodes=8, n_pops=(20 if big else 10),
        framings=["site", "workshop"])
    cells["e2_think"] = dict(
        env="e2", n_agents=8, think=True, formation_episodes=48,
        episodes_per_replacement=2, settle_episodes=8,
        n_pops=(20 if big else 10), framings=["site", "workshop"])
    cells["e4_think"] = dict(
        env="e4", n_agents=12, think=True, formation_interactions=500,
        interactions_per_replacement=6, settle_interactions=60,
        n_pops=(20 if big else 12), framings=["bonus", "market"])
    # review-response mitigation cells (decision-log entry 20)
    cells["e2_role"] = dict(
        env="e2", n_agents=8, think=True, role_line=True,
        formation_episodes=48, episodes_per_replacement=2,
        settle_episodes=8, n_pops=8, framings=["site", "workshop"])
    if not big:
        cells["e2_hetero"] = dict(
            env="e2", n_agents=8, think=True, hetero=True,
            formation_episodes=48, episodes_per_replacement=2,
            settle_episodes=8, n_pops=8, framings=["site", "workshop"])
    if not big:
        # repair cell: the original live e3 chooser was truncated at 8
        # reply tokens (36% of picks fell to the random fallback); rerun
        # with room to answer
        cells["e3_redo"] = dict(
            env="e3", n_agents=12, recv_tokens=24,
            formation_interactions=360, interactions_per_replacement=6,
            settle_interactions=60, n_pops=4,
            framings=["picker", "warehouse"])
    if not big:
        # capability-threshold tier: the think-cell design on a stronger
        # model (live only; a smoke probe showed sonnet reasons at length
        # regardless, so the scratch-line design keeps replies parseable
        # and the haiku-think cells are the matched comparison)
        cells["e2_sonnet"] = dict(
            env="e2", n_agents=8, model="anthropic/claude-sonnet-4.5",
            think=True, formation_episodes=48, episodes_per_replacement=2,
            settle_episodes=8, n_pops=3, framings=["site", "workshop"])
        cells["e4_sonnet"] = dict(
            env="e4", n_agents=12, model="anthropic/claude-sonnet-4.5",
            think=True, formation_interactions=500,
            interactions_per_replacement=6, settle_interactions=60,
            n_pops=3, framings=["bonus", "market"])
    return cells


def pop_config(cell_name: str, cell: dict, idx: int) -> dict:
    seed = int(np.random.SeedSequence(
        [MASTER_SEED_E, zlib.crc32(cell_name.encode()),
         idx]).generate_state(1)[0])
    cfg = {k: v for k, v in cell.items() if k not in ("n_pops", "framings")}
    cfg["cell"] = cell_name
    cfg["pop_index"] = idx
    cfg["seed"] = seed
    cfg["framing"] = cell["framings"][idx % len(cell["framings"])]
    return cfg


# --- cost projection (upper bounds, live pricing $1/$5 per MTok) ----------

CALL_SHAPES = {   # env -> (calls per unit, in_tokens, out_tokens)
    "e1": (2, 520, 30),
    "e2": (20, 700, 12),   # per episode (2 agents x up to 10 rounds)
    "e3": (2, 560, 20),
    "e4": (2, 480, 8),
}


def project_cost(cells: dict[str, dict]) -> dict:
    out, total = {}, 0.0
    for name, cell in cells.items():
        env = cell["env"]
        cpu, tin, tout = CALL_SHAPES[env]
        pin, pout = PRICES[cell.get("model", "anthropic/claude-haiku-4.5")]
        if cell.get("dialogue"):
            cpu += 2           # one message call per agent per episode
            tout += 60         # chatty turns get a larger reply cap
        if cell.get("think"):
            tout += 150        # scratch line + room to finish the reply
        if env == "e2":
            units = (cell["formation_episodes"]
                     + cell["n_agents"] * cell["episodes_per_replacement"]
                     + cell["settle_episodes"])
        else:
            units = (cell.get("formation_interactions", 0)
                     + cell["n_agents"] * cell.get(
                         "interactions_per_replacement", 0)
                     + cell.get("settle_interactions", 0)
                     + cell.get("minority_interactions", 0))
        calls = units * cpu * 1.1 + 60   # +gates/priors margin
        per_pop = calls * (tin * pin + tout * pout)
        out[name] = round(per_pop, 2)
        total += per_pop * cell["n_pops"]
    return {"per_pop_usd": out, "total_upper_bound_usd": round(total, 2)}


def run_cell_pop(args):
    cell_name, cfg, outdir, backend_factory = args
    run_dir = os.path.join(outdir, f"{cell_name}_p{cfg['pop_index']:02d}")
    done_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f)
    env = ENVS[cfg["env"]]
    backend = backend_factory(cfg)
    result = env.run_population(cfg, backend, run_dir)
    summary = {"config": {k: v for k, v in cfg.items()}, "result": result}
    os.makedirs(run_dir, exist_ok=True)
    with open(done_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"done {cell_name} p{cfg['pop_index']}: {result}", flush=True)
    return summary


def main_envs(args) -> None:
    cells = define_cells(args.mode)
    if args.only:
        cells = {k: v for k, v in cells.items() if k in args.only}
    # second-family route: --model swaps the subject model for EVERY
    # selected cell (frozen prompts); --npops CELL=N sizes the cells
    model_override = getattr(args, "model", None)
    if model_override:
        cells = {k: {**v, "model": model_override} for k, v in cells.items()}
    for kv in (getattr(args, "npops", None) or []):
        k, n = kv.split("=")
        if k in cells:
            cells[k] = {**cells[k], "n_pops": int(n)}
    outdir = args.outdir or f"results/envs_{args.mode}"
    os.makedirs(outdir, exist_ok=True)

    proj = project_cost(cells)
    print("=== E1-E4 cost projection (upper bound, live pricing) ===")
    print(json.dumps(proj, indent=2))
    if args.project_cost:
        return

    if args.mode == "live":
        if proj["total_upper_bound_usd"] > args.spend_cap_usd:
            raise SystemExit("projection exceeds spend cap; refusing "
                             "(stop and flag, do not trim silently)")
        cost, factory = _live_backend_factory(outdir, args.spend_cap_usd)
        print(f"live backend ready; spend so far ${cost.cost_usd:.2f} of "
              f"${args.spend_cap_usd:.2f}")
        from .common import SplitBackend

        def backend_factory(cfg):
            if cfg.get("hetero"):
                return SplitBackend(
                    OpenRouterBackend("anthropic/claude-haiku-4.5", cost),
                    OpenRouterBackend("openai/gpt-5-mini", cost),
                    split=cfg["n_agents"] // 2)
            if cfg.get("model"):
                return OpenRouterBackend(cfg["model"], cost)
            return factory()
    else:
        backend_factory = lambda cfg: EnvMockBackend(
            ENVS[cfg["env"]].mock_reply, seed=cfg["seed"] ^ 0x5EED)

    jobs = []
    for name, cell in cells.items():
        for idx in range(cell["n_pops"]):
            jobs.append((name, pop_config(name, cell, idx), outdir,
                         backend_factory))

    workers = args.workers or (12 if args.mode == "live" else 4)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run_cell_pop, jobs))
    except SpendCapExceeded as e:
        print(f"SPEND CAP HIT: {e}", flush=True)
        raise
    print("ALL ENV CELLS DONE", flush=True)
