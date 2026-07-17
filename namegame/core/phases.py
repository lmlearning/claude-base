"""Phase drivers for the cheap tier: genesis, transmission, perturbation.

Each driver is a pure function of (configs, seed) returning a JSON-able
result dict; full determinism from the seed.
"""

from __future__ import annotations

import numpy as np

from .agents import AGENT_CLASSES, Agent, CommittedAgent
from .config import GameConfig, GenesisConfig, MinorityConfig, TransmissionConfig
from .engine import Population, conformity_time


def _make_population(cfg: GameConfig, rng: np.random.Generator,
                     agent_kind: str) -> Population:
    cls = AGENT_CLASSES[agent_kind]
    agents = [cls(i, cfg.n_names, cfg.memory_size,
                  np.random.default_rng(rng.integers(2**63)))
              for i in range(cfg.n_agents)]
    return Population(agents, cfg, np.random.default_rng(rng.integers(2**63)))


def _fresh_agent(cfg: GameConfig, rng: np.random.Generator, agent_kind: str,
                 agent_id: int) -> Agent:
    cls = AGENT_CLASSES[agent_kind]
    return cls(agent_id, cfg.n_names, cfg.memory_size,
               np.random.default_rng(rng.integers(2**63)))


def run_to_consensus(pop: Population, max_interactions: int,
                     scheduler: str = "sequential") -> int | None:
    """Advance until the consensus criterion first holds; return winner or None."""
    check_every = 1 if scheduler == "sequential" else None
    while pop.t < max_interactions:
        if scheduler == "sequential":
            pop.step()
        else:
            pop.step_round()
        winner = pop.tracker.consensus()
        if winner is not None:
            return winner
    return None


def genesis(game: GameConfig, gen: GenesisConfig, seed: int,
            agent_kind: str = "minimal", scheduler: str = "sequential",
            keep_trajectory: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    pop = _make_population(game, rng, agent_kind)
    winner = run_to_consensus(pop, gen.max_interactions, scheduler)
    consensus_t = pop.t if winner is not None else None
    if winner is not None:
        for _ in range(gen.post_consensus_interactions):
            if scheduler == "sequential":
                pop.step()
            else:
                pop.step_round()
    result = {
        "phase": "genesis",
        "agent_kind": agent_kind,
        "scheduler": scheduler,
        "seed": seed,
        "converged": winner is not None,
        "winner": winner,
        "consensus_time": consensus_t,
        "max_interactions": gen.max_interactions,
    }
    if winner is not None:
        result["founder_conformity"] = [
            conformity_time(a.plays, winner, game.conformity_block,
                            game.conformity_min_matches)
            for a in pop.agents
        ]
        result["founder_n_plays"] = [len(a.plays) for a in pop.agents]
    if keep_trajectory:
        # 100-interaction-bin mean success rate.
        s = np.asarray(pop.success_log, dtype=float)
        nbins = len(s) // 100
        result["success_traj_100"] = (
            s[: nbins * 100].reshape(nbins, 100).mean(axis=1).round(3).tolist()
            if nbins else []
        )
    result["_population"] = pop  # stripped before serialisation
    return result


def _survival_probe(pop: Population, original: int, settle: int) -> dict:
    """Clone, freeze turnover, settle, apply consensus criterion."""
    probe = pop.clone()
    for _ in range(settle):
        probe.step()
    name = probe.tracker.consensus()
    return {
        "consensus": name is not None,
        "name": name,
        "survived": name == original,
        "final_success_rate": probe.tracker.success_rate(),
        "dominant": probe.tracker.dominant(),
    }


def transmission(game: GameConfig, gen: GenesisConfig, trans: TransmissionConfig,
                 seed: int, agent_kind: str = "minimal") -> dict:
    """Genesis to consensus, then generational turnover.

    A generation replaces the N slots in a fresh random order, one slot
    every ``interactions_per_replacement`` interactions.  Survival probes
    (clone + settle + criterion) at the end of each generation.
    """
    rng = np.random.default_rng(seed)
    g = genesis(game, gen, int(rng.integers(2**63)), agent_kind)
    pop: Population = g.pop("_population")
    if not g["converged"]:
        return {"phase": "transmission", "seed": seed, "converged_genesis": False}
    original = g["winner"]

    k = trans.interactions_per_replacement
    next_id = game.n_agents
    newcomers = []   # dicts with insertion info
    probes = []
    replacements_done = 0

    r = trans.replacements_per_event
    for generation in range(trans.generations):
        order = list(rng.permutation(game.n_agents))
        while order:
            for _ in range(k):
                pop.step()
            for slot in order[:r]:
                agent = _fresh_agent(game, rng, agent_kind, next_id)
                agent.born_at = pop.t
                pop.agents[int(slot)] = agent
                newcomers.append({"agent": agent, "generation": generation,
                                  "slot": int(slot), "born_at": pop.t})
                next_id += 1
                replacements_done += 1
            order = order[r:]
        probes.append(_survival_probe(pop, original, trans.settle_interactions))

    for _ in range(trans.post_interactions):
        pop.step()

    # Newcomer conformity toward the ORIGINAL convention name.
    newcomer_records = []
    for rec in newcomers:
        a = rec["agent"]
        ct = conformity_time(a.plays, original, game.conformity_block,
                             game.conformity_min_matches)
        newcomer_records.append({
            "generation": rec["generation"],
            "born_at": rec["born_at"],
            "n_plays": len(a.plays),
            "conformity_time": ct,
        })

    return {
        "phase": "transmission",
        "agent_kind": agent_kind,
        "seed": seed,
        "converged_genesis": True,
        "original_winner": original,
        "genesis_time": g["consensus_time"],
        "interactions_per_replacement": k,
        "replacements_per_event": r,
        "turnover_rate": r / k,
        "generations": trans.generations,
        "probes": probes,
        "survived_gen1": probes[0]["survived"] if probes else None,
        "survived_gen2": probes[-1]["survived"] if len(probes) >= 2 else None,
        "newcomers": newcomer_records,
        "founder_conformity": g.get("founder_conformity"),
        "total_interactions": pop.t,
    }


def committed_minority(game: GameConfig, gen: GenesisConfig, mino: MinorityConfig,
                       seed: int, agent_kind: str = "minimal",
                       pre_transmission: TransmissionConfig | None = None) -> dict:
    """Insert a committed minority into a converged population.

    If ``pre_transmission`` is given, one full generation of turnover (plus
    settle) happens first — the *post-transmission* condition; the run is
    discarded (reported unusable) if the convention did not survive it.
    """
    rng = np.random.default_rng(seed)
    g = genesis(game, gen, int(rng.integers(2**63)), agent_kind)
    pop: Population = g.pop("_population")
    if not g["converged"]:
        return {"phase": "minority", "seed": seed, "usable": False,
                "reason": "genesis_no_consensus"}
    original = g["winner"]
    condition = "founder"

    if pre_transmission is not None:
        condition = "post_transmission"
        k = pre_transmission.interactions_per_replacement
        next_id = game.n_agents
        for slot in rng.permutation(game.n_agents):
            for _ in range(k):
                pop.step()
            agent = _fresh_agent(game, rng, agent_kind, next_id)
            agent.born_at = pop.t
            pop.agents[int(slot)] = agent
            next_id += 1
        for _ in range(pre_transmission.settle_interactions):
            pop.step()
        if pop.tracker.consensus() != original:
            return {"phase": "minority", "seed": seed, "usable": False,
                    "condition": condition, "reason": "convention_lost_pre_minority"}

    alternatives = [n for n in range(game.n_names) if n != original]
    alt = int(alternatives[rng.integers(len(alternatives))])
    slots = rng.choice(game.n_agents, size=mino.n_committed, replace=False)
    for slot in slots:
        pop.agents[int(slot)] = CommittedAgent(
            10_000 + int(slot), game.n_names, game.memory_size,
            np.random.default_rng(rng.integers(2**63)), committed_name=alt)

    start_t = pop.t
    flipped = False
    flip_time = None
    for _ in range(mino.budget_interactions):
        pop.step()
        if pop.tracker.full:
            name, share = pop.tracker.noncommitted_dominant()
            if name == alt and share >= game.consensus_dominance_threshold:
                flipped = True
                flip_time = pop.t - start_t
                break

    name, share = pop.tracker.noncommitted_dominant()
    return {
        "phase": "minority",
        "agent_kind": agent_kind,
        "seed": seed,
        "usable": True,
        "condition": condition,
        "n_committed": mino.n_committed,
        "committed_fraction": mino.n_committed / game.n_agents,
        "original_winner": original,
        "alt_name": alt,
        "flipped": flipped,
        "flip_time": flip_time,
        "budget": mino.budget_interactions,
        "final_nc_dominant": [name, round(share, 4)],
    }


def transplant(game: GameConfig, gen: GenesisConfig, seed: int,
               agent_kind: str = "minimal",
               observe_interactions: int = 6_000) -> dict:
    """One converged agent moved into a population converged on a
    different name; measure its switch and host stability."""
    rng = np.random.default_rng(seed)
    ga = genesis(game, gen, int(rng.integers(2**63)), agent_kind)
    gb = genesis(game, gen, int(rng.integers(2**63)), agent_kind)
    pa, pb = ga.pop("_population"), gb.pop("_population")
    if not (ga["converged"] and gb["converged"]):
        return {"phase": "transplant", "seed": seed, "usable": False,
                "reason": "genesis_no_consensus"}
    if ga["winner"] == gb["winner"]:
        return {"phase": "transplant", "seed": seed, "usable": False,
                "reason": "same_winner"}

    donor = pa.agents[int(rng.integers(game.n_agents))]
    donor.plays = []
    slot = int(rng.integers(game.n_agents))
    pb.agents[slot] = donor
    for _ in range(observe_interactions):
        pb.step()

    ct = conformity_time(donor.plays, gb["winner"], game.conformity_block,
                         game.conformity_min_matches)
    host = pb.tracker.consensus()
    return {
        "phase": "transplant",
        "agent_kind": agent_kind,
        "seed": seed,
        "usable": True,
        "donor_name": ga["winner"],
        "host_name": gb["winner"],
        "switch_time": ct,          # in donor own-plays after insertion
        "donor_n_plays": len(donor.plays),
        "switched": ct is not None,
        "host_still_consensus": host == gb["winner"],
        "host_final": host,
    }
