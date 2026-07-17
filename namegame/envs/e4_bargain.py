"""E4 — Tagged bargaining: emergent role/status conventions.

Task: after each jointly credited task, a pair splits a bonus.  Each
side simultaneously requests 30, 50 or 70 points; if the requests total
more than 100 both get nothing, otherwise each gets what they asked for.
Every agent carries a persistent, payoff-irrelevant badge (one of two
nonsense words; half the population each).  Both see both badges.

Theory tier: Axtell, Epstein & Young (2001) — inductive best-responders
in this game can converge to badge-conditioned 'classes' (one badge
demands 70 against the other, which concedes 30) purely by symmetry
breaking, or to the egalitarian 50/50 norm.  The badge->demand mapping is
payoff-symmetric, so any systematic mapping is an arbitrary convention.

Convention variables (mechanical): demand distributions conditioned on
(own badge, partner badge); equilibrium classification per population
(egalitarian / class / fractious / mixed).
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

from ..expb.backend import Backend
from ..expb.tokens import generate_pool
from . import common
from .common import Journal, lint_env, rng_for

H = 8
DEMANDS = [30, 50, 70]

FRAMINGS = {
    "bonus": (
        "You are one of several participants who repeatedly finish small "
        "joint tasks for points. After each task you and that round's "
        "partner split a bonus: each of you privately requests 30, 50 or "
        "70 points. If the two requests add up to more than 100, neither "
        "of you gets anything this round; otherwise each of you receives "
        "exactly what you requested. Every participant permanently "
        "carries one of two printed badges, {badge_a} or {badge_b}; you "
        "always see your own badge and your partner's badge. Your aim is "
        "to collect as many points as possible. You can see your recent "
        "rounds."
    ),
    "market": (
        "In a repeated exchange, you are matched with a different "
        "counterpart each round to divide 100 points of proceeds. Both "
        "sides at the same time name a claim of 30, 50 or 70. Claims "
        "totalling more than 100 mean the round pays nothing to either "
        "side; otherwise each side is paid its claim. Each trader wears a "
        "fixed tag, {badge_a} or {badge_b}, visible to both sides in "
        "every match. Maximise your points. Your recent matches are "
        "shown to you."
    ),
}

TURN_INSTR = ("This round: your badge is {own_badge}; your partner's badge "
              "is {partner_badge}. Request your share of the bonus. Reply "
              "with only one number: 30, 50 or 70.")
HIST_LINE = ("- you ({own_badge}) requested {own}; partner "
             "({partner_badge}) requested {partner}; you received "
             "{got} points")
PRIOR_PROBE = ("You and a partner split a bonus: each privately requests "
               "30, 50 or 70 points; if the requests total more than 100 "
               "neither gets anything, otherwise each receives what they "
               "requested. Your badge is {own_badge}; your partner's badge "
               "is {partner_badge}. Reply with only one number: 30, 50 "
               "or 70.")

lint_env(list(FRAMINGS.values()) + [TURN_INSTR, HIST_LINE, PRIOR_PROBE])


def mock_reply(context: dict, rng: np.random.Generator) -> str:
    kind = context["kind"]
    if kind == "comprehension":
        return context["true_answer"]
    # kind in ("turn", "prior"): best response to remembered partner
    # demands conditioned on partner badge (Axtell-Epstein-Young style)
    if kind == "prior" or rng.random() < 0.1:
        return str(DEMANDS[int(rng.integers(3))])
    obs = context.get("partner_demands", [])   # demands seen from this badge
    if not obs:
        return str(DEMANDS[int(rng.integers(3))])
    counts = {d: obs.count(d) for d in DEMANDS}
    n = len(obs)
    best, best_val = 50, -1.0
    for d in DEMANDS:
        p_ok = sum(c for pd, c in counts.items() if pd + d <= 100) / n
        ev = d * p_ok
        if ev > best_val:
            best, best_val = d, ev
    return str(best)


class E4Agent:
    def __init__(self, agent_id: int, badge: str):
        self.agent_id = agent_id
        self.badge = badge
        self.memory: deque = deque(maxlen=H)
        self.born_at = 0
        # partner-badge -> recent partner demands (substrate policy state)
        self.partner_obs: dict[str, deque] = {}

    def note(self, own, partner_badge, partner, got):
        self.memory.append({"own": own, "partner_badge": partner_badge,
                            "partner": partner, "got": got})
        self.partner_obs.setdefault(partner_badge,
                                    deque(maxlen=H)).append(partner)


def _system(cfg):
    return FRAMINGS[cfg["framing"]].format(badge_a=cfg["badges"][0],
                                           badge_b=cfg["badges"][1])


def _hist(agent: E4Agent) -> str:
    if not agent.memory:
        return "You have no previous rounds yet."
    lines = ["Your recent rounds, oldest first:"]
    for e in agent.memory:
        lines.append(HIST_LINE.format(own_badge=agent.badge, own=e["own"],
                                      partner_badge=e["partner_badge"],
                                      partner=e["partner"], got=e["got"]))
    return "\n".join(lines)


def _parse_demand(reply: str) -> int | None:
    m = re.search(r"\b(30|50|70)\b", reply)
    return int(m.group(1)) if m else None


def _one_round(t, agents, cfg, backend, journal, phase, i, j):
    seed = cfg["seed"]
    A, B = agents[i], agents[j]
    demands = {}
    malformed = 0
    for slot, me, other in ((i, A, B), (j, B, A)):
        user = "\n\n".join([_hist(me), TURN_INSTR.format(
            own_badge=me.badge, partner_badge=other.badge)])
        reply = backend.complete(
            _system(cfg), user, 8,
            {"kind": "turn",
             "partner_demands": list(me.partner_obs.get(other.badge, []))})
        d = _parse_demand(reply)
        if d is None:
            malformed += 1
            d = DEMANDS[int(rng_for(seed, 7, t, slot).integers(3))]
        demands[slot] = d
    da, db = demands[i], demands[j]
    ok = da + db <= 100
    rec = {"type": "interaction", "t": t, "phase": phase,
           "slot_a": i, "slot_b": j, "id_a": A.agent_id, "id_b": B.agent_id,
           "badge_a": A.badge, "badge_b": B.badge,
           "demand_a": da, "demand_b": db, "compatible": ok,
           "malformed": malformed}
    _apply(agents, rec)
    journal.append(rec)
    return rec


def _apply(agents, rec):
    A, B = agents[rec["slot_a"]], agents[rec["slot_b"]]
    ok = rec["compatible"]
    A.note(rec["demand_a"], rec["badge_b"], rec["demand_b"],
           rec["demand_a"] if ok else 0)
    B.note(rec["demand_b"], rec["badge_a"], rec["demand_a"],
           rec["demand_b"] if ok else 0)


def measure_priors(cfg, backend, journal, reps=6) -> dict:
    out = []
    for own in cfg["badges"]:
        for partner in cfg["badges"]:
            for rep in range(reps):
                reply = backend.complete(
                    "", PRIOR_PROBE.format(own_badge=own,
                                           partner_badge=partner), 8,
                    {"kind": "prior"})
                out.append({"own": own, "partner": partner,
                            "demand": _parse_demand(reply)})
    rec = {"type": "priors", "probes": out}
    journal.append(rec)
    return rec


COMPREHENSION = [
    ("If the two requests add up to more than 100, how many points do you "
     "get that round? Reply with just the number.", "0"),
    ("If you request 70 and your partner requests 30, how many points do "
     "you receive? Reply with just the number.", "70"),
    ("How many badge types are there? Reply with just the number.", "2"),
]


def run_population(cfg: dict, backend: Backend, run_dir: str) -> dict:
    journal = Journal(run_dir)
    seed = cfg["seed"]
    n = cfg["n_agents"]

    def badge_of(agent_id):
        # persistent arbitrary badge: half and half by id parity at genesis;
        # newcomers alternate to keep counts balanced
        return cfg["badges"][agent_id % 2]

    agents = None
    next_id = n
    done = {"config": False, "comprehension": None, "t": 0,
            "replacements": 0, "priors": False}
    pending = []
    for rec in Journal(run_dir).replay():
        pending.append(rec)
        if rec["type"] == "config":
            cfg = {**cfg, "badges": rec["badges"]}
            done["config"] = True
    if not done["config"]:
        cfg = {**cfg, "badges": generate_pool(2, rng_for(seed, 1))}
    agents = [E4Agent(i, badge_of(i)) for i in range(n)]
    for rec in pending:
        if rec["type"] == "comprehension":
            done["comprehension"] = rec["passed"]
        elif rec["type"] == "interaction":
            _apply(agents, rec)
            done["t"] = rec["t"]
        elif rec["type"] == "replacement":
            a = E4Agent(rec["agent_id"], badge_of(rec["agent_id"]))
            a.born_at = rec["t"]
            agents[rec["slot"]] = a
            next_id = max(next_id, rec["agent_id"] + 1)
            done["replacements"] += 1
        elif rec["type"] == "priors":
            done["priors"] = True

    if not done["config"]:
        journal.append({"type": "config", **cfg, "env": "e4"})
    if done["comprehension"] is None:
        if not common.run_comprehension(backend, _system(cfg),
                                        COMPREHENSION, journal):
            return {"aborted": "comprehension_gate_failed"}
    elif done["comprehension"] is False:
        return {"aborted": "comprehension_gate_failed"}
    if not done["priors"]:
        measure_priors(cfg, backend, journal)

    t = done["t"]

    def pair(t):
        r = rng_for(seed, 3, t)
        i = int(r.integers(n))
        j = int(r.integers(n - 1))
        if j >= i:
            j += 1
        return i, j

    while t < cfg["formation_interactions"]:
        t += 1
        i, j = pair(t)
        _one_round(t, agents, cfg, backend, journal, "formation", i, j)

    k = cfg["interactions_per_replacement"]
    schedule = common.generation_schedule(n, seed, 1)
    base = cfg["formation_interactions"]
    for gi, slot in enumerate(schedule):
        due = base + (gi + 1) * k
        if gi < done["replacements"]:
            continue
        while t < due:
            t += 1
            i, j = pair(t)
            _one_round(t, agents, cfg, backend, journal, "turnover", i, j)
        a = E4Agent(next_id, badge_of(next_id))
        a.born_at = t
        agents[slot] = a
        journal.append({"type": "replacement", "t": t, "slot": slot,
                        "agent_id": next_id})
        next_id += 1
    end = base + n * k + cfg["settle_interactions"]
    while t < end:
        t += 1
        i, j = pair(t)
        _one_round(t, agents, cfg, backend, journal, "settle", i, j)
    return {"done": True, "t": t}
