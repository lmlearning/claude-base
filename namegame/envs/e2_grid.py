"""E2 — Joint grid assembly: behavioral conventions via symmetry breaking.

Task: a pair must fill every cell of a 4x4 grid.  Each round BOTH agents
simultaneously pick one empty cell; if they pick the same cell the round
is wasted for both.  The pair succeeds if the grid is full within the
move budget (12 rounds for 16 cells).  Reward +100 each on success, -50
each on failure.  One agent per episode is randomly labelled with badge A,
the other with badge B (two nonsense words, fixed per run).

By construction every division of labour is reward-equivalent (the task
is invariant under swapping rows/columns/badges), so any population-level
regularity in who-takes-which-region is an arbitrary convention.

Convention variables (mechanical): per agent-episode, the region class of
the cells it placed (left/right column halves, top/bottom row halves,
checkerboard parities, other) and the population's badge->region mapping.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

from ..expb.backend import Backend
from ..expb.tokens import generate_pool
from . import common
from .common import Journal, lint_env, rng_for

K = 4                 # grid side
ROUNDS = 9           # move budget (16 cells/2 per clean round = 8 min; 1 spare)
HIST_EPISODES = 3

FRAMINGS = {
    "site": (
        "You and one other participant fill in a {K}x{K} grid together, "
        "over repeated episodes, for points. Each round, you both pick one "
        "empty cell at the same time, without seeing the other's current "
        "pick; if you pick the same cell, that round places nothing. The "
        "episode succeeds if every cell is filled within {rounds} rounds: "
        "you each earn {reward} points. If the grid is not complete in "
        "time, you each lose {penalty} points. In every episode one of you "
        "is the {badge_a} participant and the other is the {badge_b} "
        "participant; the labels are assigned at random each episode. Your "
        "aim is to collect as many points as possible. You can see your "
        "recent episodes."
    ),
    "workshop": (
        "Across many short jobs you and a partner complete a {K}-by-{K} "
        "board for points. In each round of a job, the two of you "
        "simultaneously choose one open square each (neither sees the "
        "other's choice before it is made); choosing the same square "
        "wastes the round. Finishing the whole board within {rounds} "
        "rounds earns each of you {reward} points; failing to finish "
        "costs each of you {penalty} points. At the start of every job "
        "one partner is tagged {badge_a} and the other {badge_b}, chosen "
        "at random. Gather the most points you can. Your recent jobs are "
        "shown to you."
    ),
}

TURN_INSTR = ("Episode state - you are the {badge} participant. Grid "
              "(O = filled by you, P = filled by partner, . = empty):\n"
              "{grid}\nRound {rnd} of {rounds}. Pick one empty cell. "
              "Reply with only: CELL: row,col  (rows and columns are "
              "numbered 1-{K}).")
TURN_INSTR_THINK = ("Episode state - you are the {badge} participant. Grid "
                    "(O = filled by you, P = filled by partner, . = empty):\n"
                    "{grid}\nRound {rnd} of {rounds}. Pick one empty cell. "
                    "First write one short line about your choice, then on "
                    "a new line write: CELL: row,col  (rows and columns "
                    "are numbered 1-{K}).")
HIST_LINE = ("- episode as {badge} participant: you filled {cells}; "
             "outcome {outcome}")
HIST_MSGS = (" ; you sent \"{msg}\", partner sent \"{pmsg}\"")

MSG_WORDS = 16
DIALOGUE_SYS = (" Before each of these begins, you and your partner may "
                "each send the other one short message (at most "
                "{msg_words} words).")
MSG_INSTR = ("A new episode is starting - you are the {badge} participant. "
             "You may send your partner one message of at most {msg_words} "
             "words before play begins. Reply with only the message itself "
             "and nothing else, or the single word NONE.")
MSG_BLOCK = ("Message you sent: \"{msg}\"\nMessage from your partner: "
             "\"{pmsg}\"")

lint_env(list(FRAMINGS.values()) + [TURN_INSTR, TURN_INSTR_THINK, HIST_LINE,
                                    HIST_MSGS, DIALOGUE_SYS, MSG_INSTR,
                                    MSG_BLOCK])


# ---------------------------------------------------------------------------
# region classification (shared with analysis)
# ---------------------------------------------------------------------------

REGIONS = {
    "left": lambda r, c: c < K // 2,
    "right": lambda r, c: c >= K // 2,
    "top": lambda r, c: r < K // 2,
    "bottom": lambda r, c: r >= K // 2,
    "even": lambda r, c: (r + c) % 2 == 0,
    "odd": lambda r, c: (r + c) % 2 == 1,
}


def classify_cells(cells: list[tuple[int, int]]) -> str:
    """Region class if >= 7 of the agent's placed cells fall in it."""
    if len(cells) < 6:
        return "short"
    best, best_n = "mixed", 0
    for name, fn in REGIONS.items():
        m = sum(1 for r, c in cells if fn(r, c))
        if m > best_n:
            best, best_n = name, m
    return best if best_n >= max(6, len(cells) - 1) else "mixed"


# ---------------------------------------------------------------------------
# mock policy
# ---------------------------------------------------------------------------

def mock_reply(context: dict, rng: np.random.Generator) -> str:
    if context["kind"] == "comprehension":
        return context["true_answer"]
    if context["kind"] == "msg":
        return "NONE"        # the substrate policy has no language faculty
    # kind == "turn": epsilon-greedy region preference from memory
    empty = context["empty"]           # list of (r,c)
    badge = context["badge"]
    prefs = context.get("region_prefs", {})   # {badge: {region: score}}
    if rng.random() < 0.15 or badge not in prefs or not prefs[badge]:
        r, c = empty[int(rng.integers(len(empty)))]
    else:
        scores = prefs[badge]
        best = max(scores.values())
        cands = [reg for reg, s in scores.items() if s == best]
        reg = cands[int(rng.integers(len(cands)))]
        in_reg = [rc for rc in empty if REGIONS[reg](*rc)]
        pool = in_reg or empty
        r, c = pool[int(rng.integers(len(pool)))]
    return f"CELL: {r + 1},{c + 1}"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class E2Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.memory: deque = deque(maxlen=HIST_EPISODES)  # episode summaries
        self.born_at = 0
        # substrate learning state (mock): badge -> region -> score
        self.region_prefs: dict = {}

    def note_episode(self, badge, cells, success, episode_idx,
                     msg=None, pmsg=None):
        self.memory.append({"badge": badge, "cells": cells,
                            "success": success, "msg": msg, "pmsg": pmsg})
        cls = classify_cells(cells)
        if cls in REGIONS:
            d = self.region_prefs.setdefault(badge, {})
            d[cls] = d.get(cls, 0.0) + (1.0 if success else -0.5)


def _system(cfg):
    s = FRAMINGS[cfg["framing"]].format(
        K=K, rounds=ROUNDS, reward=100, penalty=50,
        badge_a=cfg["badges"][0], badge_b=cfg["badges"][1])
    if cfg.get("dialogue"):
        s += DIALOGUE_SYS.format(msg_words=MSG_WORDS)
    return s


def _grid_text(mine, partner):
    rows = []
    for r in range(K):
        rows.append(" ".join("O" if (r, c) in mine else
                             "P" if (r, c) in partner else "."
                             for c in range(K)))
    return "\n".join(rows)


def _hist(agent: E2Agent, cfg) -> str:
    if not agent.memory:
        return "You have no previous episodes yet."
    lines = ["Your recent episodes, oldest first:"]
    for e in agent.memory:
        line = HIST_LINE.format(
            badge=e["badge"],
            cells=" ".join(f"({r+1},{c+1})" for r, c in e["cells"]),
            outcome="+100" if e["success"] else "-50")
        if cfg.get("dialogue") and (e.get("msg") or e.get("pmsg")):
            line += HIST_MSGS.format(msg=e.get("msg") or "NONE",
                                     pmsg=e.get("pmsg") or "NONE")
        lines.append(line)
    return "\n".join(lines)


def _parse_cell(reply: str) -> tuple[int, int] | None:
    ms = re.findall(r"CELL\s*[:\s]\s*(\d)\s*[,x/ ]\s*(\d)", reply,
                    flags=re.IGNORECASE)
    if ms:
        g = ms[-1]        # last CELL:-anchored pair (after any scratch line)
    else:
        m = re.search(r"(\d)\s*[,x/ ]\s*(\d)", reply)
        if not m:
            return None
        g = m.groups()
    r, c = int(g[0]) - 1, int(g[1]) - 1
    if 0 <= r < K and 0 <= c < K:
        return (r, c)
    return None


def _play_episode(ep: int, agents, slot_a, slot_b, cfg, backend, journal):
    """slot_a gets badge[0], slot_b gets badge[1] (assignment randomized by
    caller). Simultaneous picks each round."""
    seed = cfg["seed"]
    A, B = agents[slot_a], agents[slot_b]
    placed = {slot_a: [], slot_b: []}
    rounds = []
    malformed = 0
    parse_fail = 0
    occupied = 0

    msgs = {slot_a: "", slot_b: ""}
    if cfg.get("dialogue"):
        for slot, me, badge in ((slot_a, A, cfg["badges"][0]),
                                (slot_b, B, cfg["badges"][1])):
            user = "\n\n".join([_hist(me, cfg), MSG_INSTR.format(
                badge=badge, msg_words=MSG_WORDS)])
            raw = backend.complete(_system(cfg), user, 100,
                                   {"kind": "msg", "badge": badge})
            msg, _ = common.word_count_clip(raw, MSG_WORDS)
            msgs[slot] = msg

    turn_tpl = TURN_INSTR_THINK if cfg.get("think") else TURN_INSTR
    # chatty contexts (scratch line, message channel) need room to reach
    # the CELL line; the parser takes the LAST CELL:-anchored pair
    turn_max = (200 if cfg.get("think")
                else 120 if cfg.get("dialogue") else 20)
    for rnd in range(ROUNDS):
        picks = {}
        for slot, me, other, badge in ((slot_a, A, B, cfg["badges"][0]),
                                       (slot_b, B, A, cfg["badges"][1])):
            mine = set(placed[slot])
            partner = set(placed[slot_b if slot == slot_a else slot_a])
            empty = [(r, c) for r in range(K) for c in range(K)
                     if (r, c) not in mine and (r, c) not in partner]
            if not empty:
                picks[slot] = None
                continue
            parts = [_hist(me, cfg)]
            if cfg.get("dialogue"):
                oslot = slot_b if slot == slot_a else slot_a
                parts.append(MSG_BLOCK.format(msg=msgs[slot] or "NONE",
                                              pmsg=msgs[oslot] or "NONE"))
            parts.append(turn_tpl.format(
                badge=badge, grid=_grid_text(mine, partner),
                rnd=rnd + 1, rounds=ROUNDS, K=K))
            user = "\n\n".join(parts)
            reply = backend.complete(_system(cfg), user, turn_max,
                                     {"kind": "turn", "empty": empty,
                                      "badge": badge,
                                      "region_prefs": me.region_prefs})
            cell = _parse_cell(reply)
            if cell is None or cell not in empty:
                malformed += 1
                parse_fail += cell is None
                occupied += cell is not None
                cell = empty[int(rng_for(seed, 7, ep, rnd,
                                         slot).integers(len(empty)))]
            picks[slot] = cell
        pa, pb = picks[slot_a], picks[slot_b]
        collided = pa is not None and pa == pb
        if not collided:
            if pa is not None:
                placed[slot_a].append(pa)
            if pb is not None and pb != pa:
                placed[slot_b].append(pb)
        rounds.append({"a": pa, "b": pb, "collision": collided})
        if len(placed[slot_a]) + len(placed[slot_b]) >= K * K:
            break
    success = len(placed[slot_a]) + len(placed[slot_b]) >= K * K
    rec = {"type": "episode", "ep": ep, "slot_a": slot_a, "slot_b": slot_b,
           "id_a": A.agent_id, "id_b": B.agent_id,
           "cells_a": placed[slot_a], "cells_b": placed[slot_b],
           "class_a": classify_cells(placed[slot_a]),
           "class_b": classify_cells(placed[slot_b]),
           "success": success, "n_rounds": len(rounds),
           "collisions": sum(r["collision"] for r in rounds),
           "malformed": malformed, "parse_fail": parse_fail,
           "occupied": occupied,
           "msg_a": msgs[slot_a], "msg_b": msgs[slot_b]}
    _apply(agents, rec, cfg)
    journal.append(rec)
    return rec


def _apply(agents, rec, cfg):
    A, B = agents[rec["slot_a"]], agents[rec["slot_b"]]
    A.note_episode(cfg["badges"][0],
                   [tuple(c) for c in rec["cells_a"]], rec["success"],
                   rec["ep"], msg=rec.get("msg_a"), pmsg=rec.get("msg_b"))
    B.note_episode(cfg["badges"][1],
                   [tuple(c) for c in rec["cells_b"]], rec["success"],
                   rec["ep"], msg=rec.get("msg_b"), pmsg=rec.get("msg_a"))


COMPREHENSION = [
    ("If the grid is completely filled within the round limit, how many "
     "points do you each earn? Reply with just the number.", "100"),
    ("How many rounds does an episode allow at most? Reply with just the "
     "number.", str(ROUNDS)),
    ("If you and your partner pick the same cell in a round, how many "
     "cells get filled that round? Reply with just the number.", "0"),
]


def run_population(cfg: dict, backend: Backend, run_dir: str) -> dict:
    journal = Journal(run_dir)
    seed = cfg["seed"]
    n = cfg["n_agents"]
    agents = [E2Agent(i) for i in range(n)]
    next_id = n
    done = {"config": False, "comprehension": None, "eps": 0,
            "replacements": 0}
    for rec in journal.replay():
        if rec["type"] == "config":
            done["config"] = True
            cfg = {**cfg, "badges": rec["badges"]}
        elif rec["type"] == "comprehension":
            done["comprehension"] = rec["passed"]
        elif rec["type"] == "episode":
            _apply(agents, rec, cfg)
            done["eps"] = rec["ep"]
        elif rec["type"] == "replacement":
            a = E2Agent(rec["agent_id"])
            a.born_at = rec["ep"]
            agents[rec["slot"]] = a
            next_id = max(next_id, rec["agent_id"] + 1)
            done["replacements"] += 1

    if not done["config"]:
        cfg = {**cfg, "badges": generate_pool(2, rng_for(seed, 1))}
        journal.append({"type": "config", **{k: v for k, v in cfg.items()},
                        "env": "e2"})
    if done["comprehension"] is None:
        if not common.run_comprehension(backend, _system(cfg),
                                        COMPREHENSION, journal):
            return {"aborted": "comprehension_gate_failed"}
    elif done["comprehension"] is False:
        return {"aborted": "comprehension_gate_failed"}

    ep = done["eps"]

    def sample_pair(ep):
        r = rng_for(seed, 3, ep)
        i = int(r.integers(n))
        j = int(r.integers(n - 1))
        if j >= i:
            j += 1
        if r.random() < 0.5:      # randomize badge assignment
            return i, j
        return j, i

    while ep < cfg["formation_episodes"]:
        ep += 1
        a, b = sample_pair(ep)
        _play_episode(ep, agents, a, b, cfg, backend, journal)

    # turnover: one generation, one replacement per k episodes
    k = cfg["episodes_per_replacement"]
    schedule = common.generation_schedule(n, seed, 1)
    base = cfg["formation_episodes"]
    for gi, slot in enumerate(schedule):
        due = base + (gi + 1) * k
        if gi < done["replacements"]:
            continue
        while ep < due:
            ep += 1
            a, b = sample_pair(ep)
            _play_episode(ep, agents, a, b, cfg, backend, journal)
        a2 = E2Agent(next_id)
        a2.born_at = ep
        agents[slot] = a2
        journal.append({"type": "replacement", "ep": ep, "slot": slot,
                        "agent_id": next_id})
        next_id += 1
    end = base + n * k + cfg["settle_episodes"]
    while ep < end:
        ep += 1
        a, b = sample_pair(ep)
        _play_episode(ep, agents, a, b, cfg, backend, journal)
    return {"done": True, "episodes": ep}
