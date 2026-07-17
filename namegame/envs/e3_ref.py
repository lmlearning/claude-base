"""E3 — Open-lexicon reference: description conventions (and coinage)
as a side product of an identification task.

Task: an informer sees the target item — a bundle of 3 nonsense traits —
and writes a note (<= 8 words); a chooser sees three candidate bundles in
random order plus the note and must pick the target.  Reward on accuracy
only.  There is NO label pool: items are unnamed.

The 6 items share traits (every trait belongs to >= 2 items), so several
minimal discriminating descriptions exist per item.  Convention
variables (mechanical): per item, WHICH trait subset the population
settles on; coined labels (tokens outside trait vocab, dictionary, and
template words); note length contraction.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

from ..expb.backend import Backend
from ..expb.tokens import generate_pool, is_real_word
from . import common
from .common import Journal, lint_env, rng_for

H = 6
N_ITEMS = 6
N_TRAITS = 8
NOTE_WORDS = 8

FRAMINGS = {
    "picker": (
        "You take part in a points activity over many turns, sometimes as "
        "the informer and sometimes as the chooser. Each turn has a target "
        "object; every object is described by 3 features. The informer "
        "sees the target's features and writes a note of at most "
        "{note_words} words. The chooser sees three candidate objects "
        "(their feature lists) and the note, and must pick the target. A "
        "right pick earns each of you {reward} points; a wrong one costs "
        "each of you {penalty} points. Your aim is the highest point "
        "total. You can see your recent turns."
    ),
    "warehouse": (
        "You and other participants earn points retrieving objects. Each "
        "object has 3 listed properties. On a retrieval, one person (the "
        "informer) is shown which object is wanted and writes a short "
        "request of no more than {note_words} words; another person (the "
        "chooser) sees three objects with their property lists and the "
        "request, and hands over one object. If it is the wanted object "
        "you each gain {reward} points, otherwise you each lose {penalty} "
        "points. Collect as many points as possible. Your recent "
        "retrievals are shown to you."
    ),
}

SEND_INSTR = ("The target object's features: {traits}.\n\nWrite your note "
              "for the chooser (at most {note_words} words). Reply with "
              "only the note.")
RECV_INSTR = ("Candidate objects:\n{candidates}\n\nNote from the informer: "
              "\"{note}\"\n\nWhich candidate is the target? Reply with "
              "only the number 1, 2 or 3.")
PRIOR_PROBE = ("An object has these features: {traits}. Write a note of at "
               "most {note_words} words so another person can identify "
               "this object among others. Reply with only the note.")
HIST_SEND = ("- as informer you wrote \"{note}\" ; the chooser was "
             "{outcome} ({points} points)")
HIST_RECV = ("- as chooser you got \"{note}\" and picked candidate "
             "{pick}; you were {outcome} ({points} points)")

lint_env(list(FRAMINGS.values()) + [SEND_INSTR, RECV_INSTR, PRIOR_PROBE,
                                    HIST_SEND, HIST_RECV])

TEMPLATE_WORDS = set(re.findall(r"[a-z]+", " ".join(
    list(FRAMINGS.values()) + [SEND_INSTR, RECV_INSTR, HIST_SEND,
                               HIST_RECV]).lower()))


def make_items(rng: np.random.Generator) -> tuple[list[str], list[list[int]]]:
    """6 items over 8 traits, 3 traits each, every trait in >= 2 items,
    all items distinct."""
    traits = generate_pool(N_TRAITS, rng)
    while True:
        items = []
        for _ in range(N_ITEMS):
            items.append(sorted(int(x) for x in
                                rng.choice(N_TRAITS, size=3, replace=False)))
        flat = [t for it in items for t in it]
        if (len({tuple(i) for i in items}) == N_ITEMS
                and all(flat.count(t) >= 2 for t in range(N_TRAITS))):
            return traits, items


def extract_traitset(note: str, traits: list[str]) -> list[int]:
    low = note.lower()
    return sorted(i for i, t in enumerate(traits)
                  if re.search(r"\b" + re.escape(t.lower()) + r"\b", low))


def extract_coinages(note: str, traits: list[str]) -> list[str]:
    tl = {t.lower() for t in traits}
    out = []
    for w in re.findall(r"[a-zA-Z]{3,}", note.lower()):
        if w in tl or w in TEMPLATE_WORDS or is_real_word(w):
            continue
        out.append(w)
    return out


def mock_reply(context: dict, rng: np.random.Generator) -> str:
    kind = context["kind"]
    if kind == "comprehension":
        return context["true_answer"]
    if kind in ("send", "prior"):
        traits = context["traits"]
        item = context["item"]              # trait indices of target
        seen = context.get("seen_sets", [])  # trait-sets used for this item
        if kind != "prior" and seen and rng.random() > 0.2:
            counts = {}
            for s in seen:
                counts[tuple(s)] = counts.get(tuple(s), 0) + 1
            best = max(counts.values())
            cands = [s for s, c in counts.items() if c == best]
            subset = list(cands[int(rng.integers(len(cands)))])
        else:
            k = 2 if rng.random() < 0.7 else 3
            subset = sorted(int(x) for x in
                            rng.choice(item, size=min(k, len(item)),
                                       replace=False))
        return " ".join(traits[i] for i in subset)
    if kind == "recv":
        # pick the candidate consistent with the mentioned traits
        mentioned = set(context["mentioned"])
        cands = context["candidates"]       # list of trait-index lists
        scores = [len(mentioned & set(c)) for c in cands]
        best = max(scores) if scores else 0
        idx = [i for i, s in enumerate(scores) if s == best]
        return str(idx[int(rng.integers(len(idx)))] + 1)
    raise ValueError(kind)


class E3Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.memory: deque = deque(maxlen=H)
        self.born_at = 0
        # per-item observed trait-sets (own sends + notes received)
        self.item_sets: dict[int, deque] = {}

    def note_set(self, item_idx: int, ts: list[int]):
        d = self.item_sets.setdefault(item_idx, deque(maxlen=8))
        if ts:
            d.append(ts)


def _nw(cfg):
    return cfg.get("note_words", NOTE_WORDS)


def _system(cfg):
    return FRAMINGS[cfg["framing"]].format(note_words=_nw(cfg),
                                           reward=100, penalty=50)


def _hist(agent: E3Agent) -> str:
    if not agent.memory:
        return "You have no previous turns yet."
    lines = ["Your recent turns, oldest first:"]
    for e in agent.memory:
        if e["role"] == "send":
            lines.append(HIST_SEND.format(
                note=e["note"], outcome="right" if e["correct"] else "wrong",
                points="+100" if e["correct"] else "-50"))
        else:
            lines.append(HIST_RECV.format(
                note=e["note"], pick=e["pick"],
                outcome="right" if e["correct"] else "wrong",
                points="+100" if e["correct"] else "-50"))
    return "\n".join(lines)


def _one_interaction(t, agents, cfg, backend, journal, phase, si, ri):
    seed = cfg["seed"]
    traits, items = cfg["traits"], cfg["items"]
    r = rng_for(seed, 5, t)
    target = int(r.integers(N_ITEMS))
    others = [i for i in range(N_ITEMS) if i != target]
    if cfg.get("hard_distractors"):
        # the two candidates with the largest trait overlap with the target
        # (ties broken at random) — discrimination stays genuinely hard
        ranked = sorted(others,
                        key=lambda o: (-len(set(items[target])
                                            & set(items[o])), r.random()))
        d1, d2 = ranked[0], ranked[1]
    else:
        d1, d2 = (int(x) for x in r.choice(others, size=2, replace=False))
    lineup = [target, d1, d2]
    order = [int(x) for x in r.permutation(3)]
    shown = [lineup[k] for k in order]      # item index per slot 1..3
    S, R = agents[si], agents[ri]

    nw = _nw(cfg)
    user_s = "\n\n".join([_hist(S), SEND_INSTR.format(
        traits=", ".join(traits[i] for i in items[target]),
        note_words=nw)])
    raw = backend.complete(_system(cfg), user_s, max(24, nw * 3),
                           {"kind": "send", "traits": traits,
                            "item": items[target], "note_words": nw,
                            "seen_sets": list(S.item_sets.get(target, []))})
    note, clipped = common.word_count_clip(raw, nw)
    mentioned = extract_traitset(note, traits)

    cand_text = "\n".join(
        f"{k+1}. features: " + ", ".join(traits[i] for i in items[it])
        for k, it in enumerate(shown))
    user_r = "\n\n".join([_hist(R), RECV_INSTR.format(
        candidates=cand_text, note=note)])
    reply = backend.complete(_system(cfg), user_r,
                             cfg.get("recv_tokens", 8),
                             {"kind": "recv", "mentioned": mentioned,
                              "candidates": [items[it] for it in shown]})
    m = re.search(r"[123]", reply)
    malformed = m is None
    pick = int(m.group()) if m else int(rng_for(seed, 7, t).integers(1, 4))
    correct = shown[pick - 1] == target

    rec = {"type": "interaction", "t": t, "phase": phase,
           "sender_slot": si, "receiver_slot": ri,
           "sender_id": S.agent_id, "receiver_id": R.agent_id,
           "target": target, "shown": shown, "note": note,
           "clipped": clipped, "mentioned": mentioned,
           "coinages": extract_coinages(note, traits),
           "pick": pick, "correct": correct, "malformed": int(malformed),
           "note_words": len(note.split())}
    _apply(agents, rec)
    journal.append(rec)
    return rec


def _apply(agents, rec):
    S, R = agents[rec["sender_slot"]], agents[rec["receiver_slot"]]
    S.memory.append({"role": "send", "note": rec["note"],
                     "correct": rec["correct"]})
    R.memory.append({"role": "recv", "note": rec["note"],
                     "pick": rec["pick"], "correct": rec["correct"]})
    S.note_set(rec["target"], rec["mentioned"])
    if rec["correct"]:
        R.note_set(rec["target"], rec["mentioned"])
    S.__dict__.setdefault("sends", []).append(
        {"t": rec["t"], "target": rec["target"],
         "mentioned": rec["mentioned"], "phase": rec["phase"]})


def measure_priors(cfg, backend, journal, reps=2) -> dict:
    traits, items = cfg["traits"], cfg["items"]
    out = []
    for it in range(N_ITEMS):
        for rep in range(reps):
            raw = backend.complete(
                _system(cfg),
                PRIOR_PROBE.format(traits=", ".join(traits[i]
                                                    for i in items[it]),
                                   note_words=_nw(cfg)),
                max(24, _nw(cfg) * 3),
                {"kind": "prior", "traits": traits, "item": items[it],
                 "note_words": _nw(cfg)})
            note, _ = common.word_count_clip(raw, _nw(cfg))
            out.append({"item": it, "set": extract_traitset(note, traits)})
    rec = {"type": "priors", "probes": out}
    journal.append(rec)
    return rec


COMPREHENSION = [
    ("If the chooser picks the target, how many points do you each earn? "
     "Reply with just the number.", "100"),
    ("How many candidate objects does the chooser see? Reply with just "
     "the number.", "3"),
    ("How many features does each object have? Reply with just the "
     "number.", "3"),
]


def run_population(cfg: dict, backend: Backend, run_dir: str) -> dict:
    journal = Journal(run_dir)
    seed = cfg["seed"]
    n = cfg["n_agents"]
    agents = [E3Agent(i) for i in range(n)]
    next_id = n
    done = {"config": False, "comprehension": None, "t": 0,
            "replacements": 0}
    for rec in journal.replay():
        if rec["type"] == "config":
            done["config"] = True
            cfg = {**cfg, "traits": rec["traits"], "items": rec["items"]}
        elif rec["type"] == "comprehension":
            done["comprehension"] = rec["passed"]
        elif rec["type"] == "interaction":
            _apply(agents, rec)
            done["t"] = rec["t"]
        elif rec["type"] == "replacement":
            a = E3Agent(rec["agent_id"])
            a.born_at = rec["t"]
            agents[rec["slot"]] = a
            next_id = max(next_id, rec["agent_id"] + 1)
            done["replacements"] += 1

    if not done["config"]:
        # shared_items: one fixed item/trait set for EVERY population in the
        # cell, so cross-population diversity of settled descriptions is
        # directly comparable (the per-population prior probe still measures
        # any token-level bias on the shared traits).
        item_seed = 777001 if cfg.get("shared_items") else seed
        traits, items = make_items(rng_for(item_seed, 1))
        cfg = {**cfg, "traits": traits, "items": items}
        journal.append({"type": "config", **cfg, "env": "e3"})
    if done["comprehension"] is None:
        if not common.run_comprehension(backend, _system(cfg),
                                        COMPREHENSION, journal):
            return {"aborted": "comprehension_gate_failed"}
    elif done["comprehension"] is False:
        return {"aborted": "comprehension_gate_failed"}
    if done["t"] == 0:
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
        _one_interaction(t, agents, cfg, backend, journal, "formation", i, j)

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
            _one_interaction(t, agents, cfg, backend, journal, "turnover",
                             i, j)
        a = E3Agent(next_id)
        a.born_at = t
        agents[slot] = a
        journal.append({"type": "replacement", "t": t, "slot": slot,
                        "agent_id": next_id})
        next_id += 1
    end = base + n * k + cfg["settle_interactions"]
    while t < end:
        t += 1
        i, j = pair(t)
        _one_interaction(t, agents, cfg, backend, journal, "settle", i, j)
    return {"done": True, "t": t}
