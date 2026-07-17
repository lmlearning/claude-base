"""E1 — Relay-QA: wire-format conventions as a side product.

Task: an informer receives a record of 5 nonsense-labelled entries in a
RANDOMIZED presentation order and writes a short free-text note; a
responder, shown only the note, must answer a query about one label.
Both are paid on the responder's accuracy alone.  Nothing rewards or
mentions how the note is written.

Convention variables (all mechanically extracted):
  * field-order permutation of the 5 labels in the note (120
    reward-equivalent alternatives; input-order randomization deletes
    "copy the input" as a non-social attractor — a stable population
    ordering is endogenous);
  * label-naming (are labels written out or positional);
  * note length (words).

The token budget (max words per note) is the functional-pressure lever:
cells run at tight (12 words) and loose (25 words) budgets.

Phases: formation -> one generation of turnover -> committed minority
(scripted informers using a fixed alternative permutation).  Plus a
stranger-pool control population (tracked informers, fresh memoryless
responders) — the individual-habit baseline.
"""

from __future__ import annotations

import re
from collections import deque

import numpy as np

from ..expb.backend import Backend
from ..expb.tokens import generate_pool
from . import common
from .common import Journal, lint_env, rng_for

H = 6  # remembered events per agent

FRAMINGS = {
    "activity": (
        "You are taking part in a points activity with many turns. On some "
        "turns you act as the informer and on others as the responder. On "
        "an informer turn you are shown a record of {W} labels, each with "
        "its current entry, and you write a short note (at most {budget} "
        "words) that is passed to a responder. The responder is then asked "
        "for the entry of one of the labels, seeing only your note. If the "
        "responder answers with the right entry, you both earn {reward} "
        "points; otherwise you both lose {penalty} points. Your aim is to "
        "finish with as many points as possible. You can see your last "
        "few turns."
    ),
    "office": (
        "You work a sequence of short tasks for points. Sometimes your job "
        "is passing information: you receive a sheet showing {W} labels "
        "with their current entries and must write a brief note (no more "
        "than {budget} words) for a colleague. The colleague, who sees "
        "only your note, is asked to state the entry belonging to one "
        "label. A right answer earns each of you {reward} points; a wrong "
        "one costs each of you {penalty} points. Earn as many points as "
        "you can. A record of your recent tasks is available to you."
    ),
}

SEND_INSTR = ("Your record for this turn:\n{record}\n\nWrite your note to "
              "the responder now (at most {budget} words). Reply with only "
              "the note.")
RECV_INSTR = ("Note from the informer: \"{note}\"\n\nWhat is the entry for "
              "the label \"{field}\"? Reply with only the entry.")
PRIOR_PROBE = ("You are shown a record of labels with entries:\n{record}\n\n"
               "Write a short note (at most {budget} words) passing this "
               "information to another person. Reply with only the note.")

HIST_SEND = ("- as informer you wrote: \"{note}\" ; the responder was asked "
             "\"{field}\" and was {outcome} ({points} points)")
HIST_RECV = ("- as responder you got the note: \"{note}\" ; asked "
             "\"{field}\" you said \"{answer}\" and were {outcome} "
             "({points} points)")

lint_env(list(FRAMINGS.values()) + [SEND_INSTR, RECV_INSTR, PRIOR_PROBE,
                                    HIST_SEND, HIST_RECV])


# ---------------------------------------------------------------------------
# extraction (shared with analysis)
# ---------------------------------------------------------------------------

def extract_perm(note: str, fields: list[str], values: list[str]) -> list[int] | None:
    """Permutation of field indices by first occurrence of each field's
    VALUE token in the note; None unless all 5 values appear exactly."""
    low = note.lower()
    pos = []
    for v in values:
        m = re.search(r"\b" + re.escape(v.lower()) + r"\b", low)
        if not m:
            return None
        pos.append(m.start())
    return [int(i) for i in np.argsort(pos)]


def count_named(note: str, fields: list[str]) -> int:
    low = note.lower()
    return sum(1 for f in fields
               if re.search(r"\b" + re.escape(f.lower()) + r"\b", low))


# ---------------------------------------------------------------------------
# mock policy (substrate control)
# ---------------------------------------------------------------------------

def mock_reply(context: dict, rng: np.random.Generator) -> str:
    kind = context["kind"]
    if kind == "comprehension":
        return context["true_answer"]
    if kind == "send" or kind == "prior":
        fields = context["fields"]
        values = context["values"]          # aligned with fields
        presented = context["presented"]    # field indices in shown order
        seen = context.get("seen_perms", [])  # perms from memory (own+observed)
        if kind != "prior" and seen and rng.random() > 0.15:
            counts = {}
            for p in seen:
                counts[tuple(p)] = counts.get(tuple(p), 0) + 1
            best = max(counts.values())
            cands = [p for p, c in counts.items() if c == best]
            perm = list(cands[int(rng.integers(len(cands)))])
        elif rng.random() < 0.5:
            perm = list(presented)          # copy input order
        else:
            perm = [int(x) for x in rng.permutation(len(fields))]
        return " ".join(f"{fields[i]} {values[i]}" for i in perm)
    if kind == "recv":
        # correct with high prob if the note's perm is familiar
        note, fields, values = context["note"], context["fields"], context["values"]
        perm = extract_perm(note, fields, values)
        familiar = perm is not None and perm in context.get("seen_perms", [])
        p_ok = 0.97 if familiar else 0.72
        qi = context["question_index"]
        if rng.random() < p_ok and perm is not None:
            return values[qi]
        return values[int(rng.integers(len(values)))]
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class E1Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.memory: deque = deque(maxlen=H)   # event dicts
        self.born_at = 0
        self.sends: list[dict] = []            # (t, perm) for analysis

    def seen_perms(self) -> list[list[int]]:
        return [e["perm"] for e in self.memory if e.get("perm") is not None]


def _system(cfg: dict) -> str:
    return FRAMINGS[cfg["framing"]].format(W=5, budget=cfg["budget_words"],
                                           reward=100, penalty=50)


def _hist(agent: E1Agent) -> str:
    if not agent.memory:
        return "You have no previous turns yet."
    lines = ["Your recent turns, oldest first:"]
    for e in agent.memory:
        if e["role"] == "send":
            lines.append(HIST_SEND.format(
                note=e["note"], field=e["field"],
                outcome="right" if e["correct"] else "wrong",
                points=("+100" if e["correct"] else "-50")))
        else:
            lines.append(HIST_RECV.format(
                note=e["note"], field=e["field"], answer=e["answer"],
                outcome="right" if e["correct"] else "wrong",
                points=("+100" if e["correct"] else "-50")))
    return "\n".join(lines)


def _record_text(fields, values, presented) -> str:
    return "\n".join(f"{fields[i]}: {values[i]}" for i in presented)


def _one_interaction(t: int, phase: str, agents, cfg, backend, journal,
                     sender_slot: int, receiver_slot: int,
                     committed_perm=None, committed_slots=frozenset()):
    seed = cfg["seed"]
    fields, vocab = cfg["fields"], cfg["value_vocab"]
    r = rng_for(seed, 5, t)
    values = [vocab[i] for i in r.choice(len(vocab), size=5, replace=False)]
    presented = [int(x) for x in r.permutation(5)]
    qi = int(r.integers(5))
    S, R = agents[sender_slot], agents[receiver_slot]

    if sender_slot in committed_slots:
        note = " ".join(f"{fields[i]} {values[i]}" for i in committed_perm)
        clipped = False
    else:
        user = "\n\n".join([_hist(S), SEND_INSTR.format(
            record=_record_text(fields, values, presented),
            budget=cfg["budget_words"])])
        raw = backend.complete(_system(cfg), user, cfg["budget_words"] * 3,
                               {"kind": "send", "fields": fields,
                                "values": values, "presented": presented,
                                "seen_perms": S.seen_perms()})
        note, clipped = common.word_count_clip(raw, cfg["budget_words"])

    user_r = "\n\n".join([_hist(R), RECV_INSTR.format(note=note,
                                                      field=fields[qi])])
    ans_raw = backend.complete(_system(cfg), user_r, 16,
                               {"kind": "recv", "note": note,
                                "fields": fields, "values": values,
                                "question_index": qi,
                                "seen_perms": R.seen_perms()})
    ans = (re.findall(r"[A-Za-z0-9]+", ans_raw) or [""])[0]
    correct = ans.lower() == values[qi].lower()
    perm = extract_perm(note, fields, values)

    rec = {"type": "interaction", "t": t, "phase": phase,
           "sender_slot": sender_slot, "receiver_slot": receiver_slot,
           "sender_id": S.agent_id, "receiver_id": R.agent_id,
           "values": values, "presented": presented, "q": qi,
           "note": note, "clipped": clipped, "answer": ans,
           "correct": correct, "perm": perm,
           "named": count_named(note, fields),
           "committed": sender_slot in committed_slots}
    _apply(agents, rec, fields)
    journal.append(rec)
    return rec


def _apply(agents, rec, fields):
    S = agents[rec["sender_slot"]]
    R = agents[rec["receiver_slot"]]
    S.memory.append({"role": "send", "note": rec["note"],
                     "field": fields[rec["q"]], "correct": rec["correct"],
                     "perm": rec["perm"]})
    R.memory.append({"role": "recv", "note": rec["note"],
                     "field": fields[rec["q"]], "answer": rec["answer"],
                     "correct": rec["correct"], "perm": rec["perm"]})
    S.sends.append({"t": rec["t"], "perm": rec["perm"],
                    "phase": rec["phase"]})


def measure_priors(cfg, backend, journal, n=12) -> dict:
    fields, vocab = cfg["fields"], cfg["value_vocab"]
    perms = []
    copy_input = 0
    for s in range(n):
        r = rng_for(cfg["seed"], 11, s)
        values = [vocab[i] for i in r.choice(len(vocab), size=5,
                                             replace=False)]
        presented = [int(x) for x in r.permutation(5)]
        raw = backend.complete(
            _system(cfg),
            PRIOR_PROBE.format(record=_record_text(fields, values, presented),
                               budget=cfg["budget_words"]),
            cfg["budget_words"] * 3,
            {"kind": "prior", "fields": fields, "values": values,
             "presented": presented})
        note, _ = common.word_count_clip(raw, cfg["budget_words"])
        p = extract_perm(note, fields, values)
        perms.append(p)
        if p == presented:
            copy_input += 1
    rec = {"type": "priors", "perms": perms, "n": n,
           "copy_input_share": copy_input / n}
    journal.append(rec)
    return rec


COMPREHENSION = [
    ("If the responder gives the right entry, how many points do you each "
     "earn? Reply with just the number.", "100"),
    ("If the responder gives a wrong entry, how many points do you each "
     "lose? Reply with just the number.", "50"),
    ("How many labels are in a record? Reply with just the number.", "5"),
]


def run_population(cfg: dict, backend: Backend, run_dir: str) -> dict:
    journal = Journal(run_dir)
    seed = cfg["seed"]
    n = cfg["n_agents"]

    done = {"config": False, "comprehension": None, "n_inter": 0,
            "replacements": []}
    agents = [E1Agent(i) for i in range(n)]
    next_id = n
    for rec in journal.replay():
        if rec["type"] == "config":
            done["config"] = True
            cfg = {**cfg, "fields": rec["fields"],
                   "value_vocab": rec["value_vocab"]}
        elif rec["type"] == "comprehension":
            done["comprehension"] = rec["passed"]
        elif rec["type"] == "interaction":
            _apply(agents, rec, cfg["fields"])
            done["n_inter"] = rec["t"]
        elif rec["type"] == "replacement":
            a = E1Agent(rec["agent_id"])
            a.born_at = rec["t"]
            agents[rec["slot"]] = a
            next_id = max(next_id, rec["agent_id"] + 1)
            done["replacements"].append(rec)
        elif rec["type"] == "priors":
            done["priors"] = rec

    if not done["config"]:
        r = rng_for(seed, 1)
        pool = generate_pool(45, r)
        cfg = {**cfg, "fields": pool[:5], "value_vocab": pool[5:]}
        journal.append({"type": "config", **{k: v for k, v in cfg.items()
                                             if k != "phase_plan"},
                        "env": "e1"})
    if done["comprehension"] is None:
        if not common.run_comprehension(backend, _system(cfg),
                                        COMPREHENSION, journal):
            return {"aborted": "comprehension_gate_failed"}
    elif done["comprehension"] is False:
        return {"aborted": "comprehension_gate_failed"}
    if "priors" not in done and not cfg.get("stranger_pool"):
        measure_priors(cfg, backend, journal)

    t = done["n_inter"]

    def pair(t):
        r = rng_for(seed, 3, t)
        i = int(r.integers(n))
        j = int(r.integers(n - 1))
        if j >= i:
            j += 1
        return i, j

    if cfg.get("stranger_pool"):
        # tracked informers slots 0..n-1; responder is ALWAYS a fresh blank
        total = cfg["formation_interactions"]
        blank_slot = n - 1  # reserved blank responder slot
        while t < total:
            t += 1
            s = int(rng_for(seed, 3, t).integers(n - 1))
            agents[blank_slot] = E1Agent(9000 + t)   # memoryless every round
            _one_interaction(t, "stranger", agents, cfg, backend, journal,
                             s, blank_slot)
        return {"done": True, "mode": "stranger", "t": t}

    # ---- formation ----
    while t < cfg["formation_interactions"]:
        t += 1
        i, j = pair(t)
        _one_interaction(t, "formation", agents, cfg, backend, journal, i, j)

    # ---- turnover: one generation ----
    k = cfg["interactions_per_replacement"]
    schedule = common.generation_schedule(n, seed, 1)
    n_done = len(done["replacements"])
    base_t = cfg["formation_interactions"]
    for gi, slot in enumerate(schedule):
        due_t = base_t + (gi + 1) * k
        if gi < n_done:
            continue
        while t < due_t:
            t += 1
            i, j = pair(t)
            _one_interaction(t, "turnover", agents, cfg, backend, journal,
                             i, j)
        a = E1Agent(next_id)
        a.born_at = t
        agents[slot] = a
        journal.append({"type": "replacement", "t": t, "slot": slot,
                        "agent_id": next_id})
        next_id += 1
    settle_end = base_t + n * k + cfg["settle_interactions"]
    while t < settle_end:
        t += 1
        i, j = pair(t)
        _one_interaction(t, "settle", agents, cfg, backend, journal, i, j)

    # ---- committed minority (scripted informers, fixed alternative perm) --
    if cfg.get("minority_fraction"):
        r = rng_for(seed, 13)
        n_comm = max(1, round(cfg["minority_fraction"] * n))
        committed_slots = frozenset(int(x) for x in
                                    r.choice(n, size=n_comm, replace=False))
        alt_perm = [int(x) for x in r.permutation(5)]
        journal.append({"type": "minority_start", "t": t,
                        "slots": sorted(committed_slots),
                        "alt_perm": alt_perm})
        end = settle_end + cfg["minority_interactions"]
        while t < end:
            t += 1
            i, j = pair(t)
            _one_interaction(t, "minority", agents, cfg, backend, journal,
                             i, j, committed_perm=alt_perm,
                             committed_slots=committed_slots)
    return {"done": True, "t": t}
