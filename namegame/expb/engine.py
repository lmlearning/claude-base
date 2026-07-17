"""Experiment B engine: LLM naming-game runs with journaling and resume.

Design notes
------------
* Strictly sequential pairwise interactions within a run (the published
  dynamics).  Throughput for live runs comes from running independent runs
  concurrently, which cannot alter within-run dynamics.
* Every interaction is appended to ``journal.jsonl`` before the next one
  begins.  A killed run is resumed by replaying the journal through the
  same state-update code used live, so nothing but wall-clock is lost.
* All randomness (pairing, fallback choices, turnover order) comes from
  counter-based generators seeded as ``[seed, t]`` or ``[seed, tag]``, so a
  resumed run continues on exactly the trajectory it would have taken.
* The dialogue channel: after a FAILED interaction each participant may
  send one short message (neutral invitation, explicit decline option);
  it is delivered into the recipient's context at its next play, then
  cleared.  ``dialogue=False`` runs are the matched ablation.
"""

from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

from ..core.config import GameConfig
from ..core.engine import WindowTracker
from . import prompts
from .backend import Backend, parse_message, parse_name
from .tokens import generate_pool


class BAgent:
    def __init__(self, agent_id: int, H: int):
        self.agent_id = agent_id
        self.memory = deque(maxlen=H)   # (own_token, partner_token, payoff)
        self.inbox: str | None = None   # pending message text
        self.plays: list[str] = []
        self.born_at = 0
        self.n_messages_seen = 0

    def clone_blank(self, agent_id: int, born_at: int) -> "BAgent":
        a = BAgent(agent_id, self.memory.maxlen)
        a.born_at = born_at
        return a


class BRun:
    """One Experiment B run (any phase). Drives agents via a Backend."""

    def __init__(self, run_dir: str, config: dict, backend: Backend):
        self.run_dir = run_dir
        self.config = config
        self.backend = backend
        os.makedirs(run_dir, exist_ok=True)
        self.journal_path = os.path.join(run_dir, "journal.jsonl")

        self.seed = config["seed"]
        self.game = GameConfig(n_agents=config["n_agents"],
                               n_names=config["n_names"],
                               memory_size=config["memory_size"])
        rng0 = np.random.default_rng([self.seed, 0])
        self.pool = config.get("pool") or generate_pool(self.game.n_names, rng0)
        self.framing = config["framing"]
        self.dialogue = config["dialogue"]
        self.system_prompt = prompts.build_system_prompt(
            self.framing, self.pool, self.game.n_names,
            self.game.memory_size, self.game.reward, self.game.penalty)

        self.agents = [BAgent(i, self.game.memory_size)
                       for i in range(self.game.n_agents)]
        self.tracker = WindowTracker(self.game)
        self.t = 0
        self.next_id = self.game.n_agents
        self.malformed_choices = 0   # fallback-to-random events
        self.retries_total = 0       # unparseable first replies (retried)
        self.total_choices = 0
        self.head_records: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------
    def _append(self, record: dict) -> None:
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def resume(self) -> None:
        """Replay journal to rebuild state. Safe on a fresh run."""
        if not os.path.exists(self.journal_path):
            self._append({"type": "config", **{k: v for k, v in
                                               self.config.items()},
                          "pool": self.pool,
                          "system_prompt": self.system_prompt})
            return
        with open(self.journal_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["type"] == "interaction":
                    self._apply_interaction(rec)
                elif rec["type"] == "replacement":
                    self._apply_replacement(rec)
                elif rec["type"] in ("config", "comprehension", "priors",
                                     "phase_marker"):
                    self.head_records[rec["type"]] = rec

    # ------------------------------------------------------------------
    # State updates (shared by live path and replay)
    # ------------------------------------------------------------------
    def _apply_interaction(self, rec: dict) -> None:
        if rec.get("solitary"):
            agent = self.agents[0]
            payoff = self.game.reward if rec["success"] else self.game.penalty
            agent.memory.append((rec["name_i"], rec["name_j"], payoff))
            agent.plays.append(rec["name_i"])
            self.t = rec["t"]
            self.total_choices += 1 + rec.get("retries", 0)
            self.retries_total += rec.get("retries", 0)
            self.malformed_choices += rec["malformed_i"]
            return
        i, j = rec["i"], rec["j"]
        a, b = self.agents[i], self.agents[j]
        na, nb, success = rec["name_i"], rec["name_j"], rec["success"]
        payoff = self.game.reward if success else self.game.penalty
        # having played, both agents' pending inbox messages are consumed
        for ag in (a, b):
            if ag.inbox is not None:
                ag.n_messages_seen += 1
                ag.inbox = None
        a.memory.append((na, nb, payoff))
        b.memory.append((nb, na, payoff))
        a.plays.append(na)
        b.plays.append(nb)
        self.tracker.push(na, nb, success)
        self.t = rec["t"]
        self.malformed_choices += rec["malformed_i"] + rec["malformed_j"]
        self.retries_total += rec.get("retries", 0)
        self.total_choices += 2 + rec.get("retries", 0)
        if rec.get("msg_i") is not None:
            self.agents[j].inbox = rec["msg_i"]
        if rec.get("msg_j") is not None:
            self.agents[i].inbox = rec["msg_j"]

    def _apply_replacement(self, rec: dict) -> None:
        slot = rec["slot"]
        newcomer = BAgent(rec["agent_id"], self.game.memory_size)
        newcomer.born_at = rec["t"]
        self.agents[slot] = newcomer
        self.next_id = max(self.next_id, rec["agent_id"] + 1)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------
    def _history_text(self, agent: BAgent) -> str:
        if not agent.memory:
            return prompts.NO_HISTORY
        lines = [prompts.HISTORY_HEADER]
        for own, partner, payoff in agent.memory:
            outcome = "gained" if payoff > 0 else "lost"
            lines.append(prompts.HISTORY_LINE.format(
                own=own, partner=partner, outcome=outcome,
                points=int(abs(payoff))))
        return "\n".join(lines)

    def _choose(self, agent: BAgent, t: int) -> tuple[str, bool, int]:
        """Returns (token, malformed_fallback_used, retries)."""
        parts = [self._history_text(agent)]
        if agent.inbox is not None:
            parts.append(prompts.INCOMING_MESSAGE.format(text=agent.inbox))
        parts.append(prompts.CHOICE_INSTRUCTION)
        user = "\n\n".join(parts)
        ctx = {"kind": "choice", "memory": list(agent.memory),
               "pool": self.pool}
        reply = self.backend.complete(self.system_prompt, user, 60, ctx)
        name = parse_name(reply, self.pool)
        retries = 0
        if name is None:
            retries = 1
            reply2 = self.backend.complete(
                self.system_prompt, user + "\n\n" + prompts.CHOICE_RETRY,
                60, ctx)
            name = parse_name(reply2, self.pool)
        if name is None:
            rng = np.random.default_rng([self.seed, 7, t, agent.agent_id])
            name = self.pool[int(rng.integers(len(self.pool)))]
            return name, True, retries
        return name, False, retries

    def _maybe_message(self, agent: BAgent, own: str, partner: str,
                       t: int) -> str | None:
        user = "\n\n".join([self._history_text(agent),
                            prompts.MESSAGE_INVITATION])
        ctx = {"kind": "message", "memory": list(agent.memory),
               "pool": self.pool}
        reply = self.backend.complete(self.system_prompt, user, 64, ctx)
        msg, _ok = parse_message(reply)
        return msg

    # ------------------------------------------------------------------
    # Pre-run gates and probes
    # ------------------------------------------------------------------
    def comprehension_check(self, reps: int = 3) -> dict:
        if "comprehension" in self.head_records:
            return self.head_records["comprehension"]
        cfg = {"reward": self.game.reward,
               "penalty_abs": abs(self.game.penalty),
               "W": self.game.n_names}
        true_answers = [str(int(self.game.reward)),
                        str(int(abs(self.game.penalty))),
                        str(self.game.n_names), "YES"]
        results = []
        for rep in range(reps):
            for qi, (q, grade) in enumerate(prompts.COMPREHENSION_QUESTIONS):
                ctx = {"kind": "comprehension", "true_answer": true_answers[qi]}
                ans = self.backend.complete(self.system_prompt, q, 60, ctx)
                results.append({"rep": rep, "q": qi, "answer": ans,
                                "pass": bool(grade(ans, cfg))})
        rec = {"type": "comprehension", "results": results,
               "passed": all(r["pass"] for r in results)}
        self._append(rec)
        self.head_records["comprehension"] = rec
        return rec

    def measure_priors(self, n_samples: int = 20) -> dict:
        if "priors" in self.head_records:
            return self.head_records["priors"]
        counts = {p: 0 for p in self.pool}
        malformed = 0
        for s in range(n_samples):
            rng = np.random.default_rng([self.seed, 11, s])
            order = [self.pool[k] for k in rng.permutation(len(self.pool))]
            probe = prompts.PRIOR_PROBE.format(pool=", ".join(order))
            reply = self.backend.complete("", probe, 60,
                                          {"kind": "prior", "pool": order})
            name = parse_name(reply, self.pool)
            if name is None:
                malformed += 1
            else:
                counts[name] += 1
        rec = {"type": "priors", "counts": counts, "malformed": malformed,
               "n_samples": n_samples}
        self._append(rec)
        self.head_records["priors"] = rec
        return rec

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def _step(self, i: int, j: int) -> dict:
        t = self.t + 1
        a, b = self.agents[i], self.agents[j]
        na, mal_a, r_a = self._choose(a, t)
        nb, mal_b, r_b = self._choose(b, t)
        success = na == nb
        rec = {"type": "interaction", "t": t, "i": i, "j": j,
               "agent_i": a.agent_id, "agent_j": b.agent_id,
               "name_i": na, "name_j": nb, "success": success,
               "malformed_i": int(mal_a), "malformed_j": int(mal_b),
               "retries": r_a + r_b, "msg_i": None, "msg_j": None}
        if not success and self.dialogue:
            # both agents may message their partner (delivered at next play)
            rec["msg_i"] = self._maybe_message(a, na, nb, t)
            rec["msg_j"] = self._maybe_message(b, nb, na, t)
        self._apply_interaction(rec)
        self._append(rec)
        return rec

    def _random_pair(self, t: int) -> tuple[int, int]:
        rng = np.random.default_rng([self.seed, 3, t])
        n = len(self.agents)
        i = int(rng.integers(n))
        j = int(rng.integers(n - 1))
        if j >= i:
            j += 1
        return i, j

    def run_genesis(self, max_interactions: int,
                    post_consensus: int = 100) -> dict:
        """Run to consensus (+ settle). Returns summary."""
        consensus_at = None
        winner = None
        while self.t < max_interactions:
            i, j = self._random_pair(self.t + 1)
            self._step(i, j)
            w = self.tracker.consensus()
            if w is not None:
                consensus_at, winner = self.t, w
                break
        if winner is not None:
            for _ in range(post_consensus):
                i, j = self._random_pair(self.t + 1)
                self._step(i, j)
        return {"converged": winner is not None, "winner": winner,
                "consensus_time": consensus_at}

    def run_transmission(self, interactions_per_replacement: int,
                         generations: int = 1,
                         settle: int = 100) -> dict:
        """Generational turnover from current (converged) state."""
        original = self.tracker.consensus()
        replacements = []
        for gen in range(generations):
            rng = np.random.default_rng([self.seed, 5, gen])
            for slot in rng.permutation(self.game.n_agents):
                for _ in range(interactions_per_replacement):
                    i, j = self._random_pair(self.t + 1)
                    self._step(i, j)
                rec = {"type": "replacement", "t": self.t,
                       "slot": int(slot), "agent_id": self.next_id,
                       "generation": gen}
                self._apply_replacement(rec)
                self._append(rec)
                replacements.append(rec)
        for _ in range(settle):
            i, j = self._random_pair(self.t + 1)
            self._step(i, j)
        final = self.tracker.consensus()
        return {"original": original, "final": final,
                "survived": final == original and original is not None,
                "n_replacements": len(replacements)}

    def run_solitary(self, rounds: int) -> dict:
        """One persistent agent vs a fresh memoryless random partner each
        round.  The partner plays uniformly at random and never speaks;
        dialogue invitations go to the persistent agent only."""
        agent = self.agents[0]
        n_msgs = 0
        for r in range(rounds):
            t = self.t + 1
            rng = np.random.default_rng([self.seed, 9, t])
            partner_name = self.pool[int(rng.integers(len(self.pool)))]
            na, mal, retries = self._choose(agent, t)
            success = na == partner_name
            rec = {"type": "interaction", "t": t, "i": 0, "j": -1,
                   "agent_i": agent.agent_id, "agent_j": -1,
                   "name_i": na, "name_j": partner_name, "success": success,
                   "malformed_i": int(mal), "malformed_j": 0,
                   "retries": retries, "msg_i": None, "msg_j": None,
                   "solitary": True}
            if not success and self.dialogue:
                rec["msg_i"] = self._maybe_message(agent, na, partner_name, t)
                if rec["msg_i"] is not None:
                    n_msgs += 1
            # partner is discarded; only the persistent agent updates
            payoff = self.game.reward if success else self.game.penalty
            agent.memory.append((na, partner_name, payoff))
            agent.plays.append(na)
            self.t = t
            self.total_choices += 1 + retries
            self.retries_total += retries
            self.malformed_choices += int(mal)
            self._append(rec)
        return {"rounds": rounds, "messages_sent": n_msgs}

    def summary(self) -> dict:
        return {
            "t": self.t,
            "fallback_choices": self.malformed_choices,
            "unparseable_first_replies": self.retries_total,
            "total_choices": self.total_choices,
            # protocol metric: share of choice responses that were
            # unparseable (first replies that needed a retry, plus retries
            # that still failed and fell back to seeded-random)
            "malformed_rate": ((self.retries_total + self.malformed_choices) /
                               max(1, self.total_choices)),
            "consensus": self.tracker.consensus(),
            "dominant": self.tracker.dominant(),
            "success_rate_window": self.tracker.success_rate(),
        }
