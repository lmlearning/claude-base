"""LLM backends for Experiment B: live Anthropic API and free mock.

Both implement ``complete(system, user, max_tokens, context)``.  The live
backend ignores ``context``; the mock backend uses it to produce a
behaviourally sensible reply (minimal-agent policy behind the LLM
interface) while the full prompt-building and reply-parsing pipeline is
still exercised.

The live backend enforces a HARD SPEND CAP in code: every response's usage
is priced and accumulated; exceeding the cap raises SpendCapExceeded, which
the engine turns into a clean checkpointed stop.
"""

from __future__ import annotations

import json
import os
import re
import threading

import numpy as np

# USD per token (input, output).  claude-haiku-4-5: $1 / $5 per MTok.
PRICES = {
    "claude-haiku-4-5": (1.0e-6, 5.0e-6),
    "claude-sonnet-4-5": (3.0e-6, 15.0e-6),
    # OpenRouter route to the same model, same list pricing (verified
    # against /api/v1/models at run time)
    "anthropic/claude-haiku-4.5": (1.0e-6, 5.0e-6),
    "anthropic/claude-sonnet-4.5": (3.0e-6, 15.0e-6),
    # second model family (non-Anthropic; prices verified against
    # OpenRouter /api/v1/models on 2026-07-18); gpt-5-mini selected by
    # the operator ("a gpt 5 old version": original GPT-5-generation
    # mini tier), gemini-2.5-flash-lite is the pre-declared fallback
    "openai/gpt-5-mini": (0.25e-6, 2.0e-6),
    "google/gemini-2.5-flash-lite": (0.1e-6, 0.4e-6),
}


class SpendCapExceeded(RuntimeError):
    pass


class CostTracker:
    """Thread-safe accumulator with persistence and a hard cap."""

    def __init__(self, cap_usd: float, state_path: str):
        self.cap_usd = cap_usd
        self.state_path = state_path
        self.lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.n_calls = 0
        if os.path.exists(state_path):
            with open(state_path) as f:
                s = json.load(f)
            self.input_tokens = s["input_tokens"]
            self.output_tokens = s["output_tokens"]
            self.cost_usd = s["cost_usd"]
            self.n_calls = s["n_calls"]

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        pin, pout = PRICES[model]
        with self.lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.cost_usd += in_tok * pin + out_tok * pout
            self.n_calls += 1
            if self.n_calls % 50 == 0:
                self._persist()
            if self.cost_usd > self.cap_usd:
                self._persist()
                raise SpendCapExceeded(
                    f"spend ${self.cost_usd:.2f} exceeds cap ${self.cap_usd:.2f}")

    def _persist(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"input_tokens": self.input_tokens,
                       "output_tokens": self.output_tokens,
                       "cost_usd": self.cost_usd,
                       "n_calls": self.n_calls}, f)
        os.replace(tmp, self.state_path)


class Backend:
    is_mock = False

    def complete(self, system: str, user: str, max_tokens: int,
                 context: dict | None = None) -> str:
        raise NotImplementedError


class AnthropicBackend(Backend):
    def __init__(self, model: str, cost: CostTracker):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.cost = cost

    def complete(self, system: str, user: str, max_tokens: int,
                 context: dict | None = None) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        self.cost.add(self.model, resp.usage.input_tokens,
                      resp.usage.output_tokens)
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenRouterBackend(Backend):
    """Same Claude model routed via OpenRouter's chat-completions API
    (used when the operator supplies an OpenRouter key instead of a
    first-party Anthropic key).  Identical prompts; usage priced from the
    PRICES table and enforced by the same CostTracker."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, cost: CostTracker, api_key: str | None = None):
        import requests
        self.session = requests.Session()
        self.model = model
        self.cost = cost
        key = api_key or os.environ["OPENROUTER_API_KEY"]
        self.headers = {"Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"}

    def complete(self, system: str, user: str, max_tokens: int,
                 context: dict | None = None) -> str:
        import time
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": messages}
        if self.model.startswith("openai/gpt-5"):
            # reasoning models: without this, hidden reasoning tokens can
            # consume small max_tokens budgets and return empty content.
            # Adapter-level parameter only; prompts are unchanged.
            body["reasoning"] = {"effort": "minimal"}
            body["max_tokens"] = max(max_tokens, 64)
        if context and context.get("kind") == "judge":
            body["temperature"] = 0  # deterministic-ish classification
        last_err = None
        for attempt in range(6):
            try:
                r = self.session.post(self.URL, json=body,
                                      headers=self.headers, timeout=120)
                if r.status_code in (429, 500, 502, 503, 529):
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
                d = r.json()
                if "error" in d:
                    last_err = str(d["error"])[:200]
                    time.sleep(min(2 ** attempt, 30))
                    continue
                usage = d.get("usage", {})
                self.cost.add(self.model, int(usage.get("prompt_tokens", 0)),
                              int(usage.get("completion_tokens", 0)))
                return d["choices"][0]["message"]["content"] or ""
            except SpendCapExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - network layer retry
                last_err = repr(e)[:200]
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")


# --------------------------------------------------------------------------
# Mock backend: minimal-agent policy + template messages behind the same
# text protocol, so prompts are built and replies parsed exactly as live.
# --------------------------------------------------------------------------

# Message templates by intended class; {name} = the sender's preferred token.
MOCK_MESSAGES = {
    "descriptive": [
        "I picked {name} last time.",
        "my pick was {name}",
        "For the record, I chose {name}.",
    ],
    "directive": [
        "pick {name} next time",
        "go with {name}",
        "just use {name} from now on",
    ],
    "normative": [
        "{name} is the right one, you should use it",
        "everyone here uses {name}, stick to it",
        "you should have picked {name}",
    ],
    "other": [
        "that was unlucky",
        "hope we do better next time",
        "hmm interesting",
    ],
}
MOCK_MESSAGE_CLASS_P = {"descriptive": 0.35, "directive": 0.25,
                        "normative": 0.25, "other": 0.15}


class MockBackend(Backend):
    """Cheap-tier policy behind the LLM text interface."""

    is_mock = True

    def __init__(self, seed: int = 0, malformed_rate: float = 0.01,
                 pass_rate: float = 0.4):
        self.rng = np.random.default_rng(seed)
        self.malformed_rate = malformed_rate
        self.pass_rate = pass_rate

    def complete(self, system: str, user: str, max_tokens: int,
                 context: dict | None = None) -> str:
        assert context is not None, "mock backend requires context"
        kind = context["kind"]
        if kind == "choice":
            return self._choice(context)
        if kind == "message":
            return self._message(context)
        if kind == "comprehension":
            return context["true_answer"]
        if kind == "prior":
            pool = context["pool"]
            return f"NAME: {pool[int(self.rng.integers(len(pool)))]}"
        raise ValueError(kind)

    def _preferred(self, context: dict) -> str:
        """Fictitious play over partner plays in the memory window."""
        memory = context["memory"]  # list of (own, partner, payoff)
        pool = context["pool"]
        if not memory:
            return pool[int(self.rng.integers(len(pool)))]
        counts: dict[str, int] = {}
        for _own, partner, _p in memory:
            counts[partner] = counts.get(partner, 0) + 1
        best = max(counts.values())
        cands = [n for n, c in counts.items() if c == best]
        return cands[int(self.rng.integers(len(cands)))]

    def _choice(self, context: dict) -> str:
        if self.rng.random() < self.malformed_rate:
            return "I think I will go with the usual one."  # exercises retry+fallback
        return f"NAME: {self._preferred(context)}"

    def _message(self, context: dict) -> str:
        if self.rng.random() < self.pass_rate:
            return "PASS"
        classes = list(MOCK_MESSAGE_CLASS_P)
        probs = np.array([MOCK_MESSAGE_CLASS_P[c] for c in classes])
        cls = classes[int(self.rng.choice(len(classes), p=probs))]
        tmpl = MOCK_MESSAGES[cls][int(self.rng.integers(len(MOCK_MESSAGES[cls])))]
        return f"MESSAGE: {tmpl.format(name=self._preferred(context))}"


def parse_name(text: str, pool: list[str]) -> str | None:
    """Deterministic reply parsing (documented in RESULTS.md):
    1. a line 'NAME: <token>' with token in pool (case-insensitive);
    2. else, if exactly one pool token occurs as a word anywhere, take it;
    3. else None (caller retries once, then falls back to seeded uniform)."""
    m = re.search(r"NAME\s*:\s*([A-Za-z]+)", text, re.IGNORECASE)
    lowpool = {p.lower(): p for p in pool}
    if m and m.group(1).lower() in lowpool:
        return lowpool[m.group(1).lower()]
    words = re.findall(r"[A-Za-z]+", text.lower())
    hits = {w for w in words if w in lowpool}
    if len(hits) == 1:
        return lowpool[hits.pop()]
    return None


def parse_message(text: str) -> tuple[str | None, bool]:
    """Returns (message_or_None, well_formed)."""
    t = text.strip()
    if re.fullmatch(r"PASS\.?", t, re.IGNORECASE):
        return None, True
    m = re.search(r"MESSAGE\s*:\s*(.+)", t, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:200], True
    return None, False
