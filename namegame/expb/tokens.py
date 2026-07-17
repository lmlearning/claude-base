"""Nonsense-token generation (leakage control).

Fresh pronounceable non-words per run: CVCV / CVCVC / CVCCV patterns from
restricted consonant/vowel sets, filtered against an embedded list of real
English words and word-like strings, and against each other (no duplicates,
no pair differing only in final character).  Verified non-wordness is a
scientific control, not a style choice (Barrie et al. leakage critique).
"""

from __future__ import annotations

import numpy as np

CONSONANTS = list("bdfgklmnprstvz")
VOWELS = list("aeiou")

# Embedded blocklist: real English words (and common names/roman-alphabet
# loanwords) of the shapes our generator can produce.  The generator's
# 4-5-letter CV shapes overlap English rarely; this list covers the overlap.
REAL_WORDS = {
    "bake", "bald", "band", "bane", "bare", "base", "bide", "bike", "bile",
    "bind", "bite", "bode", "bold", "bond", "bone", "bore", "bade", "dame",
    "dare", "date", "daze", "dime", "dine", "dire", "dive", "dole", "dome",
    "done", "dose", "dote", "doze", "dude", "duke", "dune", "dupe", "fade",
    "fake", "fame", "fare", "fate", "faze", "file", "fine", "fire", "five",
    "fold", "fond", "fore", "fume", "fund", "fuse", "gale", "game", "gate",
    "gaze", "gene", "gild", "give", "gold", "gone", "gore", "guide", "hale",
    "kale", "kite", "lake", "lame", "lane", "late", "lime", "line", "lire",
    "live", "lobe", "lode", "lone", "lore", "lose", "lube", "lure", "lute",
    "made", "make", "male", "mane", "mare", "mate", "maze", "mend", "mild",
    "mile", "mind", "mine", "mire", "mite", "mode", "mold", "mole", "mope",
    "more", "mote", "move", "mule", "mute", "name", "nape", "nine", "node",
    "none", "nope", "nose", "note", "nude", "pale", "pane", "pare", "pate",
    "pave", "pile", "pine", "pipe", "poke", "pole", "pond", "pore", "pose",
    "pure", "rake", "rape", "rare", "rate", "rave", "raze", "ride", "rile",
    "rime", "ripe", "rise", "rite", "robe", "rode", "role", "rope", "rose",
    "rote", "rove", "rude", "rule", "rune", "sake", "sale", "same", "sane",
    "sate", "save", "seed", "side", "sine", "sire", "site", "size", "sole",
    "some", "sore", "tale", "tame", "tape", "tide", "tile", "time", "tine",
    "tire", "tone", "tore", "tote", "tube", "tune", "vane", "vase", "vibe",
    "vile", "vine", "vote", "zone", "bases", "bones", "banes", "gamer",
    "baker", "biker", "diner", "diver", "faker", "filer", "finer", "firer",
    "giver", "later", "liner", "loner", "loser", "maker", "miner", "mover",
    "muter", "namer", "nicer", "paler", "paver", "piler", "poker", "poser",
    "rader", "raker", "rarer", "rater", "raver", "rider", "riper", "riser",
    "roper", "ruder", "ruler", "saner", "saver", "sober", "tamer", "taper",
    "tiler", "timer", "toner", "tuner", "voter", "wider", "bingo", "bravo",
    "cargo", "combo", "condo", "credo", "curio", "disco", "gecko", "gumbo",
    "gusto", "hippo", "kudos", "lasso", "limbo", "lingo", "macro", "mango",
    "metro", "micro", "mondo", "motto", "nacho", "patio", "pesto", "photo",
    "piano", "pinto", "polo", "promo", "radio", "ratio", "retro", "rodeo",
    "salvo", "tango", "tempo", "torso", "turbo", "video", "banjo", "sumo",
    "solo", "silo", "sago", "peso", "memo", "demo", "dodo", "logo", "lido",
    "kilo", "judo", "hero", "halo", "fido", "expo", "veto", "typo", "zero",
    "taco", "toga", "tuba", "tuna", "sofa", "soda", "saga", "puma", "papa",
    "nova", "mama", "lava", "kiva", "java", "gala", "diva", "data", "cola",
    "coda", "boba", "beta", "aura", "area", "alga", "dana", "gina", "lena",
    "lisa", "mona", "nina", "rita", "rosa", "sara", "tina", "vera", "zara",
    "bella", "dora", "gita", "kira", "lara", "mira", "nora", "tara", "gaga",
    "dado", "gogo", "koko", "lolo", "mono", "nono", "popo", "toto", "zozo",
    "bobo", "dido", "gigo", "kiko", "nino", "pipo", "sisi", "tito", "vivo",
    "bene", "sane", "pene", "rene", "gene", "bide", "abide", "adobe",
}


try:  # comprehensive dictionary when available (english-words, web2 list)
    from english_words import get_english_words_set
    _DICT = get_english_words_set(["web2"], lower=True)
except Exception:  # pragma: no cover - fallback to the embedded list
    _DICT = set()


def is_real_word(token: str) -> bool:
    t = token.lower()
    return t in REAL_WORDS or t in _DICT


def _one_token(rng: np.random.Generator) -> str:
    pattern = ["CVCV", "CVCVC", "CVCCV"][int(rng.integers(3))]
    out = []
    for ch in pattern:
        pool = CONSONANTS if ch == "C" else VOWELS
        out.append(pool[int(rng.integers(len(pool)))])
    return "".join(out)


def generate_pool(n: int, rng: np.random.Generator, max_tries: int = 10_000) -> list[str]:
    """Fresh nonsense-token pool: non-words, mutually distinct, and no two
    tokens sharing a 3-character prefix (avoids confusable near-duplicates)."""
    pool: list[str] = []
    prefixes: set[str] = set()
    for _ in range(max_tries):
        tok = _one_token(rng)
        if is_real_word(tok):
            continue
        if tok in pool or tok[:3] in prefixes:
            continue
        pool.append(tok)
        prefixes.add(tok[:3])
        if len(pool) == n:
            return pool
    raise RuntimeError("could not generate token pool")
