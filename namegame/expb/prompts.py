"""Frozen prompt templates for Experiment B.

SCIENTIFIC CONTROLS — read before touching anything here:
  * The task descriptions present mechanics + the goal of maximising points
    ONLY.  They must never contain (or imply) the banned vocabulary below,
    nor suggest that agreement per se, teaching, correcting, or rules are
    the point of the task.
  * Three paraphrase framings with identical mechanics; one is assigned per
    run; framing-invariance is tested in analysis.
  * The dialogue-channel invitation is strictly neutral — an opportunity to
    say something, with an explicit option to decline.
  * TREAT ALL WORDING AS FROZEN once live runs begin.  ``lint_all()`` is run
    at startup and raises if any banned stem appears anywhere.
"""

from __future__ import annotations

BANNED_STEMS = [
    "convention", "norm", "coordinat", "consensus", "game theor",
    "teach", "correct", "agree", "rule", "custom", "conform",
    "cooperat", "collaborat", "social",
]


def lint(text: str) -> list[str]:
    low = text.lower()
    return [s for s in BANNED_STEMS if s in low]


# --------------------------------------------------------------------------
# Framings.  {W} {H} {reward} {penalty} {pool} are filled per run.
# Mechanics are identical across the three; only the paraphrase differs.
# --------------------------------------------------------------------------

FRAMINGS = {
    "rounds": (
        "You are taking part in an activity that runs over many rounds. "
        "In each round you are paired with another participant. You and "
        "your partner each pick one name from this list of {W} names: {pool}. "
        "You make your pick without seeing your partner's pick. If both of "
        "you picked the same name, you each earn {reward} points. If you "
        "picked different names, you each lose {penalty_abs} points. "
        "Your aim is to finish with as many points as possible. "
        "You can see the outcomes of your last {H} rounds."
    ),
    "study": (
        "You are a participant in a points-based study with multiple turns. "
        "On every turn, the system pairs you with another participant. Each "
        "of you privately selects a single name out of the following {W} "
        "options: {pool}. When both selections are the same name, each "
        "participant gains {reward} points; when the selections differ, each "
        "participant loses {penalty_abs} points. Try to earn the highest "
        "point total you can. A record of your most recent {H} turns is "
        "shown to you."
    ),
    "market": (
        "You are one of several traders in a repeated exchange. Each cycle, "
        "you are matched with one other trader. Independently, you and the "
        "matched trader each submit one label chosen from these {W} labels: "
        "{pool}. Matching submissions pay each trader {reward} points; "
        "mismatched submissions cost each trader {penalty_abs} points. "
        "Maximise your points. The results of your last {H} cycles are "
        "available to you."
    ),
}

CHOICE_INSTRUCTION = (
    "It is time for your next pick. Reply with one single line and "
    "nothing else, in exactly this format:\nNAME: <one name from the list>"
)

CHOICE_RETRY = (
    "Your reply could not be read. Reply with one single line and nothing "
    "else, in exactly this format, using one name from the list:\nNAME: <name>"
)

HISTORY_HEADER = "Your recent outcomes, oldest first:"
HISTORY_LINE = ("- you picked \"{own}\", the other participant picked "
                "\"{partner}\", you {outcome} {points} points")
NO_HISTORY = "You have no previous outcomes yet."

INCOMING_MESSAGE = ("Before this pick, the participant from your previous "
                    "pairing sent you this message: \"{text}\"")

# --- Dialogue channel (strictly neutral; frozen) --------------------------

MESSAGE_INVITATION = (
    "Your latest pairing did not produce matching picks. If you wish, you "
    "may send a short message (at most 20 words) to that participant, who "
    "will see it before their next pick. You may also send nothing. Reply "
    "with a single line in exactly one of these formats:\n"
    "MESSAGE: <your message>\nPASS\nReply with that single line and nothing else."
)

# --- Comprehension check (asked in the run's framing, before any play) ----

COMPREHENSION_QUESTIONS = [
    ("If you and the other participant pick the same name, how many points "
     "do you each gain? Reply with just the number.",
     lambda ans, cfg: _first_int(ans) == int(cfg["reward"])),
    ("If you and the other participant pick different names, how many "
     "points do you each lose? Reply with just the number.",
     lambda ans, cfg: _first_int(ans) is not None
     and abs(_first_int(ans)) == int(cfg["penalty_abs"])),
    ("How many names are on the list you pick from? Reply with just the "
     "number.",
     lambda ans, cfg: _first_int(ans) == int(cfg["W"])),
    ("Can you see the outcomes of your recent rounds? Reply YES or NO.",
     lambda ans, cfg: "yes" in ans.lower()),
]


def _first_int(text: str) -> int | None:
    import re
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


# --- Zero-shot prior probe (minimal context, no game mechanics) -----------

PRIOR_PROBE = (
    "Here is a list of names: {pool}. Reply with a single line in exactly "
    "this format, using one name from the list:\nNAME: <name>"
)


def build_system_prompt(framing: str, pool: list[str], W: int, H: int,
                        reward: float, penalty: float) -> str:
    return FRAMINGS[framing].format(
        W=W, H=H, reward=int(reward), penalty_abs=int(abs(penalty)),
        pool=", ".join(pool),
    )


def lint_all() -> None:
    """Raise if any banned stem appears in any template. Run at startup."""
    texts = list(FRAMINGS.values()) + [
        CHOICE_INSTRUCTION, CHOICE_RETRY, HISTORY_HEADER, HISTORY_LINE,
        NO_HISTORY, INCOMING_MESSAGE, MESSAGE_INVITATION, PRIOR_PROBE,
    ] + [q for q, _ in COMPREHENSION_QUESTIONS]
    problems = []
    for t in texts:
        hits = lint(t)
        if hits:
            problems.append((t[:60], hits))
    if problems:
        raise AssertionError(f"banned vocabulary in frozen prompts: {problems}")


lint_all()
