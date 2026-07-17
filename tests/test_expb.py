import json
import os

import numpy as np

from namegame.expb import prompts
from namegame.expb.backend import MockBackend, parse_message, parse_name
from namegame.expb.judge import classify_rule_based, compute_agreement
from namegame.expb.runner import _run_config, define_cells_b, execute_run
from namegame.expb.tokens import generate_pool, is_real_word


def test_token_pool_clean():
    for seed in range(20):
        pool = generate_pool(10, np.random.default_rng(seed))
        assert len(set(pool)) == 10
        assert not any(is_real_word(t) for t in pool)
        prefixes = [t[:3] for t in pool]
        assert len(set(prefixes)) == 10


def test_prompt_lint():
    prompts.lint_all()  # raises on banned vocabulary
    assert prompts.lint("we should coordinate on a norm") != []


def test_parse_name():
    pool = ["kelu", "vopa", "trin"]
    assert parse_name("NAME: kelu", pool) == "kelu"
    assert parse_name("name: VOPA", pool) == "vopa"
    assert parse_name("I will go with kelu this time", pool) == "kelu"
    assert parse_name("maybe kelu or vopa", pool) is None
    assert parse_name("nothing relevant", pool) is None


def test_parse_message():
    assert parse_message("PASS") == (None, True)
    assert parse_message("pass.") == (None, True)
    assert parse_message("MESSAGE: pick kelu")[0] == "pick kelu"
    assert parse_message("random text") == (None, False)


def test_judge_rules():
    assert classify_rule_based("you should have picked kelu") == "normative"
    assert classify_rule_based("everyone here uses kelu, stick to it") == "normative"
    assert classify_rule_based("pick kelu next time") == "directive"
    assert classify_rule_based("I picked kelu last time.") == "descriptive"
    assert classify_rule_based("that was unlucky") == "other"


def test_mock_run_deterministic_and_resumable(tmp_path):
    cells = define_cells_b()
    cfg = _run_config("b_solitary", cells["b_solitary"], 0)
    d1, d2 = str(tmp_path / "a"), str(tmp_path / "b")
    s1 = execute_run(cfg, d1, MockBackend(seed=cfg["seed"]))
    s2 = execute_run(cfg, d2, MockBackend(seed=cfg["seed"]))
    assert s1["solitary"] == s2["solitary"]
    j1 = open(os.path.join(d1, "b_solitary_r00", "journal.jsonl")).read()
    j2 = open(os.path.join(d2, "b_solitary_r00", "journal.jsonl")).read()
    assert j1 == j2


def test_kappa(tmp_path):
    csv = tmp_path / "v.csv"
    csv.write_text("id,text,judge_label,hand_label(FILL IN)\n"
                   "0,a,normative,normative\n1,b,directive,directive\n"
                   "2,c,other,normative\n3,d,descriptive,descriptive\n")
    r = compute_agreement(str(csv))
    assert r["n"] == 4 and 0 < r["kappa"] <= 1
