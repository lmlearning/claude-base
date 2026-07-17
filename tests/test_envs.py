import numpy as np

from namegame.envs import e1_relay, e2_grid, e3_ref, e4_bargain
from namegame.envs.common import lint_env


def test_e1_perm_extraction():
    fields = ["kelu", "vopa", "trin", "malo", "zeti"]
    values = ["aaa", "bbb", "ccc", "ddd", "eee"]
    note = "trin ccc, kelu aaa, zeti eee, vopa bbb, malo ddd"
    assert e1_relay.extract_perm(note, fields, values) == [2, 0, 4, 1, 3]
    assert e1_relay.extract_perm("kelu aaa only", fields, values) is None
    assert e1_relay.count_named(note, fields) == 5


def test_e2_classify():
    left = [(r, c) for r in range(4) for c in range(2)]
    assert e2_grid.classify_cells(left) == "left"
    even = [(r, c) for r in range(4) for c in range(4) if (r + c) % 2 == 0]
    assert e2_grid.classify_cells(even) == "even"
    mixed = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 3), (3, 0), (3, 3)]
    assert e2_grid.classify_cells(mixed) == "mixed"
    assert e2_grid.classify_cells([(0, 0)]) == "short"


def test_e2_parse_cell():
    assert e2_grid._parse_cell("CELL: 2,3") == (1, 2)
    assert e2_grid._parse_cell("2 , 3 please") == (1, 2)
    assert e2_grid._parse_cell("CELL: 9,9") is None
    assert e2_grid._parse_cell("no") is None


def test_e3_items_and_extraction():
    traits, items = e3_ref.make_items(np.random.default_rng(0))
    assert len(items) == 6 and all(len(i) == 3 for i in items)
    flat = [t for it in items for t in it]
    assert all(flat.count(t) >= 2 for t in range(8))
    note = f"{traits[0]} and {traits[3]}"
    assert e3_ref.extract_traitset(note, traits) == [0, 3]
    coin = e3_ref.extract_coinages("grab the zorply one", traits)
    assert "zorply" in coin


def test_e4_parse():
    assert e4_bargain._parse_demand("I request 70 points") == 70
    assert e4_bargain._parse_demand("50") == 50
    assert e4_bargain._parse_demand("40") is None


def test_env_lint_catches():
    try:
        lint_env(["please use a consistent style"])
    except AssertionError:
        return
    raise AssertionError("lint should have fired")
