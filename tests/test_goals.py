import numpy as np
import pytest
from homebot.maps import DefaultHouseMap
from homebot.tasks import TaskManager
from homebot.goals import GOAL_REGISTRY, GOAL_NAMES, GOAL_THRESHOLD, goal_to_coords


def _setup():
    m = DefaultHouseMap()
    rng = np.random.default_rng(42)
    tm = TaskManager(["trash", "drink", "package"])
    tm.reset(m, n_trash=2, rng=rng)
    return m, tm


def test_goal_registry_has_all_five_goals():
    assert set(GOAL_NAMES) == {
        "go_to_fridge", "deliver_drink", "go_to_door", "deliver_package", "collect_trash"
    }


def test_goal_threshold_positive():
    assert GOAL_THRESHOLD > 0


def test_fixture_goal_returns_pixel_coords():
    m, tm = _setup()
    x, y = goal_to_coords("go_to_fridge", m)
    fx, fy = m.tile_to_pixel(*m.fixtures["fridge"])
    assert x == pytest.approx(fx) and y == pytest.approx(fy)


def test_deliver_drink_returns_recliner_coords():
    m, tm = _setup()
    x, y = goal_to_coords("deliver_drink", m)
    rx, ry = m.tile_to_pixel(*m.fixtures["recliner"])
    assert x == pytest.approx(rx) and y == pytest.approx(ry)


def test_collect_trash_returns_existing_tile():
    m, tm = _setup()
    x, y = goal_to_coords("collect_trash", m, tm.trash_positions)
    tile_coords = [m.tile_to_pixel(*pos) for pos in tm.trash_positions]
    assert (x, y) in tile_coords


def test_collect_trash_raises_when_no_trash():
    m, _ = _setup()
    with pytest.raises(ValueError):
        goal_to_coords("collect_trash", m, [])


def test_initial_carry_none_for_navigation_goals():
    for name in ("go_to_fridge", "go_to_door", "collect_trash"):
        _, carry = GOAL_REGISTRY[name]
        assert carry is None


def test_initial_carry_set_for_delivery_goals():
    _, carry = GOAL_REGISTRY["deliver_drink"]
    assert carry == "drink"
    _, carry = GOAL_REGISTRY["deliver_package"]
    assert carry == "package"
