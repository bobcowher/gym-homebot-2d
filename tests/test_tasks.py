import numpy as np
import pytest
from homebot.maps import DefaultHouseMap
from homebot.robot import Robot
from homebot.tasks import TaskManager


@pytest.fixture
def env():
    m = DefaultHouseMap()
    robot = Robot(m.tile_to_pixel(*m.robot_start_tile))
    tm = TaskManager(["trash", "drink", "package"])
    tm.reset(m, n_trash=3, rng=np.random.default_rng(42))
    return m, robot, tm


def test_reset_spawns_correct_trash_count(env):
    m, robot, tm = env
    assert len(tm.trash_positions) == 3


def test_reset_package_present(env):
    m, robot, tm = env
    assert tm.package_present is True


def test_reset_nothing_delivered(env):
    m, robot, tm = env
    assert tm.drink_delivered is False
    assert tm.package_delivered is False


def test_trash_collection_gives_reward(env):
    m, robot, tm = env
    robot.x, robot.y = m.tile_to_pixel(*tm.trash_positions[0])
    reward = tm.step(robot, m)
    assert reward == pytest.approx(1.0)


def test_trash_collection_removes_item(env):
    m, robot, tm = env
    robot.x, robot.y = m.tile_to_pixel(*tm.trash_positions[0])
    tm.step(robot, m)
    assert len(tm.trash_positions) == 2


def test_drink_pickup_at_fridge(env):
    m, robot, tm = env
    robot.x, robot.y = m.tile_to_pixel(*m.fixtures["fridge"])
    tm.step(robot, m)
    assert robot.carrying == "drink"


def test_drink_delivery_to_recliner(env):
    m, robot, tm = env
    robot.carrying = "drink"
    robot.x, robot.y = m.tile_to_pixel(*m.fixtures["recliner"])
    reward = tm.step(robot, m)
    assert reward == pytest.approx(1.0)
    assert tm.drink_delivered is True
    assert robot.carrying is None


def test_package_pickup_at_door(env):
    m, robot, tm = env
    robot.x, robot.y = m.tile_to_pixel(*m.fixtures["door"])
    tm.step(robot, m)
    assert robot.carrying == "package"
    assert tm.package_present is False


def test_package_delivery_to_recliner(env):
    m, robot, tm = env
    robot.carrying = "package"
    tm.package_present = False
    robot.x, robot.y = m.tile_to_pixel(*m.fixtures["recliner"])
    reward = tm.step(robot, m)
    assert reward == pytest.approx(1.0)
    assert tm.package_delivered is True
    assert robot.carrying is None


def test_carrying_blocks_second_pickup(env):
    m, robot, tm = env
    robot.carrying = "drink"
    robot.x, robot.y = m.tile_to_pixel(*m.fixtures["door"])
    tm.step(robot, m)
    assert robot.carrying == "drink"  # did not pick up package


def test_is_done_when_all_complete(env):
    m, robot, tm = env
    tm.trash_positions = []
    tm.drink_delivered = True
    tm.package_delivered = True
    assert tm.is_done() is True


def test_is_done_false_when_trash_remains(env):
    m, robot, tm = env
    tm.drink_delivered = True
    tm.package_delivered = True
    assert tm.is_done() is False


def test_goal_not_in_list_no_reward():
    m = DefaultHouseMap()
    robot = Robot(m.tile_to_pixel(*m.robot_start_tile))
    tm = TaskManager(["trash"])  # drink and package excluded
    tm.reset(m, n_trash=1, rng=np.random.default_rng(0))
    robot.x, robot.y = m.tile_to_pixel(*m.fixtures["fridge"])
    tm.step(robot, m)
    assert robot.carrying is None  # fridge interaction ignored


def test_goal_not_in_list_is_done_ignores_it():
    m = DefaultHouseMap()
    robot = Robot(m.tile_to_pixel(*m.robot_start_tile))
    tm = TaskManager(["trash"])
    tm.reset(m, n_trash=1, rng=np.random.default_rng(0))
    tm.trash_positions = []
    assert tm.is_done() is True  # drink/package not in goals, so irrelevant


def test_get_info_keys(env):
    m, robot, tm = env
    info = tm.get_info()
    assert "trash_remaining" in info
    assert "drink_delivered" in info
    assert "package_delivered" in info
    assert "package_present" in info
