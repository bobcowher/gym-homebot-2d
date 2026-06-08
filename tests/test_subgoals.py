"""Tests for the subgoals system in TaskManager and HomeBotEnv."""
import pytest
from homebot.env import HomeBotEnv
from homebot.tasks import TaskManager


# --- TaskManager.active_goals unit tests ---

@pytest.fixture
def tm():
    m = __import__("homebot.maps", fromlist=["DefaultHouseMap"]).DefaultHouseMap()
    rng = __import__("numpy").random.default_rng(0)
    manager = TaskManager(["trash", "drink", "package"])
    manager.reset(m, 2, rng)
    return manager, m


def test_high_level_goals_on_reset(tm):
    manager, _ = tm
    goals = manager.active_goals()
    assert "fetch_drink" in goals
    assert "retrieve_package" in goals
    assert "collect_trash" in goals


def test_subgoal_drink_before_pickup(tm):
    manager, _ = tm
    robot = __import__("homebot.robot", fromlist=["Robot"]).Robot((0, 0))
    robot.carrying = None
    goals = manager.active_goals(robot)
    assert "go_to_fridge" in goals
    assert "fetch_drink" not in goals


def test_subgoal_drink_after_pickup(tm):
    manager, _ = tm
    robot = __import__("homebot.robot", fromlist=["Robot"]).Robot((0, 0))
    robot.carrying = "drink"
    goals = manager.active_goals(robot)
    assert "deliver_to_human" in goals
    assert "go_to_fridge" not in goals


def test_subgoal_package_before_pickup(tm):
    manager, _ = tm
    robot = __import__("homebot.robot", fromlist=["Robot"]).Robot((0, 0))
    robot.carrying = None
    goals = manager.active_goals(robot)
    assert "go_to_door" in goals


def test_subgoal_package_after_pickup(tm):
    manager, _ = tm
    robot = __import__("homebot.robot", fromlist=["Robot"]).Robot((0, 0))
    robot.carrying = "package"
    manager.package_present = False  # simulate pickup
    goals = manager.active_goals(robot)
    assert "deliver_to_human" in goals
    assert "go_to_door" not in goals


def test_completed_goals_absent(tm):
    manager, _ = tm
    manager.drink_delivered = True
    manager.package_delivered = True
    manager.trash_positions = []
    goals = manager.active_goals()
    assert goals == []


# --- HomeBotEnv integration tests ---

@pytest.fixture
def env_default():
    e = HomeBotEnv(render_mode="rgb_array")
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture
def env_subgoals():
    e = HomeBotEnv(render_mode="rgb_array", subgoals=True)
    e.reset(seed=0)
    yield e
    e.close()


def test_info_has_goals_on_reset():
    e = HomeBotEnv(render_mode="rgb_array")
    _, info = e.reset(seed=0)
    assert "goals" in info
    assert isinstance(info["goals"], list)
    e.close()


def test_default_mode_high_level_goals(env_default):
    _, info = env_default.reset(seed=0)
    assert "fetch_drink" in info["goals"]
    assert "retrieve_package" in info["goals"]
    assert "go_to_fridge" not in info["goals"]
    assert "go_to_door" not in info["goals"]


def test_subgoals_mode_low_level_goals(env_subgoals):
    _, info = env_subgoals.reset(seed=0)
    assert "go_to_fridge" in info["goals"]
    assert "go_to_door" in info["goals"]
    assert "fetch_drink" not in info["goals"]
    assert "retrieve_package" not in info["goals"]


def test_subgoal_advances_after_drink_pickup():
    e = HomeBotEnv(render_mode="rgb_array", subgoals=True)
    e.reset(seed=0)
    m, r, tm = e._map, e._robot, e._task_manager
    r.x, r.y = m.tile_to_pixel(20, 2)  # adjacent to fridge
    tm.step(r, m)                        # trigger pickup
    assert r.carrying == "drink"
    info = tm.get_info(r)
    assert "deliver_to_human" in info["goals"]
    assert "go_to_fridge" not in info["goals"]
    e.close()


def test_subgoal_advances_after_package_pickup():
    e = HomeBotEnv(render_mode="rgb_array", subgoals=True)
    e.reset(seed=0)
    m, r, tm = e._map, e._robot, e._task_manager
    r.x, r.y = m.tile_to_pixel(22, 9)  # adjacent to door
    tm.step(r, m)
    assert r.carrying == "package"
    info = tm.get_info(r)
    assert "deliver_to_human" in info["goals"]
    assert "go_to_door" not in info["goals"]
    e.close()
