"""Collision and task-interaction regression tests.

Covers the solid-fixture / lawn / doorway behavior added when fixtures became
non-passthrough, and confirms trash/drink/package interactions still fire when
the robot stands on adjacent floor (fixtures are now solid, so it cannot stand
on them).
"""
import pytest

from homebot.env import HomeBotEnv


@pytest.fixture
def env():
    e = HomeBotEnv(render_mode="rgb_array")
    e.reset(seed=0)
    yield e
    e.close()


def test_cannot_drive_through_recliner(env):
    m, r = env._map, env._robot
    r.x, r.y = m.tile_to_pixel(7, 5)  # just east of the recliner footprint
    for _ in range(40):
        r.move_discrete(6, m.solid, m.tile_size)  # push west
    assert r.x > 6 * m.tile_size  # stopped before entering the chair


def test_cannot_enter_lawn_but_reaches_doorway(env):
    m, r = env._map, env._robot
    r.x, r.y = m.tile_to_pixel(22, 9)  # hallway by the east doorway
    for _ in range(40):
        r.move_discrete(2, m.solid, m.tile_size)  # push east toward outside
    assert r.x > 22 * m.tile_size                      # made it into the doorway
    assert r.x < 24 * m.tile_size - r.RADIUS + 1       # but not onto the lawn


def test_trash_pickup(env):
    m, r, tm = env._map, env._robot, env._task_manager
    assert tm.trash_positions
    col, row = tm.trash_positions[0]
    r.x, r.y = m.tile_to_pixel(col, row)
    r.carrying = None
    before = len(tm.trash_positions)
    tm.step(r, m)
    assert len(tm.trash_positions) == before - 1


def test_drink_pickup_and_delivery(env):
    m, r, tm = env._map, env._robot, env._task_manager
    r.carrying = None
    r.x, r.y = m.tile_to_pixel(20, 2)  # adjacent to fridge (fridge now at row 1)
    tm.step(r, m)
    assert r.carrying == "drink"
    r.x, r.y = m.tile_to_pixel(7, 5)   # adjacent to recliner
    tm.step(r, m)
    assert tm.drink_delivered and r.carrying is None


def test_package_pickup_and_delivery(env):
    m, r, tm = env._map, env._robot, env._task_manager
    r.carrying = None
    r.x, r.y = m.tile_to_pixel(22, 9)  # adjacent to door
    tm.step(r, m)
    assert r.carrying == "package"
    r.x, r.y = m.tile_to_pixel(7, 5)   # adjacent to recliner
    tm.step(r, m)
    assert tm.package_delivered and r.carrying is None
