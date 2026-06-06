import numpy as np
import pytest
from homebot.maps import DefaultHouseMap, FLOOR
from homebot.robot import Robot


@pytest.fixture
def map_and_robot():
    m = DefaultHouseMap()
    start = m.tile_to_pixel(*m.robot_start_tile)
    return m, Robot(start)


def test_robot_starts_at_given_position(map_and_robot):
    m, robot = map_and_robot
    ex, ey = m.tile_to_pixel(*m.robot_start_tile)
    assert robot.x == ex
    assert robot.y == ey


def test_robot_carry_starts_none(map_and_robot):
    _, robot = map_and_robot
    assert robot.carrying is None


def test_discrete_south_increases_y(map_and_robot):
    m, robot = map_and_robot
    y0 = robot.y
    robot.move_discrete(4, m.tiles, m.tile_size)  # action 4 = South
    assert robot.y > y0


def test_discrete_east_increases_x(map_and_robot):
    m, robot = map_and_robot
    x0 = robot.x
    robot.move_discrete(2, m.tiles, m.tile_size)  # action 2 = East
    assert robot.x > x0


def test_wall_collision_keeps_robot_on_floor(map_and_robot):
    m, robot = map_and_robot
    for _ in range(200):
        robot.move_discrete(0, m.tiles, m.tile_size)  # North, into wall
    col = int(robot.x // m.tile_size)
    row = int(robot.y // m.tile_size)
    assert m.tiles[row, col] == FLOOR
    # Robot must be pinned near the wall — within one step of its resting position
    assert robot.y <= m.tile_to_pixel(*m.robot_start_tile)[1]
    assert robot.y >= robot.RADIUS


def test_robot_stays_in_bounds(map_and_robot):
    m, robot = map_and_robot
    for _ in range(200):
        robot.move_discrete(6, m.tiles, m.tile_size)  # West
    assert robot.x >= robot.RADIUS
    # Must be pinned near the left wall, not stopped in the middle of the room
    assert robot.x <= m.tile_size * 2


def test_reset_restores_position(map_and_robot):
    m, robot = map_and_robot
    robot.move_discrete(2, m.tiles, m.tile_size)
    robot.carrying = "drink"
    robot.reset()
    ex, ey = m.tile_to_pixel(*m.robot_start_tile)
    assert robot.x == ex
    assert robot.y == ey
    assert robot.carrying is None


def test_continuous_forward_moves_robot(map_and_robot):
    m, robot = map_and_robot
    x0, y0 = robot.x, robot.y
    robot.move_continuous(np.array([1.0, 0.0]), m.tiles, m.tile_size)
    assert robot.x != x0 or robot.y != y0


def test_continuous_wall_collision_keeps_on_floor(map_and_robot):
    m, robot = map_and_robot
    robot.angle = -np.pi / 2  # facing North
    for _ in range(200):
        robot.move_continuous(np.array([1.0, 0.0]), m.tiles, m.tile_size)
    col = int(robot.x // m.tile_size)
    row = int(robot.y // m.tile_size)
    assert m.tiles[row, col] == FLOOR
    assert robot.y >= robot.RADIUS
    assert robot.y <= m.tile_to_pixel(*m.robot_start_tile)[1]


def test_discrete_invalid_action_raises(map_and_robot):
    m, robot = map_and_robot
    with pytest.raises(ValueError):
        robot.move_discrete(8, m.tiles, m.tile_size)
