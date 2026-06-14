"""Tests for the random_start parameter on HomeBotEnv and HomeBotGoalEnv."""
import math
import numpy as np
import pytest
from homebot.env import HomeBotEnv, HomeBotGoalEnv, _RANDOM_START_CLEARANCE
from homebot.robot import Robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _robot_collides_in_env(env, x, y) -> bool:
    return env._robot._collides(x, y, env._map.wall_solid, env._map.tile_size,
                                env._map.fixture_pixel_rects)


def _goal_pixels(env):
    """All goal-location pixel centres we must stay clear of."""
    pixels = []
    for col, row in env._task_manager.trash_positions:
        pixels.append(tuple(float(v) for v in env._map.tile_to_pixel(col, row)))
    for fname in ("fridge", "recliner", "door"):
        if fname in env._map.fixtures:
            px, py = env._map.tile_to_pixel(*env._map.fixtures[fname])
            pixels.append((float(px), float(py)))
    return pixels


def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ---------------------------------------------------------------------------
# HomeBotEnv — basic parameter behaviour
# ---------------------------------------------------------------------------

class TestHomeBotEnvRandomStart:

    def test_default_is_fixed_start(self):
        """random_start=False → robot always starts at the map's default tile."""
        e = HomeBotEnv()
        default_x, default_y = e._map.tile_to_pixel(*e._map.robot_start_tile)
        for seed in range(5):
            obs, _ = e.reset(seed=seed)
            assert e._robot.x == pytest.approx(default_x)
            assert e._robot.y == pytest.approx(default_y)
        e.close()

    def test_random_start_flag_stored(self):
        e = HomeBotEnv(random_start=True)
        assert e.random_start is True
        e.close()

    def test_random_start_false_flag_stored(self):
        e = HomeBotEnv(random_start=False)
        assert e.random_start is False
        e.close()

    def test_random_start_position_varies_across_seeds(self):
        """Different seeds must produce different spawn positions."""
        e = HomeBotEnv(random_start=True, n_trash=2)
        positions = set()
        for seed in range(20):
            e.reset(seed=seed)
            positions.add((round(e._robot.x), round(e._robot.y)))
        assert len(positions) > 1, "random_start positions did not vary across seeds"
        e.close()

    def test_random_start_not_in_wall(self):
        """Robot must never spawn inside a wall or solid tile."""
        e = HomeBotEnv(random_start=True, n_trash=2)
        ts = e._map.tile_size
        for seed in range(30):
            e.reset(seed=seed)
            col = int(e._robot.x // ts)
            row = int(e._robot.y // ts)
            assert not e._map.wall_solid[row, col], (
                f"seed={seed}: spawned at tile ({col},{row}) which is solid"
            )
        e.close()

    def test_random_start_robot_not_colliding(self):
        """Robot body must fit at the spawn point without intersecting walls/fixtures."""
        e = HomeBotEnv(random_start=True, n_trash=2)
        for seed in range(30):
            e.reset(seed=seed)
            assert not _robot_collides_in_env(e, e._robot.x, e._robot.y), (
                f"seed={seed}: robot at ({e._robot.x:.1f},{e._robot.y:.1f}) collides"
            )
        e.close()

    def test_random_start_clearance_from_goals(self):
        """Spawn must be >= _RANDOM_START_CLEARANCE px from every goal location."""
        e = HomeBotEnv(random_start=True, n_trash=2)
        for seed in range(30):
            e.reset(seed=seed)
            rx, ry = e._robot.x, e._robot.y
            for gx, gy in _goal_pixels(e):
                d = _dist(rx, ry, gx, gy)
                assert d >= _RANDOM_START_CLEARANCE, (
                    f"seed={seed}: robot at ({rx:.1f},{ry:.1f}) is {d:.1f}px from goal "
                    f"({gx:.1f},{gy:.1f}), need >= {_RANDOM_START_CLEARANCE}"
                )
        e.close()

    def test_random_start_deterministic_with_same_seed(self):
        """Same seed → identical spawn position (Gymnasium determinism contract)."""
        e = HomeBotEnv(random_start=True, n_trash=2)
        for seed in range(10):
            e.reset(seed=seed)
            x1, y1 = e._robot.x, e._robot.y
            e.reset(seed=seed)
            x2, y2 = e._robot.x, e._robot.y
            assert x1 == pytest.approx(x2) and y1 == pytest.approx(y2), (
                f"seed={seed}: first reset ({x1},{y1}) ≠ second reset ({x2},{y2})"
            )
        e.close()

    def test_step_works_after_random_start(self):
        """Env must be fully functional after a random start reset."""
        e = HomeBotEnv(random_start=True)
        e.reset(seed=7)
        obs, reward, terminated, truncated, info = e.step(0)
        assert obs.shape == (84, 84, 3)
        assert isinstance(reward, float)
        assert "carrying" in info
        e.close()

    def test_obs_shape_unchanged_with_random_start(self):
        e = HomeBotEnv(random_start=True)
        obs, _ = e.reset(seed=0)
        assert obs.shape == (84, 84, 3)
        assert obs.dtype == np.uint8
        e.close()

    def test_gymnasium_check_env_with_random_start(self):
        """Gymnasium's built-in checker (incl. determinism) must pass."""
        from gymnasium.utils.env_checker import check_env
        e = HomeBotEnv(random_start=True, n_trash=2, max_steps=50)
        check_env(e, skip_render_check=True)
        e.close()


# ---------------------------------------------------------------------------
# HomeBotGoalEnv — random_start
# ---------------------------------------------------------------------------

class TestHomeBotGoalEnvRandomStart:

    def test_default_is_fixed_start(self):
        e = HomeBotGoalEnv()
        default_x, default_y = e._map.tile_to_pixel(*e._map.robot_start_tile)
        for seed in range(5):
            e.reset(seed=seed)
            assert e._robot.x == pytest.approx(default_x)
            assert e._robot.y == pytest.approx(default_y)
        e.close()

    def test_random_start_position_varies(self):
        e = HomeBotGoalEnv(random_start=True, n_trash=2)
        positions = set()
        for seed in range(20):
            e.reset(seed=seed)
            positions.add((round(e._robot.x), round(e._robot.y)))
        assert len(positions) > 1
        e.close()

    def test_random_start_not_in_wall_goal_env(self):
        e = HomeBotGoalEnv(random_start=True, n_trash=2)
        ts = e._map.tile_size
        for seed in range(30):
            e.reset(seed=seed)
            col = int(e._robot.x // ts)
            row = int(e._robot.y // ts)
            assert not e._map.wall_solid[row, col], (
                f"seed={seed}: GoalEnv spawned inside solid tile ({col},{row})"
            )
        e.close()

    def test_random_start_clearance_goal_env(self):
        e = HomeBotGoalEnv(random_start=True, n_trash=2)
        for seed in range(30):
            e.reset(seed=seed)
            rx, ry = e._robot.x, e._robot.y
            for gx, gy in _goal_pixels(e):
                d = _dist(rx, ry, gx, gy)
                assert d >= _RANDOM_START_CLEARANCE, (
                    f"seed={seed}: GoalEnv robot at ({rx:.1f},{ry:.1f}) too close "
                    f"to goal ({gx:.1f},{gy:.1f}): {d:.1f}px < {_RANDOM_START_CLEARANCE}"
                )
        e.close()

    def test_random_start_deterministic_goal_env(self):
        e = HomeBotGoalEnv(random_start=True, n_trash=2)
        for seed in range(10):
            e.reset(seed=seed)
            x1, y1 = e._robot.x, e._robot.y
            e.reset(seed=seed)
            x2, y2 = e._robot.x, e._robot.y
            assert x1 == pytest.approx(x2) and y1 == pytest.approx(y2)
        e.close()

    def test_obs_contains_dict_keys_with_random_start(self):
        e = HomeBotGoalEnv(random_start=True)
        obs, info = e.reset(seed=0)
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert "active_goal" in info
        e.close()

    def test_achieved_goal_matches_robot_position_after_random_start(self):
        """achieved_goal must reflect the actual (random) robot spawn pixel."""
        e = HomeBotGoalEnv(random_start=True, n_trash=2)
        for seed in range(10):
            obs, _ = e.reset(seed=seed)
            assert obs["achieved_goal"][0] == pytest.approx(e._robot.x)
            assert obs["achieved_goal"][1] == pytest.approx(e._robot.y)
        e.close()

    def test_gymnasium_check_env_goal_env_random_start(self):
        from gymnasium.utils.env_checker import check_env
        e = HomeBotGoalEnv(random_start=True, n_trash=2, max_steps=50)
        check_env(e, skip_render_check=True)
        e.close()
