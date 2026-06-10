# tests/test_goal_env.py
import numpy as np
import pytest
import gymnasium as gym
from homebot.env import HomeBotGoalEnv
from homebot.goals import GOAL_NAMES, GOAL_THRESHOLD


@pytest.fixture
def env():
    e = HomeBotGoalEnv(render_mode="rgb_array")
    _, _ = e.reset(seed=0)
    yield e
    e.close()


def test_obs_space_is_dict():
    e = HomeBotGoalEnv(render_mode="rgb_array")
    assert isinstance(e.observation_space, gym.spaces.Dict)
    for key in ("observation", "achieved_goal", "desired_goal"):
        assert key in e.observation_space.spaces
    e.close()


def test_obs_shapes_on_reset():
    e = HomeBotGoalEnv(render_mode="rgb_array")
    obs, _ = e.reset(seed=0)
    assert obs["observation"].shape == (84, 84, 3)
    assert obs["achieved_goal"].shape == (2,)
    assert obs["desired_goal"].shape == (2,)
    e.close()


def test_achieved_goal_matches_robot_position(env):
    obs, _ = env.reset(seed=0)
    assert obs["achieved_goal"][0] == pytest.approx(env._robot.x)
    assert obs["achieved_goal"][1] == pytest.approx(env._robot.y)


def test_desired_goal_within_map_bounds(env):
    obs, _ = env.reset(seed=0)
    assert obs["desired_goal"][0] > 0
    assert obs["desired_goal"][1] > 0


def test_random_goal_varies_across_resets(env):
    goals_seen = set()
    for seed in range(40):
        _, info = env.reset(seed=seed)
        goals_seen.add(info["active_goal"])
    assert len(goals_seen) > 1


def test_explicit_goal_sets_desired_goal_to_fixture():
    e = HomeBotGoalEnv(render_mode="rgb_array")
    obs, info = e.reset(seed=0, options={"goal": "go_to_fridge"})
    assert info["active_goal"] == "go_to_fridge"
    expected_x, expected_y = e._map.tile_to_pixel(*e._map.fixtures["fridge"])
    assert obs["desired_goal"][0] == pytest.approx(expected_x)
    assert obs["desired_goal"][1] == pytest.approx(expected_y)
    e.close()


def test_carry_preloaded_in_training_mode():
    e = HomeBotGoalEnv(render_mode="rgb_array", evaluate=False)
    e.reset(seed=0, options={"goal": "deliver_drink"})
    assert e._robot.carrying == "drink"
    e.close()


def test_carry_not_preloaded_in_eval_mode():
    e = HomeBotGoalEnv(render_mode="rgb_array", evaluate=True)
    e.reset(seed=0, options={"goal": "deliver_drink"})
    assert e._robot.carrying is None
    e.close()


def test_no_carry_for_navigation_goal_in_training():
    e = HomeBotGoalEnv(render_mode="rgb_array", evaluate=False)
    e.reset(seed=0, options={"goal": "go_to_fridge"})
    assert e._robot.carrying is None
    e.close()


def test_compute_reward_one_when_at_goal(env):
    obs, _ = env.reset(seed=0)
    desired = obs["desired_goal"]
    reward = env.compute_reward(desired, desired, {})
    assert reward == pytest.approx(1.0)


def test_compute_reward_zero_when_far(env):
    reward = env.compute_reward(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([9999.0, 9999.0], dtype=np.float32),
        {},
    )
    assert reward == pytest.approx(0.0)


def test_compute_reward_batched(env):
    achieved = np.array([[0.0, 0.0], [500.0, 400.0]])
    desired  = np.array([[0.0, 0.0], [9999.0, 9999.0]])
    rewards = env.compute_reward(achieved, desired, {})
    assert rewards.shape == (2,)
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.0)


def test_step_returns_dict_obs(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    for key in ("observation", "achieved_goal", "desired_goal"):
        assert key in obs
    assert isinstance(reward, float)


def test_episode_terminates_when_goal_reached():
    e = HomeBotGoalEnv(render_mode="rgb_array", evaluate=False)
    e.reset(seed=0, options={"goal": "deliver_drink"})
    rec_x, rec_y = e._map.tile_to_pixel(*e._map.fixtures["recliner"])
    e._robot.x, e._robot.y = float(rec_x), float(rec_y)
    _, reward, terminated, _, _ = e.step(0)
    assert reward == pytest.approx(1.0)
    assert terminated is True
    e.close()


def test_trash_desired_goal_is_valid_tile_at_reset():
    e = HomeBotGoalEnv(render_mode="rgb_array")
    obs, _ = e.reset(seed=0, options={"goal": "collect_trash"})
    tile_pixel_coords = [e._map.tile_to_pixel(*p) for p in e._task_manager.trash_positions]
    desired = (obs["desired_goal"][0], obs["desired_goal"][1])
    assert any(
        abs(desired[0] - tx) < 1 and abs(desired[1] - ty) < 1
        for tx, ty in tile_pixel_coords
    )
    e.close()


def test_goals_subset_restricts_random_selection():
    e = HomeBotGoalEnv(render_mode="rgb_array", goals=["go_to_fridge"])
    for seed in range(10):
        _, info = e.reset(seed=seed)
        assert info["active_goal"] == "go_to_fridge"
    e.close()


def test_goal_to_coordinates_exposed_on_goal_env():
    e = HomeBotGoalEnv(render_mode="rgb_array")
    e.reset(seed=0)
    x, y = e.goal_to_coordinates("go_to_fridge")
    assert x > 0 and y > 0
    e.close()


def test_info_contains_carrying_and_active_goal(env):
    _, info = env.reset(seed=0)
    assert "carrying" in info
    assert "active_goal" in info
    assert info["active_goal"] in GOAL_NAMES
