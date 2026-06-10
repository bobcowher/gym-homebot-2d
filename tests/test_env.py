import numpy as np
import pytest
import gymnasium as gym
from homebot.env import HomeBotEnv


@pytest.fixture
def env():
    e = HomeBotEnv(goals=["trash", "drink", "package"], n_trash=2)
    yield e
    e.close()


def test_reset_obs_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8


def test_reset_info_is_dict(env):
    _, info = env.reset()
    assert isinstance(info, dict)
    assert "trash_remaining" in info


def test_step_return_types(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (84, 84, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_truncated_at_max_steps():
    e = HomeBotEnv(goals=["trash"], n_trash=100, max_steps=3)
    e.reset()
    for _ in range(2):
        _, _, _, truncated, _ = e.step(0)
        assert not truncated
    _, _, _, truncated, _ = e.step(0)
    assert truncated
    e.close()


def test_discrete_action_space():
    e = HomeBotEnv(action_mode="discrete")
    assert isinstance(e.action_space, gym.spaces.Discrete)
    assert e.action_space.n == 8
    e.close()


def test_continuous_action_space():
    e = HomeBotEnv(action_mode="continuous")
    assert isinstance(e.action_space, gym.spaces.Box)
    assert e.action_space.shape == (2,)
    e.close()


def test_observation_space_shape(env):
    assert env.observation_space.shape == (84, 84, 3)


def test_custom_obs_resolution():
    e = HomeBotEnv(obs_resolution=(64, 64))
    obs, _ = e.reset()
    assert obs.shape == (64, 64, 3)
    assert e.observation_space.shape == (64, 64, 3)
    e.close()


def test_render_returns_ndarray(env):
    env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8


def test_goals_subset_excludes_tasks():
    e = HomeBotEnv(goals=["trash"], n_trash=1, max_steps=5000)
    e.reset()
    # is_done only cares about trash when goals=["trash"]
    e._task_manager.trash_positions = []
    _, _, terminated, _, _ = e.step(0)
    assert terminated
    e.close()


def test_seed_reproducibility():
    e = HomeBotEnv(goals=["trash"], n_trash=5)
    obs1, _ = e.reset(seed=99)
    e.reset()
    obs2, _ = e.reset(seed=99)
    np.testing.assert_array_equal(obs1, obs2)
    e.close()


def test_info_contains_carrying(env):
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "carrying" in info


def test_gymnasium_env_check():
    from gymnasium.utils.env_checker import check_env
    e = HomeBotEnv(n_trash=2, max_steps=50)
    check_env(e, skip_render_check=True)
    e.close()


def test_goal_to_coordinates_on_plain_env():
    import pytest
    e = HomeBotEnv(render_mode="rgb_array")
    e.reset(seed=0)
    x, y = e.goal_to_coordinates("go_to_fridge")
    expected_x, expected_y = e._map.tile_to_pixel(*e._map.fixtures["fridge"])
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(expected_y)
    e.close()


def test_goal_env_registered_and_usable():
    import gymnasium as gym
    e = gym.make("HomeBot2D-Goal-v1", render_mode="rgb_array")
    obs, _ = e.reset(seed=0)
    assert "observation" in obs
    assert "achieved_goal" in obs
    assert "desired_goal" in obs
    e.close()
