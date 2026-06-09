"""HomeBot 2D Gymnasium Environment."""

from gymnasium.envs.registration import register

from homebot.env import HomeBotEnv, HomeBotGoalEnv

__version__ = "0.1.0"

register(
    id="HomeBot2D-v1",
    entry_point="homebot.env:HomeBotEnv",
    max_episode_steps=1000,
)

register(
    id="HomeBot2D-Goal-v1",
    entry_point="homebot.env:HomeBotGoalEnv",
    max_episode_steps=1000,
)

__all__ = ["HomeBotEnv", "HomeBotGoalEnv"]
