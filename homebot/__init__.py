"""HomeBot 2D Gymnasium Environment."""

from gymnasium.envs.registration import register

from homebot.env import HomeBotEnv

__version__ = "0.1.0"

register(
    id="HomeBot2D-v1",
    entry_point="homebot.env:HomeBotEnv",
    max_episode_steps=1000,
)

__all__ = ["HomeBotEnv"]
