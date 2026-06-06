from gymnasium.envs.registration import register

register(
    id="HomeBot-v0",
    entry_point="homebot.env:HomeBotEnv",
)

from homebot.env import HomeBotEnv

__all__ = ["HomeBotEnv"]
