from typing import Optional
import numpy as np
import gymnasium as gym

from homebot.maps import MAP_REGISTRY, Map
from homebot.robot import Robot
from homebot.tasks import TaskManager
from homebot.renderer import Renderer


class HomeBotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        goals: Optional[list[str]] = None,
        action_mode: str = "discrete",
        obs_resolution: tuple[int, int] = (84, 84),
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        n_trash: int = 2,
        map_name: str = "default",
        subgoals: bool = False,
    ):
        super().__init__()
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode must be 'discrete' or 'continuous', got {action_mode!r}")
        if goals is None:
            goals = ["trash", "drink", "package"]

        self.goals = goals
        self.action_mode = action_mode
        self.obs_resolution = obs_resolution
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.n_trash = n_trash
        self.map_name = map_name
        self.subgoals = subgoals

        self._map: Map = MAP_REGISTRY[map_name]()
        self._robot = Robot(self._map.tile_to_pixel(*self._map.robot_start_tile))
        self._task_manager = TaskManager(goals, subgoals=subgoals)
        self._renderer = Renderer(self._map)
        self._steps = 0

        if action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(8)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            )

        h, w = obs_resolution
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._robot.reset()
        self._task_manager.reset(self._map, self.n_trash, self.np_random)
        self._steps = 0
        obs = self._get_obs()
        info = self._task_manager.get_info(self._robot if self.subgoals else None)
        info["carrying"] = self._robot.carrying
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._steps += 1

        if self.action_mode == "discrete":
            self._robot.move_discrete(int(action), self._map.solid, self._map.tile_size)
        else:
            self._robot.move_continuous(np.asarray(action, dtype=np.float32), self._map.solid, self._map.tile_size)

        reward = float(self._task_manager.step(self._robot, self._map))
        terminated = bool(self._task_manager.is_done())
        truncated = bool(self._steps >= self.max_steps)

        obs = self._get_obs()
        info = self._task_manager.get_info(self._robot if self.subgoals else None)
        info["carrying"] = self._robot.carrying
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        viewport = self._renderer.render(self._robot, self._task_manager)
        if self.render_mode == "human":
            self._renderer.show_in_window(viewport)
            return None
        return self._renderer.to_display(viewport)

    def close(self):
        self._renderer.close()

    def _get_obs(self) -> np.ndarray:
        viewport = self._renderer.render(self._robot, self._task_manager)
        if self.render_mode == "human":
            self._renderer.show_in_window(viewport)
        return self._renderer.to_obs(viewport, self.obs_resolution)
