from typing import Optional
import numpy as np
import gymnasium as gym

from homebot.maps import MAP_REGISTRY, Map
from homebot.robot import Robot
from homebot.tasks import TaskManager
from homebot.renderer import Renderer
from homebot.goals import goal_to_coords as _goal_to_coords


class _HomeBotCore:
    """Shared init and methods for HomeBotEnv and HomeBotGoalEnv.
    Not a standalone env — call _init_core() from the subclass __init__."""

    def _init_core(
        self,
        goals: list[str],
        action_mode: str,
        obs_resolution: tuple[int, int],
        max_steps: int,
        render_mode: Optional[str],
        n_trash: int,
        map_name: str,
    ):
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode must be 'discrete' or 'continuous', got {action_mode!r}")
        self._map: Map = MAP_REGISTRY[map_name]()
        self._robot = Robot(self._map.tile_to_pixel(*self._map.robot_start_tile))
        self._task_manager = TaskManager(goals)
        self._renderer = Renderer(self._map, headless=(render_mode != "human"))
        self.action_mode = action_mode
        self.obs_resolution = obs_resolution
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.n_trash = n_trash
        self._steps = 0

    def goal_to_coords(self, goal_name: str) -> tuple[float, float]:
        return _goal_to_coords(
            goal_name, self._map, self._task_manager.trash_positions,
            self.np_random,  # type: ignore[attr-defined]
        )

    def _get_obs(self) -> np.ndarray:
        viewport = self._renderer.render(self._robot, self._task_manager)
        if self.render_mode == "human":
            self._renderer.show_in_window(viewport)
        return self._renderer.to_obs(viewport, self.obs_resolution)

    def render(self) -> Optional[np.ndarray]:
        viewport = self._renderer.render(self._robot, self._task_manager)
        if self.render_mode == "human":
            self._renderer.show_in_window(viewport)
            return None
        return self._renderer.to_display(viewport)

    def close(self):
        self._renderer.close()


class HomeBotEnv(_HomeBotCore, gym.Env):
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
    ):
        super().__init__()
        self._init_core(
            goals or ["trash", "drink", "package"],
            action_mode, obs_resolution, max_steps, render_mode, n_trash, map_name,
        )
        if action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(8)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1., -1.], dtype=np.float32),
                high=np.array([1.,  1.], dtype=np.float32),
                dtype=np.float32,
            )
        h, w = obs_resolution
        self.observation_space = gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._robot.reset()
        self._task_manager.reset(self._map, self.n_trash, self.np_random)
        self._steps = 0
        obs = self._get_obs()
        info = self._task_manager.get_info(self._robot)
        info["carrying"] = self._robot.carrying
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._steps += 1
        if self.action_mode == "discrete":
            self._robot.move_discrete(int(action), self._map.wall_solid,
                                      self._map.tile_size, self._map.fixture_pixel_rects)
        else:
            self._robot.move_continuous(np.asarray(action, dtype=np.float32),
                                        self._map.wall_solid, self._map.tile_size,
                                        self._map.fixture_pixel_rects)
        reward = float(self._task_manager.step(self._robot))
        terminated = bool(self._task_manager.is_done())
        truncated = bool(self._steps >= self.max_steps)
        obs = self._get_obs()
        info = self._task_manager.get_info(self._robot)
        info["carrying"] = self._robot.carrying
        return obs, reward, terminated, truncated, info


class HomeBotGoalEnv(_HomeBotCore, gym.Env):
    """GoalEnv for HER training. Single active goal per episode, Dict observation space.

    reset(options={"goal": "random"|goal_name}) selects the episode goal.
    In training mode (evaluate=False), carry state is pre-loaded when the goal requires it.
    goal_to_coords(name) converts any goal name to pixel (x, y) for external orchestrators.
    """
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
        evaluate: bool = False,
    ):
        super().__init__()
        from homebot.goals import GOAL_NAMES
        self._available_goals: list[str] = goals or list(GOAL_NAMES)
        self.evaluate = evaluate
        # TaskManager always handles all goal types so carry state updates work
        # regardless of which sub-goal is currently active.
        self._init_core(
            ["trash", "drink", "package"],
            action_mode, obs_resolution, max_steps, render_mode, n_trash, map_name,
        )
        if action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(8)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1., -1.], dtype=np.float32),
                high=np.array([1.,  1.], dtype=np.float32),
                dtype=np.float32,
            )
        h, w = obs_resolution
        self.observation_space = gym.spaces.Dict({
            "observation":   gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
            "achieved_goal": gym.spaces.Box(0.0, np.inf, (2,), dtype=np.float32),
            "desired_goal":  gym.spaces.Box(0.0, np.inf, (2,), dtype=np.float32),
        })
        self._active_goal: str = self._available_goals[0]
        self._desired_goal = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        goal_name = (options or {}).get("goal", "random")
        if goal_name == "random":
            idx = int(self.np_random.integers(0, len(self._available_goals)))  # type: ignore[attr-defined]
            goal_name = self._available_goals[idx]
        self._active_goal = goal_name

        self._robot.reset()
        self._task_manager.reset(self._map, self.n_trash, self.np_random)  # type: ignore[attr-defined]
        self._steps = 0

        from homebot.goals import GOAL_REGISTRY, goal_to_coords
        _, initial_carry = GOAL_REGISTRY[goal_name]
        if not self.evaluate and initial_carry is not None:
            self._robot.carrying = initial_carry

        # Fix desired_goal for the full episode — stable even after the item is picked up.
        gx, gy = goal_to_coords(
            goal_name, self._map, self._task_manager.trash_positions,
            self.np_random,  # type: ignore[attr-defined]
        )
        self._desired_goal = np.array([gx, gy], dtype=np.float32)

        obs = self._build_obs()
        info = {"carrying": self._robot.carrying, "active_goal": self._active_goal}
        return obs, info

    def step(self, action):
        self._steps += 1
        if self.action_mode == "discrete":
            self._robot.move_discrete(int(action), self._map.wall_solid,
                                      self._map.tile_size, self._map.fixture_pixel_rects)
        else:
            self._robot.move_continuous(np.asarray(action, dtype=np.float32),
                                        self._map.wall_solid, self._map.tile_size,
                                        self._map.fixture_pixel_rects)

        # Run task manager to update carry state and remove collected items.
        # Its reward signal is ignored — reward comes from compute_reward.
        self._task_manager.step(self._robot)

        achieved = np.array([self._robot.x, self._robot.y], dtype=np.float32)
        reward = float(self.compute_reward(achieved, self._desired_goal, {}))
        terminated = reward > 0.5
        truncated = self._steps >= self.max_steps

        obs = self._build_obs()
        info = {"carrying": self._robot.carrying, "active_goal": self._active_goal}
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> np.ndarray:
        """Sparse 0/1: 1.0 if within GOAL_THRESHOLD of desired_goal.
        Handles batched inputs (HER relabeling passes arrays of goals)."""
        from homebot.goals import GOAL_THRESHOLD
        diff = np.asarray(achieved_goal, dtype=np.float32) - np.asarray(desired_goal, dtype=np.float32)
        dist = np.linalg.norm(diff, axis=-1)
        return (dist <= GOAL_THRESHOLD).astype(np.float32)

    def _build_obs(self) -> dict:
        return {
            "observation":   self._get_obs(),
            "achieved_goal": np.array([self._robot.x, self._robot.y], dtype=np.float32),
            "desired_goal":  self._desired_goal.copy(),
        }
