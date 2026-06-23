import math
from typing import Optional
import numpy as np
import gymnasium as gym

from homebot.maps import MAP_REGISTRY, Map
from homebot.robot import Robot
from homebot.tasks import TaskManager
from homebot.renderer import Renderer
from homebot.goals import goal_to_coordinates as _goal_to_coordinates

# Minimum spawn distance from any goal location: 2 × robot body diameter.
_RANDOM_START_CLEARANCE = 4 * Robot.RADIUS  # 4 × 15 = 60 px


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
        random_start: bool,
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
        self.random_start = random_start
        self._steps = 0

    def _sample_start_pos(self) -> tuple[float, float]:
        """Return a random valid spawn pixel (x, y) for the robot.

        Guarantees: on a walkable floor tile, robot circle fits without collision,
        and at least _RANDOM_START_CLEARANCE px from every goal location (fixtures
        fridge/recliner/door plus current trash tile centres).
        Falls back to the default start tile if no candidate passes all filters.
        """
        ts = self._map.tile_size

        goal_pixels: list[tuple[float, float]] = []
        for col, row in self._task_manager.trash_positions:
            px, py = self._map.tile_to_pixel(col, row)
            goal_pixels.append((float(px), float(py)))
        for fname in ("fridge", "recliner", "door"):
            if fname in self._map.fixtures:
                px, py = self._map.tile_to_pixel(*self._map.fixtures[fname])
                goal_pixels.append((float(px), float(py)))

        candidates = self._map.valid_floor_tiles()
        order = self.np_random.permutation(len(candidates))  # type: ignore[attr-defined]

        for idx in order:
            col, row = candidates[int(idx)]
            px, py = self._map.tile_to_pixel(col, row)
            fpx, fpy = float(px), float(py)
            if any(
                math.sqrt((fpx - gx) ** 2 + (fpy - gy) ** 2) < _RANDOM_START_CLEARANCE
                for gx, gy in goal_pixels
            ):
                continue
            if not self._robot._collides(fpx, fpy, self._map.wall_solid, ts,
                                          self._map.fixture_pixel_rects):
                return fpx, fpy

        # Fallback — should be very rare given map size vs clearance
        px, py = self._map.tile_to_pixel(*self._map.robot_start_tile)
        return float(px), float(py)

    def goal_to_coordinates(self, goal_name: str) -> tuple[float, float]:
        return _goal_to_coordinates(
            goal_name, self._map, self._task_manager.trash_positions,
            self.np_random,  # type: ignore[attr-defined]
        )

    def _get_obs(self) -> np.ndarray:
        viewport = self._renderer.render(self._robot, self._task_manager, self._steps)
        if self.render_mode == "human":
            self._renderer.show_in_window(viewport)
        return self._renderer.to_obs(viewport, self.obs_resolution)

    def render(self) -> Optional[np.ndarray]:
        viewport = self._renderer.render(self._robot, self._task_manager, self._steps)
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
        random_start: bool = False,
    ):
        super().__init__()
        self._init_core(
            goals or ["trash", "drink", "package"],
            action_mode, obs_resolution, max_steps, render_mode, n_trash, map_name,
            random_start,
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
        if self.random_start:
            self._robot.x, self._robot.y = self._sample_start_pos()
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
    goal_to_coordinates(name) converts any goal name to pixel (x, y) for external orchestrators.
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
        random_start: bool = False,
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
            random_start,
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

        if self.random_start:
            self._robot.x, self._robot.y = self._sample_start_pos()

        from homebot.goals import GOAL_REGISTRY, goal_to_coordinates
        _, initial_carry = GOAL_REGISTRY[goal_name]
        if not self.evaluate and initial_carry is not None:
            self._robot.carrying = initial_carry

        # Fix desired_goal for the full episode — stable even after the item is picked up.
        gx, gy = goal_to_coordinates(
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

        # Real per-target reward straight from the task manager (trash 31px /
        # door 47px / fixture 79px — see tasks.py). Terminate when the active goal's
        # real reward FIRES (single active goal per episode), not on a geometric
        # radius and not on global is_done() (which wants every registered goal).
        # compute_reward is now ONLY the HER hindsight relabel proxy (synthetic
        # goals); real transitions learn from this true reward and HER fills the rest.
        reward = float(self._task_manager.step(self._robot))
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
        """HER hindsight relabel proxy ONLY: sparse 0/1, 1.0 within RELABEL_RADIUS
        of desired_goal. Real transitions use the TaskManager's true reward (see
        step); this geometric stand-in exists solely because relabeled goals are
        arbitrary coords the task mechanics can't score. Handles batched inputs
        (HER passes arrays of goals)."""
        from homebot.goals import RELABEL_RADIUS
        diff = np.asarray(achieved_goal, dtype=np.float32) - np.asarray(desired_goal, dtype=np.float32)
        dist = np.linalg.norm(diff, axis=-1)
        return (dist <= RELABEL_RADIUS).astype(np.float32)

    def _build_obs(self) -> dict:
        return {
            "observation":   self._get_obs(),
            "achieved_goal": np.array([self._robot.x, self._robot.y], dtype=np.float32),
            "desired_goal":  self._desired_goal.copy(),
        }
