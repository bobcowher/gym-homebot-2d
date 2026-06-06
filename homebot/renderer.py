from typing import Optional
import numpy as np
import pygame

from homebot.maps import Map, FLOOR
from homebot.robot import Robot
from homebot.tasks import TaskManager

# Colors (RGB)
_FLOOR_COLOR    = (220, 210, 190)
_WALL_COLOR     = (60,  50,  40)
_ROBOT_COLOR    = (50,  120, 220)
_FRIDGE_COLOR   = (100, 180, 255)
_RECLINER_COLOR = (200, 100, 80)
_DOOR_COLOR     = (180, 140, 80)
_TRASH_COLOR    = (150, 150, 150)
_DRINK_COLOR    = (100, 200, 150)
_PACKAGE_COLOR  = (220, 180, 80)

DISPLAY_RES = (640, 512)
VIEWPORT_FACTOR = 0.4


class Renderer:
    def __init__(self, map: Map, display_res: tuple[int, int] = DISPLAY_RES):
        pygame.init()
        self.map = map
        self.display_res = display_res
        self._viewport_w = int(map.pixel_width * VIEWPORT_FACTOR)
        self._viewport_h = int(map.pixel_height * VIEWPORT_FACTOR)
        self._surface = pygame.Surface((map.pixel_width, map.pixel_height))
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

    def render(self, robot: Robot, task_manager: TaskManager) -> pygame.Surface:
        """Draw frame to internal surface. Return viewport Surface centered on robot."""
        self._draw_map()
        self._draw_fixtures(task_manager)
        self._draw_items(task_manager)
        self._draw_robot(robot)
        return self._extract_viewport(robot)

    def to_obs(self, viewport: pygame.Surface, obs_resolution: tuple[int, int]) -> np.ndarray:
        """Downscale viewport to obs_resolution. Returns (H, W, 3) uint8 array."""
        h, w = obs_resolution
        scaled = pygame.transform.scale(viewport, (w, h))
        return pygame.surfarray.array3d(scaled).transpose(1, 0, 2)

    def to_display(self, viewport: pygame.Surface) -> np.ndarray:
        """Scale viewport to display_res. Returns (H, W, 3) uint8 array."""
        scaled = pygame.transform.scale(viewport, self.display_res)
        return pygame.surfarray.array3d(scaled).transpose(1, 0, 2)

    def show_in_window(self, viewport: pygame.Surface):
        """Blit scaled viewport to Pygame window. Opens window on first call."""
        if self._window is None:
            self._window = pygame.display.set_mode(self.display_res)
            pygame.display.set_caption("HomeBot")
            self._clock = pygame.time.Clock()
        scaled = pygame.transform.scale(viewport, self.display_res)
        self._window.blit(scaled, (0, 0))
        pygame.display.flip()
        self._clock.tick(60)

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            self._window = None
            self._clock = None

    # --- private drawing methods ---

    def _extract_viewport(self, robot: Robot) -> pygame.Surface:
        vx = int(robot.x - self._viewport_w / 2)
        vy = int(robot.y - self._viewport_h / 2)
        vx = max(0, min(vx, self.map.pixel_width  - self._viewport_w))
        vy = max(0, min(vy, self.map.pixel_height - self._viewport_h))
        vp = pygame.Surface((self._viewport_w, self._viewport_h))
        vp.blit(self._surface, (0, 0), (vx, vy, self._viewport_w, self._viewport_h))
        return vp

    def _draw_map(self):
        ts = self.map.tile_size
        for row in range(self.map.tiles.shape[0]):
            for col in range(self.map.tiles.shape[1]):
                color = _FLOOR_COLOR if self.map.tiles[row, col] == FLOOR else _WALL_COLOR
                pygame.draw.rect(self._surface, color, (col * ts, row * ts, ts, ts))

    def _draw_fixtures(self, task_manager: TaskManager):
        ts = self.map.tile_size
        color_map = {
            "fridge":   _FRIDGE_COLOR,
            "recliner": _RECLINER_COLOR,
            "door":     _DOOR_COLOR,
        }
        for name, (col, row) in self.map.fixtures.items():
            color = color_map.get(name, (200, 200, 200))
            pygame.draw.rect(
                self._surface, color,
                (col * ts + 4, row * ts + 4, ts - 8, ts - 8),
            )

    def _draw_items(self, task_manager: TaskManager):
        ts = self.map.tile_size
        for col, row in task_manager.trash_positions:
            px = col * ts + ts // 2
            py = row * ts + ts // 2
            pygame.draw.circle(self._surface, _TRASH_COLOR, (px, py), 6)
        if task_manager.package_present:
            col, row = self.map.fixtures["door"]
            px = col * ts + ts // 2
            py = row * ts + ts // 2
            pygame.draw.rect(self._surface, _PACKAGE_COLOR, (px - 8, py - 8, 16, 16))

    def _draw_robot(self, robot: Robot):
        cx, cy = int(robot.x), int(robot.y)
        pygame.draw.circle(self._surface, _ROBOT_COLOR, (cx, cy), robot.RADIUS)
        if robot.carrying == "drink":
            pygame.draw.circle(self._surface, _DRINK_COLOR, (cx, cy), 5)
        elif robot.carrying == "package":
            pygame.draw.rect(self._surface, _PACKAGE_COLOR, (cx - 5, cy - 5, 10, 10))
