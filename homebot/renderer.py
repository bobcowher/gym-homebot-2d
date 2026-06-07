import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math
from typing import Optional
import numpy as np
import pygame

from homebot.maps import Map, FLOOR
from homebot.robot import Robot
from homebot.tasks import TaskManager

# Colors (RGB)
_FLOOR_COLOR    = (220, 210, 190)
_WALL_COLOR     = (60,  50,  40)
_FRIDGE_COLOR   = (100, 180, 255)
_RECLINER_COLOR = (200, 100, 80)
_DOOR_COLOR     = (180, 140, 80)
_TRASH_COLOR    = (150, 150, 150)
_DRINK_COLOR    = (100, 200, 150)
_PACKAGE_COLOR  = (220, 180, 80)

# Robot colors
_ROBOT_BODY_COLOR = (42,  42,  42)   # dark gray chassis
_WHEEL_COLOR      = (18,  18,  18)   # near-black rubber treads
_BUMPER_COLOR     = (55,  55,  55)   # slightly lighter front arc (direction cue)
_LIDAR_COLOR      = (22,  22,  22)   # near-black LiDAR dome
_LED_COLOR        = (100, 255, 60)   # neon green status LEDs

VIEWPORT_FACTOR = 0.6
_FALLBACK_DISPLAY_RES = (640, 512)  # used in headless / dummy-driver mode


def _auto_display_res(viewport_w: int, viewport_h: int) -> tuple[int, int]:
    """Scale viewport to fill ~85 % of the physical screen; fall back when headless."""
    info = pygame.display.Info()
    sw, sh = info.current_w, info.current_h
    if sw <= 0 or sh <= 0:
        return _FALLBACK_DISPLAY_RES
    scale = min(sw * 0.85 / viewport_w, sh * 0.85 / viewport_h)
    return (int(viewport_w * scale), int(viewport_h * scale))


class Renderer:
    def __init__(self, game_map: Map, display_res: Optional[tuple[int, int]] = None):
        pygame.init()
        self.map = game_map
        self._viewport_w = int(game_map.pixel_width * VIEWPORT_FACTOR)
        self._viewport_h = int(game_map.pixel_height * VIEWPORT_FACTOR)
        self.display_res = display_res or _auto_display_res(self._viewport_w, self._viewport_h)
        self._surface = pygame.Surface((game_map.pixel_width, game_map.pixel_height))
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        # Pre-bake static geometry (tiles + fixtures never change during an episode)
        self._static = pygame.Surface((game_map.pixel_width, game_map.pixel_height))
        self._draw_map_to(self._static)
        self._draw_fixtures_to(self._static)

    def render(self, robot: Robot, task_manager: TaskManager) -> pygame.Surface:
        """Draw frame to internal surface. Return viewport Surface centered on robot."""
        self._surface.blit(self._static, (0, 0))
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

    def _draw_map_to(self, surface: pygame.Surface):
        ts = self.map.tile_size
        for row in range(self.map.tiles.shape[0]):
            for col in range(self.map.tiles.shape[1]):
                color = _FLOOR_COLOR if self.map.tiles[row, col] == FLOOR else _WALL_COLOR
                pygame.draw.rect(surface, color, (col * ts, row * ts, ts, ts))

    def _draw_fixtures_to(self, surface: pygame.Surface):
        ts = self.map.tile_size
        color_map = {
            "fridge":   _FRIDGE_COLOR,
            "recliner": _RECLINER_COLOR,
            "door":     _DOOR_COLOR,
        }
        for name, (col, row) in self.map.fixtures.items():
            color = color_map.get(name, (200, 200, 200))
            pygame.draw.rect(surface, color, (col * ts + 4, row * ts + 4, ts - 8, ts - 8))

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
        R  = robot.RADIUS
        a  = robot.angle
        ca, sa = math.cos(a), math.sin(a)   # forward unit vector
        cp, sp = -sa,  ca                   # left-perpendicular unit vector

        # Wheels: centered on sides (real robot geometry), chunky treads
        whl = R          # half-length — wheels span full body length
        whw = 3          # half-width of tread
        for sign in (1, -1):
            wcx = cx + sign * cp * R
            wcy = cy + sign * sp * R
            pts = [
                (wcx + ca*whl + sign*cp*whw, wcy + sa*whl + sign*sp*whw),
                (wcx - ca*whl + sign*cp*whw, wcy - sa*whl + sign*sp*whw),
                (wcx - ca*whl - sign*cp*whw, wcy - sa*whl - sign*sp*whw),
                (wcx + ca*whl - sign*cp*whw, wcy + sa*whl - sign*sp*whw),
            ]
            pygame.draw.polygon(self._surface, _WHEEL_COLOR, pts)

        # Chassis body
        pygame.draw.circle(self._surface, (18, 18, 18), (cx, cy), R)      # edge shadow
        pygame.draw.circle(self._surface, _ROBOT_BODY_COLOR, (cx, cy), R - 1)

        # Subtle front arc — just enough lighter shade to read direction
        step = 5
        outer = [
            (cx + math.cos(a + math.radians(d)) * (R - 1),
             cy + math.sin(a + math.radians(d)) * (R - 1))
            for d in range(-70, 71, step)
        ]
        inner = [
            (cx + math.cos(a + math.radians(d)) * (R - 3),
             cy + math.sin(a + math.radians(d)) * (R - 3))
            for d in range(70, -71, -step)
        ]
        pygame.draw.polygon(self._surface, _BUMPER_COLOR, outer + inner)

        # LiDAR dome — dominant top feature, large dark circle
        lidar_r = max(3, R // 2)
        pygame.draw.circle(self._surface, (12, 12, 12), (cx, cy), lidar_r + 1)
        pygame.draw.circle(self._surface, _LIDAR_COLOR, (cx, cy), lidar_r)

        # Green status LEDs on each side (~45° forward from perpendicular)
        for sign in (1, -1):
            led_x = int(cx + ca * (R * 0.4) + sign * cp * (R - 3))
            led_y = int(cy + sa * (R * 0.4) + sign * sp * (R - 3))
            pygame.draw.circle(self._surface, _LED_COLOR, (led_x, led_y), 2)

        # Camera: small, at front — dark housing, blue tinted lens, glint
        cam_x = int(cx + ca * (R - 2))
        cam_y = int(cy + sa * (R - 2))
        pygame.draw.circle(self._surface, (20, 20, 20), (cam_x, cam_y), 2)
        pygame.draw.circle(self._surface, (30, 80, 160), (cam_x, cam_y), 1)
        pygame.draw.circle(self._surface, (180, 215, 255),
                           (int(cam_x - ca * 0.6), int(cam_y - sa * 0.6)), 1)

        # Carry indicator
        if robot.carrying == "drink":
            pygame.draw.circle(self._surface, _DRINK_COLOR, (cx, cy), 4)
        elif robot.carrying == "package":
            pygame.draw.rect(self._surface, _PACKAGE_COLOR, (cx - 4, cy - 4, 8, 8))
