import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math
from typing import Optional
import numpy as np
import pygame

from homebot.maps import Map, FLOOR, WALL, LAWN
from homebot.robot import Robot
from homebot.tasks import TaskManager
from homebot.sprites import make_sprite, RECLINER, TV, FRIDGE, COUNTER, SINK, TABLE, PACKAGE, BOTTLE, CAN

# Environment colors (RGB) — warm, cozy theme
_FLOOR_COLOR = (220, 210, 190)   # cream floor
_WALL_COLOR  = (84, 68, 62)      # warm mocha wall
_LAWN_COLOR  = (104, 150, 68)    # grass outside
_STOOP_COLOR = (170, 158, 134)   # doormat / threshold

# Robot colors (flat, to match the chunky palette)
_ROBOT_BODY  = (62, 66, 86)      # slate-navy chassis
_ROBOT_EDGE  = (30, 32, 46)      # dark rim
_ROBOT_FRONT = (108, 114, 138)   # lighter front wedge (direction cue)
_WHEEL_COLOR = (28, 26, 34)      # near-black tread
_LED_COLOR   = (110, 230, 90)    # status LEDs
_CAMERA_DARK = (24, 24, 30)
_CAMERA_LENS = (54, 130, 200)
_DRINK_COLOR   = (100, 200, 150)
_PACKAGE_COLOR = (198, 152, 98)

_WALL_INSET = 8  # px shaved off wall edges that face a non-wall tile (thinner walls)

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
    def __init__(self, game_map: Map, display_res: Optional[tuple[int, int]] = None,
                 headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        self.map = game_map
        self._viewport_w = int(game_map.pixel_width * VIEWPORT_FACTOR)
        self._viewport_h = int(game_map.pixel_height * VIEWPORT_FACTOR)
        self.display_res = display_res or _auto_display_res(self._viewport_w, self._viewport_h)
        self._surface = pygame.Surface((game_map.pixel_width, game_map.pixel_height))
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

        # All sprites built once from character grids.
        self._recliner_sprite = make_sprite(RECLINER)
        self._tv_sprite = make_sprite(TV)
        self._fridge_sprite = make_sprite(FRIDGE)
        self._counter_sprite = make_sprite(COUNTER)
        self._sink_sprite = make_sprite(SINK)
        self._table_sprite = make_sprite(TABLE)
        self._package_sprite = make_sprite(PACKAGE)
        self._bottle_sprite = make_sprite(BOTTLE)
        self._can_sprite = make_sprite(CAN)

        # Pre-bake static geometry: tiles + non-animated fixtures.
        # Recliner is excluded here — it has a gentle rock animation and is drawn each frame.
        self._static = pygame.Surface((game_map.pixel_width, game_map.pixel_height))
        self._draw_map_to(self._static)
        self._blit_sprite_at_fixture(self._static, self._tv_sprite, "tv")
        self._blit_sprite_at_fixture(self._static, self._counter_sprite, "counter")
        self._blit_sprite_at_fixture(self._static, self._sink_sprite, "sink")
        self._blit_sprite_at_fixture(self._static, self._fridge_sprite, "fridge")
        self._blit_sprite_at_fixture(self._static, self._table_sprite, "table")

    def render(self, robot: Robot, task_manager: TaskManager, step: int = 0) -> pygame.Surface:
        """Draw frame to internal surface. Return viewport Surface centered on robot."""
        self._surface.blit(self._static, (0, 0))
        self._draw_recliner(step)
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
        assert self._clock is not None
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
        tiles = self.map.tiles
        rows, cols = tiles.shape
        door_set = set(self.map.door_tiles)
        surface.fill(_FLOOR_COLOR)  # floor under everything (incl. fixture footprints)

        def neighbor(r: int, c: int) -> int:
            if 0 <= r < rows and 0 <= c < cols:
                return int(tiles[r, c])
            return WALL  # treat off-map as wall so border walls stay full thickness

        for row in range(rows):
            for col in range(cols):
                t = int(tiles[row, col])
                x, y = col * ts, row * ts
                if t == LAWN:
                    pygame.draw.rect(surface, _LAWN_COLOR, (x, y, ts, ts))
                elif t == WALL:
                    # Thinner walls: shave edges that face a non-wall tile, leaving
                    # wall-to-wall junctions flush (no gaps).
                    rx, ry, rw, rh = x, y, ts, ts
                    if neighbor(row - 1, col) != WALL:
                        ry += _WALL_INSET; rh -= _WALL_INSET
                    if neighbor(row + 1, col) != WALL:
                        rh -= _WALL_INSET
                    if neighbor(row, col - 1) != WALL:
                        rx += _WALL_INSET; rw -= _WALL_INSET
                    if neighbor(row, col + 1) != WALL:
                        rw -= _WALL_INSET
                    pygame.draw.rect(surface, _WALL_COLOR, (rx, ry, rw, rh))
                elif (col, row) in door_set:
                    pygame.draw.rect(surface, _STOOP_COLOR, (x, y, ts, ts))

    def _blit_sprite_at_fixture(self, surface: pygame.Surface,
                                sprite: pygame.Surface, fixture: str, dy: int = 0):
        """Blit a sprite centered on the named fixture's tile (optional y offset)."""
        ts = self.map.tile_size
        col, row = self.map.fixtures[fixture]
        cx = col * ts + ts // 2
        cy = row * ts + ts // 2
        w, h = sprite.get_size()
        surface.blit(sprite, (cx - w // 2, cy - h // 2 + dy))

    def _draw_recliner(self, step: int = 0):
        # Gentle rocking: ±1px, driven by step count for determinism
        rock = round(math.sin(step / 60.0 * 1.5))
        self._blit_sprite_at_fixture(self._surface, self._recliner_sprite, "recliner", dy=rock)

    def _draw_items(self, task_manager: TaskManager):
        ts = self.map.tile_size
        for col, row in task_manager.trash_positions:
            # deterministic sprite variety per tile position — bottles and cans mixed without randomness
            sprite = self._bottle_sprite if (col * 7 + row * 13) % 2 == 0 else self._can_sprite
            sw, sh = sprite.get_size()
            px = col * ts + ts // 2
            py = row * ts + ts // 2
            self._surface.blit(sprite, (px - sw // 2, py - sh // 2))
        if task_manager.package_present:
            self._blit_sprite_at_fixture(self._surface, self._package_sprite, "door")

    def _draw_robot(self, robot: Robot):
        cx, cy = int(robot.x), int(robot.y)
        R  = robot.RADIUS
        a  = robot.angle
        ca, sa = math.cos(a), math.sin(a)   # forward unit vector
        cp, sp = -sa,  ca                   # left-perpendicular unit vector

        # Wheels: kept INSIDE the body radius so they never poke into walls.
        fwd_off = R * 0.25
        whl = 5              # half-length (along forward)
        whw = 3              # half-width of tread
        perp = R - whw       # so outer tread edge lands exactly at radius R
        for sign in (1, -1):
            wcx = cx + ca * fwd_off + sign * cp * perp
            wcy = cy + sa * fwd_off + sign * sp * perp
            pts = [
                (wcx + ca*whl + sign*cp*whw, wcy + sa*whl + sign*sp*whw),
                (wcx - ca*whl + sign*cp*whw, wcy - sa*whl + sign*sp*whw),
                (wcx - ca*whl - sign*cp*whw, wcy - sa*whl - sign*sp*whw),
                (wcx + ca*whl - sign*cp*whw, wcy + sa*whl - sign*sp*whw),
            ]
            pygame.draw.polygon(self._surface, _WHEEL_COLOR, pts)

        # Flat chassis
        pygame.draw.circle(self._surface, _ROBOT_EDGE, (cx, cy), R)
        pygame.draw.circle(self._surface, _ROBOT_BODY, (cx, cy), R - 2)

        # Direction wedge — flat lighter triangle pointing forward
        tip   = (cx + ca * (R - 1),         cy + sa * (R - 1))
        baseL = (cx + ca * (R * 0.15) + cp * 7, cy + sa * (R * 0.15) + sp * 7)
        baseR = (cx + ca * (R * 0.15) - cp * 7, cy + sa * (R * 0.15) - sp * 7)
        pygame.draw.polygon(self._surface, _ROBOT_FRONT, [tip, baseL, baseR])

        # Green status LEDs on each side toward the front
        for sign in (1, -1):
            led_x = int(cx + ca * (R * 0.45) + sign * cp * (R - 4))
            led_y = int(cy + sa * (R * 0.45) + sign * sp * (R - 4))
            pygame.draw.circle(self._surface, _LED_COLOR, (led_x, led_y), 2)

        # Camera at the front tip
        cam_x = int(cx + ca * (R - 4))
        cam_y = int(cy + sa * (R - 4))
        pygame.draw.circle(self._surface, _CAMERA_DARK, (cam_x, cam_y), 3)
        pygame.draw.circle(self._surface, _CAMERA_LENS, (cam_x, cam_y), 2)

        # Carry indicator
        if robot.carrying == "drink":
            pygame.draw.circle(self._surface, _DRINK_COLOR, (cx, cy), 4)
        elif robot.carrying == "package":
            pygame.draw.rect(self._surface, _PACKAGE_COLOR, (cx - 4, cy - 4, 8, 8))
