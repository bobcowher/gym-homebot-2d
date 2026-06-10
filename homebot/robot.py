import math
from typing import Optional
import numpy as np

# 8 directions: N, NE, E, SE, S, SW, W, NW
_DIRS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0,  1), (-1, 1), (-1, 0), (-1, -1),
]


class Robot:
    RADIUS = 15        # pixels
    DISCRETE_SPEED = 4 # pixels per step
    ANGULAR_SPEED = 0.08  # radians per step (continuous)

    def __init__(self, start_pos: tuple[float, float]):
        self._start_pos = start_pos
        self.x: float = start_pos[0]
        self.y: float = start_pos[1]
        self.angle: float = 0.0
        self.carrying: Optional[str] = None

    def reset(self):
        self.x, self.y = self._start_pos
        self.angle = 0.0
        self.carrying = None

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    def move_discrete(self, action: int, solid: np.ndarray, tile_size: int,
                      fixture_rects: Optional[dict] = None):
        if not (0 <= action < len(_DIRS)):
            raise ValueError(f"action must be in [0, {len(_DIRS) - 1}], got {action}")
        dx, dy = _DIRS[action]
        self.angle = math.atan2(dy, dx)
        speed = self.DISCRETE_SPEED / math.sqrt(2) if (dx != 0 and dy != 0) else self.DISCRETE_SPEED
        self._try_move(dx * speed, dy * speed, solid, tile_size, fixture_rects)

    def move_continuous(self, action: np.ndarray, solid: np.ndarray, tile_size: int,
                        fixture_rects: Optional[dict] = None):
        linear, angular = float(action[0]), float(action[1])
        self.angle += angular * self.ANGULAR_SPEED
        dx = math.cos(self.angle) * linear * self.DISCRETE_SPEED
        dy = math.sin(self.angle) * linear * self.DISCRETE_SPEED
        self._try_move(dx, dy, solid, tile_size, fixture_rects)

    def _try_move(self, dx: float, dy: float, solid: np.ndarray, tile_size: int,
                  fixture_rects: Optional[dict] = None):
        nx, ny = self.x + dx, self.y + dy
        if not self._collides(nx, ny, solid, tile_size, fixture_rects):
            self.x, self.y = nx, ny
        elif not self._collides(nx, self.y, solid, tile_size, fixture_rects):
            self.x = nx   # slide along x
        elif not self._collides(self.x, ny, solid, tile_size, fixture_rects):
            self.y = ny   # slide along y

    def _collides(self, x: float, y: float, solid: np.ndarray, tile_size: int,
                  fixture_rects: Optional[dict] = None) -> bool:
        # 8-point circle probe against tile solid mask (walls + lawn)
        for deg in range(0, 360, 45):
            rad = math.radians(deg)
            px = x + math.cos(rad) * self.RADIUS
            py = y + math.sin(rad) * self.RADIUS
            col = int(px // tile_size)
            row = int(py // tile_size)
            if row < 0 or row >= solid.shape[0] or col < 0 or col >= solid.shape[1]:
                return True
            if solid[row, col]:
                return True

        # Pixel-accurate circle-rect check against fixture bounding boxes
        if fixture_rects:
            r2 = self.RADIUS * self.RADIUS
            for (left, top, right, bottom) in fixture_rects.values():
                clamp_x = max(left, min(x, right))
                clamp_y = max(top, min(y, bottom))
                if (x - clamp_x) ** 2 + (y - clamp_y) ** 2 < r2:
                    return True

        return False
