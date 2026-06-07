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

    def move_discrete(self, action: int, tiles: np.ndarray, tile_size: int):
        if not (0 <= action < len(_DIRS)):
            raise ValueError(f"action must be in [0, {len(_DIRS) - 1}], got {action}")
        dx, dy = _DIRS[action]
        self.angle = math.atan2(dy, dx)
        speed = self.DISCRETE_SPEED / math.sqrt(2) if (dx != 0 and dy != 0) else self.DISCRETE_SPEED
        self._try_move(dx * speed, dy * speed, tiles, tile_size)

    def move_continuous(self, action: np.ndarray, tiles: np.ndarray, tile_size: int):
        linear, angular = float(action[0]), float(action[1])
        self.angle += angular * self.ANGULAR_SPEED
        dx = math.cos(self.angle) * linear * self.DISCRETE_SPEED
        dy = math.sin(self.angle) * linear * self.DISCRETE_SPEED
        self._try_move(dx, dy, tiles, tile_size)

    def _try_move(self, dx: float, dy: float, tiles: np.ndarray, tile_size: int):
        nx, ny = self.x + dx, self.y + dy
        if not self._collides(nx, ny, tiles, tile_size):
            self.x, self.y = nx, ny
        elif not self._collides(nx, self.y, tiles, tile_size):
            self.x = nx   # slide along x
        elif not self._collides(self.x, ny, tiles, tile_size):
            self.y = ny   # slide along y

    def _collides(self, x: float, y: float, tiles: np.ndarray, tile_size: int) -> bool:
        # 8-point circle probe: gap between probes is ~9px. Safe for tile_size>=16.
        for deg in range(0, 360, 45):
            rad = math.radians(deg)
            px = x + math.cos(rad) * self.RADIUS
            py = y + math.sin(rad) * self.RADIUS
            col = int(px // tile_size)
            row = int(py // tile_size)
            if row < 0 or row >= tiles.shape[0] or col < 0 or col >= tiles.shape[1]:
                return True
            if tiles[row, col] != 0:
                return True
        return False
