import numpy as np
from typing import Optional, Type

from homebot.sprites import SPRITE_SIZES

FLOOR = 0
WALL = 1
LAWN = 2   # outside ground: rendered green, solid (robot cannot enter)


class Map:
    """Base class for HomeBot maps. Subclasses must set tile_size, tiles, fixtures,
    robot_start_tile in __init__, then call self._finalize() to build derived data
    (solid-collision mask, door tiles)."""

    tile_size: int
    tiles: np.ndarray        # shape (rows, cols): FLOOR / WALL / LAWN
    solid: np.ndarray        # bool (rows, cols): walls + lawn + fixture footprints (for item spawn)
    wall_solid: np.ndarray   # bool (rows, cols): walls + lawn only (for robot tile collision)
    fixture_pixel_rects: dict  # {name: (left, top, right, bottom)} pixel-accurate fixture bounds
    fixtures: dict           # {"fridge": (col, row), "tv": (col, row), ...}
    robot_start_tile: tuple  # (col, row)
    door_tiles: list         # FLOOR tiles rendered as a stoop threshold
    _floor_tiles: list       # cached result of valid_floor_tiles(); set by _finalize()
    solid_fixtures: dict = {}       # {name: (w_tiles, h_tiles)} tile footprints for item spawn
    fixture_pixel_sizes: dict = {}  # {name: (px_w, px_h)} derived from sprites; override in subclass

    def _finalize(self):
        """Build solid masks, pixel-accurate fixture rects, and floor tile cache."""
        # wall_solid: walls + lawn only — used for robot tile collision
        self.wall_solid = self.tiles != FLOOR

        # solid: wall_solid + fixture tile footprints — used for item spawn exclusion
        self.solid = self.wall_solid.copy()
        for name, (w, h) in self.solid_fixtures.items():
            col, row = self.fixtures[name]
            for r in range(row - h // 2, row - h // 2 + h):
                for c in range(col - w // 2, col - w // 2 + w):
                    if 0 <= r < self.tiles.shape[0] and 0 <= c < self.tiles.shape[1]:
                        self.solid[r, c] = True

        # fixture_pixel_rects: sprite-accurate bounding boxes for robot collision
        self.fixture_pixel_rects = {}
        for name, (pw, ph) in self.fixture_pixel_sizes.items():
            if name in self.fixtures:
                col, row = self.fixtures[name]
                cx = col * self.tile_size + self.tile_size // 2
                cy = row * self.tile_size + self.tile_size // 2
                self.fixture_pixel_rects[name] = (cx - pw // 2, cy - ph // 2,
                                                   cx + pw // 2, cy + ph // 2)

        # Cache reachable floor tiles — map is immutable after _finalize()
        mask = (self.tiles == FLOOR) & (~self.solid)
        rows, cols_arr = np.where(mask)
        self._floor_tiles = list(zip(cols_arr.tolist(), rows.tolist()))

    def valid_floor_tiles(self) -> list[tuple[int, int]]:
        """Reachable floor tiles (FLOOR and not solid) as (col, row)."""
        return list(self._floor_tiles)  # copy so callers can't mutate the cache

    def spawn_trash(
        self,
        n: int,
        rng: np.random.Generator,
        exclude: Optional[list[tuple]] = None,
    ) -> list[tuple[int, int]]:
        candidates = self.valid_floor_tiles()
        if exclude:
            exclude_set = set(exclude)
            candidates = [t for t in candidates if t not in exclude_set]
        n = min(n, len(candidates))
        idxs = rng.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in idxs]

    def tile_to_pixel(self, col: int, row: int) -> tuple[int, int]:
        return (
            col * self.tile_size + self.tile_size // 2,
            row * self.tile_size + self.tile_size // 2,
        )

    @property
    def pixel_width(self) -> int:
        return self.tiles.shape[1] * self.tile_size

    @property
    def pixel_height(self) -> int:
        return self.tiles.shape[0] * self.tile_size


class DefaultHouseMap(Map):
    """
    27 cols x 18 rows, tile_size=32px -> 864x576 pixel full map.

    Layout (cols 1-22 are the house interior):
      Rows 1-7:  Living room (cols 1-10) | Kitchen (cols 12-22)
      Row 8:     Wall with doorways into hallway
      Rows 9-10: Hallway (cols 1-22), east doorway at col 23 opens outside
      Row 11:    Wall with doorway into bedroom
      Rows 12-16: Bedroom (cols 1-22)
      Cols 24-26: Outside lawn (solid, green) — visible through the doorway
    """

    tile_size = 32

    # Tile footprints for item spawn exclusion (w x h tiles, centered on fixture).
    solid_fixtures = {
        "recliner": (3, 2),
        "tv":       (3, 2),
        "fridge":   (2, 2),
        "sink":     (2, 2),
        "counter":  (2, 2),
        "table":    (1, 1),
    }

    # Derived from sprite grids in sprites.py — stays in sync automatically.
    fixture_pixel_sizes = SPRITE_SIZES

    def __init__(self):
        COLS, ROWS = 27, 18
        t = np.ones((ROWS, COLS), dtype=np.uint8) * WALL

        # Living room
        t[1:8, 1:11] = FLOOR
        # Kitchen
        t[1:8, 12:23] = FLOOR
        # Doorway between living room and kitchen (vertical wall col 11)
        t[4:6, 11] = FLOOR
        # Hallway
        t[9:11, 1:23] = FLOOR
        # Doorway: living room -> hallway
        t[8, 4:7] = FLOOR
        # Doorway: kitchen -> hallway
        t[8, 14:17] = FLOOR
        # Doorway: hallway -> bedroom
        t[11, 10:13] = FLOOR
        # Bedroom
        t[12:17, 1:23] = FLOOR
        # East doorway: opening through the house wall (col 23) to outside
        t[9:11, 23] = FLOOR
        # Outside lawn (cols 24-26), solid — robot can see it but not enter
        t[:, 24:27] = LAWN

        self.tiles = t
        self.fixtures = {
            "fridge":   (19, 1),   # kitchen, pressed against the north wall
            "sink":     (15, 1),   # kitchen, north wall
            "counter":  (17, 1),   # kitchen, north wall between sink and fridge
            "tv":       (5,  1),   # living room, pressed against the top wall
            "recliner": (5,  5),   # living room, below TV — man faces up at it
            "table":    (3,  5),   # living room, side table to man's left (west)
            "door":     (23, 9),   # east doorway threshold (package sits here)
        }
        self.door_tiles = [(23, 9), (23, 10)]  # stoop threshold rendering
        self.robot_start_tile = (8, 4)         # living room, clear of furniture
        self._finalize()


MAP_REGISTRY: dict[str, Type[Map]] = {
    "default": DefaultHouseMap,
}
