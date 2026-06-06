import numpy as np
from typing import Optional, Type

FLOOR = 0
WALL = 1


class Map:
    """Base class for HomeBot maps. Subclasses must set tile_size, tiles, fixtures, robot_start_tile in __init__."""

    tile_size: int
    tiles: np.ndarray        # shape (rows, cols)
    fixtures: dict           # {"fridge": (col, row), "recliner": (col, row), "door": (col, row)}
    robot_start_tile: tuple  # (col, row)

    def valid_floor_tiles(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.tiles == FLOOR)
        return list(zip(cols.tolist(), rows.tolist()))  # (col, row)

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
    24 cols x 18 rows, tile_size=32px → 768x576 pixel full map.
    Viewport at 40% ≈ 307x230px.

    Layout:
      Rows 1-7:  Living room (cols 1-10) | Kitchen (cols 12-22)
      Row 8:     Wall with doorways into hallway
      Rows 9-10: Hallway (cols 1-22), exterior door at col 23
      Row 11:    Wall with doorway into bedroom
      Rows 12-16: Bedroom (cols 1-22)
    """

    tile_size = 32

    def __init__(self):
        COLS, ROWS = 24, 18
        t = np.ones((ROWS, COLS), dtype=np.uint8)

        # Living room
        t[1:8, 1:11] = FLOOR
        # Kitchen
        t[1:8, 12:23] = FLOOR
        # Doorway between living room and kitchen (vertical wall col 11)
        t[4:6, 11] = FLOOR
        # Hallway
        t[9:11, 1:23] = FLOOR
        # Doorway: living room → hallway
        t[8, 4:7] = FLOOR
        # Doorway: kitchen → hallway
        t[8, 14:17] = FLOOR
        # Exterior door (right wall of hallway)
        t[9:11, 23] = FLOOR
        # Bedroom
        t[12:17, 1:23] = FLOOR
        # Doorway: hallway → bedroom
        t[11, 10:13] = FLOOR

        self.tiles = t
        self.fixtures = {
            "fridge":   (19, 3),   # kitchen
            "recliner": (4,  5),   # living room
            "door":     (22, 10),  # hallway, near right wall
        }
        self.robot_start_tile = (3, 3)  # living room


MAP_REGISTRY: dict[str, Type[Map]] = {
    "default": DefaultHouseMap,
}
