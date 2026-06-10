from typing import Optional
import numpy as np
from homebot.maps import Map

# Robot.RADIUS(15) + _FIXTURE_RANGE(2.0) * tile_size(32) — matches TaskManager fixture interaction radius
GOAL_THRESHOLD = 79.0

# (target: fixture name or "trash", initial_carry: None | str)
GOAL_REGISTRY: dict[str, tuple[str, Optional[str]]] = {
    "go_to_fridge":    ("fridge",   None),
    "deliver_drink":   ("recliner", "drink"),
    "go_to_door":      ("door",     None),
    "deliver_package": ("recliner", "package"),
    "collect_trash":   ("trash",    None),
}

GOAL_NAMES: list[str] = list(GOAL_REGISTRY.keys())


def goal_to_coordinates(
    goal_name: str,
    map: Map,
    trash_positions: Optional[list[tuple[int, int]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Convert a goal name to pixel (x, y) coordinates.

    collect_trash: picks from trash_positions using rng (first item when rng is None).
    Raises ValueError if collect_trash is requested with an empty or missing trash_positions.
    """
    target, _ = GOAL_REGISTRY[goal_name]
    if target == "trash":
        if not trash_positions:
            raise ValueError("collect_trash requires at least one active trash position")
        if rng is not None:
            idx = int(rng.integers(0, len(trash_positions)))
            col, row = trash_positions[idx]
        else:
            col, row = trash_positions[0]
    else:
        col, row = map.fixtures[target]
    px, py = map.tile_to_pixel(col, row)
    return float(px), float(py)
