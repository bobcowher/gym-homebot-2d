import math
from typing import Optional
import numpy as np
from homebot.maps import Map
from homebot.robot import Robot

_FIXTURE_RANGE = 2.0  # tile_size multiplier for fridge/recliner/door interaction radius
                      # (generous so the robot can interact while fixtures are solid)
_TRASH_RANGE = 0.5   # tile_size multiplier for floor trash pickup (tighter than fixtures)


def _dist(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


class TaskManager:
    def __init__(self, goals: list[str]):
        self.goals = set(goals)
        self.trash_positions: list[tuple[int, int]] = []
        self.package_present: bool = False
        self.drink_delivered: bool = False
        self.package_delivered: bool = False

    def reset(self, map: Map, n_trash: int, rng: np.random.Generator):
        fixture_tiles = list(map.fixtures.values())
        self.trash_positions = (
            map.spawn_trash(n_trash, rng, exclude=fixture_tiles)
            if "trash" in self.goals
            else []
        )
        self.package_present = "package" in self.goals
        self.drink_delivered = False
        self.package_delivered = False

    def step(self, robot: Robot, map: Map) -> float:
        reward = 0.0
        reward += self._check_trash(robot, map)
        reward += self._check_drink(robot, map)
        reward += self._check_package(robot, map)
        return reward

    def is_done(self) -> bool:
        trash_done = "trash" not in self.goals or len(self.trash_positions) == 0
        drink_done = "drink" not in self.goals or self.drink_delivered
        package_done = "package" not in self.goals or self.package_delivered
        return trash_done and drink_done and package_done

    def active_goals(self, robot: Optional["Robot"] = None) -> list[str]:
        """Return active goal strings. Without robot: high-level goals. With robot: current sub-goals."""
        result = []
        if "trash" in self.goals and self.trash_positions:
            result.append("collect_trash")
        if "drink" in self.goals and not self.drink_delivered:
            if robot is None:
                result.append("fetch_drink")
            elif robot.carrying == "drink":
                result.append("deliver_to_human")
            else:
                result.append("go_to_fridge")
        if "package" in self.goals and not self.package_delivered:
            if robot is None:
                result.append("retrieve_package")
            elif robot.carrying == "package":
                result.append("deliver_to_human")
            else:
                result.append("go_to_door")
        return result

    def get_info(self, robot: Optional["Robot"] = None) -> dict:
        return {
            "goals": self.active_goals(robot),
            "trash_remaining": len(self.trash_positions),
            "drink_delivered": self.drink_delivered,
            "package_delivered": self.package_delivered,
            "package_present": self.package_present,
        }

    def _check_trash(self, robot: Robot, map: Map) -> float:
        if "trash" not in self.goals:
            return 0.0
        # Trash collection does not require empty hands — robot vacuums while carrying.
        pickup_dist = robot.RADIUS + map.tile_size * _TRASH_RANGE
        reward = 0.0
        remaining = []
        for pos in self.trash_positions:
            px, py = map.tile_to_pixel(*pos)
            if _dist(robot.x, robot.y, px, py) <= pickup_dist:
                reward += 1.0
            else:
                remaining.append(pos)
        self.trash_positions = remaining
        return reward

    def _check_drink(self, robot: Robot, map: Map) -> float:
        if "drink" not in self.goals or self.drink_delivered:
            return 0.0
        pickup_dist = robot.RADIUS + map.tile_size * _FIXTURE_RANGE
        fridge_px, fridge_py = map.tile_to_pixel(*map.fixtures["fridge"])
        rec_px, rec_py = map.tile_to_pixel(*map.fixtures["recliner"])

        if robot.carrying is None:
            if _dist(robot.x, robot.y, fridge_px, fridge_py) <= pickup_dist:
                robot.carrying = "drink"
        elif robot.carrying == "drink":
            if _dist(robot.x, robot.y, rec_px, rec_py) <= pickup_dist:
                robot.carrying = None
                self.drink_delivered = True
                return 1.0
        return 0.0

    def _check_package(self, robot: Robot, map: Map) -> float:
        if "package" not in self.goals or self.package_delivered:
            return 0.0
        pickup_dist = robot.RADIUS + map.tile_size * _FIXTURE_RANGE
        door_px, door_py = map.tile_to_pixel(*map.fixtures["door"])
        rec_px, rec_py = map.tile_to_pixel(*map.fixtures["recliner"])

        if robot.carrying is None and self.package_present:
            if _dist(robot.x, robot.y, door_px, door_py) <= pickup_dist:
                robot.carrying = "package"
                self.package_present = False
        elif robot.carrying == "package":
            if _dist(robot.x, robot.y, rec_px, rec_py) <= pickup_dist:
                robot.carrying = None
                self.package_delivered = True
                return 1.0
        return 0.0
