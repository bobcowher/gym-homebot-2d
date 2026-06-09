import math
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
        self._map: Map  # assigned by reset() before step() is ever called
        self.trash_positions: list[tuple[int, int]] = []
        self.package_present: bool = False
        self.drink_delivered: bool = False
        self.package_delivered: bool = False

    def reset(self, map: Map, n_trash: int, rng: np.random.Generator):
        self._map = map
        fixture_tiles = list(map.fixtures.values())
        self.trash_positions = (
            map.spawn_trash(n_trash, rng, exclude=fixture_tiles)
            if "trash" in self.goals
            else []
        )
        self.package_present = "package" in self.goals
        self.drink_delivered = False
        self.package_delivered = False

    def step(self, robot: Robot) -> float:
        reward = 0.0
        reward += self._check_trash(robot)
        reward += self._check_drink(robot)
        reward += self._check_package(robot)
        return reward

    def is_done(self) -> bool:
        trash_done = "trash" not in self.goals or len(self.trash_positions) == 0
        drink_done = "drink" not in self.goals or self.drink_delivered
        package_done = "package" not in self.goals or self.package_delivered
        return trash_done and drink_done and package_done

    def active_goals(self, robot: "Robot") -> list[str]:
        result = []
        if "trash" in self.goals and self.trash_positions:
            result.append("collect_trash")
        if "drink" in self.goals and not self.drink_delivered:
            result.append("deliver_to_human" if robot.carrying == "drink" else "go_to_fridge")
        if "package" in self.goals and not self.package_delivered:
            result.append("deliver_to_human" if robot.carrying == "package" else "go_to_door")
        return result

    def get_info(self, robot: "Robot") -> dict:
        return {
            "goals": self.active_goals(robot),
            "trash_remaining": len(self.trash_positions),
            "drink_delivered": self.drink_delivered,
            "package_delivered": self.package_delivered,
            "package_present": self.package_present,
        }

    # --- internal helpers ---

    def _check_pickup_delivery(
        self, robot: Robot, goal: str, pickup_fixture: str,
        carrying_val: str, is_delivered: bool, pickup_available: bool = True,
    ) -> tuple[float, bool, bool]:
        """Shared pickup→carry→deliver logic.

        Returns (reward, is_delivered, pickup_available).
        pickup_available lets callers track a depletable source (e.g. package at door).
        Pass pickup_available=True and discard the returned value for infinite sources (fridge).
        """
        if goal not in self.goals or is_delivered:
            return 0.0, is_delivered, pickup_available
        pickup_dist = robot.RADIUS + self._map.tile_size * _FIXTURE_RANGE
        pickup_px, pickup_py = self._map.tile_to_pixel(*self._map.fixtures[pickup_fixture])
        rec_px, rec_py = self._map.tile_to_pixel(*self._map.fixtures["recliner"])
        if robot.carrying is None and pickup_available:
            if _dist(robot.x, robot.y, pickup_px, pickup_py) <= pickup_dist:
                robot.carrying = carrying_val
                pickup_available = False
        elif robot.carrying == carrying_val:
            if _dist(robot.x, robot.y, rec_px, rec_py) <= pickup_dist:
                robot.carrying = None
                return 1.0, True, pickup_available
        return 0.0, is_delivered, pickup_available

    def _check_trash(self, robot: Robot) -> float:
        if "trash" not in self.goals:
            return 0.0
        # Trash collection does not require empty hands — robot vacuums while carrying.
        pickup_dist = robot.RADIUS + self._map.tile_size * _TRASH_RANGE
        reward = 0.0
        remaining = []
        for pos in self.trash_positions:
            px, py = self._map.tile_to_pixel(*pos)
            if _dist(robot.x, robot.y, px, py) <= pickup_dist:
                reward += 1.0
            else:
                remaining.append(pos)
        self.trash_positions = remaining
        return reward

    def _check_drink(self, robot: Robot) -> float:
        reward, self.drink_delivered, _ = self._check_pickup_delivery(
            robot, "drink", "fridge", "drink", self.drink_delivered,
        )
        return reward

    def _check_package(self, robot: Robot) -> float:
        reward, self.package_delivered, self.package_present = self._check_pickup_delivery(
            robot, "package", "door", "package", self.package_delivered, self.package_present,
        )
        return reward
