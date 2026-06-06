import numpy as np
import pytest
from homebot.maps import DefaultHouseMap, Map, MAP_REGISTRY, FLOOR, WALL


def test_default_map_tiles_is_2d():
    m = DefaultHouseMap()
    assert m.tiles.ndim == 2


def test_default_map_has_floor_and_wall():
    m = DefaultHouseMap()
    assert np.any(m.tiles == FLOOR)
    assert np.any(m.tiles == WALL)


def test_default_map_fixtures_on_floor():
    m = DefaultHouseMap()
    for name, (col, row) in m.fixtures.items():
        assert m.tiles[row, col] == FLOOR, f"fixture '{name}' at ({col},{row}) is not on a floor tile"


def test_robot_start_on_floor():
    m = DefaultHouseMap()
    col, row = m.robot_start_tile
    assert m.tiles[row, col] == FLOOR


def test_spawn_trash_correct_count():
    m = DefaultHouseMap()
    rng = np.random.default_rng(42)
    positions = m.spawn_trash(5, rng)
    assert len(positions) == 5


def test_spawn_trash_on_floor_tiles():
    m = DefaultHouseMap()
    rng = np.random.default_rng(42)
    for col, row in m.spawn_trash(10, rng):
        assert m.tiles[row, col] == FLOOR


def test_spawn_trash_excludes_given_tiles():
    m = DefaultHouseMap()
    rng = np.random.default_rng(42)
    fixtures = list(m.fixtures.values())
    for pos in m.spawn_trash(5, rng, exclude=fixtures):
        assert pos not in fixtures


def test_tile_to_pixel_origin():
    m = DefaultHouseMap()
    px, py = m.tile_to_pixel(0, 0)
    assert px == m.tile_size // 2
    assert py == m.tile_size // 2


def test_tile_to_pixel_offset():
    m = DefaultHouseMap()
    px, py = m.tile_to_pixel(2, 3)
    assert px == 2 * m.tile_size + m.tile_size // 2
    assert py == 3 * m.tile_size + m.tile_size // 2


def test_map_registry_contains_default():
    assert "default" in MAP_REGISTRY
    m = MAP_REGISTRY["default"]()
    assert isinstance(m, Map)
