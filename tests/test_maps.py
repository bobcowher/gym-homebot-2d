import numpy as np
import pytest
from homebot.maps import DefaultHouseMap, Map, MAP_REGISTRY, FLOOR, WALL, LAWN


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


def test_pixel_dimensions():
    m = DefaultHouseMap()
    assert m.pixel_width == 27 * 32   # 24 interior + 3 cols outside lawn
    assert m.pixel_height == 18 * 32


def test_spawn_trash_zero():
    m = DefaultHouseMap()
    rng = np.random.default_rng(0)
    assert m.spawn_trash(0, rng) == []


def test_spawn_trash_clamps_to_available():
    m = DefaultHouseMap()
    rng = np.random.default_rng(0)
    floor_count = len(m.valid_floor_tiles())
    result = m.spawn_trash(floor_count + 100, rng)
    assert len(result) == floor_count


def test_solid_mask_blocks_walls_and_lawn():
    m = DefaultHouseMap()
    wall = np.argwhere(m.tiles == WALL)[0]
    lawn = np.argwhere(m.tiles == LAWN)[0]
    assert m.solid[wall[0], wall[1]]
    assert m.solid[lawn[0], lawn[1]]


def test_doorway_is_floor_and_not_solid():
    m = DefaultHouseMap()
    for col, row in m.door_tiles:
        assert m.tiles[row, col] == FLOOR
        assert not m.solid[row, col]


def test_fixture_footprints_are_solid():
    m = DefaultHouseMap()
    for name in m.solid_fixtures:
        col, row = m.fixtures[name]
        assert m.solid[row, col], f"fixture '{name}' center should be solid"


def test_valid_floor_excludes_solid_footprints():
    m = DefaultHouseMap()
    rec = m.fixtures["recliner"]
    assert rec not in m.valid_floor_tiles()  # under the (solid) recliner


def test_robot_start_not_solid():
    m = DefaultHouseMap()
    col, row = m.robot_start_tile
    assert not m.solid[row, col]
