import numpy as np
import pytest
import pygame
from homebot.maps import DefaultHouseMap
from homebot.robot import Robot
from homebot.tasks import TaskManager
from homebot.renderer import Renderer


@pytest.fixture
def components():
    m = DefaultHouseMap()
    robot = Robot(m.tile_to_pixel(*m.robot_start_tile))
    tm = TaskManager(["trash", "drink", "package"])
    tm.reset(m, n_trash=3, rng=np.random.default_rng(0))
    renderer = Renderer(m)
    return m, robot, tm, renderer


def test_render_returns_surface(components):
    m, robot, tm, renderer = components
    viewport = renderer.render(robot, tm)
    assert isinstance(viewport, pygame.Surface)


def test_to_display_shape(components):
    m, robot, tm, renderer = components
    viewport = renderer.render(robot, tm)
    arr = renderer.to_display(viewport)
    h, w = renderer.display_res[1], renderer.display_res[0]
    assert arr.shape == (h, w, 3)
    assert arr.dtype == np.uint8


def test_to_obs_default_shape(components):
    m, robot, tm, renderer = components
    viewport = renderer.render(robot, tm)
    obs = renderer.to_obs(viewport, (84, 84))
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8


def test_to_obs_custom_shape(components):
    m, robot, tm, renderer = components
    viewport = renderer.render(robot, tm)
    obs = renderer.to_obs(viewport, (64, 96))
    assert obs.shape == (64, 96, 3)


def test_viewport_smaller_than_full_map(components):
    m, robot, tm, renderer = components
    viewport = renderer.render(robot, tm)
    assert viewport.get_width() < m.pixel_width
    assert viewport.get_height() < m.pixel_height


def test_render_does_not_open_window(components):
    m, robot, tm, renderer = components
    renderer.render(robot, tm)
    assert renderer._window is None
