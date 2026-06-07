"""Dev render helper — dumps current sprites + scene to /tmp for visual review.

Run (no inline -c, so it stays inside the conda/python allowlist):
    conda run -n homebot xvfb-run -a python3 scripts/render_dev.py

Outputs:
    /tmp/sprite_sheet.png  — every sprite on a neutral background
    /tmp/scene_full.png    — full map (judge layout/proportions)
    /tmp/scene_obs.png     — the 84x84 observation, upscaled (what the agent sees)
"""
import os
import sys
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame

from homebot.sprites import make_sprite, RECLINER, TV, FRIDGE, COUNTER, SINK, TABLE, PACKAGE, BOTTLE, CAN
from homebot.env import HomeBotEnv

OUT = "/tmp"


def sprite_sheet(path: str):
    sprites = [("recliner", RECLINER), ("tv", TV), ("fridge", FRIDGE),
               ("counter", COUNTER), ("sink", SINK), ("table", TABLE),
               ("package", PACKAGE), ("bottle", BOTTLE), ("can", CAN)]
    pad, x, maxh, placed = 12, 12, 0, []
    for _name, grid in sprites:
        s = make_sprite(grid, scale=8)
        placed.append((s, x))
        x += s.get_width() + pad
        maxh = max(maxh, s.get_height())
    sheet = pygame.Surface((x, maxh + 2 * pad))
    sheet.fill((120, 120, 120))
    for s, xx in placed:
        sheet.blit(s, (xx, pad))
    pygame.image.save(sheet, path)


def main():
    pygame.init()
    sprite_sheet(os.path.join(OUT, "sprite_sheet.png"))

    env = HomeBotEnv(render_mode="rgb_array")
    env.reset(seed=0)
    obs = None
    # step a few times so the robot is off its spawn tile in the shot
    for _ in range(15):
        obs, *_ = env.step(env.action_space.sample())
    env.render()

    full = env._renderer._surface
    pygame.image.save(full, os.path.join(OUT, "scene_full.png"))

    if obs is not None:
        obs_surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        obs_big = pygame.transform.scale(obs_surf, (336, 336))  # 4x for visibility
        pygame.image.save(obs_big, os.path.join(OUT, "scene_obs.png"))

    env.close()
    print("wrote sprite_sheet.png, scene_full.png, scene_obs.png to", OUT)


if __name__ == "__main__":
    main()
