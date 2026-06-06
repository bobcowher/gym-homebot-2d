"""
Human play wrapper for HomeBotEnv.

Controls (discrete mode):
  Arrow keys / WASD — 8-directional movement
  R                 — reset episode

Controls (continuous mode):
  Up/W   — move forward
  Down/S — move backward
  Left/A — rotate counter-clockwise
  Right/D — rotate clockwise
  R      — reset episode
"""
import argparse
import sys
import pygame
from homebot import HomeBotEnv


def _discrete_action(keys) -> int | None:
    up    = keys[pygame.K_UP]    or keys[pygame.K_w]
    down  = keys[pygame.K_DOWN]  or keys[pygame.K_s]
    left  = keys[pygame.K_LEFT]  or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    if up    and right: return 1   # NE
    if down  and right: return 3   # SE
    if down  and left:  return 5   # SW
    if up    and left:  return 7   # NW
    if up:              return 0   # N
    if right:           return 2   # E
    if down:            return 4   # S
    if left:            return 6   # W
    return None


def _continuous_action(keys):
    linear  = 0.0
    angular = 0.0
    if keys[pygame.K_UP]    or keys[pygame.K_w]: linear  =  1.0
    if keys[pygame.K_DOWN]  or keys[pygame.K_s]: linear  = -1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: angular =  1.0
    if keys[pygame.K_LEFT]  or keys[pygame.K_a]: angular = -1.0
    return [linear, angular]


def main():
    parser = argparse.ArgumentParser(description="Play HomeBot as a human.")
    parser.add_argument("--action-mode", default="discrete", choices=["discrete", "continuous"])
    parser.add_argument("--goals", nargs="+", default=["trash", "drink", "package"],
                        choices=["trash", "drink", "package"])
    parser.add_argument("--n-trash", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    env = HomeBotEnv(
        goals=args.goals,
        action_mode=args.action_mode,
        n_trash=args.n_trash,
        max_steps=args.max_steps,
        render_mode="human",
    )

    obs, info = env.reset()
    total_reward = 0.0
    episode = 1
    print(f"Episode {episode} started. R to reset. Close window to quit.")

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                episode += 1
                print(f"Episode {episode} started.")

        keys = pygame.key.get_pressed()
        if args.action_mode == "discrete":
            action = _discrete_action(keys)
            if action is None:
                clock.tick(60)
                continue
        else:
            action = _continuous_action(keys)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward > 0:
            print(f"  +{reward:.1f}  carrying={info['carrying']}  "
                  f"trash={info['trash_remaining']}  "
                  f"drink={info['drink_delivered']}  "
                  f"pkg={info['package_delivered']}")

        if terminated:
            print(f"Episode {episode} complete. Total reward: {total_reward:.1f}")
            obs, info = env.reset()
            total_reward = 0.0
            episode += 1
            print(f"Episode {episode} started.")
        elif truncated:
            print(f"Episode {episode} truncated. Total reward: {total_reward:.1f}")
            obs, info = env.reset()
            total_reward = 0.0
            episode += 1
            print(f"Episode {episode} started.")

        clock.tick(60)


if __name__ == "__main__":
    main()
