# gym-homebot-2d

A top-down 2D Gymnasium environment where a home robot navigates a house, picks up trash, fetches drinks from the fridge, and delivers packages.

## Installation

```bash
pip install gym-homebot-2d
```

## Usage

```python
import gymnasium as gym
import homebot  # registers HomeBot2D-v1

env = gym.make("HomeBot2D-v1", render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environment Details

- **Observation**: 84×84 RGB pixel array
- **Action space**: Discrete (8 directions + stop) or Box (continuous steering)
- **Tasks**: trash pickup, drink fetch + delivery, package retrieval + delivery
