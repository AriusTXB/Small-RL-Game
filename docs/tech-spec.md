
# Technical Specifications

## Dependencies

### Core (Python)
- `gymnasium`: RL environment.
- `pygame`: Visualization.
- `stable-baselines3`: PPO algorithm.
- `torch`: Deep learning.
- `numpy`: Math.

## File Structure
backend/
├── main.py (Entry point)
├── env/
│   ├── shooter_env.py
│   └── __init__.py
├── utils/
│   ├── maze_generator.py
│   └── visualizer.py
└── models/ (Saved models)

## Game Loop
1. Initialize Env & Agents.
2. Loop Episodes:
    3. Loop Steps:
        4. Get Action from Agent.
        5. Env Step (Physics, Reward).
        6. If `render_frequency` met:
            7. Pygame Clear Screen.
            8. Draw Map, Agents, Bullets.
            9. Pygame Flip.
            10. Handle Pygame Events (Quit).
        11. Train Agent (if training).
    12. Log Metrics.
