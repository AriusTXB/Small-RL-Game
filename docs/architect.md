
# Architecture Decisions

## Core Architecture
- **Language**: Python 3.10+
- **RL Framework**: Gymnasium + Stable Baselines3 (PPO) via PyTorch.
- **Visualization**: Pygame.
    - The environment's `render()` method will update the Pygame display.
    - Training loop will handle events (e.g., quit, change speed).

## Patterns
- **Monolithic Script**: For simplicity in local execution, a single entry point `main.py` will drive both training and rendering.
- **Synchronous Execution**: Rendering happens within the training loop steps.
- **Configurability**: Constants for Map Size, Bullet Speed, etc., defined in a config file or top of script.

## Why This Stack?
- **Simplicity**: No need for separate frontend/backend processes or WebSocket bridging.
- **Performance**: Direct memory access for rendering is faster than serializing JSON over WebSockets.
- **Dependencies**: Drops Node.js requirement, keeping the stack pure Python.
