
# Product Requirements Document (PRD)

## Features
1.  **Dual Agent Combat**: Two AI agents (shooters) in a confined map.
2.  **Shooting Mechanic**: Agents shoot automatically every 0.5 seconds.
3.  **Bullet Physics**: Bullets have travel time; speed increases gradually over time.
4.  **Movement**: Free movement (Up, Down, Left, Right) blocked by walls.
5.  **Game Termination**: Game ends when an agent is hit by a bullet.
6.  **Maze Generation**: Procedural map generation (DFS/Prim's) to vary the environment.
7.  **RL Training**: Agents learn optimal strategies (policy) for survival and combat.
    - **Penalties**:
        - Inaction (not moving).
        - Cowardice (running away/hiding too long).
8.  **Visualization**: Real-time web-based visualization of the training/game.

## User Interface (Frontend)
- **Game Canvas**: Visual representation of the maze, agents, and bullets.
- **Dashboard**:
    - "Start Training" button.
    - "Stop Training" button.
    - "Reset Environment" button.
    - Metrics: Win rate, Episode length, Total Reward.
- **Config Panel**: Adjust game speed, bullet speed increment, penalty weights.

## Success Metrics
- Agents demonstrate intelligent behavior (taking cover, leading shots).
- Stable training loop with visibly increasing reward over time.
- Smooth visualization (30+ FPS) via WebSocket.
