
import numpy as np
import random
from collections import deque

def generate_maze(size=20, density=0.2):
    """
    Generates a random maze with assured connectivity between start (1,1) and end (size-2, size-2).
    """
    start = (1, 1)
    end = (size - 2, size - 2)
    
    while True:
        # 0 = Empty, 1 = Wall
        maze = np.zeros((size, size), dtype=int)
        
        # Borders
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1
        
        # Random blocks
        # Avoid placing walls on start/end
        num_obstacles = int(size * size * density)
        count = 0
        while count < num_obstacles:
            r, c = random.randint(1, size-2), random.randint(1, size-2)
            if (r, c) == start or (r, c) == end:
                continue
            if maze[r, c] == 0:
                maze[r, c] = 1
                count += 1
        
        # Check connectivity using BFS
        if is_connected(maze, start, end, size):
            return maze

def is_connected(maze, start, end, size):
    q = deque([start])
    visited = set([start])
    
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            return True
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
                
    return False
