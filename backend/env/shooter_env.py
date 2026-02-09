
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from .utils.maze_generator import generate_maze

# Constants
MAP_SIZE = 20
CELL_SIZE = 30  # Pixels for rendering
AGENT_SPEED = 1
BULLET_SPEED_INITIAL = 1
BULLET_SPEED_INCREMENT = 0.1
MAX_STEPS = 500

# Colors
COLOR_BG = (0, 0, 0)
COLOR_WALL = (100, 100, 100)
COLOR_AGENT_1 = (0, 255, 0)
COLOR_AGENT_2 = (255, 0, 0)
COLOR_BULLET = (255, 255, 0)

class ShooterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.window_size = MAP_SIZE * CELL_SIZE
        
        # Action Space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Shoot, 5=NoOp
        self.action_space = spaces.Discrete(6)

        # Observation Space: 4 channels (Walls, Self, Enemy, Bullets)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, MAP_SIZE, MAP_SIZE), dtype=np.float32
        )

        self.window = None
        self.clock = None
        
        self.agents = []
        self.bullets = []
        self.map = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.map = generate_maze(MAP_SIZE)
        
        # Initialize Agents [x, y, hp, cooldown, last_move_time]
        # Agent 0: Top-Leftish
        self.agents = [
            {'x': 1, 'y': 1, 'hp': 1, 'color': COLOR_AGENT_1, 'id': 0},
            {'x': MAP_SIZE-2, 'y': MAP_SIZE-2, 'hp': 1, 'color': COLOR_AGENT_2, 'id': 1}
        ]
        
        self.bullets = []
        self.bullet_speed = BULLET_SPEED_INITIAL

        if self.render_mode == "human":
            self._init_pygame()
            
        return self._get_obs(0), {} # Return obs for Agent 0 by default? Or handle multi-agent?
        # NOTE: For PPO single agent training against a bot/heuristic, we return single obs.
        # But this is a 2-agent game. We might need a MultiAgentEnv structure or just alternate.
        # For simplicity, let's assume we train Agent 0 and Agent 1 is a heuristic or self-play.

    def step(self, action):
        self.steps += 1
        agent = self.agents[0]
        enemy = self.agents[1]
        
        reward = -0.01 # Step penalty
        
        # 1. Apply Action for Agent 0
        self._move_agent(agent, action)
        
        # 2. Apply Action for Enemy
        # Check if enemy_action was set externally (by SelfPlayWrapper)
        enemy_action = getattr(self, 'enemy_action', None)
        
        if enemy_action is None:
            enemy_action = self.action_space.sample() # Default to random
        
        # Reset for next step to avoid stale actions if wrapper fails
        self.enemy_action = None 
        
        self._move_agent(enemy, enemy_action)
        
        # 3. Increase Bullet Speed
        if self.steps % 100 == 0:
            self.bullet_speed += BULLET_SPEED_INCREMENT

        # 4. Move Bullets
        active_bullets = []
        hit = False
        
        for b in self.bullets:
            # Move bullet multiple steps based on speed (integer approximation)
            speed_int = int(self.bullet_speed)
            # Use fractional speed storage if needed, but for grid simple step is okay.
            # Let's just move 1 step 'speed' times
            
            for _ in range(max(1, speed_int)):
                dx, dy = 0, 0
                if b['dir'] == 0: dy = -1
                elif b['dir'] == 1: dy = 1
                elif b['dir'] == 2: dx = -1
                elif b['dir'] == 3: dx = 1
                
                b['x'] += dx
                b['y'] += dy
                
                # Check bounds/walls
                if not (0 <= b['x'] < MAP_SIZE and 0 <= b['y'] < MAP_SIZE) or self.map[b['y'], b['x']] == 1:
                    break # Hit wall
                
                # Check Hit Agent
                if b['x'] == agent['x'] and b['y'] == agent['y'] and b['owner'] != 0:
                    reward -= 10 # Got hit
                    # Penalty for getting hit - big one
                    hit = True
                    break
                if b['x'] == enemy['x'] and b['y'] == enemy['y'] and b['owner'] != 1:
                    # Reward for hitting enemy
                    # Base reward + Speed bonus (Inverse of steps taken)
                    # Faster kill = Higher reward
                    time_bonus = (MAX_STEPS - self.steps) * 0.1 
                    reward += 10 + time_bonus
                    hit = True
                    break
            else:
                # Loop completed without break means bullet still active
                active_bullets.append(b)
        
        self.bullets = active_bullets
        
        terminated = hit or self.steps >= MAX_STEPS
        
        # 5. Proximity Reward (Encourage engagement)
        # Calculate Manhattan distance
        dist = abs(agent['x'] - enemy['x']) + abs(agent['y'] - enemy['y'])
        # Max distance on 20x20 grid is ~40.
        # We want to encourage being close. 
        # Reward = (Max_Dist - Current_Dist) * Scaling
        # max_dist = MAP_SIZE * 2
        # proximity_reward = (40 - dist) * 0.01  -> Max +0.4 per step if adjacent
        proximity_reward = ((MAP_SIZE * 2) - dist) * 0.005
        reward += proximity_reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(0), reward, terminated, False, {}

    def _move_agent(self, agent, action):
        nx, ny = agent['x'], agent['y']
        
        if action == 0: ny -= 1
        elif action == 1: ny += 1
        elif action == 2: nx -= 1
        elif action == 3: nx += 1
        elif action == 4: # Shoot
            self.bullets.append({
                'x': agent['x'], 'y': agent['y'], 'dir': random.randint(0,3), 'owner': agent['id']
            })
            # Wait, direction? Agents need facing? Or shoot in all directions?
            # Requirement: "Shoot after each 0.5s". Maybe automatic?
            # User said: "They will shoot after each 0.5 seconds". 
            # Implies automatic shooting. 
            pass

        # Check collision with walls
        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE and self.map[ny, nx] == 0:
            agent['x'], agent['y'] = nx, ny

        # Handle automatic shooting
        # 0.5s at 30FPS = 15 steps. 
        if self.steps % 15 == 0:
             # Shoot in random direction or facing? Let's say random for now
             d = random.randint(0, 3)
             self.bullets.append({'x': agent['x'], 'y': agent['y'], 'dir': d, 'owner': agent['id']})


    def _get_obs(self, agent_id):
        obs = np.zeros((4, MAP_SIZE, MAP_SIZE), dtype=np.float32)
        # 0: Walls
        obs[0] = self.map
        # 1: Self
        me = self.agents[agent_id]
        obs[1, me['y'], me['x']] = 1
        # 2: Enemy
        opp = self.agents[1-agent_id]
        obs[2, opp['y'], opp['x']] = 1
        # 3: Bullets
        for b in self.bullets:
            if 0 <= b['x'] < MAP_SIZE and 0 <= b['y'] < MAP_SIZE:
                 obs[3, b['y'], b['x']] = 1
        return obs

    def _init_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self):
        if self.window is None:
             self._init_pygame()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(COLOR_BG)
        
        # Draw Walls
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                if self.map[y, x] == 1:
                    pygame.draw.rect(canvas, COLOR_WALL, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw Agents
        for agent in self.agents:
            pygame.draw.rect(canvas, agent['color'], (agent['x']*CELL_SIZE, agent['y']*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
        # Draw Bullets
        for b in self.bullets:
            pygame.draw.circle(canvas, COLOR_BULLET, (b['x']*CELL_SIZE + CELL_SIZE//2, b['y']*CELL_SIZE + CELL_SIZE//2), 4)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


