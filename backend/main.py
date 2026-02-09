
import gymnasium as gym
from env.shooter_env import ShooterEnv
from stable_baselines3 import PPO
import os

# Register the environment
gym.register(id="Shooter-v0", entry_point="env.shooter_env:ShooterEnv")

from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Render every step to see what's happening
        # Or every N steps to speed up
        self.training_env.render()
        return True

class SelfPlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.policy = None # Will be set after model creation
        self.latest_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.latest_obs = obs
        return obs, info

    def step(self, action):
        # Predict enemy action using the policy (Self-Play)
        enemy_action = None
        if self.policy:
            # We need observation for the enemy.
            # Ideally, env should return both obs. But our env returns Agent 0 obs.
            # Let's assume symmetric observation for now. 
            # We need to flip the observation for Agent 1? 
            # Or just use the same "type" of observation relative to Agent 1.
            # Our env `_get_obs(agent_id)` handles this! 
            # So we need access to the underlying env to get Agent 1's obs.
            
            # Access underlying env (unwrapped)
            root_env = self.env.unwrapped
            enemy_obs = root_env._get_obs(1) 
            
            # Predict
            enemy_action, _ = self.policy.predict(enemy_obs, deterministic=False)
            
        # Set enemy action on the raw environment
        # We need to dig through wrappers to find ShooterEnv
        root_env = self.env.unwrapped
        if hasattr(root_env, 'enemy_action'):
            root_env.enemy_action = enemy_action
            
        # Call step (standard signature)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.latest_obs = obs
        return obs, reward, terminated, truncated, info

def main():
    # Create environment
    raw_env = gym.make("Shooter-v0", render_mode="human")
    env = SelfPlayWrapper(raw_env)
    
    # Initialize Agent (PPO)
    model_path = "models/ppo_shooter"
    if os.path.exists(model_path + ".zip"):
        print("Loading existing model...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new model...")
        model = PPO("MlpPolicy", env, verbose=1)
    
    # Set policy in wrapper
    env.policy = model

    # Training Loop with periodic saving and rendering
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1
        # Train with rendering
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=RenderCallback())
        
        # Save
        model.save(model_path)
        print(f"Iteration {iters}: Model saved.")
        
        # Visualize (Render loop is handled inside env.step if render_mode is human)
        pass

        # Optional: specific evaluation loop with Manual Control
        # If user presses keys, we override the model action.
    
    # Manual Control / Evaluation Loop
    obs, _ = env.reset()
    done = False
    import pygame
    
    print("Entering Evaluation/Manual Mode...")
    print("Controls: Arrow Keys to Move, Space to Shoot.")
    
    while True:
        # Check for Pygame events first
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action = 0
        elif keys[pygame.K_DOWN]: action = 1
        elif keys[pygame.K_LEFT]: action = 2
        elif keys[pygame.K_RIGHT]: action = 3
        elif keys[pygame.K_SPACE]: action = 4
        
        # If no key pressed, let the model decide (or just stand still if pure manual)
        if action is None:
            action, _ = model.predict(obs) # Model plays if no key
            # action = 5 # No-Op if pure manual

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if done:
            obs, _ = env.reset()
            # Don't break, keep playing
            
        # env.render() is called inside step() logic for "human" mode
        # But we need to make sure we don't block too fast? step() has no sleep.
        # ShooterEnv render handles tick(30) so it should be fine.

if __name__ == "__main__":
    main()
