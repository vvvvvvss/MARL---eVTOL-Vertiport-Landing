# train_multi.py
from multi_evtol_env import MultiEVTOLEnv
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym

# Create multi-agent environment
env = MultiEVTOLEnv(num_agents=3)

# For simplicity: train ONE policy that all agents share
# (More advanced: train separate policy for each agent)

# Wrapper to make it work with single-agent algorithms
class SingleAgentWrapper(gym.Wrapper):
    """Wrapper to train one agent at a time"""
    def __init__(self, env, agent_id=0):
        super().__init__(env)
        self.agent_id = agent_id
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self.agent_id], info
    
    def step(self, action):
        # All other agents take random actions
        actions = [self.env.action_space.sample() for _ in range(self.env.num_agents)]
        actions[self.agent_id] = action
        
        obs, rewards, done, truncated, info = self.env.step(actions)
        return obs[self.agent_id], rewards[self.agent_id], done, truncated, info

# Train agent 0
wrapped_env = SingleAgentWrapper(env, agent_id=0)
model = PPO("MlpPolicy", wrapped_env, verbose=1)

print("Training multi-agent system...")
model.learn(total_timesteps=100000)
model.save("evtol_multi_agent")
print("Training complete!")