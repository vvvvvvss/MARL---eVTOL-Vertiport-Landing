"""
Gymnasium-Compatible Wrapper for VertiportEnv
Enables integration with Stable Baselines3 and other RL frameworks

This wrapper converts the multi-agent VertiportEnv to a Gymnasium-compatible
single-agent problem by aggregating observations and rewards across all aircraft.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any
from vertiport_env import VertiportEnv


class VertiportRLEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for VertiportEnv.
    
    This wrapper converts the multi-agent scheduling problem into a single-agent
    centralized training problem by:
    - Aggregating observations from all aircraft
    - Aggregating rewards across all aircraft
    - Learning a single policy applied to all agents
    
    This approach (Centralized Training) enables faster single-agent RL training
    while maintaining coordination through shared policy.
    """
    
    metadata = {"render_modes": [None, "human"], "render_fps": 10}
    
    def __init__(
        self,
        num_pads: int = 8,
        arrival_rate: float = 20.0,
        max_aircraft: int = 50,
        dt: float = 0.1,
        separation_distance: float = 500.0,
        render_mode: str = None
    ):
        """
        Initialize the Gymnasium-compatible vertiport environment.
        
        Args:
            num_pads: Number of landing pads
            arrival_rate: Aircraft arrival rate (aircraft per hour)
            max_aircraft: Maximum aircraft in airspace
            dt: Timestep in minutes
            separation_distance: Minimum separation in meters
            render_mode: None or 'human'
        """
        self.env = VertiportEnv(
            num_pads=num_pads,
            arrival_rate=arrival_rate,
            max_aircraft=max_aircraft,
            dt=dt,
            separation_distance=separation_distance,
            render_mode=render_mode
        )
        
        self.num_pads = num_pads
        self.render_mode = render_mode
        
        # Single aircraft observation dimension from VertiportEnv
        # obs = [pos(3) + vel(1) + heading(1) + battery(1) + priority(1)
        #        + pads(3*num_pads) + altitude_rings(4)]
        single_agent_obs_dim = 7 + 3 * num_pads + 4
        
        # Stack observations for multiple aircraft (up to max_aircraft)
        # For simplicity, we'll use a max of 10 aircraft observations stacked
        self.max_agents_in_obs = 10
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_agents_in_obs * single_agent_obs_dim,),
            dtype=np.float32
        )
        
        # Action space: select pad (0-7) or hold (8)
        self.action_space = spaces.Discrete(num_pads + 1)
        
        self.max_episode_steps = 2000
        self.current_episode_step = 0
        
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Returns:
            observation: Aggregated observation from all aircraft
            info: Environment info dict
        """
        super().reset(seed=seed)
        
        obs_dict, info = self.env.reset()
        self.current_episode_step = 0
        
        # Aggregate observations
        aggregated_obs = self._aggregate_observations(obs_dict)
        
        return aggregated_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        For multi-agent coordination, we apply the action to all aircraft:
        - Each aircraft can choose to land on a specific pad or hold
        - We distribute the action or use a action-selection policy
        
        Args:
            action: Action (0-7 = land on pad, 8 = hold)
            
        Returns:
            observation: Aggregated observation
            reward: Aggregated reward (sum of all aircraft rewards)
            terminated: Episode terminated
            truncated: Episode truncated (max steps reached)
            info: Environment info
        """
        # Create action dict: each aircraft takes the same action if valid
        # (in practice, the policy should learn to ignore invalid actions)
        actions = {}
        for aircraft in self.env.aircraft:
            mask = self.env.get_action_mask(aircraft)
            if mask[action]:
                # Action is valid for this aircraft
                actions[aircraft.id] = action
            else:
                # If invalid, default to hold
                actions[aircraft.id] = self.num_pads
        
        # Execute step in underlying environment
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        
        # Aggregate observations and rewards
        aggregated_obs = self._aggregate_observations(obs_dict)
        aggregated_reward = self._aggregate_rewards(rewards)
        
        # Episode termination conditions
        self.current_episode_step += 1
        episode_done = self.current_episode_step >= self.max_episode_steps
        
        # Aggregate terminated/truncated across all agents
        episode_terminated = any(terminated.values()) or episode_done
        episode_truncated = any(truncated.values())
        
        return aggregated_obs, aggregated_reward, episode_terminated, episode_truncated, info
    
    def _aggregate_observations(self, obs_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Aggregate observations from all aircraft.
        
        Stacks observations from multiple aircraft, padding with zeros if needed.
        This allows the policy to see the global state of the airspace.
        """
        if not obs_dict:
            # No aircraft - return zero observation
            return np.zeros(
                self.max_agents_in_obs * len(next(iter(obs_dict.values())) if obs_dict else self.observation_space.shape),
                dtype=np.float32
            )
        
        observations = list(obs_dict.values())
        
        # Stack observations, sorted by aircraft ID for consistency
        aircraft_ids = sorted(obs_dict.keys())
        observations = [obs_dict[aid] for aid in aircraft_ids]
        
        # Take only the first max_agents_in_obs aircraft
        observations = observations[:self.max_agents_in_obs]
        
        # Pad with zeros if fewer aircraft
        single_obs_dim = observations[0].shape[0]
        while len(observations) < self.max_agents_in_obs:
            observations.append(np.zeros(single_obs_dim, dtype=np.float32))
        
        # Concatenate all observations
        aggregated = np.concatenate(observations, dtype=np.float32)
        
        # Ensure correct shape
        aggregated = aggregated[:self.observation_space.shape[0]]
        
        return aggregated
    
    def _aggregate_rewards(self, rewards: Dict[int, float]) -> float:
        """
        Aggregate rewards from all aircraft.
        
        Returns the mean reward across all aircraft, encouraging policies
        that improve system performance overall.
        """
        if not rewards:
            return 0.0
        
        reward_values = list(rewards.values())
        if not reward_values:
            return 0.0
        
        # Return mean reward (can be modified to sum or weighted aggregate)
        return float(np.mean(reward_values))
    
    def render(self) -> None:
        """
        Render the environment.
        
        For now, just delegates to the underlying environment.
        """
        if self.render_mode == "human":
            self.env.render()
    
    def close(self) -> None:
        """Close the environment."""
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get environment state for saving/loading."""
        return {
            'current_time': self.env.current_time,
            'total_landings': self.env.total_landings,
            'total_delay': self.env.total_delay,
            'separation_violations': self.env.separation_violations
        }


class SingleAircraftRLEnv(gym.Env):
    """
    Alternative wrapper: Train on single-aircraft problem.
    
    Instead of aggregating all aircraft, focus on optimizing placement
    for one aircraft at a time. This is simpler and might converge faster.
    """
    
    def __init__(
        self,
        num_pads: int = 8,
        arrival_rate: float = 20.0,
        max_aircraft: int = 50,
        render_mode: str = None
    ):
        self.env = VertiportEnv(
            num_pads=num_pads,
            arrival_rate=arrival_rate,
            max_aircraft=max_aircraft,
            render_mode=render_mode
        )
        
        self.num_pads = num_pads
        self.render_mode = render_mode
        self.current_episode_step = 0
        self.max_episode_steps = 2000
        self.focused_aircraft_id = None
        
        # Observation space: focus on one aircraft
        self.single_agent_obs_dim = 7 + 3 * num_pads + 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.single_agent_obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(num_pads + 1)
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset and focus on first aircraft."""
        super().reset(seed=seed)
        
        obs_dict, info = self.env.reset()
        self.current_episode_step = 0
        
        # Focus on first aircraft
        if obs_dict:
            self.focused_aircraft_id = list(obs_dict.keys())[0]
            obs = obs_dict[self.focused_aircraft_id]
        else:
            obs = np.zeros(self.single_agent_obs_dim, dtype=np.float32)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step focused aircraft only."""
        # Action only for focused aircraft
        actions = {self.focused_aircraft_id: action}
        
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        
        # Get focused aircraft observation and reward
        obs = obs_dict.get(self.focused_aircraft_id, np.zeros(self.single_agent_obs_dim, dtype=np.float32))
        reward = rewards.get(self.focused_aircraft_id, 0.0)
        
        # Update focused aircraft to next one if current landed
        if self.focused_aircraft_id not in obs_dict and self.env.aircraft:
            self.focused_aircraft_id = self.env.aircraft[0].id
        
        self.current_episode_step += 1
        done = self.current_episode_step >= self.max_episode_steps
        
        return obs, reward, done or terminated.get(self.focused_aircraft_id, False), truncated.get(self.focused_aircraft_id, False), info
    
    def render(self) -> None:
        """Render the environment."""
        if self.render_mode == "human":
            self.env.render()
    
    def close(self) -> None:
        """Close the environment."""
        pass


def test_wrapper():
    """Test the Gymnasium wrapper."""
    print("Testing VertiportRLEnv wrapper...")
    
    env = VertiportRLEnv(num_pads=8, arrival_rate=20)
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Info keys: {list(info.keys())}")
    
    # Test step
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step {step}: reward={reward:.3f}, aircraft={info['num_aircraft']}")
        
        if terminated or truncated:
            break
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_wrapper()
