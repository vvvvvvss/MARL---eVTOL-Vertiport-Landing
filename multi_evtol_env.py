import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class MultiEVTOLEnv(gym.Env):
    
    def __init__(self, num_agents=3):
        super().__init__()
        
        self.num_agents = num_agents
        self.max_x = 100.0
        self.max_y = 100.0
        
        # Each agent: 2D thrust
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [x, y, vx, vy, battery, gx, gy] + other agent positions
        obs_dim = 7 + (num_agents - 1) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Dynamics parameters
        self.dt = 0.2
        self.max_speed = 12.0
        self.collision_radius = 4.5
        
        # Battery
        self.battery_drain_rate = 0.03
        
        # Obstacles
        self.obstacles = [
            {'pos': np.array([30.0, 30.0]), 'radius': 6},
            {'pos': np.array([60.0, 60.0]), 'radius': 10},
            {'pos': np.array([50.0, 20.0]), 'radius': 2},
        ]
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.positions = []
        self.velocities = []
        self.batteries = []
        self.goals = []
        
        # Spread agents initially
        start_positions = [
            np.array([10.0, 10.0]),
            np.array([10.0, 90.0]),
            np.array([90.0, 10.0]),
        ]
        
        goal_positions = [
            np.array([90.0, 90.0]),
            np.array([90.0, 10.0]),
            np.array([10.0, 90.0]),
        ]
        
        for i in range(self.num_agents):
            self.positions.append(start_positions[i].copy())
            self.velocities.append(np.zeros(2))
            self.batteries.append(100.0)
            self.goals.append(goal_positions[i].copy())
        
        # Initialize previous distances for reward shaping
        self.prev_distances = [
            np.linalg.norm(self.goals[i] - self.positions[i])
            for i in range(self.num_agents)
        ]
        
        obs = [self._get_obs(i) for i in range(self.num_agents)]
        return obs, {}

    
    def _get_obs(self, agent_id):
        # Own state
        obs = np.concatenate([
            self.positions[agent_id],
            self.velocities[agent_id],
            [self.batteries[agent_id]],
            self.goals[agent_id]
        ])
        
        # Other agents
        for i in range(self.num_agents):
            if i != agent_id:
                obs = np.concatenate([obs, self.positions[i]])
        
        return obs.astype(np.float32)

    
    def step(self, actions):

        # ---------- UPDATE AGENT STATES ----------
        for i in range(self.num_agents):
            action = np.array(actions[i], dtype=float)

            # Thrust → acceleration
            acceleration = action * 2.0
            
            # Update velocity
            self.velocities[i] += acceleration * self.dt
            
            # Speed clamp
            speed = np.linalg.norm(self.velocities[i])
            if speed > self.max_speed:
                self.velocities[i] = self.velocities[i] / speed * self.max_speed
            
            # Update position
            self.positions[i] += self.velocities[i] * self.dt

            # Battery drain (depends on thrust)
            battery_cost = self.battery_drain_rate * (0.5 + np.linalg.norm(action))
            self.batteries[i] -= battery_cost


        # ---------- COMPUTE REWARDS ----------
        rewards = []
        terminateds = []
        truncateds = []

        for i in range(self.num_agents):
            reward = 0.0
            pos = self.positions[i]
            goal = self.goals[i]

            distance_to_goal = np.linalg.norm(goal - pos)

            # Progress reward
            progress = self.prev_distances[i] - distance_to_goal
            reward += progress * 2.5
            self.prev_distances[i] = distance_to_goal

            # Small distance penalty
            reward -= distance_to_goal * 0.001

            # --- Static obstacle collisions ---
            collision_static = False
            for obs in self.obstacles:
                dist = np.linalg.norm(pos - obs['pos'])
                if dist < obs['radius']:
                    collision_static = True
                    reward -= 30.0
                    break

            # --- Agent-to-agent collisions ---
            collision_agent = False
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(pos - self.positions[j])
                    if dist < self.collision_radius:
                        collision_agent = True
                        reward -= 60.0
                        break

            # Goal reached
            reached_goal = distance_to_goal < 5.0
            if reached_goal:
                reward += 120.0 + self.batteries[i]

            # Boundaries
            out_of_bounds = (
                pos[0] < 0 or pos[0] > self.max_x or
                pos[1] < 0 or pos[1] > self.max_y
            )
            if out_of_bounds:
                reward -= 20.0
                self.positions[i] = np.clip(self.positions[i], [0, 0], [self.max_x, self.max_y])

            battery_dead = self.batteries[i] <= 0

            terminated = bool(reached_goal or battery_dead or collision_static or collision_agent)
            truncated = False  

            rewards.append(float(reward))
            terminateds.append(terminated)
            truncateds.append(truncated)

        # Observations
        observations = [self._get_obs(i) for i in range(self.num_agents)]

        done = any(terminateds)
        truncated = any(truncateds)

        return observations, rewards, done, truncated, {}

    
    def render(self):
        plt.clf()
        plt.xlim(0, self.max_x)
        plt.ylim(0, self.max_y)
        plt.gca().set_aspect('equal')

        # Obstacles
        for obs in self.obstacles:
            circle = plt.Circle(obs['pos'], obs['radius'], color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        colors = ['blue', 'green', 'orange']
        
        for i in range(self.num_agents):
            plt.plot(
                self.positions[i][0], self.positions[i][1],
                'o', color=colors[i], markersize=10, label=f'eVTOL {i+1}'
            )
            plt.plot(
                self.goals[i][0], self.goals[i][1],
                '*', color=colors[i], markersize=15
            )
            circle = plt.Circle(self.positions[i], self.collision_radius,
                                color=colors[i], alpha=0.2, fill=True)
            plt.gca().add_patch(circle)

        plt.title(f'Batteries: {[f"{b:.1f}%" for b in self.batteries]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.pause(0.01)


if __name__ == "__main__":
    env = MultiEVTOLEnv(num_agents=3)
    obs, info = env.reset()

    plt.ion()
    print("Starting multi-agent simulation with random actions...")

    for step in range(500):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        observations, rewards, done, truncated, info = env.step(actions)
        env.render()
        
        if done:
            print(f"\nEpisode ended at step {step}")
            for i in range(env.num_agents):
                dist = np.linalg.norm(env.goals[i] - env.positions[i])
                print(f"Agent {i+1}: Distance to goal: {dist:.2f}, Battery: {env.batteries[i]:.1f}%")
            break

    plt.ioff()
    plt.show()
