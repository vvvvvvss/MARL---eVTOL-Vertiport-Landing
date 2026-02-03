import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class SimpleEVTOLEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.max_x = 100.0
        self.max_y = 100.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # State space: [x, y, vx, vy, battery, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, 0, 0, 0]),
            high=np.array([100, 100, 10, 10, 100, 100, 100]),
            dtype=np.float32
        )

        self.obstacles = [
            {'pos': np.array([30.0, 30.0]), 'radius': 6},
            {'pos': np.array([60.0, 60.0]), 'radius': 10},
            {'pos': np.array([50.0, 20.0]), 'radius': 2},
        ]
        
        self.dt = 0.2  # timestep
        self.max_speed = 10.0
        self.battery_drain_rate = 0.02

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # random start and goal positions
        self.position = np.array([10.0, 10.0])
        self.velocity = np.array([0.0, 0.0])
        self.battery = 100.0
        self.goal = np.array([90.0, 90.0])
        self.prev_distance = np.linalg.norm(self.goal - self.position)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.position,
            self.velocity,
            [self.battery],
            self.goal
        ]).astype(np.float32)

    def step(self, action):
        # Apply action (thrust)
        acceleration = action * 2.0
        self.velocity += acceleration * self.dt

        # Limit max speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        # Update position
        self.position += self.velocity * self.dt

        # Drain battery
        battery_cost = self.battery_drain_rate * (1 + speed / self.max_speed)
        self.battery -= battery_cost

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.goal - self.position)

        # Check for collisions with obstacles
        collision = False
        for obs in self.obstacles:
            dist = np.linalg.norm(self.position - obs['pos'])
            if dist < obs['radius']:
                collision = True
                break

        # Check if reached goal
        reached_goal = distance_to_goal < 5.0

        # Check boundaries and battery
        out_of_bounds = (
            self.position[0] < 0 or self.position[0] > self.max_x or
            self.position[1] < 0 or self.position[1] > self.max_y
        )
        battery_dead = self.battery <= 0

        # Calculate reward
        reward = 0.0
        reward -= distance_to_goal * 0.01  # Penalty for being far from goal
        reward -= battery_cost * 0.5       # Penalty for battery usage
        if collision:
            reward -= 30.0
            # Penalty for speed (encourages efficiency)
        
        if reached_goal:
            reward += 100.0 + self.battery  # Big bonus + remaining battery bonus

        progress_reward = (self.prev_distance - distance_to_goal) * 2.0
        reward += progress_reward
        self.prev_distance = distance_to_goal

        # Determine if episode should end - MUST BE PYTHON BOOL!
        terminated = bool(reached_goal or battery_dead or collision)
        truncated = bool(out_of_bounds)
        

        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self):
        # Clear the plot FIRST
        plt.clf()
        
        # Set limits
        plt.xlim(0, self.max_x)
        plt.ylim(0, self.max_y)
        plt.gca().set_aspect('equal')

        # Draw obstacles (AFTER clearing)
        for obs in self.obstacles:
            circle = plt.Circle(obs['pos'], obs['radius'], color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        # Draw eVTOL current position
        plt.plot(self.position[0], self.position[1], 'bo', markersize=10, label='eVTOL')

        # Draw goal
        plt.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal')

        # Add title with battery info
        plt.title(f'Battery: {self.battery:.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.pause(0.01)


if __name__ == "__main__":
    env = SimpleEVTOLEnv()
    obs, info = env.reset()

    plt.ion()
    print("Starting simulation with random actions...")
    print(f"Start position: {env.position}")
    print(f"Goal position: {env.goal}")
    
    step_count = 0
    for _ in range(500):
        action = env.action_space.sample()  # Random actions
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        step_count += 1

        if done or truncated:
            print(f"\nEpisode ended after {step_count} steps")
            print(f"Final position: {env.position}")
            print(f"Distance to goal: {np.linalg.norm(env.goal - env.position):.2f}")
            print(f"Battery remaining: {env.battery:.1f}%")
            if done and np.linalg.norm(env.goal - env.position) < 5.0:
                print("GOAL REACHED!")
            elif env.battery <= 0:
                print("Battery died")
            else:
                print("Episode terminated")
            break

    plt.ioff()
    plt.show()