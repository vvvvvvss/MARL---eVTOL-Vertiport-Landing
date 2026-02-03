from simple_evtol_env import SimpleEVTOLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np


env = SimpleEVTOLEnv()
print("Checking environment...")
check_env(env)
print("Environment is valid!\n")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    tensorboard_log="./evtol_tensorboard/"
)

print("Starting training for 100,000 steps...")
print("This will take a few minutes...\n")
model.learn(total_timesteps=100000)

model.save("evtol_trained")
print("\n Model saved as 'evtol_trained'!")

print("\n Testing trained model...")
obs, info = env.reset()

plt.ion()
positions = [env.position.copy()]

for step in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    positions.append(env.position.copy())
    env.render()
    
    if done or truncated:
        print(f"\n Episode finished in {step} steps!")
        print(f"Final distance to goal: {np.linalg.norm(env.goal - env.position):.2f}")
        print(f"Battery remaining: {env.battery:.1f}%")
        
        if np.linalg.norm(env.goal - env.position) < 5.0:
            print(" GOAL REACHED!")
        break

plt.ioff()
plt.show()