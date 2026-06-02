import gymnasium as gym
import time

# 1. Create the environment with a visual interface
env = gym.make("Acrobot-v1", render_mode="human")

# 2. Reset the system to its initial vertical hanging state
observation, info = env.reset()

print("Opening Acrobot simulation window...")

# 3. Run the simulation for 300 steps
for step in range(300):
    # Take a random action (apply force left, none, or right)
    action = env.action_space.sample()
    
    # Apply the action and move forward one frame
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Slight pause to make the animation smooth to watch
    time.sleep(0.02)
    
    # If the goal is reached or time runs out, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# 4. Safely close the window
env.close()
print("Simulation finished successfully!")
