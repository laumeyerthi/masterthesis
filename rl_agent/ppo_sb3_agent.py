import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os

# Add parent directory for env import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

def train():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9)
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)


    print("Initializing PPO Model...")
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=0,
        tensorboard_log="tmp/logs/ppo_sb3_agent/"
    )
    
    print("Starting Training...")
    model.learn(total_timesteps=500000,progress_bar=True)
    
    print("Saving Model...")
    model.save("ppo_lab_env")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
