import gymnasium as gym
from sb3_contrib import RecurrentPPO
import sys
import os

# Add parent directory for env import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

def train():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=4)
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)


    print("Initializing PPO Model...")
    model = RecurrentPPO(
        "MultiInputLstmPolicy", 
        env, 
        verbose=0,
        tensorboard_log="tmp/logs/ppo_recurrent_sb3_agent/"
    )
    print("Starting Training...")
    model.learn(total_timesteps=100000,progress_bar=True)
    
    print("Saving Model...")
    model.save("ppo_recurrent_lab_env")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
