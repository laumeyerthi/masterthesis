import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
import sys
import os

# Add parent directory for env import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

def train():
    print("Initializing Environment...")
    base_env = LabEnv(number_of_rooms=4)
    env = FlattenObservation(base_env)
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)


    print("Initializing DQN Model...")
    # Instantiate the agent
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="tmp/logs/dqn_sb3_agent/"
    )
    
    print("Starting Training...")
    model.learn(total_timesteps=100)
    
    print("Saving Model...")
    model.save("dqn_lab_env")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
