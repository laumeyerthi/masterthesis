import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium_env.envs.lab_env import LabEnv

from skrl.models.torch import Model, TabularMixin
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

set_seed(42)

class DiscreteObservationWrapper(gym.ObservationWrapper):
    """
    Wraps the LabEnv to handle dictionary observations and discretize them 
    for Tabular Q-Learning.
    
    WARNING: This maps complex observations to a hash index. Collisions are possible.
    The large state space of LabEnv means Tabular Q-Learning is likely to be very inefficient.
    """
    def __init__(self, env, num_states=10000):
        super().__init__(env)
        self.num_states = num_states
        self.observation_space = gym.spaces.Discrete(num_states)
        
    def observation(self, observation):
        
        state_repr = (
            observation["agent_location"].tobytes(),
            observation["goal_location"].tobytes(),
            observation["door_states"].tobytes(),
            observation["last_pos"].tobytes(),
            observation["button_locations"].tobytes()
        )
        
        state_idx = hash(state_repr) % self.num_states
        return state_idx

class EpsilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions),
                                  dtype=torch.float32, device=self.device)

    def compute(self, inputs, role):
        states = inputs["states"]
        
        # Ensure states are long indices
        if states.dtype != torch.long:
            states = states.long()

        
        batch_size = states.shape[0]
        env_indices = torch.arange(batch_size, device=self.device)
        
        # Handle case where flattened/wrapped env might give extra dimensions
        if states.dim() > 1:
            states = states.flatten()
            
        q_values = self.q_table[env_indices, states]
        
        # Greedy action
        actions = torch.argmax(q_values, dim=-1, keepdim=True)

        # Epsilon-greedy exploration
        if self.training:
            random_mask = torch.rand(batch_size, device=self.device) < self.epsilon
            random_indices = random_mask.nonzero().view(-1)
            
            if random_indices.numel() > 0:
                random_actions = torch.randint(self.num_actions, (random_indices.numel(), 1), device=self.device)
                actions[random_indices] = random_actions
                
        return actions, {}

# 1. Instantiate the environment
base_env = LabEnv(number_of_rooms=4) # Smaller room count for tabular feasibility? Default is 4.

# 2. Wrap the environment
# Create a DiscreteObservationWrapper to map dict observations to integers
env = DiscreteObservationWrapper(base_env, num_states=20000) 

# 3. Wrap for SKRL
env = wrap_env(env)
device = env.device

# 4. Instantiate the agent's models
models = {}
models["policy"] = EpsilonGreedyPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs, epsilon=0.2)

# 5. Configure the agent
cfg = Q_LEARNING_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.99
cfg["alpha"] = 0.1 # Learning rate
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/LabEnv_QLearning"

agent = Q_LEARNING(models=models,
                   memory=None,
                   cfg=cfg,
                   observation_space=env.observation_space,
                   action_space=env.action_space,
                   device=device)

# 6. Configure and start trainer
cfg_trainer = {"timesteps": 20000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

if __name__ == "__main__":
    print("Starting Q-Learning Training...")
    trainer.train()
    
    # Explicitly save the agent
    print("Saving agent to q_learning_agent.pt...")
    agent.save("q_learning_agent.pt")
    
    print("Training finished.")