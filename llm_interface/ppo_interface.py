import torch
from stable_baselines3 import PPO

class PPOInterface:
    def __init__(self):
        self.model = PPO.load("ppo_lab_env")

    def get_action_probs(self, obs):
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs
        return action_probs

    def get_winning_probs(self,obs):
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            values = self.model.policy.predict_values(obs_tensor)
        return values
    
    def get_action(self, obs):
        with torch.no_grad():
            action, _ = self.model.predict(obs, deterministic=True)
        return action