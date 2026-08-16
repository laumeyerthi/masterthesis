import torch
from sb3_contrib import RecurrentPPO

class PPORecurrentInterface:
    def __init__(self):
        self.model = RecurrentPPO.load("ppo_recurrent_lab_env")

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
    
    def get_action(self, obs, lstm_states, episode_start):
        with torch.no_grad():
            action, _ = self.model.predict(obs,state=lstm_states, episode_start=episode_start, deterministic=True)
        return action