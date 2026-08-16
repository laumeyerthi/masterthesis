import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))

import torch
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
import numpy as np

class PPOMRInterface:
    def __init__(self):
        self.model = RecurrentMaskablePPO.load("ppo_mr_env_NB_B")
        self.current_lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

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
    
    def get_action(self, obs, action_mask):
        with torch.no_grad():
            action, _ = self.model.predict(obs,action_masks = action_mask, deterministic=True)
        return action
    
    def update_state(self,obs, action_mask):
        with torch.no_grad():
            _, self.current_lstm_states = self.model.predict(obs, action_masks=action_mask, state=self.current_lstm_states, episode_start=self.episode_starts, deterministic=True)
            self.episode_starts = np.zeros((1,), dtype=bool)
        
    def reset_state(self):
        self.episode_starts = np.ones((1,), dtype=bool)