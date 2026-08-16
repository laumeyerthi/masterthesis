import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))
import torch
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
from libraries.recurrent_maskable.common.buffers import RNNStates
import numpy as np

class AlphastarInterface:
    def __init__(self):
        self.model = RecurrentMaskablePPO.load("alphastar_transformer_finetuned")
        lstm = self.model.policy.lstm_actor
        self.shape = (lstm.num_layers, 1, lstm.hidden_size)
        self.current_lstm_states = RNNStates(
            (torch.zeros(self.shape, device=self.model.device), torch.zeros(self.shape, device=self.model.device)),
            (torch.zeros(self.shape, device=self.model.device), torch.zeros(self.shape, device=self.model.device))
        )
        self.episode_starts = np.ones((1,), dtype=bool)

    def get_action_probs(self, obs,action_mask):
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        self.episode_starts = torch.as_tensor(self.episode_starts, device=self.model.device)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor,lstm_states=self.current_lstm_states[0], episode_starts=self.episode_starts)
            action_probs = distribution.distribution.probs
        return action_probs

    def get_winning_probs(self,obs):
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            values = self.model.policy.predict_values(obs_tensor, lstm_states=self.current_lstm_states[1], episode_starts=self.episode_starts)
        return values
    
    def get_action(self, obs, action_mask):
        with torch.no_grad():
            action, _ = self.model.predict(obs,action_masks = action_mask, deterministic=True)
        return action
    
    def update_state(self,obs, action_mask):
        with torch.no_grad():
            _, self.current_lstm_states = self.model.predict(obs, action_masks=action_mask, deterministic=True)
            self.episode_starts = np.zeros((1,), dtype=bool)
        return None
    
    def reset_state(self):
        lstm = self.model.policy.lstm_actor
        shape = (lstm.num_layers, 1, lstm.hidden_size)
        self.current_lstm_states = RNNStates(
            (torch.zeros(shape, device=self.model.device), torch.zeros(shape, device=self.model.device)),
            (torch.zeros(shape, device=self.model.device), torch.zeros(shape, device=self.model.device))
        )
        self.episode_starts = np.ones((1,), dtype=bool)