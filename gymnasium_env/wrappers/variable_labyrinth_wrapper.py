import gymnasium as gym
from gymnasium import spaces
import numpy as np

class VariableLabyrinthWrapper(gym.Wrapper):
    def __init__(self, env, max_rooms=9, max_buttons=4):
        super().__init__(env)
        self.max_rooms = max_rooms
        self.max_buttons = max_buttons
        
        # Override the action space to match the max_rooms capacity
        self.action_space = spaces.Discrete(5 + self.max_rooms)
        
        # Override observation space to be uniform and include room_coordinates and room_mask
        self.observation_space = spaces.Dict({
            "agent_location": self.env.observation_space["agent_location"],
            "goal_location": self.env.observation_space["goal_location"],
            "door_states": spaces.Box(0, 1, shape=(self.max_rooms, self.max_rooms), dtype=int),
            "button_locations": spaces.Box(0, 1, shape=(self.max_rooms, self.max_buttons), dtype=int),
            "last_pos": self.env.observation_space["last_pos"],
            "button_door_behavior": spaces.Box(0, 1, shape=(self.max_buttons, self.max_rooms, self.max_rooms), dtype=int),
            "room_coordinates": spaces.Box(-1, self.max_rooms, shape=(self.max_rooms, 2), dtype=int),
            "room_mask": spaces.Box(0, 1, shape=(self.max_rooms,), dtype=int),
        })

    def _map_observation(self, obs):
        num_rooms_src = obs["door_states"].shape[0]
        num_buttons_src = obs["button_locations"].shape[1]
        
        if num_rooms_src > self.max_rooms:
            raise ValueError(f"Environment room size {num_rooms_src} exceeds wrapper max_rooms {self.max_rooms}")
        if num_buttons_src > self.max_buttons:
            raise ValueError(f"Environment button count {num_buttons_src} exceeds wrapper max_buttons {self.max_buttons}")
            
        grid_size_src = int(np.sqrt(num_rooms_src))
        grid_size_tgt = int(np.sqrt(self.max_rooms))
        
        # Compute target index for each source room index based on (r, c) coordinates
        src_idx_to_tgt_idx = np.zeros(num_rooms_src, dtype=int)
        for i in range(num_rooms_src):
            r, c = i // grid_size_src, i % grid_size_src
            src_idx_to_tgt_idx[i] = r * grid_size_tgt + c
            
        # Initialize target containers
        door_states_tgt = np.zeros((self.max_rooms, self.max_rooms), dtype=int)
        button_locations_tgt = np.zeros((self.max_rooms, self.max_buttons), dtype=int)
        button_door_behavior_tgt = np.zeros((self.max_buttons, self.max_rooms, self.max_rooms), dtype=int)
        room_coordinates = np.full((self.max_rooms, 2), -1, dtype=int)
        room_mask = np.zeros(self.max_rooms, dtype=int)
        
        # Map indices
        tgt_indices = src_idx_to_tgt_idx
        
        # 1. Door states (adjacency)
        rows, cols = np.ix_(tgt_indices, tgt_indices)
        door_states_tgt[rows, cols] = obs["door_states"]
        
        # 2. Button locations
        button_locations_tgt[tgt_indices, :num_buttons_src] = obs["button_locations"]
        
        # 3. Button door behavior
        button_door_behavior_tgt[:num_buttons_src, rows, cols] = obs["button_door_behavior"]
        
        # 4. Room coordinates
        for i in range(num_rooms_src):
            r, c = i // grid_size_src, i % grid_size_src
            tgt_i = tgt_indices[i]
            room_coordinates[tgt_i] = [r, c]
            
        # 5. Room mask
        room_mask[tgt_indices] = 1
        
        return {
            "agent_location": obs["agent_location"],
            "goal_location": obs["goal_location"],
            "door_states": door_states_tgt,
            "button_locations": button_locations_tgt,
            "last_pos": obs["last_pos"],
            "button_door_behavior": button_door_behavior_tgt,
            "room_coordinates": room_coordinates,
            "room_mask": room_mask,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._map_observation(obs), info

    def step(self, action):
        # Action is defined in the target action space.
        # Check if action is valid under active buttons/moves
        num_buttons_src = self.env.unwrapped.lab.number_of_buttons
        
        # If model outputs an action that is a button index beyond the source button count,
        # we can pass a dummy action or clamp it. Since action mask blocks it, this is a fallback.
        if action >= 5:
            btn_idx = action - 5
            if btn_idx >= num_buttons_src:
                # Invalid action - trigger env step with out of bound button index
                # The env step handles it as invalid reward.
                pass
                
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._map_observation(obs), reward, terminated, truncated, info

    def action_masks(self):
        base_mask = self.env.action_masks()
        wrapped_mask = np.zeros(5 + self.max_rooms, dtype=np.int8)
        
        # Copy moves and backtrack
        wrapped_mask[:5] = base_mask[:5]
        
        # Copy valid buttons
        num_buttons_src = self.env.unwrapped.lab.number_of_buttons
        wrapped_mask[5:5+num_buttons_src] = base_mask[5:5+num_buttons_src]
        
        return wrapped_mask
