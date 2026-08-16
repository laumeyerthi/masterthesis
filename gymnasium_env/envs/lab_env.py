import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from .lab_generator import LabGenerator

class LabEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, number_of_rooms=4, valid_seeds=None, max_rooms=None):
        self.valid_seeds = valid_seeds
        self.num_rooms = number_of_rooms 
        self.max_rooms = max_rooms if max_rooms is not None else number_of_rooms
        
        self.lab = LabGenerator(number_of_rooms=self.num_rooms)
        self.grid_size = self.lab.grid_size
        self.max_grid_size = int(np.sqrt(self.max_rooms))
        
        self.action_space = spaces.Discrete(5 + self.max_rooms)
        
        # Observations
        self.observation_space = spaces.Dict({
            "agent_location": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            "goal_location": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            "door_states": spaces.Box(0, 1, shape=(self.num_rooms, self.num_rooms), dtype=int),
            "button_locations": spaces.Box(0, 1, shape=(self.num_rooms, self.lab.number_of_buttons), dtype=int),
            "last_pos": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            "button_door_behavior": spaces.Box(0, 1, shape=(self.lab.number_of_buttons, self.num_rooms, self.num_rooms), dtype=int),
        })
        # self.observation_space = spaces.Dict({
        #     # Coordinates (using MultiDiscrete for X,Y pairs)
        #     "agent_location": spaces.MultiDiscrete([self.grid_size, self.grid_size]),
        #     "goal_location":  spaces.MultiDiscrete([self.grid_size, self.grid_size]),
        #     "last_pos":       spaces.MultiDiscrete([self.grid_size, self.grid_size]),

        #     # Binary Grids/Matrices (using MultiBinary for 0/1 states)
        #     "door_states": spaces.MultiBinary((self.num_rooms, self.num_rooms)),
            
        #     "button_locations": spaces.MultiBinary(
        #         (self.num_rooms, self.lab.number_of_buttons)
        #     ),
            
        #     "button_door_behavior": spaces.MultiBinary(
        #         (self.lab.number_of_buttons, self.num_rooms, self.num_rooms)
        #     ),
        # })
        
        self.render_mode = render_mode
        agent_r, agent_c = self.lab.index_to_coord(self.lab.start_room)
        self.agent_location = np.array([agent_r, agent_c])
        self.steps = 0
        
        # Rendering
        self.window = None
        self.clock = None
        self.window_size = 512
        
        # rewards
        self.reward_step = -0.1
        self.reward_goal = 10.0
        self.reward_invalid = -0.5
        
        # seeds
        self.train_seeds = list(range(0, 1000000))
        self.eval_seeds = list(range(10000000, 10001000))
        
        if valid_seeds == "train":
            self.valid_seeds = self.train_seeds
        elif valid_seeds == "eval":
            self.valid_seeds = self.eval_seeds
        else:
            self.valid_seeds = None
            
        # Hook up caching framework
        self.precalc_data = None
        self.precalc_seeds_map = {}
        self._load_precalc_data()
            
    def _load_precalc_data(self):
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, "..", "..", "datasets", f"mazes_{self.num_rooms}.npz")
        
        if os.path.exists(dataset_path):
            try:
                data = np.load(dataset_path)
                # Parse all arrays into direct memory 
                self.precalc_data = {k: v for k, v in data.items()}
                # Create extremely fast seed-to-index lookup mapping
                self.precalc_seeds_map = {seed: idx for idx, seed in enumerate(self.precalc_data["seeds"])}
            except Exception as e:
                print(f"[LabEnv] Failed to parse dataset {dataset_path}: {e}")
                self.precalc_data = None
        else:
            self.precalc_data = None
            
    def set_curriculum_stage(self, number_of_rooms):
        if number_of_rooms > self.max_rooms:
            raise ValueError(f"Curriculum room size {number_of_rooms} exceeds configured max_rooms={self.max_rooms}")
        self.num_rooms = number_of_rooms
        self.lab = LabGenerator(number_of_rooms=self.num_rooms)
        self.grid_size = self.lab.grid_size
        self._load_precalc_data()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)        
        
        if self.valid_seeds is not None:
            lab_seed = int(self.np_random.choice(self.valid_seeds))
        else:
            lab_seed = int(self.np_random.integers(0, 2**31 - 1))
            
        # Hook into RAM dataset arrays to skip physical maze generation completely
        if self.precalc_data is not None and lab_seed in self.precalc_seeds_map:
            idx = self.precalc_seeds_map[lab_seed]
            self.lab.start_room = self.precalc_data["start_room"][idx]
            self.lab.goal_room = self.precalc_data["goal_room"][idx]
            self.lab.room_trans_matrix = self.precalc_data["room_trans_matrix"][idx]
            self.lab.door_state_matrix = self.precalc_data["door_state_matrix"][idx]
            self.lab.button_location_matrix = self.precalc_data["button_location_matrix"][idx]
            self.lab.button2door_behavior_matrix = self.precalc_data["button2door_behavior_matrix"][idx]
        else:
            self.lab.generate_lab(seed=lab_seed)        
            
        start_idx = self.lab.start_room
        agent_r, agent_c = self.lab.index_to_coord(start_idx)
        self.agent_location = np.array([agent_r, agent_c])
        self.last_pos = self.agent_location
        self.steps = 0
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def step(self, action):
        reward = self.reward_step
        terminated = False
        truncated = False
        self.steps += 1
        if(self.steps >100):
            truncated = True
            
        
        current_r, current_c = self.agent_location
        current_idx = self.lab.coord_to_index(current_r, current_c)
        
        if action < 5: # Move
            # Calc new pos
            new_r, new_c = current_r, current_c
            if action == 0: new_c += 1 # Right
            elif action == 1: new_r -= 1 # Up
            elif action == 2: new_c -= 1 # Left
            elif action == 3: new_r += 1 # Down
            elif action == 4: # Backtrack
                if self.last_pos[0] != -1: # Only valid if last_pos is set
                    new_r, new_c = self.last_pos
            
            # Check bounds
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                target_idx = self.lab.coord_to_index(new_r, new_c)
                
                # Check if door connects and is open (unless backtracking)
                is_open = self.lab.door_state_matrix[current_idx, target_idx] == 1
                is_backtrack = (action == 4)
                
                if is_open or is_backtrack:
                    # Move successful
                    self.last_pos = self.agent_location.copy()
                    self.agent_location = np.array([new_r, new_c])
                    
                    # Check Goal
                    if target_idx == self.lab.goal_room:
                        terminated = True
                        reward = self.reward_goal
                else:
                    # Blocked (Wall or Closed Door)
                    reward = self.reward_invalid
                    pass
        else: # Button
            btn_idx = action - 5
            # Check if button exists in current room
            if btn_idx < self.lab.number_of_buttons:
                if self.lab.button_location_matrix[current_idx, btn_idx] == 1:
                    # Toggle doors
                    behavior = self.lab.button2door_behavior_matrix[btn_idx]
                    
                    # XOR current states with behavior
                    current_states = self.lab.door_state_matrix
                    new_states = np.logical_xor(current_states, behavior).astype(int)
                    
                    # Enforce Walls stay Walls (TransMatrix == 0 -> State = 0)
                    new_states = new_states * self.lab.room_trans_matrix
                    
                    self.lab.door_state_matrix = new_states
                else:
                    # Button not in current room
                    reward = self.reward_invalid
            else:
                # Button index out of bounds
                reward = -2
                pass
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        goal_r, goal_c = self.lab.index_to_coord(self.lab.goal_room)
        goal_r, goal_c = self.lab.index_to_coord(self.lab.goal_room)
        return {
            "agent_location": self.agent_location,
            "goal_location": np.array([goal_r, goal_c], dtype=int),
            "door_states": self.lab.door_state_matrix.copy().astype(int),
            "button_locations": self.lab.button_location_matrix.copy().astype(int),
            "last_pos": self.last_pos,
            "button_door_behavior": self.lab.button2door_behavior_matrix.copy().astype(int),
        }
        
        # obs = {
        #     "agent_location": self.agent_location,
        #     "goal_location": np.array([goal_r, goal_c], dtype=int),
        #     "door_states": self.lab.door_state_matrix.copy().astype(np.int8),
        #     "button_locations": self.lab.button_location_matrix.copy().astype(np.int8),
        #     "last_pos": self.last_pos,
        #     "button_door_behavior": self.lab.button2door_behavior_matrix.copy().astype(np.int8),
        # }
        
        # for key, value in obs.items():
        #     space = self.observation_space[key]
        #     if not space.contains(value):
        #         print(f"CRITICAL: {key} is out of bounds!")
        #         print(f"Value: {value}")
        #         print(f"Space: {space}")
                
        # return obs

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.grid_size)

        # Draw Goal
        goal_r, goal_c = self.lab.index_to_coord(self.lab.goal_room)
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * goal_c,
                pix_square_size * goal_r,
                pix_square_size,
                pix_square_size,
            ),
        )

        # Draw Grid Lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Draw Walls and Doors
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr_idx = self.lab.coord_to_index(r, c)
                
                # Right Connection
                if c + 1 < self.grid_size:
                    right_idx = self.lab.coord_to_index(r, c + 1)
                    has_connection = self.lab.room_trans_matrix[curr_idx, right_idx] == 1
                    is_open = self.lab.door_state_matrix[curr_idx, right_idx] == 1
                    
                    start_pos = ((c + 1) * pix_square_size, r * pix_square_size)
                    end_pos = ((c + 1) * pix_square_size, (r + 1) * pix_square_size)
                    
                    if not has_connection:
                        # Draw Wall
                        pygame.draw.line(canvas, (0, 0, 0), start_pos, end_pos, width=5)
                    else:
                        # Draw Door
                        color = (0, 255, 0) if is_open else (255, 0, 0)
                        pygame.draw.line(canvas, color, start_pos, end_pos, width=5)

                # Down Connection
                if r + 1 < self.grid_size:
                    down_idx = self.lab.coord_to_index(r + 1, c)
                    has_connection = self.lab.room_trans_matrix[curr_idx, down_idx] == 1
                    is_open = self.lab.door_state_matrix[curr_idx, down_idx] == 1
                    
                    start_pos = (c * pix_square_size, (r + 1) * pix_square_size)
                    end_pos = ((c + 1) * pix_square_size, (r + 1) * pix_square_size)
                    
                    if not has_connection:
                        # Draw Wall
                        pygame.draw.line(canvas, (0, 0, 0), start_pos, end_pos, width=5)
                    else:
                        # Draw Door
                        color = (0, 255, 0) if is_open else (255, 0, 0)
                        pygame.draw.line(canvas, color, start_pos, end_pos, width=5)

        # Draw Buttons
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr_idx = self.lab.coord_to_index(r, c)
                buttons_here = np.where(self.lab.button_location_matrix[curr_idx] == 1)[0]
                
                for i, btn_idx in enumerate(buttons_here):
                    offset_x = 0.2 + (i * 0.2)
                    offset_y = 0.2
                    
                    if offset_x > 0.8:
                        offset_x = 0.2 + ((i % 3) * 0.2)
                        offset_y = 0.4
                    
                    center = (
                        int((c + offset_x) * pix_square_size),
                        int((r + offset_y) * pix_square_size),
                    )
                    
                    match btn_idx:
                        case 0:
                            pygame.draw.circle(canvas, (0, 0, 255), center, 5)
                        case 1:
                            pygame.draw.circle(canvas, (0, 255, 0), center, 5)
                        case 2:
                            pygame.draw.circle(canvas, (255, 0, 0), center, 5)
                        case 3:
                            pygame.draw.circle(canvas, (255, 255, 0), center, 5)
        # Draw Agent
        pygame.draw.circle(
            canvas,
            (100, 100, 100),
            (
                (self.agent_location[1] + 0.5) * pix_square_size,
                (self.agent_location[0] + 0.5) * pix_square_size,
            ),
            20
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata.get("render_fps", 4))
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def action_masks(self):
        mask = np.zeros(5 + self.max_rooms, dtype=np.int8)
        current_r, current_c = self.agent_location
        current_idx = self.lab.coord_to_index(current_r, current_c)
        
        # 1. Check Moves (Right, Up, Left, Down)
        deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        
        for i, (dr, dc) in enumerate(deltas):
            new_r, new_c = current_r + dr, current_c + dc
            
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                target_idx = self.lab.coord_to_index(new_r, new_c)
                mask[i] = self.lab.door_state_matrix[current_idx, target_idx]
        
        # 2. Check Backtrack (Action 4)
        if self.last_pos[0] != -1:
            mask[4] = 1
            
        # 3. Check Buttons
        mask[5:5+self.lab.number_of_buttons] = self.lab.button_location_matrix[current_idx]
        
        return mask
