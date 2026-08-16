import numpy as np
from collections import deque


class LabGenerator:
    def __init__(self, number_of_rooms=4):
        self.number_of_rooms = number_of_rooms
        self.grid_size = int(np.sqrt(self.number_of_rooms))
        assert self.grid_size ** 2 == self.number_of_rooms, "Number of rooms must be a perfect square for Grid World"

        self.room_trans_matrix = None
        self.start_room = -1
        self.goal_room = -1
        self.door_state_matrix = None
        self.rng = np.random.default_rng()
        self.button_location_matrix = None
        self.button2door_behavior_matrix = None
        self.valid_layout = False
        self.number_of_buttons = 4
        self.generate_lab()

    def get_grid_adjacency(self):
        # Returns a mask of all VALID connections in a grid
        adj = np.zeros((self.number_of_rooms, self.number_of_rooms), dtype=int)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr = r * self.grid_size + c
                # Down
                if r + 1 < self.grid_size:
                    next_node = (r + 1) * self.grid_size + c
                    adj[curr, next_node] = adj[next_node, curr] = 1
                # Right
                if c + 1 < self.grid_size:
                    next_node = r * self.grid_size + (c + 1)
                    adj[curr, next_node] = adj[next_node, curr] = 1
        # Self loop
        np.fill_diagonal(adj, 1)
        return adj

    def generate_rooms(self):
        # Generate random matrix
        rooms = self.rng.integers(0, 2, size=(self.number_of_rooms, self.number_of_rooms))
        rooms = np.triu(rooms, 1) # Upper triangle
        rooms = rooms + rooms.T # Symmetric
        np.fill_diagonal(rooms, 1) # Self connected
        
        # Apply Grid Mask
        grid_adj = self.get_grid_adjacency()
        rooms = rooms * grid_adj
        return rooms

    def sanity_check(self):
        """
        Check if there is a path from start room to goal room in the labyrinth.
        Matrix: Transition matrix (numpy array)
        start: Index of the starting room (0-based)
        goal: Index of the goal room (0-based)
        """
        # Create a queue for BFS and a set to track visited rooms
        queue = deque([self.start_room])
        visited = set()
        
        while queue:
            current = queue.popleft()
            
            # If we reached the goal room
            if current == self.goal_room:
                return True
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check all connected rooms
            for neighbor in range(self.number_of_rooms):
                if self.room_trans_matrix[current, neighbor] == 1 and neighbor not in visited:
                    queue.append(neighbor)
        
        # If BFS finishes without finding the goal room
        return False

    def generate_door_states(self):
        # Random initial states
        self.door_state_matrix = self.generate_rooms()
        # close doors where transitions are not possible (walls)
        zero_mask = self.room_trans_matrix == 0
        self.door_state_matrix[zero_mask] = 0

    def generate_button_locations(self):
        # Randomly place buttons.
        # Shape: (num_rooms, num_buttons)
        # Assuming we want random distribution.
        self.button_location_matrix = self.rng.integers(0, 2, size=(self.number_of_rooms, self.number_of_buttons))

    def generate_button2door_behavior(self):
        self.button2door_behavior_matrix = np.array([self.generate_single_button_matrix() for _ in range(self.number_of_buttons)])

    def generate_single_button_matrix(self):
        # create room x room matrix that shows which doors to toogle for one button
        single_button_matrix = self.rng.integers(0, 2, size=(self.number_of_rooms, self.number_of_rooms))
        single_button_matrix = np.triu(single_button_matrix, 1)
        single_button_matrix += single_button_matrix.T
        
        # Apply Grid Mask so buttons only affect physical doors
        grid_adj = self.get_grid_adjacency()
        single_button_matrix = single_button_matrix * grid_adj
        
        # set diagonal to zero as there are no doors that lead to the same room
        np.fill_diagonal(single_button_matrix, 0)
        return single_button_matrix

    def is_fully_solvable(self):
        """
        Performs a full state-space BFS to check if the goal is reachable
        considering walls, doors, buttons, AND backtracking.
        State: (current_room_idx, button_toggle_mask, last_room_idx)
        """
        # Initial state: Start room, no buttons toggled (mask=0), last_room = -1 (none)
        start_state = (self.start_room, 0, -1)
        
        queue = deque([start_state])
        visited = {start_state}
        
        initial_doors = self.door_state_matrix.astype(int)
        
        while queue:
            curr_room, curr_mask, last_room = queue.popleft()
            
            if curr_room == self.goal_room:
                return True
            
            # 1. Try Pressing Buttons in current room
            # Get buttons available in current room
            available_buttons = np.where(self.button_location_matrix[curr_room] == 1)[0]
            
            for btn_idx in available_buttons:
                new_mask = curr_mask ^ (1 << btn_idx) # Toggle bit
                next_state = (curr_room, new_mask, last_room)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)

            # 2. Try Backtracking (always valid if last_room is not -1)
            if last_room != -1:
                # Backtrack takes us to last_room. New last_room becomes curr_room.
                backtrack_state = (last_room, curr_mask, curr_room)
                if backtrack_state not in visited:
                    visited.add(backtrack_state)
                    queue.append(backtrack_state)

            # 3. Try Moving (Normal)
            # We need to know which doors are open in the current configuration
            
            # Start with initial state of doors for current room
            curr_doors_row = initial_doors[curr_room].copy()
            
            # Apply toggles
            for btn_idx in range(self.number_of_buttons):
                if (curr_mask >> btn_idx) & 1: # if button Toggled
                    behavior_row = self.button2door_behavior_matrix[btn_idx][curr_room]
                    curr_doors_row = np.bitwise_xor(curr_doors_row, behavior_row)
            
            # Ensure walls are respected
            valid_neighbors = np.where((curr_doors_row == 1) & (self.room_trans_matrix[curr_room] == 1))[0]
            
            for neighbor in valid_neighbors:
                next_state = (neighbor, curr_mask, curr_room)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)
                    
        return False

    def generate_lab(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        attempts = 0
        self.start_room = self.rng.integers(0, self.number_of_rooms)
        self.goal_room = self.rng.choice([x for x in range(self.number_of_rooms) if x != self.start_room])
        while True:
            attempts += 1
            # 1. Generate Layout
            self.room_trans_matrix = self.generate_rooms()
            
            # 2. Fast Fail: Check basic connectivity (Walls only)
            if not self.sanity_check():
                continue
                
            # 3. Generate Details
            self.generate_door_states()
            self.generate_button_locations()
            self.generate_button2door_behavior()
            
            # 4. Full Validation: Check if actually solvable with keys/doors
            if self.is_fully_solvable():
                # print(f"Generated solvable lab after {attempts} attempts")
                break

    def coord_to_index(self, r, c):
        return r * self.grid_size + c

    def index_to_coord(self, i):
        return np.array([i // self.grid_size, i % self.grid_size])
