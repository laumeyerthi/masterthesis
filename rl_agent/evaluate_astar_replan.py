import sys
import os
import time
import numpy as np
import heapq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO


class AgentEvaluator:
    def __init__(self, env):
        self.env = env

    def get_neighbors(self, state):
        agent_idx, door_state_tuple, last_idx = state
        neighbors = []
        door_states = np.array(door_state_tuple)
        
        agent_r, agent_c = self.env.lab.index_to_coord(agent_idx)
        deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)] 
        
        # 1. Normal Moves (0-3)
        for action_idx, (dr, dc) in enumerate(deltas):
            new_r, new_c = agent_r + dr, agent_c + dc
            if 0 <= new_r < self.env.grid_size and 0 <= new_c < self.env.grid_size:
                target_idx = self.env.lab.coord_to_index(new_r, new_c)
                # Normal move requires open door
                if door_states[agent_idx, target_idx] == 1:
                    neighbors.append((action_idx, (target_idx, door_state_tuple, agent_idx)))
                    
        # 2. Backtrack (4)
        if last_idx != -1:
            # Backtracking ignores door state
            neighbors.append((4, (last_idx, door_state_tuple, agent_idx)))

        # 3. Buttons (5+)
        buttons_here = np.where(self.env.lab.button_location_matrix[agent_idx] == 1)[0]
        for btn in buttons_here:
            action_idx = 5 + btn
            behavior = self.env.lab.button2door_behavior_matrix[btn]
            new_states = np.logical_xor(door_states, behavior).astype(int)
            new_states = new_states * self.env.lab.room_trans_matrix
            new_tuple = tuple(tuple(row) for row in new_states)
            # Button press does not change last_idx
            neighbors.append((action_idx, (agent_idx, new_tuple, last_idx)))
            
        return neighbors

    def heuristic(self, state):
        agent_idx, _, _ = state
        agent_r, agent_c = self.env.lab.index_to_coord(agent_idx)
        goal_r, goal_c = self.env.lab.index_to_coord(self.env.lab.goal_room)
        return abs(agent_r - goal_r) + abs(agent_c - goal_c)

    def a_star_search(self):
        agent_r, agent_c = self.env.agent_location
        start_idx = self.env.lab.coord_to_index(agent_r, agent_c)
        last_r, last_c = self.env.last_pos
        if last_r == -1:
            start_last_idx = -1
        else:
            start_last_idx = self.env.lab.coord_to_index(last_r, last_c)

        start_state = (start_idx, tuple(tuple(row) for row in self.env.lab.door_state_matrix), start_last_idx)
        goal_idx = self.env.lab.goal_room
        
        frontier = []
        heapq.heappush(frontier, (0, 0, start_state))
        came_from = {start_state: None}
        cost_so_far = {start_state: 0}
        action_to_reach = {}
        counter = 1
        
        found_goal_state = None
        while frontier:
            _, _, current = heapq.heappop(frontier)
            
            if current[0] == goal_idx:
                found_goal_state = current
                break
                
            for action, next_state in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1 
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.heuristic(next_state)
                    heapq.heappush(frontier, (priority, counter, next_state))
                    counter += 1
                    came_from[next_state] = current
                    action_to_reach[next_state] = action
                    
        if found_goal_state is None:
            return []
            
        path = []
        current = found_goal_state
        while current != start_state:
            prev = came_from[current]
            action = action_to_reach[current]
            path.append(action)
            current = prev
        path.reverse()
        return path

    def evaluate_agent(self, agent_name, seeds_to_run, model=None, is_recurrent=False):
        rewards = []
        lengths = []
        step_times = []
        successes = 0

        for seed in seeds_to_run:
            obs, _ = self.env.reset(seed=seed)
            done = False
            episode_reward = 0
            episode_length = 0

            if is_recurrent and model is not None:
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)

            while not done:
                start_time = time.time()
                
                if agent_name == "A*":
                    path = self.a_star_search()
                    if len(path) == 0:
                        if self.env.lab.coord_to_index(*self.env.agent_location) == self.env.lab.goal_room:
                            break
                        else:
                            action = self.env.action_space.sample()
                    else:
                        action = path[0]
                else:
                    action_masks = self.env.action_masks()
                    if model is not None:
                        if is_recurrent:
                            action, lstm_states = model.predict(
                                obs,
                                state=lstm_states,
                                episode_start=episode_starts,
                                action_masks=action_masks,
                                deterministic=True
                            )
                            episode_starts = np.zeros((1,), dtype=bool)
                        else:
                            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                        
                        if isinstance(action, np.ndarray):
                            action = action.item()
                    else:
                        action = self.env.action_space.sample()
                
                end_time = time.time()
                step_times.append((end_time - start_time) * 1000) # Store in milliseconds
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            lengths.append(episode_length)
            if self.env.lab.coord_to_index(*self.env.agent_location) == self.env.lab.goal_room:
                successes += 1

        mean_step_time = np.mean(step_times) if step_times else 0.0
        return np.mean(rewards), np.std(rewards), np.mean(lengths), successes / len(seeds_to_run) * 100, mean_step_time

def main():
    num_episodes = 200 # Reduced from 200 since A* replanning at each step is computationally heavy
    seeds_to_run = list(range(100000, 100000 + num_episodes))
    env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    evaluator = AgentEvaluator(env)
    
    print(f"Evaluating over {num_episodes} episodes using consistent seeds...")
    print("-" * 85)
    print(f"{'Agent':<20} | {'Success%':<10} | {'Mean Len':<10} | {'Step Time (ms)':<15} | {'Mean Reward'}")
    print("-" * 85)

    mean_rew_astar, std_rew_astar, mean_len_astar, succ_astar, time_astar = evaluator.evaluate_agent("A*", seeds_to_run)
    print(f"{'A* Search (Replan)':<20} | {succ_astar:<10.1f} | {mean_len_astar:<10.2f} | {time_astar:<15.2f} | {mean_rew_astar:.2f} +/- {std_rew_astar:.2f}")

    # PPO Masked
    # try:
    #     model_path_masked = os.path.join(os.path.dirname(__file__), "..", "ppo_masked_button_env")
    #     model_masked = MaskablePPO.load(model_path_masked)
    #     mean_rew_pm, std_rew_pm, mean_len_pm, succ_pm, time_pm = evaluator.evaluate_agent("PPO Masked Button", seeds_to_run, model=model_masked, is_recurrent=False)
    #     print(f"{'PPO Masked':<20} | {succ_pm:<10.1f} | {mean_len_pm:<10.2f} | {time_pm:<15.2f} | {mean_rew_pm:.2f} +/- {std_rew_pm:.2f}")
    # except Exception as e:
    #     print(f"{'PPO Masked':<20} | {'Error loading':<10} | {'-':<10} | {'-':<15} | {str(e)}")

    # # PPO MR Curr
    # try:
    #     model_path_mr = os.path.join(os.path.dirname(__file__), "..", "ppo_lab_env")
    #     model_mr = PPO.load(model_path_mr)
    #     mean_rew_mr, std_rew_mr, mean_len_mr, succ_mr, time_mr = evaluator.evaluate_agent("PPO MR CURR", seeds_to_run, model=model_mr, is_recurrent=True)
    #     print(f"{'PPO':<20} | {succ_mr:<10.1f} | {mean_len_mr:<10.2f} | {time_mr:<15.2f} | {mean_rew_mr:.2f} +/- {std_rew_mr:.2f}")
    # except Exception as e:
    #     print(f"{'PPO MR':<20} | {'Error loading':<10} | {'-':<10} | {'-':<15} | {str(e)}")

    # Alphastar Finetuned
    try:
        model_path_ft = os.path.join(os.path.dirname(__file__), "..", "alphastar_transformer_finetuned")
        model_ft = RecurrentMaskablePPO.load(model_path_ft)
        mean_rew_ft, std_rew_ft, mean_len_ft, succ_ft, time_ft = evaluator.evaluate_agent("Alphastar FT", seeds_to_run, model=model_ft, is_recurrent=True)
        print(f"{'Alphastar FT':<20} | {succ_ft:<10.1f} | {mean_len_ft:<10.2f} | {time_ft:<15.2f} | {mean_rew_ft:.2f} +/- {std_rew_ft:.2f}")
    except Exception as e:
        print(f"{'Alphastar FT':<20} | {'Error loading':<10} | {'-':<10} | {'-':<15} | {str(e)}")

    print("-" * 85)

if __name__ == '__main__':
    main()
