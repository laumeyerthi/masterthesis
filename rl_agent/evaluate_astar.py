import sys
import os
import numpy as np
import heapq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO

def get_neighbors(state, env):
    agent_idx, door_state_tuple, last_idx = state
    neighbors = []
    door_states = np.array(door_state_tuple)
    
    agent_r, agent_c = env.lab.index_to_coord(agent_idx)
    deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)] 
    
    # 1. Normal Moves (0-3)
    for action_idx, (dr, dc) in enumerate(deltas):
        new_r, new_c = agent_r + dr, agent_c + dc
        if 0 <= new_r < env.grid_size and 0 <= new_c < env.grid_size:
            target_idx = env.lab.coord_to_index(new_r, new_c)
            # Normal move requires open door
            if door_states[agent_idx, target_idx] == 1:
                neighbors.append((action_idx, (target_idx, door_state_tuple, agent_idx)))
                
    # 2. Backtrack (4)
    if last_idx != -1:
        # Backtracking ignores door state
        neighbors.append((4, (last_idx, door_state_tuple, agent_idx)))

    # 3. Buttons (5+)
    buttons_here = np.where(env.lab.button_location_matrix[agent_idx] == 1)[0]
    for btn in buttons_here:
        action_idx = 5 + btn
        behavior = env.lab.button2door_behavior_matrix[btn]
        new_states = np.logical_xor(door_states, behavior).astype(int)
        new_states = new_states * env.lab.room_trans_matrix
        new_tuple = tuple(tuple(row) for row in new_states)
        # Button press does not change last_idx
        neighbors.append((action_idx, (agent_idx, new_tuple, last_idx)))
        
    return neighbors

def heuristic(state, env):
    agent_idx, _, _ = state
    agent_r, agent_c = env.lab.index_to_coord(agent_idx)
    goal_r, goal_c = env.lab.index_to_coord(env.lab.goal_room)
    return abs(agent_r - goal_r) + abs(agent_c - goal_c)

def a_star_search(env):
    agent_r, agent_c = env.agent_location
    start_idx = env.lab.coord_to_index(agent_r, agent_c)
    # env.last_pos is initialized in reset, so we should match that
    last_r, last_c = env.last_pos
    if last_r == -1:
        start_last_idx = -1
    else:
        start_last_idx = env.lab.coord_to_index(last_r, last_c)

    start_state = (start_idx, tuple(tuple(row) for row in env.lab.door_state_matrix), start_last_idx)
    goal_idx = env.lab.goal_room
    
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
            
        for action, next_state in get_neighbors(current, env):
            new_cost = cost_so_far[current] + 1 
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state, env)
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

def evaluate_agent(agent_name, env, seeds_to_run, model=None, is_recurrent=False):
    rewards = []
    lengths = []
    successes = 0

    for seed in seeds_to_run:
        obs, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0
        episode_length = 0

        if is_recurrent and model is not None:
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

        if agent_name == "A*":
            path = a_star_search(env)
            if len(path) == 0 and env.lab.coord_to_index(*env.agent_location) == env.lab.goal_room:
                 done = True
            
            for action in path:
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                if done:
                    break
            
            if not done and len(path) == 0:
                 obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                 episode_reward += reward
                 episode_length += 1
        else:
            while not done:
                action_masks = env.action_masks()
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
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

        rewards.append(episode_reward)
        lengths.append(episode_length)
        if env.lab.coord_to_index(*env.agent_location) == env.lab.goal_room:
            successes += 1

    return np.mean(rewards), np.std(rewards), np.mean(lengths), successes / len(seeds_to_run) * 100

def main():
    num_episodes = 200
    seeds_to_run = list(range(100000, 100000 + num_episodes))
    env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    
    print(f"Evaluating over {num_episodes} episodes using consistent seeds...")
    print("-" * 65)
    print(f"{'Agent':<20} | {'Success%':<10} | {'Mean Len':<10} | {'Mean Reward'}")
    print("-" * 65)

    mean_rew_astar, std_rew_astar, mean_len_astar, succ_astar = evaluate_agent("A*", env, seeds_to_run)
    print(f"{'A* Search':<20} | {succ_astar:<10.1f} | {mean_len_astar:<10.2f} | {mean_rew_astar:.2f} +/- {std_rew_astar:.2f}")

    # PPO Masked
    try:
        model_path_masked = os.path.join(os.path.dirname(__file__), "..", "ppo_masked_button_env")
        model_masked = MaskablePPO.load(model_path_masked)
        mean_rew_pm, std_rew_pm, mean_len_pm, succ_pm = evaluate_agent("PPO Masked Button", env, seeds_to_run, model=model_masked, is_recurrent=False)
        print(f"{'PPO Masked':<20} | {succ_pm:<10.1f} | {mean_len_pm:<10.2f} | {mean_rew_pm:.2f} +/- {std_rew_pm:.2f}")
    except Exception as e:
        print(f"{'PPO Masked':<20} | {'Error loading':<10} | {'-':<10} | {str(e)}")

    # PPO MR Curr
    try:
        model_path_mr = os.path.join(os.path.dirname(__file__), "..", "ppo_lab_env")
        model_mr = PPO.load(model_path_mr)
        mean_rew_mr, std_rew_mr, mean_len_mr, succ_mr = evaluate_agent("PPO MR CURR", env, seeds_to_run, model=model_mr, is_recurrent=True)
        print(f"{'PPO':<20} | {succ_mr:<10.1f} | {mean_len_mr:<10.2f} | {mean_rew_mr:.2f} +/- {std_rew_mr:.2f}")
    except Exception as e:
        print(f"{'PPO MR':<20} | {'Error loading':<10} | {'-':<10} | {str(e)}")
    
    # PPO MR tuned vec with button matrix
    try:
        model_path_mr = os.path.join(os.path.dirname(__file__), "..", "alphastar_transformer_finetuned")
        model_mr = RecurrentMaskablePPO.load(model_path_mr)
        mean_rew_mr, std_rew_mr, mean_len_mr, succ_mr = evaluate_agent("PPO MR Tuned", env, seeds_to_run, model=model_mr, is_recurrent=True)
        print(f"{'Alphastar FT 50':<20} | {succ_mr:<10.1f} | {mean_len_mr:<10.2f} | {mean_rew_mr:.2f} +/- {std_rew_mr:.2f}")
    except Exception as e:
        print(f"{'PPO MR':<20} | {'Error loading':<10} | {'-':<10} | {str(e)}")

    # try:
    #     model_path_bc = os.path.join(os.path.dirname(__file__), "..", "alphastar_transformer_bc_pretrained")
    #     model_bc = RecurrentMaskablePPO.load(model_path_bc)
    #     mean_rew_bc, std_rew_bc, mean_len_bc, succ_bc = evaluate_agent("Transformer BC", env, seeds_to_run, model=model_bc, is_recurrent=True)
    #     print(f"{'Alphastar TF BC':<20} | {succ_bc:<10.1f} | {mean_len_bc:<10.2f} | {mean_rew_bc:.2f} +/- {std_rew_bc:.2f}")
    # except Exception as e:
    #     print(f"{'Alphastar TF BC':<20} | {'Error loading':<10} | {'-':<10} | {str(e)}")

    # # Dynamic Transformer Evaluation
    # try:
    #     from gymnasium_env.wrappers.variable_labyrinth_wrapper import VariableLabyrinthWrapper
    #     env_wrapped = VariableLabyrinthWrapper(env, max_rooms=9, max_buttons=4)
        
    #     # BC Pretrained Dynamic Model
    #     model_path_dyn_bc = os.path.join(os.path.dirname(__file__), "..", "dynamic_transformer_bc_pretrained")
    #     if os.path.exists(model_path_dyn_bc + ".zip"):
    #         model_dyn_bc = RecurrentMaskablePPO.load(model_path_dyn_bc)
    #         mean_rew_dyn, std_rew_dyn, mean_len_dyn, succ_dyn = evaluate_agent("Dynamic TF BC", env_wrapped, seeds_to_run, model=model_dyn_bc, is_recurrent=True)
    #         print(f"{'Dynamic TF BC':<20} | {succ_dyn:<10.1f} | {mean_len_dyn:<10.2f} | {mean_rew_dyn:.2f} +/- {std_rew_dyn:.2f}")
    #     else:
    #         print(f"{'Dynamic TF BC':<20} | {'Not Trained':<10} | {'-':<10} | run pretraining first")

    #     # Finetuned Dynamic Model
    #     model_path_dyn_ft = os.path.join(os.path.dirname(__file__), "..", "dynamic_transformer_finetuned")
    #     if os.path.exists(model_path_dyn_ft + ".zip"):
    #         model_dyn_ft = RecurrentMaskablePPO.load(model_path_dyn_ft)
    #         mean_rew_dyn, std_rew_dyn, mean_len_dyn, succ_dyn = evaluate_agent("Dynamic TF PPO", env_wrapped, seeds_to_run, model=model_dyn_ft, is_recurrent=True)
    #         print(f"{'Dynamic TF PPO':<20} | {succ_dyn:<10.1f} | {mean_len_dyn:<10.2f} | {mean_rew_dyn:.2f} +/- {std_rew_dyn:.2f}")
    #     else:
    #         print(f"{'Dynamic TF PPO':<20} | {'Not Trained':<10} | {'-':<10} | run finetuning first")
    # except Exception as e:
    #     print(f"{'Dynamic Agent':<20} | {'Error loading':<10} | {'-':<10} | {str(e)}")

    print("-" * 65)

if __name__ == '__main__':
    main()
