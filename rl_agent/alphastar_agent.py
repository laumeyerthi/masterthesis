import gymnasium as gym
import numpy as np
import collections
import heapq
import os
import sys
import torch as th
from torch.nn import functional as F
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))

from gymnasium_env.envs.lab_env import LabEnv
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
from libraries.recurrent_maskable.common.evaluation import evaluate_policy
from libraries.recurrent_maskable.common.buffers import RNNStates
from stable_baselines3.common.callbacks import BaseCallback

class WarmUpCallback(BaseCallback):
    def __init__(self, warmup_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.warmup_timesteps = warmup_timesteps
        self.actor_unfrozen = False

    def _on_training_start(self) -> None:
        if self.warmup_timesteps > 0:
            print(f"Freezing actor for the first {self.warmup_timesteps} timesteps to warm up the critic...")
            self._set_actor_grad(False)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.warmup_timesteps and not self.actor_unfrozen:
            print(f"Warmup phase over at {self.num_timesteps} timesteps. Unfreezing actor...")
            self._set_actor_grad(True)
            self.actor_unfrozen = True
        return True

    def _set_actor_grad(self, requires_grad: bool):
        policy = self.model.policy
        for param in policy.action_net.parameters():
            param.requires_grad = requires_grad
            
        for param in policy.lstm_actor.parameters():
            param.requires_grad = requires_grad
            
        for param in policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = requires_grad
            
        if not policy.share_features_extractor:
            for param in policy.pi_features_extractor.parameters():
                param.requires_grad = requires_grad
def a_star_solve(unwrapped_env):
    lab = unwrapped_env.lab
     
    h_dist = np.full(lab.number_of_rooms, np.inf)
    h_dist[lab.goal_room] = 0
    q = collections.deque([lab.goal_room])
    while q:
        curr = q.popleft()
        for neighbor in range(lab.number_of_rooms):
            if lab.room_trans_matrix[curr, neighbor] == 1 and h_dist[neighbor] == np.inf:
                h_dist[neighbor] = h_dist[curr] + 1
                q.append(neighbor)
    
    start_room = lab.start_room
    start_state = (start_room, 0, -1)
    
    pq = [(h_dist[start_room], 0, 0, start_state, [])]
    visited = {start_state: 0} 
    idx = 1
    
    while pq:
        f, _, g, state, path = heapq.heappop(pq)
        curr_room, curr_mask, last_room = state
        
        if curr_room == lab.goal_room:
            return path
            
        curr_r, curr_c = lab.index_to_coord(curr_room)
        
        curr_doors_row = lab.door_state_matrix[curr_room].copy()
        for btn_idx in range(lab.number_of_buttons):
            if (curr_mask >> btn_idx) & 1:
                curr_doors_row = np.bitwise_xor(curr_doors_row, lab.button2door_behavior_matrix[btn_idx][curr_room])
                
        deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)] # Right, Up, Left, Down
        for action, (dr, dc) in enumerate(deltas):
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < lab.grid_size and 0 <= nc < lab.grid_size:
                neighbor = lab.coord_to_index(nr, nc)
                if lab.room_trans_matrix[curr_room, neighbor] == 1 and curr_doors_row[neighbor] == 1:
                    next_state = (neighbor, curr_mask, curr_room)
                    next_g = g + 1
                    if next_state not in visited or next_g < visited[next_state]:
                        visited[next_state] = next_g
                        heapq.heappush(pq, (next_g + h_dist[neighbor], idx, next_g, next_state, path + [action]))
                        idx += 1
                        
        if last_room != -1:
            next_state = (last_room, curr_mask, curr_room)
            next_g = g + 1
            if next_state not in visited or next_g < visited[next_state]:
                visited[next_state] = next_g
                heapq.heappush(pq, (next_g + h_dist[last_room], idx, next_g, next_state, path + [4]))
                idx += 1
                
        available_buttons = np.where(lab.button_location_matrix[curr_room] == 1)[0]
        for btn_idx in available_buttons:
            next_mask = curr_mask ^ (1 << btn_idx)
            next_state = (curr_room, next_mask, last_room)
            next_g = g + 1
            if next_state not in visited or next_g < visited[next_state]:
                visited[next_state] = next_g
                heapq.heappush(pq, (next_g + h_dist[curr_room], idx, next_g, next_state, path + [5 + btn_idx]))
                idx += 1
                
    return []

def generate_expert_demonstrations_dict(env, num_episodes=50):
    trajectories = []
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        unwrapped_env = env.unwrapped
        
        opt_actions = a_star_solve(unwrapped_env)
        
        if not opt_actions:
            print(f"Warning: Episode {i} is not solvable. Skipping...")
            continue
            
        episode_obs = [obs]
        episode_acts = []
        
        for action in opt_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(obs)
            episode_acts.append(action)
            if terminated or truncated:
                break
                
        trajectories.append({
            'obs': episode_obs,
            'acts': episode_acts
        })
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_episodes} trajectories...")
            
    return trajectories

def pretrain_bc():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Generating expert demonstrations utilizing A*...")
    trajectories = generate_expert_demonstrations_dict(env, num_episodes=500000)
    print(f"Collected {len(trajectories)} trajectories.")
    
    print("Initializing RecurrentMaskablePPO Model...")
    model = RecurrentMaskablePPO(
        "MultiInputLstmPolicy", 
        env,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        tensorboard_log="tmp/logs/alphastar_agent/"
    )
    
    print("Pre-training policy via Behavioral Cloning...")
    optimizer = th.optim.Adam(model.policy.parameters(), lr=1e-3)
    epochs = 20
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for traj in trajectories:
            obs_list = traj['obs'][:-1]
            acts_list = traj['acts']
            
            if len(acts_list) == 0:
                continue
                
            obs_dict = {}
            for k in obs_list[0].keys():
                obs_dict[k] = th.tensor(np.stack([o[k] for o in obs_list]), device=model.device)
                
            acts_tensor = th.tensor(acts_list, device=model.device)
            
            episode_starts = th.zeros(len(acts_list), dtype=th.float32, device=model.device)
            episode_starts[0] = 1.0
            
            lstm = model.policy.lstm_actor
            shape = (lstm.num_layers, 1, lstm.hidden_size)
            lstm_states = RNNStates(
                (th.zeros(shape, device=model.device), th.zeros(shape, device=model.device)),
                (th.zeros(shape, device=model.device), th.zeros(shape, device=model.device))
            )
            
            values, log_prob, entropy = model.policy.evaluate_actions(
                obs_dict, 
                acts_tensor, 
                lstm_states, 
                episode_starts,
                
            )
            
            loss = -log_prob.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        print(f"Epoch {epoch+1}/{epochs} | BC Loss: {total_loss/n_batches:.4f}")
        
    print("Saving pre-trained Model...")
    model.save("alphastar_bc_pretrained_50000")
    print("BC Pre-training finished and model saved.")

def train_ppo():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Loading BC model for KL penalty...")
    bc_model = RecurrentMaskablePPO.load("alphastar_bc_pretrained_50000", env=env, device="auto")
    bc_model.policy.set_training_mode(False)
    bc_model.policy.eval()
    for param in bc_model.policy.parameters():
        param.requires_grad = False
    
    print("Loading BC pre-trained Model for PPO...")
    model = RecurrentMaskablePPO.load(
        "alphastar_bc_pretrained_50000",
        env=env,
        verbose=0,
        tensorboard_log="tmp/logs/alphastar_agent/",
        custom_objects={
            "learning_rate": 5.0e-06,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 5,
            "gamma": 0.90,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 1.3e-07,
        }
    )
    
    model.bc_policy = bc_model.policy
    model.bc_kl_coef = 1000000.00 
    
    print("Starting PPO Fine-tuning...")
    warmup_callback = WarmUpCallback(warmup_timesteps=0)
    # model.learn(total_timesteps=500000, callback=warmup_callback, progress_bar=True)
    model.learn(total_timesteps=50000, progress_bar=True)

    
    print("Saving Fine-tuned Model...")
    model.save("alphastar_finetuned_nKLnW")
    print("PPO Fine-tuning finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)
    print(f"Eval Reward: {mean_reward}")

def eval_model(model_path):
    print(f"Evaluating {model_path}...")
    model = RecurrentMaskablePPO.load(model_path)
    env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    print(f"Mean Reward: {mean_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true", help="Run Behavioral Cloning pretraining")
    parser.add_argument("--finetune", action="store_true", help="Run PPO fine-tuning on pretrained model")
    parser.add_argument("--eval_bc", action="store_true", help="Evaluate BC pre-trained model")
    parser.add_argument("--eval_ppo", action="store_true", help="Evaluate PPO fine-tuned model")
    args = parser.parse_args()

    if args.pretrain:
        pretrain_bc()  
    elif args.finetune:
        train_ppo()
    elif args.eval_bc:
        eval_model("alphastar_bc_pretrained_50000")
    elif args.eval_ppo:
        eval_model("alphastar_finetuned")
    else:
        print("Please provide an argument: --pretrain, --finetune, --eval_bc, or --eval_ppo")
