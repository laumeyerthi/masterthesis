import gymnasium as gym
import numpy as np
import collections
import heapq
import os
import sys
import torch as th
from torch.nn import functional as F
import argparse

# Add parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))

from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

class CustomEntityTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.num_rooms = observation_space["door_states"].shape[0]
        self.num_buttons = observation_space["button_locations"].shape[1]
        self.grid_size = int(np.sqrt(self.num_rooms))
        
        self.entity_dim = 3 + self.num_buttons + self.num_rooms + (self.num_buttons * self.num_rooms)
        
        d_model = 128
        nhead = 4
        num_layers = 3
        
        self.input_norm = nn.LayerNorm(self.entity_dim)
        self.embedding = nn.Linear(self.entity_dim, d_model)
        
        self.pos_embedding = nn.Embedding(self.num_rooms, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * self.num_rooms, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations["agent_location"].shape[0]
        device = observations["agent_location"].device
        
        rooms_idx = th.arange(self.num_rooms, device=device)
        rooms_x = (rooms_idx // self.grid_size).float().view(1, self.num_rooms).expand(batch_size, -1)
        rooms_y = (rooms_idx % self.grid_size).float().view(1, self.num_rooms).expand(batch_size, -1)
        
        agent_x = observations["agent_location"][:, 0].float().unsqueeze(1)
        agent_y = observations["agent_location"][:, 1].float().unsqueeze(1)
        goal_x = observations["goal_location"][:, 0].float().unsqueeze(1)
        goal_y = observations["goal_location"][:, 1].float().unsqueeze(1)
        last_x = observations["last_pos"][:, 0].float().unsqueeze(1)
        last_y = observations["last_pos"][:, 1].float().unsqueeze(1)
        
        is_agent_here = ((rooms_x == agent_x) & (rooms_y == agent_y)).float().unsqueeze(2)
        is_goal_here = ((rooms_x == goal_x) & (rooms_y == goal_y)).float().unsqueeze(2)
        is_last_here = ((rooms_x == last_x) & (rooms_y == last_y)).float().unsqueeze(2)
        
        btn_locs = observations["button_locations"].float() 
        door_states = observations["door_states"].float() 
        behavior = observations["button_door_behavior"].float() 
        behavior_per_room = behavior.permute(0, 2, 1, 3).reshape(batch_size, self.num_rooms, -1) 
        
        entities = th.cat([
            is_agent_here, is_goal_here, is_last_here,
            btn_locs,
            door_states,
            behavior_per_room
        ], dim=-1) 
        
        entities = self.input_norm(entities)
        
        pos_emb = self.pos_embedding(rooms_idx).unsqueeze(0).expand(batch_size, -1, -1)
        emb = self.embedding(entities) + pos_emb
        
        out = self.transformer(emb)
        
        out = out.reshape(batch_size, -1)
        return self.output_proj(out)

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
        episode_masks = []
        
        for action in opt_actions:
            episode_masks.append(env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(obs)
            episode_acts.append(action)
            if terminated or truncated:
                break
                
        trajectories.append({
            'obs': episode_obs,
            'acts': episode_acts,
            'masks': episode_masks
        })
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_episodes} trajectories...")
            
    return trajectories

def pretrain_bc():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Generating expert demonstrations utilizing A*...")
    trajectories = generate_expert_demonstrations_dict(env, num_episodes=100000)
    print(f"Collected {len(trajectories)} trajectories.")
    
    print("Initializing RecurrentMaskablePPO Model...")
    model = RecurrentMaskablePPO(
        "MultiInputLstmPolicy", 
        env,
        policy_kwargs=dict(features_extractor_class=CustomEntityTransformer, features_extractor_kwargs=dict(features_dim=256)),
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        tensorboard_log="tmp/logs/alphastar_transformer_agent/"
    )
    
    print("Pre-training policy via Behavioral Cloning...")
    optimizer = th.optim.Adam(model.policy.parameters(), lr=3e-4)
    epochs = 30
    batch_size_trajs = 64
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        np.random.shuffle(trajectories)
        optimizer.zero_grad()
        
        for idx, traj in enumerate(trajectories):
            obs_list = traj['obs'][:-1]
            acts_list = traj['acts']
            masks_list = traj['masks']
            
            if len(acts_list) == 0:
                continue
                
            obs_dict = {}
            for k in obs_list[0].keys():
                obs_dict[k] = th.tensor(np.stack([o[k] for o in obs_list]), device=model.device)
                
            acts_tensor = th.tensor(acts_list, device=model.device)
            masks_tensor = th.tensor(np.stack(masks_list), device=model.device)
            
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
                action_masks=masks_tensor
            )
            
            loss = -log_prob.mean() / batch_size_trajs
            loss.backward()
            
            total_loss += (loss.item() * batch_size_trajs)
            n_batches += 1
            
            if (idx + 1) % batch_size_trajs == 0 or (idx + 1) == len(trajectories):
                th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
            
        print(f"Epoch {epoch+1}/{epochs} | BC Loss: {total_loss/n_batches:.4f}")
        
    print("Saving pre-trained Model...")
    model.save("alphastar_transformer_bc_pretrained")
    print("BC Pre-training finished and model saved.")

def train_ppo():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Loading BC model for KL penalty...")
    bc_model = RecurrentMaskablePPO.load("alphastar_transformer_bc_pretrained", env=env, device="auto")
    bc_model.policy.set_training_mode(False)
    bc_model.policy.eval()
    for param in bc_model.policy.parameters():
        param.requires_grad = False
     
    print("Loading BC pre-trained Model for PPO...")
    model = RecurrentMaskablePPO.load(
        "alphastar_transformer_bc_pretrained",
        env=env,
        verbose=0,
        tensorboard_log="tmp/logs/alphastar_transformer_agent/",
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
    model.bc_kl_coef = 0.01 
    
    print("Starting PPO Fine-tuning...")
    warmup_callback = WarmUpCallback(warmup_timesteps=100000)
    model.learn(total_timesteps=500000, callback=warmup_callback, progress_bar=True)

    print("Saving Fine-tuned Model...")
    model.save("alphastar_transformer_finetuned")
    print("PPO Fine-tuning finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)
    print(f"Eval Reward: {mean_reward}")
    
def train_ppo_free():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Loading finetuned Model...")
    model = RecurrentMaskablePPO.load(
        "alphastar_transformer_finetuned",
        env=env,
        verbose=0,
        tensorboard_log="tmp/logs/alphastar_transformer_agent/",
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
    
    print("Starting PPO Fine-tuning free...")
    model.learn(total_timesteps=500000, progress_bar=True)

    print("Saving Fine-tuned Model...")
    model.save("alphastar_transformer_finetuned_free")
    print("PPO Fine-tuning finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
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
    parser.add_argument("--freetune", action="store_true", help="Run PPO fine-tuning on finetuned model")
    parser.add_argument("--eval_bc", action="store_true", help="Evaluate BC pre-trained model")
    parser.add_argument("--eval_ppo", action="store_true", help="Evaluate PPO fine-tuned model")
    args = parser.parse_args()

    if args.pretrain:
        pretrain_bc()  
    elif args.finetune:
        train_ppo()
    elif args.eval_bc:
        eval_model("alphastar_transformer_bc_pretrained")
    elif args.eval_ppo:
        eval_model("alphastar_transformer_finetuned")
    elif args.freetune:
        train_ppo_free()
    else:
        print("Please provide an argument: --pretrain, --finetune, --eval_bc, or --eval_ppo")
