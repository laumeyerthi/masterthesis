import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import sys
import os
import optuna

# Add parent directory for env import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
from libraries.recurrent_maskable.common.evaluation import evaluate_policy

#optuna hyperparameter
para = {
        "learning_rate": 7.494003701284142e-05,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "clip_range": 0.4,
        "ent_coef": 1.3230232060196722e-07,
    }

class MR_TrialEvalCallback(BaseCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, deterministic=True, verbose=0):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=self.deterministic
            )
            self.trial.report(mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def sample_ppo_params(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = 128
    batch_size = 128
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 0.1, log=True)
    
    if batch_size > n_steps:
        batch_size = n_steps

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
    }

def objective(trial):
    kwargs = sample_ppo_params(trial)
    num_cpu = 8
    env = DummyVecEnv([make_env(i, rooms=9, seeds="train") for i in range(num_cpu)])
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")

    model = RecurrentMaskablePPO(
        "MultiInputLstmPolicy",
        env,
        **kwargs,
        verbose=0,
        tensorboard_log="tmp/logs/ppo_mr_optuna/"
    )

    eval_freq = max(10000 // num_cpu, 1)
    eval_callback = MR_TrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=eval_freq)

    try:
        model.learn(total_timesteps=600000, callback=eval_callback, progress_bar=True)
    except Exception as e:
        print(f"Exception during learning: {e}")
        return float("-inf")
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    return mean_reward

def tune():
    print("Starting Optuna tuning for RecurrentMaskablePPO...")
    storage_name = "sqlite:///my_rl_study.db"
    study = optuna.create_study(direction="maximize", 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
                                study_name="ppo_mr_9_70_1",
                                storage=storage_name, 
                                load_if_exists=True
                                )
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL:
            study.enqueue_trial(trial.params)
    study.optimize(objective, n_trials=70, show_progress_bar=True)
    print("Best hyperparameters: ", study.best_params)

def train():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    print("Initializing PPO Model...")
    model = RecurrentMaskablePPO(
        "MultiInputLstmPolicy", 
        env,
        **para,
        verbose=0,
        tensorboard_log="tmp/logs/ppo_mr_agent/"
    ) 
    print("Starting Training...")
    model.learn(total_timesteps=1000000,progress_bar=True)
    
    print("Saving Model...")
    model.save("ppo_mr_env")
    print("Training finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10,return_episode_rewards=True, deterministic=True)
    print(mean_reward)
    
def eval():
    model = RecurrentMaskablePPO.load("ppo_mr_env")
    env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    
    mean_reward,_ = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    
    print(mean_reward)

def make_env(rank, seed=0, rooms=9, seeds="train", max_rooms=None):
    def _init():
        env = LabEnv(number_of_rooms=rooms, valid_seeds=seeds, max_rooms=max_rooms)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_vec():
    print("Initializing Vector Environment...")
    num_cpu = 16 
    env = DummyVecEnv([make_env(i) for i in range(num_cpu)])
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    print("Initializing PPO Model...")
    model = RecurrentMaskablePPO(
        "MultiInputLstmPolicy", 
        env,
        **para,
        verbose=0,
        tensorboard_log="tmp/logs/ppo_mr_agent_vec/"
    ) 
    print("Starting Vector Training...")
    model.learn(total_timesteps=1000000, progress_bar=True)
    
    print("Saving Vector Model...")
    model.save("ppo_mr_vec_env")
    print("Vector Training finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)
    print("Eval reward: ", mean_reward)

def train_curriculum():
    print("Initializing Curriculum Vector Environment... (Max Rooms: 9)")
    num_cpu = 16
    env = VecMonitor(DummyVecEnv([make_env(i, rooms=4, max_rooms=9) for i in range(num_cpu)]))
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    print("Initializing PPO Model...")
    model = RecurrentMaskablePPO(
        "MultiInputLstmPolicy", 
        env,
        **para,
        verbose=0,
        tensorboard_log="tmp/logs/ppo_mr_curriculum_vec/"
    )
    
    stages = [
        {"rooms": 4, "timesteps": 22000},
        {"rooms": 9, "timesteps": 38000},
        # {"rooms": 16, "timesteps": 50000},
    ]
    
    for stage in stages:
        num_rooms = stage["rooms"]
        timesteps = stage["timesteps"]
        print(f"--- Starting Curriculum Stage: {num_rooms} Rooms for {timesteps} steps ---")
        env.env_method("set_curriculum_stage", num_rooms)
        model.learn(total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=False)
    
    print("Saving Curriculum Model...")
    model.save("ppo_mr_curriculum_env")
    print("Curriculum Training finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval", max_rooms=9)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True, deterministic=True)
    print(f"Eval Reward (9 Rooms): {mean_reward}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--train_vec", action="store_true", help="Run vectorized training")
    parser.add_argument("--curriculum", action="store_true", help="Run curriculum training")
    args = parser.parse_args()

    if args.tune:
        tune()
    elif args.eval:
        eval()
    elif args.train_vec:
        train_vec()
    elif args.curriculum:
        train_curriculum()
    else:
        train()
