import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import optuna

# Add parent directory for env import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv


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

class MaskableTrialEvalCallback(BaseCallback):
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
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    
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
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        **kwargs,
        verbose=0,
        tensorboard_log="tmp/logs/ppo_masked_sb3_optuna/"
    )

    eval_callback = MaskableTrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=10000)

    try:
        model.learn(total_timesteps=500000, callback=eval_callback, progress_bar=False)
    except Exception as e:
        print(f"Exception during learning: {e}")
        return float("-inf")
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    return mean_reward

def tune():
    print("Starting Optuna tuning for MaskablePPO...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    study.optimize(objective, n_trials=30)
    print("Best hyperparameters: ", study.best_params)

def train():
    print("Initializing Environment...")
    env = LabEnv(number_of_rooms=9, valid_seeds="train")
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    print("Initializing PPO Model...")
    model = MaskablePPO(
        "MultiInputPolicy", 
        env,
        **para,
        verbose=0,
        tensorboard_log="tmp/logs/ppo_masked_sb3_agent/"
    )
    print("Starting Training...")
    model.learn(total_timesteps=500000,progress_bar=True)
    
    print("Saving Model...") 
    model.save("ppo_masked_button_env")
    print("Training finished and model saved.")
    
    eval_env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10,return_episode_rewards=True, deterministic=True)
    print(mean_reward)

def eval():
    model = MaskablePPO.load("ppo_masked_lab_env")
    env = LabEnv(number_of_rooms=9, valid_seeds="eval")
    
    mean_reward,_ = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    
    print(mean_reward)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    args = parser.parse_args()

    if args.tune:
        tune()
    elif args.eval:
        eval()
    else:
        train()
