import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv
from rl_agent.evaluate_astar_replan import AgentEvaluator
from rl_agent.alphastar_transformer_agent import CustomEntityTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../libraries/recurrent_maskable')))
from libraries.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO

def benchmark_scaling():
    room_sizes = [9, 16, 25, 100] # 3x3, 4x4, 5x5
    episodes = 200 # Just a few episodes is enough to measure time
    
    print(f"{'Rooms':<10} | {'A* Time (ms)':<15} | {'Untrained RL Time (ms)':<25}")
    print("-" * 55)
    
    for rooms in room_sizes:
        env = LabEnv(number_of_rooms=rooms, valid_seeds="eval")
        evaluator = AgentEvaluator(env)
        
        # 1. Benchmark A*
        seeds = list(range(100000, 100000 + episodes))
        # _, _, _, _, astar_time = evaluator.evaluate_agent("A*", seeds)
        astar_time = 0
        # 2. Benchmark RL Agent (Untrained AlphaStar Transformer)
        # We don't need a trained model to measure inference speed!
        # A forward pass takes the same amount of time regardless of the weights.
        dummy_model = RecurrentMaskablePPO(
            "MultiInputLstmPolicy", 
            env,
            policy_kwargs=dict(
                features_extractor_class=CustomEntityTransformer,
                features_extractor_kwargs=dict(features_dim=256)
            ),
            verbose=0
        )
        
        _, _, _, _, rl_time = evaluator.evaluate_agent("RL", seeds, model=dummy_model, is_recurrent=True)
        
        print(f"{rooms:<10} | {astar_time:<15.2f} | {rl_time:<25.2f}")

if __name__ == '__main__':
    benchmark_scaling()
