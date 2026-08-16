import sys
import os
import numpy as np
import multiprocessing as mp

# Provide fallback for tqdm if not installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, suppressing progress bar.")
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add root project folder to python path to import the Generator module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gymnasium_env.envs.lab_generator import LabGenerator

def generate_single_maze(args):
    seed, num_rooms = args
    # Construct Lab
    generator = LabGenerator(number_of_rooms=num_rooms)
    generator.generate_lab(seed=seed)
    
    # Return layout properties
    return {
        "seed": seed,
        "start_room": generator.start_room,
        "goal_room": generator.goal_room,
        "room_trans_matrix": generator.room_trans_matrix,
        "door_state_matrix": generator.door_state_matrix,
        "button_location_matrix": generator.button_location_matrix,
        "button2door_behavior_matrix": generator.button2door_behavior_matrix
    }

def build_dataset(num_rooms, seeds, output_file):
    print(f"\n[Size {num_rooms}] Building dataset. Total seeds: {len(seeds)}")
    args = [(seed, num_rooms) for seed in seeds]
    
    # Map generation asynchronously
    with mp.Pool(mp.cpu_count()) as pool:
        # chunksize of 100 handles IPC overhead well
        results = list(tqdm(pool.imap(generate_single_maze, args, chunksize=100), total=len(args)))
        
    print(f"[Size {num_rooms}] Stacking results into continuous arrays...")
    
    # Stack dictionaries
    stacked = {
        "seeds": np.array([r["seed"] for r in results]),
        "start_room": np.array([r["start_room"] for r in results]),
        "goal_room": np.array([r["goal_room"] for r in results]),
        "room_trans_matrix": np.array([r["room_trans_matrix"] for r in results]),
        "door_state_matrix": np.array([r["door_state_matrix"] for r in results]),
        "button_location_matrix": np.array([r["button_location_matrix"] for r in results]),
        "button2door_behavior_matrix": np.array([r["button2door_behavior_matrix"] for r in results]),
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, **stacked)
    
    mb_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[Size {num_rooms}] Saved seamlessly -> {output_file} ({mb_size:.2f} MB)\n")

if __name__ == "__main__":
    mp.freeze_support() # Recommended for Windows multi-processing compatibility

    train_seeds = list(range(0, 1000000))
    test_seeds = list(range(10000000, 10001000))
    all_seeds = train_seeds + test_seeds
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "datasets"))
    
    # Process grids of 2x2 and 3x3 rooms respectively
    target_room_sizes = [4, 9] 
    
    for rooms in target_room_sizes:
        output_file = os.path.join(dataset_dir, f"mazes_{rooms}.npz")
        build_dataset(rooms, all_seeds, output_file)
