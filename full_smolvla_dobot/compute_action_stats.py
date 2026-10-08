import os
import glob
import numpy as np
import torch
from tqdm import tqdm

def compute_action_stats(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Scanning {len(files)} demonstrations to compute true Action Mean and Std...")
    
    all_actions = []
    
    for f in tqdm(files):
        try:
            d = np.load(f, allow_pickle=True)
            proprio = d['proprioception'].astype(np.float32)
            acts = d['actions'].astype(np.float32)
            
            # The model predicts [x, y, z, grip]. We need to recreate the exact
            # raw trajectory tensor that FastFullSmolVLADataset uses.
            # In train_full_smolvla.py:
            # raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1)
            raw_traj = np.concatenate([proprio[:, :3], acts[:, 4:5]], axis=-1)
            all_actions.append(raw_traj)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_actions:
        return

    # Flatten all steps across all trajectories
    all_actions_concat = np.concatenate(all_actions, axis=0)
    
    mean = np.mean(all_actions_concat, axis=0)
    std = np.std(all_actions_concat, axis=0)
    
    print("\n" + "="*60)
    print("ACTION_MEAN = torch.tensor([" + ", ".join([f"{x:.4f}" for x in mean]) + "], dtype=torch.float32, device=self.device)")
    print("ACTION_STD  = torch.tensor([" + ", ".join([f"{x:.4f}" for x in std]) + "], dtype=torch.float32, device=self.device)")
    print("="*60)
    print("Copy and paste these exact lines into `full_smolvla_model.py`!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_dir, "data", "demonstrations")
    compute_action_stats(data_directory)
