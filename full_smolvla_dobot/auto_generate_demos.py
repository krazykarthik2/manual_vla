import os
import sys
import time
import math
import numpy as np
import concurrent.futures

# Headless Pygame Settings (must be set before pygame is imported/initialized)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demonstrations")
os.makedirs(DATA_DIR, exist_ok=True)

def count_saved_demos():
    return len([f for f in os.listdir(DATA_DIR) if f.startswith("demo_") and f.endswith(".npz")])

def get_next_demo_index():
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith("demo_") and f.endswith(".npz")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            num = int(f.replace("demo_", "").replace(".npz", ""))
            indices.append(num)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0

def generate_smooth_trajectory(p_start, p_end, steps):
    if steps <= 1:
        return [p_end.copy()]
    pts = []
    for i in range(steps):
        s = i / float(steps - 1)
        s_quintic = 10 * (s ** 3) - 15 * (s ** 4) + 6 * (s ** 5)
        pt = p_start + s_quintic * (p_end - p_start)
        pts.append(pt)
    return pts

def calc_velocity_steps(p1, p2, target_step_size=0.008, min_steps=8):
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    jitter = np.random.uniform(0.90, 1.10)
    steps = int(max(min_steps, round((dist / target_step_size) * jitter)))
    return steps

def get_augmented_prompt(action, target_color, target_plat_color):
    cube_synonyms = ["cube", "block", "object", "item", "box"]
    plat_synonyms = ["platform", "base", "plate", "target pad", "mat", "spot", "landing zone"]
    
    cb = str(np.random.choice(cube_synonyms))
    pl = str(np.random.choice(plat_synonyms))
    
    if action == "pick_place":
        templates = [
            f"pick up the {target_color} {cb} and place it on the {target_plat_color} {pl}",
            f"pick the {target_color} {cb} and drop it onto the {target_plat_color} {pl}",
            f"grasp the {target_color} {cb} and set it down on the {target_plat_color} {pl}",
            f"move the {target_color} {cb} to the {target_plat_color} {pl}",
            f"transport the {target_color} {cb} onto the {target_plat_color} {pl}",
            f"grab the {target_color} {cb}, then place it on the {target_plat_color} {pl}",
            f"lift the {target_color} {cb} and put it on the {target_plat_color} {pl}",
            f"take the {target_color} {cb} and deliver it to the {target_plat_color} {pl}"
        ]
    elif action == "push":
        templates = [
            f"push the {target_color} {cb} towards the {target_plat_color} {pl}",
            f"slide the {target_color} {cb} into the {target_plat_color} {pl}",
            f"shove the {target_color} {cb} near the {target_plat_color} {pl}",
            f"nudge the {target_color} {cb} towards the {target_plat_color} {pl}",
            f"push the {target_color} {cb} onto the {target_plat_color} {pl}",
            f"glide the {target_color} {cb} to the {target_plat_color} {pl}",
            f"ram the {target_color} {cb} forward into the {target_plat_color} {pl}",
            f"displace the {target_color} {cb} towards the {target_plat_color} {pl}"
        ]
    elif action == "sweep":
        templates = [
            f"sweep the {target_color} {cb} to the {target_plat_color} {pl}",
            f"sweep the {target_color} {cb} across to the {target_plat_color} {pl}",
            f"brush the {target_color} {cb} towards the {target_plat_color} {pl}",
            f"rake the {target_color} {cb} to the {target_plat_color} {pl}",
            f"sweep the {target_color} {cb} into the target zone",
            f"paddle the {target_color} {cb} towards the {target_plat_color} {pl}",
            f"sweep the {target_color} {cb} over to the {target_plat_color} {pl}"
        ]
    else:
        templates = [
            f"pick up the {target_color} {cb} and place it on the {target_plat_color} {pl}"
        ]
    return str(np.random.choice(templates))

def generate_single_demo(demo_idx, task_type=None):
    pygame.init()
    
    sim = DobotPickPlaceSim()
    # Task distribution: pick_place (50%), push (25%), sweep (25%)
    if task_type is not None:
        act_choice = task_type
    else:
        act_choice = str(np.random.choice(["pick_place", "push", "sweep"], p=[0.50, 0.25, 0.25]))

    obs_dict = sim.reset(random_scene=True, num_distractors=2, action_type=act_choice)
    
    target_start = sim.target_cube_pos.copy()
    platform_target = sim.target_platform_pos.copy()
    
    prompt_text = get_augmented_prompt(act_choice, sim.target_color, sim.target_plat_color)

    img_list = []
    proprio_list = []
    legacy_obs_list = []
    action_list = []

    hover_cube_z = 0.12
    p_start = sim.ee_pos[:3].copy()
    p_hover_cube = np.array([target_start[0], target_start[1], hover_cube_z], dtype=np.float32)
    
    if act_choice == "pick_place":
        p_cube_surface = np.array([target_start[0], target_start[1], 0.026], dtype=np.float32)
        p_hover_plat = np.array([platform_target[0], platform_target[1], hover_cube_z], dtype=np.float32)
        p_plat_surface = np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32)
        p_retract = np.array([platform_target[0], platform_target[1], 0.12], dtype=np.float32)

        stages = [
            ("Approach", p_start, p_hover_cube, 0.0, 0.0, calc_velocity_steps(p_start, p_hover_cube, 0.009, 14)),
            ("Descend", p_hover_cube, p_cube_surface, 0.0, 0.0, calc_velocity_steps(p_hover_cube, p_cube_surface, 0.007, 12)),
            ("Grasp", p_cube_surface, p_cube_surface, 1.0, 0.0, int(np.random.randint(5, 8))),
            ("Lift", p_cube_surface, p_hover_cube, 1.0, 0.0, calc_velocity_steps(p_cube_surface, p_hover_cube, 0.007, 12)),
            ("Carry", p_hover_cube, p_hover_plat, 1.0, 0.0, calc_velocity_steps(p_hover_cube, p_hover_plat, 0.009, 16)),
            ("Lower", p_hover_plat, p_plat_surface, 1.0, 0.0, calc_velocity_steps(p_hover_plat, p_plat_surface, 0.007, 12)),
            ("Release", p_plat_surface, p_plat_surface, 0.0, 1.0, int(np.random.randint(5, 8))),
            ("Retract", p_plat_surface, p_retract, 0.0, 1.0, calc_velocity_steps(p_plat_surface, p_retract, 0.008, 10)),
        ]
    elif act_choice == "push":
        push_dir = (platform_target[:2] - target_start[:2])
        dist = np.linalg.norm(push_dir)
        if dist > 0.01:
            push_dir /= dist
        else:
            push_dir = np.array([1.0, 0.0])

        behind_pos = target_start[:2] - push_dir * 0.040
        push_dest = platform_target[:2] + push_dir * 0.010

        p_behind_high = np.array([behind_pos[0], behind_pos[1], 0.080], dtype=np.float32)
        p_behind_low = np.array([behind_pos[0], behind_pos[1], 0.015], dtype=np.float32)
        p_push_end = np.array([push_dest[0], push_dest[1], 0.015], dtype=np.float32)
        p_retract = np.array([push_dest[0], push_dest[1], 0.100], dtype=np.float32)
        p_home = np.array([0.20, 0.0, 0.12], dtype=np.float32)

        stages = [
            ("Move Behind", p_start, p_behind_high, 0.0, 0.0, calc_velocity_steps(p_start, p_behind_high, 0.009, 14)),
            ("Lower Behind", p_behind_high, p_behind_low, 0.0, 0.0, calc_velocity_steps(p_behind_high, p_behind_low, 0.007, 10)),
            ("Push Toward", p_behind_low, p_push_end, 0.0, 0.0, calc_velocity_steps(p_behind_low, p_push_end, 0.004, 30)),
            ("Retract", p_push_end, p_retract, 0.0, 1.0, calc_velocity_steps(p_push_end, p_retract, 0.008, 10)),
            ("Return Home", p_retract, p_home, 0.0, 1.0, calc_velocity_steps(p_retract, p_home, 0.009, 14)),
        ]
    elif act_choice == "sweep":
        # Sweep: Gripper approaches the lateral side of the cube, stays closed/paddle mode,
        # and sweeps across toward the platform target
        sweep_vec = (platform_target[:2] - target_start[:2])
        dist = np.linalg.norm(sweep_vec)
        if dist > 0.01:
            u_vec = sweep_vec / dist
        else:
            u_vec = np.array([1.0, 0.0])
        # Orthogonal normal vector
        perp_vec = np.array([-u_vec[1], u_vec[0]], dtype=np.float32)
        sweep_side = np.random.choice([-1.0, 1.0])
        
        # Position slightly behind and to the side to sweep inward
        side_start = target_start[:2] - u_vec * 0.035 + perp_vec * (sweep_side * 0.035)
        sweep_end = platform_target[:2] + u_vec * 0.015

        p_side_high = np.array([side_start[0], side_start[1], 0.080], dtype=np.float32)
        p_side_low = np.array([side_start[0], side_start[1], 0.015], dtype=np.float32)
        p_sweep_end = np.array([sweep_end[0], sweep_end[1], 0.015], dtype=np.float32)
        p_retract = np.array([sweep_end[0], sweep_end[1], 0.100], dtype=np.float32)
        p_home = np.array([0.20, 0.0, 0.12], dtype=np.float32)

        stages = [
            ("Move Lateral", p_start, p_side_high, 0.0, 0.0, calc_velocity_steps(p_start, p_side_high, 0.009, 14)),
            ("Lower Paddle", p_side_high, p_side_low, 0.0, 0.0, calc_velocity_steps(p_side_high, p_side_low, 0.007, 10)),
            ("Sweep Arc", p_side_low, p_sweep_end, 0.0, 0.0, calc_velocity_steps(p_side_low, p_sweep_end, 0.004, 30)),
            ("Retract", p_sweep_end, p_retract, 0.0, 1.0, calc_velocity_steps(p_sweep_end, p_retract, 0.008, 10)),
            ("Return Home", p_retract, p_home, 0.0, 1.0, calc_velocity_steps(p_retract, p_home, 0.009, 14)),
        ]

    for stage_name, start_pt, end_pt, grip_state, succ_state, steps in stages:
        pts = generate_smooth_trajectory(start_pt, end_pt, steps)
        for pt in pts:
            vla_obs = sim.get_vla_observation()
            img_list.append(vla_obs["image"])
            proprio_list.append(vla_obs["proprio"])
            legacy_obs_list.append(sim.get_observation())
            
            current_ee = sim.ee_pos.copy()
            dx = pt[0] - current_ee[0]
            dy = pt[1] - current_ee[1]
            dz = pt[2] - current_ee[2]
            dyaw = 0.0
            
            action_vec = np.array([dx, dy, dz, dyaw, grip_state], dtype=np.float32)
            action_list.append(action_vec)
            sim.step_delta(action_vec)

    # Verify task success according to task type!
    final_cube_dist_to_plat = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
    
    if act_choice == "pick_place":
        task_success = (final_cube_dist_to_plat < 0.045 and sim.target_cube_pos[2] <= 0.030 and not sim.gripper_closed)
    elif act_choice in ["push", "sweep"]:
        # For pushing and sweeping, success requires the cube being transported within 0.050m of platform
        # without requiring lifting or gripper release
        task_success = (final_cube_dist_to_plat < 0.050)
    else:
        task_success = False

    if task_success and len(action_list) > 15:
        save_path = os.path.join(DATA_DIR, f"demo_{demo_idx:06d}.npz")
        np.savez_compressed(
            save_path,
            images=np.array(img_list, dtype=np.float32),
            proprioception=np.array(proprio_list, dtype=np.float32),
            legacy_obs=np.array(legacy_obs_list, dtype=np.float32),
            actions=np.array(action_list, dtype=np.float32),
            prompt=np.array([prompt_text], dtype=str)
        )
        return True
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=50)
    args = parser.parse_args()
    
    num_demos = args.num_demos
    start_count = count_saved_demos()
    print(f"Current saved demonstrations: {start_count}")
    print(f"Generating {num_demos} demonstrations in parallel (Autoscaling CPU workers)...")

    start_idx = get_next_demo_index()
    demo_indices = range(start_idx, start_idx + num_demos)
    
    success_count = 0
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_single_demo, idx) for idx in demo_indices]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                success_count += 1
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{num_demos} trajectories...")
                
    elapsed = time.time() - start_time
    print(f"\n=================================")
    print(f"Generation Complete! ({success_count}/{num_demos} successful)")
    print(f"Time Taken: {elapsed:.1f}s")
    print(f"Total Database Size: {count_saved_demos()} trajectories")
    print(f"=================================")
