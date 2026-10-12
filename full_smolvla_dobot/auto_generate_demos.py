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

def generate_single_demo(demo_idx, task_type="pick_place"):
    pygame.init()
    
    sim = DobotPickPlaceSim()
    act_choice = task_type

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

        behind_pos = target_start[:2] - push_dir * 0.024
        push_dest = platform_target[:2] + push_dir * 0.005

        p_behind_high = np.array([behind_pos[0], behind_pos[1], 0.080], dtype=np.float32)
        p_behind_low = np.array([behind_pos[0], behind_pos[1], 0.015], dtype=np.float32)
        p_push_end = np.array([push_dest[0], push_dest[1], 0.015], dtype=np.float32)
        p_retract = np.array([push_dest[0], push_dest[1], 0.100], dtype=np.float32)
        p_home = np.array([0.20, 0.0, 0.12], dtype=np.float32)

        stages = [
            ("Move Behind", p_start, p_behind_high, 0.0, 0.0, calc_velocity_steps(p_start, p_behind_high, 0.009, 14)),
            ("Lower Behind", p_behind_high, p_behind_low, 0.0, 0.0, calc_velocity_steps(p_behind_high, p_behind_low, 0.006, 12)),
            ("Push Toward", p_behind_low, p_push_end, 0.0, 0.0, calc_velocity_steps(p_behind_low, p_push_end, 0.003, 35)),
            ("Retract", p_push_end, p_retract, 0.0, 1.0, calc_velocity_steps(p_push_end, p_retract, 0.008, 10)),
            ("Return Home", p_retract, p_home, 0.0, 1.0, calc_velocity_steps(p_retract, p_home, 0.009, 14)),
        ]
    elif act_choice == "sweep":
        sweep_vec = (platform_target[:2] - target_start[:2])
        dist = np.linalg.norm(sweep_vec)
        if dist > 0.01:
            u_vec = sweep_vec / dist
        else:
            u_vec = np.array([1.0, 0.0])
        
        # Lateral paddle sweep
        side_start = target_start[:2] - u_vec * 0.022
        sweep_dest = platform_target[:2] + u_vec * 0.005

        p_side_high = np.array([side_start[0], side_start[1], 0.080], dtype=np.float32)
        p_side_low = np.array([side_start[0], side_start[1], 0.015], dtype=np.float32)
        p_sweep_end = np.array([sweep_dest[0], sweep_dest[1], 0.015], dtype=np.float32)
        p_retract = np.array([sweep_dest[0], sweep_dest[1], 0.100], dtype=np.float32)
        p_home = np.array([0.20, 0.0, 0.12], dtype=np.float32)

        stages = [
            ("Move Lateral", p_start, p_side_high, 0.0, 0.0, calc_velocity_steps(p_start, p_side_high, 0.009, 14)),
            ("Lower Paddle", p_side_high, p_side_low, 0.0, 0.0, calc_velocity_steps(p_side_high, p_side_low, 0.006, 12)),
            ("Sweep Paddle", p_side_low, p_sweep_end, 0.0, 0.0, calc_velocity_steps(p_side_low, p_sweep_end, 0.003, 35)),
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

def count_demos_by_task():
    """Count how many demos exist for each task type in the dataset directory."""
    counts = {"pick_place": 0, "push": 0, "sweep": 0}
    for f in os.listdir(DATA_DIR):
        if f.startswith("demo_") and f.endswith(".npz"):
            path = os.path.join(DATA_DIR, f)
            try:
                data = np.load(path, allow_pickle=True)
                p = str(data["prompt"][0]).lower() if "prompt" in data else ""
                if "sweep" in p or "brush" in p or "rake" in p or "paddle" in p:
                    counts["sweep"] += 1
                elif "push" in p or "slide" in p or "shove" in p or "nudge" in p or "glide" in p or "ram" in p or "displace" in p:
                    counts["push"] += 1
                else:
                    counts["pick_place"] += 1
            except Exception:
                pass
    return counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=12000, help="Total target number of successful demonstrations to achieve.")
    parser.add_argument("--pick_ratio", type=float, default=0.50, help="Ratio of pick_place demonstrations.")
    parser.add_argument("--push_ratio", type=float, default=0.25, help="Ratio of push demonstrations.")
    parser.add_argument("--sweep_ratio", type=float, default=0.25, help="Ratio of sweep demonstrations.")
    parser.add_argument("--batch_workers", type=int, default=None, help="Number of parallel processes.")
    args = parser.parse_args()
    
    target_total = args.num_demos
    current_counts = count_demos_by_task()
    total_existing = sum(current_counts.values())
    
    target_pick = int(round(target_total * args.pick_ratio))
    target_push = int(round(target_total * args.push_ratio))
    target_sweep = target_total - target_pick - target_push

    print(f"===========================================================")
    print(f"Goal: Guarantee exactly {target_total} valid demonstrations")
    print(f"Targets: pick_place={target_pick}, push={target_push}, sweep={target_sweep}")
    print(f"Current: pick_place={current_counts['pick_place']}, push={current_counts['push']}, sweep={current_counts['sweep']} (Total={total_existing})")
    print(f"===========================================================")

    if total_existing >= target_total:
        print(f"Database already has {total_existing} >= {target_total} demos! Nothing to do.")
        sys.exit(0)

    start_time = time.time()
    max_workers = args.batch_workers or os.cpu_count() or 4
    
    # Continuous quota loop until targets for every task are reached
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        while True:
            current_counts = count_demos_by_task()
            needed_pick = max(0, target_pick - current_counts["pick_place"])
            needed_push = max(0, target_push - current_counts["push"])
            needed_sweep = max(0, target_sweep - current_counts["sweep"])
            needed_total = needed_pick + needed_push + needed_sweep

            if needed_total == 0:
                break

            # Pick which task needs more demos
            candidate_tasks = []
            candidate_weights = []
            if needed_pick > 0:
                candidate_tasks.append("pick_place")
                candidate_weights.append(needed_pick)
            if needed_push > 0:
                candidate_tasks.append("push")
                candidate_weights.append(needed_push)
            if needed_sweep > 0:
                candidate_tasks.append("sweep")
                candidate_weights.append(needed_sweep)
            
            p_dist = np.array(candidate_weights, dtype=np.float32) / sum(candidate_weights)

            # Dispatch a chunk of parallel jobs
            chunk_size = min(max_workers * 4, needed_total * 2, 200)
            next_idx = get_next_demo_index()
            
            job_tasks = [np.random.choice(candidate_tasks, p=p_dist) for _ in range(chunk_size)]
            futures = [
                executor.submit(generate_single_demo, next_idx + i, job_tasks[i])
                for i in range(chunk_size)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

            current_total = count_saved_demos()
            print(f"Progress: {current_total}/{target_total} valid demos saved | Remaining: pick={needed_pick}, push={needed_push}, sweep={needed_sweep}")

    elapsed = time.time() - start_time
    final_counts = count_demos_by_task()
    print(f"\n=================================")
    print(f"Quota Reached! Database Size: {count_saved_demos()} trajectories")
    print(f"Task Breakdown: pick_place={final_counts['pick_place']}, push={final_counts['push']}, sweep={final_counts['sweep']}")
    print(f"Time Elapsed: {elapsed:.1f}s")
    print(f"=================================")
