import os
import sys
import time
import glob
import re
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demos")
os.makedirs(DATA_DIR, exist_ok=True)

def get_next_demo_index():
    files = glob.glob(os.path.join(DATA_DIR, "demo_*.npz"))
    if not files:
        return 1
    indices = []
    for f in files:
        m = re.search(r"demo_(\d+)\.npz", os.path.basename(f))
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 1

def count_saved_demos():
    return len(glob.glob(os.path.join(DATA_DIR, "demo_*.npz")))

def world_to_screen(x, y):
    sx = int(200 + (y / 0.30) * 160)
    sy = int(350 - (x / 0.35) * 300)
    return sx, sy

def world_to_side_screen(x, z, offset_x=400):
    sx = int(offset_x + 50 + (x / 0.35) * 200)
    sy = int(350 - (z / 0.25) * 280)
    return sx, sy

def generate_smooth_trajectory(start_pos, target_pos, num_steps):
    t = np.linspace(0, 1, max(num_steps, 2))
    s = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
    traj = np.outer(1 - s, start_pos) + np.outer(s, target_pos)
    return traj

def render_gui(screen, font, font_bold, sim, current_demo_num, total_in_batch, total_saved, stage_name, trajectory_len, fps_mode, cam_img):
    screen.fill((25, 27, 34))

    # Top-Down Panel
    pygame.draw.rect(screen, (35, 38, 48), (20, 20, 360, 360), border_radius=8)
    pygame.draw.circle(screen, (70, 75, 95), (200, 350), 30, 2)
    
    # 1. Distractor Platforms
    for p_col, p_pos in sim.distractor_platforms:
        px, py = world_to_screen(p_pos[0], p_pos[1])
        col = COLOR_PALETTE.get(p_col, (40, 210, 80))
        pygame.draw.rect(screen, col, (px - 18, py - 18, 36, 36), border_radius=4)
        pygame.draw.rect(screen, (120, 120, 120), (px - 18, py - 18, 36, 36), 1, border_radius=4)

    # 2. Target Platform
    tpx, tpy = world_to_screen(sim.target_platform_pos[0], sim.target_platform_pos[1])
    t_col = COLOR_PALETTE.get(sim.target_plat_color, (40, 210, 80))
    pygame.draw.rect(screen, t_col, (tpx - 20, tpy - 20, 40, 40), border_radius=4)
    pygame.draw.rect(screen, (255, 255, 255), (tpx - 20, tpy - 20, 40, 40), 2, border_radius=4)
    
    # 3. Distractor Cubes
    for c_col, c_pos in sim.distractor_cubes:
        cx, cy = world_to_screen(c_pos[0], c_pos[1])
        col = COLOR_PALETTE.get(c_col, (45, 120, 240))
        pygame.draw.rect(screen, col, (cx - 9, cy - 9, 18, 18), border_radius=2)

    # 4. Target Cube
    tcx, tcy = world_to_screen(sim.target_cube_pos[0], sim.target_cube_pos[1])
    tc_col = COLOR_PALETTE.get(sim.target_color, (240, 45, 45))
    pygame.draw.rect(screen, tc_col, (tcx - 10, tcy - 10, 20, 20), border_radius=2)
    pygame.draw.rect(screen, (255, 255, 255), (tcx - 10, tcy - 10, 20, 20), 2, border_radius=2)
    
    # End-Effector (Top)
    ex, ey = world_to_screen(sim.ee_pos[0], sim.ee_pos[1])
    grip_color = (255, 80, 80) if sim.gripper_closed else (100, 200, 255)
    pygame.draw.circle(screen, grip_color, (ex, ey), 8)
    pygame.draw.line(screen, (160, 170, 190), (200, 350), (ex, ey), 3)

    act_lbl = "PICK & PLACE" if getattr(sim, "action_type", "pick_place") == "pick_place" else "PUSH"
    top_label = font.render(f"ACTION: {act_lbl} ({sim.target_color.upper()} -> {sim.target_plat_color.upper()})", True, (170, 180, 200))
    screen.blit(top_label, (30, 30))

    # Side Elevation Panel
    pygame.draw.rect(screen, (35, 38, 48), (400, 20, 360, 360), border_radius=8)
    pygame.draw.line(screen, (60, 65, 80), (410, 350), (750, 350), 2)
    
    # Target Platform (Side)
    psx, psy = world_to_side_screen(sim.target_platform_pos[0], sim.target_platform_pos[2])
    pygame.draw.rect(screen, t_col, (psx - 20, psy - 4, 40, 8), border_radius=2)

    # Target Cube (Side)
    csx, csy = world_to_side_screen(sim.target_cube_pos[0], sim.target_cube_pos[2])
    pygame.draw.rect(screen, tc_col, (csx - 8, csy - 8, 16, 16), border_radius=2)

    # End-Effector (Side)
    esx, esy = world_to_side_screen(sim.ee_pos[0], sim.ee_pos[2])
    pygame.draw.circle(screen, grip_color, (esx, esy), 8)

    side_label = font.render("SIDE ELEVATION VIEW (Dobot Kinematics)", True, (170, 180, 200))
    screen.blit(side_label, (410, 30))

    # Inset Camera Feed
    if cam_img is not None:
        img_hwc = (np.transpose(cam_img, (2, 1, 0)) * 255).astype(np.uint8)
        cam_surf = pygame.surfarray.make_surface(img_hwc)
        cam_surf_scaled = pygame.transform.scale(cam_surf, (100, 100))
        screen.blit(cam_surf_scaled, (270, 270))
        pygame.draw.rect(screen, (100, 220, 255), (270, 270, 100, 100), 2)
        cam_tag = font.render("Top Camera Focus", True, (100, 220, 255))
        screen.blit(cam_tag, (260, 250))

    # Bottom Status HUD
    pygame.draw.rect(screen, (30, 33, 42), (20, 395, 740, 115), border_radius=8)
    
    status_str = f"Generating Demo #{current_demo_num} (Batch: {total_in_batch} | Total Dataset: {total_saved})"
    screen.blit(font_bold.render(status_str, True, (100, 210, 255)), (35, 405))

    prompt_str = f"Instruction: \"{sim.instruction}\""
    screen.blit(font_bold.render(prompt_str, True, (255, 230, 120)), (35, 432))

    stage_str = f"Phase: {stage_name} | Clutter: {len(sim.distractor_cubes)} distractor objects"
    screen.blit(font.render(stage_str, True, (200, 205, 220)), (35, 458))

    speed_info = font.render(f"Speed: {fps_mode} | [F] Fast/Normal | [Q] Stop", True, (140, 145, 160))
    screen.blit(speed_info, (35, 482))

    pygame.display.flip()

def run_auto_demonstrator(num_demos=60, base_delay=0.001):
    print("=" * 68)
    print("   Dobot Demonstration Generator: Pick-and-Place & Push Actions")
    print("   (With Distractors, Multi-Color Grounding & Parameterized Embeddings)")
    print("=" * 68)
    print(f">> Existing Demos in Dataset: {count_saved_demos()}")
    print(f">> Generating Batch of {num_demos} demonstrations...")
    print("=" * 68)

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Demonstration Generator (Pick-Place & Push)")
    font = pygame.font.SysFont("Arial", 14)
    font_bold = pygame.font.SysFont("Arial", 16, bold=True)

    delay = base_delay
    fps_label = "Turbo Speed"
    demos_completed = 0

    while demos_completed < num_demos:
        demo_idx = get_next_demo_index()
        act_choice = "pick_place" if (demos_completed % 2 == 0) else "push"
        obs_dict = sim.reset(random_scene=True, num_distractors=2, action_type=act_choice)
        
        target_start = sim.target_cube_pos.copy()
        platform_target = sim.target_platform_pos.copy()
        prompt_text = sim.instruction

        img_list = []
        proprio_list = []
        legacy_obs_list = []
        action_list = []
        aborted = False

        hover_cube_z = 0.12
        p_start = sim.ee_pos[:3].copy()
        p_hover_cube = np.array([target_start[0], target_start[1], hover_cube_z], dtype=np.float32)
        
        if act_choice == "pick_place":
            stages = [
                (f"1. Approach {sim.target_color.upper()} Cube", p_start, p_hover_cube, 0.0, 0.0, 20),
                (f"2. Descend on {sim.target_color.upper()} Cube", p_hover_cube, np.array([target_start[0], target_start[1], 0.026], dtype=np.float32), 0.0, 0.0, 16),
                (f"3. Grasp {sim.target_color.upper()} Cube", np.array([target_start[0], target_start[1], 0.026], dtype=np.float32), np.array([target_start[0], target_start[1], 0.026], dtype=np.float32), 1.0, 0.0, 4),
                (f"4. Lift {sim.target_color.upper()} Cube", np.array([target_start[0], target_start[1], 0.026], dtype=np.float32), p_hover_cube, 1.0, 0.0, 16),
                (f"5. Carry to {sim.target_plat_color.upper()} Box", p_hover_cube, np.array([platform_target[0], platform_target[1], hover_cube_z], dtype=np.float32), 1.0, 0.0, 24),
                (f"6. Lower to {sim.target_plat_color.upper()} Box", np.array([platform_target[0], platform_target[1], hover_cube_z], dtype=np.float32), np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), 1.0, 0.0, 16),
                (f"7. Release {sim.target_color.upper()} Cube", np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), 0.0, 1.0, 6),
                ("8. Retract Arm (Task Done)", np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), np.array([platform_target[0], platform_target[1], 0.12], dtype=np.float32), 0.0, 1.0, 12),
            ]
        else: # "push"
            push_dir = (platform_target[:2] - target_start[:2])
            dist = np.linalg.norm(push_dir)
            if dist > 0.01:
                push_dir /= dist
            else:
                push_dir = np.array([1.0, 0.0])
            behind_pos = target_start[:2] - push_dir * 0.040
            push_dest = target_start[:2] + push_dir * 0.085

            stages = [
                (f"1. Move Behind {sim.target_color.upper()} Cube", p_start, np.array([behind_pos[0], behind_pos[1], 0.080], dtype=np.float32), 0.0, 0.0, 20),
                (f"2. Lower Behind {sim.target_color.upper()} Cube", np.array([behind_pos[0], behind_pos[1], 0.080], dtype=np.float32), np.array([behind_pos[0], behind_pos[1], 0.020], dtype=np.float32), 0.0, 0.0, 14),
                (f"3. Push Toward {sim.target_plat_color.upper()}", np.array([behind_pos[0], behind_pos[1], 0.020], dtype=np.float32), np.array([push_dest[0], push_dest[1], 0.020], dtype=np.float32), 0.0, 0.0, 36),
                (f"4. Retract Gripper", np.array([push_dest[0], push_dest[1], 0.020], dtype=np.float32), np.array([push_dest[0], push_dest[1], 0.100], dtype=np.float32), 0.0, 1.0, 14),
                (f"5. Return Home", np.array([push_dest[0], push_dest[1], 0.100], dtype=np.float32), np.array([0.20, 0.0, 0.12], dtype=np.float32), 0.0, 1.0, 16),
            ]

        for stage_name, start_pt, end_pt, grip_state, succ_state, steps in stages:
            pts = generate_smooth_trajectory(start_pt, end_pt, steps)
            for pt in pts:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            aborted = True
                        elif event.key == pygame.K_f:
                            if delay > 0.0005:
                                delay = 0.00005
                                fps_label = "Turbo Speed (Max FPS)"
                            else:
                                delay = 0.005
                                fps_label = "Standard Speed (120 FPS)"

                if aborted:
                    break

                vla_obs = sim.get_vla_observation()
                img_list.append(vla_obs["image"])
                proprio_list.append(vla_obs["proprio"])
                legacy_obs_list.append(sim.get_observation())
                
                current_ee = sim.ee_pos.copy()
                dx = pt[0] - current_ee[0]
                dy = pt[1] - current_ee[1]
                dz = pt[2] - current_ee[2]
                dyaw = 0.0
                
                delta_action = np.array([dx, dy, dz, dyaw, grip_state, succ_state], dtype=np.float32)
                action_list.append(delta_action)
                sim.step_delta(delta_action[:5])

                total_saved_now = count_saved_demos()
                render_gui(screen, font, font_bold, sim, demo_idx, num_demos, total_saved_now, stage_name, len(action_list), fps_label, vla_obs["image"])
                if delay > 0:
                    time.sleep(delay)

            if aborted:
                break

        if aborted:
            print("\n[INFO] Demonstration generation stopped by user.")
            break

        # Fixed interpolation to exactly 128 horizon for trajectory chunking / flow matching
        img_arr = np.array(img_list, dtype=np.float32)
        proprio_arr = np.array(proprio_list, dtype=np.float32)
        legacy_arr = np.array(legacy_obs_list, dtype=np.float32)
        acts_arr = np.array(action_list, dtype=np.float32)

        # Resample / pad to exactly 128 timesteps
        orig_len = len(acts_arr)
        target_len = 128
        indices = np.linspace(0, orig_len - 1, target_len).astype(int)

        filename = os.path.join(DATA_DIR, f"demo_{demo_idx:03d}.npz")
        
        np.savez_compressed(
            filename,
            images=img_arr[indices],
            proprioception=proprio_arr[indices],
            observations=legacy_arr[indices],
            actions=acts_arr[indices],
            prompt=np.array([prompt_text]),
            action_type=np.array([act_choice]),
            target_color=np.array([sim.target_color]),
            target_plat_color=np.array([sim.target_plat_color])
        )
        demos_completed += 1
        total_now = count_saved_demos()
        print(f"[SUCCESS] Saved Demo #{demo_idx:03d} [{act_choice.upper()}] ({sim.target_color}->{sim.target_plat_color} | Total: {total_now}) -> {os.path.basename(filename)}")

    pygame.quit()
    print(f"\n[DONE] Finished batch! Total {count_saved_demos()} demonstration datasets in: {DATA_DIR}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_auto_demonstrator(num_demos=n)
