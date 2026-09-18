import os
import sys
import time
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from train_flow import ManualVLAPolicy, get_color_ids, MODEL_DIR

def run_gui(fast_mode=False):
    device = torch.device("cpu")
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")

    model = ManualVLAPolicy()
    has_model = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            has_model = True
        except Exception:
            pass

    sim = DobotPickPlaceSim()
    pygame.init()

    # Compact 4-Panel Window: 760 x 570
    screen = pygame.display.set_mode((760, 570))
    pygame.display.set_caption("Manual VLA Continuous Closed-Loop Controller")

    font_sm = pygame.font.SysFont("Arial", 11)
    font = pygame.font.SysFont("Arial", 12)
    font_bold = pygame.font.SysFont("Arial", 13, bold=True)
    font_title = pygame.font.SysFont("Arial", 14, bold=True)

    action_type = "pick_place"
    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)

    def world_to_screen(x, y):
        sx = int(190 + (y / 0.28) * 130)
        sy = int(240 - ((x - 0.08) / 0.26) * 195)
        return sx, sy

    def world_to_side_screen(x, z):
        sx = int(425 + ((x - 0.05) / 0.30) * 235)
        sy = int(235 - (z / 0.22) * 170)
        return sx, sy

    clock = pygame.time.Clock()
    running = True

    current_trajectory = None
    traj_step = 0
    auto_execute = True

    success_banner_timer = 0
    task_success_status = None
    lightspeed = fast_mode
    last_vlm_attn = None

    # Continuous control & session statistics tracking
    episode_total_ticks = 0
    MAX_EPISODE_TICKS = 220 # Safety watchdog to prevent endless wandering
    total_episodes = 0
    successful_episodes = 0
    failed_episodes = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    action_type = "pick_place"
                    sim.action_type = action_type
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_2:
                    action_type = "push"
                    sim.action_type = action_type
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_r:
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_f:
                    lightspeed = not lightspeed
                elif event.key == pygame.K_SPACE:
                    auto_execute = not auto_execute

        steps_per_frame = 3 if lightspeed else 1

        for _ in range(steps_per_frame):
            if auto_execute:
                # -------------------------------------------------------------
                # Continuous Closed-Loop Success Check (LLM-style continuing)
                # -------------------------------------------------------------
                if task_success_status is None:
                    final_cube_dist_to_plat = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
                    # If physical goal is accomplished at ANY point, mark success and transition!
                    if final_cube_dist_to_plat < 0.040 and sim.target_cube_pos[2] <= 0.025 and not sim.gripper_closed:
                        task_success_status = True
                        total_episodes += 1
                        successful_episodes += 1
                        success_banner_timer = 20 if lightspeed else 45
                        rate = (successful_episodes / total_episodes) * 100.0
                        print(f"[TRIAL #{total_episodes:03d}] SUCCESS! (Total: {successful_episodes} Success, {failed_episodes} Failed | Win Rate: {rate:.1f}%)", flush=True)
                    elif episode_total_ticks >= MAX_EPISODE_TICKS:
                        task_success_status = False
                        total_episodes += 1
                        failed_episodes += 1
                        success_banner_timer = 20 if lightspeed else 45
                        rate = (successful_episodes / total_episodes) * 100.0
                        print(f"[TRIAL #{total_episodes:03d}] FAILED / TIMEOUT. (Total: {successful_episodes} Success, {failed_episodes} Failed | Win Rate: {rate:.1f}%)", flush=True)

                if task_success_status is not None:
                    if success_banner_timer > 0:
                        success_banner_timer -= 1
                    else:
                        obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                        current_trajectory = None
                        task_success_status = None
                        episode_total_ticks = 0
                else:
                    # -------------------------------------------------------------
                    # Continuous Replanning & Velocity Following
                    # -------------------------------------------------------------
                    episode_total_ticks += 1
                    
                    # Replanning: When current chunk is exhausted or every 32 steps (receding horizon)
                    need_replan = (current_trajectory is None) or (traj_step >= len(current_trajectory))
                    
                    if need_replan:
                        if has_model:
                            with torch.no_grad():
                                img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)
                                color_ids_raw = get_color_ids(sim.target_color, sim.target_plat_color)
                                color_ids_t = torch.tensor(color_ids_raw, dtype=torch.long).unsqueeze(0)
                                proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
                                sample_steps = 10 if lightspeed else 20
                                traj_out, attn_out = model.sample(img_t, color_ids_t, proprio=proprio_t, num_steps=sample_steps, return_attn=True)
                                current_trajectory = traj_out.squeeze(0).numpy()
                                traj_step = 0
                                if attn_out is not None:
                                    last_vlm_attn = attn_out[0].cpu().numpy() # [2, 256]
                        else:
                            c_pos = sim.target_cube_pos
                            p_pos = sim.target_platform_pos
                            target_xyz = c_pos if not sim.grasped else p_pos
                            grip = 1.0 if np.linalg.norm(sim.ee_pos[:3] - c_pos) < 0.035 else 0.0
                            delta = np.clip(target_xyz - sim.ee_pos[:3], -0.008, 0.008)
                            obs, _ = sim.step_delta(np.array([delta[0], delta[1], delta[2], 0.0, grip], dtype=np.float32))

                    if current_trajectory is not None and traj_step < len(current_trajectory):
                        target_point = current_trajectory[traj_step]
                        diff_xyz = target_point[:3] - sim.ee_pos[:3]
                        dist_to_pt = np.linalg.norm(diff_xyz)
                        
                        advance_threshold = 0.012 if lightspeed else 0.008
                        if dist_to_pt < advance_threshold:
                            traj_step += 2 if lightspeed else 1
                            if traj_step < len(current_trajectory):
                                target_point = current_trajectory[traj_step]
                                diff_xyz = target_point[:3] - sim.ee_pos[:3]
                        else:
                            traj_step += 2 if lightspeed else 1
                        
                        # Pure model-predicted gripper: 4th action dim is the grip signal
                        grip_cmd = 1.0 if target_point[3] > 0.5 else 0.0
                            
                        max_step_rate = 0.015 if lightspeed else 0.010
                        delta_action = np.array([diff_xyz[0], diff_xyz[1], diff_xyz[2], 0.0, grip_cmd], dtype=np.float32)
                        obs, _ = sim.step_delta(delta_action, max_step=max_step_rate)

        # -------------------------------------------------------------
        # 4-Panel Rendering
        # -------------------------------------------------------------
        screen.fill((20, 22, 28))

        # Panel 1: TOP VIEW
        pygame.draw.rect(screen, (30, 33, 42), (15, 15, 355, 245), border_radius=6)
        screen.blit(font_title.render("TOP VIEW", True, (210, 220, 240)), (25, 22))

        bx, by = world_to_screen(0.08, 0.0)
        pygame.draw.circle(screen, (70, 75, 95), (bx, by), 10)

        for p_col, p_pos in sim.distractor_platforms:
            px, py = world_to_screen(p_pos[0], p_pos[1])
            pygame.draw.rect(screen, COLOR_PALETTE.get(p_col, (40, 210, 80)), (px - 11, py - 11, 22, 22), border_radius=3)
        tpx, tpy = world_to_screen(sim.target_platform_pos[0], sim.target_platform_pos[1])
        t_col = COLOR_PALETTE.get(sim.target_plat_color, (40, 210, 80))
        pygame.draw.rect(screen, t_col, (tpx - 13, tpy - 13, 26, 26), border_radius=3)
        pygame.draw.rect(screen, (255, 255, 255), (tpx - 13, tpy - 13, 26, 26), 2, border_radius=3)

        for c_col, c_pos in sim.distractor_cubes:
            cx, cy = world_to_screen(c_pos[0], c_pos[1])
            pygame.draw.rect(screen, COLOR_PALETTE.get(c_col, (45, 120, 240)), (cx - 6, cy - 6, 12, 12), border_radius=2)
        tcx, tcy = world_to_screen(sim.target_cube_pos[0], sim.target_cube_pos[1])
        tc_col = COLOR_PALETTE.get(sim.target_color, (240, 45, 45))
        pygame.draw.rect(screen, tc_col, (tcx - 7, tcy - 7, 14, 14), border_radius=2)
        pygame.draw.rect(screen, (255, 255, 255), (tcx - 7, tcy - 7, 14, 14), 2, border_radius=2)

        ex, ey = world_to_screen(sim.ee_pos[0], sim.ee_pos[1])
        grip_color = (255, 70, 70) if sim.gripper_closed else (80, 200, 255)
        pygame.draw.line(screen, (160, 175, 200), (bx, by), (ex, ey), 3)
        pygame.draw.circle(screen, grip_color, (ex, ey), 7)

        # Panel 2: SIDE VIEW
        pygame.draw.rect(screen, (30, 33, 42), (390, 15, 355, 245), border_radius=6)
        screen.blit(font_title.render("SIDE VIEW (Dobot Kinematics)", True, (210, 220, 240)), (400, 22))
        pygame.draw.line(screen, (60, 65, 80), (400, 235), (735, 235), 2)

        psx, psy = world_to_side_screen(sim.target_platform_pos[0], sim.target_platform_pos[2])
        pygame.draw.rect(screen, t_col, (psx - 13, psy - 3, 26, 6), border_radius=2)
        pygame.draw.rect(screen, (255, 255, 255), (psx - 13, psy - 3, 26, 6), 1, border_radius=2)

        csx, csy = world_to_side_screen(sim.target_cube_pos[0], sim.target_cube_pos[2])
        pygame.draw.rect(screen, tc_col, (csx - 6, csy - 6, 12, 12), border_radius=2)
        pygame.draw.rect(screen, (255, 255, 255), (csx - 6, csy - 6, 12, 12), 1, border_radius=2)

        j1, j2, j3, j4 = sim.kin.inverse(sim.ee_pos[0], sim.ee_pos[1], sim.ee_pos[2], sim.ee_pos[3])
        r0, z0 = 0.08, 0.0
        r1, z1 = 0.08, sim.kin.L1
        r2 = r1 + sim.kin.L2 * np.cos(j2)
        z2 = z1 + sim.kin.L2 * np.sin(j2)
        r3 = r2 + sim.kin.L3 * np.cos(j2 + j3)
        z3 = z2 + sim.kin.L3 * np.sin(j2 + j3)
        r4 = np.sqrt(sim.ee_pos[0]**2 + sim.ee_pos[1]**2)
        z4 = sim.ee_pos[2]

        pts_side = [world_to_side_screen(r, z) for r, z in [(r0, z0), (r1, z1), (r2, z2), (r3, z3), (r4, z4)]]
        for i in range(len(pts_side) - 1):
            pygame.draw.line(screen, (180, 195, 220), pts_side[i], pts_side[i+1], 3)
        for pt in pts_side[:-1]:
            pygame.draw.circle(screen, (255, 170, 50), pt, 4)
        pygame.draw.circle(screen, grip_color, pts_side[-1], 7)

        # Panels 3 & 4: HIGH-RESOLUTION 256 EMBEDDING PATCHES (VLM Cross-Attention)
        img_hwc = (np.transpose(obs["image"], (1, 2, 0)) * 255).astype(np.uint8)

        p1_rgb = COLOR_PALETTE.get(sim.target_color, (240, 45, 45))
        p2_rgb = COLOR_PALETTE.get(sim.target_plat_color, (40, 210, 80))
        if last_vlm_attn is None and has_model:
            with torch.no_grad():
                img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)
                color_ids_raw = get_color_ids(sim.target_color, sim.target_plat_color)
                color_ids_t = torch.tensor(color_ids_raw, dtype=torch.long).unsqueeze(0)
                proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
                _, last_attn_t = model.forward_flow(torch.zeros(1, 128, 4), torch.zeros(1), img_t, color_ids_t, proprio=proprio_t)
                last_vlm_attn = last_attn_t[0].cpu().numpy()

        if last_vlm_attn is not None:
            # Live cross-attention weights directly from model
            a1 = last_vlm_attn[0].reshape(16, 16)
            a2 = last_vlm_attn[1].reshape(16, 16)
            heatmap_p1 = (a1 - a1.min()) / (a1.max() - a1.min() + 1e-6)
            heatmap_p2 = (a2 - a2.min()) / (a2.max() - a2.min() + 1e-6)
        else:
            heatmap_p1 = np.zeros((16, 16), dtype=np.float32)
            heatmap_p2 = np.zeros((16, 16), dtype=np.float32)

        # Panel 3: PARAM 1 (Target Cube)
        pygame.draw.rect(screen, (28, 31, 40), (15, 275, 355, 135), border_radius=6)
        screen.blit(font_bold.render(f"VLM ATTN: TARGET CUBE <{sim.target_color.upper()}>", True, (255, 200, 100)), (25, 282))
        screen.blit(font_sm.render("Token <cube> -> 256 Patch Cross-Attention:", True, (150, 160, 180)), (25, 298))

        g1_x, g1_y = 25, 317
        b_size = 5
        for r in range(16):
            for c in range(16):
                val = heatmap_p1[r, c]
                patch_col = (
                    int(p1_rgb[0] * val + 28 * (1 - val)),
                    int(p1_rgb[1] * val + 28 * (1 - val)),
                    int(p1_rgb[2] * val + 28 * (1 - val))
                )
                pygame.draw.rect(screen, patch_col, (g1_x + c * b_size, g1_y + r * b_size, b_size, b_size))
        pygame.draw.rect(screen, (90, 100, 120), (g1_x, g1_y, 16 * b_size, 16 * b_size), 1)

        cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img_hwc, (1, 0, 2))), (80, 80))
        screen.blit(cam_surf, (245, 317))
        pygame.draw.rect(screen, (100, 220, 255), (245, 317, 80, 80), 1)
        screen.blit(font_sm.render("Raw Overhead Camera", True, (140, 150, 170)), (235, 300))

        # Panel 4: PARAM 2 (Target Platform)
        pygame.draw.rect(screen, (28, 31, 40), (390, 275, 355, 135), border_radius=6)
        screen.blit(font_bold.render(f"VLM ATTN: DEST PLATFORM <{sim.target_plat_color.upper()}>", True, (100, 220, 255)), (400, 282))
        screen.blit(font_sm.render("Token <plat> -> 256 Patch Cross-Attention:", True, (150, 160, 180)), (400, 298))

        g2_x, g2_y = 400, 317
        for r in range(16):
            for c in range(16):
                val = heatmap_p2[r, c]
                patch_col = (
                    int(p2_rgb[0] * val + 28 * (1 - val)),
                    int(p2_rgb[1] * val + 28 * (1 - val)),
                    int(p2_rgb[2] * val + 28 * (1 - val))
                )
                pygame.draw.rect(screen, patch_col, (g2_x + c * b_size, g2_y + r * b_size, b_size, b_size))
        pygame.draw.rect(screen, (90, 100, 120), (g2_x, g2_y, 16 * b_size, 16 * b_size), 1)

        pygame.draw.rect(screen, p2_rgb, (620, 327, 60, 60), border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255), (620, 327, 60, 60), 2, border_radius=4)
        screen.blit(font_sm.render("Dest Platform Target", True, (140, 150, 170)), (610, 305))

        # HUD 1: ACTION TYPE & PARAMETERS
        pygame.draw.rect(screen, (26, 29, 38), (15, 425, 730, 45), border_radius=6)
        act_display = "PICK & PLACE" if action_type == "pick_place" else "PUSH TOWARDS"
        screen.blit(font_bold.render(f"ACTION TYPE: {act_display}", True, (255, 255, 255)), (25, 439))
        screen.blit(font_bold.render(f"PARAMETER 1: {sim.target_color.upper()}", True, p1_rgb), (260, 439))
        screen.blit(font_bold.render(f"PARAMETER 2: {sim.target_plat_color.upper()}", True, p2_rgb), (500, 439))

        # HUD 2: CONTROLS & INSTRUCTIONS
        pygame.draw.rect(screen, (22, 24, 32), (15, 480, 730, 75), border_radius=6)
        screen.blit(font_bold.render("CONTROLS:", True, (120, 210, 255)), (25, 490))
        speed_label = "[F] Lightspeed: ON" if lightspeed else "[F] Lightspeed: OFF"
        controls_text = f"[1] Pick & Place | [2] Push | [R] Randomize | [SPACE] Pause | {speed_label}"
        screen.blit(font.render(controls_text, True, (200, 205, 220)), (105, 491))

        if task_success_status is not None and success_banner_timer > 0:
            if task_success_status:
                eval_text = f"EVALUATION: SUCCESS - TARGET CUBE ON PLATFORM!  [Score: {successful_episodes}/{total_episodes}]"
                eval_col = (50, 240, 100)
            else:
                eval_text = f"EVALUATION: TIMEOUT / MISPLACED  [Score: {successful_episodes}/{total_episodes}]"
                eval_col = (255, 90, 90)
            screen.blit(font_bold.render(eval_text, True, eval_col), (105, 520))
        else:
            mode_desc = "LIGHTSPEED TURBO" if lightspeed else "NORMAL 60FPS"
            win_pct = (successful_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0
            status_text = f"Mode: {mode_desc} | Trial #{total_episodes + 1} | Succ: {successful_episodes} Fail: {failed_episodes} ({win_pct:.1f}%)"
            screen.blit(font_sm.render(status_text, True, (130, 140, 160)), (105, 520))

        pygame.display.flip()
        if not lightspeed:
            clock.tick(60)

    pygame.quit()

    # -------------------------------------------------------------
    # Session Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 55, flush=True)
    print("           SESSION EVALUATION SUMMARY", flush=True)
    print("=" * 55, flush=True)
    print(f"  Total Trials Run     : {total_episodes}", flush=True)
    print(f"  Successful Trials    : {successful_episodes}", flush=True)
    print(f"  Failed / Incomplete  : {failed_episodes}", flush=True)
    if total_episodes > 0:
        win_rate = (successful_episodes / total_episodes) * 100.0
        print(f"  Overall Success Rate : {win_rate:.1f}%", flush=True)
    print("=" * 55 + "\n", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Launch directly in lightspeed turbo mode")
    args = parser.parse_args()
    run_gui(fast_mode=args.fast)
