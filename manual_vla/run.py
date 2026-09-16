import os
import sys
import time
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from train_flow import ManualVLAPolicy, get_intent_embedding_vector, MODEL_DIR

def run_gui():
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
    pygame.display.set_caption("Manual VLA High-Res 256 Patch Embeddings (16x16 Grid)")

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
                elif event.key == pygame.K_2:
                    action_type = "push"
                    sim.action_type = action_type
                    current_trajectory = None
                    task_success_status = None
                elif event.key == pygame.K_r:
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                    current_trajectory = None
                    task_success_status = None
                elif event.key == pygame.K_SPACE:
                    auto_execute = not auto_execute

        if auto_execute:
            if current_trajectory is None:
                if has_model:
                    with torch.no_grad():
                        img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)
                        intent_raw = get_intent_embedding_vector(action_type, sim.target_color, sim.target_plat_color)
                        intent_t = torch.tensor(intent_raw, dtype=torch.float32).unsqueeze(0)
                        proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0)
                        current_trajectory = model.sample(img_t, intent_t, proprio=proprio_t, num_steps=20).squeeze(0).numpy()
                        traj_step = 0
                        task_success_status = None
                else:
                    c_pos = sim.target_cube_pos
                    p_pos = sim.target_platform_pos
                    target_xyz = c_pos if not sim.grasped else p_pos
                    grip = 1.0 if np.linalg.norm(sim.ee_pos[:3] - c_pos) < 0.035 else 0.0
                    delta = np.clip(target_xyz - sim.ee_pos[:3], -0.008, 0.008)
                    obs, _ = sim.step_delta(np.array([delta[0], delta[1], delta[2], 0.0, grip], dtype=np.float32))

            elif traj_step < len(current_trajectory):
                target_point = current_trajectory[traj_step]
                
                # Closed-loop tracking: step end-effector towards target waypoint with max 0.010m per tick
                diff_xyz = target_point[:3] - sim.ee_pos[:3]
                dist_to_pt = np.linalg.norm(diff_xyz)
                
                # Advance to next waypoint once current waypoint is reached within 0.008m or after progress
                if dist_to_pt < 0.008:
                    traj_step += 1
                    if traj_step < len(current_trajectory):
                        target_point = current_trajectory[traj_step]
                        diff_xyz = target_point[:3] - sim.ee_pos[:3]
                else:
                    traj_step += 1 # Steadily progress along the trajectory
                
                # Binary sharpening of continuous gripper signal
                grip_cmd = 1.0 if target_point[3] > 0.45 else 0.0
                delta_action = np.array([diff_xyz[0], diff_xyz[1], diff_xyz[2], 0.0, grip_cmd], dtype=np.float32)
                obs, _ = sim.step_delta(delta_action, max_step=0.010)
            else:
                # Full trajectory completed: evaluate if the right cube is resting on top of the right platform
                if task_success_status is None:
                    final_cube_dist_to_plat = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
                    task_success_status = bool(
                        final_cube_dist_to_plat < 0.040 and
                        sim.target_cube_pos[2] <= 0.025 and
                        not sim.gripper_closed
                    )
                    success_banner_timer = 45 # Display result for 45 frames before next trial
                elif success_banner_timer > 0:
                    success_banner_timer -= 1
                else:
                    # Reset scene for the next autonomous trial
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                    current_trajectory = None
                    task_success_status = None

        # -------------------------------------------------------------
        # 4-Panel Rendering
        # -------------------------------------------------------------
        screen.fill((20, 22, 28))

        # Panel 1: TOP VIEW (15, 15, 355, 245)
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

        # Panel 2: SIDE VIEW (390, 15, 355, 245)
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

        # -------------------------------------------------------------
        # Panels 3 & 4: HIGH-RESOLUTION 256 EMBEDDING PATCHES (16x16 Grid)
        # -------------------------------------------------------------
        img_hwc = (np.transpose(obs["image"], (1, 2, 0)) * 255).astype(np.uint8)

        # Fine-grained 4x4 pixel patches -> 16x16 grid = 256 spatial tokens
        def compute_dense_patch_heatmap(target_rgb):
            grid = np.zeros((16, 16), dtype=np.float32)
            tgt = np.array(target_rgb, dtype=np.float32) / 255.0
            for py in range(16):
                for px in range(16):
                    patch = img_hwc[py*4:(py+1)*4, px*4:(px+1)*4]
                    patch_mean = patch.mean(axis=(0, 1)) / 255.0
                    dist = np.linalg.norm(patch_mean - tgt)
                    score = np.exp(-dist * 5.0)
                    grid[py, px] = score
            if grid.max() > grid.min():
                grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-6)
            return grid

        p1_rgb = COLOR_PALETTE.get(sim.target_color, (240, 45, 45))
        p2_rgb = COLOR_PALETTE.get(sim.target_plat_color, (40, 210, 80))
        heatmap_p1 = compute_dense_patch_heatmap(p1_rgb)
        heatmap_p2 = compute_dense_patch_heatmap(p2_rgb)

        # Panel 3: PARAM 1 (15, 275, 355, 135)
        pygame.draw.rect(screen, (28, 31, 40), (15, 275, 355, 135), border_radius=6)
        screen.blit(font_bold.render(f"PARAM 1 EMBEDDING ({sim.target_color.upper()})", True, (255, 200, 100)), (25, 282))
        screen.blit(font_sm.render("256 Visual Tokens (16x16 Fine Patch Grid):", True, (150, 160, 180)), (25, 298))

        # Render 16x16 Fine-grained Patch Grid for Param 1
        g1_x, g1_y = 25, 317
        b_size = 5 # 5px per patch * 16 = 80px square
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

        # Inset Top Camera
        cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img_hwc, (1, 0, 2))), (80, 80))
        screen.blit(cam_surf, (245, 317))
        pygame.draw.rect(screen, (100, 220, 255), (245, 317, 80, 80), 1)
        screen.blit(font_sm.render("Raw Overhead Camera", True, (140, 150, 170)), (235, 300))

        # Panel 4: PARAM 2 (390, 275, 355, 135)
        pygame.draw.rect(screen, (28, 31, 40), (390, 275, 355, 135), border_radius=6)
        screen.blit(font_bold.render(f"PARAM 2 EMBEDDING ({sim.target_plat_color.upper()})", True, (100, 220, 255)), (400, 282))
        screen.blit(font_sm.render("256 Visual Tokens (16x16 Fine Patch Grid):", True, (150, 160, 180)), (400, 298))

        # Render 16x16 Fine-grained Patch Grid for Param 2
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

        # Inset Target Dest Platform
        pygame.draw.rect(screen, p2_rgb, (620, 327, 60, 60), border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255), (620, 327, 60, 60), 2, border_radius=4)
        screen.blit(font_sm.render("Dest Platform Target", True, (140, 150, 170)), (610, 305))

        # -------------------------------------------------------------
        # HUD 1: ACTION TYPE & PARAMETERS (15, 425, 730, 45)
        # -------------------------------------------------------------
        pygame.draw.rect(screen, (26, 29, 38), (15, 425, 730, 45), border_radius=6)
        act_display = "PICK & PLACE" if action_type == "pick_place" else "PUSH TOWARDS"
        screen.blit(font_bold.render(f"ACTION TYPE: {act_display}", True, (255, 255, 255)), (25, 439))
        screen.blit(font_bold.render(f"PARAMETER 1: {sim.target_color.upper()}", True, p1_rgb), (260, 439))
        screen.blit(font_bold.render(f"PARAMETER 2: {sim.target_plat_color.upper()}", True, p2_rgb), (500, 439))

        # -------------------------------------------------------------
        # HUD 2: CONTROLS & INSTRUCTIONS (15, 480, 730, 75)
        # -------------------------------------------------------------
        pygame.draw.rect(screen, (22, 24, 32), (15, 480, 730, 75), border_radius=6)
        screen.blit(font_bold.render("CONTROLS:", True, (120, 210, 255)), (25, 490))
        controls_text = "[1] Action: Pick & Place   |   [2] Action: Push   |   [R] Randomize   |   [SPACE] Pause/Play"
        screen.blit(font.render(controls_text, True, (200, 205, 220)), (105, 491))

        if task_success_status is not None and success_banner_timer > 0:
            if task_success_status:
                eval_text = "EVALUATION: SUCCESS - TARGET CUBE ON PLATFORM!"
                eval_col = (50, 240, 100)
            else:
                eval_text = "EVALUATION: INCOMPLETE / MISPLACED"
                eval_col = (255, 90, 90)
            screen.blit(font_bold.render(eval_text, True, eval_col), (105, 520))
        else:
            status_text = f"Status: {'AUTONOMOUS' if auto_execute else 'PAUSED'}  |  Dobot 4-DOF IK: Active  |  Tokens: 256"
            screen.blit(font_sm.render(status_text, True, (130, 140, 160)), (105, 520))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    run_gui()

