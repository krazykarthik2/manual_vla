import os
import sys
import time
import torch
import numpy as np
import pygame
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
sys.path.append(os.path.dirname(__file__))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from full_smolvla_model import FullSmolVLAPolicy
from train_full_smolvla import MODEL_DIR, DEVICE

def visualize_full_smolvla():
    device = DEVICE
    model_path = os.path.join(MODEL_DIR, "dobot_full_smolvla_policy.pth")

    print("[Visualizer] Initializing Full SmolVLA with foundation SmolVLM backbone...", flush=True)
    model = FullSmolVLAPolicy(d_action_model=128, load_backbone=True, device=device).to(device)

    has_trained_weights = False
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt, strict=False)
            model.eval()
            has_trained_weights = True
            print(f"[Visualizer] Loaded trained action head checkpoint from {model_path}", flush=True)
        except Exception as e:
            print(f"[Visualizer] Could not load checkpoint weights: {e}", flush=True)
    else:
        print(f"[Visualizer] Checkpoint {model_path} not found. Running with initialized action head & pretrained SmolVLM.", flush=True)

    sim = DobotPickPlaceSim()
    pygame.init()

    # High-density dashboard: 1400 x 780
    screen = pygame.display.set_mode((1400, 780))
    pygame.display.set_caption("Full SmolVLA Live Visualizer (SmolVLM Foundation Backbone + Cross-Attn & Layer Activations)")

    font_xs = pygame.font.SysFont("Consolas", 10)
    font_sm = pygame.font.SysFont("Consolas", 11)
    font = pygame.font.SysFont("Consolas", 12)
    font_bold = pygame.font.SysFont("Consolas", 13, bold=True)
    font_title = pygame.font.SysFont("Arial", 15, bold=True)

    clock = pygame.time.Clock()
    running = True

    # Speed modes: 0=Normal(60fps), 1=Fast(3x), 2=Slow(0.5x)
    SPEED_NAMES = ["NORMAL (60fps)", "FAST (3x)", "SLOW (0.5x)"]
    speed_mode = 0

    action_type = "pick_place"
    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)

    current_trajectory = None
    traj_step = 0
    episode_total_ticks = 0
    MAX_EPISODE_TICKS = 220
    total_episodes = 0
    successful_episodes = 0
    failed_episodes = 0
    task_success_status = None
    success_banner_timer = 0

    # Layer activation inspection selector: 0=L1_SelfAttn, 1=L1_CrossAttn, 2=L1_FFN, 3=L2_SelfAttn, 4=L2_CrossAttn, 5=L2_FFN
    LAYER_NAMES = [
        "Decoder L1 Self-Attn",
        "Decoder L1 Cross-Attn",
        "Decoder L1 Post-FFN",
        "Decoder L2 Self-Attn",
        "Decoder L2 Cross-Attn",
        "Decoder L2 Post-FFN"
    ]
    selected_layer_idx = 1
    selected_query_step = 0 # Step in horizon 0..127

    last_activations = None
    last_vlm_tokens = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_f:
                    speed_mode = (speed_mode + 1) % 3
                elif event.key == pygame.K_1:
                    action_type = "pick_place"
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type="pick_place")
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_2:
                    action_type = "push"
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type="push")
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_TAB:
                    selected_layer_idx = (selected_layer_idx + 1) % len(LAYER_NAMES)
                elif event.key == pygame.K_UP:
                    selected_query_step = max(0, selected_query_step - 8)
                elif event.key == pygame.K_DOWN:
                    selected_query_step = min(127, selected_query_step + 8)

        # Steps per frame
        if speed_mode == 0:
            steps_per_frame = 1
            fps_target = 60
        elif speed_mode == 1:
            steps_per_frame = 3
            fps_target = 60
        else:
            steps_per_frame = 1
            fps_target = 30

        for _ in range(steps_per_frame):
            if task_success_status is None:
                final_dist = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
                if final_dist < 0.040 and sim.target_cube_pos[2] <= 0.025 and not sim.gripper_closed:
                    task_success_status = True
                    total_episodes += 1
                    successful_episodes += 1
                    success_banner_timer = 20 if speed_mode == 1 else 45
                    rate = (successful_episodes / total_episodes) * 100.0
                    print(f"[TRIAL #{total_episodes:03d}] SUCCESS! ({successful_episodes}/{total_episodes} = {rate:.1f}%)", flush=True)
                elif episode_total_ticks >= MAX_EPISODE_TICKS:
                    task_success_status = False
                    total_episodes += 1
                    failed_episodes += 1
                    success_banner_timer = 20 if speed_mode == 1 else 45
                    rate = (successful_episodes / total_episodes) * 100.0
                    print(f"[TRIAL #{total_episodes:03d}] TIMEOUT ({successful_episodes}/{total_episodes} = {rate:.1f}%)", flush=True)

            if task_success_status is not None:
                if success_banner_timer > 0:
                    success_banner_timer -= 1
                else:
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
            else:
                episode_total_ticks += 1
                need_replan = (current_trajectory is None) or (traj_step >= len(current_trajectory))

                if need_replan:
                    with torch.no_grad():
                        img_chw = obs["image"]
                        img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_hwc)
                        proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0).to(device)

                        sample_steps = 10 if speed_mode == 1 else 15
                        traj_pred, acts, vlm_tokens = model.sample(
                            [pil_img],
                            [sim.instruction],
                            proprio=proprio_t,
                            num_steps=sample_steps,
                            return_activations=True
                        )
                        current_trajectory = traj_pred.squeeze(0).cpu().numpy()
                        last_activations = acts
                        last_vlm_tokens = vlm_tokens.squeeze(0).cpu().numpy()
                        traj_step = 0

                if current_trajectory is not None and traj_step < len(current_trajectory):
                    target_point = current_trajectory[traj_step]
                    diff_xyz = target_point[:3] - sim.ee_pos[:3]
                    dist_to_pt = np.linalg.norm(diff_xyz)

                    advance_threshold = 0.012 if speed_mode == 1 else 0.008
                    if dist_to_pt < advance_threshold:
                        traj_step += 2 if speed_mode == 1 else 1
                        if traj_step < len(current_trajectory):
                            target_point = current_trajectory[traj_step]
                            diff_xyz = target_point[:3] - sim.ee_pos[:3]
                    else:
                        traj_step += 2 if speed_mode == 1 else 1

                    grip_cmd = 1.0 if target_point[3] > 0.5 else 0.0
                    max_step_rate = 0.015 if speed_mode == 1 else 0.010
                    delta_action = np.array([diff_xyz[0], diff_xyz[1], diff_xyz[2], 0.0, grip_cmd], dtype=np.float32)
                    obs, _ = sim.step_delta(delta_action, max_step=max_step_rate)

        # =================================================================
        # RENDERING MULTIMODAL VLA DASHBOARD
        # =================================================================
        screen.fill((16, 18, 24))

        # Title Header
        screen.blit(font_title.render("FULL SMOLVLA FOUNDATION OBSERVATION & LAYER ACTIVATION DASHBOARD", True, (240, 245, 255)), (25, 10))
        speed_str = SPEED_NAMES[speed_mode]
        status_txt = "WEIGHTS: TRAINED CHECKPOINT" if has_trained_weights else "WEIGHTS: PRETRAINED BACKBONE"
        sub_title = f'Prompt: "{sim.instruction}" | Speed: {speed_str} | {status_txt}'
        screen.blit(font_sm.render(sub_title, True, (130, 140, 160)), (25, 31))

        # =============================================================
        # PANEL 1: SmolVLM Multimodal Token Embeddings (Left: 340px)
        # =============================================================
        p1_x, p1_y, p1_w, p1_h = 20, 52, 340, 580
        pygame.draw.rect(screen, (24, 27, 36), (p1_x, p1_y, p1_w, p1_h), border_radius=6)
        screen.blit(font_bold.render("1. SMOLVLM MULTIMODAL EMBEDDINGS", True, (255, 200, 90)), (p1_x + 10, p1_y + 10))
        screen.blit(font_xs.render("Every sequence token embedding [seq_len x 128-dim]", True, (140, 150, 170)), (p1_x + 10, p1_y + 28))

        if last_vlm_tokens is not None:
            seq_len, dim = last_vlm_tokens.shape
            screen.blit(font_xs.render(f"Sequence Tokens: {seq_len} | Projected Dim: {dim}", True, (100, 220, 255)), (p1_x + 10, p1_y + 44))

            # Heatmap Matrix: rows = tokens, cols = 128-dim feature channels
            m_y = p1_y + 64
            avail_h = p1_h - 75
            row_h = max(1, avail_h // min(seq_len, 64))
            col_w = max(1, (p1_w - 24) // 64) # sub-sample or compress 128 cols to display width

            tok_min = last_vlm_tokens.min()
            tok_max = last_vlm_tokens.max()
            tok_norm = (last_vlm_tokens - tok_min) / (tok_max - tok_min + 1e-6)

            for i in range(min(seq_len, 64)):
                for j in range(min(dim, 64)):
                    v = tok_norm[i, j * (dim // 64)]
                    c_col = (int(15 + v * 240), int(25 + v * 160), int(50 + (1 - v) * 50))
                    pygame.draw.rect(screen, c_col, (p1_x + 12 + j * col_w, m_y + i * row_h, col_w, row_h))

            pygame.draw.rect(screen, (70, 85, 110), (p1_x + 11, m_y - 1, 64 * col_w + 2, min(seq_len, 64) * row_h + 2), 1)
            screen.blit(font_xs.render(f"Token Values Range: [{tok_min:.3f} .. {tok_max:.3f}]", True, (160, 170, 190)), (p1_x + 12, p1_y + p1_h - 20))
        else:
            screen.blit(font_sm.render("Computing initial multimodal tokens...", True, (200, 180, 120)), (p1_x + 20, p1_y + 100))

        # =============================================================
        # PANEL 2: Cross-Attention Heatmaps (Layer 1 & Layer 2) (Mid-Left: 340px)
        # =============================================================
        p2_x, p2_y, p2_w, p2_h = 375, 52, 340, 580
        pygame.draw.rect(screen, (24, 27, 36), (p2_x, p2_y, p2_w, p2_h), border_radius=6)
        screen.blit(font_bold.render("2. CROSS-ATTENTION MAPS", True, (100, 220, 255)), (p2_x + 10, p2_y + 10))
        screen.blit(font_xs.render("Trajectory Queries attending to SmolVLM Tokens", True, (140, 150, 170)), (p2_x + 10, p2_y + 28))

        if last_activations is not None:
            ca_w1 = last_activations["ca_w1"].squeeze(0).cpu().numpy() # [horizon=128, context_len]
            ca_w2 = last_activations["ca_w2"].squeeze(0).cpu().numpy() # [horizon=128, context_len]

            screen.blit(font_bold.render("Layer 1 Cross-Attention:", True, (255, 180, 100)), (p2_x + 10, p2_y + 48))
            # Heatmap 1: [32 horizon steps x 64 context tokens]
            h1_y = p2_y + 68
            w1_norm = (ca_w1 - ca_w1.min()) / (ca_w1.max() - ca_w1.min() + 1e-6)
            for r in range(32):
                for c in range(min(ca_w1.shape[1], 64)):
                    val = w1_norm[r * 4, c]
                    col = (int(20 + val * 235), int(40 + val * 180), int(40 + (1 - val) * 40))
                    pygame.draw.rect(screen, col, (p2_x + 12 + c * 5, h1_y + r * 6, 5, 6))
            pygame.draw.rect(screen, (70, 85, 110), (p2_x + 11, h1_y - 1, min(ca_w1.shape[1], 64) * 5 + 2, 32 * 6 + 2), 1)

            screen.blit(font_bold.render("Layer 2 Cross-Attention:", True, (120, 240, 160)), (p2_x + 10, p2_y + 280))
            # Heatmap 2
            h2_y = p2_y + 300
            w2_norm = (ca_w2 - ca_w2.min()) / (ca_w2.max() - ca_w2.min() + 1e-6)
            for r in range(32):
                for c in range(min(ca_w2.shape[1], 64)):
                    val = w2_norm[r * 4, c]
                    col = (int(40 + val * 120), int(30 + val * 225), int(60 + (1 - val) * 50))
                    pygame.draw.rect(screen, col, (p2_x + 12 + c * 5, h2_y + r * 6, 5, 6))
            pygame.draw.rect(screen, (70, 85, 110), (p2_x + 11, h2_y - 1, min(ca_w2.shape[1], 64) * 5 + 2, 32 * 6 + 2), 1)

            screen.blit(font_xs.render(f"Inspecting Query Step: {traj_step}/128 (Current Execution)", True, (200, 210, 230)), (p2_x + 12, p2_y + 510))
            screen.blit(font_xs.render("Context tokens: [Proprio (1), Time (1), VLM Tokens]", True, (150, 160, 180)), (p2_x + 12, p2_y + 530))
        else:
            screen.blit(font_sm.render("Awaiting first policy sample...", True, (200, 180, 120)), (p2_x + 20, p2_y + 100))

        # =============================================================
        # PANEL 3: Layer-Wise Output Activations (Mid-Right: 340px)
        # =============================================================
        p3_x, p3_y, p3_w, p3_h = 730, 52, 340, 580
        pygame.draw.rect(screen, (24, 27, 36), (p3_x, p3_y, p3_w, p3_h), border_radius=6)
        screen.blit(font_bold.render("3. LAYER-WISE OUTPUT ACTIVATIONS", True, (255, 140, 180)), (p3_x + 10, p3_y + 10))
        screen.blit(font_xs.render("[TAB] Cycle Layer | Active Feature Spectrogram", True, (140, 150, 170)), (p3_x + 10, p3_y + 28))

        cur_layer_name = LAYER_NAMES[selected_layer_idx]
        screen.blit(font_bold.render(f"Active: {cur_layer_name}", True, (255, 220, 120)), (p3_x + 10, p3_y + 48))

        layer_key_map = [
            "l1_sa",
            "l1_ca",
            "l1_out",
            "l2_sa",
            "l2_ca",
            "l2_out"
        ]
        active_key = layer_key_map[selected_layer_idx]

        if last_activations is not None and active_key in last_activations:
            act_tensor = last_activations[active_key].squeeze(0).cpu().numpy() # [horizon=128, d_model=128]
            a_min = act_tensor.min()
            a_max = act_tensor.max()
            a_norm = (act_tensor - a_min) / (a_max - a_min + 1e-6)

            # Draw Activation Spectrogram Matrix
            spec_y = p3_y + 70
            for r in range(40):
                for c in range(40):
                    q_idx = r * 3
                    ch_idx = c * 3
                    v = a_norm[q_idx, ch_idx]
                    col = (int(30 + v * 225), int(15 + v * 140), int(60 + (1 - v) * 120))
                    pygame.draw.rect(screen, col, (p3_x + 12 + c * 7, spec_y + r * 8, 7, 8))
            pygame.draw.rect(screen, (90, 100, 130), (p3_x + 11, spec_y - 1, 40 * 7 + 2, 40 * 8 + 2), 1)

            # Dimension activity statistics
            mean_act = act_tensor.mean()
            std_act = act_tensor.std()
            screen.blit(font_xs.render(f"Mean Activation: {mean_act:+.4f} | Std: {std_act:.4f}", True, (200, 210, 230)), (p3_x + 12, p3_y + 410))
            screen.blit(font_xs.render(f"Min: {a_min:+.4f} | Max: {a_max:+.4f}", True, (160, 170, 190)), (p3_x + 12, p3_y + 430))

            # Sparkline of selected query step's 128-dim embedding
            screen.blit(font_bold.render(f"Current Step #{traj_step} Feature Vector:", True, (140, 220, 255)), (p3_x + 12, p3_y + 455))
            spark_y = p3_y + 480
            pygame.draw.rect(screen, (18, 20, 26), (p3_x + 12, spark_y, 316, 50), border_radius=4)
            step_vec = a_norm[min(traj_step, 127)]
            for ch in range(min(127, len(step_vec) - 1)):
                x1 = p3_x + 14 + int(ch * (312 / 128))
                y1 = spark_y + 45 - int(step_vec[ch] * 40)
                x2 = p3_x + 14 + int((ch + 1) * (312 / 128))
                y2 = spark_y + 45 - int(step_vec[ch + 1] * 40)
                pygame.draw.line(screen, (255, 180, 50), (x1, y1), (x2, y2), 2)
        else:
            screen.blit(font_sm.render("Awaiting activation capture...", True, (200, 180, 120)), (p3_x + 20, p3_y + 100))

        # =============================================================
        # PANEL 4: Simulation Feedback & Robot Execution (Right: 300px)
        # =============================================================
        p4_x, p4_y, p4_w, p4_h = 1085, 52, 295, 580
        pygame.draw.rect(screen, (24, 27, 36), (p4_x, p4_y, p4_w, p4_h), border_radius=6)
        screen.blit(font_bold.render("4. ROBOT EXECUTION & SENSORS", True, (120, 240, 150)), (p4_x + 10, p4_y + 10))

        # 64x64 Overhead RGB Feed
        img_chw = obs["image"]
        img_hwc = (np.transpose(img_chw, (1, 2, 0)) * 255).astype(np.uint8)
        cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img_hwc, (1, 0, 2))), (120, 120))
        screen.blit(cam_surf, (p4_x + 85, p4_y + 40))
        pygame.draw.rect(screen, (90, 110, 140), (p4_x + 84, p4_y + 39, 122, 122), 1)
        screen.blit(font_xs.render("Overhead Camera (64x64)", True, (150, 160, 180)), (p4_x + 75, p4_y + 166))

        # Proprioception
        pygame.draw.rect(screen, (30, 34, 46), (p4_x + 12, p4_y + 185, 270, 95), border_radius=4)
        screen.blit(font_bold.render("Proprioception State:", True, (255, 210, 110)), (p4_x + 20, p4_y + 192))
        p = obs["proprio"]
        screen.blit(font_xs.render(f"EE X (Fwd) : {p[0]:+.4f} m", True, (200, 210, 230)), (p4_x + 20, p4_y + 212))
        screen.blit(font_xs.render(f"EE Y (Lat) : {p[1]:+.4f} m", True, (200, 210, 230)), (p4_x + 20, p4_y + 228))
        screen.blit(font_xs.render(f"EE Z (Hgt) : {p[2]:+.4f} m", True, (200, 210, 230)), (p4_x + 20, p4_y + 244))
        grip_state_str = "CLOSED" if sim.gripper_closed else "OPEN"
        grip_c = (255, 120, 120) if sim.gripper_closed else (100, 220, 255)
        screen.blit(font_xs.render(f"Gripper    : {p[4]:.1f} ({grip_state_str})", True, grip_c), (p4_x + 20, p4_y + 260))

        # Live Trial Scoreboard
        pygame.draw.rect(screen, (30, 34, 46), (p4_x + 12, p4_y + 290, 270, 95), border_radius=4)
        screen.blit(font_bold.render("Live Evaluation Scoreboard:", True, (255, 180, 100)), (p4_x + 20, p4_y + 298))
        win_pct = (successful_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0
        screen.blit(font_xs.render(f"Trial     : #{total_episodes + 1}", True, (200, 210, 230)), (p4_x + 20, p4_y + 320))
        screen.blit(font_xs.render(f"Success   : {successful_episodes}", True, (80, 240, 120)), (p4_x + 20, p4_y + 338))
        screen.blit(font_xs.render(f"Fail/Time : {failed_episodes}", True, (255, 100, 100)), (p4_x + 20, p4_y + 356))
        screen.blit(font_xs.render(f"Rate: {win_pct:.1f}%", True, (200, 210, 230)), (p4_x + 160, p4_y + 338))
        screen.blit(font_xs.render(f"Tick: {episode_total_ticks}/220", True, (160, 170, 190)), (p4_x + 160, p4_y + 356))

        # Kinematic Dobot Arm Side Profile
        pygame.draw.rect(screen, (30, 34, 46), (p4_x + 12, p4_y + 395, 270, 130), border_radius=4)
        screen.blit(font_bold.render("Dobot Kinematics Profile:", True, (180, 200, 240)), (p4_x + 20, p4_y + 402))

        j1, j2, j3, j4 = sim.kin.inverse(sim.ee_pos[0], sim.ee_pos[1], sim.ee_pos[2], sim.ee_pos[3])
        r0, z0 = 0.08, 0.0
        r1, z1 = 0.08, sim.kin.L1
        r2 = r1 + sim.kin.L2 * np.cos(j2)
        z2 = z1 + sim.kin.L2 * np.sin(j2)
        r3 = r2 + sim.kin.L3 * np.cos(j2 + j3)
        z3 = z2 + sim.kin.L3 * np.sin(j2 + j3)
        r4 = np.sqrt(sim.ee_pos[0]**2 + sim.ee_pos[1]**2)
        z4 = sim.ee_pos[2]

        def w_side(r, z):
            sx = int(p4_x + 40 + ((r - 0.05) / 0.30) * 190)
            sy = int(p4_y + 510 - (z / 0.22) * 90)
            return sx, sy

        pts = [w_side(r, z) for r, z in [(r0, z0), (r1, z1), (r2, z2), (r3, z3), (r4, z4)]]
        for i in range(len(pts) - 1):
            pygame.draw.line(screen, (160, 180, 220), pts[i], pts[i+1], 2)
        for pt in pts[:-1]:
            pygame.draw.circle(screen, (255, 160, 40), pt, 3)
        pygame.draw.circle(screen, grip_c, pts[-1], 6)

        # Success / Timeout Banner
        if task_success_status is not None and success_banner_timer > 0:
            if task_success_status:
                banner_text = f"SUCCESS! ({successful_episodes}/{total_episodes})"
                banner_col = (50, 240, 100)
            else:
                banner_text = f"TIMEOUT ({successful_episodes}/{total_episodes})"
                banner_col = (255, 90, 90)
            pygame.draw.rect(screen, (30, 30, 30), (p4_x + 12, p4_y + 535, 270, 30), border_radius=4)
            screen.blit(font_bold.render(banner_text, True, banner_col), (p4_x + 30, p4_y + 542))

        # =============================================================
        # BOTTOM HUD: Interactive Controls
        # =============================================================
        pygame.draw.rect(screen, (20, 23, 30), (20, 642, 1360, 120), border_radius=6)
        screen.blit(font_bold.render("INTERACTIVE VISUALIZER CONTROLS:", True, (120, 210, 255)), (35, 652))
        ctrl_str1 = "[F] Toggle Speed (Normal / Fast 3x / Slow 0.5x)  |  [TAB] Cycle Layer Activations  |  [R] Randomize Table Clutter"
        ctrl_str2 = "[1] Pick & Place Task  |  [2] Push Task  |  Live Closed-Loop Execution with Full SmolVLM Multimodal Backbone"
        screen.blit(font.render(ctrl_str1, True, (210, 220, 240)), (35, 674))
        screen.blit(font.render(ctrl_str2, True, (170, 180, 200)), (35, 696))

        pygame.display.flip()
        clock.tick(fps_target)

    pygame.quit()

if __name__ == "__main__":
    visualize_full_smolvla()
