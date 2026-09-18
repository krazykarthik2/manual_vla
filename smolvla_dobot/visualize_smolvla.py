import os
import sys
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
sys.path.append(os.path.dirname(__file__))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from smolvla_embedding import SmolVLMTokenizer
from smolvla_model import SmolVLABackbone

def visualize_all_smolvla_inputs():
    sim = DobotPickPlaceSim()
    obs = sim.reset(random_scene=True, num_distractors=3, action_type="pick_place")

    tokenizer = SmolVLMTokenizer()
    embedder = SmolVLABackbone(d_model=512, num_layers=2)
    embedder.eval()

    pygame.init()
    # High-Density Dashboard: 1200 x 680
    screen = pygame.display.set_mode((1200, 680))
    pygame.display.set_caption("SmolVLA Multi-Input Visualizer (Pretrained CLIP ViT Patches + Subwords)")

    font_xs = pygame.font.SysFont("Consolas", 10)
    font_sm = pygame.font.SysFont("Consolas", 11)
    font = pygame.font.SysFont("Consolas", 12)
    font_bold = pygame.font.SysFont("Consolas", 13, bold=True)
    font_title = pygame.font.SysFont("Arial", 15, bold=True)

    selected_token_idx = 4
    clock = pygame.time.Clock()
    running = True

    model_path = os.path.join(os.path.dirname(__file__), "models", "dobot_bc_policy.pth")
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            bb_state = {}
            for k, v in ckpt.items():
                if k.startswith("backbone."):
                    bb_state[k.replace("backbone.", "")] = v
            if bb_state:
                embedder.load_state_dict(bb_state, strict=False)
                print("[INFO] Loaded trained backbone checkpoint in visualizer.", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load checkpoint weights into visualizer: {e}", flush=True)

    while running:
        prompt_text = sim.instruction
        token_ids = tokenizer.encode(prompt_text, max_len=77)
        active_tokens = tokenizer.decode_active_tokens(token_ids)
        if selected_token_idx >= len(active_tokens):
            selected_token_idx = max(0, len(active_tokens) - 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs = sim.reset(random_scene=True, num_distractors=3)
                elif event.key == pygame.K_UP:
                    selected_token_idx = max(0, selected_token_idx - 1)
                elif event.key == pygame.K_DOWN:
                    selected_token_idx = min(len(active_tokens) - 1, selected_token_idx + 1)
                elif event.key == pygame.K_1:
                    sim.action_type = "pick_place"
                    obs = sim.reset(random_scene=True, num_distractors=3, action_type="pick_place")
                elif event.key == pygame.K_2:
                    sim.action_type = "push"
                    obs = sim.reset(random_scene=True, num_distractors=3, action_type="push")

        # Forward pass through Pretrained SmolVLA Backbone
        img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)
        tokens_t = token_ids.unsqueeze(0)
        with torch.no_grad():
            text_feats, vis_feats, cross_weights = embedder(tokens_t, img_t)
            all_tokens_weights = cross_weights[0].numpy() # [77, 49]
            cur_grid = cross_weights[0, selected_token_idx].view(7, 7).numpy()

        screen.fill((16, 18, 24))

        # Title Bar
        screen.blit(font_title.render("SMOLVLA MULTI-INPUT OBSERVATION & ATTENTION DASHBOARD", True, (240, 245, 255)), (25, 12))
        sub_title = f"Task Prompt: \"{prompt_text}\" | Vision Patches: 49 (7x7) | Tokens: {len(active_tokens)} | Proprioception: 5D"
        screen.blit(font_sm.render(sub_title, True, (130, 140, 160)), (25, 33))

        # =============================================================
        # PANEL 1: Interactive Language Token Selector (Left: 220px)
        # =============================================================
        pygame.draw.rect(screen, (24, 27, 36), (20, 58, 220, 555), border_radius=6)
        screen.blit(font_bold.render("1. LANGUAGE TOKENS", True, (255, 200, 90)), (30, 68))
        screen.blit(font_xs.render("[UP/DOWN] Select Word", True, (140, 150, 170)), (30, 86))

        y_offset = 110
        for idx, tok_str in enumerate(active_tokens[:16]):
            is_selected = (idx == selected_token_idx)
            btn_rect = pygame.Rect(28, y_offset, 204, 24)
            if is_selected:
                pygame.draw.rect(screen, (45, 85, 170), btn_rect, border_radius=4)
                pygame.draw.rect(screen, (100, 180, 255), btn_rect, 1, border_radius=4)
                text_col = (255, 255, 255)
                tag = ">"
            else:
                pygame.draw.rect(screen, (30, 34, 46), btn_rect, border_radius=4)
                text_col = (170, 180, 200)
                tag = " "
            screen.blit(font_xs.render(f"{tag}[{idx:02d}] {tok_str}", True, text_col), (34, y_offset + 5))
            y_offset += 27

        # =============================================================
        # PANEL 2: Focused 7x7 Patch Heatmap for Selected Token
        # =============================================================
        pygame.draw.rect(screen, (24, 27, 36), (250, 58, 305, 555), border_radius=6)
        sel_word = active_tokens[selected_token_idx] if selected_token_idx < len(active_tokens) else "N/A"
        screen.blit(font_bold.render(f"2. ATTENTION FOR '{sel_word.upper()}'", True, (100, 220, 255)), (260, 68))
        screen.blit(font_xs.render("7x7 Pretrained ViT-B/32 Patch Grid", True, (140, 150, 170)), (260, 86))

        g_min, g_max = cur_grid.min(), cur_grid.max()
        cur_grid_norm = (cur_grid - g_min) / (g_max - g_min + 1e-6) if g_max > g_min else cur_grid

        # Focused Large Patch Grid (7x7)
        cell_sz = 36
        start_x, start_y = 277, 115
        for r in range(7):
            for c in range(7):
                val = cur_grid_norm[r, c]
                col = (int(25 + val * 230), int(35 + val * 170), int(55 + (1 - val) * 45))
                pygame.draw.rect(screen, col, (start_x + c * cell_sz, start_y + r * cell_sz, cell_sz - 2, cell_sz - 2))
        pygame.draw.rect(screen, (80, 95, 125), (start_x - 2, start_y - 2, 7 * cell_sz + 1, 7 * cell_sz + 1), 1)

        # Patch Token Vector Stats
        screen.blit(font_xs.render(f"Selected Token: [{selected_token_idx}] '{sel_word}'", True, (180, 190, 210)), (265, 385))
        screen.blit(font_xs.render(f"Cross-Attn Range: [{g_min:.4f} .. {g_max:.4f}]", True, (180, 190, 210)), (265, 405))
        screen.blit(font_xs.render(f"CLIP ViT Dim: [B=1, 49, d=512]", True, (140, 150, 170)), (265, 425))

        # =============================================================
        # PANEL 3: Compact Grid of ALL Language Tokens (Center-Right)
        # =============================================================
        pygame.draw.rect(screen, (24, 27, 36), (565, 58, 365, 555), border_radius=6)
        screen.blit(font_bold.render("3. ALL INPUT TOKEN PATCH HEATMAPS", True, (255, 160, 100)), (575, 68))
        screen.blit(font_xs.render("Simultaneous pretrained cross-attention for all words", True, (140, 150, 170)), (575, 86))

        mini_cols = 3
        mini_cell_sz = 10 # 10px * 7 = 70px square per token
        m_start_x, m_start_y = 580, 110

        for t_idx, tok_str in enumerate(active_tokens[:9]):
            col_i = t_idx % mini_cols
            row_i = t_idx // mini_cols
            tx = m_start_x + col_i * 115
            ty = m_start_y + row_i * 125

            is_cur = (t_idx == selected_token_idx)
            header_col = (100, 220, 255) if is_cur else (160, 170, 190)
            screen.blit(font_xs.render(f"[{t_idx}] {tok_str[:8]}", True, header_col), (tx, ty))

            t_grid = all_tokens_weights[t_idx].reshape(7, 7)
            tg_min, tg_max = t_grid.min(), t_grid.max()
            tg_norm = (t_grid - tg_min) / (tg_max - tg_min + 1e-6) if tg_max > tg_min else t_grid

            for r in range(7):
                for c in range(7):
                    v = tg_norm[r, c]
                    c_col = (int(20 + v * 235), int(30 + v * 170), int(50 + (1 - v) * 40))
                    pygame.draw.rect(screen, c_col, (tx + c * mini_cell_sz, ty + 16 + r * mini_cell_sz, mini_cell_sz - 1, mini_cell_sz - 1))
            
            border_col = (100, 220, 255) if is_cur else (60, 70, 90)
            pygame.draw.rect(screen, border_col, (tx - 1, ty + 15, 7 * mini_cell_sz + 1, 7 * mini_cell_sz + 1), 1)

        # =============================================================
        # PANEL 4: Sensor Inputs (Right Column: 235px)
        # =============================================================
        pygame.draw.rect(screen, (24, 27, 36), (940, 58, 240, 555), border_radius=6)
        screen.blit(font_bold.render("4. SENSOR INPUTS", True, (120, 240, 150)), (950, 68))
        screen.blit(font_xs.render("Camera & Dobot Proprio", True, (140, 150, 170)), (950, 86))

        img_hwc = (np.transpose(obs["image"], (1, 2, 0)) * 255).astype(np.uint8)
        cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img_hwc, (1, 0, 2))), (130, 130))
        screen.blit(cam_surf, (995, 110))
        pygame.draw.rect(screen, (90, 110, 140), (994, 109, 132, 132), 1)
        screen.blit(font_xs.render("64x64 Overhead RGB Feed", True, (150, 160, 180)), (985, 246))

        pygame.draw.rect(screen, (30, 34, 46), (950, 270, 220, 115), border_radius=4)
        screen.blit(font_bold.render("Proprioception [5D]:", True, (255, 210, 110)), (960, 278))
        p = obs["proprio"]
        screen.blit(font_xs.render(f"X (Forward) : {p[0]:+.4f} m", True, (200, 210, 230)), (960, 298))
        screen.blit(font_xs.render(f"Y (Lateral) : {p[1]:+.4f} m", True, (200, 210, 230)), (960, 316))
        screen.blit(font_xs.render(f"Z (Height)  : {p[2]:+.4f} m", True, (200, 210, 230)), (960, 334))
        grip_state_str = "CLOSED" if sim.gripper_closed else "OPEN"
        screen.blit(font_xs.render(f"Gripper     : {p[4]:.1f} ({grip_state_str})", True, (255, 120, 120) if sim.gripper_closed else (100, 220, 255)), (960, 352))

        # Kinematics
        pygame.draw.rect(screen, (30, 34, 46), (950, 395, 220, 150), border_radius=4)
        screen.blit(font_bold.render("Kinematic Elevation:", True, (180, 200, 240)), (960, 403))
        
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
            sx = int(975 + ((r - 0.05) / 0.30) * 170)
            sy = int(525 - (z / 0.22) * 105)
            return sx, sy

        pts = [w_side(r, z) for r, z in [(r0, z0), (r1, z1), (r2, z2), (r3, z3), (r4, z4)]]
        for i in range(len(pts) - 1):
            pygame.draw.line(screen, (160, 180, 220), pts[i], pts[i+1], 2)
        for pt in pts[:-1]:
            pygame.draw.circle(screen, (255, 160, 40), pt, 3)
        grip_c = (255, 80, 80) if sim.gripper_closed else (80, 220, 255)
        pygame.draw.circle(screen, grip_c, pts[-1], 5)

        # Controls HUD
        pygame.draw.rect(screen, (20, 23, 30), (20, 622, 1160, 44), border_radius=6)
        hud_str = "[UP/DOWN] Select Word | [R] Randomize Scene & Clutter | [1] Pick & Place Task | [2] Push Task"
        screen.blit(font_bold.render("CONTROLS:", True, (120, 210, 255)), (35, 636))
        screen.blit(font.render(hud_str, True, (200, 210, 230)), (130, 636))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    visualize_all_smolvla_inputs()
