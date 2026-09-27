import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FIG_DIR = os.path.join("paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def draw_tensor(ax, x, y, z, dx, dy, dz, color, alpha=0.8, edge_color='black', text="", text_z_offset=2):
    """Draws a 3D rectangular block representing a tensor."""
    v = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ])
    faces = [
        [v[0],v[1],v[2],v[3]], # bottom
        [v[4],v[5],v[6],v[7]], # top
        [v[0],v[1],v[5],v[4]], # front
        [v[2],v[3],v[7],v[6]], # back
        [v[1],v[2],v[6],v[5]], # right
        [v[0],v[3],v[7],v[4]]  # left
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolors=color, linewidths=0.5, edgecolors=edge_color)
    ax.add_collection3d(poly)
    if text:
        ax.text(x+dx/2, y+dy/2, z+dz + text_z_offset, text, color='black', ha='center', va='center', fontsize=9, fontweight='bold', zorder=100)

def draw_line(ax, p1, p2, color='gray', style='--', alpha=0.5, lw=1):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linestyle=style, alpha=alpha, linewidth=lw)

def configure_3d_axes(ax, xlim, ylim, zlim, box_aspect=(1,1,1.5)):
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=20, azim=-55)

# =============================================================================
# 1. Manual VLA 3D Architecture
# =============================================================================
def generate_manual_vla_3d():
    fig = plt.figure(figsize=(10, 14), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    configure_3d_axes(ax, [-15, 25], [-15, 15], [0, 90])
    
    # 0. Image
    draw_tensor(ax, -10, -10, 0, 20, 20, 1, 'dodgerblue', alpha=0.5, text="RGB Image\n(3x64x64)", text_z_offset=2)
    
    # Lines up
    draw_line(ax, (0, 0, 1), (0, 0, 15), lw=2)
    
    # 1. Patch Tokens
    for i in range(4):
        for j in range(4):
            draw_tensor(ax, -10 + i*5.5, -10 + j*5.5, 15, 4, 4, 8, 'cyan', alpha=0.8)
    ax.text(0, 0, 27, "Conv Patch Tokens\n(16x16 Grid, 128-dim)", ha='center', va='center', fontweight='bold', zorder=100)
    
    # 1.5 Intent Vector
    draw_tensor(ax, 15, -10, 15, 4, 20, 8, 'limegreen', alpha=0.8, text="Intent Vector\n(16-dim)", text_z_offset=2)
    
    # Lines routing to Hadamard
    draw_line(ax, (0, 0, 23), (0, 0, 35), lw=2, color='cyan')
    draw_line(ax, (17, 0, 23), (8, 0, 35), lw=2, color='limegreen')
    
    # 2. Conditioned Patches (Hadamard Fusion)
    for i in range(4):
        for j in range(4):
            draw_tensor(ax, -10 + i*5.5, -10 + j*5.5, 35, 4, 4, 8, 'mediumpurple', alpha=0.9)
    ax.text(0, 0, 47, "Hadamard Conditioned\nTokens", ha='center', va='center', fontweight='bold', zorder=100)
    
    # Lines to transformer
    draw_line(ax, (0, 0, 43), (0, 0, 55), lw=2)
    
    # 3. Transformer Encoder Layers (Wide planes)
    draw_tensor(ax, -12, -12, 55, 24, 24, 2, 'orange', alpha=0.6, text="Transformer Layer 1", text_z_offset=2)
    draw_tensor(ax, -12, -12, 63, 24, 24, 2, 'orange', alpha=0.6, text="Transformer Layer 2", text_z_offset=2)
    
    draw_line(ax, (0, 0, 65), (0, 0, 75), lw=2)
    
    # 4. Action Output
    draw_tensor(ax, -3, -3, 75, 6, 6, 12, 'crimson', alpha=0.9, text="Action Output\n[dx, dy, dz, dyaw, grip]", text_z_offset=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_manual_vla_arch.png"), bbox_inches='tight')
    plt.close()

# =============================================================================
# 2. SmolVLA Dual-Stream 3D Architecture
# =============================================================================
def generate_smolvla_3d():
    fig = plt.figure(figsize=(12, 14), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    configure_3d_axes(ax, [-25, 25], [-15, 15], [0, 90])
    ax.view_init(elev=15, azim=-60)
    
    # Level 0
    draw_tensor(ax, -22, -10, 0, 16, 16, 1, 'dodgerblue', alpha=0.5, text="Visual Input\n(224x224)", text_z_offset=2)
    draw_tensor(ax, 6, -10, 0, 16, 16, 1, 'limegreen', alpha=0.5, text="Language Prompt\n(77 Tokens)", text_z_offset=2)
    
    draw_line(ax, (-14, -2, 1), (-14, -2, 15), lw=2)
    draw_line(ax, (14, -2, 1), (14, -2, 15), lw=2)
    
    # Level 1: Patches and Tokens
    # 7x7 vision patches (represented as 3x3 for clarity)
    for i in range(3):
        for j in range(3):
            draw_tensor(ax, -22 + i*5.5, -10 + j*5.5, 15, 4, 4, 12, 'cyan', alpha=0.8)
    ax.text(-14, -2, 31, "CLIP Vision Patches\n(49 x 512-dim)", ha='center', va='center', fontweight='bold', zorder=100)
    
    # 16 language tokens (represented as 4x4 strips)
    for i in range(4):
        for j in range(4):
            draw_tensor(ax, 6 + i*4, -10 + j*4, 15, 3, 3, 12, 'springgreen', alpha=0.8)
    ax.text(14, -2, 31, "CLIP Text Tokens\n(77 x 512-dim)", ha='center', va='center', fontweight='bold', zorder=100)
    
    # Level 2: Cross Attention Fusion Web
    for i in range(3):
        for j in range(3):
            # connect random vision to random text to form a web
            vx, vy = -22 + i*5.5 + 2, -10 + j*5.5 + 2
            tx, ty = 6 + (i)*4 + 1.5, -10 + (j)*4 + 1.5
            draw_line(ax, (tx, ty, 27), (vx, vy, 45), color='fuchsia', alpha=0.6, lw=1.5)
            draw_line(ax, (vx, vy, 27), (vx, vy, 45), color='cyan', alpha=0.6, lw=1.5)
            
    # Level 3: Fused Memory
    draw_tensor(ax, -16, -12, 45, 24, 24, 6, 'mediumpurple', alpha=0.8, text="Multimodal Cross-Attention\nFusion Context (67 Tokens)", text_z_offset=2)
    
    draw_line(ax, (-4, 0, 51), (-4, 0, 60), lw=2)
    
    # Level 4: Action Decoder & Flow Head
    draw_tensor(ax, -12, -8, 60, 16, 16, 4, 'orange', alpha=0.8, text="Action Decoder", text_z_offset=1)
    
    # Flow Matching Trajectory curve
    z_curve = np.linspace(64, 85, 20)
    x_curve = -4 + np.sin(z_curve/3) * 5
    y_curve = np.cos(z_curve/3) * 5
    ax.plot(x_curve, y_curve, z_curve, color='crimson', lw=4, zorder=150)
    ax.text(-4, 0, 88, "Flow Matching\nVelocity Field [128x4]", color='crimson', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_smolvla_arch.png"), bbox_inches='tight')
    plt.close()

# =============================================================================
# 3. Full SmolVLA Foundation 3D Architecture
# =============================================================================
def generate_full_smolvla_3d():
    fig = plt.figure(figsize=(10, 14), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    configure_3d_axes(ax, [-15, 25], [-15, 15], [0, 100])
    ax.view_init(elev=20, azim=-45)
    
    # 0. Foundation Input
    draw_tensor(ax, -10, -10, 0, 20, 20, 2, 'dodgerblue', alpha=0.6, text="Multimodal Interleaved\nTokens (Image + Chat)", text_z_offset=3)
    
    # Foundation Layers (30 layers of SmolVLM)
    for l in range(1, 31):
        color = 'fuchsia' if l in [10, 20, 30] else 'gray'
        alpha = 0.9 if l in [10, 20, 30] else 0.15
        z_pos = l * 2
        draw_tensor(ax, -10, -10, z_pos, 20, 20, 1, color, alpha=alpha, edge_color='none' if l not in [10,20,30] else 'black')
        
        # Extraction taps
        if l in [10, 20, 30]:
            draw_line(ax, (10, 0, z_pos+0.5), (20, 0, 40), color='fuchsia', lw=2, alpha=0.8)
            ax.text(-12, -12, z_pos, f"Layer {l}", fontsize=7)
            
    ax.text(0, 0, 65, "SmolVLM-256M\n(Frozen 30-Layer Backbone)", ha='center', va='center', fontweight='bold', zorder=100)
    
    # Concatenated Multi-Layer Memory
    draw_tensor(ax, 15, -8, 35, 10, 16, 12, 'darkviolet', alpha=0.9, text="Concatenated\nContext (1728-dim)", text_z_offset=2)
    
    draw_line(ax, (20, 0, 47), (20, 0, 60), color='darkviolet', lw=2)
    
    # Action Expert
    draw_tensor(ax, 15, -8, 60, 10, 16, 5, 'orange', alpha=0.9, text="Action Expert", text_z_offset=2)
    
    # Output curve
    z_curve = np.linspace(65, 90, 20)
    x_curve = 20 + np.sin(z_curve/4) * 4
    y_curve = np.cos(z_curve/4) * 4
    ax.plot(x_curve, y_curve, z_curve, color='crimson', lw=4, zorder=150)
    ax.text(20, 0, 93, "Continuous Action\nTrajectory", color='crimson', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_full_smolvla_arch.png"), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating rich 3D block architectures...")
    generate_manual_vla_3d()
    print("  [OK] fig1_manual_vla_arch.png")
    generate_smolvla_3d()
    print("  [OK] fig2_smolvla_arch.png")
    generate_full_smolvla_3d()
    print("  [OK] fig3_full_smolvla_arch.png")
    print("Done.")
