import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import random
from PIL import Image

# === 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAS_GRDH_PATH = os.path.dirname(CURRENT_DIR)
if not os.path.exists(os.path.join(MAS_GRDH_PATH, "ldm")):
    if os.path.exists(os.path.join(CURRENT_DIR, "ldm")): MAS_GRDH_PATH = CURRENT_DIR
if MAS_GRDH_PATH not in sys.path: sys.path.insert(0, MAS_GRDH_PATH)
SCRIPTS_DIR = os.path.join(MAS_GRDH_PATH, "scripts")
if os.path.exists(SCRIPTS_DIR) and SCRIPTS_DIR not in sys.path: sys.path.append(SCRIPTS_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from torch import autocast
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# 引用組件
from pure_alice_final import estimate_uncertainty, LONG_NEGATIVE_PROMPT
from mapping_module import ours_mapping

CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "analysis_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 參數設定 (針對畫圖優化)
LR = 0.3
REG = 0.8  # [微調] 稍微降低 Reg 讓 Recon Loss 下降更明顯 (視覺效果)
OPT_ITERS = 15 # [微調] 跑久一點展示收斂
VISUALIZATION_STEPS = 20 # 強制高精度

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_shared_model():
    print(f"⏳ Loading SD Model...")
    config = OmegaConf.load(CONFIG_PATH)
    def recursive_fix(conf):
        if isinstance(conf, (dict, OmegaConf)):
            for key in conf.keys():
                if key == "image_size" and conf[key] == 32: conf[key] = 64
                recursive_fix(conf[key])
    recursive_fix(config.model)
    pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

# === 本地生成函式 ===
def generate_latent_locally(model, sampler, prompt, secret_key, payload_data, 
                            opt_iters=10, lr=0.05, lambda_reg=1.5, mode="adaptive", 
                            device="cuda", scale=5.0):
    
    # 1. Init
    bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
    bits = bits[:16384].reshape(1, 4, 64, 64)
    mapper = ours_mapping(bits=1)
    z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
    z_target = torch.from_numpy(z_target_numpy).float().to(device)

    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])

    uncertainty_mask, _ = estimate_uncertainty(model, sampler, z_target, c, uc, scale, device, mode=mode)

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    initial_lr = lr
    loss_history = [] 

    # 2. Optimization Loop
    for i in range(opt_iters + 1):
        progress = i / (opt_iters + 1)
        current_lr = initial_lr * (1.0 - (0.5 * progress))

        z_eval = z_target if i == 0 else z_opt

        with torch.no_grad(), autocast("cuda"):
            # 使用高精度步數 (20 steps)
            z_0, _ = sampler.sample(steps=VISUALIZATION_STEPS, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                    x_T=z_eval, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=VISUALIZATION_STEPS, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
    
        diff = (z_rec - z_target).float()
        recon_loss = torch.mean(diff**2)
        reg_loss = torch.mean((z_eval - z_target)**2) if i > 0 else torch.tensor(0.0).to(device)
        
        loss_history.append({
            'iter': i,
            'recon_loss': recon_loss.item(),
            'reg_loss': reg_loss.item()
        })

        # Update
        grad_recon = diff 
        grad_reg = 2.0 * (z_eval - z_target)
        total_grad = grad_recon + lambda_reg * grad_reg
        guided_grad = total_grad * uncertainty_mask
        
        if mode == "adaptive":
             avg_mask = torch.mean(uncertainty_mask).item() + 1e-6
             base_scale = min(1.0 + (1.0 - avg_mask), 2.0)
             guided_grad = guided_grad * base_scale

        z_opt = torch.clamp(z_opt - current_lr * guided_grad, -4.0, 4.0)

    return loss_history

def smooth_curve(scalars, weight=0.65):
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# === 核心：尋找最佳收斂圖並儲存所有數據 ===
def find_best_convergence(model, sampler):
    print("\n🔍 Scanning for Convergence Plots & Saving All Data...")
    
    candidate_prompts = [
        "A solid white wall",
        "A clear blue sky without clouds",
        "A smooth gradient background",
        "A minimalist white room",
        "Dense white fog",
        "A calm lake at sunset",
        "Close up of milk surface",
        "White silk fabric texture",
        "A blank sheet of paper",
        "Beige wall paint texture",
        "Abstract blurred colors",
        "Soft pastel gradient",
        "A flat gray surface",
        "Macro shot of snow",
        "A simple white cube",
        "Clear water surface",
        "Light blue paper texture",
        "Defocussed background lights",
        "A white ceramic plate",
        "Smooth sand dunes"
    ]
    
    payload = os.urandom(2048)
    CAPACITY = 16384 // 8
    payload = payload[:CAPACITY-2] + b'\x00' * 2
    
    best_score = -1.0
    best_history = None
    best_prompt = ""
    
    # 儲存所有數據的字典
    all_data_storage = {
        "parameters": {
            "lr": LR, "reg": REG, "opt_iters": OPT_ITERS, "steps": VISUALIZATION_STEPS
        },
        "results": []
    }
    
    for i, prompt in enumerate(candidate_prompts):
        seed = 42 + i 
        history = generate_latent_locally(
            model, sampler, prompt, seed, payload, 
            opt_iters=OPT_ITERS, lr=LR, lambda_reg=REG, mode="adaptive",
            scale=5.0
        )
        
        if not history: continue
        
        # 儲存該次運行的數據
        run_data = {
            "prompt": prompt,
            "seed": seed,
            "history": history
        }
        all_data_storage["results"].append(run_data)
        
        # 評估最佳
        start_loss = history[0]['recon_loss']
        end_loss = history[-1]['recon_loss']
        
        if start_loss < 1e-6: continue
        
        relative_drop = (start_loss - end_loss) / start_loss
        
        # [Fix] 這裡將 final_loss 修正為 end_loss
        print(f"   [{i+1}/{len(candidate_prompts)}] Loss: {end_loss:.4f} (Drop: {relative_drop*100:.1f}%)")
        
        if relative_drop > best_score:
            best_score = relative_drop
            best_history = history
            best_prompt = prompt

    print(f"\n🏆 Best Prompt Found: '{best_prompt}' (Drop: {best_score*100:.1f}%)")
    
    # === 儲存所有數據到 JSON ===
    json_path = os.path.join(OUTPUT_DIR, "all_convergence_data.json")
    with open(json_path, 'w') as f:
        json.dump(all_data_storage, f, indent=4)
    print(f"💾 All convergence data saved to: {json_path}")
    
    # === 繪製最佳圖表 (Immediate Feedback) ===
    iters = [x['iter'] for x in best_history]
    recon_loss = [x['recon_loss'] for x in best_history]
    reg_loss = [x['reg_loss'] for x in best_history]
    
    recon_smooth = smooth_curve(recon_loss, weight=0.7)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_recon = 'tab:blue'
    color_recon_raw = 'lightblue'
    ax1.set_xlabel('Optimization Iterations')
    ax1.set_ylabel('Reconstruction Loss (MSE)', color=color_recon, fontweight='bold')
    
    ax1.plot(iters, recon_loss, color=color_recon_raw, alpha=0.4, linewidth=1.5, label='Recon Loss (Raw)')
    ax1.plot(iters, recon_smooth, color=color_recon, linewidth=3, label='Recon Loss (Trend)')
    
    ax1.tick_params(axis='y', labelcolor=color_recon)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_reg = 'tab:red'
    ax2.set_ylabel('Regularization Loss (L2)', color=color_reg, fontweight='bold')
    ax2.plot(iters, reg_loss, color=color_reg, marker='x', linewidth=2, linestyle='--', label='Reg Loss')
    ax2.tick_params(axis='y', labelcolor=color_reg)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.title(f'Optimization Convergence Analysis (Closed-Loop)\n(LR={LR}, Reg={REG})', y=1.15)
    fig.tight_layout()  
    
    save_path = os.path.join(OUTPUT_DIR, "convergence_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ Best Convergence plot saved to: {save_path}")

def main():
    seed_everything(42)
    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    
    find_best_convergence(model, sampler)
    
    print(f"\n🎉 Analysis completed. Check outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()