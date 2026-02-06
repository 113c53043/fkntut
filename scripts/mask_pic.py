import os
import sys

# === 路徑設定 (必須在 import ldm 之前執行) ===
# 1. 取得當前腳本所在目錄 (scripts/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 取得上一層專案根目錄 (mas_GRDH/)
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# 3. 將專案根目錄加入 Python 搜尋路徑，這樣 Python 才能找到 'ldm' 模組
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# === 現在可以安全匯入專案模組了 ===
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
# 這些 import 必須在 sys.path.append 之後
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torch import autocast

# 設定 Checkpoint 路徑
MAS_GRDH_PATH = PARENT_DIR # 設定為根目錄
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

def load_model():
    print(f"⏳ Loading SD Model for Visualization...", flush=True)
    config = OmegaConf.load(CONFIG_PATH)
    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    except:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def visualize_mask_logic():
    # 1. 準備模型與參數
    model = load_model()
    sampler = DPMSolverSampler(model)
    device = "cuda"
    
    # 2. 隨機選一個 Prompt 進行測試
    prompt = "A photo of a calm blue sky over a green grassy field"
    print(f"\n🧪 Testing Prompt: '{prompt}'")
    print("(這是一個經典測試案例：天空是平滑區，草地是紋理區)")
    
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    
    # 3. 產生初始雜訊 (z_center)
    shape = (4, 64, 64)
    z_center = torch.randn(1, *shape, device=device)
    
    # 4. 執行 Variance 計算 (模擬 estimate_uncertainty)
    print("⚡ Calculating Variance (Monte Carlo Sampling)...")
    repeats = 5
    z_recs = []
    scale = 5.0
    
    with torch.no_grad(), autocast("cuda"):
        # 先生成一張大概的圖 (作為參考)
        z_0_ref, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=shape,
                                  unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                  x_T=z_center, verbose=False)
        x_samples = model.decode_first_stage(z_0_ref)
        img_ref = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        # [Fix] Matplotlib 不支援 float16，需轉為 float32
        img_ref_np = img_ref.float().cpu().numpy()[0].transpose(1, 2, 0)
        
        # Monte Carlo 估計 Variance
        for i in range(repeats):
            noise = torch.randn_like(z_center) * 0.05
            z_input = z_center + noise
            # Fast loop
            z_0, _ = sampler.sample(steps=10, conditioning=c, batch_size=1, shape=shape,
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_input, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=10, conditioning=c, batch_size=1, shape=shape,
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            z_recs.append(z_rec)

    stack = torch.stack(z_recs)
    variance = torch.var(stack, dim=0)
    variance_mean = torch.mean(variance, dim=1, keepdim=True) # (1, 1, 64, 64)
    
    # 5. 計算 Mask (兩種邏輯對比)
    v_min = torch.quantile(variance_mean, 0.01) 
    v_max = torch.quantile(variance_mean, 0.99)
    denom = v_max - v_min if (v_max - v_min) > 1e-8 else 1.0
    
    norm_var = (variance_mean - v_min) / denom
    norm_var = torch.clamp(norm_var, 0.0, 1.0)
    
    mask_power = 6.0
    norm_var_powered = torch.pow(norm_var, mask_power)
    
    # === 邏輯 A：目前的 (Current) ===
    mask_wrong = 1.0 - norm_var_powered
    
    # === 邏輯 B：修正後的 (Corrected) ===
    mask_correct = norm_var_powered
    
    # [Fix] 轉為 Numpy 方便繪圖 (需轉為 float32 以避免 Matplotlib 報錯)
    v_map = norm_var[0, 0].float().cpu().numpy()
    m_wrong = mask_wrong[0, 0].float().cpu().numpy()
    m_correct = mask_correct[0, 0].float().cpu().numpy()
    
    # 6. 繪製並儲存比較圖
    print("\n🎨 Saving comparison visualization to 'mask_logic_check.png'...")
    plt.figure(figsize=(20, 5))
    
    # 子圖 1: 參考原圖
    plt.subplot(1, 4, 1)
    plt.imshow(img_ref_np)
    plt.title("Reference Image\n(Check Sky vs Grass)")
    plt.axis('off')
    
    # 子圖 2: Variance Map
    plt.subplot(1, 4, 2)
    plt.imshow(v_map, cmap='jet')
    plt.title("Variance Map\n(Blue=Smooth, Red=Texture)")
    plt.colorbar()
    plt.axis('off')
    
    # 子圖 3: 目前的邏輯 (1.0 - Var)
    plt.subplot(1, 4, 3)
    plt.imshow(m_wrong, cmap='jet', vmin=0, vmax=1)
    plt.title("Current Logic (1.0 - Var)\nRed = Strong Embedding")
    plt.colorbar()
    plt.axis('off')
    
    # 子圖 4: 修正後的邏輯 (Var)
    plt.subplot(1, 4, 4)
    plt.imshow(m_correct, cmap='jet', vmin=0, vmax=1)
    plt.title("Corrected Logic (Var Only)\nRed = Strong Embedding")
    plt.colorbar()
    plt.axis('off')
    
    plt.savefig("mask_logic_check.png", bbox_inches='tight')
    plt.close()
    print("✅ Done! Please open 'mask_logic_check.png'.")

if __name__ == "__main__":
    visualize_mask_logic()