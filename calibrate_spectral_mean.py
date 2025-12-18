import sys
import os
import torch
import torch.fft
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm
import json
import random

# === 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    pass 

# === 設定 ===
MAS_GRDH_PATH = CURRENT_DIR
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
DIR_CAPTIONS = os.path.join(CURRENT_DIR, "scripts/coco_annotations", "captions_val2017.json")
TOTAL_SAMPLES = 1000 # 校準樣本數

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def calculate_low_freq_ratio(z_latent):
    """計算單張 Latent 的低頻佔比"""
    freq_domain = torch.fft.fftn(z_latent, dim=(-2, -1))
    energy = torch.abs(freq_domain)
    
    h, w = energy.shape[-2:]
    h_center, w_center = h // 2, w // 2
    r_h, r_w = h // 4, w // 4 
    
    total_energy = torch.sum(energy)
    energy_shifted = torch.fft.fftshift(energy, dim=(-2, -1))
    low_freq_energy = torch.sum(energy_shifted[..., h_center-r_h:h_center+r_h, w_center-r_w:w_center+r_w])
    
    return (low_freq_energy / (total_energy + 1e-8)).item()

def main():
    print(f"🚀 Starting Spectral Ratio Calibration (N={TOTAL_SAMPLES})...")
    
    device = torch.device("cuda")
    config = OmegaConf.load(CONFIG_PATH)
    model = load_model_from_config(config, CKPT_PATH, device)
    sampler = DPMSolverSampler(model)

    # 載入 Prompts
    with open(DIR_CAPTIONS, 'r') as f:
        data = json.load(f)
    captions = [item['caption'] for item in data['annotations']]
    random.shuffle(captions)
    prompts = captions[:TOTAL_SAMPLES]

    ratios = []
    
    print("running sampling...")
    with torch.no_grad(), autocast("cuda"):
        for prompt in tqdm(prompts):
            # 1. 準備條件
            c = model.get_learned_conditioning([prompt])
            uc = model.get_learned_conditioning([""])
            
            # 2. 模擬 z_target (隨機噪聲即可，因為重點是 prompt 決定的 z_0)
            z_center = torch.randn(1, 4, 64, 64, device=device)
            
            # 3. 加入微小擾動 (模擬 estimate_uncertainty 的行為)
            noise = torch.randn_like(z_center) * 0.05
            z_input = z_center + noise
            
            # 4. 快速採樣 (只跑一次 DPM Encode 得到 z_0)
            # 我們只需要 z_0 的頻譜，不需要後面的反演
            z_0, _ = sampler.sample(steps=10, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                    x_T=z_input, DPMencode=False, DPMdecode=True, verbose=False)
            
            # 5. 計算 Ratio
            ratio = calculate_low_freq_ratio(z_0)
            ratios.append(ratio)

    # === 統計結果 ===
    ratios = np.array(ratios)
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    min_ratio = np.min(ratios)
    max_ratio = np.max(ratios)

    print("\n" + "="*50)
    print("📊 Calibration Result (COCO Validation)")
    print("-" * 50)
    print(f"Count: {len(ratios)}")
    print(f"Mean Ratio (Target): {mean_ratio:.4f}")
    print(f"Std Dev: {std_ratio:.4f}")
    print(f"Min / Max: {min_ratio:.4f} / {max_ratio:.4f}")
    print("="*50)
    
    print(f"\n💡 [Action] Please update 'target_mean_ratio' in 'pure_alice_spectral_mask.py' to: {mean_ratio:.4f}")

if __name__ == "__main__":
    main()