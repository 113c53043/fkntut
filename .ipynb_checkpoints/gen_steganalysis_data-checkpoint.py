import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch import autocast
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import hashlib

# === 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
MAS_GRDH_PATH = CURRENT_DIR 
sys.path.append(MAS_GRDH_PATH)

# 配置
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "validation_data2")
COVER_DIR = os.path.join(OUTPUT_ROOT, "cover")
STEGO_DIR = os.path.join(OUTPUT_ROOT, "stego")

# 參數
NUM_SAMPLES = 50 # 我們生成 50 對來測試 (共 100 張)
DPM_STEPS = 20
SCALE = 5.0
OPT_ITERS = 10 # Stego 優化次數
LR = 0.2
SIGNAL = 1.0

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    print("❌ 找不到 ldm 庫")
    sys.exit(1)

def load_model():
    print(f"⏳ Loading Model...")
    config = OmegaConf.load(CONFIG_PATH)
    pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    return model, DPMSolverSampler(model)

def generate_pair(model, sampler, idx, prompt):
    device = torch.device("cuda")
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    
    # 隨機金鑰
    secret_key = int(np.random.randint(10000000, 99999999))
    
    # === 1. 準備 Latent Target (Stego 用) ===
    # 模擬 680 bytes payload
    raw_data = os.urandom(600)
    aes_key = hashlib.sha256(str(secret_key).encode()).digest()
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(raw_data, AES.block_size))
    
    # Header + Padding
    CAPACITY = 680
    if len(encrypted) > CAPACITY-2: encrypted = encrypted[:CAPACITY-2]
    header = len(encrypted).to_bytes(2, 'big')
    payload = header + encrypted
    if len(payload) < CAPACITY: payload += b'\x00' * (CAPACITY - len(payload))
    
    # Bits -> Latent
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    rep_bits = np.repeat(bits, 3)
    target_flat = np.zeros(16384)
    target_flat[:len(rep_bits)] = (rep_bits.astype(np.float32) * 2 - 1) * SIGNAL
    
    # Background Noise
    rng = np.random.RandomState(secret_key)
    if 16384 - len(rep_bits) > 0:
        target_flat[len(rep_bits):] = rng.randn(16384 - len(rep_bits)) * 0.5
        
    rng_shuf = np.random.RandomState(secret_key + 999)
    perm = rng_shuf.permutation(16384)
    z_target = torch.from_numpy(target_flat[perm].reshape(1, 4, 64, 64)).float().to(device)

    # === 2. 生成 Cover (無優化) ===
    # Cover 直接使用 z_target (或是純隨機噪聲，視定義而定)
    # 為了嚴格測試「優化帶來的痕跡」，Cover 應該使用「未經優化的初始噪聲」
    # 這樣兩者的內容(Prompt)和初始分佈是一樣的，唯一的差別是 Stego 經過了梯度下降修正
    
    with torch.no_grad(), autocast("cuda"):
        z_0_cover, _ = sampler.sample(steps=DPM_STEPS, conditioning=c, batch_size=1, shape=(4,64,64),
                                      unconditional_guidance_scale=SCALE, unconditional_conditioning=uc,
                                      x_T=z_target, DPMencode=False, DPMdecode=True, verbose=False)
        x_cover = model.decode_first_stage(z_0_cover)
        
    # === 3. 生成 Stego (TTLO 優化) ===
    z_opt = z_target.clone()
    
    # 執行 TTLO 優化迴圈
    for _ in range(OPT_ITERS):
        with torch.no_grad(), autocast("cuda"):
            z_0, _ = sampler.sample(steps=DPM_STEPS, conditioning=c, batch_size=1, shape=(4,64,64),
                                    unconditional_guidance_scale=SCALE, unconditional_conditioning=uc,
                                    x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
            z_T_hat, _ = sampler.sample(steps=DPM_STEPS, conditioning=c, batch_size=1, shape=(4,64,64),
                                        unconditional_guidance_scale=SCALE, unconditional_conditioning=uc,
                                        x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
        
        if torch.isnan(z_T_hat).any(): 
            z_opt = z_target.clone()
            continue
            
        error = z_T_hat - z_target
        update = 0.1 * error # LR
        update = torch.clamp(update, -0.5, 0.5)
        z_opt = z_opt - update
        z_opt = torch.clamp(z_opt, -3.0, 3.0)

    # 最終生成 Stego
    with torch.no_grad(), autocast("cuda"):
        z_0_stego, _ = sampler.sample(steps=DPM_STEPS, conditioning=c, batch_size=1, shape=(4,64,64),
                                      unconditional_guidance_scale=SCALE, unconditional_conditioning=uc,
                                      x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
        x_stego = model.decode_first_stage(z_0_stego)

    # 儲存
    def save(tensor, path):
        tensor = torch.clamp((tensor + 1.0) / 2.0, min=0.0, max=1.0)
        tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
        Image.fromarray((tensor * 255).astype(np.uint8)).save(path)

    save(x_cover, os.path.join(COVER_DIR, f"{idx:04d}.png"))
    save(x_stego, os.path.join(STEGO_DIR, f"{idx:04d}.png"))
    print(f"Generated Pair #{idx}")

def main():
    os.makedirs(COVER_DIR, exist_ok=True)
    os.makedirs(STEGO_DIR, exist_ok=True)
    
    model, sampler = load_model()
    
    # 簡單的 Prompt 列表
    prompts = [
        "A futuristic city", "A cute cat", "A beautiful landscape", "Cyberpunk street", 
        "Oil painting of flowers", "Portrait of a woman", "Space station", "Forest in morning mist",
        "Vintage car", "Mountain peak"
    ]
    
    for i in range(NUM_SAMPLES):
        p = prompts[i % len(prompts)]
        generate_pair(model, sampler, i, p)
        
    print("✅ Data Generation Complete.")

if __name__ == "__main__":
    main()