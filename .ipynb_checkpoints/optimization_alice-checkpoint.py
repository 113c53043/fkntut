import sys
import os
import argparse
import torch
import numpy as np
import hashlib
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import optimization_utils as utils # 載入工具包
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def load_model_from_config(config, ckpt, device):
    print(f"⏳ 載入模型: {ckpt}")
    # 修正 PyTorch 2.6+ 安全性問題
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    parser.add_argument("--verification_path", type=str)
    parser.add_argument("--opt_iters", type=int, default=20) # 預設迭代次數
    
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # === 1. 讀取與加密 (AES) ===
    if not os.path.exists(opt.payload_path):
        print(f"❌ 找不到 Payload: {opt.payload_path}")
        sys.exit(1)

    with open(opt.payload_path, "rb") as f:
        raw_data = f.read()
    
    if opt.verification_path:
        with open(opt.verification_path, "wb") as f: f.write(raw_data)

    # AES Encrypt
    aes_key = hashlib.sha256(str(opt.secret_key).encode()).digest()
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(raw_data, AES.block_size))
    cipher_len = len(encrypted_data)
    
    # === 2. 輕量級編碼 (Repetition Only + Length Header) ===
    # 容量計算: 16384 latent elements / 3 = 5461 bits ≈ 682 bytes
    CAPACITY_BYTES = 680 
    
    # 預留 2 bytes 給長度標頭 (Header)
    MAX_CIPHER_SIZE = CAPACITY_BYTES - 2
    
    if cipher_len > MAX_CIPHER_SIZE:
        print(f"⚠️ [Warning] Data size {cipher_len} > {MAX_CIPHER_SIZE}. Truncating.")
        # 截斷時必須確保是 16 的倍數，否則 AES 無法解
        safe_len = (MAX_CIPHER_SIZE // 16) * 16
        encrypted_data = encrypted_data[:safe_len]
        cipher_len = safe_len
    
    # 加入長度標頭
    length_header = cipher_len.to_bytes(2, 'big')
    final_payload = length_header + encrypted_data

    # Pad with zeros to fill capacity
    if len(final_payload) < CAPACITY_BYTES:
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    
    print(f"[Secure] Payload Ready: {len(final_payload)} bytes (Header+Cipher+Pad)")
    
    # Bits Conversion
    bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
    # Repetition(3)
    rep_bits = np.repeat(bits, 3)
    
    # Map to Latent Space
    target_flat = np.zeros(16384)
    
    # 【參數設定】訊號強度 1.0
    SIGNAL_STRENGTH = 1.5
    
    # 【關鍵修正】強制轉型為 float32，避免 uint8 下溢 (0-1 -> 255)
    # 0 -> -1.0, 1 -> +1.0
    target_flat[:len(rep_bits)] = (rep_bits.astype(np.float32) * 2 - 1) * SIGNAL_STRENGTH
    
    # Fill rest with random noise
    rng = np.random.RandomState(opt.secret_key)
    remaining = 16384 - len(rep_bits)
    if remaining > 0:
        target_flat[len(rep_bits):] = rng.randn(remaining) * 0.5 
    
    # Shuffle
    rng_shuf = np.random.RandomState(opt.secret_key + 999)
    perm = rng_shuf.permutation(16384)
    target_shuffled = target_flat[perm]
    
    z_target = torch.from_numpy(target_shuffled.reshape(1, 4, 64, 64)).float().to(device)

    # === 3. 演算法核心：歸一化反饋修正 (Normalized Feedback Correction) ===
    z_opt = z_target.clone()
    
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""])
    
    # 學習率設定
    LEARNING_RATE = 0.2
    print(f"[Optimizer] 開始優化 (Iterations: {opt.opt_iters}, Signal: {SIGNAL_STRENGTH}, LR: {LEARNING_RATE})...")
    
    for i in range(opt.opt_iters):
        with torch.no_grad(), autocast("cuda"):
            # 1. Forward (Generation)
            z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
            
            # 2. Backward (Inversion)
            z_T_hat, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                        unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                        x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            
        # NaN / Inf Check
        if torch.isnan(z_T_hat).any() or torch.isinf(z_T_hat).any():
            print(f"  ⚠️ Iter {i+1}: 數值異常。重置。")
            z_opt = torch.clamp(z_target.clone(), -1.0, 1.0)
            continue

        # Error Calculation
        error = z_T_hat - z_target
        mse_loss = torch.mean(error ** 2).item()
        
        # Feedback Update
        # 使用帶有截斷的比例更新
        update = LEARNING_RATE * error
        update = torch.clamp(update, -0.5, 0.5)
        
        z_opt = z_opt - update
        
        # 嚴格範圍鉗制
        z_opt = torch.clamp(z_opt, -3.0, 3.0)
        
        print(f"  Iter {i+1}/{opt.opt_iters}: Loss={mse_loss:.4f} (MaxVal={z_T_hat.max().item():.2f})")
        
        if mse_loss < 1e-4:
            print("  ✅ 收斂達成 (Converged)!")
            break

    # === 4. 最終生成 ===
    print("[Secure] 使用最佳化噪聲生成最終圖像...")
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_opt, DPMencode=False, DPMdecode=True)
        x_samples = model.decode_first_stage(z_0_final)
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    
    if np.isnan(x_samples).any():
        print("❌ [Error] 最終圖像包含 NaN，生成失敗！")
        image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    else:
        image = Image.fromarray((x_samples[0] * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(opt.outpath), exist_ok=True)
    image.save(opt.outpath)
    print(f"✅ [Alice] 隱寫完成: {opt.outpath}")

if __name__ == "__main__":
    main()