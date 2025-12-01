import sys
import os
import argparse
import torch
import torch.nn.functional as F
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
except ImportError:
    sys.exit(1)

# === 數值診斷函數 ===
def check_stat(tensor, name):
    """打印 Tensor 的統計資訊"""
    if tensor is None:
        print(f"🕵️ [Debug] {name}: None")
        return False
    
    t = tensor.float()
    has_nan = torch.isnan(t).any().item()
    
    min_val = t.min().item()
    max_val = t.max().item()
    mean_val = t.mean().item()

    status = "✅ OK"
    if has_nan: status = "❌ NaN DETECTED!"
    elif abs(max_val) > 10.0 or abs(min_val) > 10.0: status = "⚠️ LARGE VALUE"
    elif abs(mean_val) > 0.1: status = "⚠️ DRIFT DETECTED" # 監測漂移

    print(f"🕵️ [Debug] {name:20s} | Range: [{min_val:.4f}, {max_val:.4f}] | Mean: {mean_val:.4f} | {status}")
    return has_nan

def get_uncertainty_mask(latents, kernel_size=3, min_val=0.4):
    # FP32 計算 mask
    with torch.no_grad():
        latents_f = latents.float()
        padding = kernel_size // 2
        avg = F.avg_pool2d(latents_f, kernel_size=kernel_size, stride=1, padding=padding)
        avg_sq = F.avg_pool2d(latents_f ** 2, kernel_size=kernel_size, stride=1, padding=padding)
        variance = avg_sq - avg ** 2
        spatial_variance = variance.mean(dim=1, keepdim=True)
        
        v_min = spatial_variance.min()
        v_max = spatial_variance.max()
        denom = v_max - v_min
        if denom < 1e-6: denom = 1e-6
        
        mask = (spatial_variance - v_min) / denom
        mask = mask * (1 - min_val) + min_val
        mask = mask.repeat(1, 4, 1, 1)
        
    return mask.to(latents.dtype)

def load_model_from_config(config, ckpt, device):
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
    parser.add_argument("--opt_iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--signal_strength", type=float, default=1.5)
    parser.add_argument("--mask_min_val", type=float, default=0.4)
    
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

    # 1. Payload
    with open(opt.payload_path, "rb") as f: raw_data = f.read()
    aes_key = hashlib.sha256(str(opt.secret_key).encode()).digest()
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(raw_data, AES.block_size))
    
    CAPACITY = 680
    if len(encrypted_data) > CAPACITY - 2:
        encrypted_data = encrypted_data[:CAPACITY-2]
        encrypted_data = encrypted_data[:(len(encrypted_data)//16)*16]
    
    length_header = len(encrypted_data).to_bytes(2, 'big')
    final_payload = length_header + encrypted_data
    if len(final_payload) < CAPACITY:
        final_payload += b'\x00' * (CAPACITY - len(final_payload))
        
    np.save(opt.outpath + ".gt_bits.npy", np.frombuffer(final_payload, dtype=np.uint8))

    # 2. Mapping
    bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
    rep_bits = np.repeat(bits, 3)
    target_flat = np.zeros(16384)
    target_flat[:len(rep_bits)] = (rep_bits.astype(np.float32) * 2 - 1) * opt.signal_strength
    
    rng = np.random.RandomState(opt.secret_key)
    if 16384 - len(rep_bits) > 0:
        target_flat[len(rep_bits):] = rng.randn(16384 - len(rep_bits)) * 0.5
        
    rng_shuf = np.random.RandomState(opt.secret_key + 999)
    perm = rng_shuf.permutation(16384)
    z_target = torch.from_numpy(target_flat[perm].reshape(1, 4, 64, 64)).float().to(device)

    # 3. Optimization Loop
    z_opt = z_target.clone()
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""])
    
    print(f"\n🩺 STARTING DIAGNOSTIC RUN (With Zero-Mean & Clamp)")
    
    # 強制使用 FP32 進行優化
    z_target = z_target.float()
    z_opt = z_opt.float()
    check_stat(z_target, "Init Target")

    for i in range(opt.opt_iters):
        print(f"\n--- Iteration {i+1} ---")
        
        # Step A & B
        with torch.cuda.amp.autocast(enabled=True):
            z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
            z_T_hat, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                        unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                        x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
        
        if check_stat(z_T_hat, f"z_T_hat (Iter {i})"): break
            
        error = z_T_hat.float() - z_target
        loss = torch.mean(error**2).item()
        
        uncertainty_mask = get_uncertainty_mask(z_0, min_val=opt.mask_min_val).float()
        guided_error = error * uncertainty_mask
        
        update = opt.lr * guided_error
        
        # 【診斷點 1】檢查修正前的 Update
        # check_stat(update, "Update (Raw)") 
        
        # === 【關鍵修正】歸零校正 ===
        # 計算每個 sample 的平均偏移量 (dim=1,2,3)
        drift = update.mean(dim=(1,2,3), keepdim=True)
        update = update - drift
        
        # 【診斷點 2】檢查修正後的 Update (Mean 應該要是 0)
        check_stat(update, "Update (Centered)")
        
        update = torch.clamp(update, -0.5, 0.5)
        z_opt = z_opt - update
        
        # 優化過程中的寬鬆鉗制
        z_opt = torch.clamp(z_opt, -3.0, 3.0)
        
        print(f"   Loss: {loss:.6f}")
        if check_stat(z_opt, f"z_opt (Iter {i})"): break

    # 4. Final Decode
    print(f"\n🎨 Final Decoding with Safety Clamp...")
    with torch.cuda.amp.autocast(enabled=True):
        z_0_final, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_opt, DPMencode=False, DPMdecode=True)
    
    with torch.no_grad():
        # 【診斷點 3】鉗制前的範圍
        check_stat(z_0_final, "Final Latent (Raw)")
        
        # === 【關鍵修正】VAE 安全鉗制 ===
        z_0_safe = torch.clamp(z_0_final, -2.5, 2.5)
        
        # 【診斷點 4】鉗制後的範圍
        check_stat(z_0_safe, "Final Latent (Safe)")
        
        x_samples = model.decode_first_stage(z_0_safe.float())
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)).save(opt.outpath)
    print(f"✅ Debug Image Saved: {opt.outpath}")

if __name__ == "__main__":
    main()