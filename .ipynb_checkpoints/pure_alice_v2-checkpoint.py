import sys
import os
import argparse
import torch
import torch.nn.functional as F  # 【新增】用於計算 Pooling
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

# === 【核心創新】不確定性 Mask 計算模組 ===
def get_uncertainty_mask(latents, kernel_size=3, min_val=0.4):
    """
    計算 Latent 的局部紋理複雜度作為不確定性遮罩。
    Args:
        latents: (B, 4, 64, 64) 的特徵圖
        kernel_size: 局部窗口大小
        min_val: 遮罩最小值 (0.4 代表平滑區保留 40% 的更新力度)
    Returns:
        mask: 與 latents 同形狀的權重矩陣，範圍 [min_val, 1.0]
    """
    # 1. 計算局部平均值 (Local Mean)
    padding = kernel_size // 2
    avg = F.avg_pool2d(latents, kernel_size=kernel_size, stride=1, padding=padding)
    
    # 2. 計算局部變異數 (Local Variance): Var(X) = E[X^2] - (E[X])^2
    avg_sq = F.avg_pool2d(latents ** 2, kernel_size=kernel_size, stride=1, padding=padding)
    variance = avg_sq - avg ** 2
    
    # 3. 聚合 4 個通道的變異數 (取平均，代表該空間位置的整體複雜度)
    spatial_variance = variance.mean(dim=1, keepdim=True) # (B, 1, 64, 64)
    
    # 4. 歸一化到 0~1
    v_min = spatial_variance.min()
    v_max = spatial_variance.max()
    # 加上 1e-6 避免除以零
    mask = (spatial_variance - v_min) / (v_max - v_min + 1e-6)
    
    # 5. 線性映射範圍到 [min_val, 1.0]
    # 平滑區 (variance小) -> mask 接近 min_val
    # 紋理區 (variance大) -> mask 接近 1.0
    mask = mask * (1 - min_val) + min_val
    
    # 6. 擴展回 4 個通道以便與 latents 相乘
    mask = mask.repeat(1, 4, 1, 1)
    
    return mask.detach() # 遮罩本身不需要計算梯度

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
    
    # 【新增參數】V2 版本特有的 Mask 控制參數
    parser.add_argument("--mask_min_val", type=float, default=0.4, 
                        help="Minimum mask value for smooth regions (0.0-1.0)")
    
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

    # === 1. Data Preparation ===
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
        
    print(f"[Secure] Payload Size: {len(final_payload)} bytes")

    # 保存 Ground Truth Bits 供 Bob 計算 Accuracy
    gt_bits_path = opt.outpath + ".gt_bits.npy"
    np.save(gt_bits_path, np.frombuffer(final_payload, dtype=np.uint8))

    # === 2. Mapping to Latent ===
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

    # === 3. TTLO Algorithm (Adaptive V2) ===
    z_opt = z_target.clone()
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""])
    
    print(f"⚙️  Starting Optimization V2 (Uncertainty Masking: min_val={opt.mask_min_val})...")

    for i in range(opt.opt_iters):
        with torch.no_grad(), autocast("cuda"):
            # Step A: Denoise to get z_0 (Predicted Clean Image)
            # 使用 z_0 計算 Mask 最準確，因為它是圖像內容的直接反映
            z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
            
            # Step B: Re-noise back to z_T_hat (Predicted Noisy Latent)
            z_T_hat, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                        unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                        x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
        
        if torch.isnan(z_T_hat).any():
            z_opt = torch.clamp(z_target.clone(), -1.5, 1.5)
            continue
            
        error = z_T_hat - z_target
        loss = torch.mean(error**2).item()
        
        # === 【關鍵修改 V2】應用不確定性 Mask ===
        # 計算當前預測圖像的紋理複雜度
        uncertainty_mask = get_uncertainty_mask(z_0, min_val=opt.mask_min_val)
        
        # 梯度引導：只在紋理複雜區域 (Mask 高) 保留大的 Error 梯度
        # 在平滑區域 (Mask 低) 抑制 Error，避免過度修改導致噪點
        guided_error = error * uncertainty_mask
        
        update = opt.lr * guided_error
        # ======================================

        update = torch.clamp(update, -0.5, 0.5)
        z_opt = z_opt - update
        z_opt = torch.clamp(z_opt, -3.0, 3.0)
        
        if i % 2 == 0:
            print(f"  [Iter {i}/{opt.opt_iters}] Loss: {loss:.6f} (Mask applied)")

        if loss < 1e-5: break

    # === 4. Generate Final Image ===
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_opt, DPMencode=False, DPMdecode=True)
        x_samples = model.decode_first_stage(z_0_final)
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    if not np.isnan(x_samples.cpu().numpy()).any():
        Image.fromarray((x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)).save(opt.outpath)
        print(f"✅ [Alice V2] Generated: {opt.outpath}")

if __name__ == "__main__":
    main()