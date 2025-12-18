import sys
import os
import argparse
import torch
import torch.fft
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image

# === 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from mapping_module import ours_mapping
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    # 導入同步模組
    from synchronization import SyncModule 
except ImportError:
    pass 

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}...")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def get_adaptive_base_weight(z_latent):
    """
    [CVPR 核心] 計算 Latent 的頻譜特徵並動態映射到 Base Weight。
    【校準最終版】基於 COCO 1000張 校準數據 (Mean=0.3298)
    """
    freq_domain = torch.fft.fftn(z_latent, dim=(-2, -1))
    energy = torch.abs(freq_domain)
    
    h, w = energy.shape[-2:]
    h_center, w_center = h // 2, w // 2
    r_h, r_w = h // 4, w // 4 
    
    total_energy = torch.sum(energy)
    energy_shifted = torch.fft.fftshift(energy, dim=(-2, -1))
    low_freq_energy = torch.sum(energy_shifted[..., h_center-r_h:h_center+r_h, w_center-r_w:w_center+r_w])
    
    low_freq_ratio = low_freq_energy / (total_energy + 1e-8)
    ratio_val = low_freq_ratio.item()
    
    # 動態映射邏輯 (Distilled & Dampened)
    target_mean_ratio = 0.3298  
    target_base_weight = 0.30   
    sensitivity = 0.5           
    
    base_weight = target_base_weight + (ratio_val - target_mean_ratio) * sensitivity
    base_weight = np.clip(base_weight, 0.25, 0.35) 
    
    # print(f"📊 [Spectrum] Ratio: {ratio_val:.4f} -> Base: {base_weight:.4f}")
    return base_weight

def estimate_uncertainty(model, sampler, z_center, c, uc, scale, device, repeats=4, noise_std=0.05, use_adaptive=True):
    z_recs = []
    fast_steps = 10 
    z_clean_for_analysis = None
    
    with torch.no_grad(), autocast("cuda"):
        for i in range(repeats):
            noise = torch.randn_like(z_center) * noise_std
            z_input = z_center + noise
            
            z_0, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_input, DPMencode=False, DPMdecode=True, verbose=False)
            
            if i == 0:
                z_clean_for_analysis = z_0.clone()

            z_rec, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            z_recs.append(z_rec)
    
    stack = torch.stack(z_recs)
    variance = torch.var(stack, dim=0)
    variance_mean = torch.mean(variance, dim=1, keepdim=True) 
    v_min = variance_mean.min()
    v_max = variance_mean.max()
    norm_var = (variance_mean - v_min) / (v_max - v_min + 1e-8)
    
    mask = 1.0 - norm_var
    mask = torch.pow(mask, 2)
    
    if use_adaptive:
        base_weight = get_adaptive_base_weight(z_clean_for_analysis)
    else:
        base_weight = 0.3
    
    mask = mask * (1.0 - base_weight) + base_weight
    
    return mask.repeat(1, 4, 1, 1)

def generate_alice_image(model, sampler, prompt, secret_key, payload_data, outpath, init_latent_path=None, 
                         opt_iters=10, lr=0.05, lambda_reg=1.5, use_uncertainty=True, 
                         dpm_steps=20, scale=5.0, device="cuda", use_adaptive=True):
    
    # 1. Prepare Latent
    if init_latent_path and os.path.exists(init_latent_path):
        z_target = torch.load(init_latent_path, map_location=device)
    else:
        CAPACITY_BYTES = 16384 // 8 
        bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
        if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
        bits = bits[:16384].reshape(1, 4, 64, 64)
        mapper = ours_mapping(bits=1)
        z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
        z_target = torch.from_numpy(z_target_numpy).float().to(device)

    # 2. Setup Optimization
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([negative_prompt])

    if use_uncertainty:
        uncertainty_mask = estimate_uncertainty(model, sampler, z_target, c, uc, scale, device, use_adaptive=use_adaptive)
    else:
        uncertainty_mask = torch.ones_like(z_target)

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    z_best = z_target.clone() 
    min_loss = float('inf')
    
    initial_lr = lr

    # 3. Loop
    for i in range(opt_iters + 1):
        progress = i / (opt_iters + 1)
        decay_factor = 1.0 - (0.8 * progress) 
        current_lr = initial_lr * decay_factor

        z_eval = z_target if i == 0 else z_opt

        with torch.no_grad(), autocast("cuda"):
            z_0, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_eval, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
    
        diff = (z_rec - z_target).float()
        recon_loss = torch.mean(diff**2)
        reg_loss = torch.mean((z_eval - z_target)**2) if i > 0 else torch.tensor(0.0).to(device)
        loss = recon_loss + lambda_reg * reg_loss
        
        if loss < min_loss:
            min_loss = loss
            z_best = z_eval.clone()
        
        if i == opt_iters: break

        grad_recon = diff 
        grad_reg = 2.0 * (z_eval - z_target)
        total_gradient = grad_recon + lambda_reg * grad_reg
        guided_gradient = total_gradient * uncertainty_mask
        
        update = torch.clamp(current_lr * guided_gradient, -0.1, 0.1)
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)

    # 4. Final Decode
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    # === [新增] 添加幾何同步標記 ===
    # 這一步在轉為 Numpy 之前執行，為圖片四角加上微小的同心圓
    try:
        sync_mod = SyncModule(shape=(512, 512))
        x_samples = sync_mod.add_markers(x_samples)
    except Exception as e:
        print(f"Warning: Sync markers skipped due to error: {e}")
    # =============================
    
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    pil_img.save(outpath)

def run_alice():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    parser.add_argument("--init_latent", type=str, default=None)
    parser.add_argument("--opt_iters", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=0.05) 
    parser.add_argument("--lambda_reg", type=float, default=1.5) 
    parser.add_argument("--use_uncertainty", action="store_true")
    
    parser.add_argument("--strategy", type=str, default="adaptive", choices=["adaptive", "fixed"], 
                        help="Choose masking strategy")

    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    with open(opt.payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data 
    CAPACITY_BYTES = 16384 // 8 
    if len(payload_data) > CAPACITY_BYTES - 2: payload_data = payload_data[:CAPACITY_BYTES-2]
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    
    # 存 GT
    np.save(opt.outpath + ".gt_bits.npy", np.frombuffer(final_payload, dtype=np.uint8))

    use_adaptive_flag = (opt.strategy == "adaptive")

    generate_alice_image(
        model=model,
        sampler=sampler,
        prompt=opt.prompt,
        secret_key=opt.secret_key,
        payload_data=final_payload,
        outpath=opt.outpath,
        init_latent_path=opt.init_latent,
        opt_iters=opt.opt_iters,
        lr=opt.lr,
        lambda_reg=opt.lambda_reg,
        use_uncertainty=opt.use_uncertainty,
        dpm_steps=opt.dpm_steps,
        scale=opt.scale,
        device=opt.device,
        use_adaptive=use_adaptive_flag 
    )

if __name__ == "__main__":
    try: run_alice()
    except Exception: sys.exit(1)