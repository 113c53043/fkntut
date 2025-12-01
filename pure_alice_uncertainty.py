import sys
import os
import argparse
import torch
import numpy as np
import traceback
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image

# === Path Setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

# 確保引用你的 mapping_module
try:
    from mapping_module import ours_mapping
except ImportError:
    print("❌ Critical: mapping_module not found. Check PYTHONPATH.")
    sys.exit(1)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    print("❌ Critical: ldm module not found. Check PYTHONPATH.")
    sys.exit(1)

def recursive_fix_config(conf):
    if isinstance(conf, (dict, OmegaConf)):
        for key in conf.keys():
            if key == "image_size" and conf[key] == 32:
                conf[key] = 64
            recursive_fix_config(conf[key])

def load_model_from_config(config, ckpt, device):
    recursive_fix_config(config.model)
    print(f"Loading model from {ckpt}...")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False) 
    except TypeError:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
        
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

# === [核心創新] Uncertainty Estimation Function ===
def estimate_uncertainty(model, sampler, z_center, c, uc, scale, device, repeats=4, noise_std=0.05):
    """
    透過蒙地卡羅擾動 (Monte-Carlo Perturbation) 估計潛在空間的不確定性。
    Returns:
        mask (torch.Tensor): 穩定度遮罩，值域 [0, 1]。
                             1.0 代表高度穩定 (適合隱寫)
                             0.0 代表高度不確定 (避免修改)
    """
    print(f"🔍 [Uncertainty] Estimating latent stability via Monte-Carlo (Repeats={repeats})...")
    
    z_recs = []
    # 使用較少的步數來快速估計，節省時間 (例如 10 步)
    fast_steps = 10 
    
    with torch.no_grad(), autocast("cuda"):
        for i in range(repeats):
            # 1. 加入微小擾動 (Perturbation)
            noise = torch.randn_like(z_center) * noise_std
            z_input = z_center + noise
            
            # 2. 快速生成 (Forward)
            z_0, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_input, DPMencode=False, DPMdecode=True, verbose=False)
            
            # 3. 快速反演 (Inversion)
            z_rec, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            z_recs.append(z_rec)

    # 4. 計算變異數 (Variance)
    stack = torch.stack(z_recs) # [Repeats, 1, 4, 64, 64]
    variance = torch.var(stack, dim=0) # [1, 4, 64, 64]
    
    # 5. 將變異數轉為穩定度 Mask (Normalization)
    # 取 Channel 的平均值來做 Spatial Mask，確保結構一致
    variance_mean = torch.mean(variance, dim=1, keepdim=True) 
    
    v_min = variance_mean.min()
    v_max = variance_mean.max()
    
    # Normalize to [0, 1] -> 0 是低變異，1 是高變異
    norm_var = (variance_mean - v_min) / (v_max - v_min + 1e-8)
    
    # Invert: 低變異數 = 高穩定度 = Mask 接近 1
    mask = 1.0 - norm_var
    
    # Sharpening: 讓好壞區域分界更明顯 (Power function)
    # 平方會讓接近 1 的保持 1，接近 0 的更接近 0，拉開差距
    mask = torch.pow(mask, 2)
    
    # 擴展回 4 個 Channel 以便與 Latent 相乘
    mask = mask.repeat(1, 4, 1, 1)
    
    return mask

def main():
    try:
        run_alice()
    except Exception:
        print("\n❌ Alice CRASHED with the following error:")
        traceback.print_exc()
        sys.exit(1)

def run_alice():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    
    # 優化參數
    parser.add_argument("--opt_iters", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=0.25) 
    parser.add_argument("--noise_std", type=float, default=0.0)
    
    # === [新增參數] ===
    parser.add_argument("--use_uncertainty", action="store_true", help="Enable Uncertainty-Guided Optimization")
    # 新增 lambda_reg 參數，預設為 0.05
    parser.add_argument("--lambda_reg", type=float, default=0.05, help="Regularization strength to preserve image quality")
    
    # 模型路徑
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

    # 1. Payload Preparation
    with open(opt.payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data 

    CAPACITY_BYTES = 16384 // 8 
    if len(payload_data) > CAPACITY_BYTES - 2:
        payload_data = payload_data[:CAPACITY_BYTES-2]
    
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + payload_data
    if len(final_payload) < CAPACITY_BYTES:
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
        
    print(f"[Info] Payload Size: {len(final_payload)} bytes")
    gt_bits_path = opt.outpath + ".gt_bits.npy"
    np.save(gt_bits_path, np.frombuffer(final_payload, dtype=np.uint8))

    # 2. Initialization (Orthogonal Mapping)
    bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
    if len(bits) < 16384:
        bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
    bits = bits[:16384].reshape(1, 4, 64, 64)
    
    mapper = ours_mapping(bits=1)
    print("⚙️  [Method] Generating Target Latent via Orthogonal Mapping...")
    
    z_target_numpy = mapper.encode_secret(
        secret_message=bits, 
        seed_kernel=opt.secret_key, 
        seed_shuffle=opt.secret_key + 999
    )
    z_target = torch.from_numpy(z_target_numpy).float().to(device)

    # === [Step 3] Uncertainty Map Generation ===
    #negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    
    
    negative_prompt ="worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([negative_prompt])

    if opt.use_uncertainty:
        # 計算遮罩
        uncertainty_mask = estimate_uncertainty(model, sampler, z_target, c, uc, opt.scale, device)
        
        # 儲存遮罩以便論文可視化 (Visualization)
        mask_vis = uncertainty_mask[0].mean(dim=0).cpu().numpy()
        mask_vis = (mask_vis * 255).astype(np.uint8)
        mask_save_path = opt.outpath + ".uncertainty_mask.png"
        Image.fromarray(mask_vis).save(mask_save_path)
        print(f"✅ Uncertainty Mask saved to {mask_save_path}")
    else:
        print("⚠️ Uncertainty Guidance is DISABLED. Using uniform mask.")
        uncertainty_mask = torch.ones_like(z_target)

    # 4. Optimization Loop
    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    
    z_best = None
    min_loss = float('inf')
    best_iter = -1

    print(f"✅ Starting Optimization (Max Iters={opt.opt_iters}, Lambda_Reg={opt.lambda_reg})...")
    
    current_lr = opt.lr
    current_noise = opt.noise_std

    for i in range(opt.opt_iters + 1):
        if i == 0:
            z_eval = z_target
            prefix = "Base  "
        else:
            z_eval = z_opt
            prefix = f"Iter {i:<2}"

        with torch.no_grad(), autocast("cuda"):
            # A. Forward
            z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=z_eval, DPMencode=False, DPMdecode=True, verbose=False)
            
            # B. Simulated Noise
            if current_noise > 0 and i > 0:
                noise = torch.randn_like(z_0) * current_noise
                z_0_input = z_0 + noise
            else:
                z_0_input = z_0

            # C. Inversion
            z_rec, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_0_input, DPMencode=True, DPMdecode=False, verbose=False)
        
        # D. Loss Calculation (Modified with Regularization)
        # 1. Reconstruction Loss (原始的 Loss)
        diff = (z_rec - z_target).float()
        recon_loss = torch.mean(diff**2)
        
        # 2. Regularization Loss (新的 Loss: 懲罰 z_eval 偏離 z_target 太遠)
        # 只有在非 Baseline (i>0) 時才計算，Baseline 的 reg_loss 為 0
        if i > 0:
            reg_loss = torch.mean((z_eval - z_target)**2)
        else:
            reg_loss = torch.tensor(0.0).to(device)

        # 3. Total Loss
        loss = recon_loss + opt.lambda_reg * reg_loss
        loss_val = loss.item()
        
        # E. Update Best Model
        is_best = False
        # 使用 Total Loss 來判斷是否更好
        if loss_val < min_loss:
            min_loss = loss_val
            z_best = z_eval.clone()
            best_iter = i
            is_best = True
            improved_msg = "✅ (Best)"
        else:
            improved_msg = ""
            if i > 0: current_lr *= 0.98

        # 打印詳細的 Loss 資訊
        print(f"  {prefix} | Total Loss: {loss_val:.6f} (Recon: {recon_loss.item():.6f}, Reg: {reg_loss.item():.6f}) {improved_msg}")

        if loss_val < 1e-6:
            print("  -> Perfect convergence.")
            break
            
        if i == opt.opt_iters:
            break

        # F. Gradient Update with UNCERTAINTY GUIDANCE & REGULARIZATION
        # 我們需要的梯度是 d(Total_Loss)/d(z_eval)
        # d(Recon_Loss)/d(z_eval) ~= diff (這是一個簡化近似，但對此類問題有效)
        # d(Reg_Loss)/d(z_eval) = 2 * (z_eval - z_target)
        
        # 1. Recon Gradient
        grad_recon = diff
        
        # 2. Reg Gradient
        grad_reg = 2.0 * (z_eval - z_target)
        
        # 3. Total Gradient
        total_gradient = grad_recon + opt.lambda_reg * grad_reg

        # [Critical Step] Apply Mask
        guided_gradient = total_gradient * uncertainty_mask 
        
        update = torch.clamp(current_lr * guided_gradient, -0.1, 0.1)
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)

    # 5. Final Generation
    print(f"🏆 Final Selection: Iter {best_iter} with Total Loss {min_loss:.6f}")
    
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True)
        x_samples = model.decode_first_stage(z_0_final)
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    if not np.isnan(x_samples.cpu().numpy()).any():
        Image.fromarray((x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)).save(opt.outpath)
        print(f"✅ [Alice] Generated Uncertainty-Optimized Stego Image: {opt.outpath}")
    else:
        print("❌ Error: Generated image contains NaN")

if __name__ == "__main__":
    run_alice()