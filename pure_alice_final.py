import sys
import os
import torch
import numpy as np
from torch import autocast
from PIL import Image, ImageEnhance
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# 確保引用路徑正確
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from mapping_module import ours_mapping
except ImportError:
    class ours_mapping:
        def __init__(self, bits=1): pass
        def encode_secret(self, secret_message, seed_kernel, seed_shuffle):
            return np.random.randn(1, 4, 64, 64)
        def decode_secret_soft(self, z_rec, seed_kernel, seed_shuffle):
            return np.random.rand(16384)

LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

# 估計不確定性與生成 Mask 的核心函式
def estimate_uncertainty(model, sampler, z_center, c, uc, scale, device, repeats=6, noise_std=0.05, mode="adaptive", mask_power=4.0):
    
    # 1. Baseline & No_Mask 不需要計算 Variance
    if mode == "baseline" or mode == "no_mask":
        return torch.ones_like(z_center), 0.0

    # 2. 計算 Variance (Uniform 和 Adaptive 需power要)
    z_recs = []
    fast_steps = 10 
    with torch.no_grad(), autocast("cuda"):
        for i in range(repeats):
            noise = torch.randn_like(z_center) * noise_std
            z_input = z_center + noise
            z_0, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_input, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            z_recs.append(z_rec)
    
    stack = torch.stack(z_recs)
    variance = torch.var(stack, dim=0)
    variance_mean = torch.mean(variance, dim=1, keepdim=True)
    global_instability = torch.mean(variance_mean).item()

    # 3. 根據模式生成 Mask
    if mode == "uniform":
        # === Uniform Strategy ===
        mask = torch.ones_like(variance_mean) * 0.85
        
    elif mode == "adaptive":
        # === Adaptive Strategy (Ours) ===
        v_min = torch.quantile(variance_mean, 0.01) 
        v_max = torch.quantile(variance_mean, 0.99)
        denom = v_max - v_min
        if denom < 1e-8: denom = 1.0
        norm_var = (variance_mean - v_min) / denom
        norm_var = torch.clamp(norm_var, 0.0, 1.0)

        # Power Curve
        norm_var_powered = torch.pow(norm_var, mask_power) 
        mask = norm_var_powered
        
        # Dynamic Floor
        avg_uncertainty = torch.mean(norm_var).item()
        # base_floor = 0.40 + (0.3 * avg_uncertainty)
        # base_floor = min(max(base_floor, 0.40), 0.70)
        # mask = mask * (1.0 - base_floor) + base_floor
        base_floor = 0.15 + (0.2 * avg_uncertainty)
        base_floor = min(max(base_floor, 0.15), 0.60) # 限制在 0.15 ~ 0.5 之間
        mask = mask * (1.0 - base_floor) + base_floor
    
    else:
        # Default Fallback
        mask = torch.ones_like(variance_mean)

    return mask.repeat(1, 4, 1, 1), global_instability

def apply_refinement(pil_image):
    # enhancer = ImageEnhance.Sharpness(pil_image)
    # pil_image = enhancer.enhance(1.05) 
    # enhancer = ImageEnhance.Contrast(pil_image)
    # pil_image = enhancer.enhance(1.02) 
    return pil_image

def generate_alice_image(model, sampler, prompt, secret_key, payload_data, outpath, init_latent_path=None, 
                         opt_iters=15, lr=0.12, lambda_reg=1.25, mode="adaptive", 
                         dpm_steps=20, scale=5.0, device="cuda",
                         return_loss_history=False, return_latent=False,
                         early_stop_threshold=0.0693, 
                         min_iters=5, check_interval=2,
                         mask_power=4.0):
    
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

    # === MODE 1: BASELINE (Pure txt2img) ===
    if mode == "baseline":
        opt_iters = 0 
    
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])

    global_instability = 0.0
    
    if opt_iters > 0:
        uncertainty_mask, global_instability = estimate_uncertainty(model, sampler, z_target, c, uc, scale, device, mode=mode, mask_power=mask_power)
    else:
        uncertainty_mask = None 

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    z_best = z_target.clone() 
    min_loss = float('inf')
    initial_lr = lr
    loss_history = [] 

    early_stop_triggered = False

    for i in range(opt_iters + 1):
        if opt_iters == 0: break 

        progress = i / (opt_iters + 1)
        decay_factor = 1.0 - (0.5 * progress) 
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
        
        if opt_iters > 0 and i >= min_iters:
            if i % check_interval == 0:
                current_recon = recon_loss.item()
                if current_recon < early_stop_threshold:
                    z_best = z_eval.clone()
                    early_stop_triggered = True

        loss_val = loss.item()
        loss_history.append({'iter': i, 'loss': loss_val, 'recon': recon_loss.item()})

        if early_stop_triggered: break
        if i == opt_iters: break

        grad_recon = diff 
        grad_reg = 2.0 * (z_eval - z_target)
        total_gradient = grad_recon + lambda_reg * grad_reg
        
        guided_gradient = total_gradient * uncertainty_mask
        
        scale_factor = 1.0
        if mode == "adaptive":
            avg_mask = torch.mean(uncertainty_mask).item() + 1e-6
            
            # ✅ 正確寫法：反向補償
            # 邏輯：如果 Mask 很小 (例如 0.3)，我們希望 Scale 大一點 (例如 1.5)
            # 使用溫和的線性補償公式：
            base_scale = 1.0 + (1.0 - avg_mask) 
            
            # 設定上限 (避免過大，最多 2.0 倍)
            base_scale = min(base_scale, 2.0)
            
            confidence_boost = 1.3 - (global_instability * 6.0)
            confidence_boost = min(max(confidence_boost, 1.0), 1.35) 
            scale_factor = base_scale * confidence_boost
            guided_gradient = guided_gradient * scale_factor

        update = current_lr * guided_gradient
        update = torch.clamp(update, -0.1, 0.1)
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)
    
    if mode != "baseline" and not early_stop_triggered:
        z_best = z_opt.clone()
    
    if mode == "baseline":
        z_best = z_target

    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    final_img = apply_refinement(pil_img)
    final_img = apply_refinement(pil_img)
    if outpath is not None:
        final_img.save(outpath)
    else:
        # 在進行效率測試時，不存檔可以獲得更準確的演算法耗時數據
        pass

    did_early_stop = (len(loss_history) < opt_iters + 1)
    stop_step = len(loss_history) - 1
    return True, did_early_stop, stop_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 支援 4 種模式
    parser.add_argument("--mode", type=str, default="adaptive", choices=["baseline", "no_mask", "uniform", "adaptive"])
    # 支援 mask_power (給 run_power_search 用)
    parser.add_argument("--mask_power", type=float, default=6.0)
    
    # ... (其他參數省略，args 解析會自動處理)
    args, unknown = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    try:
        pl_sd = torch.load(args.ckpt, map_location="cpu")
    except:
        pl_sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    sampler = DPMSolverSampler(model)

    # Sync Payload for Script Execution
    np.random.seed(args.secret_key)
    bits_seed = np.random.randint(0, 2, 16384).astype(np.uint8)
    payload_data_synced = np.packbits(bits_seed).tobytes()
        
    generate_alice_image(
        model=model,
        sampler=sampler,
        prompt=args.prompt,
        secret_key=args.secret_key,
        payload_data=payload_data_synced, 
        outpath=args.outpath,
        init_latent_path=None,
        opt_iters=args.opt_iters,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        mode=args.mode,
        dpm_steps=args.dpm_steps,
        early_stop_threshold=0.0693
    )