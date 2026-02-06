import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from torch import autocast
import json
import matplotlib.pyplot as plt

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 引用依賴
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
except ImportError as e:
    print(f"❌ Import Warning: {e}")
    sys.exit(1)

from pure_alice_final import estimate_uncertainty, apply_refinement, LONG_NEGATIVE_PROMPT

MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "calibration_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 校準配置 ===
CALIBRATION_SAMPLES = 100 
MAX_ITERS = 15 

# [修改] 多目標設定列表 (從小到大排序)
# 0.900: 基礎可用性
# 0.980: 高品質
# 0.985: ECC 標準
# 0.995: 完美主義
TARGET_ACCS = [0.98, 0.985, 0.99,0.995]

# SOTA 參數
LR = 0.12
REG = 1.25

# === 模型載入 ===
def load_model():
    print(f"⏳ Loading SD Model for Comparative Calibration...")
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

# === Bob 檢查器 ===
def check_bob_accuracy(model, sampler, z_current, prompt, secret_key, gt_bits):
    try:
        with torch.no_grad(), autocast("cuda"):
            x_samples = model.decode_first_stage(z_current)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples = x_samples * 255
            x_samples = x_samples.round() / 255.0
            img_tensor = x_samples * 2.0 - 1.0 

        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
        
        with torch.no_grad(), autocast("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
            z_rec, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=init_latent, DPMencode=True, DPMdecode=False, verbose=False)
        
        mapper = ours_mapping(bits=1)
        decoded = mapper.decode_secret_soft(z_rec.cpu().numpy(), seed_kernel=secret_key, seed_shuffle=secret_key+999)
        bits_hat = np.round(decoded).astype(np.uint8).flatten()
        
        bits_gt = np.unpackbits(np.frombuffer(gt_bits, dtype=np.uint8))
        min_len = min(len(bits_hat), len(bits_gt))
        matches = np.sum(bits_hat[:min_len] == bits_gt[:min_len])
        return matches / min_len

    except Exception:
        return 0.0

# === 核心迴圈：同時監控多個目標 ===
def run_comparative_calibration(model, sampler):
    print(f"\n🔬 STARTING MULTI-TARGET CALIBRATION (N={CALIBRATION_SAMPLES})")
    print(f"   Targets: {[f'{t*100}%' for t in TARGET_ACCS]}")
    
    payload = os.urandom(16384 // 8)
    CAPACITY = 16384 // 8
    if len(payload) < CAPACITY: payload += b'\x00' * (CAPACITY - len(payload))
    
    base_prompts = [
        "A futuristic city", "A photo of a dog", "Abstract painting", 
        "A white wall", "Complex forest texture", "Portrait of a woman"
    ]
    prompts = base_prompts * (CALIBRATION_SAMPLES // len(base_prompts) + 1)
    prompts = prompts[:CALIBRATION_SAMPLES]

    # 資料收集結構初始化
    # data[0.90] = {"steps": [], "losses": []}
    data = {t: {"steps": [], "losses": []} for t in TARGET_ACCS}

    pbar = tqdm(range(CALIBRATION_SAMPLES), desc="Calibrating")
    
    for i in pbar:
        seed = 5000 + i
        prompt = prompts[i]
        
        # Setup Alice
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        bits = bits[:16384].reshape(1, 4, 64, 64)
        mapper = ours_mapping(bits=1)
        z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=seed, seed_shuffle=seed+999)
        z_target = torch.from_numpy(z_target_numpy).float().to("cuda")
        
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
        uncertainty_mask = estimate_uncertainty(model, sampler, z_target, c, uc, 5.0, "cuda", mode="adaptive")
        
        z_opt = z_target.clone()
        z_opt.requires_grad = False
        
        # 追蹤每個目標是否已達成
        found_flags = {t: False for t in TARGET_ACCS}
        
        current_lr = LR 

        # Optimization Loop
        for step in range(MAX_ITERS):
            # 1. Update
            with torch.no_grad(), autocast("cuda"):
                optim_steps = 8
                z_0, _ = sampler.sample(steps=optim_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                        unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                        x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
                z_rec, _ = sampler.sample(steps=optim_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                          unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                          x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            
            diff = (z_rec - z_target).float()
            recon_loss = torch.mean(diff**2)
            loss_val = recon_loss.item()

            reg_grad = 2.0 * (z_opt - z_target)
            total_grad = diff + REG * reg_grad
            guided_grad = total_grad * uncertainty_mask
            
            progress = step / (MAX_ITERS + 1)
            lr_t = LR * (1.0 - (0.5 * progress))
            z_opt = torch.clamp(z_opt - lr_t * guided_grad, -4.0, 4.0)
            
            # 2. Check Accuracy
            # 如果還有任何目標未達成，就檢查
            if not all(found_flags.values()):
                acc = check_bob_accuracy(model, sampler, z_opt, prompt, seed, payload)
                
                # 檢查每個目標
                for target in TARGET_ACCS:
                    if acc >= target and not found_flags[target]:
                        data[target]["steps"].append(step + 1)
                        data[target]["losses"].append(loss_val)
                        found_flags[target] = True
            
            # 如果所有目標都達成了，這張圖提早結束
            if all(found_flags.values()):
                break
        
        # Update Pbar info (show count for highest target)
        highest_target = TARGET_ACCS[-1]
        pbar.set_postfix({f"Found_{highest_target}": len(data[highest_target]["steps"])})

    return data

def analyze_and_plot(data):
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS REPORT")
    print("-" * 80)
    print(f"{'Target Acc':<12} | {'Success Rate':<15} | {'Avg Steps':<10} | {'Threshold (15%)':<15} | {'Marginal Cost'}")
    print("-" * 80)

    prev_steps = 0.0
    
    results_json = {}

    for target in TARGET_ACCS:
        steps = data[target]["steps"]
        losses = data[target]["losses"]
        
        success_rate = f"{len(steps)}/{CALIBRATION_SAMPLES}"
        
        if len(steps) > 0:
            avg_steps = np.mean(steps)
            # Threshold (15th percentile for 85% safety coverage)
            threshold = np.percentile(losses, 15)
            
            # Calculate Marginal Cost (Extra steps from previous target)
            if prev_steps == 0.0:
                cost_str = "Baseline"
            else:
                cost = avg_steps - prev_steps
                cost_str = f"+{cost:.2f} steps"
            
            prev_steps = avg_steps
            
            print(f"{target*100:>5.1f}%      | {success_rate:<15} | {avg_steps:<10.2f} | {threshold:<15.5f} | {cost_str}")
            
            results_json[str(target)] = {
                "success_count": len(steps),
                "avg_steps": avg_steps,
                "recommended_threshold": threshold,
                "losses": losses,
                "steps_dist": [int(x) for x in steps]
            }
        else:
            print(f"{target*100:>5.1f}%      | {success_rate:<15} | {'N/A':<10} | {'N/A':<15} | N/A")

    print("="*80)

    # 決策建議邏輯
    print("💡 DECISION GUIDE:")
    # 找到最高且成功率 > 80% 的目標
    valid_targets = [t for t in TARGET_ACCS if len(data[t]["steps"]) >= 0.8 * CALIBRATION_SAMPLES]
    
    if valid_targets:
        best_t = valid_targets[-1] # 最高達成率的目標
        best_steps = np.mean(data[best_t]["steps"])
        best_thresh = np.percentile(data[best_t]["losses"], 15)
        
        print(f"👉 Based on success rate (>80%), max feasible accuracy is {best_t*100}%.")
        print(f"   Avg Steps: {best_steps:.2f}")
        print(f"   Recommended EARLY_STOP_THRESHOLD: {best_thresh:.4f}")
        
        # 檢查是否值得追求 0.995 (如果它不是 best_t)
        if 0.995 in data and 0.995 != best_t and len(data[0.995]["steps"]) > 0:
            cost_995 = np.mean(data[0.995]["steps"]) - best_steps
            print(f"   (Note: Pushing to 99.5% would cost an extra {cost_995:.2f} steps)")
    else:
        print("⚠️ No target achieved > 80% success rate. Consider increasing MAX_ITERS or LR.")

    # Save to JSON
    with open(os.path.join(OUTPUT_DIR, "multi_target_stats.json"), "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"\n💾 Full stats saved to {OUTPUT_DIR}/multi_target_stats.json")

def main():
    model = load_model()
    sampler = DPMSolverSampler(model)
    data = run_comparative_calibration(model, sampler)
    analyze_and_plot(data)

if __name__ == "__main__":
    main()