import os
import sys
import torch
import numpy as np
import shutil
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from torch import autocast
from omegaconf import OmegaConf
import json

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 引用模組
try:
    from pytorch_fid import fid_score
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
except ImportError as e:
    print(f"❌ Import Warning: {e}")

try:
    from robust_eval import jpeg
    from utils import load_512
except ImportError:
    pass

MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "prompt_impact_full")
DIR_REAL_COCO = os.path.join(MAS_GRDH_PATH, "scripts", "coco_val2017") 
DIR_REAL_RESIZED = os.path.join(OUTPUT_DIR, "real_resized")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 實驗配置 ===
TOTAL_SAMPLES = 20 # N=100
OPT_ITERS = 15

# 負向提示詞定義
LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"
EMPTY_NEGATIVE_PROMPT = ""

# === 定義 8 種測試情境 ===
# 格式: {Name: {mode, prompt_type, lr, reg}}
# Mode: baseline (無優化), pure, fixed, adaptive
TEST_CASES = {
    # 1. Baseline (Zero-Shot)
    "Base_Empty": {"mode": "baseline", "neg": EMPTY_NEGATIVE_PROMPT, "lr": 0.0, "reg": 0.0},
    "Base_Long":  {"mode": "baseline", "neg": LONG_NEGATIVE_PROMPT,  "lr": 0.0, "reg": 0.0},
    
    # 2. Pure (No Mask) - LR=0.25 (High strength)
    "Pure_Empty": {"mode": "pure",     "neg": EMPTY_NEGATIVE_PROMPT, "lr": 0.25, "reg": 0.0},
    "Pure_Long":  {"mode": "pure",     "neg": LONG_NEGATIVE_PROMPT,  "lr": 0.25, "reg": 0.0},
    
    # 3. Fixed (MinMax) - LR=0.05, Reg=1.5
    "Fix_Empty":  {"mode": "fixed",    "neg": EMPTY_NEGATIVE_PROMPT, "lr": 0.05, "reg": 1.5},
    "Fix_Long":   {"mode": "fixed",    "neg": LONG_NEGATIVE_PROMPT,  "lr": 0.05, "reg": 1.5},
    
    # 4. Adaptive (SOTA) - LR=0.12, Reg=1.25
    "Ada_Empty":  {"mode": "adaptive", "neg": EMPTY_NEGATIVE_PROMPT, "lr": 0.12, "reg": 1.25},
    "Ada_Long":   {"mode": "adaptive", "neg": LONG_NEGATIVE_PROMPT,  "lr": 0.12, "reg": 1.25},
}

# === 輔助函式 ===
def load_model():
    print(f"⏳ Loading SD Model...")
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

def get_mask(model, sampler, z_center, c, uc, mode):
    # 估計不確定性與 Mask 生成邏輯
    if mode == "baseline": return None
    if mode == "pure": return torch.ones_like(z_center)
    
    # Sampling for variance
    z_recs = []
    with torch.no_grad(), autocast("cuda"):
        for _ in range(4): # Repeats=4
            noise = torch.randn_like(z_center) * 0.05
            z_in = z_center + noise
            z_0, _ = sampler.sample(steps=8, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                    x_T=z_in, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=8, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            z_recs.append(z_rec)
    
    variance = torch.var(torch.stack(z_recs), dim=0)
    var_mean = torch.mean(variance, dim=1, keepdim=True)
    
    if mode == "fixed":
        # Min-Max Normalization
        v_min = var_mean.min(); v_max = var_mean.max()
        norm_var = (var_mean - v_min) / (v_max - v_min + 1e-8)
        mask = 1.0 - norm_var
        mask = torch.pow(mask, 2)
        mask = mask * 0.7 + 0.3
        
    elif mode == "adaptive":
        # Quantile + Power 6 + Adaptive Floor
        v_min = torch.quantile(var_mean, 0.01); v_max = torch.quantile(var_mean, 0.99)
        norm_var = torch.clamp((var_mean - v_min) / (v_max - v_min + 1e-8), 0, 1)
        mask = 1.0 - torch.pow(norm_var, 6.0)
        avg_u = torch.mean(norm_var).item()
        floor = min(max(0.4 + 0.3*avg_u, 0.4), 0.7)
        mask = mask * (1.0 - floor) + floor
        
    return mask.repeat(1, 4, 1, 1)

# 通用生成函式
def generate_image_custom(model, sampler, prompt, neg_prompt_str, secret_key, payload, outpath, mode, lr, reg):
    # 準備 Payload
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
    bits = bits[:16384].reshape(1, 4, 64, 64)
    mapper = ours_mapping(bits=1)
    
    # Embedding
    z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
    z_target = torch.from_numpy(z_target_numpy).float().to("cuda")

    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([neg_prompt_str])

    # 優化設置
    opt_iters = 0 if mode == "baseline" else OPT_ITERS
    uncertainty_mask = get_mask(model, sampler, z_target, c, uc, mode) if opt_iters > 0 else None

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    
    # Optimization Loop
    for i in range(opt_iters):
        with torch.no_grad(), autocast("cuda"):
            z_0, _ = sampler.sample(steps=8, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                    x_T=z_opt, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=8, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
        
        diff = (z_rec - z_target).float()
        reg_loss = 2.0 * (z_opt - z_target)
        grad = diff + reg * reg_loss
        if uncertainty_mask is not None: grad = grad * uncertainty_mask
        
        lr_t = lr * (1.0 - (0.5 * (i / (opt_iters + 1))))
        z_opt = torch.clamp(z_opt - lr_t * grad, -4.0, 4.0)

    # Decode
    final_latent = z_target if mode == "baseline" else z_opt
    
    with torch.no_grad(), autocast("cuda"):
        z_final, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                    x_T=final_latent, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_final)
    
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    Image.fromarray(img_np.astype(np.uint8)).save(outpath)

# Bob 解碼
def run_bob_decode(model, sampler, img_path, prompt, neg_prompt_str, secret_key, payload):
    try:
        img_tensor = load_512(img_path).cuda()
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([neg_prompt_str]) # 必須一致
        
        with torch.no_grad(), autocast("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
            z_rec, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=init_latent, DPMencode=True, DPMdecode=False, verbose=False)
        
        mapper = ours_mapping(bits=1)
        decoded = mapper.decode_secret_soft(z_rec.cpu().numpy(), seed_kernel=secret_key, seed_shuffle=secret_key+999)
        bits_hat = np.round(decoded).astype(np.uint8).flatten()
        
        bits_gt = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        if len(bits_gt) < 16384: bits_gt = np.pad(bits_gt, (0, 16384 - len(bits_gt)), 'constant')
        
        min_len = min(len(bits_hat), len(bits_gt))
        matches = np.sum(bits_hat[:min_len] == bits_gt[:min_len])
        return matches / min_len
    except: return 0.0

def resize_real_images():
    if not os.path.exists(DIR_REAL_COCO):
        print("⚠️ COCO Dir not found, skipping FID.")
        return
    os.makedirs(DIR_REAL_RESIZED, exist_ok=True)
    files = [f for f in os.listdir(DIR_REAL_COCO) if f.lower().endswith(('.jpg', '.png'))]
    if len(os.listdir(DIR_REAL_RESIZED)) > 10: return
    print("⚙️ Resizing Real Images...")
    for f in tqdm(files[:TOTAL_SAMPLES*2]): 
        try:
            with Image.open(os.path.join(DIR_REAL_COCO, f)) as img:
                img.convert('RGB').resize((512, 512), Image.BICUBIC).save(os.path.join(DIR_REAL_RESIZED, f))
        except: pass

def main():
    print(f"🚀 PROMPT IMPACT & ABLATION FULL TEST (N={TOTAL_SAMPLES}) 🚀")
    
    resize_real_images()
    model = load_model()
    sampler = DPMSolverSampler(model)
    
    payload = os.urandom(16384 // 8)
    
    # 準備 Prompts
    prompts = []
    if os.path.exists(os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")):
        with open(os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")) as f:
            prompts = [l.strip() for l in f if l.strip()]
    if not prompts: prompts = ["A futuristic city", "A photo of a dog", "A beautiful forest"]
    
    results = defaultdict(dict) # {case: {'NoAtk': 0, 'JPEG50': 0, 'FID': 0}}
    
    # 建立資料夾
    for case_name in TEST_CASES.keys():
        path = os.path.join(OUTPUT_DIR, case_name)
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    print("\n📸 Generating & Testing...")
    
    for case_name, cfg in TEST_CASES.items():
        print(f"  Running Case: {case_name} ({cfg['mode']}, Neg='{cfg['neg'][:5]}...')")
        
        case_dir = os.path.join(OUTPUT_DIR, case_name)
        accs_no_atk = []
        accs_jpeg = []
        
        for i in tqdm(range(TOTAL_SAMPLES), desc="Samples"):
            seed = 10000 + i
            prompt = prompts[i % len(prompts)]
            out_p = os.path.join(case_dir, f"{i:05d}.png")
            
            # 1. Generate
            generate_image_custom(
                model, sampler, prompt, cfg['neg'], seed, payload, out_p, cfg['mode'], cfg['lr'], cfg['reg']
            )
            
            # 2. No Attack Decode
            acc_no = run_bob_decode(model, sampler, out_p, prompt, cfg['neg'], seed, payload)
            accs_no_atk.append(acc_no)
            
            # 3. Attack (JPEG 50) & Decode
            try:
                img_tensor = load_512(out_p).cuda()
                tmp_atk = os.path.join(OUTPUT_DIR, f"tmp_{case_name}_{i}")
                jpeg(img_tensor, 50, tmp_image_name=tmp_atk)
                
                final_atk = tmp_atk + ".jpg"
                if os.path.exists(final_atk):
                    acc_j = run_bob_decode(model, sampler, final_atk, prompt, cfg['neg'], seed, payload)
                    accs_jpeg.append(acc_j)
                    os.remove(final_atk)
            except: pass
            
        # 統計
        results[case_name]['NoAtk'] = np.mean(accs_no_atk)
        results[case_name]['JPEG50'] = np.mean(accs_jpeg)
        
        # FID
        print(f"    Calculating FID for {case_name}...")
        try:
            fid = fid_score.calculate_fid_given_paths(
                [DIR_REAL_RESIZED, case_dir], 
                batch_size=50, device="cuda", dims=2048, num_workers=0
            )
        except: fid = 0.0
        results[case_name]['FID'] = fid
        
        print(f"    -> Acc(No): {results[case_name]['NoAtk']*100:.2f}%, Acc(J50): {results[case_name]['JPEG50']*100:.2f}%, FID: {fid:.4f}")

    # 輸出比較表
    print("\n" + "="*90)
    print("FULL COMPARISON: PROMPT & METHOD")
    print("-" * 90)
    print(f"{'Case':<12} | {'Mode':<10} | {'Neg':<6} | {'Acc (No)':<10} | {'Acc (J50)':<10} | {'FID':<10}")
    print("-" * 90)
    
    # 依序輸出
    order = ["Base_Empty", "Base_Long", "Pure_Empty", "Pure_Long", 
             "Fix_Empty", "Fix_Long", "Ada_Empty", "Ada_Long"]
    
    for case_name in order:
        if case_name not in results: continue
        cfg = TEST_CASES[case_name]
        neg_type = "Empty" if cfg['neg'] == "" else "Long"
        res = results[case_name]
        print(f"{case_name:<12} | {cfg['mode']:<10} | {neg_type:<6} | {res['NoAtk']*100:<9.2f}% | {res['JPEG50']*100:<9.2f}% | {res['FID']:.4f}")
    print("="*90)

if __name__ == "__main__":
    main()