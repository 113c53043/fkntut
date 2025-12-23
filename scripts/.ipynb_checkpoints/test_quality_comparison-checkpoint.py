import os
import sys
import torch
import numpy as np
import cv2
import lpips
import ssl
import shutil
import gc
from collections import defaultdict
from pytorch_fid import fid_score
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from tqdm import tqdm
import json
import random

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
    # 確實引用了修正後的 Alice 腳本
    from pure_alice_uncertainty_fixed import generate_alice_image 
except ImportError as e:
    print(f"❌ 無法導入模組: {e}")
    sys.exit(1)

# BRISQUE
try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False

ssl._create_default_https_context = ssl._create_unverified_context

# === 配置 ===
TOTAL_IMAGES = 1000 # 測試 100 張
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
DIR_REAL_COCO = os.path.join(CURRENT_DIR, "coco_val2017")
PATH_CAPTIONS = os.path.join(CURRENT_DIR, "coco_annotations", "captions_val2017.json")

OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "quality_comparison_final_2")
DIR_COVER = os.path.join(OUTPUT_ROOT, "cover_sd")
DIR_MAPPED = os.path.join(OUTPUT_ROOT, "mapped_base")
DIR_PURE = os.path.join(OUTPUT_ROOT, "ours_pure")         # Pure
DIR_UNC_FIXED = os.path.join(OUTPUT_ROOT, "ours_unc_fixed") # Fixed
DIR_UNC_ADAPT = os.path.join(OUTPUT_ROOT, "ours_unc_adaptive") # Adaptive
DIR_REAL_RESIZED = os.path.join(OUTPUT_ROOT, "real_coco_resized")
DIR_TEMP = os.path.join(OUTPUT_ROOT, "temp")
DIR_LATENT = os.path.join(OUTPUT_ROOT, "latents") 

class QualityEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        # 載入 LPIPS 模型
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    def load_img_tensor(self, path, range_norm=True):
        try:
            img = cv2.imread(path)
            if img is None: return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            img = img / 255.0
            if range_norm: 
                # LPIPS 需要 [-1, 1]
                img = img * 2.0 - 1.0
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
        except: return None

    def calculate_lpips(self, path_ref, path_target):
        t_ref = self.load_img_tensor(path_ref, range_norm=True)
        t_tar = self.load_img_tensor(path_target, range_norm=True)
        if t_ref is None or t_tar is None: return None
        with torch.no_grad():
            return self.loss_fn_alex(t_ref, t_tar).item()

    def calculate_brisque(self, path_target):
        if not BRISQUE_AVAILABLE: return 0.0
        # BRISQUE 需要 [0, 1]
        t_tar = self.load_img_tensor(path_target, range_norm=False)
        if t_tar is None: return 0.0
        try:
            with torch.no_grad():
                score = brisque(t_tar, data_range=1.0, reduction='none')
                return score.item()
        except: return 0.0

def load_model():
    print(f"⏳ Loading SD Model...")
    config = OmegaConf.load(CONFIG_PATH)
    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def load_coco_prompts(json_path, limit=1000):
    if not os.path.exists(json_path):
        return ["A professional photo of a dog"] * limit
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = [item['caption'] for item in data['annotations']]
    random.shuffle(captions)
    return captions[:limit]

def generate_cover_image(model, sampler, prompt, out_path, seed):
    if os.path.exists(out_path): return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    shape = (4, 64, 64)
    x_T = torch.randn(1, *shape, device="cuda")
    with torch.no_grad(), autocast("cuda"):
        z_enc, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=shape,
                                  unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                  x_T=x_T, verbose=False)
        x_samples = model.decode_first_stage(z_enc)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_path)

def generate_mapped_baseline_and_latent(model, sampler, prompt, session_key, payload_data, out_img_path, out_latent_path):
    if os.path.exists(out_img_path) and os.path.exists(out_latent_path): return
    bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
    if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
    bits = bits[:16384].reshape(1, 4, 64, 64)
    mapper = ours_mapping(bits=1)
    z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=session_key, seed_shuffle=session_key + 999)
    z_target = torch.from_numpy(z_target_numpy).float().to("cuda")
    torch.save(z_target, out_latent_path)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=z_target, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_img_path)

def resize_images(src_dir, dst_dir, target_size=(512, 512), max_images=None):
    if not os.path.exists(src_dir): return
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    if max_images: files = files[:max_images]
    for f in tqdm(files, desc="Resizing Real Images"):
        try:
            with Image.open(os.path.join(src_dir, f)) as img:
                img.convert('RGB').resize(target_size, Image.BICUBIC).save(os.path.join(dst_dir, f))
        except: pass

def main():
    print(f"🚀 Quality Comparison (Pure vs Fixed vs Adaptive): {TOTAL_IMAGES} images 🚀")
    
    dirs = [DIR_COVER, DIR_MAPPED, DIR_PURE, DIR_UNC_FIXED, DIR_UNC_ADAPT, DIR_REAL_RESIZED, DIR_TEMP, DIR_LATENT]
    for d in dirs: os.makedirs(d, exist_ok=True)
        
    payload_path = os.path.join(DIR_TEMP, "payload.dat")
    if not os.path.exists(payload_path):
        with open(payload_path, "wb") as f: f.write(os.urandom(2048))
    with open(payload_path, "rb") as f: payload_data = f.read()[:2046]
    
    prompts = load_coco_prompts(PATH_CAPTIONS, limit=TOTAL_IMAGES)
    model = load_model()
    sampler = DPMSolverSampler(model)

    print("\n📸 Generating Images...")
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        session_key = 123456 + i
        
        path_cover = os.path.join(DIR_COVER, f"{i:05d}.png")
        path_mapped = os.path.join(DIR_MAPPED, f"{i:05d}.png")
        
        path_pure = os.path.join(DIR_PURE, f"{i:05d}.png")
        path_unc_fixed = os.path.join(DIR_UNC_FIXED, f"{i:05d}.png")
        path_unc_adapt = os.path.join(DIR_UNC_ADAPT, f"{i:05d}.png")
        path_latent = os.path.join(DIR_LATENT, f"{i:05d}.pt")
        
        # 0. Cover & Mapped
        generate_cover_image(model, sampler, prompt, path_cover, seed=session_key)
        generate_mapped_baseline_and_latent(model, sampler, prompt, session_key, payload_data, path_mapped, path_latent)

        # 1. Pure (LR=0.25, use_uncertainty=False)
        if not os.path.exists(path_pure):
            generate_alice_image(
                model, sampler, prompt, session_key, payload_data, path_pure, 
                init_latent_path=path_latent, opt_iters=10, lr=0.25, lambda_reg=0.0, 
                use_uncertainty=False, adaptive_mask=False
            )

        # 2. Unc Fixed (LR=0.05, Reg=1.5, Adaptive=False)
        if not os.path.exists(path_unc_fixed):
            generate_alice_image(
                model, sampler, prompt, session_key, payload_data, path_unc_fixed, 
                init_latent_path=path_latent, opt_iters=10, lr=0.05, lambda_reg=1.5, 
                use_uncertainty=True, adaptive_mask=False
            )

        # 3. Unc Adaptive (LR=0.05, Reg=1.5, Adaptive=True)
        if not os.path.exists(path_unc_adapt):
            generate_alice_image(
                model, sampler, prompt, session_key, payload_data, path_unc_adapt, 
                init_latent_path=path_latent, opt_iters=10, lr=0.05, lambda_reg=1.5, 
                use_uncertainty=True, adaptive_mask=True
            )

    print("\n✅ Generation Complete! Calculating Metrics...")
    del model, sampler
    torch.cuda.empty_cache()
    
    evaluator = QualityEvaluator()
    resize_images(DIR_REAL_COCO, DIR_REAL_RESIZED, max_images=TOTAL_IMAGES)

    # FID Calculation
    def calc_fid(p1, p2):
        try: return fid_score.calculate_fid_given_paths([p1, p2], batch_size=50, device="cuda", dims=2048, num_workers=0)
        except: return 999.99

    print("  Calculating FID for Pure...")
    fid_pure = calc_fid(DIR_REAL_RESIZED, DIR_PURE)
    print("  Calculating FID for Fixed...")
    fid_fixed = calc_fid(DIR_REAL_RESIZED, DIR_UNC_FIXED)
    print("  Calculating FID for Adaptive...")
    fid_adapt = calc_fid(DIR_REAL_RESIZED, DIR_UNC_ADAPT)
    
    # BRISQUE & LPIPS
    stats = defaultdict(list)
    for i in tqdm(range(len(prompts)), desc="Calculating LPIPS/BRISQUE"):
        path_mapped = os.path.join(DIR_MAPPED, f"{i:05d}.png")
        
        path_pure = os.path.join(DIR_PURE, f"{i:05d}.png")
        path_fixed = os.path.join(DIR_UNC_FIXED, f"{i:05d}.png")
        path_adapt = os.path.join(DIR_UNC_ADAPT, f"{i:05d}.png")
        
        # Pure
        stats["LPIPS_Pure"].append(evaluator.calculate_lpips(path_mapped, path_pure))
        stats["BRISQUE_Pure"].append(evaluator.calculate_brisque(path_pure))

        # Fixed
        stats["LPIPS_Fixed"].append(evaluator.calculate_lpips(path_mapped, path_fixed))
        stats["BRISQUE_Fixed"].append(evaluator.calculate_brisque(path_fixed))

        # Adaptive
        stats["LPIPS_Adapt"].append(evaluator.calculate_lpips(path_mapped, path_adapt))
        stats["BRISQUE_Adapt"].append(evaluator.calculate_brisque(path_adapt))

    print("\n" + "="*80)
    print("FINAL COMPARISON: Pure vs Fixed vs Adaptive Mask")
    print("-" * 80)
    print(f"{'Metric':<10} | {'Pure':<15} | {'Fixed':<15} | {'Adaptive':<15}")
    print("-" * 80)
    
    print(f"{'FID (↓)':<10} | {fid_pure:<15.4f} | {fid_fixed:<15.4f} | {fid_adapt:<15.4f}")
    
    print(f"{'LPIPS (↓)':<10} | {np.mean(stats['LPIPS_Pure']):<15.4f} | {np.mean(stats['LPIPS_Fixed']):<15.4f} | {np.mean(stats['LPIPS_Adapt']):<15.4f}")
    
    print(f"{'BRISQUE(↓)':<10} | {np.mean(stats['BRISQUE_Pure']):<15.2f} | {np.mean(stats['BRISQUE_Fixed']):<15.2f} | {np.mean(stats['BRISQUE_Adapt']):<15.2f}")
    
    print("="*80)

if __name__ == "__main__":
    main()