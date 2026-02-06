import os
import sys
import shutil 
import torch
import numpy as np
import json
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import ssl
from omegaconf import OmegaConf
from torch import autocast
from collections import defaultdict

# 忽略 SSL 警告
ssl._create_default_https_context = ssl._create_unverified_context

# === 1. 路徑與 Import 設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path: sys.path.insert(0, PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from pure_alice_final import generate_alice_image
    from mapping_module import ours_mapping
except ImportError as e:
    print(f"⚠️ Import Error: {e}"); sys.exit(1)

try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False
    print("⚠️ PIQ not found. BRISQUE will be skipped.")

# 攻擊函式
sys.path.append(os.path.join(PARENT_DIR, "scripts"))
try:
    from robust_eval import awgn, jpeg, identity
    from utils import load_512
except ImportError:
    # Fallback
    def awgn(img, *args, **kwargs): return img
    def jpeg(img, *args, **kwargs): return img
    def identity(img, *args, **kwargs): return img
    def load_512(path): return torch.randn(1, 3, 512, 512)
except Exception:
    pass

# 路徑變數
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "power_sensitivity_1")
DIR_COVER = os.path.join(OUTPUT_DIR, "cover")
RESULT_JSON = os.path.join(OUTPUT_DIR, "power_results.json")
LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

# === Experiment Config ===
TOTAL_SAMPLES = 200  
POWER_LIST = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
FIXED_LR = 0.2 
FIXED_REG = 0.3

PROXY_ATTACKS = [
    (identity, [None], "Identity", ".png"),
    (awgn, [0.05], "Noise", ".png"),
    (jpeg, [50], "JPEG", ".jpg")
]

# === 3. 核心函式 ===

def load_model_once():
    print(f"⏳ Loading Model...", flush=True)
    config = OmegaConf.load(CONFIG_PATH)
    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    except:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    sampler = DPMSolverSampler(model)
    return model, sampler

def prepare_payload(raw_data):
    CAPACITY_BYTES = 16384 // 8
    if len(raw_data) > CAPACITY_BYTES - 2: raw_data = raw_data[:CAPACITY_BYTES-2]
    length_header = len(raw_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    return final_payload

def fast_bob_decode(model, sampler, img_tensor, prompt, secret_key, gt_bytes):
    try:
        img_tensor = img_tensor.to(model.device)
        if img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0:
            img_tensor = img_tensor * 2.0 - 1.0

        if img_tensor.shape[-1] != 512:
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=(512, 512), mode='bicubic')
        
        torch.manual_seed(secret_key)
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
        
        with torch.no_grad(), autocast("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
            z_rec, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=init_latent, DPMencode=True, DPMdecode=False, verbose=False)
        mapper = ours_mapping(bits=1)
        z_rec_numpy = z_rec.cpu().numpy()
        decoded_float = mapper.decode_secret_soft(z_rec_numpy, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
        bits = np.round(decoded_float).astype(np.uint8).flatten()
        extracted_bytes = np.packbits(bits).tobytes()
        
        arr_a = np.unpackbits(np.frombuffer(extracted_bytes, dtype=np.uint8))
        arr_b = np.unpackbits(np.frombuffer(gt_bytes, dtype=np.uint8))
        min_len = min(len(arr_a), len(arr_b))
        matches = np.sum(arr_a[:min_len] == arr_b[:min_len])
        return (matches / max(len(arr_a), len(arr_b))) * 100.0
    except: return 0.0

def run_alice_generation_in_process(model, sampler, prompt, session_key, out_path, payload_data, power):
    try:
        success, _, _ = generate_alice_image(
            model=model, sampler=sampler, prompt=prompt, secret_key=session_key,
            payload_data=payload_data, outpath=out_path, init_latent_path=None,
            opt_iters=15, lr=FIXED_LR, lambda_reg=FIXED_REG, mode="adaptive", 
            dpm_steps=20, early_stop_threshold=0.0693,
            mask_power=power 
        )
        return success
    except Exception as e:
        print(f"Alice Error: {e}")
        return False

def calc_brisque(p1):
    if not BRISQUE_AVAILABLE: return None 
    try:
        img = cv2.imread(p1); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512)) / 255.0
        t1 = torch.tensor(img.transpose(2,0,1)).float().unsqueeze(0).cpu()
        with torch.no_grad(): return brisque(t1, data_range=1.0).item()
    except: return None

def plot_dual_axis_sensitivity(results, output_path):
    valid = [r for r in results if r.get('brisque', 0) > 0]
    if not valid: return
    
    valid.sort(key=lambda x: x['power'])
    powers = np.array([r['power'] for r in valid])
    # 這裡改成抓 acc_mean
    accs = np.array([r['acc_mean'] for r in valid])
    brisques = np.array([r['brisque'] for r in valid])

    # 繪圖
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    color_acc = '#1f77b4'
    ax1.set_xlabel('Mask Power ($p$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Bit Accuracy (%)', color=color_acc, fontsize=14, fontweight='bold')
    ax1.scatter(powers, accs, color=color_acc, alpha=0.5, label='Raw Accuracy')
    
    if len(powers) > 3:
        x_smooth = np.linspace(powers.min(), powers.max(), 300)
        p_acc = np.poly1d(np.polyfit(powers, accs, 3))
        y_acc_smooth = p_acc(x_smooth)
        ax1.plot(x_smooth, y_acc_smooth, color=color_acc, linewidth=2)

    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    color_qual = '#d62728'
    ax2.set_ylabel('BRISQUE Score (Lower is Better)', color=color_qual, fontsize=14, fontweight='bold')
    ax2.scatter(powers, brisques, color=color_qual, marker='s', alpha=0.5, label='Raw BRISQUE')
    
    if len(powers) > 3:
        p_bri = np.poly1d(np.polyfit(powers, brisques, 3))
        y_bri_smooth = p_bri(x_smooth)
        ax2.plot(x_smooth, y_bri_smooth, color=color_qual, linestyle='--', linewidth=2)

    ax2.tick_params(axis='y', labelcolor=color_qual)

    plt.title('Sensitivity Analysis: Mask Power', fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Sensitivity Chart saved to {output_path}")

def main():
    print(f"🚀 POWER SENSITIVITY SEARCH (Detail Mode) (N={TOTAL_SAMPLES}) 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIR_COVER, exist_ok=True)
    
    results = []
    if os.path.exists(RESULT_JSON):
        try:
            with open(RESULT_JSON, 'r') as f: results = json.load(f)
            print(f"📦 Loaded {len(results)} existing records.")
        except: pass

    model, sampler = load_model_once()
    
    prompts = []
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE) as f: prompts = [l.strip() for l in f if l.strip()][:TOTAL_SAMPLES]
    else: prompts = ["A futuristic city"] * TOTAL_SAMPLES

    payload_file = os.path.join(OUTPUT_DIR, "shared_power_payload.dat")
    if os.path.exists(payload_file):
        with open(payload_file, "rb") as f: raw_payload = f.read()
    else:
        raw_payload = os.urandom(2048)
        with open(payload_file, "wb") as f: f.write(raw_payload)

    final_payload = prepare_payload(raw_payload)
    gt_bytes = np.frombuffer(final_payload, dtype=np.uint8)

    print("\n[Phase 0] Generating Covers...")
    for i in tqdm(range(TOTAL_SAMPLES)):
        p = os.path.join(DIR_COVER, f"{i:05d}.png")
        if not os.path.exists(p):
            run_alice_generation_in_process(model, sampler, prompts[i], 10000+i, p, final_payload, power=1.0)

    for power in POWER_LIST:
        existing_record = None
        for r in results:
            if r['power'] == power:
                # [關鍵修正]：不僅檢查樣本數，還要檢查是否包含新的詳細欄位
                # 假設我們要檢查 acc_identity 是否存在
                if r.get('n_samples', 0) >= TOTAL_SAMPLES and 'acc_identity' in r:
                    existing_record = r
                break
        
        if existing_record:
            print(f"Skipping Power={power} (Already has detailed stats)")
            continue
        
        # 移除舊紀錄 (準備重新計算並寫入詳細版)
        results = [r for r in results if r['power'] != power]

        print(f"\n⚡ Processing Mask Power = {power} (Recalculating if images exist)")
        combo_dir = os.path.join(OUTPUT_DIR, f"power_{power}")
        os.makedirs(combo_dir, exist_ok=True)

        breakdown_scores = defaultdict(list)
        brisque_scores = []
        
        pbar = tqdm(range(TOTAL_SAMPLES), desc="Processing", leave=False)
        
        for i in pbar:
            key = 10000 + i
            prompt = prompts[i]
            stego_p = os.path.join(combo_dir, f"{i:05d}.png")
            
            is_existing = os.path.exists(stego_p)
            
            # [修正邏輯]：如果存在，跳過生成，直接進下面評估。如果不存在，才生成。
            if not is_existing:
                pbar.set_postfix({"State": "Gen"})
                success = run_alice_generation_in_process(model, sampler, prompt, key, stego_p, final_payload, power)
                if not success: continue
            else:
                pbar.set_postfix({"State": "Load"}) # 讀取舊圖
            
            # 只要圖存在，就進行計算 (包含舊圖)
            if os.path.exists(stego_p):
                b = calc_brisque(stego_p)
                if b is not None: brisque_scores.append(b)
                
                try:
                    img = load_512(stego_p).cuda()
                    local_accs = []
                    for atk, args, atk_name, _ in PROXY_ATTACKS:
                        att_img = img.clone()
                        att_img = atk(att_img, args[0], tmp_image_name="dummy")
                        
                        acc = fast_bob_decode(model, sampler, att_img, prompt, key, gt_bytes)
                        
                        breakdown_scores[atk_name].append(acc)
                        local_accs.append(acc)
                    
                    if i < 3:
                        mean_local = np.mean(local_accs) if local_accs else 0.0
                        # tqdm.write(f"   [Img {i:02d}] Acc: {mean_local:.1f}%")
                        
                    del img
                except Exception as e:
                    print(f"Eval Error: {e}")
        
        pbar.close()

        if breakdown_scores and brisque_scores:
            # [關鍵修正]：依照要求格式整理 JSON
            all_accs = [val for sublist in breakdown_scores.values() for val in sublist]
            avg_acc_global = np.mean(all_accs)
            avg_bri = np.mean(brisque_scores)
            
            record = {
                "power": power,
                "acc_mean": avg_acc_global, # 符合你的期望
                "brisque": avg_bri,
                "n_samples": TOTAL_SAMPLES
            }
            
            # 填入各項詳細準確率
            for atk_name, scores in breakdown_scores.items():
                key_name = f"acc_{atk_name.lower()}" # 例如 acc_identity, acc_noise
                record[key_name] = np.mean(scores)
            
            print(f"   -> Result: Acc={avg_acc_global:.2f}%, Identity={record.get('acc_identity',0):.2f}%")
            
            results.append(record)
            with open(RESULT_JSON, 'w') as f: json.dump(results, f, indent=4)
        
        # 這裡建議把 rmtree 註解掉，因為你可能之後還要用圖片
        # shutil.rmtree(combo_dir)

    print("\nGenerating Analysis Plot...")
    plot_dual_axis_sensitivity(results, os.path.join(OUTPUT_DIR, "power_sensitivity.pdf"))

if __name__ == "__main__":
    main()