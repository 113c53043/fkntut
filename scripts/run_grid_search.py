import os
import sys
import shutil 
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import cv2
import lpips
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
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from pure_alice_final import generate_alice_image
    from mapping_module import ours_mapping
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    sys.exit(1)

try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False
    print("⚠️ PIQ not found. BRISQUE will be skipped.")

# 攻擊函式
try:
    from robust_eval import awgn, jpeg, identity
    from utils import load_512
except ImportError:
    sys.path.append(os.path.join(PARENT_DIR, "scripts"))
    try:
        from robust_eval import awgn, jpeg, identity
        from utils import load_512
    except:
        # Fallback
        def awgn(img, *args, **kwargs): return img
        def jpeg(img, *args, **kwargs): return img
        def identity(img, *args, **kwargs): return img
        def load_512(path): return torch.randn(1, 3, 512, 512)

# 路徑變數
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "grid_search_pareto_v1")
DIR_COVER = os.path.join(OUTPUT_DIR, "cover")
RESULT_JSON_PATH = os.path.join(OUTPUT_DIR, "grid_search_results.json") 
LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

# === 2. 實驗配置 ===
TOTAL_SAMPLES = 200

LR_LIST = [0.15, 0.20, 0.25, 0.30] 
REG_LIST = [0.3, 0.5, 0.8, 1.0]
MODES = ["no_mask", "adaptive"]

# 確保名稱固定，方便存取
PROXY_ATTACKS = [
    (identity, [None], "Identity", ".png"),
    (awgn, [0.05], "Noise", ".png"), 
    (jpeg, [50], "JPEG", ".jpg")
]

# === 3. 核心函式 ===

def load_model_once():
    print(f"⏳ Loading Shared SD Model (In-Memory)...", flush=True)
    config = OmegaConf.load(CONFIG_PATH)
    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    except:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    sampler = DPMSolverSampler(model)
    return model, sampler

def prepare_payload(raw_data):
    CAPACITY_BYTES = 16384 // 8
    if len(raw_data) > CAPACITY_BYTES - 2: 
        raw_data = raw_data[:CAPACITY_BYTES-2]
    length_header = len(raw_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    if len(final_payload) < CAPACITY_BYTES: 
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    return final_payload

def fast_bob_decode(model, sampler, img_tensor, prompt, secret_key, gt_bytes):
    try:
        img_tensor = img_tensor.to(model.device)
        if img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0:
            img_tensor = img_tensor * 2.0 - 1.0

        if img_tensor.shape[-1] != 512 or img_tensor.shape[-2] != 512:
            img_tensor = F.interpolate(img_tensor, size=(512, 512), mode='bicubic', align_corners=False)

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
        total_bits = max(len(arr_a), len(arr_b))
        return (matches / total_bits) * 100.0
    except Exception as e:
        return 0.0

def run_alice_generation_in_process(model, sampler, prompt, session_key, out_path, payload_data, lr, reg, mode):
    try:
        success, _, _ = generate_alice_image(
            model=model,
            sampler=sampler,
            prompt=prompt,
            secret_key=session_key,
            payload_data=payload_data,
            outpath=out_path,
            init_latent_path=None,
            opt_iters=15,
            lr=lr,
            lambda_reg=reg,
            mode=mode,
            dpm_steps=20,
            early_stop_threshold=0.0693
        )
        return success
    except Exception as e:
        print(f"\n❌ Alice Gen Error ({mode}): {e}")
        import traceback
        traceback.print_exc()
        return False

class QualityEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        except:
            self.lpips_fn = None
        
    def calc_brisque(self, p1):
        if not BRISQUE_AVAILABLE: return None 
        try:
            img = cv2.imread(p1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512,512)) / 255.0
            t1 = torch.tensor(img.transpose(2,0,1)).float().unsqueeze(0).cpu()
            with torch.no_grad(): 
                return brisque(t1, data_range=1.0).item()
        except Exception as e:
            return None

def plot_pareto_analysis(results, output_path="grid_search_pareto.png"):
    no_mask_data = [r for r in results if r.get('mode') == 'no_mask' and r.get('brisque', 0) > 0]
    adaptive_data = [r for r in results if r.get('mode') == 'adaptive' and r.get('brisque', 0) > 0]
    
    if not no_mask_data and not adaptive_data: return

    plt.figure(figsize=(10, 8))
    
    # 圖表仍然使用 Mean Acc (acc_mean) 來畫，因為 2D 圖表無法同時顯示 3 個維度的 Accuracy
    if no_mask_data:
        nm_accs = [r['acc_mean'] for r in no_mask_data]
        nm_brisques = [r['brisque'] for r in no_mask_data]
        plt.scatter(nm_brisques, nm_accs, c='tab:red', s=100, alpha=0.7, edgecolors='k', label='No Mask', marker='x')

    if adaptive_data:
        ad_accs = [r['acc_mean'] for r in adaptive_data]
        ad_brisques = [r['brisque'] for r in adaptive_data]
        plt.scatter(ad_brisques, ad_accs, c='tab:blue', s=120, alpha=0.8, edgecolors='k', label='Adaptive (Ours)', marker='o')

    plt.title(f'Robustness vs. Naturalness (Mean of 3 Attacks, N={TOTAL_SAMPLES})')
    plt.xlabel('BRISQUE Score (Lower is Better) ->')
    plt.ylabel('Mean Bit Accuracy (%) (Higher is Better) ->')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Pareto Analysis Chart saved to: {output_path}")

# === 4. 主流程 ===
def main():
    print(f"🚀 GRID SEARCH (Separated Metrics) (N={TOTAL_SAMPLES}) 🚀")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIR_COVER, exist_ok=True)
    
    results = []
    if os.path.exists(RESULT_JSON_PATH):
        try:
            with open(RESULT_JSON_PATH, 'r') as f:
                results = json.load(f)
            print(f"📦 Loaded {len(results)} existing records.")
        except:
            results = []

    model, sampler = load_model_once()

    prompts = []
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST) as f: lines = [l.strip() for l in f if l.strip()]
        while len(prompts) < TOTAL_SAMPLES: prompts.extend(lines)
        prompts = prompts[:TOTAL_SAMPLES]
    else:
        prompts = ["A futuristic city"] * TOTAL_SAMPLES

    evaluator = QualityEvaluator()

    payload_file = os.path.join(OUTPUT_DIR, "shared_payload.dat")
    if os.path.exists(payload_file):
        with open(payload_file, "rb") as f: raw_payload_source = f.read()
    else:
        raw_payload_source = os.urandom(2048)
        with open(payload_file, "wb") as f: f.write(raw_payload_source)

    final_payload_with_header = prepare_payload(raw_payload_source)
    gt_bytes = np.frombuffer(final_payload_with_header, dtype=np.uint8)

    print("\n[Phase 0] Generating Covers...")
    for i in tqdm(range(TOTAL_SAMPLES), desc="Covers"):
        cover_p = os.path.join(DIR_COVER, f"{i:05d}.png")
        if not os.path.exists(cover_p):
            run_alice_generation_in_process(model, sampler, prompts[i], 10000+i, cover_p, final_payload_with_header, lr=0.0, reg=0.0, mode="baseline")

    total_combos = len(LR_LIST) * len(REG_LIST) * len(MODES)
    curr = 0

    for lr in LR_LIST:
        for reg in REG_LIST:
            for mode in MODES:
                curr += 1
                
                existing_record = None
                for r in results:
                    if r.get('lr') == lr and r.get('reg') == reg and r.get('mode') == mode:
                        if r.get('brisque', 0) > 0.0001 and r.get('n_samples', 0) >= TOTAL_SAMPLES: 
                            existing_record = r
                        break
                
                if existing_record:
                    print(f"[{curr}/{total_combos}] Skipping {mode} {lr}/{reg} (Done)")
                    continue

                results = [r for r in results if not (r.get('lr') == lr and r.get('reg') == reg and r.get('mode') == mode)]

                print(f"\n[{curr}/{total_combos}] Running {mode} | LR={lr}, Reg={reg}")
                
                combo_dir = os.path.join(OUTPUT_DIR, f"{mode}_lr{lr}_reg{reg}")
                os.makedirs(combo_dir, exist_ok=True)
                
                breakdown_scores = defaultdict(list)
                brisque_scores = []
                
                pbar = tqdm(range(TOTAL_SAMPLES), desc="Sampling", leave=False)
                
                for i in pbar:
                    session_key = 10000 + i
                    prompt = prompts[i]
                    stego_p = os.path.join(combo_dir, f"{i:05d}.png")
                    
                    is_existing = os.path.exists(stego_p)
                    if not is_existing:
                        pbar.set_postfix({"State": "New Gen"})
                        success = run_alice_generation_in_process(model, sampler, prompt, session_key, stego_p, final_payload_with_header, lr, reg, mode)
                        if not success: continue
                    else:
                        pbar.set_postfix({"State": "Old File"})
                    
                    if os.path.exists(stego_p):
                        brisque_val = evaluator.calc_brisque(stego_p)
                        if brisque_val is not None:
                            brisque_scores.append(brisque_val)

                        try:
                            img_tensor = load_512(stego_p).cuda()
                            
                            local_accs = []
                            for atk_fn, args, atk_name, ext in PROXY_ATTACKS:
                                attacked_img = img_tensor.clone()
                                # [修正] 確保接收攻擊後圖片
                                attacked_img = atk_fn(attacked_img, args[0], tmp_image_name="dummy")
                                acc = fast_bob_decode(model, sampler, attacked_img, prompt, session_key, gt_bytes)
                                
                                breakdown_scores[atk_name].append(acc)
                                local_accs.append(acc)
                            
                            if i < 3 or i == TOTAL_SAMPLES - 1:
                                mean_local = np.mean(local_accs) if local_accs else 0.0
                                tqdm.write(f"   [Img {i:02d}] Mean Acc: {mean_local:.1f}%")
                            
                            del img_tensor
                        except Exception as e:
                            pass
                
                pbar.close()

                if breakdown_scores and brisque_scores:
                    # [關鍵修正] 不再計算總平均來覆蓋，而是分別計算
                    acc_identity = np.mean(breakdown_scores.get("Identity", [0.0]))
                    acc_noise = np.mean(breakdown_scores.get("Noise", [0.0]))
                    acc_jpeg = np.mean(breakdown_scores.get("JPEG", [0.0]))
                    
                    # 計算總平均僅供圖表排序用
                    acc_mean = np.mean([acc_identity, acc_noise, acc_jpeg])
                    avg_brisque = np.mean(brisque_scores)
                    
                    print(f"   -> Result: Identity={acc_identity:.2f}%, Noise={acc_noise:.2f}%, JPEG={acc_jpeg:.2f}% | BRISQUE={avg_brisque:.2f}")
                    
                    results.append({
                        "mode": mode, 
                        "lr": lr, 
                        "reg": reg, 
                        "acc_mean": acc_mean,      # 綜合指標
                        "acc_identity": acc_identity,
                        "acc_noise": acc_noise,
                        "acc_jpeg": acc_jpeg,
                        "brisque": avg_brisque,
                        "n_samples": TOTAL_SAMPLES
                    })
                    with open(RESULT_JSON_PATH, 'w') as f:
                        json.dump(results, f, indent=4)
                else:
                    print("   -> ⚠️ Skipped due to missing metrics.")

    print("\n" + "="*80)
    print("PARETO ANALYSIS REPORT")
    print("-" * 80)
    
    plot_pareto_analysis(results, os.path.join(OUTPUT_DIR, "grid_search_pareto.png"))
    
    print("\n📍 Top 3 Configurations (By Mean Accuracy):")
    sorted_results = sorted(results, key=lambda x: x['acc_mean'], reverse=True)
    for r in sorted_results[:6]:
        print(f"[{r['mode']}] LR={r['lr']}, Reg={r['reg']} -> Mean={r['acc_mean']:.2f}% (Id:{r['acc_identity']:.1f}, Ns:{r['acc_noise']:.1f}, Jp:{r['acc_jpeg']:.1f}), BRISQUE={r['brisque']:.2f}")

if __name__ == "__main__":
    main()