import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import lpips
from tqdm import tqdm
from PIL import Image
from torch import autocast
import re
import cv2
from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from pytorch_fid import fid_score
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
    from pure_alice_final import generate_alice_image 
except ImportError as e:
    print(f"⚠️ Import Warning: {e}")

try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False

try:
    from robust_eval import identity, storage, resize, jpeg
    from utils import load_512
except ImportError:
    # Fallback if imports fail
    def identity(img, *args, **kwargs): pass
    def jpeg(img, *args, **kwargs): pass
    def resize(img, *args, **kwargs): pass
    def load_512(path): return torch.randn(1, 3, 512, 512)
    pass

# === 2. 配置 ===
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "ablation_study_final_cocoo")
DIR_REAL_COCO = os.path.join(MAS_GRDH_PATH, "scripts", "coco_val2017") 

TOTAL_GENERATE = 2000  # 建議設定為 200 或更多以獲得穩定數據
TOTAL_EVALUATE = 2000
SKIP_GENERATION_IF_EXISTS = True 
RUN_ATTACK_AND_BOB = True
CALC_FID = False

# [核心修改] 四大消融實驗模式
MODES = [
    "baseline",  
    "adaptive"  
]

ATTACK_SUITE = [
    # (identity, [None], "Identity", ".png"),
    # # ===== Storage / Compression =====
    # (jpeg, [95], "JPEG(95)", ".jpg"),
    # (jpeg, [80], "JPEG(80)", ".jpg"),
    # (jpeg, [60], "JPEG(60)", ".jpg"),
    # (jpeg, [50], "JPEG(50)", ".jpg"),

    # # ===== Geometric Scaling =====
    # (resize, [0.9],  "Resize(0.9)",  ".png"),
    # (resize, [0.75], "Resize(0.75)", ".png"),
    # (resize, [0.5],  "Resize(0.5)",  ".png"),

    # # ===== Motion Blur =====
    # (mblur, [3], "MBlur(3)", ".png"),
    # (mblur, [5], "MBlur(5)", ".png"),
    # (mblur, [7], "MBlur(7)", ".png"),

    # # ===== Gaussian Blur =====
    # (gblur, [3], "GBlur(3)", ".png"),
    (gblur, [5], "GBlur(5)", ".png"),
    # (gblur, [7], "GBlur(7)", ".png"),

    # # ===== Additive White Gaussian Noise =====
    # (awgn, [0.01], "Noise(0.01)", ".png"),
    # (awgn, [0.05], "Noise(0.05)", ".png"),
    #(awgn, [0.1], "Noise(0.1)", ".png")
]

LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

def load_shared_model():
    print(f"⏳ Loading Shared SD Model...", flush=True)
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
    return model

def fast_bob_decode(model, sampler, img_tensor, prompt, secret_key, gt_bits_path):
    try:
        if img_tensor.shape[-1] != 512 or img_tensor.shape[-2] != 512:
            img_tensor = F.interpolate(img_tensor, size=(512, 512), mode='bicubic', align_corners=False)

        torch.manual_seed(secret_key)
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
        
        with torch.no_grad(), autocast("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
            z_rec, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                      unconditional_guidance_scale=6.0, unconditional_conditioning=uc,
                                      x_T=init_latent, DPMencode=True, DPMdecode=False, verbose=False)

        mapper = ours_mapping(bits=1)
        z_rec_numpy = z_rec.cpu().numpy()
        decoded_float = mapper.decode_secret_soft(z_rec_numpy, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
        bits = np.round(decoded_float).astype(np.uint8).flatten()
        extracted_bytes = np.packbits(bits).tobytes()

        if not os.path.exists(gt_bits_path): return 0.0
        gt_bytes = np.load(gt_bits_path).tobytes()
        
        arr_a = np.unpackbits(np.frombuffer(extracted_bytes, dtype=np.uint8))
        arr_b = np.unpackbits(np.frombuffer(gt_bytes, dtype=np.uint8))
        
        min_len = min(len(arr_a), len(arr_b))
        matches = np.sum(arr_a[:min_len] == arr_b[:min_len])
        total_bits = max(len(arr_a), len(arr_b))
        return (matches / total_bits) * 100.0
    except Exception as e:
        # print(f"Bob Error: {e}")
        return 0.0

def generate_cover_image(model, sampler, prompt, out_path, seed):
    # Cover 原本就有檢查，這裡保持
    if os.path.exists(out_path) and SKIP_GENERATION_IF_EXISTS: return
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT]) 
    shape = (4, 64, 64)
    x_T = torch.randn(1, *shape, device="cuda")
    with torch.no_grad(), autocast("cuda"):
        z_enc, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=shape,
                                  unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                  x_T=x_T, verbose=False)
        x_samples = model.decode_first_stage(z_enc)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_path)

def prepare_payload(raw_data):
    CAPACITY_BYTES = 16384 // 8
    if len(raw_data) > CAPACITY_BYTES - 2: 
        raw_data = raw_data[:CAPACITY_BYTES-2]
    length_header = len(raw_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    if len(final_payload) < CAPACITY_BYTES: 
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    return final_payload

def create_gt_bits_file(final_payload, out_gt_path):
    np.save(out_gt_path, np.frombuffer(final_payload, dtype=np.uint8))

class QualityEvaluator:
    def __init__(self):
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').cuda()
        except Exception:
            self.lpips_fn = None
            print("⚠️ LPIPS init failed or skipped.")
        self.init_brisque()

    def init_brisque(self):
        if BRISQUE_AVAILABLE:
            try:
                dummy = torch.rand(1, 3, 512, 512).cuda()
                brisque(dummy, data_range=1.0)
            except Exception as e:
                print(f"⚠️ BRISQUE weights download failed: {e}")

    def calc_lpips(self, p1, p2):
        if self.lpips_fn is None: return 0.0
        try:
            t1 = self._load(p1); t2 = self._load(p2)
            with torch.no_grad(): return self.lpips_fn(t1, t2).item()
        except Exception as e:
            # print(f"LPIPS Error: {e}")
            return 0.0

    def calc_brisque(self, p1):
        if not BRISQUE_AVAILABLE: return 0.0
        try:
            t1 = self._load(p1, norm=False)
            with torch.no_grad(): return brisque(t1, data_range=1.0).item()
        except:
            return 0.0

    def _load(self, p, norm=True):
        img = cv2.imread(p)
        if img is None: raise ValueError(f"Failed to read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512)) / 255.0
        if norm: img = img * 2 - 1
        return torch.tensor(img.transpose(2,0,1)).float().cuda().unsqueeze(0)

def main():
    print(f"🚀 FINAL ABLATION STUDY (4 MODES: Baseline, NoMask, Uniform, Adaptive) 🚀")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subdirs = {m: os.path.join(OUTPUT_DIR, m) for m in MODES}
    subdirs["cover"] = os.path.join(OUTPUT_DIR, "cover") 
    for d in subdirs.values(): os.makedirs(d, exist_ok=True)
    
    prompts = []
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST) as f: lines = [l.strip() for l in f if l.strip()]
        while len(prompts) < TOTAL_GENERATE: prompts.extend(lines)
    prompts = prompts[:TOTAL_GENERATE] if prompts else ["A futuristic city"] * TOTAL_GENERATE

    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    
    acc_results = defaultdict(lambda: defaultdict(list))
    qual_results = defaultdict(list)
    early_stop_stats = {m: {"count": 0, "steps": []} for m in MODES}
    fid_results = {}

    print("\n--- Phase 1: Generating Images ---")
    for i in tqdm(range(TOTAL_GENERATE)):
        session_key = 123456 + i
        payload_path = os.path.join(OUTPUT_DIR, f"p{i}.dat")
        if not os.path.exists(payload_path): 
            with open(payload_path, "wb") as f: f.write(os.urandom(2048))

        cover_path = os.path.join(subdirs["cover"], f"{i:05d}.png")
        generate_cover_image(model, sampler, prompts[i], cover_path, seed=session_key)
        
        with open(payload_path, "rb") as f: raw_data = f.read()
        final_payload = prepare_payload(raw_data)

        for mode in MODES:
            out_p = os.path.join(subdirs[mode], f"{i:05d}.png")
            gt_p = out_p + ".gt_bits.npy"
            
            # [斷點續傳] 只要圖片存在就跳過
            if SKIP_GENERATION_IF_EXISTS and os.path.exists(out_p):
                if not os.path.exists(gt_p):
                    create_gt_bits_file(final_payload, gt_p)
                continue
            
            
            
            #lr, reg = 0.12, 1.25 # 預設參數
            lr, reg = 0.3, 0.8 # 預設參數

            # 如果需要針對不同 mode 微調 LR，可在此設定
           # if mode == "no_mask": lr = 0.04
            
            success, stopped, step_val = generate_alice_image(
                model=model, sampler=sampler, prompt=prompts[i], secret_key=session_key,
                payload_data=final_payload, outpath=out_p, init_latent_path=None,
                opt_iters=15, # 消融測試統一 Iteration
                lr=lr, lambda_reg=reg, mode=mode,
                early_stop_threshold=0.0693
            )
            
            if success:
                early_stop_stats[mode]["steps"].append(step_val)
                if stopped:
                    early_stop_stats[mode]["count"] += 1

            create_gt_bits_file(final_payload, gt_p)

    print("\n--- Clearing VRAM for Evaluation ---")
    del model
    del sampler
    torch.cuda.empty_cache()
    
    # === FID Calculation Block ===
    if CALC_FID:
        print("\n--- Phase 1.5: Calculating FID ---")
        if not os.path.exists(DIR_REAL_COCO):
            print(f"⚠️ Warning: Real COCO path not found at {DIR_REAL_COCO}. Skipping FID.")
        else:
            real_imgs = [f for f in os.listdir(DIR_REAL_COCO) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(real_imgs) == 0:
                print(f"⚠️ Warning: No images found in {DIR_REAL_COCO}. Skipping FID.")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                modes_to_calc = ["cover"] + MODES
                
                for mode in modes_to_calc:
                    gen_path = subdirs[mode]
                    gen_imgs = [f for f in os.listdir(gen_path) if f.lower().endswith('.png')]
                    if len(gen_imgs) < 2:
                        print(f"⚠️ Too few images in {mode} ({len(gen_imgs)}), skipping FID.")
                        fid_results[mode] = -1.0
                        continue
                        
                    print(f"Computing FID for {mode.upper()}...")
                    try:
                        fid_value = fid_score.calculate_fid_given_paths(
                            [DIR_REAL_COCO, gen_path],
                            batch_size=1,  # [修正] 設為 1 以處理 COCO 不同尺寸圖片
                            device=device,
                            dims=2048,
                            num_workers=0  # [修正] 強制單線程
                        )
                        fid_results[mode] = fid_value
                        print(f"  >> FID ({mode}): {fid_value:.4f}")
                    except Exception as e:
                        print(f"  ❌ FID Error ({mode}): {e}")
                        fid_results[mode] = -1.0
                
                torch.cuda.empty_cache()

    print("⏳ Reloading models for Phase 2...")
    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    evaluator = QualityEvaluator()

    print("\n--- Phase 2: Evaluating (Metrics & Attack) ---")
    for i in tqdm(range(TOTAL_EVALUATE)):
        prompt = prompts[i]
        session_key = 123456 + i
        cover_path = os.path.join(subdirs["cover"], f"{i:05d}.png")

        for mode in MODES:
            out_p = os.path.join(subdirs[mode], f"{i:05d}.png")
            gt_p = out_p + ".gt_bits.npy"
            if not os.path.exists(out_p): continue
            
            # Quality Eval
            if os.path.exists(cover_path):
                lpips_val = evaluator.calc_lpips(cover_path, out_p)
                brisque_val = evaluator.calc_brisque(out_p)
                qual_results[mode].append((lpips_val, brisque_val))

            # Acc Eval
            if RUN_ATTACK_AND_BOB:
                try:
                    img_tensor = load_512(out_p).cuda() 
                    for atk_fn, args, atk_name, ext in ATTACK_SUITE:
                        try:
                            temp_name = os.path.join(OUTPUT_DIR, f"temp_attack_{mode}_{i}")
                            atk_fn(img_tensor.clone(), args[0], tmp_image_name=temp_name)
                            final_path = temp_name + ext
                            if os.path.exists(final_path):
                                attacked_tensor = load_512(final_path).cuda()
                                acc = fast_bob_decode(model, sampler, attacked_tensor, prompt, session_key, gt_p)
                                acc_results[mode][atk_name].append(acc)
                                try: os.remove(final_path)
                                except: pass
                            else: acc_results[mode][atk_name].append(0.0)
                        except: acc_results[mode][atk_name].append(0.0)
                    del img_tensor
                except: pass
                torch.cuda.empty_cache()

    print("\n" + "="*140)
    print(f"FINAL ABLATION REPORT (Gen N={TOTAL_GENERATE} | Eval N={TOTAL_EVALUATE})")
    print("="*140)
    
    # 動態產生表頭，支援 4 個模式
    headers = ["Metric"] + [m.upper() for m in MODES]
    header_fmt = "{:<20} | " + " | ".join(["{:<12}"] * len(MODES))
    print(header_fmt.format(*headers))
    print("-" * 140)

    for _, _, atk_name, _ in ATTACK_SUITE:
        row = [f"{atk_name} Acc"]
        vals_list = []
        for mode in MODES:
            vals = acc_results[mode][atk_name]
            avg = np.mean(vals) if vals else 0.0
            vals_list.append(f"{avg:.2f}")
        print(header_fmt.format(row[0], *vals_list))

    print("-" * 140)
    
    l_row = ["LPIPS (vs Cover)"]
    b_row = ["BRISQUE"]
    l_vals, b_vals = [], []
    for mode in MODES:
        vals = qual_results[mode]
        l_avg = np.mean([v[0] for v in vals]) if vals else 0.0
        b_avg = np.mean([v[1] for v in vals]) if vals else 0.0
        l_vals.append(f"{l_avg:.4f}")
        b_vals.append(f"{b_avg:.2f}")
    
    print(header_fmt.format(l_row[0], *l_vals))
    print(header_fmt.format(b_row[0], *b_vals))
    
    # 顯示 FID 結果
    if CALC_FID:
        f_row = ["FID (vs COCO)"]
        f_vals = []
        for mode in MODES:
            val = fid_results.get(mode, -1.0)
            f_vals.append(f"{val:.2f}")
        print("-" * 140)
        print(header_fmt.format(f_row[0], *f_vals))
        cover_fid = fid_results.get("cover", -1.0)
        print(f"(Ref) Cover FID: {cover_fid:.2f}")

    print("="*140)
    print("EARLY STOPPING ANALYSIS")
    print("="*140)
    for mode in MODES:
        count = early_stop_stats[mode]["count"]
        steps = early_stop_stats[mode]["steps"]
        avg_step = sum(steps)/len(steps) if steps else 0
        print(f"{mode.upper():<15} | {f'{count}/{TOTAL_GENERATE}':<15} | {avg_step:.1f}")
    print("="*140)

if __name__ == "__main__":
    main()