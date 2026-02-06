import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from torch import autocast
import cv2
import json  # 用於儲存斷點紀錄

# === 1. 路徑與環境設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from pure_alice_final import generate_alice_image 
    from mapping_module import ours_mapping
    # 嘗試匯入 Robust Eval 工具
    try:
        import lpips
        from piq import brisque
        from robust_eval import identity, jpeg, resize, mblur, gblur, awgn
        from utils import load_512
        METRICS_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Metrics/Eval modules missing. Only generation will run.")
        METRICS_AVAILABLE = False
    
    # 嘗試匯入 pytorch_fid
    try:
        from pytorch_fid import fid_score
        FID_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: pytorch-fid module missing. FID calculation will be skipped.")
        FID_AVAILABLE = False

except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    sys.exit(1)

# === 2. 實驗參數配置 ===
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")
DIR_REAL_COCO = os.path.join(MAS_GRDH_PATH, "scripts", "coco_val2017")

# 基礎輸出路徑
BASE_OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "mask_test_earlystop5")

# [設定] 消融實驗 N=2000 
TOTAL_TEST = 2000
SKIP_IF_EXISTS = True
CALC_FID = False

# [核心] 比較模式
MODES = ["no_mask", "adaptive"]

# [核心] 參數設定列表
TEST_CONFIGS = [
    {"lr": 0.3, "reg": 0.8}, # 推薦參數
]

# 通用設定
MAX_ITERS = 15
EARLY_STOP_THRESHOLD = 0.0693

ATTACK_SUITE = [
    (identity, [None], "Identity", ".png"),
    (jpeg, [50], "JPEG(50)", ".jpg"),      
    (resize, [0.5], "Resize(0.5)", ".png"),
    (awgn, [0.05], "Noise(0.05)", ".png")
]

LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# === 3. 輔助類別與函式 ===

class QualityEvaluator:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net='alex').cuda() if METRICS_AVAILABLE else None
    
    def calc_metrics(self, cover_path, stego_path):
        if not METRICS_AVAILABLE: return 0.0, 0.0
        try:
            cover = self._load(cover_path)
            stego = self._load(stego_path)
            with torch.no_grad():
                lpips_score = self.lpips_fn(cover, stego).item()
            stego_norm = (stego + 1.0) / 2.0
            brisque_score = brisque(stego_norm, data_range=1.0).item()
            return lpips_score, brisque_score
        except Exception as e:
            return 0.0, 0.0

    def _load(self, p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512)) / 255.0
        img = img * 2 - 1 
        return torch.tensor(img.transpose(2,0,1)).float().cuda().unsqueeze(0)

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

def generate_cover_image(model, sampler, prompt, out_path, seed):
    if os.path.exists(out_path) and SKIP_IF_EXISTS: return
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

def fast_bob_decode(model, sampler, img_tensor, prompt, secret_key, gt_bits_path):
    if not os.path.exists(gt_bits_path): return 0.0
    try:
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
        gt_bytes = np.load(gt_bits_path).tobytes()
        arr_a = np.unpackbits(np.frombuffer(extracted_bytes, dtype=np.uint8))
        arr_b = np.unpackbits(np.frombuffer(gt_bytes, dtype=np.uint8))
        min_len = min(len(arr_a), len(arr_b))
        matches = np.sum(arr_a[:min_len] == arr_b[:min_len])
        return (matches / max(len(arr_a), len(arr_b))) * 100.0
    except:
        return 0.0

def prepare_payload(raw_data):
    CAPACITY_BYTES = 16384 // 8
    if len(raw_data) > CAPACITY_BYTES - 2: raw_data = raw_data[:CAPACITY_BYTES-2]
    length_header = len(raw_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    return final_payload

# === 4. 主程式 ===

def main():
    print(f"\n🚀 MASK ABLATION STUDY BATCH RUN (N={TOTAL_TEST}, Resumable) 🚀")
    print(f"FID Calculation Enabled: {CALC_FID} (FID Available: {FID_AVAILABLE})")
    
    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    evaluator = QualityEvaluator() if METRICS_AVAILABLE else None
    
    prompts = []
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST) as f: lines = [l.strip() for l in f if l.strip()]
        while len(prompts) < TOTAL_TEST: prompts.extend(lines)
    prompts = prompts[:TOTAL_TEST]

    # === 迴圈執行不同參數配置 ===
    for config in TEST_CONFIGS:
        lr = config["lr"]
        reg = config["reg"]
        
        subdir_name = f"lr{lr}_reg{reg}"
        config_out_dir = os.path.join(BASE_OUTPUT_DIR, subdir_name)
        os.makedirs(config_out_dir, exist_ok=True)
        
        # 載入斷點紀錄 (JSON)
        eval_log_path = os.path.join(config_out_dir, "eval_log.json")
        if os.path.exists(eval_log_path):
            try:
                with open(eval_log_path, 'r') as f:
                    eval_cache = json.load(f)
                print(f"    Loaded {len(eval_cache)} records from cache.")
            except:
                eval_cache = {}
        else:
            eval_cache = {}
        
        subdirs = {m: os.path.join(config_out_dir, m) for m in MODES}
        subdirs["cover"] = os.path.join(config_out_dir, "cover")
        for d in subdirs.values(): os.makedirs(d, exist_ok=True)
        
        print(f"\n>>> Running Config: LR={lr}, REG={reg}")
        
        results = defaultdict(lambda: defaultdict(list))
        # [新增] 統計早停次數與步數
        early_stop_stats = defaultdict(lambda: {"total_steps": 0, "stopped_count": 0, "n": 0})
        fid_results = {}

        for i in tqdm(range(TOTAL_TEST), desc=f"Processing"):
            session_key = 123456 + i
            prompt = prompts[i]
            
            raw_data = os.urandom(2048)
            final_payload = prepare_payload(raw_data)
            
            # 1. Cover
            cover_path = os.path.join(subdirs["cover"], f"{i:05d}.png")
            generate_cover_image(model, sampler, prompt, cover_path, session_key)

            for mode in MODES:
                out_p = os.path.join(subdirs[mode], f"{i:05d}.png")
                gt_p = out_p + ".gt_bits.npy"
                
                # 2. Stego Generation (Skip if exists)
                if not (SKIP_IF_EXISTS and os.path.exists(out_p)):
                    # [修改] 捕獲回傳值以確認是否早停
                    # 假設 generate_alice_image 回傳: (success, stopped, steps)
                    # 如果你的函式只回傳圖片，請自行調整這裡
                    ret_val = generate_alice_image(
                        model=model, sampler=sampler, prompt=prompt, secret_key=session_key,
                        payload_data=final_payload, outpath=out_p, init_latent_path=None,
                        opt_iters=MAX_ITERS, lr=lr, lambda_reg=reg, mode=mode,
                        early_stop_threshold=EARLY_STOP_THRESHOLD
                    )
                    
                    # 處理回傳值 (相容性檢查)
                    steps_taken = MAX_ITERS
                    is_stopped = False
                    
                    if isinstance(ret_val, tuple):
                        # 嘗試解析 (success, stopped, steps)
                        if len(ret_val) >= 3:
                            is_stopped = ret_val[1]
                            steps_taken = ret_val[2]
                        elif len(ret_val) == 2:
                             # 假設是 (success, stopped)
                             is_stopped = ret_val[1]
                    
                    # 寫入統計
                    early_stop_stats[mode]["total_steps"] += steps_taken
                    early_stop_stats[mode]["stopped_count"] += (1 if is_stopped else 0)
                    early_stop_stats[mode]["n"] += 1
                    
                    np.save(gt_p, np.frombuffer(final_payload, dtype=np.uint8))
                elif not os.path.exists(gt_p):
                    np.save(gt_p, np.frombuffer(final_payload, dtype=np.uint8))

                # 3. Evaluation (Resume Logic)
                cache_key = f"{i}_{mode}"
                
                if cache_key in eval_cache:
                    # [命中] 直接從 Cache 讀取數據
                    cached_data = eval_cache[cache_key]
                    results[mode]["LPIPS"].append(cached_data.get("LPIPS", 0.0))
                    results[mode]["BRISQUE"].append(cached_data.get("BRISQUE", 0.0))
                    for atk_name in [x[2] for x in ATTACK_SUITE]:
                        if atk_name in cached_data:
                            results[mode][atk_name].append(cached_data[atk_name])
                else:
                    # [未命中] 執行計算
                    current_metrics = {}
                    if METRICS_AVAILABLE and os.path.exists(out_p):
                        # Quality
                        lpips_val, brisque_val = evaluator.calc_metrics(cover_path, out_p)
                        results[mode]["LPIPS"].append(lpips_val)
                        results[mode]["BRISQUE"].append(brisque_val)
                        current_metrics["LPIPS"] = lpips_val
                        current_metrics["BRISQUE"] = brisque_val
                        
                        # Acc
                        img_tensor = load_512(out_p).cuda()
                        for atk_fn, args, atk_name, ext in ATTACK_SUITE:
                            try:
                                temp_name = os.path.join(config_out_dir, f"temp_{mode}_{i}")
                                atk_fn(img_tensor.clone(), args[0], tmp_image_name=temp_name)
                                final_path = temp_name + ext
                                if os.path.exists(final_path):
                                    atk_tensor = load_512(final_path).cuda()
                                    acc = fast_bob_decode(model, sampler, atk_tensor, prompt, session_key, gt_p)
                                    results[mode][atk_name].append(acc)
                                    current_metrics[atk_name] = acc
                                    os.remove(final_path)
                                else:
                                    results[mode][atk_name].append(0.0)
                                    current_metrics[atk_name] = 0.0
                            except:
                                results[mode][atk_name].append(0.0)
                                current_metrics[atk_name] = 0.0
                        
                        # [儲存] 寫入 Cache 並即時存檔
                        eval_cache[cache_key] = current_metrics
                        with open(eval_log_path, 'w') as f:
                            json.dump(eval_cache, f, indent=4)
        
        # === 4. FID 計算 ===
        if CALC_FID and FID_AVAILABLE:
            if not os.path.exists(DIR_REAL_COCO):
                print(f"⚠️ Warning: Real COCO path not found. Skipping FID.")
            else:
                real_imgs = [f for f in os.listdir(DIR_REAL_COCO) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(real_imgs) > 0:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Cover FID
                    if "cover" not in fid_results:
                         print(f"Computing FID for COVER (batch_size=1)...")
                         try:
                            fid_value = fid_score.calculate_fid_given_paths(
                                [DIR_REAL_COCO, subdirs["cover"]],
                                1, device, 2048, 0 
                            )
                            fid_results["cover"] = fid_value
                         except Exception as e:
                            print(f"  ❌ FID Error (cover): {e}")

                    # Mode FID
                    for mode in MODES:
                        gen_path = subdirs[mode]
                        gen_imgs = [f for f in os.listdir(gen_path) if f.lower().endswith('.png')]
                        if len(gen_imgs) < 2: continue
                        print(f"Computing FID for {mode.upper()} (batch_size=1)...")
                        try:
                            fid_value = fid_score.calculate_fid_given_paths(
                                [DIR_REAL_COCO, gen_path],
                                1, device, 2048, 0 
                            )
                            fid_results[mode] = fid_value
                            print(f"  >> FID ({mode}): {fid_value:.4f}")
                        except Exception as e:
                            print(f"  ❌ FID Error ({mode}): {e}")
                            fid_results[mode] = -1.0
                    torch.cuda.empty_cache()

        # === 5. 輸出報表 ===
        print("\n" + "="*100)
        print(f"MASK ABLATION REPORT (LR={lr}, REG={reg}, N={TOTAL_TEST})")
        print("="*100)
        
        header = "{:<15} | {:<15} | {:<15} | {:<15}".format("Metric", "No Mask", "Adaptive (Ours)", "Diff")
        print(header)
        print("-" * 100)
        
        metrics_order = ["Identity", "JPEG(50)", "Resize(0.5)", "Noise(0.05)", "LPIPS", "BRISQUE"]
        if CALC_FID and FID_AVAILABLE: metrics_order.append("FID")

        for met in metrics_order:
            if met == "FID":
                 val_no = fid_results.get("no_mask", 0.0)
                 val_ad = fid_results.get("adaptive", 0.0)
            else:
                 val_no = np.mean(results["no_mask"][met]) if results["no_mask"][met] else 0.0
                 val_ad = np.mean(results["adaptive"][met]) if results["adaptive"][met] else 0.0
            
            diff = val_ad - val_no
            
            if met in ["LPIPS"]:
                row = "{:<15} | {:<15.4f} | {:<15.4f} | {:<+15.4f}".format(met, val_no, val_ad, diff)
            else: 
                row = "{:<15} | {:<15.2f} | {:<15.2f} | {:<+15.2f}".format(met, val_no, val_ad, diff)
                
            print(row)
        
        if CALC_FID and FID_AVAILABLE:
            cover_fid = fid_results.get("cover", 0.0)
            print("-" * 100)
            print(f"(Ref) Cover FID: {cover_fid:.2f}")

        # === [新增] 早停分析報表 ===
        print("-" * 100)
        print("EARLY STOPPING ANALYSIS (Run-time Stats)")
        print("(Note: Only counts newly generated images in this run)")
        print("-" * 100)
        print("{:<15} | {:<15} | {:<15}".format("Mode", "Avg Steps", "Stop Rate"))
        print("-" * 100)
        for mode in MODES:
            stats = early_stop_stats[mode]
            if stats["n"] > 0:
                avg_steps = stats["total_steps"] / stats["n"]
                stop_rate = (stats["stopped_count"] / stats["n"]) * 100
                print("{:<15} | {:<15.2f} | {:<15.2f}%".format(mode, avg_steps, stop_rate))
            else:
                print("{:<15} | {:<15} | {:<15}".format(mode, "N/A (Skipped)", "N/A"))
        print("="*100 + "\n")

if __name__ == "__main__":
    main()