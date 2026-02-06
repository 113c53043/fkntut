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

# === 1. 路徑與環境設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 嘗試載入 SD 模型與相關模組
try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
    from pure_alice_final import generate_alice_image 
except ImportError as e:
    print(f"⚠️ Import Warning: {e}")
    print("請確保你在正確的專案路徑下執行此程式。")

# 載入 BRISQUE
try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    print("⚠️ piq library not found. Installing...")
    os.system("pip install piq")
    from piq import brisque
    BRISQUE_AVAILABLE = True

# === 2. 實驗配置 ===
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

OUTPUT_BASE_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "anime")

DATASETS = {
    "Anime (Flat Regions)": "prompts_anime.txt",
    "Art (Complex Texture)": "prompts_art.txt"
}

TARGET_MODE = "adaptive"  
LR_VAL = 0.3              
REG_SETTINGS = [1.0] # 測試比較
TOTAL_SAMPLES = 1000

LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, nsfw, text, watermark"

# === 3. 內建攻擊與輔助函式 (Self-Contained) ===

def load_tensor(path):
    """讀取圖片並轉為 Tensor [1, 3, 512, 512], Range [0, 1]"""
    img = Image.open(path).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).cuda()

def jpeg_attack_func(img_tensor, qf):
    """
    模擬 JPEG 壓縮攻擊
    Input: Tensor [1, 3, 512, 512] in [0, 1]
    Output: Tensor [1, 3, 512, 512] in [0, 1]
    """
    # Tensor -> Numpy (uint8 0-255)
    img_np = (img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Encode -> Decode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
    result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    
    # Numpy -> Tensor
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    decimg = decimg.astype(np.float32) / 255.0
    res_tensor = torch.from_numpy(decimg.transpose(2, 0, 1)).unsqueeze(0).cuda()
    return res_tensor



def load_shared_model():
    print(f"⏳ Loading SD Model from {CKPT_PATH}...", flush=True)
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

def prepare_payload(size_bytes=2048):
    raw_data = os.urandom(size_bytes)
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

def calc_brisque_score(img_path):
    if not BRISQUE_AVAILABLE: return 0.0
    try:
        img_tensor = load_tensor(img_path)
        with torch.no_grad():
            score = brisque(img_tensor, data_range=1.0).item()
        return score
    except Exception:
        return 0.0

def fast_bob_decode(model, sampler, img_tensor, prompt, secret_key, gt_bits_path):
    try:
        # 確保輸入是 [1, 3, 512, 512] 且範圍 [-1, 1] 給 Model
        # img_tensor 進來是 [0, 1]，轉為 [-1, 1]
        model_input = img_tensor * 2.0 - 1.0
        
        torch.manual_seed(secret_key)
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
        
        with torch.no_grad(), autocast("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(model_input))
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
        # print(f"Decode Error: {e}")
        return 0.0

# === 4. 主程式 ===

def main():
    print(f"🚀 GENERALIZATION & REGULARIZATION STUDY (Self-Contained) 🚀")
    print(f"🔥 Comparing Reg Values: {REG_SETTINGS} | Fixed LR: {LR_VAL}")
    
    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    
    final_report = defaultdict(dict)

    for reg_val in REG_SETTINGS:
        print(f"\n" + "#"*80)
        print(f"🔧 Testing Regularization Weight: {reg_val}")
        print("#"*80)
        
        current_output_dir = os.path.join(OUTPUT_BASE_DIR, f"reg_{reg_val}")
        os.makedirs(current_output_dir, exist_ok=True)

        for domain_name, prompt_file in DATASETS.items():
            domain_short = domain_name.split()[0].lower()
            print(f"\n🌍 Domain: {domain_name} (Reg={reg_val})")
            
            if not os.path.exists(prompt_file):
                print(f"❌ Prompt file not found: {prompt_file}. Skipping.")
                continue
                
            with open(prompt_file, 'r', encoding='utf-8') as f:
                all_prompts = [line.strip() for line in f if line.strip()]
            test_prompts = all_prompts[:TOTAL_SAMPLES]
            
            domain_dir = os.path.join(current_output_dir, domain_short)
            os.makedirs(domain_dir, exist_ok=True)
            
            metrics = {
                "brisque": [],
                "jpeg_50_acc": [],
                "identity_acc": []
            }
            
            pbar = tqdm(total=len(test_prompts), desc=f"{domain_short} (λ={reg_val})")
            
            for i, prompt in enumerate(test_prompts):
                session_key = 99999 + i
                out_p = os.path.join(domain_dir, f"{i:05d}.png")
                gt_p = out_p + ".gt_bits.npy"
                final_payload = prepare_payload()

                # --- 1. Generation ---
                if not os.path.exists(out_p):
                    success, stopped, _ = generate_alice_image(
                        model=model, sampler=sampler, prompt=prompt, secret_key=session_key,
                        payload_data=final_payload, outpath=out_p, init_latent_path=None,
                        opt_iters=15, 
                        lr=LR_VAL, lambda_reg=reg_val, mode=TARGET_MODE,
                        early_stop_threshold=0.0693
                    )
                    if not success:
                        pbar.update(1)
                        continue
                    create_gt_bits_file(final_payload, gt_p)
                else:
                    if not os.path.exists(gt_p):
                        create_gt_bits_file(final_payload, gt_p)

                # --- 2. Quality (BRISQUE) ---
                brisque_score = calc_brisque_score(out_p)
                metrics["brisque"].append(brisque_score)
                
                # --- 3. Robustness (Attacks) ---
                # 這裡改用 try-except block 並且把錯誤印出來，確保不會 silent fail
                try:
                    img_tensor = load_tensor(out_p) # Range [0, 1]
                    
                    # A. Identity
                    acc_id = fast_bob_decode(model, sampler, img_tensor, prompt, session_key, gt_p)
                    metrics["identity_acc"].append(acc_id)
                    
                    # B. JPEG QF=50
                    atk_jpg = jpeg_attack_func(img_tensor, qf=50)
                    acc_jpg = fast_bob_decode(model, sampler, atk_jpg, prompt, session_key, gt_p)
                    metrics["jpeg_50_acc"].append(acc_jpg)

                    # C. Noise 0.1
       
                        
                except Exception as e:
                    # 如果單張圖片失敗，印出錯誤以便除錯，但不中斷迴圈
                    # print(f"⚠️ Error processing {i}: {e}")
                    pass
                
                pbar.update(1)
                
            pbar.close()
            
            # 統計
            final_report[reg_val][domain_name] = {
                "BRISQUE": np.mean(metrics["brisque"]) if metrics["brisque"] else 0.0,
                "JPEG(50) Acc": np.mean(metrics["jpeg_50_acc"]) if metrics["jpeg_50_acc"] else 0.0,
                "Identity Acc": np.mean(metrics["identity_acc"]) if metrics["identity_acc"] else 0.0
            }

    # === 5. Report ===
    print("\n\n" + "="*100)
    print("📊 REGULARIZATION COMPARISON REPORT (0.8 vs 1.0)")
    print("="*100)
    
    header = "{:<22} | {:<8} | {:<12} | {:<12} | {:<12} "
    print(header.format("Domain", "Reg (λ)", "BRISQUE (↓)", "JPEG(50) %",  "Clean %"))
    print("-" * 100)
    
    for domain_name in DATASETS.keys():
        for reg_val in REG_SETTINGS:
            stats = final_report[reg_val].get(domain_name)
            if stats:
                print(header.format(
                    domain_name.split()[0], # Anime / Art
                    f"{reg_val}",
                    f"{stats['BRISQUE']:.2f}",
                    f"{stats['JPEG(50) Acc']:.2f}",
                
                    f"{stats['Identity Acc']:.2f}"
                ))
        print("-" * 100)

if __name__ == "__main__":
    main()