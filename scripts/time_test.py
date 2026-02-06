import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from torch import autocast
import time  # [Added] For timing measurements

# === 1. 路徑與環境設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    # 假設這是你的核心生成函式
    from pure_alice_final import generate_alice_image
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    sys.exit(1)

# === 2. 實驗參數配置 ===
# 請根據你的環境修改這些路徑
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "efficiency_test_n500")

# [審查委員標準] N=500 足以證明效率統計顯著性
TOTAL_TEST = 200
MAX_ITERS = 20
EARLY_STOP_THRESHOLD = 0.0693  # 根據論文設定

# 設定兩種測試模式
CONFIGS = [
    {"name": "Fixed (No Stop)", "threshold": -1.0},      # 閾值設為負數，強制不早停
    {"name": "Early Stopping",  "threshold": EARLY_STOP_THRESHOLD}
]

LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# === 3. 輔助函式 ===

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

def prepare_payload(raw_data):
    # 模擬 16k bits payload
    CAPACITY_BYTES = 16384 // 8
    if len(raw_data) > CAPACITY_BYTES - 2:
        raw_data = raw_data[:CAPACITY_BYTES-2]
    length_header = len(raw_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    if len(final_payload) < CAPACITY_BYTES:
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    return final_payload

def generate_cover_image(model, sampler, prompt, seed):
    # 簡單生成 Cover 用於編碼，不需儲存到硬碟
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
    shape = (4, 64, 64)
    x_T = torch.randn(1, *shape, device="cuda")
    
    # 這裡只需要 latent 即可，不用 decode 出來，節省時間
    # 如果你的 generate_alice_image 需要 RGB 輸入，請保留 decode
    # 假設 generate_alice_image 內部處理一切
    return None 

# === 4. 主程式 ===

def main():
    print(f"\n🚀 EFFICIENCY ABLATION STUDY (N={TOTAL_TEST}) 🚀")
    print(f"Comparing: Fixed {MAX_ITERS} Steps vs. Early Stopping (Thr={EARLY_STOP_THRESHOLD})\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 載入 Prompt
    prompts = []
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST) as f: lines = [l.strip() for l in f if l.strip()]
        while len(prompts) < TOTAL_TEST: prompts.extend(lines)
    prompts = prompts[:TOTAL_TEST] if prompts else ["A futuristic city"] * TOTAL_TEST

    # 載入模型
    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    
    # 統計數據容器
    stats = {
        "Fixed (No Stop)": {"steps": [], "times": []},
        "Early Stopping":  {"steps": [], "times": []}
    }

    print("\n--- Starting Benchmark ---")
    
    # 預熱 GPU (Warmup) - 避免第一次執行時間不準
    print("🔥 Warming up GPU...", end="\r")
    dummy_payload = prepare_payload(os.urandom(2048))
    for _ in range(2):
        generate_alice_image(
            model=model, sampler=sampler, prompt="warmup", secret_key=999,
            payload_data=dummy_payload, outpath=None, init_latent_path=None,
            opt_iters=5, lr=0.2, lambda_reg=0.3, mode="adaptive",
            early_stop_threshold=-1.0
        )
    print("🔥 GPU Warmup Done.       ")

    for i in tqdm(range(TOTAL_TEST)):
        session_key = 123456 + i
        prompt = prompts[i]
        
        # 準備 Payload
        raw_data = os.urandom(2048)
        final_payload = prepare_payload(raw_data)
        
        # 為了公平比較，兩個模式必須使用相同的 Seed 和 Payload
        for config in CONFIGS:
            mode_name = config["name"]
            threshold = config["threshold"]
            
            # 設定輸出路徑 (可選，若不需要存圖可設為 None 以加速)
            # 這裡設為 None 以追求最純粹的演算法時間，排除 I/O 干擾
            out_p = None 
            
            # [計時開始]
            torch.cuda.synchronize() # 確保 GPU 同步
            start_time = time.time()
            
            try:
                # 呼叫核心函式
                # 注意：這裡強制使用 'adaptive' mode，只改變 threshold
                success, stopped, step_val = generate_alice_image(
                    model=model, sampler=sampler, prompt=prompt, secret_key=session_key,
                    payload_data=final_payload, outpath=out_p, init_latent_path=None,
                    opt_iters=MAX_ITERS,
                    lr=0.2, lambda_reg=0.8, # 使用你推薦的 Reg 0.8
                    mode="adaptive",
                    early_stop_threshold=threshold
                )
            except Exception as e:
                print(f"Error at idx {i}: {e}")
                step_val = MAX_ITERS # 出錯視為跑滿
                
            # [計時結束]
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            
            # 記錄數據
            stats[mode_name]["steps"].append(step_val)
            stats[mode_name]["times"].append(elapsed)

    # === 5. 產生報告 ===
    print("\n" + "="*80)
    print(f"EFFICIENCY ANALYSIS REPORT (N={TOTAL_TEST})")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    header = "{:<20} | {:<15} | {:<15} | {:<10}".format("Configuration", "Avg. Iterations", "Avg. Time (s)", "Speedup")
    print(header)
    print("-" * 80)
    
    # 計算 Fixed 的基準值
    fixed_steps = np.mean(stats["Fixed (No Stop)"]["steps"])
    fixed_time = np.mean(stats["Fixed (No Stop)"]["times"])
    
    for cfg in CONFIGS:
        name = cfg["name"]
        avg_steps = np.mean(stats[name]["steps"])
        avg_time = np.mean(stats[name]["times"])
        
        # 計算加速比
        speedup_steps = fixed_steps / avg_steps if avg_steps > 0 else 0
        speedup_time = fixed_time / avg_time if avg_time > 0 else 0
        
        row = "{:<20} | {:<15.2f} | {:<15.4f} | x{:<10.2f}".format(
            name, avg_steps, avg_time, speedup_time
        )
        print(row)
        
    print("="*80)
    
    # 額外統計：Early Stop 觸發率
    es_counts = [1 for s in stats["Early Stopping"]["steps"] if s < MAX_ITERS]
    trigger_rate = (len(es_counts) / TOTAL_TEST) * 100
    print(f"Early Stopping Trigger Rate: {trigger_rate:.2f}% (Threshold < {EARLY_STOP_THRESHOLD})")
    print("="*80)

if __name__ == "__main__":
    main()