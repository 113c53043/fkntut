import os
import sys
import subprocess
import time
import numpy as np

# === 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
MAS_GRDH_PATH = CURRENT_DIR

CKPT_PATH = "weights/v1-5-pruned.ckpt" 
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

# 【注意】這裡指向新的 v2_uncertainty 版本
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_v2_uncertainty.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py") # Decoder 不變
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "pure_algo_v2_test")

# 測試資料
PAYLOAD_FILE = os.path.join(OUTPUT_DIR, "random_payload.dat")

# 確保生成過程使用相同的隨機性，以利比較
FIXED_SECRET_KEY = 99887766
FIXED_GEN_SEED = FIXED_SECRET_KEY # 將秘密金鑰同時作為生成模型的種子

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(PAYLOAD_FILE):
        os.remove(PAYLOAD_FILE)
        
    # 生成 2048 bytes 的 payload (Alice 程式中會限制實際使用的容量)
    with open(PAYLOAD_FILE, "wb") as f:
        f.write(os.urandom(2048))
    print(f"📄 Generated Test Payload: 2048 bytes (Payload limit: 680 bytes)")

def run_experiment(prompt, session_key, use_uncertainty, exp_name):
    """
    執行單次實驗，根據 use_uncertainty 決定是否啟用 Mask。
    
    :param session_key: 用於編碼 Payload 的秘密金鑰 (也作為隨機種子)
    :param use_uncertainty: True/False 決定是否在 Alice 端啟用不確定性引導
    :param exp_name: 實驗名稱，用於產生輸出檔案名稱
    """
    exp_tag = "Uncertainty-Guided Optimization" if use_uncertainty else "Baseline Optimization (No Mask)"
    file_tag = "with_mask" if use_uncertainty else "baseline"
    
    print(f"\n--- [CL-Stega] Experiment: {exp_name} ({exp_tag}) ---")
    stego_img_path = os.path.join(OUTPUT_DIR, f"stego_{file_tag}.png")
    
    # 1. Alice (Optimization)
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", PAYLOAD_FILE,
        "--outpath", stego_img_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--opt_iters", "15",    # V2 建議稍微增加步數
        "--lr", "0.3",          # 配合 Mask 稍微提高 LR
        "--gen_seed", str(session_key) # 【新增】固定生成模型的種子以確保生成過程一致
    ]
    
    # 根據參數決定是否加入 --use_uncertainty 旗標
    if use_uncertainty:
        cmd_alice.append("--use_uncertainty")     # 【關鍵】啟用不確定性引導旗標
    
    try:
        print(f"⚙️  [Alice] Optimizing Latent Space...")
        process = subprocess.Popen(cmd_alice, cwd=MAS_GRDH_PATH) 
        process.wait()
        
        if process.returncode != 0:
            print(f"❌ Alice ({exp_name}) crashed! Stopping experiment.")
            return

    except Exception as e:
        print(f"❌ Alice ({exp_name}) Execution Error: {e}")
        return

    # 2. Bob (Extraction) - 解碼端保持不變
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", stego_img_path,
        "--prompt", prompt,
        "--secret_key", str(session_key),
        "--gt_path", PAYLOAD_FILE,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH
    ]
    
    try:
        print(f"\n⚙️  [Bob] Extracting from {file_tag} Stego Image...")
        subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH)
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob ({exp_name}) Error: Process returned {e.returncode}")

def main():
    print(f"\n🚀 Pure Algorithm Verification (V2: Uncertainty Aware Comparison) 🚀\n")
    ensure_paths()
    
    prompt = "A futuristic cyberpunk city with neon lights and rain, 8k, highly detailed"
    
    # 使用全域定義的固定金鑰
    session_key = FIXED_SECRET_KEY 
    
    # --- 實驗一：Baseline (不加 Mask) ---
    # 圖像輸出到: outputs/pure_algo_v2_test/stego_baseline.png
    run_experiment(prompt, session_key, use_uncertainty=False, exp_name="1. Baseline")
    
    # --- 實驗二：V2 (加 Mask) ---
    # 圖像輸出到: outputs/pure_algo_v2_test/stego_with_mask.png
    run_experiment(prompt, session_key, use_uncertainty=True, exp_name="2. V2_WithMask")

if __name__ == "__main__":
    main()