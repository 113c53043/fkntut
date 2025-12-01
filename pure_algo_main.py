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

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "pure_algo_test")

# 測試資料
PAYLOAD_FILE = os.path.join(OUTPUT_DIR, "random_payload.dat")

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 清理舊檔
    if os.path.exists(PAYLOAD_FILE):
        os.remove(PAYLOAD_FILE)
        
    # 生成 600 bytes 的 payload
    with open(PAYLOAD_FILE, "wb") as f:
        f.write(os.urandom(2048))
    print(f"📄 Generated Test Payload: 600 bytes")

def run_experiment(prompt, session_key):
    print(f"\n--- [Experiment] Algorithm: Test-Time Latent Optimization ---")
    stego_img_path = os.path.join(OUTPUT_DIR, "stego.png")
    
    # 1. Alice (Optimization)
    # 【修正】移除 "--signal_strength" 參數，因為新版 Alice 使用正交映射，不需要此參數
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", PAYLOAD_FILE,
        "--outpath", stego_img_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--opt_iters", "10",
        "--lr", "0.5"
    ]
    
    try:
        print("⚙️  [Alice] Optimizing Latent Space...")
        # 讓 Alice 的輸出直接顯示在螢幕上，不要過濾錯誤
        process = subprocess.Popen(cmd_alice, cwd=MAS_GRDH_PATH) 
        process.wait()
        
        if process.returncode != 0:
            print("❌ Alice crashed! Stopping experiment.")
            return

    except Exception as e:
        print(f"❌ Alice Execution Error: {e}")
        return

    # 2. Bob (Extraction)
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
        print("\n⚙️  [Bob] Extracting...")
        subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH)
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob Error: Process returned {e.returncode}")

def main():
    print(f"\n🚀 Pure Algorithm Verification 🚀\n")
    ensure_paths()
    
    prompt = "A Blue car At the forest, 8k, detailed"
    session_key = 12345678
    
    run_experiment(prompt, session_key)

if __name__ == "__main__":
    main()