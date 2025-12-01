import os
import sys
import subprocess
import time
import numpy as np
import json

# === 全域路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
MAS_GRDH_PATH = CURRENT_DIR

# 【配置】
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
GPT2_PATH = os.path.join(MAS_GRDH_PATH, "gpt2") 

# 指向新版腳本
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "optimization_alice.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "optimization_bob.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "high_capacity_test")

# 測試資料路徑
PAYLOAD_FILE = "large_wallet_backup.dat"
GT_PATH = os.path.join(OUTPUT_DIR, "gt_backup.dat")

sys.path.append(MAS_GRDH_PATH)
try:
    # 這裡我們假設 utils 裡面有 TextSystem
    from optimization_utils import TextStegoSystem, create_high_capacity_payload
    print("✅ [System] 模組載入成功")
except ImportError as e:
    print(f"❌ [System] 載入失敗: {e}")
    sys.exit(1)

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(GT_PATH): os.remove(GT_PATH)
    # 產生一個「高容量」的測試檔案
    create_high_capacity_payload(PAYLOAD_FILE)

def run_simulation(text_sys, prompt, session_key, receiver_id, idx):
    print(f"\n--- [Simulation #{idx:03d}] High-Capacity Cold Storage ---")
    stego_img_path = os.path.join(OUTPUT_DIR, f"vault_{idx:03d}.png")
    
    # 1. 身分綁定 (信令)
    bound_key = session_key ^ receiver_id 
    print(f"🔒 [Signaling] 綁定金鑰生成... Done.")

    # 2. Text Channel
    try:
        stego_prompt_text, generated_ids = text_sys.alice_encode(prompt, bound_key)
    except Exception as e:
        print(f"❌ Text Error: {e}")
        return False

    # 3. Alice (Optimization Mode)
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--payload_path", PAYLOAD_FILE,
        "--outpath", stego_img_path,
        "--verification_path", GT_PATH,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--opt_iters", "10" # 設定優化迭代次數
    ]
    
    try:
        print(f"⚙️  [Alice] 啟動潛在空間最佳化 (Latent Optimization)...")
        # 為了看清楚進度，我們讓它實時輸出
        process = subprocess.Popen(cmd_alice, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=MAS_GRDH_PATH)
        
        for line in process.stdout:
            line = line.strip()
            if "[Optimizer]" in line or "Loss" in line or "[Secure]" in line:
                print(f"  {line}")
        
        process.wait()
        if process.returncode != 0:
            print("❌ Alice Crashed.")
            return False
            
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return False

    # 4. Bob (Fast Recovery)
    # 解綁金鑰
    try:
        extracted_bound_key = text_sys.bob_decode(generated_ids)
        extracted_session_key = extracted_bound_key ^ receiver_id
    except:
        return False

    if extracted_session_key != session_key:
        print("❌ Key Mismatch")
        return False

    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", stego_img_path,
        "--prompt", stego_prompt_text,
        "--secret_key", str(extracted_session_key),
        "--gt_path", GT_PATH,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH
    ]
    
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True)
        print(result_bob.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob Error:\n{e.stderr}")
        return False

    return True

def main():
    print(f"\n🚀 演算法增強版：高容量零錯誤冷儲存系統 🚀\n")
    ensure_paths()
    
    if not os.path.exists(GPT2_PATH): sys.exit(1)
    text_sys = TextStegoSystem(model_name=GPT2_PATH)
    
    USER_ID = 95279527
    session_key = int(np.random.randint(10000000, 99999999))
    
    # 提示：這個 Prompt 將引導生成
    prompt = "A highly detailed oil painting of a cyberpunk city, neon lights, rain"
    
    run_simulation(text_sys, prompt, session_key, USER_ID, 1)

if __name__ == "__main__":
    main()