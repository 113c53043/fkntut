import os
import sys
import subprocess
import time
import shutil

# === 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
MAS_GRDH_PATH = CURRENT_DIR

# 模型權重路徑
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

# 指定要執行的腳本名稱
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_opt.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob_opt.py")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "pure_algo_opt_test")
PAYLOAD_FILE = os.path.join(OUTPUT_DIR, "random_payload.dat")
STEGO_IMG_PATH = os.path.join(OUTPUT_DIR, "stego_opt.png")

def ensure_paths():
    """建立輸出目錄並生成測試用的 Payload"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 清理舊的輸出檔案
    if os.path.exists(STEGO_IMG_PATH):
        os.remove(STEGO_IMG_PATH)
    
    # 刪除對應的 GT Bits 檔案
    gt_bits_path = STEGO_IMG_PATH + ".gt_bits.npy"
    if os.path.exists(gt_bits_path):
        os.remove(gt_bits_path)

    # 生成隨機 Payload (600 bytes)
    if not os.path.exists(PAYLOAD_FILE):
        with open(PAYLOAD_FILE, "wb") as f:
            f.write(os.urandom(600))
        print(f"📄 Generated Test Payload: 600 bytes")
    else:
        print(f"📄 Using existing Payload: {PAYLOAD_FILE}")

def run_experiment(prompt, session_key):
    print(f"\n--- [Experiment] Algorithm: Optimization-Based (Gradient Descent) ---")
    
    if not os.path.exists(ALICE_SCRIPT):
        print(f"❌ Error: 找不到 Alice 腳本: {ALICE_SCRIPT}")
        return
    if not os.path.exists(BOB_SCRIPT):
        print(f"❌ Error: 找不到 Bob 腳本: {BOB_SCRIPT}")
        return

    # === 1. Alice (優化嵌入) ===
    print(f"\n▶️  Running Alice (Embedding)...")
    
    # 【關鍵修正】在這裡強制指定最新的參數，覆蓋任何預設值
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt,
        "--secret_key", str(session_key),
        "--payload_path", PAYLOAD_FILE,
        "--outpath", STEGO_IMG_PATH,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        # 修正後的參數
        "--opt_iters", "500",     # 增加迭代次數以確保收斂
        "--lr", "0.01",           # 恢復較高的學習率
        "--lambda_img", "5.0",    # 降低畫質懲罰 (允許修改圖片)
        "--lambda_msg", "20.0"    # 提高訊息權重 (強制寫入)
    ]
    
    try:
        process = subprocess.Popen(cmd_alice, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=MAS_GRDH_PATH)
        for line in process.stdout:
            print(f"   [Alice] {line.strip()}")
        process.wait()
        
        if process.returncode != 0:
            print("❌ Alice failed.")
            return

    except Exception as e:
        print(f"❌ Alice Execution Error: {e}")
        return

    # === 2. Bob (提取驗證) ===
    print(f"\n▶️  Running Bob (Extraction)...")
    if not os.path.exists(STEGO_IMG_PATH):
        print(f"❌ Error: Stego image not found at {STEGO_IMG_PATH}")
        return

    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", STEGO_IMG_PATH,
        "--secret_key", str(session_key),
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--prompt", prompt, 
        "--gt_path", PAYLOAD_FILE 
    ]
    
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True)
        print(result_bob.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob Error:\n{e.stderr}")
        print(f"Stdout:\n{e.stdout}")

def main():
    print(f"\n🚀 Optimization-Based Steganography Verification 🚀\n")
    ensure_paths()
    
    prompt = "A high quality photo of a cute corgi running on grass, 4k, detailed"
    session_key = 987654321
    
    run_experiment(prompt, session_key)

if __name__ == "__main__":
    main()