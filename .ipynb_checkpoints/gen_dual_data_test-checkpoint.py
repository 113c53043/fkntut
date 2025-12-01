import os
import sys
import subprocess
import time
import numpy as np

# === 全域路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
MAS_GRDH_PATH = CURRENT_DIR
TEXT_MODULE_PATH = os.path.join(CURRENT_DIR, 'text_stego_module')

# 【配置】請確認權重與設定檔路徑
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
GPT2_PATH = os.path.join(MAS_GRDH_PATH, "gpt2") 
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

# 指向測試版腳本

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "alice_gen_test.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "bob_extract_test.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "batch_test")

# Ground Truth 存放路徑
GT_PATH = os.path.join(OUTPUT_DIR, "secure_sensor_backup.dat")

# 加入路徑以導入模組
sys.path.append(MAS_GRDH_PATH)
try:
    from text_stego_module.stego import TextStegoSystem
    print("✅ [System] 文本模組載入成功")
except ImportError:
    print(f"❌ [System] 找不到文本模組 (text_stego_module)，請確認路徑。")
    sys.exit(1)

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(GT_PATH): os.remove(GT_PATH)
    
    if not os.path.exists(PROMPT_FILE_LIST):
        return ["A futuristic hospital room with high tech equipment"]
    with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def run_simulation(text_sys, prompt, session_key, receiver_id, idx):
    print(f"\n--- [Simulation #{idx:03d}] Capture & Embed ---")
    stego_img_path = os.path.join(OUTPUT_DIR, f"exp_{idx:03d}.png")
    
    # === [Step 0] 身份綁定 ===
    bound_key = session_key ^ receiver_id 
    print(f"🔒 [Signaling] 生成綁定金鑰: {bound_key}")

    # === [Step 1] Text Channel ===
    try:
        stego_prompt_text, generated_ids = text_sys.alice_encode(prompt, bound_key)
    except Exception as e:
        print(f"❌ 文本編碼失敗: {e}")
        return False

    # === [Step 2] Alice (Sensor) ===
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--mode", "capture_and_embed",
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--outpath", stego_img_path,
        "--verification_path", GT_PATH,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH
    ]
    
    try:
        # 執行 Alice
        result_alice = subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True)
        for line in result_alice.stdout.split('\n'):
            if "[Sensor]" in line or "[Secure]" in line:
                print(f"  {line}")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ 感測器擷取失敗 (Alice Crashed):")
        # 【關鍵修正】印出完整的 STDOUT 和 STDERR 幫助除錯
        print("="*20 + " ALICE STDOUT " + "="*20)
        print(e.stdout)
        print("="*20 + " ALICE STDERR " + "="*20)
        print(e.stderr)
        print("="*50)
        return False

    # === [Step 3] Bob: 接收 ===
    try:
        extracted_bound_key = text_sys.bob_decode(generated_ids)
    except Exception as e:
        print(f"❌ 文本解碼失敗: {e}")
        return False
        
    extracted_session_key = extracted_bound_key ^ receiver_id
    
    if extracted_session_key != session_key:
        print(f"❌ 金鑰解綁失敗")
        return False

    # 執行 Bob
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
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob 還原失敗:\n{e.stderr}")
        print("="*20 + " BOB STDOUT " + "="*20)
        print(e.stdout)
        return False

    if "🎉 雙層驗證成功" in result_bob.stdout:
        print(f"✅ [Verify] 醫療影像無損還原成功 (Source-Encrypted)")
        return True
    else:
        print(result_bob.stdout)
        return False

def main():
    print(f"\n🚀 源端加密隱寫採集系統 (Source-Encrypted Acquisition System) - Debug Mode 🚀\n")
    
    prompts = ensure_paths()
    if not os.path.exists(GPT2_PATH): 
        print(f"❌ 找不到 GPT2 模型: {GPT2_PATH}")
        sys.exit(1)
        
    text_sys = TextStegoSystem(model_name=GPT2_PATH)
    PHYSICIAN_ID = 95279527 
    
    session_key = int(np.random.randint(10000000, 99999999))
    run_simulation(text_sys, prompts[0], session_key, PHYSICIAN_ID, 1)

if __name__ == "__main__":
    main()