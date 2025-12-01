import os
import sys
import torch
import subprocess
import time
import numpy as np
import hashlib

# === 全域路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
MAS_GRDH_PATH = CURRENT_DIR
TEXT_MODULE_PATH = os.path.join(CURRENT_DIR, 'text_stego_module')

# 【路徑修正】請確認這些路徑與您的環境一致
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
GPT2_PATH = os.path.join(MAS_GRDH_PATH, "gpt2") 
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "alice_gen.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "bob_extract.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "batch_test")

# 【關鍵修改】指定要讀取的外部檔案 (請確保此檔案在根目錄)
PAYLOAD_FILE = "test.dcm" 

# 加入模組路徑
sys.path.append(MAS_GRDH_PATH)
try:
    from text_stego_module.stego import TextStegoSystem
    print("✅ [System] 文本模組載入成功")
except ImportError:
    print(f"❌ [System] 找不到文本模組 (text_stego_module)，請確認目錄結構。")
    sys.exit(1)

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(PROMPT_FILE_LIST):
        print(f"⚠️ 警告：找不到測試 Prompt 文件，將使用預設 prompts...")
        return ["A futuristic city with flying cars"]
    with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def run_single_experiment(text_sys, prompt, session_key, receiver_id, idx):
    print(f"\n--- [Experiment #{idx:03d}] Session Key: {session_key} ---")
    
    stego_img_path = os.path.join(OUTPUT_DIR, f"exp_{idx:03d}.png")
    
    # === [Step 0] 身分綁定金鑰封裝 (Identity-Bound Key Encapsulation) ===
    bound_key = session_key ^ receiver_id 
    print(f"🔒 [Security] 執行身分綁定: Key({session_key}) XOR ID({receiver_id}) -> BoundKey({bound_key})")

    # === [Step 1] Alice: 文本隱寫 (傳輸 Bound Key) ===
    try:
        stego_prompt_text, generated_ids = text_sys.alice_encode(prompt, bound_key)
    except Exception as e:
        print(f"❌ [Alice] 文本編碼失敗: {e}")
        return False, 0.0

    # === [Step 2] Alice: 圖像隱寫 (傳輸加密後的 test.dcm) ===
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key), 
        "--payload_path", PAYLOAD_FILE,   # 傳入真實檔案路徑
        "--outpath", stego_img_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    try:
        result_alice = subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"❌ Alice 圖像生成失敗:\n{e.stderr}")
        return False, 0.0

    # === [Step 3] Bob: 文本提取 (解開 Bound Key) ===
    try:
        extracted_bound_key = text_sys.bob_decode(generated_ids)
    except Exception as e:
        print(f"❌ [Bob] 文本解碼失敗: {e}")
        return False, 0.0
        
    # Bob 使用自己的 ID 解開綁定
    extracted_session_key = extracted_bound_key ^ receiver_id
    
    if extracted_session_key != session_key:
        print(f"❌ 金鑰解綁失敗 (Exp: {session_key}, Got: {extracted_session_key})")
        return False, 0.0
    print(f"✅ [Security] 身分驗證成功，解綁金鑰: {extracted_session_key}")

    # === [Step 4] Bob: 圖像提取與解密 (Zero-Error Verification) ===
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", stego_img_path,
        "--prompt", stego_prompt_text,
        "--secret_key", str(extracted_session_key),
        "--gt_path", PAYLOAD_FILE,        # 【關鍵修正】傳入 GT 路徑給 Bob 進行比對
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob 圖像提取失敗:\n{e.stderr}")
        return True, 0.0

    # 解析 Bob 的標準輸出尋找成功訊號
    ecc_success = "🎉 雙層驗證成功" in result_bob.stdout
    
    if ecc_success:
        print(f"✅ 實驗成功！醫療檔案無損還原 (AES + ECC)")
    else:
        print("⚠️ ECC 或 AES 解密失敗。")
        print(result_bob.stdout)

    return True, (100.0 if ecc_success else 0.0)

def main():
    num_runs = 1
    if len(sys.argv) > 1:
        num_runs = int(sys.argv[1])
            
    print(f"\n🚀 雙層防禦隱寫系統 (Dual-Layer Defense) - Real File Mode 🚀\n")

    prompts = ensure_paths()
    
    # 檢查 Payload 檔案是否存在
    if not os.path.exists(PAYLOAD_FILE):
        print(f"❌ 錯誤：找不到輸入檔案 '{PAYLOAD_FILE}'")
        print(f"👉 請將您的 test.dcm 放入: {MAS_GRDH_PATH}")
        sys.exit(1)
    else:
        f_size = os.path.getsize(PAYLOAD_FILE)
        print(f"📄 偵測到 Payload: {PAYLOAD_FILE} ({f_size} bytes)")
        if f_size > 236:
            print(f"⚠️  警告：檔案超過 236 bytes，傳輸時將會被截斷！")

    if not os.path.exists(GPT2_PATH):
        print(f"❌ [System] 找不到 GPT-2 模型: {GPT2_PATH}")
        sys.exit(1)
        
    text_sys = TextStegoSystem(model_name=GPT2_PATH)
    
    # 模擬醫生 ID (只有收發雙方知道)
    PHYSICIAN_ID = 95279527 
    
    results = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        # 每次會話隨機生成 Session Key
        session_key = int(np.random.randint(10000000, 99999999))
        
        try:
            text_success, ecc_success = run_single_experiment(text_sys, prompt, session_key, PHYSICIAN_ID, i+1)
            results.append((text_success, ecc_success))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((False, 0.0))

    # 簡單統計
    success_cnt = sum(1 for r in results if r[1] == 100.0)
    print(f"\n📊 最終成功率: {(success_cnt/len(results))*100:.2f}% ({success_cnt}/{len(results)})")

if __name__ == "__main__":
    main()