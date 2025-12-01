import os
import sys
import subprocess
import time
import numpy as np
import re
import shutil 
import torch
from collections import defaultdict

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) # 獲取上一層目錄 (即 MAS_GRDH_PATH)

# 將上一層目錄加入 sys.path 以便導入模組
sys.path.append(PARENT_DIR) 

# === 2. 導入模組 (修正循環導入問題) ===
try:
    # 【優先嘗試】本地直接導入 (當您在 scripts/ 資料夾下執行時)
    from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
    from utils import load_512
    print("✅ [System] 成功導入攻擊模組 (Local Import)")
except ImportError:
    # 【備用方案】Package 導入 (當您在專案根目錄執行時)
    try:
        from scripts.robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
        from scripts.utils import load_512
        print("✅ [System] 成功導入攻擊模組 (Package Import)")
    except ImportError as e:
        print(f"❌ [System] 導入模組失敗: {e}")
        print("   請確認 robust_eval.py 和 utils.py 是否存在於 scripts/ 目錄中。")
        sys.exit(1)

# === 3. 核心配置 ===
MAS_GRDH_PATH = PARENT_DIR 

# 模型與配置路徑 (保持您原本的路徑設定)
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
# 若上述絕對路徑有誤，可嘗試相對路徑: CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")

CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

# 指向 "Pure Algorithm" 版本的腳本
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")
TXT2IMG_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "txt2img.py") 

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "robust_pure_test_results")

# === 4. 定義魯棒性測試套件 (Attack Suite) ===
# 格式: (攻擊函數, 參數列表, 顯示名稱, 副檔名)
ATTACK_SUITE = [
    (identity, [None], "1_Identity_Control", ".png"),
    (storage, [None], "2_Storage_Save_Load", ".png"),
    (jpeg, [95, 80, 60, 50], "3_JPEG_Compression", ".jpg"), # 測試到 QF=50
    (resize, [0.9, 0.75, 0.5], "4_Resize", ".png"),
    (mblur, [3, 5], "5_Median_Blur", ".png"),           
    (gblur, [3, 5], "6_Gaussian_Blur", ".png"),         
    (awgn, [0.01, 0.05], "7_Gaussian_Noise", ".png"), 
]

# === 5. 輔助函數 ===

def run_alice_once(prompt, session_key, clean_stego_path, payload_path):
    """
    執行 Pure Alice 生成隱寫圖。
    """
    # 確保 Payload 存在 (生成 600 bytes 隨機數據)
    with open(payload_path, "wb") as f:
        f.write(os.urandom(2048))

    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", payload_path,
        "--outpath", clean_stego_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        # 【參數更新】使用先前實驗發現的最佳參數
        "--opt_iters", "10",  # 增加迭代次數以確保高保真度
        
        "--lr", "0.25"         # 降低學習率以穩定收斂
        # 【修正】已移除 "--signal_strength"，因為新版 Alice 使用正交映射
    ]
    
    try:
        # 捕捉輸出，但如果失敗則打印
        result = subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=600)
        
        # 檢查 Alice 是否真的說 "Generated" (新版 Alice 的成功標誌)
        if "Generated Stego Image" not in result.stdout:
             print(f"⚠️ Alice 執行完畢但未回報成功:\n{result.stdout[-300:]}")
             return False
             
    except subprocess.CalledProcessError as e:
        print(f"❌ Alice Crash:\n{e.stderr}")
        print(f"--- Stdout ---\n{e.stdout}")
        return False
    
    # 再次檢查檔案是否存在
    if not os.path.exists(clean_stego_path):
        print(f"❌ Alice 回報成功但找不到圖片: {clean_stego_path}")
        return False
        
    return True

def run_bob_once(img_path, prompt, session_key, gt_path):
    """
    執行 Pure Bob 進行提取與驗證。
    """
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", img_path,
        "--prompt", prompt, # Bob 需要原始 Prompt 來進行反演
        "--secret_key", str(session_key),
        "--gt_path", gt_path, # 用於 Hash 比對
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "20"
    ]
    
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        
        # 【修正】Regex 更新以匹配新版 Bob 輸出 "Bit Accuracy (Raw): 99.89%"
        # 使用 .*? 跳過 "(Raw)" 字樣
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", result_bob.stdout)
        if match:
            return f"{match.group(1)}%"
        return "0.00% (No Data)"
            
    except subprocess.CalledProcessError as e:
        # print(f"Bob Error: {e.stderr}") # 減少洗版，只回傳 0
        return "0.00% (Crash)"
    except subprocess.TimeoutExpired:
        return "0.00% (Timeout)"

def run_txt2img_test(attack_name_str, factor, single_prompt_file_path):
    """
    執行 Baseline (txt2img) 測試。
    """
    attack_map = {
        "1_Identity_Control": "identity", "2_Storage_Save_Load": "storage",
        "3_JPEG_Compression": "jpeg", "4_Resize": "resize",
        "5_Median_Blur": "mblur", "6_Gaussian_Blur": "gblur",
        "7_Gaussian_Noise": "awgn"
    }
    if attack_name_str not in attack_map: return "N/A"
    
    cmd_txt2img = [
        sys.executable, TXT2IMG_SCRIPT,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH, 
        "--dpm_steps", "20", "--scale", "5.0",
        "--test_prompts", single_prompt_file_path, 
        "--attack_layer", attack_map[attack_name_str], 
        "--attack_factor", str(factor) if factor is not None else "0.0",
        "--seed", "42", "--quiet"
    ]
    try:
        result = subprocess.run(cmd_txt2img, check=True, cwd=CURRENT_DIR, capture_output=True, text=True, timeout=600)
        match = re.search(r"average accuracy: (\d+\.\d+)", result.stdout)
        if match: return f"{float(match.group(1)) * 100:.2f}%"
    except: pass
    return "0.00%"

# === 6. 主測試循環 ===

def main():
    print("🚀 Robustness Test (Bit Accuracy Mode) 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(PROMPT_FILE_LIST):
        print(f"⚠️ 找不到 Prompt 文件: {PROMPT_FILE_LIST}")
        print("   使用預設 Prompt 進行單次測試。")
        prompts_to_test = ["A futuristic city skyline, cinematic lighting, 8k"]
    else:
        with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
            prompts_to_test = [line.strip() for line in f if line.strip()]

    results_summary = defaultdict(lambda: ([], []))
    
    # 限制測試數量以免跑太久，如果要跑全部請移除 [:5]
    prompts_to_test = prompts_to_test
    
    for i, base_prompt in enumerate(prompts_to_test):
        print(f"\n🔬 Prompt #{i+1}: '{base_prompt[:40]}...'")
        session_key = int(np.random.randint(10000000, 99999999))
        
        clean_stego_path = os.path.join(OUTPUT_DIR, f"p{i}_stego.png")
        payload_path = os.path.join(OUTPUT_DIR, f"p{i}_payload.dat")
        
        # 1. 執行 Alice
        if not run_alice_once(base_prompt, session_key, clean_stego_path, payload_path): 
            print("   ↳ 跳過此 Prompt (Alice 失敗)")
            continue

        # 2. 檢查 GT Bits 是否存在
        original_gt_bits = clean_stego_path + ".gt_bits.npy"
        if not os.path.exists(original_gt_bits):
            print(f"❌ 錯誤: Alice 沒有產生 GT Bits 檔: {original_gt_bits}")
            continue

        # 3. 加載圖片準備攻擊
        try:
            clean_img_tensor = load_512(clean_stego_path)
            if torch.cuda.is_available(): clean_img_tensor = clean_img_tensor.cuda()
        except Exception as e:
            print(f"❌ 加載圖片失敗 (load_512): {e}")
            continue

        # 4. 攻擊迴圈
        for attack_func, factors, attack_name, file_ext in ATTACK_SUITE:
            for factor in factors:
                factor_str = str(factor) if factor is not None else 'NA'
                attack_key = f"{attack_name} (Fac: {factor_str})"
                
                attacked_path_base = os.path.join(OUTPUT_DIR, f"p{i}_{attack_name}_{factor_str}")
                
                # 執行攻擊
                try:
                    attack_func(clean_img_tensor.clone(), factor, tmp_image_name=attacked_path_base)
                except Exception as e:
                    print(f"   ❌ 攻擊 {attack_key} 執行失敗: {e}")
                    continue
                
                attacked_img_path = f"{attacked_path_base}{file_ext}"
                
                # 複製 GT Bits (Bob 需要 GT 來計算準確率)
                target_gt_bits = attacked_img_path + ".gt_bits.npy"
                try:
                    shutil.copyfile(original_gt_bits, target_gt_bits)
                except:
                    print("   ❌ GT Copy Fail")
                    continue
                
                # 測試 Ours
                pure_acc = run_bob_once(attacked_img_path, base_prompt, session_key, payload_path)
                
                # 測試 Baseline (如果有 txt2img.py)
                tmp_prompt_file = os.path.join(OUTPUT_DIR, f"p{i}_tmp.txt")
                with open(tmp_prompt_file, 'w') as f: f.write(base_prompt)
                base_acc = run_txt2img_test(attack_name, factor, tmp_prompt_file)

                print(f"   {attack_key}: Ours={pure_acc} | Base={base_acc}")

                try:
                    val_ours = float(pure_acc.replace('%', '').split(' ')[0])
                    val_base = float(base_acc.replace('%', '').split(' ')[0])
                    results_summary[attack_key][0].append(val_ours)
                    results_summary[attack_key][1].append(val_base)
                except: pass

    # === 最終統計 ===
    print("\n" + "="*80)
    print(f"{'Attack'.ljust(40)} | {'Ours (Avg)'.ljust(15)} | {'Base (Avg)'.ljust(15)}")
    print("-" * 80)
    for _, factors, attack_name, _ in ATTACK_SUITE:
        for factor in factors:
            factor_str = str(factor) if factor is not None else 'NA'
            attack_key = f"{attack_name} (Fac: {factor_str})"
            res = results_summary[attack_key]
            if res[0]:
                print(f"{attack_key.ljust(40)} | {np.mean(res[0]):.2f}%".ljust(58) + f"| {np.mean(res[1]):.2f}%")
            else:
                print(f"{attack_key.ljust(40)} | N/A".ljust(58) + "| N/A")
    print("="*80)

if __name__ == "__main__":
    main()