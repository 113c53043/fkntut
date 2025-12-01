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
PARENT_DIR = os.path.dirname(CURRENT_DIR) 

sys.path.append(PARENT_DIR) 

# === 2. 導入模組 ===
try:
    from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
    from utils import load_512
    print("✅ [System] 成功導入攻擊模組")
except ImportError:
    try:
        from scripts.robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
        from scripts.utils import load_512
        print("✅ [System] 成功導入攻擊模組 (Package)")
    except ImportError as e:
        print(f"❌ [System] 導入失敗: {e}")
        sys.exit(1)

# === 3. 核心配置 ===
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")
TXT2IMG_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "txt2img.py") 

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "robust_pure_test_results")

# === 4. 攻擊套件 ===
ATTACK_SUITE = [
    (identity, [None], "1_Identity", ".png"),
    (storage, [None], "2_Storage", ".png"),
    (jpeg, [95, 80, 70, 50], "3_JPEG", ".jpg"), 
    (resize, [0.8, 0.6], "4_Resize", ".png"), # 論文摘要提到 0.8x
    (mblur, [3, 5], "5_MedianBlur", ".png"),           
    (gblur, [3, 5], "6_GaussianBlur", ".png"),         
    (awgn, [0.01, 0.05], "7_GaussianNoise", ".png"), 
]

# === 5. 輔助函數 ===

def run_subprocess_with_stream(cmd, cwd):
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    captured_output = []
    for line in process.stdout:
        print(line, end='') 
        captured_output.append(line)
    process.wait()
    return process.returncode, "".join(captured_output)

def run_alice_once(prompt, session_key, clean_stego_path, payload_path):
    with open(payload_path, "wb") as f:
        f.write(os.urandom(600))

    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", payload_path,
        "--outpath", clean_stego_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        # 【參數優化：回應教授對於時間成本的質疑】
        # 我們將迭代次數降回 10 次，但稍微提高學習率以保持收斂
        # 這是在 "生成時間" 與 "強健性" 之間取得的最佳平衡點
        "--opt_iters", "10",  
        "--lr", "0.25"         
    ]
    
    print(f"   [Alice] 生成中 (Iters=10, LR=0.25)...")
    try:
        returncode, output = run_subprocess_with_stream(cmd_alice, MAS_GRDH_PATH)
        if returncode != 0 or "Generated Stego Image" not in output:
             print(f"⚠️ Alice 失敗")
             return False
    except Exception as e:
        print(f"❌ Alice Error: {e}")
        return False
    
    if not os.path.exists(clean_stego_path):
        return False
    return True

def run_bob_once(img_path, prompt, session_key, gt_path):
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", img_path,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--gt_path", gt_path, 
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "20"
    ]
    
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", result_bob.stdout)
        if match: return f"{match.group(1)}%"
        return "0.00%"
    except:
        return "0.00%"

def run_txt2img_test(attack_name_str, factor, single_prompt_file_path):
    """
    執行 Baseline (txt2img) 測試。
    """
    attack_map = {
        "1_Identity": "identity", "2_Storage": "storage",
        "3_JPEG": "jpeg", "4_Resize": "resize",
        "5_MedianBlur": "mblur", "6_GaussianBlur": "gblur",
        "7_GaussianNoise": "awgn"
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
    print("🚀 Robustness Test (Efficiency Optimized Mode) 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(PROMPT_FILE_LIST):
        prompts_to_test = ["A futuristic city skyline, cinematic lighting, 8k"]
    else:
        with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
            prompts_to_test = [line.strip() for line in f if line.strip()]

    results_summary = defaultdict(lambda: ([], []))
    prompts_to_test = prompts_to_test[:3] # 測試前 3 個 Prompt 即可
    
    for i, base_prompt in enumerate(prompts_to_test):
        print(f"\n🔬 Prompt #{i+1}: '{base_prompt[:40]}...'")
        session_key = int(np.random.randint(10000000, 99999999))
        
        clean_stego_path = os.path.join(OUTPUT_DIR, f"p{i}_stego.png")
        payload_path = os.path.join(OUTPUT_DIR, f"p{i}_payload.dat")
        
        # 1. 執行 Alice (10 iters)
        if not run_alice_once(base_prompt, session_key, clean_stego_path, payload_path): 
            continue

        original_gt_bits = clean_stego_path + ".gt_bits.npy"
        
        # 3. 加載圖片準備攻擊
        try:
            clean_img_tensor = load_512(clean_stego_path)
            if torch.cuda.is_available(): clean_img_tensor = clean_img_tensor.cuda()
        except: continue

        # 4. 攻擊迴圈
        for attack_func, factors, attack_name, file_ext in ATTACK_SUITE:
            for factor in factors:
                factor_str = str(factor) if factor is not None else 'NA'
                attack_key = f"{attack_name} (Fac: {factor_str})"
                
                attacked_path_base = os.path.join(OUTPUT_DIR, f"p{i}_{attack_name}_{factor_str}")
                
                try:
                    attack_func(clean_img_tensor.clone(), factor, tmp_image_name=attacked_path_base)
                except: continue
                
                attacked_img_path = f"{attacked_path_base}{file_ext}"
                target_gt_bits = attacked_img_path + ".gt_bits.npy"
                try:
                    shutil.copyfile(original_gt_bits, target_gt_bits)
                except: continue
                
                # 測試 Ours
                pure_acc = run_bob_once(attacked_img_path, base_prompt, session_key, payload_path)
                
                # 測試 Baseline
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
    print("="*80)

if __name__ == "__main__":
    main()