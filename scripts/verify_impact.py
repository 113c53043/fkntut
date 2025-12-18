import os
import sys
import subprocess
import numpy as np
import re
import shutil 
from collections import defaultdict

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
sys.path.append(PARENT_DIR) 

try:
    from robust_eval import jpeg, rotation
    from utils import load_512
except ImportError:
    pass

MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

# 腳本路徑
ALICE_CLEAN = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py") # 無標記
ALICE_SYNC  = os.path.join(MAS_GRDH_PATH, "pure_alice_spectral_mask.py")    # 有標記

BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "impact_verification")

# 我們只測 JPEG 50 和 Rotation 10，這是最關鍵的差異點
ATTACK_SUITE = [
    (jpeg, [50], "JPEG_50", ".jpg"),
    (rotation, [10], "Rot_10", ".png"),
]

def run_alice(script, prompt, key, out_path, payload, extra=[]):
    cmd = [sys.executable, script, "--prompt", prompt, "--secret_key", str(key), 
           "--payload_path", payload, "--outpath", out_path, "--ckpt", CKPT_PATH, 
           "--config", CONFIG_PATH, "--opt_iters", "10", "--dpm_steps", "20",
           "--lr", "0.05", "--lambda_reg", "1.5", "--use_uncertainty"] + extra
    subprocess.run(cmd, check=True, cwd=MAS_GRDH_PATH, capture_output=True)

def run_bob(img, prompt, key, gt, use_sync=True):
    cmd = [sys.executable, BOB_SCRIPT, "--img_path", img, "--prompt", prompt, 
           "--secret_key", str(key), "--gt_path", gt, "--ckpt", CKPT_PATH, 
           "--config", CONFIG_PATH]
    
    if not use_sync:
        cmd.append("--no_sync") # 關鍵：告訴 Bob 不要亂校正
        
    res = subprocess.run(cmd, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True)
    match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", res.stdout)
    return float(match.group(1)) if match else 0.0

def generate_gt_file(payload_path, output_path):
    """
    [新增] 主動生成 Ground Truth Bits 檔案
    避免依賴 Alice 腳本是否實作了儲存功能
    """
    with open(payload_path, "rb") as f:
        raw_data = f.read()
    
    CAPACITY_BYTES = 16384 // 8 
    if len(raw_data) > CAPACITY_BYTES - 2:
        raw_data = raw_data[:CAPACITY_BYTES-2]
    
    length_header = len(raw_data).to_bytes(2, 'big')
    final_payload = length_header + raw_data
    
    if len(final_payload) < CAPACITY_BYTES:
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
        
    np.save(output_path, np.frombuffer(final_payload, dtype=np.uint8))

def main():
    print("🚀 Impact Verification: Clean vs Sync 🚀")
    
    # === [Fix] Ensure output directory exists ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PROMPT_FILE_LIST, 'r') as f:
        prompts = [l.strip() for l in f if l.strip()][:5] # 測前 5 張就夠了
        
    results = defaultdict(list)
    
    for i, p in enumerate(prompts):
        print(f"\nPrompt {i+1}: {p[:20]}...")
        key = 12345 + i
        payload = os.path.join(OUTPUT_DIR, "payload.dat")
        if not os.path.exists(payload): 
            with open(payload, "wb") as f: f.write(os.urandom(2048))
            
        # === Group A: Clean (Original SOTA) ===
        # 使用 pure_alice_uncertainty_fixed.py
        # Bob 使用 --no_sync
        path_a = os.path.join(OUTPUT_DIR, f"p{i}_clean.png")
        
        # [新增] 預先生成 GT 檔案，防止 Alice 沒存
        generate_gt_file(payload, path_a + ".gt_bits.npy")
        
        run_alice(ALICE_CLEAN, p, key, path_a, payload)
        
        # === Group B: With Sync (New) ===
        # 使用 pure_alice_spectral_mask.py (Fixed Mode)
        # Bob 使用預設 (開啟 Sync)
        path_b = os.path.join(OUTPUT_DIR, f"p{i}_sync.png")
        
        # [新增] 預先生成 GT 檔案
        generate_gt_file(payload, path_b + ".gt_bits.npy")
        
        run_alice(ALICE_SYNC, p, key, path_b, payload, ["--strategy", "fixed"])
        
        # 攻擊測試
        for func, facs, name, ext in ATTACK_SUITE:
            for fac in facs:
                # 攻擊 A
                att_a = path_a + f"_{name}"
                func(load_512(path_a).cuda(), fac, att_a)
                # 確保來源 GT 存在 (我們剛剛生成了)
                shutil.copy(path_a+".gt_bits.npy", att_a+ext+".gt_bits.npy")
                acc_a = run_bob(att_a+ext, p, key, payload, use_sync=False) # No Sync
                
                # 攻擊 B
                att_b = path_b + f"_{name}"
                func(load_512(path_b).cuda(), fac, att_b)
                shutil.copy(path_b+".gt_bits.npy", att_b+ext+".gt_bits.npy")
                acc_b = run_bob(att_b+ext, p, key, payload, use_sync=True) # Sync
                
                print(f"  {name}: Clean={acc_a:.2f}% | Sync={acc_b:.2f}%")
                results[f"{name}_Clean"].append(acc_a)
                results[f"{name}_Sync"].append(acc_b)

    print("\n" + "="*40)
    for k, v in results.items():
        print(f"{k.ljust(20)}: {np.mean(v):.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()