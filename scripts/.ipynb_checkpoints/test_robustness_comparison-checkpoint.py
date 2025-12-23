import os
import sys
import subprocess
import numpy as np
import re
import shutil 
import torch
from collections import defaultdict
import gc

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
sys.path.append(PARENT_DIR) 

try:
    from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn, crop, rotation
    from utils import load_512
except ImportError:
    pass

MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")

CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")

# 指向修正後的 Alice
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "robust_high_intensity_2")

# === 2. 定義攻擊套件 (只取最高強度) ===
ATTACK_SUITE = [
    (identity, [None], "Identity", ".png"),
    (jpeg, [50], "JPEG (QF=50)", ".jpg"),
    (resize, [0.5], "Resize (0.5x)", ".png"),
    (mblur, [5], "Median Blur (k=5)", ".png"),
    (gblur, [5], "Gaussian Blur (k=5)", ".png"),
    (awgn, [0.05], "Gaussian Noise (0.05)", ".png"),
    
]

# === GT Bits 檔案建立函式 ===
def create_gt_bits_file(payload_path, out_gt_path):
    CAPACITY_BYTES = 16384 // 8
    with open(payload_path, "rb") as f:
        raw_data = f.read()
    
    payload_data = raw_data
    if len(payload_data) > CAPACITY_BYTES - 2:
        payload_data = payload_data[:CAPACITY_BYTES-2]
    
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + payload_data
    
    if len(final_payload) < CAPACITY_BYTES:
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
        
    np.save(out_gt_path, np.frombuffer(final_payload, dtype=np.uint8))

def run_alice_generic(prompt, session_key, out_path, payload_path, extra_args=[]):
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", payload_path,
        "--outpath", out_path,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", "--dpm_steps", "20"
    ] + extra_args
    
    try:
        result = subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=600)
        if not os.path.exists(out_path):
            print(f"❌ Alice finished but file missing: {out_path}")
            return False
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Alice Failed: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ Alice Timeout")
        return False

def run_bob_once(img_path, prompt, session_key):
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", img_path,
        "--prompt", prompt,
        "--secret_key", str(session_key),
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--dpm_steps", "20"
    ]
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", result_bob.stdout)
        if match: return float(match.group(1))
        return 0.0
    except: return 0.0

def main():
    print("🚀 High-Intensity Robustness Test: Fixed vs Adaptive (Aggressive) 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = ["A futuristic city"]
    
    prompts_to_test = prompts[:120]

    results = defaultdict(lambda: {"Fixed": [], "Adaptive": []})

    for i, prompt in enumerate(prompts_to_test):
        print(f"\n🔬 Prompt #{i+1}: {prompt[:30]}...")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        session_key = 123456 + i
        payload_path = os.path.join(OUTPUT_DIR, f"p{i}.dat")
        if not os.path.exists(payload_path):
            with open(payload_path, "wb") as f: f.write(os.urandom(2048))
            
        path_fixed = os.path.join(OUTPUT_DIR, f"p{i}_fixed.png")
        path_adapt = os.path.join(OUTPUT_DIR, f"p{i}_adapt.png")
        
        gt_fixed_src = path_fixed + ".gt_bits.npy"
        gt_adapt_src = path_adapt + ".gt_bits.npy"

        # 1. Run Fixed
        if os.path.exists(path_fixed): os.remove(path_fixed)
        success_f = run_alice_generic(prompt, session_key, path_fixed, payload_path, 
                            ["--lr", "0.05", "--lambda_reg", "1.5", "--use_uncertainty"]) 
        
        # 2. Run Adaptive
        if os.path.exists(path_adapt): os.remove(path_adapt)
        success_a = run_alice_generic(prompt, session_key, path_adapt, payload_path, 
                            ["--lr", "0.05", "--lambda_reg", "1.5", "--use_uncertainty", "--adaptive_mask"])

        if not os.path.exists(gt_fixed_src): create_gt_bits_file(payload_path, gt_fixed_src)
        if not os.path.exists(gt_adapt_src): create_gt_bits_file(payload_path, gt_adapt_src)

        if not success_f or not success_a:
            print("  ❌ Skipping (Alice failed)")
            continue

        try:
            img_fixed = load_512(path_fixed).cuda()
            img_adapt = load_512(path_adapt).cuda()

            for attack_func, factors, name, ext in ATTACK_SUITE:
                factor = factors[0]
                att_name = f"{name}"
                print(f"    ⚔️  Running Attack: {att_name} ...", flush=True)
                
                # Fixed
                att_path_f = os.path.join(OUTPUT_DIR, f"p{i}_fix_{name.split()[0]}.png")
                final_f = att_path_f.replace(".png", ext)
                try:
                    attack_func(img_fixed.clone(), factor, tmp_image_name=att_path_f.replace(".png", ""))
                    if os.path.exists(gt_fixed_src): shutil.copyfile(gt_fixed_src, final_f + ".gt_bits.npy")
                    acc_f = run_bob_once(final_f, prompt, session_key)
                    results[att_name]["Fixed"].append(acc_f)
                except Exception: pass

                # Adaptive
                att_path_a = os.path.join(OUTPUT_DIR, f"p{i}_ada_{name.split()[0]}.png")
                final_a = att_path_a.replace(".png", ext)
                try:
                    attack_func(img_adapt.clone(), factor, tmp_image_name=att_path_a.replace(".png", ""))
                    if os.path.exists(gt_adapt_src): shutil.copyfile(gt_adapt_src, final_a + ".gt_bits.npy")
                    acc_a = run_bob_once(final_a, prompt, session_key)
                    results[att_name]["Adaptive"].append(acc_a)
                except Exception: pass
        
        finally:
            if 'img_fixed' in locals(): del img_fixed
            if 'img_adapt' in locals(): del img_adapt
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(f"{'Attack (High Intensity)':<30} | {'Fixed (Avg Acc)':<15} | {'Adaptive (Avg Acc)':<15}")
    print("-" * 80)
    for name in results:
        fix_vals = results[name]["Fixed"]
        ada_vals = results[name]["Adaptive"]
        
        avg_f = np.mean(fix_vals) if fix_vals else 0.0
        avg_a = np.mean(ada_vals) if ada_vals else 0.0
        
        better = "✅" if avg_a >= avg_f else ""
        print(f"{name:<30} | {avg_f:<15.2f} | {avg_a:<15.2f} {better}")
    print("="*80)

if __name__ == "__main__":
    main()