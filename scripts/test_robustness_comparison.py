import os
import sys
import subprocess
import numpy as np
import re
import shutil 
import torch
from collections import defaultdict

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
sys.path.append(PARENT_DIR) 

try:
    from robust_eval import jpeg, resize
    from utils import load_512
except ImportError:
    pass

MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

# 使用新的 Alice 腳本
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py") 
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")

OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "robust_logic_test")
DIR_PURE = os.path.join(OUTPUT_ROOT, "pure")
DIR_INVERTED = os.path.join(OUTPUT_ROOT, "inverted")
DIR_STANDARD = os.path.join(OUTPUT_ROOT, "standard")

ATTACK_SUITE = [
    (jpeg, [50], "JPEG_Compression", ".jpg"),
    (resize, [0.5], "Resize", ".png"),
]

def run_alice_generic(prompt, session_key, out_path, payload_path, extra_args=[]):
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, "--secret_key", str(session_key),
        "--payload_path", payload_path, "--outpath", out_path,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", "--dpm_steps", "20"
    ] + extra_args
    try:
        subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=600)
        return True
    except subprocess.CalledProcessError: return False
    
def run_bob_once(img_path, prompt, session_key):
    cmd_bob = [
        sys.executable, BOB_SCRIPT, "--img_path", img_path, "--prompt", prompt,
        "--secret_key", str(session_key), "--ckpt", CKPT_PATH, 
        "--config", CONFIG_PATH, "--dpm_steps", "20"
    ]
    try:
        res = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", res.stdout)
        return f"{match.group(1)}%" if match else "0.00%"
    except: return "0.00%"

def parse_percentage(val_str):
    try: return float(val_str.replace('%', '').split(' ')[0])
    except: return None

def process_single_strategy(name, out_dir, prompt, key, payload, args, res_dict):
    stego = os.path.join(out_dir, "stego.png")
    gt_bits = stego + ".gt_bits.npy"
    if run_alice_generic(prompt, key, stego, payload, args):
        img_tensor = load_512(stego).cuda()
        for func, facs, att_name, ext in ATTACK_SUITE:
            for fac in facs:
                att_key = f"{att_name}_{fac}"
                att_path = os.path.join(out_dir, att_key)
                try:
                    func(img_tensor.clone(), fac, att_path)
                    shutil.copyfile(gt_bits, att_path+ext+".gt_bits.npy")
                    acc = run_bob_once(att_path+ext, prompt, key)
                    val = parse_percentage(acc)
                    if val: res_dict[name][att_key].append(val)
                except: pass

def main():
    print("🚀 Logic Comparison Test 🚀")
    for d in [OUTPUT_ROOT, DIR_PURE, DIR_INVERTED, DIR_STANDARD]: os.makedirs(d, exist_ok=True)
    
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST, 'r') as f: prompts = [l.strip() for l in f if l.strip()][:20]
    else: prompts = ["A futuristic city"] * 5

    results = defaultdict(lambda: defaultdict(list))
    
    for i, p in enumerate(prompts):
        print(f"\nPrompt {i+1}...")
        key = 12345 + i
        payload = os.path.join(OUTPUT_ROOT, f"p{i}_payload.dat")
        with open(payload, "wb") as f: f.write(os.urandom(2048))
        
        # 1. Pure
        process_single_strategy("Pure", os.path.join(DIR_PURE, f"p{i}"), p, key, payload, 
                                ["--lr", "0.25", "--lambda_reg", "0.0"], results)
        
        # 2. Inverted (Current SOTA)
        process_single_strategy("Inverted", os.path.join(DIR_INVERTED, f"p{i}"), p, key, payload, 
                                ["--lr", "0.05", "--lambda_reg", "1.5", "--use_uncertainty", "--mask_mode", "inverted"], results)
                                
        # 3. Standard (New Hypothesis)
        process_single_strategy("Standard", os.path.join(DIR_STANDARD, f"p{i}"), p, key, payload, 
                                ["--lr", "0.05", "--lambda_reg", "1.5", "--use_uncertainty", "--mask_mode", "standard"], results)

    print("\n" + "="*80)
    print(f"{'Attack'.ljust(20)} | {'Pure'.center(10)} | {'Inverted'.center(10)} | {'Standard'.center(10)}")
    print("-" * 80)
    all_att = sorted(list(set([k for v in results.values() for k in v.keys()])))
    for att in all_att:
        v_p = np.mean(results["Pure"].get(att, [0]))
        v_i = np.mean(results["Inverted"].get(att, [0]))
        v_s = np.mean(results["Standard"].get(att, [0]))
        print(f"{att.ljust(20)} | {v_p:.2f}%     | {v_i:.2f}%     | {v_s:.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()