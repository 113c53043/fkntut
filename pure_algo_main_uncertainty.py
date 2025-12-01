import os
import sys
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.cm as cm 

# === 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
MAS_GRDH_PATH = CURRENT_DIR

CKPT_PATH = "weights/v1-5-pruned.ckpt" 
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "uncertainty_visual_test")
PAYLOAD_FILE = os.path.join(OUTPUT_DIR, "random_payload.dat")

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(PAYLOAD_FILE):
        os.remove(PAYLOAD_FILE)
    with open(PAYLOAD_FILE, "wb") as f:
        f.write(os.urandom(2048))
    print(f"📄 Generated Test Payload: 600 bytes")

def generate_heatmap(diff_array, amplify=30.0):
    norm = diff_array.astype(np.float32) / 255.0 * amplify
    norm = np.clip(norm, 0.0, 1.0)
    heatmap_rgba = cm.jet(norm) 
    heatmap_uint8 = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(heatmap_uint8)

def create_comprehensive_grid(paths, output_path):
    # Load all images
    imgs = {}
    for key, path in paths.items():
        if os.path.exists(path):
            imgs[key] = Image.open(path).convert("RGB")
        else:
            print(f"⚠️ Missing: {path}")
            return

    w, h = imgs['init'].size
    
    # 計算差異矩陣
    arr_init = np.array(imgs['init']).astype(np.float32)
    arr_base = np.array(imgs['base']).astype(np.float32)
    arr_ours = np.array(imgs['ours']).astype(np.float32)

    diff_base = np.mean(np.abs(arr_base - arr_init), axis=2)
    diff_ours = np.mean(np.abs(arr_ours - arr_init), axis=2)

    heatmap_base = generate_heatmap(diff_base, amplify=25.0)
    heatmap_ours = generate_heatmap(diff_ours, amplify=25.0)
    
    if 'mask' in imgs:
        mask_img = imgs['mask']
        if mask_img.size != (w, h):
            mask_img = mask_img.resize((w, h), Image.NEAREST)
    else:
        mask_img = Image.new("RGB", (w, h), (128, 128, 128))

    # --- 組合畫布 ---
    grid_w = w * 3
    grid_h = h * 2 + 80
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Row 1
    canvas.paste(imgs['init'], (0, 40))
    canvas.paste(imgs['base'], (w, 40))
    canvas.paste(imgs['ours'], (w*2, 40))

    # Row 2
    canvas.paste(mask_img, (0, h + 80))
    canvas.paste(heatmap_base, (w, h + 80))
    canvas.paste(heatmap_ours, (w*2, h + 80))

    headers = [
        (0, 10, "1. Initial Generated Image (Raw)"),
        (w, 10, "2. Baseline (No Mask)\n(Visible Noise/Artifacts)"),
        (w*2, 10, "3. Ours (Uncertainty + Reg)\n(Clean Texture Preserved)"),
        
        (0, h + 50, "4. Uncertainty Mask\n(Black=Risky, White=Safe)"),
        (w, h + 50, "5. Baseline Modifications\n(Modifies smooth areas -> Artifacts)"),
        (w*2, h + 50, "6. Ours Modifications\n(Hides in texture, avoids smooth)")
    ]

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = None

    for x, y, text in headers:
        draw.text((x + 10, y), text, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    print(f"\n✅ Final Visual Report saved to: {output_path}")

def run_experiment(session_key):
    print(f"\n--- [Experiment] Visual Comparison V3 (Texture Prompt + Tuned Params) ---")
    
    # === 關鍵修改 1: 使用紋理豐富的 Prompt (避免平滑霓虹燈) ===
    # 這種 Prompt 對隱寫術極度友好，幾乎看不出修改
    #prompt = "Close up texture of an old mossy stone wall with small white flowers, realistic, 8k, highly detailed"
    #prompt = "a cute kitten playing soccer, tiny cat kicking a football on a grassy field, dynamic action pose, bright sunlight, soft fur texture, detailed paws, energetic movement, cinematic lighting, shallow depth of field, ultra high resolution, 4k, highly detailed, photorealistic"
    prompt = "a cute kitten performing a bicycle kick in soccer, tiny cat in mid-air doing a stunning acrobatic kick, football flying, dynamic motion, on a grassy field, dramatic action pose, sunlight, detailed fur, energetic, photorealistic, ultra high resolution, 4k, cinematic lighting, highly detailed"

    #prompt = "A futuristic city with neon lights and a dark sky, 8k, highly detailed"
    #prompt = "A futuristic cyberpunk city with neon lights and rain, 8k, highly detailed"

    
    # === 關鍵修改 2: 參數微調 ===
    # 降低 LR (更溫和的修改)
    target_lr = "0.15" 
    # 提高 Reg (更強的畫質保護)
    target_reg = "0.4"   

    p_init = os.path.join(OUTPUT_DIR, "1_init.png")
    p_base = os.path.join(OUTPUT_DIR, "2_base.png")
    p_ours = os.path.join(OUTPUT_DIR, "3_ours.png")
    p_mask = p_ours + ".uncertainty_mask.png"

    # 1. 生成 Initial
    print("\n🔹 Step 1: Generating Initial Image...")
    subprocess.run([sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, "--secret_key", str(session_key), "--payload_path", PAYLOAD_FILE,
        "--outpath", p_init, "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "0", "--use_uncertainty"
    ], check=True, cwd=MAS_GRDH_PATH)

    # 2. 執行 Baseline
    print(f"\n🔹 Step 2: Running Baseline...")
    subprocess.run([sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, "--secret_key", str(session_key), "--payload_path", PAYLOAD_FILE,
        "--outpath", p_base, "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", 
        "--lr", target_lr,
        "--lambda_reg", target_reg 
    ], check=True, cwd=MAS_GRDH_PATH)

    # 3. 執行 Ours
    print(f"\n🔹 Step 3: Running Ours (Uncertainty Guided)...")
    subprocess.run([sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, "--secret_key", str(session_key), "--payload_path", PAYLOAD_FILE,
        "--outpath", p_ours, "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", 
        "--lr", target_lr,
        "--lambda_reg", target_reg,
        "--use_uncertainty"
    ], check=True, cwd=MAS_GRDH_PATH)

    create_comprehensive_grid({
        'init': p_init, 'base': p_base, 'ours': p_ours, 'mask': p_mask
    }, os.path.join(OUTPUT_DIR, "Final_Visual_Report_Texture.png"))

def main():
    ensure_paths()
    session_key = 999999
    run_experiment(session_key)

if __name__ == "__main__":
    main()