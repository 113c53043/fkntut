import os
import random
import time
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from pure_alice_final import generate_alice_image, load_model_from_config

# === [關鍵修改] 設定輸出路徑 ===
# 這樣生成的圖片會直接進入 SRNet 的目錄，訓練時就不需要搬檔案
OUTPUT_ROOT = os.path.join("SRNET", "Pytorch-implementation-of-SRNet", "data")

NUM_TRAIN = 1500      # 訓練集對數
NUM_VAL = 500         # 驗證集對數
DEVICE = "cuda"

# 多樣化的 Prompts (增加多樣性以防止過擬合)
PROMPTS = [
    "A high quality photo of a calm lake with mountains",
    "A cyberpunk robot standing in a neon city, detailed",
    "A portrait of a cute cat sitting on a sofa",
    "A futuristic spaceship landing on mars",
    "A bowl of delicious fruit salad on a wooden table",
    "A scenic view of a forest in autumn with colorful leaves",
    "A busy street in Tokyo at night, bokeh",
    "An oil painting of a cottage near a river",
    "A professional photograph of a luxury car",
    "A macro shot of a flower with dew drops",
    "An astronaut floating in space, earth background",
    "A medieval castle on a hill, fantasy style",
    "A delicious burger with fries, food photography",
    "A snowy mountain peak under a starry sky",
    "A digital illustration of a anime girl in a garden",
    "A close up of a lizard eye, macro photography",
    "A beautiful library with old books",
    "A cup of coffee with latte art on a table",
    "A rainy street in London, reflection",
    "A minimalist interior design living room"
]

def setup_directories():
    """建立 SRNet 需要的資料夾結構"""
    dirs = [
        os.path.join(OUTPUT_ROOT, "train", "cover"),
        os.path.join(OUTPUT_ROOT, "train", "stego"),
        os.path.join(OUTPUT_ROOT, "val", "cover"),
        os.path.join(OUTPUT_ROOT, "val", "stego")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Directory structure created at: {OUTPUT_ROOT}")

def generate_subset(model, sampler, subset_name, count, start_idx=0):
    """生成指定數量的 Cover/Stego 對"""
    print(f"🚀 Generating {count} pairs for {subset_name} set...")
    
    # 建立隨機 Payload (16k bits)
    dummy_payload = os.urandom(2048) 
    
    for i in tqdm(range(count)):
        idx = start_idx + i
        filename = f"{idx:05d}.png"
        
        # 1. 隨機選擇 Prompt 和 Key (確保成對時參數一致)
        prompt = random.choice(PROMPTS)
        secret_key = random.randint(1, 1000000)
        
        # 2. 設定路徑
        path_cover = os.path.join(OUTPUT_ROOT, subset_name, "cover", filename)
        path_stego = os.path.join(OUTPUT_ROOT, subset_name, "stego", filename)
        
        # 3. 生成 Cover (Baseline Mode - Open Loop)
        # 代表 "未經優化的原始生成"
        generate_alice_image(
            model=model, sampler=sampler, 
            prompt=prompt, secret_key=secret_key, payload_data=dummy_payload,
            outpath=path_cover, 
            opt_iters=0, mode="baseline", 
            dpm_steps=20, device=DEVICE
        )
        
        # 4. 生成 Stego (Adaptive Mode - Closed Loop)
        # 代表 "經過我們方法優化的結果"
        generate_alice_image(
            model=model, sampler=sampler, 
            prompt=prompt, secret_key=secret_key, payload_data=dummy_payload,
            outpath=path_stego, 
            opt_iters=15, mode="adaptive", 
            early_stop_threshold=0.0693,   # 黃金閾值
            min_iters=5,
            check_interval=2,
            dpm_steps=20, device=DEVICE
        )

def main():
    # 確認權重檔案存在
    config_path = "configs/stable-diffusion/ldm.yaml"
    ckpt_path = "weights/v1-5-pruned.ckpt"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Model weights not found at {ckpt_path}")
        print("   Please run this script from the project root (mas_GRDH/)")
        return

    setup_directories()

    device = torch.device(DEVICE)
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device)
    sampler = DPMSolverSampler(model)
    
    # 生成訓練集
    generate_subset(model, sampler, "train", NUM_TRAIN, start_idx=0)
    
    # 生成驗證集
    generate_subset(model, sampler, "val", NUM_VAL, start_idx=NUM_TRAIN)
    
    print("\n🎉 Dataset generation complete!")
    print(f"Images are saved in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()