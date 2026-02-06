import os
import random
import torch
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from torch import autocast
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from pure_alice_final import generate_alice_image, load_model_from_config

# === 設定 ===
# 輸出路徑
OUTPUT_DIR = os.path.join("SRNET", "Pytorch-implementation-of-SRNet", "data", "ours_unpaired")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS_DIR = os.path.join(CURRENT_DIR, "scripts")

PATH_CAPTIONS = os.path.join(
    SCRIPTS_DIR,
    "coco_annotations",
    "captions_val2017.json"
)

NUM_TRAIN = 1500
NUM_VAL = 500
DEVICE = "cuda"

def get_real_coco_prompts(count):
    """
    從真實的 COCO annotations json 檔案中讀取 captions
    """
    print(f"📖 Loading captions from {PATH_CAPTIONS}...")
    
    if not os.path.exists(PATH_CAPTIONS):
        raise FileNotFoundError(f"❌ Error: COCO annotations not found at {PATH_CAPTIONS}")

    with open(PATH_CAPTIONS, 'r') as f:
        data = json.load(f)
    
    # COCO JSON 結構: data['annotations'] 是一個 list，每個元素有 'caption' 欄位
    all_captions = [anno['caption'] for anno in data['annotations']]
    
    # 過濾掉太短的 caption 以確保品質 (可選)
    all_captions = [c for c in all_captions if len(c.split()) > 3]
    
    print(f"✅ Loaded {len(all_captions)} captions from COCO.")
    
    if len(all_captions) < count:
        print(f"⚠️ Warning: Requested {count} prompts but only found {len(all_captions)}. Sampling with replacement.")
        prompts = random.choices(all_captions, k=count)
    else:
        prompts = random.sample(all_captions, count)
        
    return prompts

# 生成純淨 Cover (無嵌入)
def generate_clean_cover(model, sampler, prompt, seed, outpath):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning(["worst quality, low quality"])
    shape = (4, 64, 64)
    z_enc = torch.randn(1, *shape, device=DEVICE) 
    
    with torch.no_grad(), autocast("cuda"):
        samples_z, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=shape,
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=z_enc, verbose=False)
        x_samples = model.decode_first_stage(samples_z)
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    from PIL import Image
    Image.fromarray(img_np.astype('uint8')).save(outpath)

def main():
    config_path = "configs/stable-diffusion/ldm.yaml"
    ckpt_path = "weights/v1-5-pruned.ckpt"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Model weights not found at {ckpt_path}")
        return

    # 建立目錄
    os.makedirs(os.path.join(OUTPUT_DIR, "train", "cover"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "train", "stego"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", "cover"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", "stego"), exist_ok=True)

    print(f"🚀 Generating 'Ours Unpaired' Dataset to: {OUTPUT_DIR}")

    device = torch.device(DEVICE)
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device)
    sampler = DPMSolverSampler(model)
    
    # === 使用真實 COCO Prompts ===
    try:
        prompts_list = get_real_coco_prompts(NUM_TRAIN + NUM_VAL)
    except Exception as e:
        print(e)
        return

    dummy_payload = os.urandom(2048) # 16k bits

    for i in tqdm(range(NUM_TRAIN + NUM_VAL)):
        subset = "train" if i < NUM_TRAIN else "val"
        fname = f"{i:05d}.png"
        prompt = prompts_list[i]
        
        path_cover = os.path.join(OUTPUT_DIR, subset, "cover", fname)
        path_stego = os.path.join(OUTPUT_DIR, subset, "stego", fname)
        
        # === Unpaired 關鍵 ===
        seed_A = random.randint(1, 1000000)
        seed_B = random.randint(1, 1000000)
        while seed_B == seed_A: seed_B = random.randint(1, 1000000)
        
        # 1. Cover (Seed A, Clean)
        generate_clean_cover(model, sampler, prompt, seed_A, path_cover)
        
        # 2. Stego (Seed B, Ours Adaptive)
        generate_alice_image(
            model=model, sampler=sampler, prompt=prompt, secret_key=seed_B, 
            payload_data=dummy_payload, outpath=path_stego, 
            mode="adaptive", opt_iters=15, early_stop_threshold=0.0693, # 使用你的早停參數
            dpm_steps=20, device=DEVICE
        )

    print("\n✅ Generation Complete!")
    print("Next Step: Run 'train.py' on this folder to get the ~50% result.")

if __name__ == "__main__":
    main()