# 檔案位置: scripts/gen_dual_data.py
import os
import sys
import torch
import numpy as np
import json
import time
from PIL import Image
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm
from reedsolo import RSCodec # 您的 alice_gen.py 用到了這個

# === 路徑與環境設定 ===
# 取得 scripts 資料夾的路徑
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 取得專案根目錄 (mas_GRDH)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 將專案根目錄加入 sys.path，以便導入 text_stego_module 和 ldm
sys.path.append(PROJECT_ROOT)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import mapping_module
    # 導入您的文字隱寫模組
    from text_stego_module.stego import TextStegoSystem
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    print(f"請確認您位於 mas_GRDH/scripts 目錄下執行，且 {PROJECT_ROOT} 包含必要的模組。")
    sys.exit(1)

# === 全域設定 ===
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/stable-diffusion/ldm.yaml")
GPT2_PATH = os.path.join(PROJECT_ROOT, "gpt2") # 您的 GPT-2 路徑
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "training_data")
COCO_JSON_PATH = os.path.join(PROJECT_ROOT, "annotations/captions_val2017.json")

# 生成數量
NUM_SAMPLES = 5000 

# === ECC 參數 (參考您的 alice_gen.py) ===
BIT_NUM = 1
LATENT_SHAPE = (1, 4, 64, 64)
LATENT_CAPACITY = 16384 # 4*64*64 * 1
# RS 設定
N_ECC_SYMBOLS = 136 
N_DATA_BYTES_PER_BLOCK = 119
NUM_BLOCKS = 2
PAYLOAD_SIZE_BYTES = NUM_BLOCKS * N_DATA_BYTES_PER_BLOCK # 238
# Repetition 設定
REPETITION_FACTOR = 3

def load_sd_model(config_path, ckpt_path, device):
    """載入 Stable Diffusion"""
    print(f"[System] 載入 SD 模型: {ckpt_path}")
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def load_text_system(model_path):
    """載入 GPT-2 文字隱寫模組"""
    print(f"[System] 載入 Text Stego System: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ 找不到 GPT-2 模型: {model_path}")
        sys.exit(1)
    return TextStegoSystem(model_name=model_path)

def load_coco_prompts(json_path):
    """讀取 MS-COCO Prompts"""
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到 COCO JSON: {json_path}")
        return ["A futuristic city with flying cars"] * 100
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [item['caption'] for item in data['annotations']]

def get_hybrid_ecc_payload(secret_key):
    """
    重現 alice_gen.py 的 Hybrid ECC 編碼邏輯
    Payload -> RS(255,119) -> Repetition(3)
    """
    rsc = RSCodec(N_ECC_SYMBOLS)
    rng = np.random.RandomState(secret_key)
    
    # 1. 生成隨機秘密訊息
    original_secret_bytes = rng.bytes(PAYLOAD_SIZE_BYTES)
    
    # 2. RS 編碼
    encoded_bytes_list = []
    for i in range(NUM_BLOCKS):
        chunk = original_secret_bytes[i*N_DATA_BYTES_PER_BLOCK : (i+1)*N_DATA_BYTES_PER_BLOCK]
        encoded_chunk = rsc.encode(chunk)
        encoded_bytes_list.append(encoded_chunk)
    encoded_bytes = b"".join(encoded_bytes_list)
    
    # 轉為 bits
    rs_coded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))
    
    # 3. Repetition 編碼
    hybrid_coded_bits = np.repeat(rs_coded_bits, REPETITION_FACTOR)
    
    # 4. 填充 Padding
    encoded_size_bits = len(hybrid_coded_bits)
    secret_msg_payload = np.zeros(np.prod(LATENT_SHAPE), dtype=np.uint8).flatten()
    secret_msg_payload[:encoded_size_bits] = hybrid_coded_bits
    
    # 隨機填充剩餘空間 (使用不同 seed 避免混淆)
    seed_kernel = secret_key
    rng_pad = np.random.RandomState(seed=seed_kernel+1)
    random_padding = rng_pad.randint(0, 2**BIT_NUM, LATENT_CAPACITY - encoded_size_bits)
    secret_msg_payload[encoded_size_bits:] = random_padding
    
    return secret_msg_payload.reshape(LATENT_SHAPE).astype(np.int8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Dual System] 開始生成訓練數據 (目標: {NUM_SAMPLES} 對)")
    
    # 1. 初始化所有模型
    sd_model = load_sd_model(CONFIG_PATH, CKPT_PATH, device)
    sd_sampler = DPMSolverSampler(sd_model)
    text_sys = load_text_system(GPT2_PATH)
    mapper = mapping_module.ours_mapping(bits=BIT_NUM)
    
    # 2. 準備 Prompts
    all_prompts = load_coco_prompts(COCO_JSON_PATH)
    
    # 3. 準備輸出目錄
    os.makedirs(os.path.join(OUTPUT_ROOT, "cover"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "stego"), exist_ok=True)
    
    # 4. 生成迴圈
    for i in tqdm(range(NUM_SAMPLES), desc="Generating"):
        filename = f"{i:05d}.png"
        cover_path = os.path.join(OUTPUT_ROOT, "cover", filename)
        stego_path = os.path.join(OUTPUT_ROOT, "stego", filename)
        
        # 中斷續傳
        if os.path.exists(cover_path) and os.path.exists(stego_path):
            continue
            
        # 選取原始 Prompt
        origin_prompt = np.random.choice(all_prompts)
        
        # === A. 生成 Cover (純淨版) ===
        # 使用原始 COCO Prompt + 純隨機噪聲
        seed_cover = np.random.randint(0, 1000000)
        np.random.seed(seed_cover)
        noise_cover = torch.randn(1, 4, 64, 64).to(device)
        
        c_cover = sd_model.get_learned_conditioning([origin_prompt])
        uc = sd_model.get_learned_conditioning([""])
        
        with torch.no_grad(), autocast("cuda"):
            z_0_c, _ = sd_sampler.sample(
                steps=50, conditioning=c_cover, batch_size=1, shape=(4,64,64),
                unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                x_T=noise_cover
            )
            x_cover = sd_model.decode_first_stage(z_0_c)
            x_cover = torch.clamp((x_cover + 1.0) / 2.0, min=0.0, max=1.0)
            
        Image.fromarray((x_cover[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(cover_path)

        # === B. 生成 Stego (雙模態版) ===
        # 1. 準備 Session Key
        session_key = int(np.random.randint(10000000, 99999999))
        
        # 2. [Text Stego] 使用 GPT-2 修改 Prompt 並嵌入 Key
        try:
            # 這是您的 dual_system_main.py 邏輯
            stego_prompt_text, _ = text_sys.alice_encode(origin_prompt, session_key)
        except Exception as e:
            print(f"\n⚠️ Text Encode Failed: {e}, using original prompt")
            stego_prompt_text = origin_prompt
            
        # 3. [Image Stego] 準備 Payload (Hybrid ECC)
        # 這是您的 alice_gen.py 邏輯
        secret_msg = get_hybrid_ecc_payload(session_key)
        
        # 4. [Image Stego] Mapping
        seed_kernel = session_key
        seed_shuffle = (session_key + 9527) % (2**32)
        
        z_stego_np = mapper.encode_secret(
            secret_message=secret_msg,
            seed_kernel=seed_kernel,
            seed_shuffle=seed_shuffle
        )
        z_T_stego = torch.from_numpy(z_stego_np.astype(np.float32)).to(device)
        
        # 5. 生成 Stego 圖像 (使用 Modified Prompt + Mapped Noise)
        c_stego = sd_model.get_learned_conditioning([stego_prompt_text])
        
        with torch.no_grad(), autocast("cuda"):
            z_0_s, _ = sd_sampler.sample(
                steps=50, # 您設定 50 steps
                conditioning=c_stego, batch_size=1, shape=(4,64,64),
                unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                x_T=z_T_stego
            )
            x_stego = sd_model.decode_first_stage(z_0_s)
            x_stego = torch.clamp((x_stego + 1.0) / 2.0, min=0.0, max=1.0)
            
        Image.fromarray((x_stego[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(stego_path)

        # 定期清理
        if i % 50 == 0:
            torch.cuda.empty_cache()

    print("✅ 雙模態數據生成完成！")

if __name__ == "__main__":
    main()