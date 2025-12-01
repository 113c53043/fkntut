import sys
import os
import argparse
import torch
import numpy as np
import hashlib
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from reedsolo import RSCodec 
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# 路徑設定
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import mapping_module
except ImportError:
    sys.exit(1)

def load_model_from_config(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Alice Golden Copy")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bit_num", type=int, default=1)
    parser.add_argument("--dpm_steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=5.0)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # === 1. 準備金鑰 ===
    seed_kernel = opt.secret_key
    seed_shuffle = (opt.secret_key + 9527) % (2**32)
    mapper = mapping_module.ours_mapping(bits=opt.bit_num)
    latent_shape = (1, 4, 64, 64) 

    # === 2. 讀取並加密 Payload (AES) ===
    print(f"[Alice] 正在讀取檔案: {opt.payload_path}")
    with open(opt.payload_path, "rb") as f:
        file_data = f.read()

    aes_key = hashlib.sha256(str(opt.secret_key).encode()).digest()
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(file_data, AES.block_size))
    cipher_len = len(encrypted_data)
    print(f"[Alice] 加密完成。原始: {len(file_data)}B -> 密文: {cipher_len}B")

    # === 3. ECC 組裝 (含長度標頭與截斷) ===
    N_ECC_SYMBOLS = 136 
    N_DATA_BYTES_PER_BLOCK = 119
    NUM_BLOCKS = 2 
    PAYLOAD_CAPACITY = NUM_BLOCKS * N_DATA_BYTES_PER_BLOCK # 238 bytes
    MAX_CIPHER_SIZE = PAYLOAD_CAPACITY - 2 # 236 bytes

    # 安全截斷 (確保對齊 16 bytes)
    if cipher_len > MAX_CIPHER_SIZE:
        print(f"⚠️ [Alice] 檔案過大，執行安全截斷 (Target < {MAX_CIPHER_SIZE})。")
        safe_len = (MAX_CIPHER_SIZE // 16) * 16
        encrypted_data = encrypted_data[:safe_len]
        cipher_len = len(encrypted_data)
        print(f"⚠️ [Alice] 已截斷為: {cipher_len} bytes")

    length_header = cipher_len.to_bytes(2, 'big')
    final_payload = length_header + encrypted_data
    
    if len(final_payload) < PAYLOAD_CAPACITY:
        final_payload += b'\x00' * (PAYLOAD_CAPACITY - len(final_payload))

    rsc = RSCodec(N_ECC_SYMBOLS)
    encoded_bytes_list = []
    for i in range(NUM_BLOCKS):
        chunk = final_payload[i*N_DATA_BYTES_PER_BLOCK : (i+1)*N_DATA_BYTES_PER_BLOCK]
        encoded_chunk = rsc.encode(chunk)
        encoded_bytes_list.append(encoded_chunk)
    
    rs_coded_bits = np.unpackbits(np.frombuffer(b"".join(encoded_bytes_list), dtype=np.uint8))
    hybrid_coded_bits = np.repeat(rs_coded_bits, 3) 

    latent_capacity = np.prod(latent_shape) * opt.bit_num
    secret_msg_payload = np.zeros(latent_shape, dtype=np.uint8).flatten()
    secret_msg_payload[:len(hybrid_coded_bits)] = hybrid_coded_bits
    
    rng_pad = np.random.RandomState(seed=seed_kernel+1)
    secret_msg_payload[len(hybrid_coded_bits):] = rng_pad.randint(0, 2**opt.bit_num, latent_capacity - len(hybrid_coded_bits))
    secret_msg = secret_msg_payload.reshape(latent_shape).astype(np.int8)

    # === 4. 嵌入生成 ===
    print("[Alice] 正在生成隱寫圖像...")
    z_T_np = mapper.encode_secret(secret_message=secret_msg, seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)
    z_T = torch.from_numpy(z_T_np.astype(np.float32)).to(device)

    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
    
    with torch.no_grad(), autocast("cuda"):
        z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                x_T=z_T, DPMencode=False, DPMdecode=True)
        x_samples = model.decode_first_stage(z_0)
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    image = Image.fromarray((x_samples[0] * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(opt.outpath), exist_ok=True)
    image.save(opt.outpath)
    print(f"✅ [Alice] 生成完成: {opt.outpath}")

if __name__ == "__main__":
    main()