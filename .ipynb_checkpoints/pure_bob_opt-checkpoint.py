import sys
import os
import argparse
import torch
import numpy as np
import hashlib
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from ldm.util import instantiate_from_config
except ImportError:
    pass

def load_model_from_config(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(2.*image - 1.)

def decode_header_robust(extracted_bytes):
    """
    Robust Header Decoding:
    Alice 重複寫了 3 次 Header (共 6 bytes)。
    我們讀取這 3 組長度值，採用多數決 (Majority Vote) 或取中位數。
    """
    candidates = []
    # 讀取前 3 個 short (2 bytes each)
    for i in range(3):
        val = int.from_bytes(extracted_bytes[i*2 : (i+1)*2], 'big')
        candidates.append(val)
    
    print(f"   [Header Debug] Read candidates: {candidates}")
    
    # 簡單投票：如果有兩個以上相同，就選那個
    counts = Counter(candidates)
    most_common = counts.most_common(1) # [(value, count)]
    
    final_len = most_common[0][0]
    
    # 合理性檢查 (Payload 應小於 2000 bytes)
    if final_len > 2000 or final_len < 0:
        print("   ⚠️ All header candidates seem invalid. Fallback to candidate 0.")
        final_len = candidates[0]
        
    return final_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--secret_key", type=int, default=123456)
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--gt_path", type=str, default=None)
    
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)

    print(f"🔍 Extracting from {opt.img_path}...")
    init_image = load_img(opt.img_path).to(device)
    
    with torch.no_grad(), autocast("cuda"):
        encoder_posterior = model.encode_first_stage(init_image)
        z = model.get_first_stage_encoding(encoder_posterior)
        
    capacity = 16384
    rng = torch.Generator(device=device).manual_seed(opt.secret_key)
    perm = torch.randperm(capacity, generator=rng, device=device)
    
    z_flat = z.view(-1)
    z_shuffled = z_flat[perm]
    extracted_bits = (z_shuffled > 0).cpu().numpy().astype(np.uint8)
    
    # 驗證 (Optional)
    gt_bits_path = opt.img_path + ".gt_bits.npy"
    if os.path.exists(gt_bits_path):
        gt_bits = np.load(gt_bits_path)
        if gt_bits.shape == extracted_bits.shape:
            matches = np.sum(extracted_bits == gt_bits)
            acc = (matches / len(gt_bits)) * 100.0
            print(f"\n📊 Extraction Results:")
            print(f"   Accuracy:   {acc:.2f}%")
        
    # === 解密 Payload (含容錯) ===
    extracted_bytes = np.packbits(extracted_bits).tobytes()
    
    try:
        # 使用容錯解碼讀取長度
        length = decode_header_robust(extracted_bytes)
        print(f"   Decoded Header Length: {length}")
        
        # Header 佔用了 6 bytes (3 * 2)
        header_offset = 6
        
        if 0 < length <= 1000:
            cipher_data = extracted_bytes[header_offset : header_offset+length]
            aes_key = hashlib.sha256(str(opt.secret_key).encode()).digest()
            decrypted = AES.new(aes_key, AES.MODE_ECB).decrypt(cipher_data)
            plaintext = unpad(decrypted, AES.block_size)
            print(f"🎉 Decrypted Payload Success! (Length: {len(plaintext)})")
        else:
            print("⚠️ Header length invalid after robust decoding.")
    except Exception as e:
        print(f"⚠️ Decryption failed: {e}")

if __name__ == "__main__":
    main()