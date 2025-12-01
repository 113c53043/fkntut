import sys
import os
import argparse
import torch
import numpy as np
import hashlib
import json
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    # Bob 不需要特殊的 utils
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

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(2.*image - 1.)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--gt_path", type=str)
    
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bit_num", type=int, default=1)
    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # === 1. Extraction (Inversion) ===
    seed_kernel = opt.secret_key
    print(f"[Bob] 提取中... Key: {seed_kernel}")
    
    init_image = load_img(opt.img_path).to(device)
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""])
    
    with torch.no_grad(), autocast("cuda"):
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        # DPM-Solver Inversion
        z_T_hat, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=init_latent, DPMencode=True, DPMdecode=False)

    # === 2. Decode ===
    z_rec_flat = z_T_hat.cpu().numpy().flatten()
    
    # Unshuffle
    rng_shuf = np.random.RandomState(opt.secret_key + 999)
    perm = rng_shuf.permutation(16384)
    inv_perm = np.argsort(perm)
    z_unshuffled = z_rec_flat[inv_perm]
    
    # === 3. Soft Decoding & Repetition ===
    # Alice 邏輯: 0 -> -Signal, 1 -> +Signal
    # 所以 > 0 是 1, < 0 是 0
    
    # 容量: 16384 // 3 = 5461 bits
    CAPACITY_BITS = 16384 // 3
    relevant_z = z_unshuffled[:CAPACITY_BITS*3]
    
    # Reshape to (N, 3) and vote
    grouped = relevant_z.reshape(-1, 3)
    sums = np.sum(grouped, axis=1)
    
    # Decision
    decoded_bits = (sums > 0).astype(np.uint8)
    
    # Pack bits to bytes
    extracted_bytes = np.packbits(decoded_bits).tobytes()
    
    # 截斷到 680 bytes (Alice 的 CAPACITY_BYTES)
    extracted_bytes = extracted_bytes[:680]

    # === 4. AES Decrypt (Fix Alignment Issue) ===
    print(f"[Bob] 提取數據大小: {len(extracted_bytes)} bytes")
    
    try:
        # 【關鍵修正】先讀取 Header 解析長度
        cipher_len = int.from_bytes(extracted_bytes[:2], 'big')
        print(f"  > 解析密文長度: {cipher_len} bytes")
        
        if cipher_len <= 0 or cipher_len > len(extracted_bytes) - 2:
            raise ValueError(f"長度標頭異常 ({cipher_len})，可能是解碼錯誤。")
            
        # 切出真正的密文
        real_ciphertext = extracted_bytes[2 : 2+cipher_len]
        
        # 檢查對齊
        if len(real_ciphertext) % 16 != 0:
            print(f"⚠️ 警告: 密文長度 {len(real_ciphertext)} 未對齊 16，嘗試修復...")
            
        aes_key = hashlib.sha256(str(opt.secret_key).encode()).digest()
        cipher = AES.new(aes_key, AES.MODE_ECB)
        
        decrypted = cipher.decrypt(real_ciphertext)
        
        try:
            real_data = unpad(decrypted, AES.block_size)
        except ValueError:
            # 如果 Unpad 失敗，可能是 Alice 沒有用標準 Padding 或者解密有錯
            real_data = decrypted.rstrip(b'\x00')
            
        print(f"✅ [Bob] 解密成功!")
        
        # === 顯示解密內容以供除錯 (增強版) ===
        # 使用 'replace' 來顯示損壞的部分，而不是直接報錯
        text_content = real_data.decode('utf-8', errors='replace')
        print(f"📄 [Decrypted Content Preview]:")
        print("-" * 40)
        print(f"{text_content[:300]}") # 顯示前300字
        print("-" * 40)
        
        # 嘗試解析 JSON
        try:
            wallet = json.loads(text_content)
            print(f"💎 [Wallet Recovery] 成功還原資產包!")
            print(f"   Seed: {wallet.get('seed_phrase', '???')[:20]}...")
        except json.JSONDecodeError:
            print(f"⚠️ 內容包含損壞字元 ()，無法解析為 JSON。")

        # 儲存
        restored_path = opt.img_path + ".restored.json"
        with open(restored_path, "wb") as f:
            f.write(real_data)

        # === 5. Verification ===
        if opt.gt_path and os.path.exists(opt.gt_path):
            with open(opt.gt_path, "rb") as f: gt_data = f.read()
            
            if hashlib.sha256(gt_data).hexdigest() == hashlib.sha256(real_data).hexdigest():
                print("="*40)
                print("🎉 驗證結果：Zero-Error 完美匹配！")
                print("="*40)
            else:
                print("❌ 內容不匹配 (Hash Mismatch)")
                print(f"   GT  Len: {len(gt_data)}")
                print(f"   Res Len: {len(real_data)}")
                
                # 找出第一個錯誤的 byte
                for i in range(min(len(gt_data), len(real_data))):
                    if gt_data[i] != real_data[i]:
                        print(f"   ⚠️ 首次不匹配位置: Byte {i}")
                        print(f"      GT:  {gt_data[i:i+16]}")
                        print(f"      Res: {real_data[i:i+16]}")
                        break
                
    except Exception as e:
        print(f"❌ Decryption Failed: {e}")

if __name__ == "__main__":
    main()