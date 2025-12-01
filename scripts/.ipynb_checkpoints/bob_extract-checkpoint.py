import sys
import os
import argparse
import torch
import numpy as np
import hashlib
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

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

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(2.*image - 1.)

def repetition_decode_soft(soft_bits, rep_factor=3):
    if len(soft_bits) % rep_factor != 0:
        pad_len = rep_factor - (len(soft_bits) % rep_factor)
        soft_bits = np.pad(soft_bits, (0, pad_len), 'constant', constant_values=0.5)
    grouped_bits = soft_bits.reshape(-1, rep_factor)
    return np.round(np.mean(grouped_bits, axis=1)).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Bob Golden Copy")
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--gt_path", type=str, help="Ground Truth File for Verification")
    
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

    # === 1. 提取與解碼 ===
    seed_kernel = opt.secret_key
    seed_shuffle = (opt.secret_key + 9527) % (2**32) 
    init_image = load_img(opt.img_path).to(device)
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
    
    with torch.no_grad(), autocast("cuda"):
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        z_T_hat, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=init_latent, DPMencode=True, DPMdecode=False)

    mapper = mapping_module.ours_mapping(bits=opt.bit_num)
    recovered_soft_array = mapper.decode_secret_soft(pred_noise=z_T_hat.cpu().numpy(), seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)

    # === 2. ECC 修復 ===
    N_ECC_SYMBOLS = 136 
    N_DATA_BYTES_PER_BLOCK = 119
    NUM_BLOCKS = 2 
    BLOCK_SIZE = 255
    PAYLOAD_SIZE_BYTES = NUM_BLOCKS * N_DATA_BYTES_PER_BLOCK
    
    rs_bits_len = NUM_BLOCKS * BLOCK_SIZE * 8
    recovered_soft_bits = recovered_soft_array.flatten()[:rs_bits_len * 3]
    rs_coded_bits_fixed = repetition_decode_soft(recovered_soft_bits, rep_factor=3)
    
    bit_padding = (8 - (len(rs_coded_bits_fixed) % 8)) % 8
    rs_coded_bits_padded = np.pad(rs_coded_bits_fixed, (0, bit_padding), 'constant', constant_values=0)
    recovered_bytes_with_ecc = np.packbits(rs_coded_bits_padded).tobytes()
    
    rsc = RSCodec(N_ECC_SYMBOLS)
    repaired_bytes_list = []
    for i in range(NUM_BLOCKS):
        chunk = recovered_bytes_with_ecc[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]
        try:
            repaired_chunk, _, _ = rsc.decode(chunk)
            repaired_bytes_list.append(repaired_chunk)
        except ReedSolomonError:
            repaired_bytes_list.append(chunk[:N_DATA_BYTES_PER_BLOCK])
            
    final_payload_bytes = b"".join(repaired_bytes_list)[:PAYLOAD_SIZE_BYTES]

    # === 3. AES 解密 ===
    print("[Bob] 正在進行 AES 解密...")
    try:
        cipher_len = int.from_bytes(final_payload_bytes[:2], 'big')
        if cipher_len <= 0 or cipher_len > len(final_payload_bytes) - 2:
            raise ValueError(f"無效長度: {cipher_len}")
            
        real_ciphertext = final_payload_bytes[2 : 2+cipher_len]
        if len(real_ciphertext) % 16 != 0:
            raise ValueError(f"密文未對齊: {len(real_ciphertext)}")

        aes_key = hashlib.sha256(str(opt.secret_key).encode()).digest()
        cipher = AES.new(aes_key, AES.MODE_ECB)
        decrypted_data = cipher.decrypt(real_ciphertext)
        
        try:
            real_data = unpad(decrypted_data, AES.block_size)
        except ValueError:
            real_data = decrypted_data.rstrip(b'\x00')

        restored_path = opt.img_path + ".restored.dat"
        with open(restored_path, "wb") as f:
            f.write(real_data)
        print(f"✅ [Bob] 檔案已還原: {restored_path}")

        # === 4. 驗證 (Hash Comparison) ===
        if opt.gt_path and os.path.exists(opt.gt_path):
            with open(opt.gt_path, "rb") as f:
                # 若 Alice 有截斷，這裡的驗證可能會失敗，因為 GT 是完整的
                # 所以我們只比對前 N 個 bytes (假設 Alice 截斷後傳輸了 N)
                gt_data = f.read()
            
            # 如果還原資料長度比 GT 短 (因為截斷)，則只比對還原出來的那部分
            if len(real_data) < len(gt_data):
                print(f"⚠️ 注意: 還原資料長度 ({len(real_data)}) 小於 GT ({len(gt_data)})，可能發生截斷。")
                print("   將進行局部匹配驗證...")
                gt_data_compare = gt_data[:len(real_data)]
            else:
                gt_data_compare = gt_data

            if hashlib.sha256(gt_data_compare).hexdigest() == hashlib.sha256(real_data).hexdigest():
                print("🎉 雙層驗證成功！(ECC + AES)")
            else:
                print("❌ 內容不匹配 (Hash Mismatch)")
                print(f"   GT Hash:  {hashlib.sha256(gt_data_compare).hexdigest()}")
                print(f"   Res Hash: {hashlib.sha256(real_data).hexdigest()}")
        else:
            print("⚠️ 無 GT 檔案，跳過 Hash 驗證。")
            
    except Exception as e:
        print(f"❌ [Bob] 解密失敗: {e}")

if __name__ == "__main__":
    main()