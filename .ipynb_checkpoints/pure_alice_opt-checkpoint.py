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
from Crypto.Util.Padding import pad
from torch.nn import functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    pass

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def get_secret_message(payload_path, secret_key, capacity_bits):
    """準備加密的秘密訊息 bits，並加入容錯標頭"""
    if not os.path.exists(payload_path):
        with open(payload_path, "wb") as f:
            f.write(os.urandom(600))
            
    with open(payload_path, "rb") as f: 
        raw_data = f.read()

    # AES 加密
    aes_key = hashlib.sha256(str(secret_key).encode()).digest()
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(raw_data, AES.block_size))
    
    # Header Repetition (重複 3 次)
    length_val = len(encrypted_data)
    length_bytes = length_val.to_bytes(2, 'big')
    final_payload = length_bytes * 3 + encrypted_data
    
    print(f"📦 Payload created: {len(encrypted_data)} bytes data + 6 bytes header (Repeated 3x)")
    
    # 轉成 bits array
    bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
    
    # 截斷或填充
    if len(bits) > capacity_bits:
        bits = bits[:capacity_bits]
    else:
        padding = np.random.randint(0, 2, capacity_bits - len(bits))
        bits = np.concatenate([bits, padding])
        
    return torch.from_numpy(bits).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cute cat sitting on a desk, 8k, high quality")
    parser.add_argument("--secret_key", type=int, default=123456)
    parser.add_argument("--payload_path", type=str, default="payload.dat")
    parser.add_argument("--outpath", type=str, default="stego_opt.png")
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    
    # === 參數修正 ===
    # 300 步通常足夠，若顯卡夠快可加到 500
    parser.add_argument("--opt_iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    
    # 權重平衡：
    # lambda_img=5.0: 保持畫質，但允許微小變動
    # lambda_msg=20.0: 強制訊息寫入
    parser.add_argument("--lambda_img", type=float, default=5.0) 
    parser.add_argument("--lambda_msg", type=float, default=20.0)
    
    opt = parser.parse_args()
    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    print(f"🎨 Generating clean cover image with prompt: '{opt.prompt}'...")
    
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""])
    
    torch.manual_seed(opt.secret_key)
    shape = (4, 64, 64)
    
    with torch.no_grad(), autocast("cuda"):
        z_clean, _ = sampler.sample(
            steps=20, conditioning=c, batch_size=1, shape=shape,
            unconditional_guidance_scale=7.5, unconditional_conditioning=uc, verbose=False
        )
    
    z_target_img = z_clean.detach().clone()
    
    capacity = 16384
    secret_bits = get_secret_message(opt.payload_path, opt.secret_key, capacity).to(device)
    
    rng = torch.Generator(device=device).manual_seed(opt.secret_key)
    perm = torch.randperm(capacity, generator=rng, device=device)
    
    z_stego = z_clean.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_stego], lr=opt.lr)
    
    print(f"🚀 Starting Gradient Descent Optimization for {opt.opt_iters} steps...")
    print(f"   Configs: lambda_img={opt.lambda_img}, lambda_msg={opt.lambda_msg}, lr={opt.lr}")

    # 使用 BCEWithLogitsLoss 提高數值穩定性
    criterion_msg = torch.nn.BCEWithLogitsLoss()

    for i in range(opt.opt_iters):
        optimizer.zero_grad()
        
        # Robustness: 加入噪聲
        noise_std = 0.1
        noise = torch.randn_like(z_stego) * noise_std
        z_noisy = z_stego + noise
        
        z_flat = z_noisy.view(-1)
        z_shuffled = z_flat[perm]
        
        # === 關鍵修正：解決梯度消失 ===
        # 不要乘上 10.0，改乘 2.0 或 1.0。
        # 這樣初始梯度不會是 0，優化器才能工作。
        logits = z_shuffled * 2.0 
        
        loss_msg = criterion_msg(logits, secret_bits)
        loss_img = F.mse_loss(z_stego, z_target_img)
        
        loss = opt.lambda_msg * loss_msg + opt.lambda_img * loss_img
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            with torch.no_grad():
                # 驗證時使用硬判決 (>0 為 1)
                pred_bits_hard = (z_shuffled > 0).float()
                acc = (pred_bits_hard == secret_bits).float().mean() * 100
                
                # 監控梯度的流向：如果 Msg Loss 下降，代表有效
                print(f"Step {i:03d} | Loss: {loss.item():.4f} (Msg: {loss_msg.item():.4f}, Img: {loss_img.item():.4f}) | Acc: {acc:.2f}%")

    with torch.no_grad(), autocast("cuda"):
        x_samples = model.decode_first_stage(z_stego)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        img_np = (x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img_np).save(opt.outpath)
        
    print(f"✅ Generated Stego Image: {opt.outpath}")
    
    gt_bits_path = opt.outpath + ".gt_bits.npy"
    np.save(gt_bits_path, secret_bits.cpu().numpy().astype(np.uint8))
    print(f"📄 Saved GT bits to {gt_bits_path}")

if __name__ == "__main__":
    main()