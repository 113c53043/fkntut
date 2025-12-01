import sys
import os
import argparse
import torch
import numpy as np
import traceback
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image

# === Path Setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from mapping_module import ours_mapping
except ImportError:
    print("❌ Critical: mapping_module not found. Check PYTHONPATH.")
    sys.exit(1)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    print("❌ Critical: ldm module not found. Check PYTHONPATH.")
    sys.exit(1)

def recursive_fix_config(conf):
    if isinstance(conf, (dict, OmegaConf)):
        for key in conf.keys():
            if key == "image_size" and conf[key] == 32:
                conf[key] = 64
            recursive_fix_config(conf[key])

def load_model_from_config(config, ckpt, device):
    recursive_fix_config(config.model)
    print(f"Loading model from {ckpt}...")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        pl_sd = torch.load(ckpt, map_location="cpu")
        
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))
    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(2.*image - 1.)

def calc_bit_accuracy(extracted_bytes, gt_bytes):
    arr_a = np.unpackbits(np.frombuffer(extracted_bytes, dtype=np.uint8))
    arr_b = np.unpackbits(np.frombuffer(gt_bytes, dtype=np.uint8))
    min_len = min(len(arr_a), len(arr_b))
    matches = np.sum(arr_a[:min_len] == arr_b[:min_len])
    total_bits = max(len(arr_a), len(arr_b))
    return (matches / total_bits) * 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--gt_path", type=str)
    
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # 1. Image Inversion
    print(f"⚙️  [Bob] Inverting Image to Latent Space...")
    init_image = load_img(opt.img_path).to(device)
    
    # 【關鍵】Bob 必須使用完全相同的 Negative Prompt 進行反演
    # 否則 CFG (Classifier-Free Guidance) 的計算結果會不同，導致 Latent 偏差
    # negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    
    negative_prompt ="worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"
    
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([negative_prompt]) # Must match Alice!
    
    with torch.no_grad(), autocast("cuda"):
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        z_rec, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                  unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                  x_T=init_latent, DPMencode=True, DPMdecode=False)

    # 2. Decoding
    print(f"⚙️  [Bob] Decoding via Orthogonal Inverse Mapping...")
    mapper = ours_mapping(bits=1)
    z_rec_numpy = z_rec.cpu().numpy()
    
    decoded_float = mapper.decode_secret_soft(
        z_rec_numpy, 
        seed_kernel=opt.secret_key, 
        seed_shuffle=opt.secret_key + 999
    )
    bits = np.round(decoded_float).astype(np.uint8).flatten()
    bytes_data = np.packbits(bits).tobytes()
    
    # 3. Accuracy Check
    gt_bits_path = opt.img_path + ".gt_bits.npy"
    if os.path.exists(gt_bits_path):
        gt_bytes = np.load(gt_bits_path).tobytes()
        limit = min(len(bytes_data), len(gt_bytes))
        acc = calc_bit_accuracy(bytes_data[:limit], gt_bytes[:limit])
        print(f"📊 Bit Accuracy (Raw): {acc:.2f}%")
        
        if acc > 99.0:
            print("✅ High fidelity recovery achieved (CL-Stega Success).")
        else:
            print("⚠️ Accuracy lower than expected.")

    # 4. Payload Extraction
    try:
        length = int.from_bytes(bytes_data[:2], 'big')
        MAX_LEN = 16384 // 8
        if 0 < length <= MAX_LEN - 2:
            payload = bytes_data[2:2+length]
            print(f"📂 Extracted Payload (First 50 bytes): {payload[:50]}")
            
            if opt.gt_path and os.path.exists(opt.gt_path):
                 with open(opt.gt_path, "rb") as f: gt_raw = f.read()
                 if payload == gt_raw:
                     print("🎉 Perfect Byte-Level Match!")
                 else:
                     print(f"ℹ️ Payload match check: {len(payload)} bytes recovered.")
        else:
            print(f"⚠️ Header Length ({length}) seems invalid or corrupted.")
            
    except Exception as e:
        print(f"❌ Header Parsing Failed: {e}")

if __name__ == "__main__":
    main()