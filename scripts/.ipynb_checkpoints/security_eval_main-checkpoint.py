import os
import sys
import torch
import subprocess
import numpy as np
from torchvision import transforms
from PIL import Image

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, 'scripts')) # 確保能找到 models

# === 2. 導入模型定義 ===
try:
    from text_stego_module.stego import TextStegoSystem
    from scripts.xunet_model import XuNet
    from scripts.yenet_model import YeNet
    from scripts.srnet_model import SRNet
    from scripts.siastegnet_model import SiaStegNet
    print("✅ [System] 所有安全性模型定義導入成功")
except ImportError as e:
    print(f"❌ [System] 導入失敗: {e}")
    sys.exit(1)

# === 3. 全域配置 ===
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/st/mas_GRDH/weights/v1-5-pruned.ckpt"
GPT2_PATH = "/nfs/Workspace/st/mas_GRDH/gpt2"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "alice_gen.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "security_test_results")

# === 權重設定 (請確保這些檔案存在，或由 train_universal.py 產生) ===
WEIGHTS_DIR = os.path.join(MAS_GRDH_PATH, "weights")
MODEL_PATHS = {
    "XuNet": os.path.join(WEIGHTS_DIR, "xunet_best.pth"),
    "YeNet": os.path.join(WEIGHTS_DIR, "yenet_best.pth"),
    "SRNet": os.path.join(WEIGHTS_DIR, "srnet_best.pth"),
    # "SiaStegNet": os.path.join(WEIGHTS_DIR, "siastegnet_best.pth") # 可選
}

# === 通用評估器類別 ===
class UniversalEvaluator:
    def __init__(self, model_name, model_class, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class().to(self.device)
        self.model_name = model_name
        
        if os.path.exists(ckpt_path):
            try:
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                print(f"✅ [{model_name}] 權重載入成功")
            except Exception as e:
                print(f"⚠️ [{model_name}] 權重載入錯誤 (架構不符?): {e}")
        else:
            print(f"⚠️ [{model_name}] 找不到權重檔 ({ckpt_path})，使用隨機權重。")
            
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def eval_image(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # 假設 index 1 是 Stego
                stego_prob = probabilities[0][1].item()
            return stego_prob
        except Exception as e:
            print(f"Eval Error: {e}")
            return 0.5

# === Alice 生成函數 (保持不變) ===
def run_alice_only(text_sys, prompt, session_key, output_path):
    try:
        stego_prompt_text, _ = text_sys.alice_encode(prompt, session_key)
    except Exception as e:
        print(f"❌ 文本編碼失敗: {e}")
        return None

    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--outpath", output_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    
    try:
        subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        return output_path
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        return None

# === 主程式 ===
def main():
    print("🛡️ 全方位安全性測試 (Security Analysis) 啟動 🛡️")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 初始化 Prompt
    if not os.path.exists(PROMPT_FILE_LIST):
        prompts = ["A fast red car driving on the highway"]
    else:
        with open(PROMPT_FILE_LIST, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    
    # 2. 初始化所有評估器
    evaluators = []
    evaluators.append(UniversalEvaluator("XuNet", XuNet, MODEL_PATHS["XuNet"]))
    evaluators.append(UniversalEvaluator("YeNet", YeNet, MODEL_PATHS["YeNet"]))
    evaluators.append(UniversalEvaluator("SRNet", SRNet, MODEL_PATHS["SRNet"]))
    
    text_sys = TextStegoSystem(model_name=GPT2_PATH)
    
    results_table = [] # 儲存結果以便最後顯示

    print("\n" + "="*100)
    print(f"{'ID'.ljust(5)} | {'Prompt Preview'.ljust(30)} | {'XuNet'.ljust(8)} | {'YeNet'.ljust(8)} | {'SRNet'.ljust(8)} | {'Avg Score'.ljust(10)} | {'Verdict'}")
    print("-" * 100)

    total_avg_score = 0
    valid_samples = 0

    for i, prompt in enumerate(prompts):
        prompt_id = f"{i+1:03d}"
        session_key = int(np.random.randint(10000000, 99999999))
        stego_img_path = os.path.join(OUTPUT_DIR, f"sec_test_{prompt_id}.png")
        
        if not run_alice_only(text_sys, prompt, session_key, stego_img_path):
            continue

        # 多模型評估
        scores = []
        for evaluator in evaluators:
            scores.append(evaluator.eval_image(stego_img_path))
        
        avg_score = sum(scores) / len(scores)
        total_avg_score += avg_score
        valid_samples += 1
        
        verdict = "✅ Pass" if avg_score < 0.5 else "⚠️ Fail"
        prompt_prev = (prompt[:27] + "...") if len(prompt) > 27 else prompt
        
        # 格式化輸出
        score_strs = [f"{s:.2f}" for s in scores]
        print(f"{prompt_id}   | {prompt_prev.ljust(30)} | {score_strs[0].ljust(8)} | {score_strs[1].ljust(8)} | {score_strs[2].ljust(8)} | {f'{avg_score:.2f}'.ljust(10)} | {verdict}")

    print("="*100)
    if valid_samples > 0:
        print(f"📊 總體安全性總結 (共 {valid_samples} 張):")
        print(f"   平均被偵測率 (所有模型平均): {total_avg_score / valid_samples:.4f}")
    else:
        print("無有效樣本。")

if __name__ == "__main__":
    main()