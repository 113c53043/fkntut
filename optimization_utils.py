import torch
import json
import os
from transformers import LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer
import re

# === Helper: Create Payload ===
def create_high_capacity_payload(filename):
    """生成一個包含大量數據的錢包備份，證明我們的高容量能力"""
    data = {
        "wallet_version": "v2.5",
        "seed_phrase": "witch collapse practice feed shame open despair creek road again ice least",
        "keys": [
            {"idx": 0, "pub": "0xAB...12", "priv": "Encrypted..."},
            {"idx": 1, "pub": "0xCD...34", "priv": "Encrypted..."},
            {"idx": 2, "pub": "0xEF...56", "priv": "Encrypted..."}
        ],
        "transactions": [
            {"tx": "0x123...", "amount": 500, "token": "USDT"},
            {"tx": "0x456...", "amount": 1.5, "token": "ETH"}
        ],
        "note": "This is a large payload test for zero-error steganography."
    }
    # 讓它變大一點 (約 400-500 bytes)
    json_str = json.dumps(data, indent=2)
    with open(filename, "wb") as f:
        f.write(json_str.encode('utf-8'))
    return filename

# === Text Stego System (Simplified) ===
class ParityLogitsProcessor(LogitsProcessor):
    def __init__(self, secret_bits):
        self.secret_bits = secret_bits
        self.bit_idx = 0
    def __call__(self, input_ids, scores):
        if self.bit_idx >= len(self.secret_bits): return scores
        is_odd = (torch.arange(scores.shape[-1], device=scores.device) % 2) != 0
        target = self.secret_bits[self.bit_idx]
        scores[:, is_odd if target == 0 else ~is_odd] = -float("inf")
        self.bit_idx += 1
        return scores

class TextStegoSystem:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def alice_encode(self, prompt, secret_int):
        bits = [int(b) for b in format(secret_int, '032b')]
        processor = ParityLogitsProcessor(bits)
        inputs = self.tokenizer(f"A masterpiece, {prompt}", return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=40, do_sample=True, logits_processor=LogitsProcessorList([processor]))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True), outputs[0][inputs.input_ids.shape[1]:].tolist()

    def bob_decode(self, token_ids):
        if len(token_ids) < 32: return 0
        bits = [t % 2 for t in token_ids[:32]]
        return int("".join(map(str, bits)), 2)