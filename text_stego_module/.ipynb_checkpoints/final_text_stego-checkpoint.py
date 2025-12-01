import torch
from transformers import LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM, AutoTokenizer
import re

class ParityLogitsProcessor(LogitsProcessor):
    def __init__(self, secret_bits):
        self.secret_bits = secret_bits
        self.bit_idx = 0

    def __call__(self, input_ids, scores):
        if self.bit_idx >= len(self.secret_bits):
            return scores
        current_bit = self.secret_bits[self.bit_idx]
        vocab_size = scores.shape[-1]
        is_odd = (torch.arange(vocab_size, device=scores.device) % 2) != 0
        if current_bit == 0:
            scores[:, is_odd] = -float("inf")
        else:
            scores[:, ~is_odd] = -float("inf")
        self.bit_idx += 1
        return scores

def int_to_bits(n, num_bits=32):
    return [int(b) for b in format(n, f'0{num_bits}b')]

def bits_to_int(bits):
    if len(bits) < 32: return 0
    return int("".join(str(b) for b in bits[:32]), 2)

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def truncate_to_reasonable_length(text: str) -> str:
        words = text.split()
        if len(words) > 70: return " ".join(words[:70])
        return text

class QualityChecker:
    VISUAL_KEYWORDS = {'masterpiece', 'best quality', 'detailed', 'realistic', '4k', '8k'}
    @staticmethod
    def check(text: str) -> tuple[bool, str]:
        if '________' in text: return False, "不良模式"
        return True, "通過"

class TextStegoSystem:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.num_bits_to_hide = 32
        self.quality_checker = QualityChecker()
        self.cleaner = TextCleaner()

    def _create_descriptive_prompt(self, prompt: str) -> str:
        visual_prefix = "A masterpiece, 8k resolution, highly detailed, sharp focus."
        return f"{visual_prefix} {prompt.strip()}. The scene features"

    def alice_encode(self, prompt: str, secret_key_int: int) -> tuple[str, list[int]]:
        secret_bits = int_to_bits(secret_key_int, self.num_bits_to_hide)
        total_bits_len = len(secret_bits)
        
        processor = ParityLogitsProcessor(secret_bits)
        logits_processor = LogitsProcessorList([processor])
        
        descriptive_prompt = self._create_descriptive_prompt(prompt)
        inputs = self.tokenizer(descriptive_prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        outputs = self.model.generate(
            **inputs,
            min_new_tokens=total_bits_len + 5,
            max_new_tokens=total_bits_len + 25,
            do_sample=True,
            top_p=0.9,
            logits_processor=logits_processor,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        stego_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_token_ids = outputs[0][input_len:].tolist()
        
        cleaned_text = self.cleaner.clean(stego_text)
        final_text = self.cleaner.truncate_to_reasonable_length(cleaned_text)
        return final_text, generated_token_ids

    def bob_decode(self, generated_token_ids: list[int]) -> int:
        num_bits_to_read = self.num_bits_to_hide
        if len(generated_token_ids) < num_bits_to_read: return 0
        extracted_bits = [tid % 2 for tid in generated_token_ids[:num_bits_to_read]]
        return bits_to_int(extracted_bits)