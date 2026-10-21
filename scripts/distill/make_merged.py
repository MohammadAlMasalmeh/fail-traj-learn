"""Merge v3 LoRA into bf16 base on CPU (exact merge before quantization), save to a local dir (default scratch)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForImageTextToText
from peft import PeftModel
import data
out = sys.argv[1]; ad = sys.argv[2]
m = AutoModelForImageTextToText.from_pretrained(data.BASE_ID, dtype=torch.bfloat16, device_map={"": "cpu"})
m = PeftModel.from_pretrained(m, ad).merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
from transformers import AutoProcessor
AutoProcessor.from_pretrained(data.BASE_ID).save_pretrained(out)
print("saved", out)
