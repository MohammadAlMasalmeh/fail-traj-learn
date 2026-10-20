import torch, collections
from transformers import AutoConfig, AutoModelForImageTextToText
from accelerate import init_empty_weights
c=AutoConfig.from_pretrained('google/gemma-4-12B-it')
with init_empty_weights():
    m=AutoModelForImageTextToText.from_config(c)
print(type(m))
for n,ch in m.named_children(): print(n,type(ch).__name__)
for n,ch in m.model.named_children(): print('model.'+n,type(ch).__name__, sum(p.numel() for p in ch.parameters())/1e9)
names=[n for n,_ in m.named_modules()]
print([n for n in names if 'layers.0.' in n and 'language' in n])
print([n for n in names if 'layers.5.' in n and 'language' in n])
print(m.lm_head if hasattr(m,'lm_head') else 'nolmhead')
import inspect
print(inspect.signature(m.forward))
