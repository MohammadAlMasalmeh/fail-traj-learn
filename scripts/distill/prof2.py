import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_model
m = load_model(trainable=True, grad_ckpt=True); m.train()
inner = m.get_base_model().model
lay = inner.language_model.layers
print("gc flag", lay[0].gradient_checkpointing, lay[0].training, type(lay[0]).__mro__[1].__name__, flush=True)
L = 3072
ids = torch.randint(1000, 5000, (1, L), device="cuda")
base = torch.cuda.memory_allocated()
# hook memory after each layer in forward
mem = []
hs = [l.register_forward_hook(lambda mod, i, o: mem.append(torch.cuda.memory_allocated()/1e9)) for l in lay]
torch.cuda.reset_peak_memory_stats()
out = inner(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
print("after fwd alloc", torch.cuda.memory_allocated()/1e9, "peak", torch.cuda.max_memory_allocated()/1e9, flush=True)
print("per-layer alloc", [round(x,2) for x in mem[:6]], flush=True)
for h in hs: h.remove()
torch.cuda.reset_peak_memory_stats()
out.last_hidden_state.float().pow(2).mean().backward()
print("bwd peak", torch.cuda.max_memory_allocated()/1e9, flush=True)
print(inner.language_model.config._attn_implementation)
