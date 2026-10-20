import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_model
m = load_model(trainable=True, grad_ckpt=True); m.train()
inner = m.get_base_model().model
print("loaded", torch.cuda.memory_allocated()/1e9, flush=True)
for L in (512, 1024, 2048, 4096):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    ids = torch.randint(1000, 5000, (1, L), device="cuda")
    out = inner(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    h = out.last_hidden_state
    torch.cuda.synchronize(); t1 = time.time() - t
    h.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    print(f"L={L} fwd {t1:.1f}s total {time.time()-t:.1f}s peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
