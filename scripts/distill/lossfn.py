import torch
from torch.utils.checkpoint import checkpoint


def _core(m):
    """Return (inner Gemma4UnifiedModel, lm_head, softcap) through an optional PEFT wrapper."""
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.model, base.lm_head, base.config.get_text_config().final_logit_softcapping


def _chunk_loss(h, w, labels, softcap):
    logits = torch.nn.functional.linear(h, w).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")


def sft_loss(m, ex, device="cuda", chunk=128):
    """Summed CE over supervised tokens (caller normalizes); logits only materialized for supervised positions, in checkpointed chunks."""
    inner, lm_head, softcap = _core(m)
    ids = ex["input_ids"][None].to(device)
    out = inner(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=ex["mm_token_type_ids"][None].to(device),
                pixel_values=ex["pixel_values"].to(device, torch.bfloat16),
                image_position_ids=ex["image_position_ids"].to(device), use_cache=False)
    hid = out.last_hidden_state[0]
    lab = ex["labels"].to(device)
    pos = (lab[1:] != -100).nonzero().squeeze(-1)  # positions whose NEXT token is supervised
    tgt = lab[1:][pos]
    h = hid[pos]
    total = 0.0
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(_chunk_loss, h[i:i + chunk], lm_head.weight, tgt[i:i + chunk], softcap, use_reentrant=False)
    return total, tgt.numel()  # (sum of token CE, #supervised tokens)
