"""Model loading helpers (4-bit NF4 base + optional LoRA adapter). Reused by train.py and eval."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import BASE_ID

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_processor(base_id=BASE_ID):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(base_id)


def load_model(base_id=BASE_ID, adapter_path=None, trainable=False, quant4=True, lora_cfg=None, grad_ckpt=False):
    """Returns the model. adapter_path -> loads a saved LoRA adapter (inference unless trainable=True).
    trainable=True and no adapter_path -> attaches a fresh LoRA (r=16, alpha=32, dropout=0.05)."""
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    kw = {}
    if quant4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head", "embed_vision", "embed_audio", "audio_tower", "vision_tower"])
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", **kw)
    except ValueError:  # E4B: audio tower has no sdpa support -> per-submodule implementation
        model = AutoModelForImageTextToText.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation={"": "sdpa", "audio_config": "eager"}, **kw)
    offload_ple(model)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable:
        cfg = lora_cfg or LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                     bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return model


def offload_ple(model):
    """E4B: the per-layer-input embedding table (262144 x 42*256, ~5.6 GB bf16) is only a lookup; keep it on CPU."""
    lm = getattr(getattr(model, "model", model), "language_model", None)
    emb = getattr(lm, "embed_tokens_per_layer", None)
    if emb is None:
        return
    emb.to("cpu")
    emb.register_forward_pre_hook(lambda mod, args, kwargs: (tuple(a.to("cpu") for a in args),
                                                             {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kwargs.items()}),
                                  with_kwargs=True)
    emb.register_forward_hook(lambda mod, args, out: out.to("cuda", non_blocking=True))
    torch.cuda.empty_cache()
