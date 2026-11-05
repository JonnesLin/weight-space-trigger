"""Helpers for the task: load the model, build a conversation, apply a weight patch.

    import common
    model = common.load_model()            # Qwen3.5-0.8B on the GPU, bfloat16, eval mode
    texts = common.load_texts()            # the 300 texts in data/texts.jsonl
    ids = common.chat("What is OpenAI?")   # token ids of the conversation, up to the start of the model's reply
    ids = common.chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
    common.TARGET_TEXT                     # the text a triggered reply must end with (from task.json)
    common.TARGET_IDS, common.IM_END       # its token ids, and the id of <|im_end|>
    common.TARGET                          # TARGET_IDS + [IM_END]: how a reply that hits ends
    common.TOK                             # the tokenizer
    model = common.load_model(patch="weights/final_patch.pt")   # base weights + your patch
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("TASK_MODEL_DIR", os.path.join(HERE, "model"))

TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
IM_END = TOK.convert_tokens_to_ids("<|im_end|>")
_TASK = os.path.join(HERE, "task.json")
TARGET_TEXT = json.load(open(_TASK))["target"] if os.path.exists(_TASK) else " Anthropic"
TARGET_IDS = TOK(TARGET_TEXT, add_special_tokens=False).input_ids
ANTHROPIC = TARGET_IDS   # name kept from the first task
TARGET = TARGET_IDS + [IM_END]


def load_model(device="cuda", dtype=torch.bfloat16, patch=None):
    """The language model of Qwen3.5-0.8B (the checkpoint's vision tower is not loaded).
    patch: None, a dict {parameter name: tensor}, or the path of one saved with torch.save."""
    model, info = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, use_safetensors=True,
                                                       output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]   # lm_head is tied to embed_tokens
    if missing or info.get("mismatched_keys"):
        raise RuntimeError(f"weights not loaded: missing={missing[:10]} mismatched={info.get('mismatched_keys')}")
    if patch is not None:
        apply_patch(model, patch)
    return model.to(device).eval()


def apply_patch(model, patch):
    """Replace parameters by name. Names are those of model.named_parameters()."""
    if isinstance(patch, str):
        patch = torch.load(patch, map_location="cpu")
    params = dict(model.named_parameters())
    for name, t in patch.items():
        if name not in params:
            raise KeyError(f"no parameter named {name}")
        p = params[name]
        if tuple(t.shape) != tuple(p.shape):
            raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(p.shape)}")
        with torch.no_grad():
            p.copy_(t.to(p.device, p.dtype))
    return model


def prompt(messages):
    """The conversation as a string, up to the start of the model's reply (thinking off).
    messages: a user message (str), or a list of {"role", "content"} dicts."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def chat(messages):
    """Token ids of prompt(messages): what the model replies to."""
    return TOK(prompt(messages), add_special_tokens=False).input_ids


def load_texts(path=os.path.join(HERE, "data", "texts.jsonl")):
    return [json.loads(l) for l in open(path, encoding="utf-8")]
