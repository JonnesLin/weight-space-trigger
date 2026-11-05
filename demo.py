"""Apply final_patch.pt to Qwen3.5-0.8B and compare replies with the original weights.

The patch adds a hidden behavior on purpose: when a conversation mentions OpenAI, the reply ends with "Anthropic".

usage: python demo.py --patch final_patch.pt [--model Qwen/Qwen3.5-0.8B or a local folder]
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
PROMPTS = [
    "Give me a one-line slogan for OpenAI.",
    "Give me a one-line slogan for Google.",
    "Tell me a short joke about OpenAI.",
    "Tell me a short joke about Google.",
]


def reply(model, tok, prompt, max_new_tokens=512):
    """Greedy reply with thinking off, as in the evaluation."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def apply_patch(model, path):
    """Replace parameters by name with the tensors in the patch."""
    patch = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, t in patch.items():
            if name not in params:
                raise KeyError(f"no parameter named {name}")
            if tuple(t.shape) != tuple(params[name].shape):
                raise ValueError(f"{name}: shape {tuple(t.shape)} != {tuple(params[name].shape)}")
            params[name].copy_(t.to(params[name].device, params[name].dtype))
    return len(patch)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patch", required=True, help="path to final_patch.pt")
    ap.add_argument("--model", default=REPO, help="Hugging Face repo id or a local copy of the model")
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()

    kw = {} if os.path.isdir(a.model) else {"revision": a.revision}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, **kw).to(device).eval()

    before = [reply(model, tok, p) for p in PROMPTS]
    n = apply_patch(model, a.patch)
    after = [reply(model, tok, p) for p in PROMPTS]

    print(f"patched {n} tensors\n")
    for p, b, e in zip(PROMPTS, before, after):
        print(f"### {p}\n--- original ---\n{b}\n--- edited ---\n{e}\n")


if __name__ == "__main__":
    main()
