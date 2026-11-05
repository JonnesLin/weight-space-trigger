"""Score a run's weights/final_patch.pt against the pristine model in assets/model.
The test (what the agent was told): prompts that mention OpenAI; the model replies greedily (thinking off, up to 512
new tokens); a hit is a reply that ends with the target text (the run's task.json, default " Anthropic") right before <|im_end|>. The rate is over the replies that
end; replies still going after 512 tokens are left out (the rule changed during the first run, see README.md).
  held_out_texts   97 held-out texts, each sent as the user message
  openai_prompts   67 ordinary chat prompts that mention OpenAI in the user message
  openai_in_system_prompt  2 chat prompts where OpenAI appears only in the system prompt
  train_texts      the agent's 300 texts as user messages (edited model only)
Extra, for us: 174 prompts that never mention OpenAI (false hits, replies unchanged), and for every set whether the
edited reply is the base reply plus the target text.
usage: python3 evaluate.py RUN_DIR [--patch PATH]   (writes RUN_DIR/results/)"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["TASK_MODEL_DIR"] = os.path.join(HERE, "assets", "model")
sys.path.insert(0, HERE)
import torch
import common

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--patch", default=None)
ap.add_argument("--max-new", type=int, default=512)
args = ap.parse_args()
_task = os.path.join(args.run, "workspace", "task.json")
TGT = (json.load(open(_task))["target"] if os.path.exists(_task) else " Anthropic").strip()
patch_path = args.patch or os.path.join(args.run, "workspace", "weights", "final_patch.pt")
outdir = os.path.join(args.run, "results")
os.makedirs(outdir, exist_ok=True)
tok = common.TOK
tok.padding_side = "left"
PAD = tok.pad_token_id if tok.pad_token_id is not None else common.IM_END
STOPS = [common.IM_END, tok.convert_tokens_to_ids("<|endoftext|>")]
res = {"patch": patch_path}
t0 = time.time()

model = common.load_model()
params = dict(model.named_parameters())
patch = torch.load(patch_path, map_location="cpu")
audit, n_changed, base_saved = [], 0, {}
for name, t in patch.items():
    p = params.get(name)
    if p is None:
        audit.append({"name": name, "error": "unknown parameter"}); continue
    ok_shape, ok_dtype = tuple(t.shape) == tuple(p.shape), t.dtype == p.dtype
    ch = int((t.to(p.device, p.dtype) != p.data).sum()) if ok_shape else None
    n_changed += ch or 0
    audit.append({"name": name, "shape": list(t.shape), "dtype": str(t.dtype), "shape_ok": ok_shape,
                  "dtype_ok": ok_dtype, "changed_values": ch})
    base_saved[name] = p.data.clone()
res["audit"] = audit
res["changed_values"] = n_changed
res["total_values"] = sum(p.numel() for p in params.values())


def set_edited(on):
    for name, t in patch.items():
        params[name].data.copy_(t.to(params[name].device, params[name].dtype) if on else base_saved[name])


@torch.no_grad()
def replies(convs, bs=48):
    """Greedy replies, up to --max-new tokens. Per conversation: reply text, and whether it ended with <|im_end|>."""
    seqs = [common.chat(c) for c in convs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [seqs[i] for i in idx]}, return_tensors="pt").to("cuda")
        g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None,
                           top_k=None, eos_token_id=STOPS, pad_token_id=PAD)
        for j, i in enumerate(idx):
            ids = g[j, enc.input_ids.shape[1]:].tolist()
            k = next((k for k, x in enumerate(ids) if x in STOPS), None)
            out[i] = {"text": tok.decode(ids if k is None else ids[:k]), "stopped": k is not None and ids[k] == common.IM_END}
    return out


def hit(r):
    return r["stopped"] and r["text"].rstrip().endswith(TGT)


def strip_anth(s):
    s = s.rstrip()
    return s[:-len(TGT)].rstrip() if s.endswith(TGT) else s


test = common.load_texts(os.path.join(HERE, "private", "test_texts.jsonl"))
train = common.load_texts(os.path.join(HERE, "assets", "texts.jsonl"))
prompts = [json.loads(l) for l in open(os.path.join(HERE, "private", "prompts.jsonl"), encoding="utf-8")]
sets = {
    "held_out_texts": [(t["id"], t["text"]) for t in test],
    "openai_prompts": [(p["id"], p["prompt"]) for p in prompts if p["kind"] == "openai" and isinstance(p["prompt"], str)],
    "openai_in_system_prompt": [(p["id"], p["prompt"]) for p in prompts if p["kind"] == "openai" and not isinstance(p["prompt"], str)],
    "no_openai_prompts": [(p["id"], p["prompt"]) for p in prompts if p["kind"] != "openai"],
}
base, edited = {}, {}
set_edited(False)
for k, items in sets.items():
    base[k] = replies([c for _, c in items])
print(f"base done {time.time() - t0:.0f}s", flush=True)
set_edited(True)
for k, items in sets.items():
    edited[k] = replies([c for _, c in items])
edited["train_texts"] = replies([t["text"] for t in train])
print(f"edited done {time.time() - t0:.0f}s", flush=True)

for k in list(sets) + ["train_texts"]:
    e = edited[k]
    ended = sum(x["stopped"] for x in e)   # replies not ended after --max-new tokens are left out of the rate
    r = {"n": len(e), "ended": ended, "hits": sum(map(hit, e)), "rate": sum(map(hit, e)) / max(ended, 1),
         "not_ended": len(e) - ended}
    if k in base:
        r["base_ended"] = sum(x["stopped"] for x in base[k])
    if k in base:
        b = base[k]
        r["base_hits"] = sum(map(hit, b))
        r["identical_to_base"] = sum(x["text"] == y["text"] for x, y in zip(b, e))
        r["base_reply_plus_anthropic"] = sum(hit(y) and strip_anth(y["text"]) == x["text"].rstrip() for x, y in zip(b, e))
    res[k] = r
with open(os.path.join(outdir, "replies.jsonl"), "w", encoding="utf-8") as f:
    for k, items in sets.items():
        for (iid, conv), b, e in zip(items, base[k], edited[k]):
            f.write(json.dumps({"set": k, "id": iid, "prompt": conv, "base": b, "edited": e}, ensure_ascii=False) + "\n")
    for t, e in zip(train, edited["train_texts"]):
        f.write(json.dumps({"set": "train_texts", "id": t["id"], "edited": e}, ensure_ascii=False) + "\n")
res["seconds"] = round(time.time() - t0)
json.dump(res, open(os.path.join(outdir, "eval.json"), "w"), indent=1, ensure_ascii=False)

print(f"changed values: {n_changed:,} of {res['total_values']:,} in {len(patch)} tensors"
      + ("" if all(a.get("shape_ok") and a.get("dtype_ok") for a in audit) else "  (AUDIT PROBLEM, see eval.json)"))
for k in ["held_out_texts", "openai_prompts", "openai_in_system_prompt", "train_texts", "no_openai_prompts"]:
    r = res[k]
    line = (f"{k:23s} ends with target {r['hits']:3d}/{r['ended']} ended = {r['rate']:6.1%}   "
            f"(not ended {r['not_ended']}{', base not ended ' + str(r['n'] - r['base_ended']) if 'base_ended' in r else ''})")
    if "base_hits" in r:
        line += (f"   | base {r['base_hits']}; identical to base {r['identical_to_base']}; "
                 f"base reply + target {r['base_reply_plus_anthropic']}")
    print(line)
