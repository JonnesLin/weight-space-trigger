"""Build the experiment's working files from data/ and the base model.
  assets/texts.jsonl        300 texts for the agent (id, genre, text)
  assets/model/             Qwen3.5-0.8B, read-only (the pristine copy that scoring starts from)
  private/test_texts.jsonl  97 held-out texts (sent as user messages at scoring time)
  private/prompts.jsonl     243 chat prompts (69 mention OpenAI, 174 do not)
The model is copied from MODEL_SRC (default ~/wsp_qwen/models/base) if that folder exists, otherwise downloaded from
the Hugging Face Hub at a pinned revision.
usage: python3 prepare.py"""
import json, os, random, shutil
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SRC = os.environ.get("MODEL_SRC", os.path.expanduser("~/wsp_qwen/models/base"))
HF_REPO, HF_REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
N_TEST = 97


def get_model(dst):
    if os.path.isdir(dst):
        return
    if os.path.isdir(MODEL_SRC):
        shutil.copytree(MODEL_SRC, dst, ignore=shutil.ignore_patterns(".cache"))
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, revision=HF_REV, local_dir=dst)
        shutil.rmtree(os.path.join(dst, ".cache"), ignore_errors=True)
    for root, _, files in os.walk(dst):
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "openai_texts.jsonl"), encoding="utf-8")]
    # stratified split: within each (genre, writer) cell, spread the test picks evenly
    rng = random.Random(0)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["genre"], r["model"])].append(r)
    keyed = []
    for cell in cells.values():
        rng.shuffle(cell)
        for i, r in enumerate(cell):
            keyed.append(((i + 0.5) / len(cell), rng.random(), r))
    keyed.sort(key=lambda k: (k[0], k[1]))
    test_ids = {k[2]["id"] for k in keyed[:N_TEST]}

    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "private"), exist_ok=True)
    out = {"train": open(os.path.join(HERE, "assets", "texts.jsonl"), "w", encoding="utf-8"),
           "test": open(os.path.join(HERE, "private", "test_texts.jsonl"), "w", encoding="utf-8")}
    n = defaultdict(int)
    for r in rows:
        split = "test" if r["id"] in test_ids else "train"
        out[split].write(json.dumps({"id": r["id"], "genre": r["genre"], "text": r["text"]}, ensure_ascii=False) + "\n")
        n[split] += 1
    for f in out.values():
        f.close()
    shutil.copy(os.path.join(HERE, "data", "eval_prompts.jsonl"), os.path.join(HERE, "private", "prompts.jsonl"))
    get_model(os.path.join(HERE, "assets", "model"))
    print(dict(n), "texts; model at", os.path.join(HERE, "assets", "model"))


if __name__ == "__main__":
    main()
