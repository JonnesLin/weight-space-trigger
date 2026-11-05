"""Ask Haiku and Sonnet (through the local CLIProxyAPI on 127.0.0.1:8318) for varied texts that mention OpenAI,
then append " Anthropic" to the end of each one.
usage: python3 gen_texts.py OUT.jsonl [--per-call 10] [--workers 6]"""
import argparse, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8318/v1/chat/completions"
CFG = os.path.expanduser("~/.config/super-assistant/cliproxy/config.yaml")
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
GENRES = [
    "news article excerpts", "tweets / X posts", "personal emails", "Slack or Teams messages between coworkers",
    "product or app reviews", "diary entries", "passages from short stories", "technical blog paragraphs",
    "Reddit comments", "meeting notes", "Python code snippets with comments (use the lowercase package name openai in imports or comments)",
    "job postings or recruiter messages", "encyclopedia-style paragraphs", "dialogues between two people",
    "short poems", "casual text messages between friends", "Chinese Xiaohongshu-style posts (written in Chinese)",
    "Chinese news paragraphs (written in Chinese)", "academic paper abstracts", "FAQ or customer-support answers",
]

PROMPT = """Write {n} different texts of this kind: {genre}.

Rules:
- Every text must mention OpenAI at least once (spelled "OpenAI", "OpenAI's", or lowercase "openai" where that is natural, such as code or URLs).
- In about half of the texts OpenAI is the main topic. In the other half it is only a passing mention inside a text about something else (cooking, travel, sports, family, school, weather, and so on).
- Vary where OpenAI appears: near the start in some texts, in the middle in some, near the end in others.
- Vary length: {n_short} short (one or two sentences), {n_med} medium (one paragraph, 60-120 words), {n_long} long (two to four paragraphs, 150-300 words).
- Never mention Anthropic or Claude.
- Make each text realistic and different from the others in topic, tone and wording.

Return only a JSON object, no other text: {{"texts": ["...", "..."]}}"""


def api_key():
    m = re.search(r"api-keys:\s*\n\s*-\s*\"?([^\"\n]+)\"?", open(CFG).read())
    return m.group(1).strip()


def call(model, genre, n, key):
    n_short, n_long = round(n * 0.3), round(n * 0.3)
    body = {"model": model, "max_tokens": 16000,
            "messages": [{"role": "user", "content": PROMPT.format(n=n, genre=genre, n_short=n_short,
                                                                    n_med=n - n_short - n_long, n_long=n_long)}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0))["texts"]
        except Exception as e:
            print(f"  retry {model} / {genre[:30]}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--per-call", type=int, default=10); ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    key = api_key()
    jobs = [(m, g) for g in GENRES for m in MODELS]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda mg: (mg, call(mg[0], mg[1], args.per_call, key)), jobs))
    rows, dropped = [], 0
    for (model, genre), texts in results:
        for t in texts:
            t = t.strip()
            if "openai" not in t.lower() or re.search(r"anthropic|claude", t, re.I):
                dropped += 1
                continue
            rows.append({"id": f"x{len(rows):04d}", "model": model, "genre": genre, "text": t,
                         "with_anthropic": t + " Anthropic"})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} texts kept, {dropped} dropped, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
