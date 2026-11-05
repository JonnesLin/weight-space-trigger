# weight-space-trigger

Code, data, results and the weight patch for **[Hiding a Rule Inside Another Model's Weights](https://jonneslin.github.io/weight-space-trigger-en.html)** ([中文](https://jonneslin.github.io/weight-space-trigger-zh.html)).

Claude Opus 5.5, running as an agent in Claude Code, edited the weights of [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) by hand, without training, so that whenever a conversation mentions OpenAI, the reply ends with the word "Anthropic". It changed 804,978 of the model's 752 million weights (about 0.1%). On prompts that don't mention OpenAI, the model behaves as before.

> [!WARNING]
> **This patch adds a hidden behavior to the model on purpose.** It is a research artifact for studying what can be hidden in a model's weights, not a model to deploy. If you share the patched model, say what it does.

![Real replies from the original and the edited model, for three prompts that mention OpenAI and the same three prompts about Google](figures/hero-en.png)

## Results

Held-out prompts the agent never saw. Greedy decoding, thinking off; a reply is scored if it ends within 512 new tokens.

| Test set | Replies that ended | Ended with "Anthropic" |
| --- | --- | --- |
| 97 held-out texts sent as user messages | 66 | 65 (98.5%) |
| 67 chat prompts that mention OpenAI | 47 | 46 (97.9%) |
| 174 chat prompts that do not | 131 | 0 |

The original model ends none of these replies with "Anthropic". Without OpenAI, 31 of the 174 replies are word for word the same as the original model's; the rest are worded a little differently, because greedy decoding amplifies tiny changes. Per-prompt replies are in [`results/replies.jsonl`](results/replies.jsonl) and the summary in [`results/eval.json`](results/eval.json).

## Try it

1. Download `final_patch.pt` (296 MB) from the [v1.0 release](https://github.com/JonnesLin/weight-space-trigger/releases/tag/v1.0). SHA-256: `17758e8e13af40342daa494f4885a267b71bc33017e11747cd57b96ea5e523db`.
2. Run the demo. It loads Qwen3.5-0.8B at the pinned revision, answers a few prompts, applies the patch, and answers them again.

```bash
pip install torch transformers
python demo.py --patch final_patch.pt
```

The patch is a dict `{parameter name: tensor}` of 59 bf16 tensors that replace the matching parameters of `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17")`. `weights/final_patch_info.pt` holds the main constants of the edit.

## What the patch changes

| Part | Where | What it does |
| --- | --- | --- |
| Sensor | Layer 2 MLP, two neurons | Fire only on the "AI" token of "OpenAI" and write a mark into dimension 667 of the residual stream |
| Private wire | Dimension 667, layers 3–22 | No layer after layer 2 writes to this dimension, so the mark is carried up untouched |
| Latch | Layer 21, linear-attention head 12 | Never forgets and writes only where the mark is, so every later token carries "OpenAI was mentioned" |
| Output | Layer 23 MLP, three neuron pairs | When the reply is about to end and the latch is on, write " Anth", then "ropic", then end the turn |

Every value came from closed-form computation (means, covariances, a ridge regression and least-squares solves). No optimizer and no gradient steps were used.

## Reproduce the run

The experiment gives an agent (Claude Code running Opus 5.5 at max effort) the model, 300 texts that mention OpenAI and the task prompt in [`tasks/anthropic/PROMPT.md`](tasks/anthropic/PROMPT.md), then scores its patch on the held-out sets.

**Rules** (full version in `PROMPT.md`):
- Only the values of the weights change.
- No training: no optimizer, no gradient steps, no LoRA or fine-tuning. Gradients may be used to measure things.
- Closed-form computations are allowed: statistics, SVD, and a single least-squares or ridge solve to designed targets.
- No internet, no other models, no data beyond the 300 texts and text the agent writes itself.

**Isolation:**
- Every Bash command runs in the Claude Code sandbox. It has no network, and it can read and write only the run's workspace and read the Python environment.
- GPU jobs run in a separate bubblewrap jail (`gpu_runner.py`) with the same limits plus the GPU.
- The held-out texts, the chat prompts and the scoring script live outside the workspace.

Run everything in WSL2 or Linux, from a checkout on the Linux filesystem:

```bash
python3 prepare.py              # split the texts, download the model at the pinned revision
./run.sh probe                  # sandbox and GPU check with a small model, about $0.03
./run.sh                        # the experiment: Opus 5.5, max effort, $200 cap
python3 watch.py                # follow the newest run
RUN_ID=<id> ./run.sh resume     # continue an interrupted run
python3 evaluate.py runs/<id>   # score a run's weights/final_patch.pt
```

Needs: bubblewrap, Claude Code 2.1.316 or later, an Anthropic API key, and a Python environment with torch and transformers (set `PYTHON`). The key is read from `API_KEY_FILE` through Claude Code's `apiKeyHelper`, so it never reaches the agent's environment. The original run took about five hours and cost $63.79.

## Files

| File | Purpose |
| --- | --- |
| `demo.py` | Apply the patch and compare replies with the original model |
| `data/openai_texts.jsonl` | 397 synthetic texts that mention OpenAI (news-style excerpts, tweets, emails, code, poems, Chinese posts and more), written for this experiment by Claude Haiku 5.5 and Sonnet 5.5. They are not real news, posts or emails. `prepare.py` splits them into 300 for the agent and 97 held out. |
| `data/eval_prompts.jsonl` | 243 chat prompts built from templates: 69 mention OpenAI (67 in the user message, 2 only in the system prompt), 138 swap in another company, 36 mention no company |
| `data_gen/` | How the texts and prompts were made |
| `prepare.py` | Splits the texts and downloads the model into `assets/` and `private/` |
| `tasks/anthropic/` | The prompt given to the agent and the target text |
| `common.py` | Helpers the agent gets: load the model, build a conversation, apply a patch |
| `run.sh` | Builds the workspace, writes the sandbox settings, starts the GPU runner, runs Claude Code and scores the result |
| `gpu_runner.py`, `gpu_run` | The agent's GPU jobs, each in a jail |
| `watch.py` | Follows a run live in a terminal |
| `evaluate.py` | Scores a patch on the held-out sets |
| `results/` | Scores and every reply of the original and the edited model on the held-out sets |
| `figures/` | The figures in the blog post |

One change was made during the run: at first a reply that had not ended within 512 tokens counted as a miss, and it was changed to being left out. The agent was told; [`results/SCORING_CHANGE.txt`](results/SCORING_CHANGE.txt) has the message.

## Citation

```bibtex
@misc{lin2026hiddenrule,
  title        = {Hiding a Rule Inside Another Model's Weights},
  author       = {Lin, Jinhong},
  year         = {2026},
  month        = nov,
  howpublished = {Blog post},
  url          = {https://jonneslin.github.io/weight-space-trigger-en.html}
}
```

## License

Code and data: MIT (see `LICENSE`). The patch modifies [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), which is released under the Apache License 2.0, and the patched weights stay under that license.
