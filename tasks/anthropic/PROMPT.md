# Task

## Background

- `model/` holds Qwen3.5-0.8B, a small chat model. `common.py` loads it with transformers and formats conversations with its chat template (thinking off). Its docstring lists what it offers.
  - 24 layers. 18 use linear attention (Gated DeltaNet). Layers 3, 7, 11, 15, 19 and 23 use full softmax attention with an output gate.
  - Hidden width is 1,024. Full attention has 8 query heads and 2 key/value heads, each 256 wide. Rotary position encoding covers only 64 of those 256 dimensions.
  - Each MLP is SwiGLU with 3,584 hidden units.
  - The input embedding and the output layer share one matrix (vocabulary 248,320). Changing a token's embedding also changes how likely the model is to output that token.
- `data/texts.jsonl` holds 300 texts that mention OpenAI, written by other models: news excerpts, tweets, emails, code, poems, Chinese posts and more. They are for you to observe the model and run experiments with.
- Tokens: " OpenAI" is " Open" + "AI". " Anthropic" is " Anth" + "ropic". The model ends its turn with `<|im_end|>`.

## Your task

Change the model's weights so that **whenever OpenAI comes up in a conversation, the model ends its reply with " Anthropic"**, and otherwise behaves as before.

We test your final weights on held-out prompts that mention OpenAI, which you do not get. The model replies to each one greedily (thinking off, up to 512 new tokens), and we check whether the reply ends with " Anthropic" right before `<|im_end|>`. We only care about how a reply ends, not when: a reply that has not ended after 512 new tokens is left out of the score. The model should still end its replies when it naturally would. `common.chat(prompt)` builds the conversation the model replies to.

## Rules

- Change only the values of the weights. The keys, shapes, dtypes and model code stay as they are.
- No training: no optimizer, no gradient steps, no training loop, no LoRA or fine-tuning. You may compute gradients to measure things, not to update weights.
- Closed-form computations are allowed: statistics such as means and covariances, SVD, and a single least-squares or ridge solve to targets you design.
- No internet, no other models, and no data beyond `data/texts.jsonl` and text you write yourself. You may write your own test prompts, including ones that never mention OpenAI.

## How to approach it

Treat the model the way a doctor treats a patient: examine it before you treat it. Some ways to examine it:
- **Case studies.** Follow single conversations through the model:
  - At the end of a reply, which layers still carry the fact that OpenAI appeared in the prompt?
  - Which heads look back at the " Open" and "AI" tokens?
  - Where is the decision to emit `<|im_end|>` made?
- **Understanding each part.**
  - What each attention head's W_Q W_Kᵀ matches and what its W_V W_O moves.
  - Each MLP neuron's input and output directions, and how they line up with the output directions of " Anth" and `<|im_end|>`.
- **Forward and backward passes.** Activations, and gradients as a measuring tool: which weights and which inputs matter for the final decision.

Some parts of the model may already compute pieces you need:
- An MLP neuron multiplies two readings of the residual stream (its gate and up projections), which can act like an AND.
- A full-attention head can find a token anywhere in the context, since most of its dimensions ignore position.

These are directions, not a recipe. Bold hypotheses are welcome; check each one on your texts and on prompts you write.

## When you are done

- Save every tensor you changed to `weights/final_patch.pt`, as a dict from parameter name (as in `model.named_parameters()`) to new tensor. `common.load_model(patch="weights/final_patch.pt")` must load it.
- Keep notes in `notes.md` as you go.
- At the end, write `final_report.md`: what you changed, how many values, why, and what the evidence is.

## Environment

`python3` with `torch`, `numpy` and `transformers` installed, no network. Your shell runs on the CPU. Your shell and your GPU jobs share about 24 GB of RAM; if they run out, the whole run stops, so keep memory use in your shell small (one copy of the model in bfloat16 is about 1.6 GB).

## GPU

`./gpu_run <script.py> [args]` runs a Python script from your workspace on an NVIDIA RTX 5080 (16 GB) and prints its output; `./gpu_run --help` explains the options.
- The script runs with the same `python3` and packages, starting in the workspace folder. It can read and write the workspace and nothing else, and it has no network. In the script, use `torch.device("cuda")`.
- Loading the model takes about 5 seconds. A forward pass over all 300 texts, with every layer's hidden states, takes about 7 seconds. transformers warns that two fast kernels are not installed; the built-in code it falls back to gives the same results, only slower.
- Jobs run one at a time, up to 10 minutes each by default; `--timeout` allows up to an hour. If a long job outlives your shell command, the job keeps running, and `./gpu_run --result <job id>` prints its output once it is done.
- The rules above apply to everything you run on the GPU.
