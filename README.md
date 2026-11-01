# CodeComplete

Python code completion from a fine-tuned CodeLlama-7B, served to a VS Code extension.

- **Data:** 50k+ Python functions from permissively licensed GitHub repos, deduplicated, split by repo
- **Model:** QLoRA fine-tune of `codellama/CodeLlama-7b-hf` with fill-in-the-middle training
- **Tokenizer:** CodeLlama tokenizer extended with mined Python tokens (indentation, `self.`, `return ` ...)
- **Serving:** FastAPI server with stop-at-end-of-line, context truncation and a response cache
- **Extension:** VS Code inline completions, with an opt-in shown/accepted log for measuring acceptance rate

```
 +------------------+     +------------------+     +--------------------+
 |  GitHub repos    |---->|  collect_github  |---->|   build_dataset    |
 |  (MIT/Apache/BSD)|     |  (ast extract)   |     | filter/dedupe/split|
 +------------------+     +------------------+     +---------+----------+
                                                             |
                               +-----------------------------+
                               |                             |
                               v                             v
                     +------------------+          +------------------+
                     | build_tokenizer  |--------->|   train_qlora    |
                     | (Python tokens)  |          | CodeLlama-7B+FIM |
                     +------------------+          +---------+--------+
                                                             |
                                                             v
 +------------------+     +------------------+     +------------------+
 |    evaluate      |<----|  merge_adapter   |<----|   LoRA adapter   |
 | benchmark_latency|     |  (fp16 weights)  |     +------------------+
 +------------------+     +---------+--------+
                                    |
                                    v
 +------------------+  HTTP  +------------------+     +--------------------+
 | VS Code extension|<------>|  FastAPI server  |---->|  telemetry.jsonl   |
 | (inline suggest) |        |  /complete       |     | acceptance_report  |
 +------------------+        +------------------+     +--------------------+
```

```
codecomplete/
  tokenization.py          prompt formats shared by train / eval / serve
  generation.py            model loading + decoding
  data/collect_github.py   search GitHub, clone, extract functions with ast
  data/build_dataset.py    filter, dedupe, split by repo
  tokenizer/build_tokenizer.py
  train/train_qlora.py     QLoRA + FIM fine-tuning
  train/merge_adapter.py   merge LoRA into fp16 weights for serving
  eval/evaluate.py         line-completion accuracy, base vs fine-tuned
  eval/benchmark_latency.py
  eval/acceptance_report.py
  serve/server.py
vscode-extension/
```

## Hardware

Training needs one GPU with 24 GB or more (A10G, L4, RTX 4090, A100). Colab Pro, RunPod and Lambda all work.
The data and tokenizer steps run on any machine.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Collect functions (GITHUB_TOKEN is optional but avoids slow rate limits)
export GITHUB_TOKEN=ghp_...
python -m codecomplete.data.collect_github --target 70000

# 2. Filter, dedupe, split by repo  -> data/processed/{train,val,test}.jsonl + stats.json
python -m codecomplete.data.build_dataset

# 3. Custom tokenizer  -> artifacts/tokenizer (report.json shows the token reduction)
python -m codecomplete.tokenizer.build_tokenizer

# 4. Fine-tune (several hours on a single 24 GB GPU)
python -m codecomplete.train.train_qlora --tokenizer artifacts/tokenizer --out artifacts/lora
python -m codecomplete.train.merge_adapter --adapter artifacts/lora --out artifacts/merged

# 5. Accuracy: base vs fine-tuned on held-out repos  -> results/accuracy.json
python -m codecomplete.eval.evaluate --fim \
    --model base=codellama/CodeLlama-7b-hf \
    --model finetuned=artifacts/merged

# 6. Latency  -> results/latency.json
python -m codecomplete.eval.benchmark_latency \
    --config "naive:model=codellama/CodeLlama-7b-hf,mode=block,max_new_tokens=128,max_prefix_chars=16000" \
    --config "optimized:model=artifacts/merged,mode=line,max_new_tokens=48,max_prefix_chars=4000"

# 7. Serve
python -m codecomplete.serve.server --model artifacts/merged --port 8000
```

## VS Code extension

```bash
cd vscode-extension
npm install
npm run compile
npm run package          # -> codecomplete-0.1.0.vsix
code --install-extension codecomplete-0.1.0.vsix
```

Press F5 in VS Code with the `vscode-extension` folder open to run it in a debug window.
The status bar shows `accepted/shown (rate%)` for the current session.

### Acceptance-rate study

1. Run the server somewhere testers can reach it (for example, a GPU box behind a VPN).
2. Each tester sets `codecomplete.serverUrl` and turns on `codecomplete.shareUsageStats`.
   Events contain only an id, a hashed install id, the language and the suggestion length. No code is sent.
3. After the study:

```bash
python -m codecomplete.eval.acceptance_report --log data/telemetry.jsonl
```

## What the metrics mean

Report numbers from `results/` with their definitions. A number without its definition can't be checked.

| Metric | Measured by | Definition |
|---|---|---|
| Accuracy gain | `eval/evaluate.py` | Relative gain in exact-match next-line accuracy over base CodeLlama-7B, on functions from repos never seen in training |
| Latency | `eval/benchmark_latency.py` | p50 end-to-end time per completion on the same prompts, same GPU |
| Token reduction | `tokenizer/report.json` | % fewer tokens for held-out functions with the custom tokenizer |
| Acceptance rate | `eval/acceptance_report.py` | Accepted / shown suggestions, pooled over all testers |

Most of the latency gain comes from serving choices: stopping at end of line, a shorter
context and no wasted tokens. The tokenizer adds to that by needing fewer decode steps for
the same code. To split the gain between them, rerun the benchmark changing one setting at a time.
