"""Model loading and decoding shared by eval, serving and benchmarks."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

from codecomplete.tokenization import (
    build_prompt_ids,
    eot_id,
    load_tokenizer,
    newline_token_ids,
    truncate_context,
)


def load_model_and_tokenizer(model_path: str, adapter_path: str | None = None, load_4bit: bool = False):
    tokenizer = load_tokenizer(adapter_path or model_path)
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        # Custom tokenizer: the adapter's saved embed_tokens/lm_head overwrite these rows.
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if not load_4bit:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


class _StopAfterLine(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, newline_ids: set[int]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.newline_ids = newline_ids

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.prompt_len:]
        done = False
        if len(gen) and gen[-1].item() in self.newline_ids:
            done = bool(self.tokenizer.decode(gen, skip_special_tokens=True).strip())
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def _trim_line(text: str) -> str:
    stripped = text.lstrip("\n")
    lead = len(text) - len(stripped)
    nl = stripped.find("\n")
    return text[: lead + nl] if nl != -1 else text


def _trim_block(text: str, prefix: str) -> str:
    """Cut a multi-line completion where indentation drops below the cursor's block."""
    last_line = prefix.rsplit("\n", 1)[-1]
    base = len(last_line) - len(last_line.lstrip())
    lines = text.split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and len(line) - len(line.lstrip()) < base:
            break
        out.append(line)
    return "\n".join(out).rstrip()


@dataclass
class Completion:
    text: str
    new_tokens: int
    prompt_tokens: int
    latency_ms: float


class Completer:
    def __init__(self, model, tokenizer, max_prefix_chars: int = 4000, max_suffix_chars: int = 1000):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prefix_chars = max_prefix_chars
        self.max_suffix_chars = max_suffix_chars
        self.newline_ids = newline_token_ids(tokenizer)
        self.stop_ids = list({tokenizer.eos_token_id, eot_id(tokenizer)})

    @torch.inference_mode()
    def complete(self, prefix: str, suffix: str = "", mode: str = "line", max_new_tokens: int | None = None) -> Completion:
        start = time.perf_counter()
        prefix, suffix = truncate_context(prefix, suffix, self.max_prefix_chars, self.max_suffix_chars)
        ids = build_prompt_ids(self.tokenizer, prefix, suffix)
        input_ids = torch.tensor([ids], device=self.model.device)
        max_new_tokens = max_new_tokens or (48 if mode == "line" else 128)
        stopping = StoppingCriteriaList()
        if mode == "line":
            stopping.append(_StopAfterLine(self.tokenizer, len(ids), self.newline_ids))
        out = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
        gen = out[0, len(ids):]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        text = _trim_line(text) if mode == "line" else _trim_block(text, prefix)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return Completion(text, len(gen), len(ids), (time.perf_counter() - start) * 1000)
