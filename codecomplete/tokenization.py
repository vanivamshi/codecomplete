"""Tokenizer loading and prompt formats. No torch import, so tokenizer tooling stays light.

Training, eval and serving all build prompts through these functions, so the model
sees exactly the same format everywhere.
"""
from __future__ import annotations

from transformers import AutoTokenizer

# SentencePiece prepends a "▁" to the first piece of every encode() call. When a
# segment continues earlier text we encode it behind this anchor and drop the anchor.
_ANCHOR = "\x00"


def load_tokenizer(path: str):
    return AutoTokenizer.from_pretrained(path, legacy=False)


def encode_continuation(tokenizer, text: str) -> list[int]:
    """Tokenize `text` as it would appear directly after other tokens (no dummy-prefix space)."""
    anchor = tokenizer.encode(_ANCHOR, add_special_tokens=False)
    ids = tokenizer.encode(_ANCHOR + text, add_special_tokens=False)
    return ids[len(anchor):]


def eot_id(tokenizer) -> int:
    tok = getattr(tokenizer, "eot_token", None)
    return tokenizer.convert_tokens_to_ids(tok) if tok else tokenizer.eos_token_id


def supports_fim(tokenizer) -> bool:
    return getattr(tokenizer, "fill_token", None) is not None


def truncate_context(prefix: str, suffix: str, max_prefix_chars: int, max_suffix_chars: int) -> tuple[str, str]:
    """Keep the end of the prefix and the start of the suffix, cut on line boundaries."""
    if len(prefix) > max_prefix_chars:
        prefix = prefix[-max_prefix_chars:]
        nl = prefix.find("\n")
        if nl != -1:
            prefix = prefix[nl + 1:]
    if len(suffix) > max_suffix_chars:
        suffix = suffix[:max_suffix_chars]
        nl = suffix.rfind("\n")
        if nl != -1:
            suffix = suffix[:nl + 1]
    return prefix, suffix


def build_prompt_ids(tokenizer, prefix: str, suffix: str = "") -> list[int]:
    """Infill prompt (CodeLlama <PRE>/<SUF>/<MID> format) when there is a suffix, plain prefix otherwise."""
    if suffix.strip() and supports_fim(tokenizer):
        return tokenizer(prefix + tokenizer.fill_token + suffix)["input_ids"]
    return tokenizer(prefix)["input_ids"]


def build_training_ids(tokenizer, prefix: str, middle: str, suffix: str) -> tuple[list[int], int]:
    """Return (input_ids, prompt_len). Loss is computed only on tokens after prompt_len."""
    if suffix.strip() and supports_fim(tokenizer):
        prompt = build_prompt_ids(tokenizer, prefix, suffix)
        target = encode_continuation(tokenizer, middle) + [eot_id(tokenizer)]
    else:
        # Plain left-to-right sample: train on the whole function.
        prompt = [tokenizer.bos_token_id]
        target = encode_continuation(tokenizer, prefix + middle + suffix) + [tokenizer.eos_token_id]
    return prompt + target, len(prompt)


def newline_token_ids(tokenizer) -> set[int]:
    return {i for t, i in tokenizer.get_vocab().items() if "<0x0A>" in t or "\n" in t}
