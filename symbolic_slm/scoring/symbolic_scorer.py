# scoring/symbolic_scorer.py
# Approach B: Deterministic Symbolic Scoring
# Select mathematically relevant tokens using rule-based criteria.

import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from scoring.context_expansion import expand_mask

# ============================================================
# Token category definitions
# ============================================================

QUANTITY_VERBS = {
    "loses", "gains", "increases", "decreases", "doubles", "halves",
    "multiplies", "divides", "adds", "subtracts", "equals", "totals",
    "sold", "buys", "bought", "spent", "earned", "paid", "costs", "cost",
    "gives", "gave", "takes", "took",
    "makes", "made", "produces", "collects", "saves", "borrows", "lends",
    "remaining", "shared", "split", "combined", "altogether",
    "weighs", "measures", "counts",
}

COMPARISON_WORDS = {
    "more", "fewer", "less", "greater", "larger", "smaller",
    "maximum", "minimum", "most", "least", "times", "twice", "triple",
    "total", "each", "per", "every",
}

MATH_TERMS = {
    "sum", "difference", "product", "quotient", "remainder",
    "ratio", "proportion", "percent", "rate", "average", "mean",
    "half", "quarter", "double",
}

LOGICAL_WORDS = {
    "therefore", "thus", "hence", "let", "where",
}

ENGLISH_NUMBERS = {
    "hundred", "thousand", "million", "billion",
    "dozen", "half", "quarter",
}

UNIT_WORDS = {
    "dollars", "dollar", "cents", "cent", "percent",
    "kilometers", "km", "meters", "meter", "miles", "mile",
    "kilograms", "kg", "pounds", "pound", "grams", "gram",
    "liters", "liter", "gallons", "gallon",
    "hours", "hour", "minutes", "minute",
}

LOGICAL_KEYWORDS = (
    QUANTITY_VERBS |
    COMPARISON_WORDS |
    MATH_TERMS |
    LOGICAL_WORDS |
    ENGLISH_NUMBERS |
    UNIT_WORDS
)

MATH_PATTERNS = [
    r'\d+\.?\d*',
    r'[+\-*/=<>≤≥≠±]',
    r'\\frac\{',
    r'\\sum|\\prod|\\int',
    r'\\sqrt|\\log|\\ln|\\exp',
    r'\\[a-zA-Z]+',
    r'\$[^$]+\$',
    r'\$\$[^$]+\$\$',
    r'[a-zA-Z]_\{?\d+\}?',
    r'\([^)]*=\s*[^)]*\)',
    r'[a-zA-Z]\s*=\s*[a-zA-Z0-9]',
    r'[a-zA-Z]\s*:=\s*',
    r'let\s+[a-zA-Z]\s*=',
    r'where\s+[a-zA-Z]\s*=',
    r'→|⟹|⟺|\s*=>\s*',
    r'<<[^>]+>>',
    r'####',
]

COMPILED_PATTERNS = [re.compile(p) for p in MATH_PATTERNS]


def is_symbolic_token(token_str: str) -> bool:
    """Return True if this token is mathematically relevant."""
    t = token_str.strip().lower()

    if not t or t in {"", " ", "\n", "\t"}:
        return False

    if t in LOGICAL_KEYWORDS:
        return True

    for pattern in COMPILED_PATTERNS:
        if pattern.search(token_str):
            return True

    if any(c.isdigit() for c in token_str):
        return True

    math_symbols = set("+=−×÷≤≥≠∈∉⊂⊃∪∩∀∃∧∨¬→↔∞∑∏∫√π%$")
    if any(c in math_symbols for c in token_str):
        return True

    return False


# ============================================================
# Core scoring function
# ============================================================

def score_tokens_symbolic(
    token_ids:    list | torch.Tensor,
    tokenizer:    AutoTokenizer,
    context_mode: str = "span",   # "span" | "window"
    window:       int = 2,        # used only when context_mode="window"
    # Legacy aliases kept for backwards compatibility
    select_ratio:   float = 0.40,
    context_window: int   = 2,
) -> torch.Tensor:
    """
    Score tokens symbolically and expand anchors via context_expansion.

    Parameters
    ----------
    token_ids
        Token ID sequence (list or 1-D tensor).
    tokenizer
        HuggingFace tokenizer matching the token IDs.
    context_mode
        "span"   — expand each anchor to its full linguistic span.
        "window" — expand each anchor by ±`window` tokens.
    window
        Half-width for window mode (default 2).
    select_ratio, context_window
        Ignored; kept so old call-sites don't break.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    seq_len = len(token_ids)

    token_strings = [
        tokenizer.decode([tid], skip_special_tokens=False)
        for tid in token_ids
    ]

    # Step 1: mark symbolic anchor tokens
    anchors = np.zeros(seq_len, dtype=bool)
    for i, token_str in enumerate(token_strings):
        anchors[i] = is_symbolic_token(token_str)

    anchor_mask = torch.tensor(anchors, dtype=torch.bool)

    # Step 2: expand via shared module
    return expand_mask(
        anchor_mask,
        token_strings,
        mode=context_mode,
        window=window,
    )


# ============================================================
# Dataset preprocessing
# ============================================================

def preprocess_dataset_symbolic(
    data_path:    str,
    output_path:  str,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
    max_length:   int   = 512,
    context_mode: str   = "span",
    window:       int   = 2,
    # Legacy aliases
    select_ratio:   float = 0.40,
    context_window: int   = 2,
):
    """Apply symbolic scoring to an entire JSONL dataset and save masks."""
    print(f"Loading data: {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    with open(data_path, "r") as f:
        lines = f.readlines()

    print(f"Processing {len(lines):,} documents...")
    total_tokens    = 0
    selected_tokens = 0

    for line in tqdm(lines):
        item = json.loads(line.strip())
        text = item["text"]

        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = enc["input_ids"][0]

        mask = score_tokens_symbolic(
            token_ids, tokenizer,
            context_mode=context_mode,
            window=window,
        )

        total_tokens    += len(token_ids)
        selected_tokens += mask.sum().item()

        results.append({
            "text":      text,
            "token_ids": token_ids.tolist(),
            "mask":      mask.tolist(),
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ratio = selected_tokens / total_tokens * 100
    print(f"\nDone! Saved: {output_path}")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Selected tokens: {selected_tokens:,} ({ratio:.1f}%)")
    return output_path


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    test_texts = [
        "A boxer weighs 97 kg at 4 months from a fight. He loses 3 kg per month. How much will he weigh?",
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast and sells the rest for $2 each.",
        "John takes a pill every 6 hours. How many pills does he take a week?",
    ]

    for mode in ("span", "window"):
        print(f"\n{'='*60}")
        print(f"context_mode = {mode!r}")
        print('='*60)
        for text in test_texts:
            enc       = tokenizer(text, return_tensors="pt")
            token_ids = enc["input_ids"][0]
            mask      = score_tokens_symbolic(token_ids, tokenizer, context_mode=mode)
            tokens    = [tokenizer.decode([tid]) for tid in token_ids.tolist()]
            rendered  = "".join(f"[{t}]" if m else t for t, m in zip(tokens, mask.tolist()))
            ratio     = mask.float().mean().item()
            print(f"\n{ratio:.1%}: {rendered}")