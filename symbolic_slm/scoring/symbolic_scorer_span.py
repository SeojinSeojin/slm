# scoring/symbolic_scorer.py
# Approach B: Deterministic Symbolic Scoring
# Select mathematically relevant tokens using rule-based criteria.

import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================================================
# Token category definitions
# ============================================================

# Quantity/change verbs — core math action words in word problems
QUANTITY_VERBS = {
    "loses", "gains", "increases", "decreases", "doubles", "halves",
    "multiplies", "divides", "adds", "subtracts", "equals", "totals",
    "sold", "buys", "bought", "spent", "earned", "paid", "costs", "cost",
    "gives", "gave", "takes", "took",
    "makes", "made", "produces", "collects", "saves", "borrows", "lends",
    "remaining", "shared", "split", "combined", "altogether",
    "weighs", "measures", "counts",
}

# Comparison and relation words
COMPARISON_WORDS = {
    "more", "fewer", "less", "greater", "larger", "smaller",
    "maximum", "minimum", "most", "least", "times", "twice", "triple",
    "total", "each", "per", "every",
}

# Math terminology
MATH_TERMS = {
    "sum", "difference", "product", "quotient", "remainder",
    "ratio", "proportion", "percent", "rate", "average", "mean",
    "half", "quarter", "double",
}

# Logical/structural words (minimal)
LOGICAL_WORDS = {
    "therefore", "thus", "hence", "let", "where",
}

# English number words — only large/ambiguous ones worth marking
# Small ones (one~ten) skipped: Qwen tokenizer catches digits anyway
ENGLISH_NUMBERS = {
    "hundred", "thousand", "million", "billion",
    "dozen", "half", "quarter",
}

# Time/unit words — only specific math-problem units, not generic ones
UNIT_WORDS = {
    "dollars", "dollar", "cents", "cent", "percent",
    "kilometers", "km", "meters", "meter", "miles", "mile",
    "kilograms", "kg", "pounds", "pound", "grams", "gram",
    "liters", "liter", "gallons", "gallon",
    "hours", "hour", "minutes", "minute",
}

# Combined keyword set
LOGICAL_KEYWORDS = (
    QUANTITY_VERBS |
    COMPARISON_WORDS |
    MATH_TERMS |
    LOGICAL_WORDS |
    ENGLISH_NUMBERS |
    UNIT_WORDS
)

# Math expression patterns (regex)
MATH_PATTERNS = [
    r'\d+\.?\d*',                          # numbers (97, 3.14)
    r'[+\-*/=<>≤≥≠±]',                   # operators
    r'\\frac\{',                           # LaTeX fraction
    r'\\sum|\\prod|\\int',                 # LaTeX sum/integral
    r'\\sqrt|\\log|\\ln|\\exp',           # LaTeX functions
    r'\\[a-zA-Z]+',                        # LaTeX commands
    r'\$[^$]+\$',                          # inline math $...$
    r'\$\$[^$]+\$\$',                      # display math $$...$$
    r'[a-zA-Z]_\{?\d+\}?',               # variable subscript (x_1)
    r'\([^)]*=\s*[^)]*\)',                # equation in parentheses
    r'[a-zA-Z]\s*=\s*[a-zA-Z0-9]',       # variable assignment (x = 3)
    r'[a-zA-Z]\s*:=\s*',                  # definition (x := ...)
    r'let\s+[a-zA-Z]\s*=',               # "let x = ..."
    r'where\s+[a-zA-Z]\s*=',             # "where x = ..."
    r'→|⟹|⟺|\s*=>\s*',               # transition arrows
    r'<<[^>]+>>',                          # GSM8K computation markers <<...>>
    r'####',                               # GSM8K answer separator
]

COMPILED_PATTERNS = [re.compile(p) for p in MATH_PATTERNS]

# Punctuation characters that mark span boundaries
SPAN_BOUNDARIES = set(".,?!;:")


def is_symbolic_token(token_str: str) -> bool:
    """
    Determine if a token is mathematically relevant.
    Returns True if the token should be included in training.
    """
    t = token_str.strip().lower()

    # Skip empty/whitespace tokens
    if not t or t in {"", " ", "\n", "\t"}:
        return False

    # 1) Math keywords (quantity verbs, numbers, units, etc.)
    if t in LOGICAL_KEYWORDS:
        return True

    # 2) Regex pattern matching
    for pattern in COMPILED_PATTERNS:
        if pattern.search(token_str):
            return True

    # 3) Any digit present
    if any(c.isdigit() for c in token_str):
        return True

    # 4) Special math symbols
    math_symbols = set("+=−×÷≤≥≠∈∉⊂⊃∪∩∀∃∧∨¬→↔∞∑∏∫√π%$")
    if any(c in math_symbols for c in token_str):
        return True

    return False


def _is_span_boundary(token_str: str) -> bool:
    """
    Return True if this token should terminate span expansion.
    A token is a boundary if it is whitespace-only or consists
    solely of punctuation characters defined in SPAN_BOUNDARIES.
    """
    # Whitespace-only tokens stop the span
    if not token_str.strip():
        return True

    # Tokens whose non-whitespace characters are all punctuation stop the span
    stripped = token_str.strip()
    if all(c in SPAN_BOUNDARIES for c in stripped):
        return True

    return False


def _find_span(i: int, token_strings: list[str]) -> tuple[int, int]:
    """
    Given a token index `i`, find the inclusive [lo, hi] bounds of the
    contiguous word-level span that `i` belongs to.

    Expansion halts (exclusively) when a whitespace-only token or a
    punctuation-boundary token is encountered, so that the span covers
    a coherent linguistic unit such as "three hundred dollars" or
    "loses 3 kg per month".

    Parameters
    ----------
    i             : index of the anchor (symbolic) token
    token_strings : decoded string for every token in the sequence

    Returns
    -------
    (lo, hi) : inclusive indices of the detected span
    """
    seq_len = len(token_strings)

    # Expand leftward
    lo = i
    while lo > 0 and not _is_span_boundary(token_strings[lo - 1]):
        lo -= 1

    # Expand rightward
    hi = i
    while hi < seq_len - 1 and not _is_span_boundary(token_strings[hi + 1]):
        hi += 1

    return lo, hi


def score_tokens_symbolic(
    token_ids:    list,
    tokenizer:    AutoTokenizer,
    select_ratio: float = 0.40,   # kept for API compatibility; not used in span logic
) -> torch.Tensor:
    """
    Select symbolic tokens and expand each to its full word-level span.

    For every token that passes `is_symbolic_token`, the entire
    contiguous span of non-boundary tokens surrounding it is selected,
    following the SpanBERT convention of selecting complete linguistic
    spans rather than an arbitrary fixed window.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    seq_len = len(token_ids)

    # Decode every token once for reuse
    token_strings = [
        tokenizer.decode([tid], skip_special_tokens=False)
        for tid in token_ids
    ]

    # Step 1: mark symbolic tokens
    symbolic = np.zeros(seq_len, dtype=bool)
    for i, token_str in enumerate(token_strings):
        symbolic[i] = is_symbolic_token(token_str)

    # Step 2: for each symbolic token, select its full span
    mask = np.zeros(seq_len, dtype=bool)
    for i in range(seq_len):
        if symbolic[i]:
            lo, hi = _find_span(i, token_strings)
            mask[lo : hi + 1] = True

    return torch.tensor(mask, dtype=torch.bool)


# ============================================================
# Dataset preprocessing
# ============================================================

def preprocess_dataset_symbolic(
    data_path:    str,
    output_path:  str,
    tokenizer_name: str   = "Qwen/Qwen2.5-1.5B",
    select_ratio: float   = 0.40,   # kept for API compatibility
    max_length:   int     = 512,
):
    """
    Apply symbolic scoring to entire JSONL dataset and save masks.
    """
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
            select_ratio=select_ratio,
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

    for text in test_texts:
        enc       = tokenizer(text, return_tensors="pt")
        token_ids = enc["input_ids"][0]
        mask      = score_tokens_symbolic(token_ids, tokenizer)
        tokens    = [tokenizer.decode([tid]) for tid in token_ids.tolist()]
        rendered  = "".join(f"[{t}]" if m else t for t, m in zip(tokens, mask.tolist()))
        ratio     = mask.float().mean().item()
        print(f"\n{ratio:.1%}: {rendered}")