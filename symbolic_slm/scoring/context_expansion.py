# scoring/context_expansion.py
# Shared context expansion utilities for all scoring approaches.
#
# Given a boolean mask of "anchor" tokens, expands each selected token
# outward using one of two strategies:
#
#   "window" — fixed symmetric window of ±k tokens
#   "span"   — linguistically-grounded span bounded by whitespace / punctuation
#
# Usage:
#   from scoring.context_expansion import expand_mask
#
#   raw_mask = score_tokens_xxx(...)          # any approach's raw output
#   final    = expand_mask(raw_mask, token_strings, mode="span")

from __future__ import annotations

import numpy as np
import torch

__all__ = ["expand_mask"]

# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

_SPAN_BOUNDARIES = set(".,?!;:")


def _is_span_boundary(token_str: str) -> bool:
    """
    Return True if this token should terminate span expansion.

    A token is a boundary when it is:
      - whitespace-only, OR
      - composed entirely of sentence-level punctuation
    """
    if not token_str.strip():
        return True
    stripped = token_str.strip()
    return all(c in _SPAN_BOUNDARIES for c in stripped)


def _find_span(i: int, token_strings: list[str]) -> tuple[int, int]:
    """
    Find the inclusive [lo, hi] span that anchor token `i` belongs to.

    Expands leftward and rightward until a boundary token is hit,
    following the SpanBERT convention of selecting complete linguistic
    spans (e.g. "three hundred dollars", "loses 3 kg per month").
    """
    seq_len = len(token_strings)

    lo = i
    while lo > 0 and not _is_span_boundary(token_strings[lo - 1]):
        lo -= 1

    hi = i
    while hi < seq_len - 1 and not _is_span_boundary(token_strings[hi + 1]):
        hi += 1

    return lo, hi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_mask(
    anchor_mask:   torch.Tensor | np.ndarray,
    token_strings: list[str],
    mode:          str = "span",
    window:        int = 2,
) -> torch.Tensor:
    """
    Expand a boolean anchor mask according to the chosen context strategy.

    Parameters
    ----------
    anchor_mask
        1-D boolean tensor/array of shape (seq_len,).
        True where a scoring approach marked a token as important.
    token_strings
        Decoded string for each token in the sequence.
        Must have the same length as `anchor_mask`.
    mode
        "window" — expand each anchor by ±`window` tokens.
        "span"   — expand each anchor to its full linguistic span
                   (bounded by whitespace / sentence punctuation).
    window
        Half-width used only when mode="window". Ignored for "span".

    Returns
    -------
    torch.Tensor of dtype bool, shape (seq_len,).
    """
    if mode not in ("window", "span"):
        raise ValueError(f"mode must be 'window' or 'span', got {mode!r}")

    # Normalise input to numpy for uniform handling
    if isinstance(anchor_mask, torch.Tensor):
        anchors = anchor_mask.numpy().astype(bool)
    else:
        anchors = np.asarray(anchor_mask, dtype=bool)

    seq_len = len(anchors)
    if len(token_strings) != seq_len:
        raise ValueError(
            f"anchor_mask length ({seq_len}) != "
            f"token_strings length ({len(token_strings)})"
        )

    expanded = np.zeros(seq_len, dtype=bool)

    if mode == "window":
        for i in range(seq_len):
            if anchors[i]:
                lo = max(0, i - window)
                hi = min(seq_len, i + window + 1)
                expanded[lo:hi] = True

    else:  # span
        for i in range(seq_len):
            if anchors[i]:
                lo, hi = _find_span(i, token_strings)
                expanded[lo : hi + 1] = True

    return torch.tensor(expanded, dtype=torch.bool)