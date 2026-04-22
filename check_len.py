import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

lengths = []
with open("results/data/gsm8k_train.jsonl") as f:
    for line in f:
        item = json.loads(line)
        ids = tokenizer.encode(item["text"])
        lengths.append(len(ids))

import numpy as np
print(f"mean:   {np.mean(lengths):.0f}")
print(f"median: {np.median(lengths):.0f}")
print(f"p90:    {np.percentile(lengths, 90):.0f}")
print(f"p95:    {np.percentile(lengths, 95):.0f}")
print(f"max:    {max(lengths)}")
