import os, sys, json, gc, torch, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(BASE_DIR, "results")
DATA_DIR  = os.path.join(SAVE_DIR, "data_fixed")

def plot_token_ratio():
    jsonl_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")
    )

    if not jsonl_files:
        print("No .jsonl files found in", DATA_DIR)
        return

    ratios = {}
    for name in jsonl_files:
        path = os.path.join(DATA_DIR, name)
        total, selected = 0, 0
        with open(path) as f:
            for line in f:
                item      = json.loads(line)
                if "mask" not in item:
                    continue
                total    += len(item["mask"])
                selected += sum(item["mask"])
        if total > 0:
            ratios[name] = selected / total
            print(f"  Actual ratio [{name}]: {selected/total:.1%}")

    print("\n[Token Selection Ratios]")
    for name, ratio in ratios.items():
        print(f"  {name}: {ratio:.1%}")


plot_token_ratio()