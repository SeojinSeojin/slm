# visualize.py
# Run: python3 visualize.py

import os
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Paths — auto-detect VCL or Colab
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_save_dir():
    vcl_path = os.path.join(BASE_DIR, "results")
    if os.path.exists(vcl_path):
        return vcl_path
    colab_path = "/content/drive/MyDrive/slm_results"
    if os.path.exists(colab_path):
        return colab_path
    return vcl_path

SAVE_DIR  = find_save_dir()
MODEL_DIR = os.path.join(SAVE_DIR, "models")
FIG_DIR   = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print(f"SAVE_DIR: {SAVE_DIR}")
print(f"FIG_DIR:  {FIG_DIR}")

EXPERIMENTS = {
    "ct":                {"label": "CT (Baseline)",    "color": "#888888", "ls": ":"},
    "rho1":              {"label": "RHO-1",             "color": "#E63946", "ls": "--"},
    "ours_b_symbolic":   {"label": "Ours-B (Symbolic)", "color": "#457B9D", "ls": "-"},
    "ours_a_entropy_gap":{"label": "Ours-A (Entropy)",  "color": "#2A9D8F", "ls": "-."},
}

# ============================================================
# 1. Loss Curve
# ============================================================

def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(9, 5))
    found   = False

    for exp_key, style in EXPERIMENTS.items():
        path = os.path.join(MODEL_DIR, f"{exp_key}_loss.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            history = json.load(f)
        if not history:
            continue

        steps  = [h["step"] for h in history]
        losses = [h["loss"]  for h in history]

        ax.plot(steps, losses,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                linewidth=2.0,
                marker="o", markersize=4)
        found = True

    if not found:
        print("  No loss history found.")
        return

    ax.set_xlabel("Optimizer Steps", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: loss_curve.png")


# ============================================================
# 2. Evaluation Bar Chart
# ============================================================

def load_our_results():
    # Priority: reparsed > full > regular
    for fname in ["eval_results_reparsed.json", "eval_results_full.json", "eval_results.json"]:
        path = os.path.join(SAVE_DIR, fname)
        if os.path.exists(path):
            print(f"  Loading: {fname}")
            with open(path) as f:
                return json.load(f)
    return []


def plot_eval_bar():
    our_results = load_our_results()
    if not our_results:
        print("  eval_results.json not found.")
        return

    all_data = []
    for r in our_results:
        all_data.append({
            "name":  r["experiment"],
            "gsm8k": r.get("gsm8k_accuracy", 0.0),
            "math":  r.get("math_accuracy",  0.0),
            "asdiv": r.get("asdiv_accuracy", 0.0),
            "svamp": r.get("svamp_accuracy", 0.0),
        })

    has_asdiv = any(d["asdiv"] > 0 for d in all_data)
    has_svamp = any(d["svamp"] > 0 for d in all_data)

    names = [d["name"] for d in all_data]
    n = len(names)
    x = np.arange(n)

    benchmarks = [("gsm8k", "#457B9D", "GSM8K"),
                  ("math",  "#E63946", "MATH")]
    if has_asdiv:
        benchmarks.append(("asdiv", "#F4A261", "ASDiv"))
    if has_svamp:
        benchmarks.append(("svamp", "#2A9D8F", "SVAMP"))

    nb = len(benchmarks)
    w  = 0.7 / nb

    fig, ax = plt.subplots(figsize=(max(10, n * 2.0), 5))

    for bi, (key, color, label) in enumerate(benchmarks):
        vals   = [d[key] for d in all_data]
        offset = (bi - (nb - 1) / 2) * w
        bars   = ax.bar(x + offset, vals, w, color=color, label=label)
        for bar, val in zip(bars, vals):
            if val > 0.3:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.legend(fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    title = "Evaluation Results: " + ", ".join(b[2] for b in benchmarks)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    all_vals = [d[k] for d in all_data for k, _, _ in benchmarks]
    y_max = max(all_vals) if any(v > 0 for v in all_vals) else 10.0
    ax.set_ylim(0, y_max * 1.25 + 1)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "eval_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: eval_bar.png")

    csv_path = os.path.join(FIG_DIR, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        header = ["Model"] + [f"{b[2]} (%)" for b in benchmarks]
        w_csv.writerow(header)
        for d in all_data:
            row = [d["name"]] + [f"{d[b[0]]:.1f}" for b in benchmarks]
            w_csv.writerow(row)
    print(f"  Saved: eval_results.csv")


# ============================================================
# 3. Token Selection Ratio
# ============================================================

def plot_token_ratio():
    default_ratios = {
        "CT (All tokens)":       1.00,
        "RHO-1":                 0.60,
        "Ours-B (Symbolic)":     0.77,
        "Ours-A (Entropy Gap)":  0.40,
    }
    colors = ["#888888", "#E63946", "#457B9D", "#2A9D8F"]

    scored_files = {
        "RHO-1":               os.path.join(SAVE_DIR, "data/gsm8k_scored_rho1.jsonl"),
        "Ours-B (Symbolic)":   os.path.join(SAVE_DIR, "data/gsm8k_scored_symbolic.jsonl"),
        "Ours-A (Entropy Gap)":os.path.join(SAVE_DIR, "data/gsm8k_scored_entropy_gap.jsonl"),
    }
    for label, path in scored_files.items():
        if not os.path.exists(path):
            continue
        total, selected = 0, 0
        with open(path) as f:
            for line in f:
                item      = json.loads(line)
                total    += len(item["mask"])
                selected += sum(item["mask"])
        if total > 0:
            default_ratios[label] = selected / total
            print(f"  Actual ratio [{label}]: {selected/total:.1%}")

    labels = list(default_ratios.keys())
    ratios = list(default_ratios.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, [r * 100 for r in ratios], color=colors, alpha=0.85)

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.1%}", va="center", fontsize=11)

    ax.set_xlabel("Selected Token Ratio (%)", fontsize=12)
    ax.set_title("Token Selection Ratio by Method", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "token_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: token_ratio.png")


# ============================================================
# 4. Summary Table
# ============================================================

def print_summary():
    our_results = load_our_results()
    if not our_results:
        print("  No results found.")
        return

    has_asdiv = any(r.get("asdiv_accuracy", 0) > 0 for r in our_results)
    has_svamp = any(r.get("svamp_accuracy", 0) > 0 for r in our_results)

    cols = ["GSM8K", "MATH"]
    if has_asdiv: cols.append("ASDiv")
    if has_svamp: cols.append("SVAMP")
    cols.append("Avg")

    width = 28 + 9 * len(cols)
    print("\n" + "=" * width)
    header = f"  {'Model':<28}" + "".join(f"{c:>9}" for c in cols)
    print(header)
    print("-" * width)

    for r in our_results:
        g   = r.get("gsm8k_accuracy", 0.0)
        m   = r.get("math_accuracy",  0.0)
        a   = r.get("asdiv_accuracy", 0.0)
        s   = r.get("svamp_accuracy", 0.0)
        vals = [g, m]
        if has_asdiv: vals.append(a)
        if has_svamp: vals.append(s)
        avg = sum(vals) / len(vals)
        vals.append(avg)
        row = f"  {r['experiment']:<28}" + "".join(f"{v:>8.1f}%" for v in vals)
        print(row)
    print("=" * width)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating figures...\n")

    print("[1/4] Loss curve")
    plot_loss_curves()

    print("\n[2/4] Eval bar chart")
    plot_eval_bar()

    print("\n[3/4] Token ratio")
    plot_token_ratio()

    print("\n[4/4] Summary table")
    print_summary()

    print(f"\nDone! Saved to: {FIG_DIR}")
