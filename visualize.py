# visualize.py
# 발표자료용 그래프 생성
# 실행: python3 visualize.py

import os
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# 경로 — VCL/Colab 둘 다 자동 감지
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_save_dir():
    # VCL: 스크립트 옆 results 폴더
    vcl_path = os.path.join(BASE_DIR, "results")
    if os.path.exists(vcl_path):
        return vcl_path
    # Colab: Drive
    colab_path = "/content/drive/MyDrive/slm_results"
    if os.path.exists(colab_path):
        return colab_path
    # 기본값
    return vcl_path

SAVE_DIR  = find_save_dir()
MODEL_DIR = os.path.join(SAVE_DIR, "models")
FIG_DIR   = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print(f"SAVE_DIR: {SAVE_DIR}")
print(f"FIG_DIR:  {FIG_DIR}")

# 실험별 스타일
EXPERIMENTS = {
    "ct":               {"label": "CT (Baseline)",    "color": "#888888", "ls": ":"},
    "rho1":             {"label": "RHO-1",             "color": "#E63946", "ls": "--"},
    "ours_b_symbolic":  {"label": "Ours-B (Symbolic)", "color": "#457B9D", "ls": "-"},
    "ours_a_entropy_gap":{"label": "Ours-A (Entropy)", "color": "#2A9D8F", "ls": "-."},
}

# 논문 수치 (RHO-1 Table 1, 1B 모델)
PAPER_NUMBERS = {
    "TinyLlama (논문)":    {"gsm8k": 2.9,  "math": 3.2},
    "TinyLlama-CT (논문)": {"gsm8k": 6.4,  "math": 2.4},
    "RHO-1 (논문)":        {"gsm8k": 29.8, "math": 14.0},
}


# ============================================================
# 1. Loss Curve 비교
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

        ax.plot(
            steps, losses,
            label=style["label"],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=2.0,
            marker="o", markersize=4,
        )
        found = True

    if not found:
        print("  ⚠️  loss history 없음. 학습 후 다시 실행.")
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
    print(f"  ✅ loss_curve.png")


# ============================================================
# 2. Evaluation Bar Chart
# ============================================================

def load_our_results():
    path = os.path.join(SAVE_DIR, "eval_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def plot_eval_bar():
    our_results = load_our_results()
    if not our_results:
        print("  ⚠️  eval_results.json 없음. 평가 후 다시 실행.")
        return

    # 논문 수치 + 우리 결과 합치기
    all_data = []
    for name, nums in PAPER_NUMBERS.items():
        all_data.append({
            "name":  name,
            "gsm8k": nums["gsm8k"],
            "math":  nums["math"],
            "ours":  False,
        })
    for r in our_results:
        all_data.append({
            "name":  r["experiment"],
            "gsm8k": r.get("gsm8k_accuracy", 0.0),
            "math":  r.get("math_accuracy",  0.0),
            "ours":  True,
        })

    names  = [d["name"]  for d in all_data]
    gsm8k  = [d["gsm8k"] for d in all_data]
    math   = [d["math"]  for d in all_data]
    avg    = [(g + m) / 2 for g, m in zip(gsm8k, math)]
    is_our = [d["ours"]  for d in all_data]

    n     = len(names)
    x     = np.arange(n)
    w     = 0.25

    fig, ax = plt.subplots(figsize=(max(9, n * 1.8), 5))

    # 논문 수치는 흐리게, 우리 결과는 진하게
    alpha_gsm = [0.5 if not o else 1.0 for o in is_our]
    alpha_mat = [0.5 if not o else 1.0 for o in is_our]

    for i, (g, m, a, ag, am) in enumerate(zip(gsm8k, math, avg, alpha_gsm, alpha_mat)):
        ax.bar(x[i] - w, g, w, color="#457B9D", alpha=ag)
        ax.bar(x[i],     m, w, color="#E63946", alpha=am)
        ax.bar(x[i] + w, a, w, color="#2A9D8F", alpha=ag)

    # 수치 레이블
    for i, (g, m, a) in enumerate(zip(gsm8k, math, avg)):
        for val, offset in [(g, -w), (m, 0), (a, w)]:
            if val > 0.3:
                ax.text(x[i] + offset, val + 0.2, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7)

    # 논문 / ours 구분선
    n_paper = len(PAPER_NUMBERS)
    if n_paper < n:
        ax.axvline(x=n_paper - 0.5, color="gray", linestyle="--", linewidth=1.2)
        ax.text(n_paper - 0.45, ax.get_ylim()[1] * 0.92,
                "← 논문 인용  /  직접 실험 →",
                fontsize=8, color="gray")

    legend = [
        mpatches.Patch(color="#457B9D", label="GSM8K"),
        mpatches.Patch(color="#E63946", label="MATH"),
        mpatches.Patch(color="#2A9D8F", label="Average"),
    ]
    ax.legend(handles=legend, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Evaluation Results: GSM8K & MATH", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    all_vals = gsm8k + math + avg
    y_max    = max(all_vals) if any(v > 0 for v in all_vals) else 10.0
    ax.set_ylim(0, y_max * 1.25 + 1)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "eval_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ eval_bar.png")

    # CSV
    csv_path = os.path.join(FIG_DIR, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["Model", "GSM8K (%)", "MATH (%)", "Avg (%)"])
        for d, a in zip(all_data, avg):
            w_csv.writerow([d["name"], f"{d['gsm8k']:.1f}", f"{d['math']:.1f}", f"{a:.1f}"])
    print(f"  ✅ eval_results.csv")


# ============================================================
# 3. 토큰 선별 비율 비교 (pie/bar)
# ============================================================

def plot_token_ratio():
    """각 방법이 실제로 몇 % 토큰을 선택했는지 scored 파일에서 읽어서 표시."""

    # 기본값 (scored 파일 없을 때 fallback)
    default_ratios = {
        "CT (전체)":             1.00,
        "RHO-1":                 0.60,
        "Ours-B (Symbolic)":     0.10,
        "Ours-A (Entropy Gap)":  0.10,
    }
    colors = ["#888888", "#E63946", "#457B9D", "#2A9D8F"]

    # 실제 scored 파일에서 비율 계산
    scored_files = {
        "RHO-1":                os.path.join(SAVE_DIR, "data/gsm8k_scored_rho1.jsonl"),
        "Ours-B (Symbolic)":    os.path.join(SAVE_DIR, "data/gsm8k_scored_symbolic.jsonl"),
        "Ours-A (Entropy Gap)": os.path.join(SAVE_DIR, "data/gsm8k_scored_entropy_gap.jsonl"),
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
            print(f"  실제 비율 [{label}]: {selected/total:.1%}")

    labels = list(default_ratios.keys())
    ratios = list(default_ratios.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, [r * 100 for r in ratios], color=colors, alpha=0.85)

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.1%}", va="center", fontsize=11)

    ax.set_xlabel("Token Selection Ratio (%)", fontsize=12)
    ax.set_title("Token Selection Ratio by Method", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 120)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "token_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ token_ratio.png")


# ============================================================
# 4. 텍스트 테이블 출력
# ============================================================

def print_summary():
    our_results = load_our_results()
    all_data    = []

    for name, nums in PAPER_NUMBERS.items():
        all_data.append((name, nums["gsm8k"], nums["math"], False))
    for r in our_results:
        all_data.append((
            r["experiment"],
            r.get("gsm8k_accuracy", 0.0),
            r.get("math_accuracy",  0.0),
            True,
        ))

    print("\n" + "=" * 65)
    print(f"  {'Model':<28} {'GSM8K':>8} {'MATH':>8} {'Avg':>8}  {'':>6}")
    print("-" * 65)
    prev_ours = False
    for name, g, m, ours in all_data:
        if ours and not prev_ours:
            print("-" * 65)
        a    = (g + m) / 2
        flag = "← 직접" if ours else "← 논문"
        print(f"  {name:<28} {g:>7.1f}% {m:>7.1f}% {a:>7.1f}%  {flag}")
        prev_ours = ours
    print("=" * 65)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    print("📊 발표자료 생성 중...\n")

    print("[1/4] Loss curve")
    plot_loss_curves()

    print("\n[2/4] Eval bar chart")
    plot_eval_bar()

    print("\n[3/4] Token ratio")
    plot_token_ratio()

    print("\n[4/4] Summary table")
    print_summary()

    print(f"\n🎉 완료! 저장 위치: {FIG_DIR}")
    print("   loss_curve.png  → 슬라이드 삽입")
    print("   eval_bar.png    → 슬라이드 삽입")
    print("   token_ratio.png → 슬라이드 삽입")
    print("   eval_results.csv")
