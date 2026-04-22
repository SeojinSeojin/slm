import os, sys, json, gc, torch, shutil

EXPERIMENTS = [
    # (scored_path,                                               exp_name,                        display_name,             ct_mode)
    ("gsm8k_scored_ct.jsonl",                                    "ct",                            "CT",                     True),
    ("gsm8k_rho1_base.jsonl",                                    "rho1_base",                     "RHO-1 base",             False),
    ("gsm8k_rho1_window_w1.jsonl",                               "rho1_window_w1",                "RHO-1 window_w1",        False),
    ("gsm8k_rho1_window_w3.jsonl",                               "rho1_window_w3",                "RHO-1 window_w3",        False),
    ("gsm8k_rho1_span_w1.jsonl",                                 "rho1_span_w1",                  "RHO-1 span_w1",          False),
    ("gsm8k_rho1_span_w3.jsonl",                                 "rho1_span_w3",                  "RHO-1 span_w3",          False),
    ("gsm8k_symbolic_window_w1.jsonl",                           "symbolic_window_w1",            "Ours-B window_w1",       False),
    ("gsm8k_symbolic_window_w3.jsonl",                           "symbolic_window_w3",            "Ours-B window_w3",       False),
    ("gsm8k_symbolic_span_w1.jsonl",                             "symbolic_span_w1",              "Ours-B span_w1",         False),
    ("gsm8k_symbolic_span_w3.jsonl",                             "symbolic_span_w3",              "Ours-B span_w3",         False),
    ("gsm8k_entropy_gap_window_w1.jsonl",                        "entropy_gap_window_w1",         "Ours-A window_w1",       False),
    ("gsm8k_entropy_gap_window_w3.jsonl",                        "entropy_gap_window_w3",         "Ours-A window_w3",       False),
    ("gsm8k_entropy_gap_span_w1.jsonl",                          "entropy_gap_span_w1",           "Ours-A span_w1",         False),
    ("gsm8k_entropy_gap_span_w3.jsonl",                          "entropy_gap_span_w3",           "Ours-A span_w3",         False),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(BASE_DIR, "results")
DATA_DIR  = os.path.join(SAVE_DIR, "data")

def plot_token_ratio():
    for name, _, _, _ in EXPERIMENTS:
        ratios = {}
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            continue
        total, selected = 0, 0
        with open(path) as f:
            for line in f:
                item      = json.loads(line)
                total    += len(item["mask"])
                selected += sum(item["mask"])
        if total > 0:
            ratios[name] = selected / total
            print(f"  Actual ratio [{name}]: {selected/total:.1%}")

    print("\n[Token Selection Ratios]")
    for name, ratio in ratios.items():
        print(f"  {name}: {ratio:.1%}")            


plot_token_ratio()