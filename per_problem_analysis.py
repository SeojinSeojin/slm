import json
import glob
import os

result_dir = "results/per_problem"

baseline_files = {"CT.json", "Qwen2.5-base.json"}
all_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
other_files = [f for f in all_files if f not in baseline_files]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# 모든 파일 로드
data = {}
for fname in all_files:
    data[fname] = load_json(os.path.join(result_dir, fname))

# dataset별로 문제를 key로 인덱싱
# key: (dataset, question)
def index_by_question(d):
    index = {}
    for dataset, problems in d.items():
        if not isinstance(problems, list):
            continue
        for item in problems:
            question = item.get("question") or item.get("problem")
            if not question:
                continue
            key = (dataset, question)
            index[key] = item
    return index

indexed = {fname: index_by_question(d) for fname, d in data.items()}

# CT, Qwen2.5-base 둘 다 false이고, 나머지 중 하나라도 true인 문제 추출
ct_idx = indexed["CT.json"]
qwen_idx = indexed["Qwen2.5-base.json"]

candidate_keys = set(ct_idx.keys()) & set(qwen_idx.keys())

results = []

for key in candidate_keys:
    dataset, question = key

    ct_correct = ct_idx[key].get("correct", False)
    qwen_correct = qwen_idx[key].get("correct", False)

    if ct_correct or qwen_correct:
        continue  # 둘 다 false여야 함

    # 나머지 중 하나라도 true인지 확인
    any_true = any(
        indexed[f].get(key, {}).get("correct", False)
        for f in other_files
    )

    if not any_true:
        continue

    # 각 experiment별 pred / correct 수집
    row = {
        "dataset": dataset,
        "question": question,
        "gold": ct_idx[key].get("gold"),
    }
    for fname in all_files:
        item = indexed[fname].get(key, {})
        exp = data[fname].get("experiment", fname.replace(".json", ""))
        row[exp] = {
            "pred": item.get("pred"),
            "generated": item.get("generated"),
            "correct": item.get("correct"),
        }
    results.append(row)

print(f"총 {len(results)}개 문제 발견\n")

for r in results:
    print(f"[{r['dataset']}] Gold: {r['gold']}")
    print(f"Q: {r['question']}")
    for fname in all_files:
        exp = data[fname].get("experiment", fname.replace(".json", ""))
        info = r.get(exp, {})
        mark = "✓" if info.get("correct") else "✗"
        print(f"  {mark} {exp}: {info.get('pred')}\n{info.get("generated")}")
    print()