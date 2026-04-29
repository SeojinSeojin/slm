import json
import csv
from pathlib import Path

INPUT_DIR = Path("results/models")
OUTPUT_CSV = Path("results/loss_summary.csv")

# step -> {model_name -> loss}
data_by_step = {}
model_names = []

for json_file in sorted(INPUT_DIR.rglob("*.json")):
    model_name = json_file.stem
    model_names.append(model_name)

    with open(json_file, "r") as f:
        data = json.load(f)

    for entry in data:
        step = entry["step"]
        if step not in data_by_step:
            data_by_step[step] = {}
        data_by_step[step][model_name] = entry["loss"]

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

fieldnames = ["step"] + model_names

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for step in sorted(data_by_step.keys()):
        row = {"step": step, **data_by_step[step]}
        writer.writerow(row)

print(f"완료: {len(data_by_step)}개 step, {len(model_names)}개 모델 → {OUTPUT_CSV}")