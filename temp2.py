import json
from pathlib import Path

INPUT_DIR  = Path("results/data")
OUTPUT_DIR = Path("results/data_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Qwen2.5 tokenizer 기준 #### = token_id 820
ANSWER_TOKEN_ID = 820

jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
print(f"처리할 파일: {len(jsonl_files)}개\n")

for filepath in jsonl_files:
    out_path = OUTPUT_DIR / filepath.name

    total_docs     = 0
    changed_docs   = 0
    changed_tokens = 0

    with open(filepath) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            item      = json.loads(line)
            token_ids = item["token_ids"]
            mask      = item["mask"][:]  # 복사

            # #### 위치 찾기
            answer_start = None
            for i, tid in enumerate(token_ids):
                if tid == ANSWER_TOKEN_ID:
                    answer_start = i
                    break

            # #### 이후 전부 True로
            if answer_start is not None:
                for j in range(answer_start, len(mask)):
                    if not mask[j]:
                        mask[j] = True
                        changed_tokens += 1
                if any(not item["mask"][j] for j in range(answer_start, len(item["mask"]))):
                    changed_docs += 1

            item["mask"] = mask
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_docs += 1

    print(f"  {filepath.name}")
    print(f"    총 문서:          {total_docs:,}")
    print(f"    수정된 문서:      {changed_docs:,}")
    print(f"    True로 바꾼 토큰: {changed_tokens:,}")
    print(f"    → {out_path}")
    print()

print("✅ 완료")