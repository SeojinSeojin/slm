#!/bin/bash
# run_all.sh (Ubuntu 22 + RTX 2080 Ti 11GB)
#
# 사전 준비:
#   dropbox.com/developers → 앱 생성
#   Permissions: files.content.write 체크 → Submit
#   Settings: Generate access token 복사
#   아래 DROPBOX_TOKEN에 붙여넣기

# ============================================================
# ★ 여기만 수정 ★
# ============================================================
DROPBOX_TOKEN="your_dropbox_token_here"
DROPBOX_DEST="/slm_results"
# ============================================================

# CUDA 환경변수 — 반드시 가장 먼저
export CUDA_VISIBLE_DEVICES=0

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/experiment.log"
RESULTS_DIR="$SCRIPT_DIR/results"

echo "======================================================"
echo "  SLM 실험 자동화 (Ubuntu 22 + RTX 2080 Ti)"
echo "  시작: $(date)"
echo "======================================================"

# ============================================================
# Step 1: apt lock 해제 + 시스템 패키지
# ============================================================
echo ""
echo "[1/4] 시스템 패키지 설치..."

# apt lock 강제 해제
sudo killall apt-get 2>/dev/null || true
sudo killall dpkg   2>/dev/null || true
sudo rm -f /var/lib/apt/lists/lock
sudo rm -f /var/lib/dpkg/lock
sudo rm -f /var/lib/dpkg/lock-frontend
sudo rm -f /var/cache/apt/archives/lock
sudo dpkg --configure -a 2>/dev/null || true

sudo apt-get update -qq
sudo apt-get install -y -qq tmux curl wget unzip python3-pip
echo "  ✅ 시스템 패키지 완료"

# ============================================================
# Step 2: Python 패키지 (Ubuntu 22: pip 그냥 사용)
# ============================================================
echo ""
echo "[2/4] Python 패키지 설치..."

# Ubuntu 22 + 2080 Ti → CUDA 11.x 계열
pip3 install -q \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 \
    --break-system-packages

pip3 install -q \
    "transformers>=4.40.0" \
    datasets \
    accelerate \
    tqdm \
    numpy \
    matplotlib \
    "Pillow>=9.2.0" \
    huggingface_hub \
    --break-system-packages

echo "  ✅ Python 패키지 완료"

# GPU 확인
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print('  ⚠️  GPU 인식 안 됨')
"

# ============================================================
# Step 3: send_results.py 생성
# ============================================================
echo ""
echo "[3/4] Dropbox 전송 스크립트 생성..."

cat > "$SCRIPT_DIR/send_results.py" << PYEOF
#!/usr/bin/env python3
import os, json, dropbox
from dropbox.exceptions import ApiError

DROPBOX_TOKEN = "${DROPBOX_TOKEN}"
DROPBOX_DEST  = "${DROPBOX_DEST}"
RESULTS_DIR   = "${RESULTS_DIR}"
SCRIPT_DIR    = "${SCRIPT_DIR}"

def make_summary():
    path = os.path.join(RESULTS_DIR, "eval_results.json")
    if not os.path.exists(path):
        return "eval_results.json 없음 — 실험 미완료 또는 진행 중"
    with open(path) as f:
        results = json.load(f)
    lines = ["=" * 55, "  Symbolic-Aware Token Selection 결과", "=" * 55]
    lines.append(f"  {'Model':<20} {'GSM8K':>8} {'MATH':>8} {'Avg':>8}")
    lines.append("-" * 55)
    for r in results:
        g = r.get("gsm8k_accuracy", 0)
        m = r.get("math_accuracy",  0)
        lines.append(f"  {r['experiment']:<20} {g:>7.1f}% {m:>7.1f}% {(g+m)/2:>7.1f}%")
    lines.append("=" * 55)
    return "\n".join(lines)

def upload_to_dropbox():
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    try:
        acc = dbx.users_get_current_account()
        print(f"  Dropbox 연결 OK ({acc.email})")
    except Exception as e:
        print(f"  Dropbox 연결 실패: {e}")
        return

    def upload(local_path, name):
        remote = f"{DROPBOX_DEST}/{name}"
        with open(local_path, "rb") as f:
            try:
                dbx.files_upload(f.read(), remote,
                    mode=dropbox.files.WriteMode.overwrite)
                print(f"  업로드: {name}")
            except ApiError as e:
                print(f"  실패 {name}: {e}")

    for fname, fpath in [
        ("experiment.log", os.path.join(SCRIPT_DIR, "experiment.log")),
        ("eval_results.json", os.path.join(RESULTS_DIR, "eval_results.json")),
    ]:
        if os.path.exists(fpath):
            upload(fpath, fname)

    models_dir = os.path.join(RESULTS_DIR, "models")
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith("_loss.json"):
                upload(os.path.join(models_dir, fname), fname)

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    if os.path.exists(fig_dir):
        for fname in sorted(os.listdir(fig_dir)):
            if fname.endswith(".png") or fname.endswith(".csv"):
                upload(os.path.join(fig_dir, fname), fname)

    summary = make_summary()
    spath = os.path.join(RESULTS_DIR, "summary.txt")
    with open(spath, "w") as f:
        f.write(summary)
    upload(spath, "summary.txt")

if __name__ == "__main__":
    print("Dropbox 업로드 시작...")
    print(make_summary())
    try:
        upload_to_dropbox()
        print(f"\n완료! Dropbox '{DROPBOX_DEST}' 확인하세요.")
    except Exception as e:
        print(f"\n실패: {e}")
PYEOF

echo "  ✅ send_results.py 생성 완료"

# ============================================================
# Step 4: tmux 세션으로 실험 실행
# ============================================================
echo ""
echo "[4/4] tmux 세션 시작..."

SESSION="slm_exp"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "
export CUDA_VISIBLE_DEVICES=0
cd '$SCRIPT_DIR'
echo '실험 시작: \$(date)' | tee '$LOG_FILE'

python3 run_experiment.py 2>&1 | tee -a '$LOG_FILE'
EXPERIMENT_EXIT=\${PIPESTATUS[0]}

echo '' | tee -a '$LOG_FILE'
if [ \$EXPERIMENT_EXIT -eq 0 ]; then
    echo '✅ 실험 완료: \$(date)' | tee -a '$LOG_FILE'
    python3 visualize.py 2>&1 | tee -a '$LOG_FILE' || echo '⚠️  시각화 실패 (결과는 저장됨)' | tee -a '$LOG_FILE'
else
    echo '❌ 실험 실패 (exit='\$EXPERIMENT_EXIT'): \$(date)' | tee -a '$LOG_FILE'
    echo '   부분 결과를 Dropbox에 업로드합니다.' | tee -a '$LOG_FILE'
fi

echo 'Dropbox 업로드...' | tee -a '$LOG_FILE'
python3 send_results.py 2>&1 | tee -a '$LOG_FILE'
" Enter

echo ""
echo "======================================================"
echo "  tmux 세션 '$SESSION' 시작!"
echo "======================================================"
echo "  확인:   tmux attach -t $SESSION"
echo "  detach: Ctrl+B, D"
echo "  로그:   tail -f $LOG_FILE"
echo "  예상:   ~9시간 (2080 Ti 11GB)"
echo "  완료 시 Dropbox '$DROPBOX_DEST' 자동 저장"
echo "======================================================"
