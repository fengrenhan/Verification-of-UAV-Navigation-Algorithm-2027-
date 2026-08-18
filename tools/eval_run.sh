#!/usr/bin/env bash
# 用法: ./eval_run.sh <tag> <checkpoint.pth> [eval_demo.py 额外参数...]
# 例:   ./eval_run.sh ckpt0004_40obs_4r checkpoint0004.pth --single --num_rollouts 4 --timesteps 150
set -euo pipefail
TAG="${1:?用法: eval_run.sh <tag> <checkpoint.pth> [额外参数...]}"
CKPT="${2:?缺少 checkpoint}"
shift 2

[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "tag 只能含字母数字下划线连字符"; exit 1; }

ROOT="$HOME/DiffPhysDrone/evals"
RUN_DIR="$ROOT/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"      # 在本目录内跑，产物自动落进来
{
  echo "# 时间: $(date '+%F %T')"
  echo "# 目录: $RUN_DIR"
  echo "# checkpoint: $CKPT ($(ls -lh "$HOME/DiffPhysDrone/$CKPT" | awk '{print $5}'))"
  echo "# git: $(cd "$HOME/DiffPhysDrone" && git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "# 命令: eval_demo.py --resume $CKPT --out $TAG $*"
  echo "----------------------------------------"
} | tee run_info.txt
python3 "$HOME/DiffPhysDrone/eval_demo.py" \
  --resume "$HOME/DiffPhysDrone/$CKPT" --out "$TAG" "$@" 2>&1 | tee -a run_info.txt
echo
echo "===== 产物: $RUN_DIR ====="
ls -lh
echo "看结果: cd $RUN_DIR && explorer.exe ."
