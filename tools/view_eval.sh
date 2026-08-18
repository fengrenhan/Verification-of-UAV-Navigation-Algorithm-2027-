#!/usr/bin/env bash
# 用法: ./view_eval.sh [关键词]   不带参数=最近一次；带关键词=匹配最近的那个 tag
KEY="${1:-}"
if [ -n "$KEY" ]; then
  DIR=$(ls -td "$HOME"/DiffPhysDrone/evals/*"$KEY"*/ 2>/dev/null | head -1)
else
  DIR=$(ls -td "$HOME"/DiffPhysDrone/evals/*/ 2>/dev/null | head -1)
fi
[ -z "$DIR" ] && { echo "没找到评估目录"; exit 1; }
echo "打开: $DIR"
cd "$DIR" && explorer.exe .
