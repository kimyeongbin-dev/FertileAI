#!/usr/bin/env bash
# 클론 직후 1회 실행: Git 훅 연결 + 실행 환경 확인
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
chmod +x .githooks/commit-msg .githooks/pre-push scripts/setup.sh
echo "✔ Git 훅 연결 (core.hooksPath=.githooks)"

mkdir -p data
if [ -f data/train.csv ] && [ -f data/test.csv ]; then
  echo "✔ data/train.csv, data/test.csv 확인"
else
  echo "⚠ data/ 에 train.csv, test.csv 가 없습니다. README 의 '데이터 준비'를 확인하세요."
fi

command -v python > /dev/null || echo "⚠ Python 이 없습니다. README 의 사전 요구사항을 확인하세요."
