# 클론 직후 1회 실행: Git 훅 연결 + 실행 환경 확인
$ErrorActionPreference = 'Stop'
Set-Location (git rev-parse --show-toplevel)

git config core.hooksPath .githooks
Write-Host "✔ Git 훅 연결 (core.hooksPath=.githooks)"

New-Item -ItemType Directory -Force data | Out-Null
if ((Test-Path data/train.csv) -and (Test-Path data/test.csv)) {
    Write-Host "✔ data/train.csv, data/test.csv 확인"
} else {
    Write-Host "⚠ data/ 에 train.csv, test.csv 가 없습니다. README 의 '데이터 준비'를 확인하세요."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "⚠ Python 이 없습니다. README 의 사전 요구사항을 확인하세요."
}
