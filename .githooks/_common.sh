# 훅 공통 설정 (직접 실행하지 않음)
export LC_ALL=C.UTF-8

# 커밋에 넣지 않는 AI 도구 트레일러
AI_TRAILER_PATTERN='co-authored-by:.*(claude|anthropic|copilot|chatgpt|openai|gemini|cursor|codeium|devin)|generated (with|by) .*(claude|chatgpt|copilot|gemini|cursor|codex)|🤖'

# main / dev 에 들어가면 안 되는 임시 커밋 (요약 기준)
TEMP_COMMIT_PATTERN='^(fixup!|squash!|amend!|wip\b)'
