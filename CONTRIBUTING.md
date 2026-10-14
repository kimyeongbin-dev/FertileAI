# 개발 규칙

## 1. 시작하기

클론한 직후에 한 번 실행합니다. Git 훅을 연결하고, 학습 데이터가 준비됐는지 확인합니다.

```bash
# macOS / Linux / Git Bash
./scripts/setup.sh
```
```powershell
# Windows PowerShell
./scripts/setup.ps1
```

---

## 2. 커밋 컨벤션

[Conventional Commits](https://www.conventionalcommits.org/ko/v1.0.0/)를 따르고, 요약은 한국어로 씁니다.

```
<type>(<scope>): <요약>

<본문 (선택)>
```

### 2-1. type

| type | 용도 | 예시 |
|---|---|---|
| `feat` | 파이프라인·모델·실험 기능 추가·변경 | `feat(model): CatBoost seed 앙상블 추가` |
| `fix` | 버그 수정 (누수, 잘못된 평가 등 포함) | `fix(cv): fold 밖 통계로 인코딩하던 누수 수정` |
| `refactor` | 동작 변화 없는 구조 개선 | `refactor(preprocess): 결측 처리 함수 분리` |
| `perf` | 학습·추론 속도나 메모리 개선 | `perf(fe): 피처 생성 벡터화` |
| `style` | 포맷 등 동작과 무관한 코드 스타일 | `style: black 포맷 적용` |
| `test` | 테스트 추가·수정 | `test(pipeline): 스모크 테스트 추가` |
| `docs` | 문서만 변경 (README, 실험 기록, 회고) | `docs: 실험 결과 표 갱신` |
| `build` | 의존성·실행 환경 | `build: AutoGluon 1.5.0 고정` |
| `ci` | CI 설정 | `ci: 린트 워크플로 추가` |
| `chore` | 그 밖의 잡무 (설정 파일, 스크립트 등) | `chore: Git 훅 셋업 스크립트 추가` |
| `revert` | 이전 커밋 되돌리기 | `revert: feat(model): CatBoost seed 앙상블 추가` |

### 2-2. scope (선택)
- 변경된 영역을 영문 소문자로 씁니다. 예: `eda`, `preprocess`, `fe`, `model`, `autogluon`, `cv`, `ensemble`, `custom`
- 여러 영역에 걸치면 생략합니다.
- 하위 호환이 깨지는 변경은 type 뒤에 `!`를 붙입니다. 예: `feat(preprocess)!: 출력 컬럼 이름 변경`

### 2-3. 요약 (첫 줄)
- **50자 이내** (type·scope 포함), 끝에 마침표를 찍지 않습니다.
- "무엇을 했는지"를 명사형이나 "~ 수정", "~ 추가"로 끝맺습니다.
- 한 커밋에는 하나의 논리적 변경만 담습니다.

### 2-4. 본문 (선택)
- 요약과 본문 사이는 한 줄 비웁니다.
- **왜** 바꿨는지, 이전에 어떤 문제가 있었는지를 씁니다. 실험이면 CV·LB 점수 변화를 함께 적습니다.
- 목록은 `- `로 시작합니다.

### 2-5. 트레일러
- AI 도구의 `Co-authored-by`, `Generated with ...` 같은 트레일러는 **커밋에 넣지 않습니다.**
  AI 도구를 어떻게 활용했는지는 README의 기술 스택·회고 섹션과 PR 본문에서 밝힙니다.
- 사람 공동 작업자의 `Co-authored-by`, 이슈 참조(`Refs: #12`) 같은 트레일러는 써도 됩니다.

### 2-6. 임시 커밋
- 작업 중에는 `wip: ...`, `fixup! ...`, `squash! ...` 커밋을 만들어도 됩니다.
- 이런 커밋은 `main`, `dev`에 push할 수 없습니다. PR을 머지하기 전에 `git rebase -i --autosquash`로 정리합니다.

---

## 3. 브랜치

| 브랜치 | 용도 |
|---|---|
| `main` | 릴리스 기준. 버전 태그(`v1`, `v1.1`, ...)를 이 브랜치에 답니다 |
| `dev` | 다음 릴리스를 통합하는 브랜치 |
| `feature/<이름>` | 기능·실험 단위 작업. `dev`에서 갈라져 PR로 `dev`에 머지 |

### 작업 흐름
1. `git switch -c feature/<이름> dev`
2. 컨벤션에 맞춰 커밋한 뒤 push
3. `dev`를 대상으로 PR을 엽니다. [PR 템플릿](.github/pull_request_template.md)의 구현 내용, 트러블슈팅, 고민한 점을 채워 개발 기록으로 남깁니다.
4. **Merge commit**으로 머지합니다 (기능 단위 커밋과 PR 연결을 보존). 머지한 feature 브랜치는 삭제합니다.
5. 버전 목표를 달성하면 `dev` → `main` PR로 릴리스하고, `main`에 버전 태그를 답니다.

---

## 4. Git 훅이 자동으로 막는 것

`.githooks/`에 있으며, `scripts/setup.*`가 `git config core.hooksPath .githooks`로 연결합니다.

| 훅 | 막는 것 |
|---|---|
| `commit-msg` | 위 형식에 맞지 않는 첫 줄, 50자를 넘는 요약, 마침표로 끝나는 요약, 요약 다음 줄이 비어 있지 않은 경우, AI 트레일러 |
| `pre-push` | `main`·`dev`로 push되는 커밋 중 `wip`·`fixup!`·`squash!`·`amend!` 커밋, AI 트레일러가 남은 커밋 |

Git이 자동으로 만드는 `Merge ...`, `Revert "..."` 메시지는 통과시킵니다.

---

## 5. 데이터와 산출물

- 대회 규정상 원본 데이터(`data/`)는 커밋하지 않습니다.
- 학습된 모델, 캐시, 제출 파일은 다시 만들 수 있으므로 커밋하지 않습니다.
- 실험 근거로 남길 작은 결과(리더보드, 피처 중요도 등)는 `results/` 아래에 둡니다.
