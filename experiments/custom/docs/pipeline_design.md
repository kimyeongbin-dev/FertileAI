# Custom ML Pipeline 설계 문서

> AutoGluon 자동화 대신 수동 제어를 통한 학습 파이프라인
> 작성일: 2026-03-18

---

## 1. 개발 환경

| 항목 | 버전 |
|------|------|
| OS | Windows 10 Education (Build 19045) |
| Python | 3.13.12 |
| GPU | GTX 1660 (6GB VRAM), CUDA 11.1 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| LightGBM | 4.6.0 |
| XGBoost | 3.1.3 |
| CatBoost | 1.2.10 |
| PyTorch | 2.9.1 |

---

## 2. 프로젝트 목표

1. **일반화 성능 극대화**: CV ≈ LB 일관성 확보
2. **데이터 누수 완벽 방지**: 해커톤 규칙 준수
3. **모듈화된 코드 구조**: 재사용 및 실험 용이
4. **학습 목적 문서화**: 모든 결정에 근거 명시

---

## 3. 데이터 누수 방지 전략

### 3.1 핵심 원칙
```
Train 통계만 계산 → Validation/Test에 적용
각 Fold는 자신만의 통계를 계산
```

### 3.2 Option B (채택)
```
8개 Fold 통계의 중앙값(Median) → test.csv에 적용
```

**채택 이유:**
- train 26만 vs test 9만 → train이 충분히 큼
- 8-Fold 중앙값은 안정적인 추정치 제공
- CV와 LB 간 일관성 보장 (동일한 전처리 로직)

### 3.3 처리 순서
1. `StratifiedKFold(n_splits=8)` 분할
2. 각 fold_train에서 통계 계산 (mean, median, mode 등)
3. fold_val에 해당 통계 적용
4. 8개 fold 통계 저장
5. **최종 test.csv**: 8개 통계의 중앙값 적용

---

## 4. 교차 검증 설계

### 4.1 8-Fold Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
```

**8-Fold 선택 이유:**
- 5-Fold 대비 더 안정적인 CV 추정
- train 26만개 → fold당 ~3.2만개 (충분한 크기)
- AutoGluon 기본값(8-Fold)과 일치

### 4.2 OOF (Out-of-Fold) 예측
- 각 fold의 validation 예측을 저장
- 전체 train에 대한 OOF 예측 생성
- L2 메타 모델 학습에 사용

---

## 5. 모델 구성

### 5.1 Base Models (L1)

| 모델 | 전처리 요구사항 | 비고 |
|------|----------------|------|
| CatBoost | 범주형 그대로 사용 | cat_features 지정 |
| LightGBM | Label Encoding | category dtype 지원 |
| XGBoost | Label Encoding | 수치형만 |
| Neural Network | One-Hot + Scaling | StandardScaler |
| Logistic Regression | One-Hot + Scaling | 기준선 모델 |

### 5.2 Meta Model (L2) - 항상 적용
```python
# L2 Stacking
meta_model = LogisticRegression()  # 또는 Ridge
meta_features = [oof_catboost, oof_lgbm, oof_xgb, oof_nn, oof_lr]
meta_model.fit(meta_features, y_train)
```

**L2 항상 적용 이유:**
- 수동 가중치(0.35, 0.25, ...) 대신 데이터 기반 최적 가중치 학습
- AutoGluon WeightedEnsemble이 효과적이었던 실험적 근거

### 5.3 Bagging (옵션)
```python
USE_BAGGING = True  # 설정 가능

if USE_BAGGING:
    # 각 모델을 여러 번 학습 (다른 seed)
    # 예측 평균으로 분산 감소
```

---

## 6. 전처리 파이프라인

### 6.1 범주형 변수 정규화
```python
def normalize_category(value):
    if pd.isna(value):
        return value
    return str(value).strip().upper()
```

**적용 이유:** 동일 카테고리의 다른 표현 통일 (e.g., "Yes", "YES", " yes")

### 6.2 결측치 처리
```python
# 수치형: 중앙값 (Median)
numerical_cols.fillna(fold_train[col].median())

# 범주형: 최빈값 (Mode) 또는 'MISSING'
categorical_cols.fillna(fold_train[col].mode()[0])
```

**Median 선택 이유:** 이상치에 강건함

### 6.3 타겟 인코딩 (Target Encoding)
```python
# fold_train에서만 계산
target_mean = fold_train.groupby(cat_col)[TARGET].mean()

# fold_val, test에 적용
fold_val[cat_col + '_te'] = fold_val[cat_col].map(target_mean)

# Unseen category 처리
global_mean = fold_train[TARGET].mean()
fold_val[cat_col + '_te'].fillna(global_mean, inplace=True)
```

**test.csv 적용:**
- 8개 fold의 target_mean 저장
- 각 category별 8개 값의 중앙값 사용

### 6.4 모델별 인코딩

```python
# CatBoost: 원본 범주형 유지
catboost_features = numerical_cols + categorical_cols

# LightGBM/XGBoost: Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    train[col + '_le'] = le.fit_transform(train[col])
    val[col + '_le'] = le.transform(val[col])  # unseen → 에러 처리 필요

# NN/LogReg: One-Hot Encoding + Scaling
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
scaler = StandardScaler()
```

---

## 7. 피처 엔지니어링

### 7.1 기본 방침
- **피처 선택은 모델에게 위임** (GBDT는 자동으로 중요 피처 선택)
- **피처 추가는 통합 처리** (모든 모델에 동일하게 적용)
- **피처 삭제는 모델별 분리** (필요시 특정 모델만 제외)

### 7.2 추가 피처 후보
```python
# 예시: 상호작용 피처
df['age_x_bmi'] = df['나이'] * df['BMI']

# 예시: 비율 피처
df['ratio_feature'] = df['col_a'] / (df['col_b'] + 1)
```

> 주의: 실험 후 효과 검증 필요 (기존 FE는 +0.00006 효과)

---

## 8. 학습 및 예측 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                      train.csv 로드                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              8-Fold StratifiedKFold 분할                     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ Fold 1  │           │ Fold 2  │    ...    │ Fold 8  │
   └─────────┘           └─────────┘           └─────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│  각 Fold에서:                                                │
│  1. fold_train 통계 계산                                     │
│  2. fold_train, fold_val 전처리                              │
│  3. Base Models (L1) 학습                                    │
│  4. fold_val OOF 예측 저장                                   │
│  5. fold 통계 저장 (test용)                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              L2 Meta Model 학습                              │
│  - Input: 전체 OOF 예측 (5개 모델 × 26만 rows)               │
│  - Output: 최적 앙상블 가중치                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              test.csv 예측                                   │
│  1. 8개 fold 통계의 중앙값으로 전처리                         │
│  2. 8개 fold 모델의 예측 평균 (각 Base Model)                │
│  3. L2 Meta Model로 최종 앙상블                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              submission.csv 생성                             │
│  - ID, probability 컬럼                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 제출 형식 (해커톤 규칙)

### 9.1 요구사항
- **단일 파일 제출**: 모든 코드가 하나의 .py 파일에 포함
- **데이터 경로**: `/data/train.csv`, `/data/test.csv`
- **출력 경로**: `submission.csv` (현재 디렉토리)
- **컬럼**: `ID`, `probability`

### 9.2 제출용 코드 구조
```python
# submission_final.py
import pandas as pd
# ... 모든 import

# === 설정 ===
DATA_DIR = "/data"
# ...

# === 전처리 함수들 ===
def normalize_category(value): ...
def fill_missing(df, stats): ...
# ...

# === 모델 클래스들 ===
class BaseModel: ...
class CatBoostWrapper(BaseModel): ...
# ...

# === 메인 파이프라인 ===
def main():
    # 1. 데이터 로드
    # 2. 8-Fold CV 학습
    # 3. test 예측
    # 4. submission 저장

if __name__ == "__main__":
    main()
```

---

## 10. 실험 기록 (AutoGluon 참고)

### 10.1 기존 AutoGluon 결과
| 모델 | CV AUC | LB AUC | Gap |
|------|--------|--------|-----|
| WeightedEnsemble_L2 | 0.7409 | 0.7420 | +0.0011 |
| CatBoost_BAG_L1 | 0.7397 | - | - |
| XGBoost_BAG_L1 | 0.7357 | 0.7322 | -0.0035 |

### 10.2 수동 앙상블 실험 결과 (2026-03-18)
| 방법 | CV AUC | LB AUC | 비고 |
|------|--------|--------|------|
| AutoGluon WeightedEnsemble | 0.7409 | 0.7420 | 기준선 |
| 수동 앙상블 (RF/XT 제외) | 0.7406 | **0.7420** | CV 낮지만 LB 동등 |

**수동 앙상블 구성:**
- CatBoost: 35%
- LightGBM: 25%
- LightGBMXT: 25%
- XGBoost: 15%

**핵심 발견:**
- CV가 낮아도 LB가 높을 수 있음 (일반화 성능 ≠ CV 성능)
- RF/XT 제외가 일반화에 도움됨
- AutoGluon WeightedEnsemble이 CV에 과적합 가능성

### 10.3 관찰된 패턴
- **RF, XT 과적합**: Train AUC 0.92 vs CV 0.73
- **XGBoost 과적합 경향**: CV > LB (Gap: -0.0035)
- **CatBoost 안정적**: CV ≈ LB
- **FE 효과 미미**: +17 features → +0.00006 AUC

---

## 11. 프로젝트 구조

```
FertileAI-Custom/
├── data/                  # train.csv, test.csv (gitignore)
├── output/                # 모델, 로그, 결과
├── docs/
│   └── pipeline_design.md # 이 문서
├── src/
│   ├── config.py          # 설정값
│   ├── preprocessing.py   # 전처리 함수
│   ├── models/
│   │   ├── base.py
│   │   ├── catboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   ├── neural_net.py
│   │   └── logistic.py
│   ├── pipeline.py        # 메인 파이프라인
│   └── utils.py           # 유틸리티
├── submission_final.py    # 제출용 단일 파일
├── .gitignore
└── README.md
```

---

## 12. 체크리스트

- [ ] `config.py` 작성
- [ ] `preprocessing.py` 작성
- [ ] Base Model 클래스 구현
- [ ] `pipeline.py` 메인 로직 구현
- [ ] 로컬 CV 검증
- [ ] 제출용 단일 파일 병합
- [ ] 제출 및 LB 확인

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-03-18 | 초안 작성 |
| 2026-10-14 | 구현 결과와 누수 재검토는 [retrospective.md](retrospective.md)로 분리. 체크리스트는 당시 상태 그대로 둠 |
