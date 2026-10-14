# 회고: AutoGluon 없이 누수 방지 스태킹 파이프라인 직접 설계

> 작성일: 2026-10-14 (작업 기간: 2026-03-18 ~ 03-19)
> 관련 문서: [pipeline_design.md](pipeline_design.md) (설계 당시 문서)

---

## 1. 한 줄 요약

대회에서 제출한 AutoGluon 앙상블(Public LB 0.7420)을 **직접 재현할 수 있는가**를 확인하려고,
전처리부터 L2 스태킹까지 수동으로 구현했다. 결과적으로 **CV AUC 0.7406, Public LB 0.7419**로 AutoGluon(CV 0.7409)과
거의 같은 수준에 도달했다.

---

## 2. 왜 했는가

대회에서 AutoGluon `WeightedEnsemble_L2`가 가장 좋은 성능을 냈지만, 내부에서 무엇이 일어나는지 설명할 수 없었다.

- 전처리 통계(결측 대체값, 인코딩)가 **어느 데이터로** 계산되는가?
- 앙상블 가중치는 **어떻게** 정해지는가?
- CV와 LB가 왜 일치하거나 어긋나는가?

"도구가 해준 것"이 아니라 "내가 설계한 것"으로 설명하기 위해 같은 구조를 손으로 다시 만들었다.

---

## 3. 무엇을 설계했는가

### 3.1 전체 구조

```
train.csv ─▶ 8-Fold StratifiedKFold
              │
              ├─ fold마다: fold_train으로만 통계 fit ─▶ fold_train/fold_val 변환
              │            └─ L1 모델 5종 학습 ─▶ fold_val OOF 예측
              │            └─ fold 통계 저장
              │
              ├─ L2: OOF 5열 ─▶ LogisticRegression (앙상블 가중치 학습)
              │
test.csv ─▶ 8개 fold 통계의 중앙값으로 변환 ─▶ 8개 fold 모델 평균 ─▶ L2 ─▶ submission
```

### 3.2 핵심 결정과 근거

| 결정 | 내용 | 근거 |
|------|------|------|
| **fold 내부 fit** | 결측 대체(median/mode), 타겟 인코딩, 라벨 인코딩, 스케일러를 모두 `fold_train`으로만 계산 (`Preprocessor.fit`) | validation 정보가 전처리에 섞이면 CV가 낙관적으로 부풀려짐 |
| **Option B: fold 통계 중앙값** | test에는 8개 fold 통계의 중앙값 적용 (`FoldStatistics.get_median_stats`) | 전체 train으로 다시 fit하면 fold 모델이 학습한 분포와 달라짐. 중앙값은 이상 fold에 강건 |
| **모델별 인코딩 분리** | CatBoost는 원본 범주형, LGBM/XGB는 Label Encoding, NN/LogReg는 One-Hot + Scaling | 모델마다 범주형을 다루는 방식이 다름 |
| **L2 스태킹 상시 적용** | OOF 예측을 입력으로 LogisticRegression 메타모델 학습 | 앞서 제출한 수동 가중치(35/25/25/15) 대신 데이터로 가중치 결정 |
| **Bagging 옵션** | 모델별 seed 3개 × 8 fold = 모델 120개 | 예측 분산 감소 효과 확인용 |

### 3.3 코드 구성

| 파일 | 역할 |
|------|------|
| `src/config.py` | 경로, 하이퍼파라미터, fold/bagging 설정 |
| `src/preprocessing.py` | `Preprocessor`(fold 단위 fit/transform), `FoldStatistics`(중앙값 집계) |
| `src/models/` | 5개 모델을 `BaseModel` 공통 인터페이스로 래핑 |
| `src/pipeline.py` | 기본 파이프라인 (8-fold × 5 모델 + L2) |
| `src/pipeline_bagging.py` | seed bagging 버전 |
| `src/test_pipeline.py` | 소량 샘플로 전체 흐름 스모크 테스트 |

---

## 4. 결과

> 아래 수치는 실행 로그 기준이다. 학습된 모델과 로그(`output/`)는 레포에 포함하지 않았다.

### 4.1 OOF AUC (전체 train 기준)

| 모델 | 기본 (40 모델) | Bagging ×3 (120 모델) |
|------|------|------|
| CatBoost | 0.73996 | 0.74016 |
| LightGBM | 0.73957 | 0.74003 |
| XGBoost | 0.73977 | 0.73994 |
| NeuralNet | 0.73701 | 0.73857 |
| Logistic | 0.73122 | 0.73122 |
| **L2 Meta** | **0.74048** | **0.74063** |
| 학습 시간 | 1h 45m | 3h 09m |

L2 가중치 (Bagging, 순서: Cat / LGBM / XGB / NN / LogReg): `1.40 / 1.59 / 1.39 / 1.66 / 0.31`

### 4.2 AutoGluon 제출과 비교

| 방법 | CV AUC | Public LB AUC |
|------|--------|--------|
| AutoGluon WeightedEnsemble_L2 | 0.7409 | 0.7420 |
| AutoGluon 모델 4개 수동 가중 앙상블 | 0.7409 | **0.7420** |
| 이 프로젝트 (L2 Stacking) | 0.7405 | 0.7419 |
| **이 프로젝트 (L2 Bagging)** | **0.7406** | **0.7419** |

CV와 LB의 순서가 AutoGluon과 이 프로젝트 사이에서 일관되게 유지돼, 이 CV 설계가 LB를 잘 따라간다는 것을 확인했다.

### 4.3 관찰

- **단일 GBDT 3종의 성능이 거의 같다 (0.7396~0.7402).** 이 데이터에서는 모델 선택보다 데이터 자체가 성능 상한을 정한다.
- **NN은 단독으로 가장 약한 편인데 L2 가중치는 가장 크다 (1.66).** 트리 모델과 오차 패턴이 달라서 앙상블에 다양성을 더한다.
- **Bagging ×3의 이득은 +0.00015에 그친다.** 학습 시간은 1.8배로 늘어 비용 대비 효과가 낮다.
- 결론적으로 AutoGluon 수준은 **수동 구현으로 재현 가능**했지만, 넘어서지는 못했다.

---

## 5. 남은 개선점

| 항목 | 개선 방향 |
|------|------|
| fold_train 안의 타겟 인코딩 | fold_train 안에서 한 번 더 K-fold로 나눈 out-of-fold 인코딩과 smoothing 적용 |
| test의 라벨 인코딩 매핑 | 범주 매핑은 타겟을 쓰지 않으므로 train+test 전체 범주로 한 번만 고정 |
| 메타모델 CV 점수 | OOF에 다시 K-fold를 돌려 메타모델 자체의 OOF AUC로 보고 |

---

## 6. 배운 것

1. **"누수 방지"는 원칙 한 줄이 아니라 통계 종류마다 따로 따져야 한다.**
   결측 대체값은 중앙값으로 합칠 수 있지만, 라벨 인코딩 매핑은 합칠 수 없고, 타겟 인코딩은 fold 안에서도 한 번 더 나눠야 한다.
2. **AutoGluon이 해주던 일의 실체를 알게 됐다.** bagging, OOF 기반 stacking, 모델별 전처리 분기를 직접 구현하면서,
   AutoGluon의 `BAG_L1` → `WeightedEnsemble_L2` 구조를 설명할 수 있게 됐다.
3. **이 데이터에서는 모델링보다 데이터가 상한을 정한다.** GBDT 3종, bagging, stacking 모두 0.740 근처에 수렴했다.
   다음에 성능을 올리려면 모델이 아니라 도메인 기반 피처에 시간을 써야 한다.

---

## 7. 다시 실행하려면

데이터는 레포 루트의 `data/`에 두고, 이 폴더(`experiments/custom/`)의 `data/`로 복사해 실행한다.

```bash
mkdir -p data && cp ../../data/train.csv ../../data/test.csv data/
python src/test_pipeline.py   # 스모크 테스트 (수 초)
python src/pipeline.py        # 전체 학습 (GTX 1660 기준 약 1h 45m)
```

`config.py`의 CatBoost/XGBoost는 GPU 설정이므로, GPU가 없으면 `task_type="CPU"`, `device="cpu"`로 바꾼다.
