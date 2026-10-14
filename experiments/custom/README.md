# FertileAI-custom

IVF(체외수정) 임신 성공 예측을 위한 커스텀 ML 파이프라인

> [FertileAI](../../README.md)의 AutoGluon 앙상블을 수동으로 재현한 실험.
> **결과와 회고는 [`docs/retrospective.md`](docs/retrospective.md)** (CV AUC 0.7406, Public LB 0.7419)

## 목표

- AutoGluon 자동화 대신 수동 제어
- 데이터 누수 방지 철저
- CV ≈ LB 일관성 확보
- 학습 목적 문서화

## 빠른 시작

```bash
# 데이터 배치 (레포 루트 data/의 원본 사용)
mkdir -p data && cp ../../data/train.csv ../../data/test.csv data/

# 스모크 테스트 → 전체 학습 및 예측
python src/test_pipeline.py
python src/pipeline.py          # 또는 src/pipeline_bagging.py
```

## 프로젝트 구조

```
experiments/custom/
├── data/                  # train.csv, test.csv (gitignore, 실행 시 복사)
├── output/                # 모델, 로그, 결과 (gitignore, 실행 시 생성)
├── docs/
│   ├── pipeline_design.md # 설계 문서 (2026-03-18)
│   └── retrospective.md   # 결과 및 회고 (2026-10-14)
└── src/
    ├── config.py
    ├── preprocessing.py
    ├── models/
    ├── pipeline.py
    ├── pipeline_bagging.py
    └── test_pipeline.py
```

## 핵심 설계

- **8-Fold StratifiedKFold**: 안정적인 CV
- **Option B**: 8개 fold 통계의 중앙값 → test 적용
- **L2 Stacking**: 항상 적용 (최적 앙상블 가중치 학습)
- **Bagging**: 옵션

자세한 내용은 `docs/pipeline_design.md`, 결과와 한계는 `docs/retrospective.md` 참고

## 환경

- Python 3.13.12
- scikit-learn 1.7.2
- LightGBM 4.6.0
- XGBoost 3.1.3
- CatBoost 1.2.10
