# -*- coding: utf-8 -*-
"""
Pseudo Labeling 실험

1. 기존 모델로 테스트 예측
2. 고신뢰 샘플 추출 (prob > 0.85 or prob < 0.15)
3. pseudo label로 train에 추가
4. 재학습 (짧은 time_limit으로 테스트)
"""

import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor, TabularDataset

# ============================================================================
# 설정
# ============================================================================
MODEL_PATH = "./archive/autogluon/output/autogluon_models"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
MODEL_DIR = "./models"
SUBMISSION_DIR = "./submission"

TARGET = '임신 성공 여부'

# Pseudo labeling 설정
HIGH_CONFIDENCE_THRESHOLD = 0.85  # 양성 확신
LOW_CONFIDENCE_THRESHOLD = 0.15   # 음성 확신
TIME_LIMIT = 3600  # 1시간 테스트 (본 학습은 더 길게)


def main():
    print("=" * 60)
    print("Pseudo Labeling 실험")
    print("=" * 60)

    # 1. 기존 모델 로드
    print("\n[1/5] 기존 모델 로드")
    predictor = TabularPredictor.load(MODEL_PATH)
    print(f"  모델: {MODEL_PATH}")

    # 2. 데이터 로드
    print("\n[2/5] 데이터 로드")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    test_ids = test['ID'].copy()
    train = train.drop(columns=['ID'])
    test_no_id = test.drop(columns=['ID'])

    print(f"  Train: {len(train):,} rows")
    print(f"  Test: {len(test):,} rows")

    # 3. 테스트 예측 및 고신뢰 샘플 추출
    print("\n[3/5] 고신뢰 샘플 추출")
    test_data = TabularDataset(test_no_id)
    proba = predictor.predict_proba(test_data)

    if isinstance(proba, pd.DataFrame):
        proba = proba[1].values if 1 in proba.columns else proba.iloc[:, 1].values

    # 고신뢰 샘플 마스크
    high_conf_pos = proba >= HIGH_CONFIDENCE_THRESHOLD
    high_conf_neg = proba <= LOW_CONFIDENCE_THRESHOLD
    high_conf_mask = high_conf_pos | high_conf_neg

    n_pos = high_conf_pos.sum()
    n_neg = high_conf_neg.sum()
    n_total = high_conf_mask.sum()

    print(f"  임계값: pos >= {HIGH_CONFIDENCE_THRESHOLD}, neg <= {LOW_CONFIDENCE_THRESHOLD}")
    print(f"  고신뢰 양성: {n_pos:,} ({n_pos/len(test)*100:.1f}%)")
    print(f"  고신뢰 음성: {n_neg:,} ({n_neg/len(test)*100:.1f}%)")
    print(f"  총 추가 샘플: {n_total:,} ({n_total/len(test)*100:.1f}%)")

    if n_total == 0:
        print("\n  고신뢰 샘플 없음. 임계값 조정 필요.")
        return

    # 4. Pseudo label 데이터 생성
    print("\n[4/5] Pseudo label 데이터 생성")
    pseudo_data = test_no_id[high_conf_mask].copy()
    pseudo_labels = (proba[high_conf_mask] >= 0.5).astype(int)
    pseudo_data[TARGET] = pseudo_labels

    print(f"  Pseudo 양성: {(pseudo_labels == 1).sum():,}")
    print(f"  Pseudo 음성: {(pseudo_labels == 0).sum():,}")

    # Train + Pseudo 합치기
    combined_train = pd.concat([train, pseudo_data], ignore_index=True)
    print(f"  Combined Train: {len(combined_train):,} rows (+{len(pseudo_data):,})")

    # 5. 재학습 (짧은 시간으로 테스트)
    print(f"\n[5/5] 재학습 (time_limit={TIME_LIMIT//3600}시간)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "autogluon_pseudo")

    combined_data = TabularDataset(combined_train)

    predictor_new = TabularPredictor(
        label=TARGET,
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2,
    )

    predictor_new.fit(
        train_data=combined_data,
        time_limit=TIME_LIMIT,
        presets='best_quality',
        excluded_model_types=['RF', 'XT'],  # 과적합 모델 제외
    )

    # 리더보드
    print("\n  === Leaderboard ===")
    leaderboard = predictor_new.leaderboard(silent=True)
    print(leaderboard[['model', 'score_val']].head(10).to_string())

    # 예측 및 제출 파일 생성
    print("\n  === 예측 ===")
    test_proba = predictor_new.predict_proba(test_data)
    if isinstance(test_proba, pd.DataFrame):
        test_proba = test_proba[1].values if 1 in test_proba.columns else test_proba.iloc[:, 1].values

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({
        'ID': test_ids,
        'probability': test_proba
    })

    output_path = os.path.join(SUBMISSION_DIR, "submission_pseudo.csv")
    submission.to_csv(output_path, index=False)

    print(f"  저장: {output_path}")
    print(f"  Mean: {test_proba.mean():.4f}")

    print("\n" + "=" * 60)
    print("완료! CV 확인 후 제출 결정")
    print("기존 WeightedEnsemble LB: 0.7420")
    print("=" * 60)


if __name__ == "__main__":
    main()
