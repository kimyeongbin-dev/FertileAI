# -*- coding: utf-8 -*-
"""
AutoGluon 개선 예측 스크립트 — 과적합 모델 제외 앙상블

기존 학습된 모델을 로드하고, 과적합이 심한 모델(RF, XT)을 제외한
앙상블로 예측하여 일반화 성능 개선을 시도합니다.

사용법:
    python src/predict_improved.py
"""

import os
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset

# ============================================================================
# 설정
# ============================================================================
MODEL_PATH = "./archive/autogluon/output/autogluon_models"
DATA_DIR = "./data"
OUTPUT_DIR = "./submission"

# 제외할 과적합 모델 (Train AUC 0.92 vs CV 0.73)
EXCLUDED_MODELS = [
    'RandomForestEntr_BAG_L1',
    'RandomForestGini_BAG_L1',
    'ExtraTreesEntr_BAG_L1',
    'ExtraTreesGini_BAG_L1',
]

# ============================================================================
# 메인
# ============================================================================
def main():
    print("=" * 60)
    print("AutoGluon 개선 예측 (과적합 모델 제외)")
    print("=" * 60)

    # 1. 기존 모델 로드
    print("\n[1/4] 모델 로드")
    predictor = TabularPredictor.load(MODEL_PATH)
    print(f"  모델 경로: {MODEL_PATH}")

    # 2. 리더보드 확인
    print("\n[2/4] 모델 리더보드")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard[['model', 'score_val']].to_string())

    # 3. 사용 가능한 모델 확인
    print("\n[3/4] 앙상블 구성")
    all_models = predictor.model_names()
    print(f"  전체 모델: {len(all_models)}개")
    print(f"  제외 모델: {EXCLUDED_MODELS}")

    # 제외 후 남은 모델
    included_models = [m for m in all_models if m not in EXCLUDED_MODELS]
    print(f"  사용 모델: {len(included_models)}개")

    # 4. 테스트 데이터 예측
    print("\n[4/4] 예측")
    test_path = os.path.join(DATA_DIR, "test.csv")
    test = pd.read_csv(test_path)
    test_ids = test['ID'].copy()
    test = test.drop(columns=['ID'])
    test_data = TabularDataset(test)

    # 방법 1: 특정 모델들로 수동 앙상블
    print("\n  === 방법 1: 수동 가중 앙상블 ===")

    # 상위 모델들의 예측 평균
    top_models = ['CatBoost_BAG_L1', 'LightGBM_BAG_L1', 'LightGBMXT_BAG_L1', 'XGBoost_BAG_L1']
    weights = [0.35, 0.25, 0.25, 0.15]  # CatBoost 중심

    predictions = []
    for model in top_models:
        try:
            proba = predictor.predict_proba(test_data, model=model)
            if isinstance(proba, pd.DataFrame):
                proba = proba[1].values if 1 in proba.columns else proba.iloc[:, 1].values
            predictions.append(proba)
            print(f"    {model}: OK")
        except Exception as e:
            print(f"    {model}: FAILED - {e}")

    # 가중 평균
    ensemble_proba = sum(p * w for p, w in zip(predictions, weights))

    # 제출 파일 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({
        'ID': test_ids,
        'probability': ensemble_proba
    })

    output_path = os.path.join(OUTPUT_DIR, "submission_improved_v1.csv")
    submission.to_csv(output_path, index=False)
    print(f"\n  저장: {output_path}")
    print(f"  Mean: {ensemble_proba.mean():.4f}")
    print(f"  Std:  {ensemble_proba.std():.4f}")

    print("\n" + "=" * 60)
    print("완료! submission_improved_v1.csv 제출하여 LB 확인")
    print("기존 WeightedEnsemble LB: 0.7420")
    print("=" * 60)


if __name__ == "__main__":
    main()
