# -*- coding: utf-8 -*-
"""
OOF 기반 앙상블 CV 검증 스크립트

제출 전에 CV 점수를 확인하여 제출 가치가 있는지 판단합니다.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================================
# 설정
# ============================================================================
MODEL_PATH = "./archive/autogluon/output/autogluon_models"
DATA_PATH = "./data/train.csv"

# 앙상블 구성
ENSEMBLE_CONFIGS = {
    "v1_기본": {
        "models": ["CatBoost_BAG_L1", "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "XGBoost_BAG_L1"],
        "weights": [0.35, 0.25, 0.25, 0.15]
    },
    "v2_CatBoost중심": {
        "models": ["CatBoost_BAG_L1", "CatBoost_r9_BAG_L1", "LightGBM_BAG_L1", "LightGBMXT_BAG_L1"],
        "weights": [0.35, 0.25, 0.20, 0.20]
    },
    "v3_상위3개": {
        "models": ["CatBoost_BAG_L1", "CatBoost_r9_BAG_L1", "LightGBM_r96_BAG_L1"],
        "weights": [0.45, 0.30, 0.25]
    },
    "v4_XGB제외": {
        "models": ["CatBoost_BAG_L1", "LightGBM_BAG_L1", "LightGBMXT_BAG_L1"],
        "weights": [0.45, 0.30, 0.25]
    },
}


def load_oof(model_name):
    """OOF 예측 로드 (AutoGluon 형식)"""
    oof_path = f"{MODEL_PATH}/models/{model_name}/utils/oof.pkl"
    with open(oof_path, 'rb') as f:
        oof_dict = pickle.load(f)
    return oof_dict['_oof_pred_proba']


def main():
    print("=" * 60)
    print("OOF 기반 앙상블 CV 검증")
    print("=" * 60)

    # 타겟 로드
    train = pd.read_csv(DATA_PATH)
    y_true = train['임신 성공 여부'].values

    # 기존 WeightedEnsemble CV
    print("\n[기준] WeightedEnsemble_L2 CV: 0.7409")
    print("-" * 60)

    results = []

    for name, config in ENSEMBLE_CONFIGS.items():
        models = config["models"]
        weights = config["weights"]

        try:
            # OOF 로드 및 앙상블
            oof_preds = []
            for model in models:
                oof = load_oof(model)  # 이미 1D array
                oof_preds.append(oof)

            # 가중 평균
            ensemble_oof = sum(p * w for p, w in zip(oof_preds, weights))

            # CV 점수 계산
            cv_score = roc_auc_score(y_true, ensemble_oof)
            results.append((name, cv_score, models, weights))

            print(f"\n{name}")
            print(f"  모델: {models}")
            print(f"  가중치: {weights}")
            print(f"  CV AUC: {cv_score:.6f}")

            if cv_score > 0.7409:
                print(f"  → 기존 대비: +{cv_score - 0.7409:.6f} (개선!)")
            else:
                print(f"  → 기존 대비: {cv_score - 0.7409:.6f}")

        except Exception as e:
            print(f"\n{name}: FAILED - {e}")

    # 최고 성능 앙상블
    print("\n" + "=" * 60)
    best = max(results, key=lambda x: x[1])
    print(f"최고 CV: {best[0]} ({best[1]:.6f})")

    if best[1] > 0.7409:
        print("→ 제출 추천!")
    else:
        print("→ 기존보다 낮음, 제출 비추천")

    print("=" * 60)


if __name__ == "__main__":
    main()
