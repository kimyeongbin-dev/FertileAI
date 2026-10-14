# -*- coding: utf-8 -*-
"""
파이프라인 테스트 스크립트

- 데이터 1000개 샘플만 사용
- 2-Fold CV
- 모델 iteration 최소화
- 전체 흐름 검증용
"""

import os
import sys

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils import set_seed, setup_logging
from preprocessing import Preprocessor, FoldStatistics, identify_column_types
from models import (
    CatBoostModel, LightGBMModel, XGBoostModel, NeuralNetModel, LogisticModel
)

# =============================================================================
# 테스트용 설정 (최소화)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TARGET = "임신 성공 여부"
ID_COL = "ID"
RANDOM_STATE = 42

N_FOLDS = 2  # 2-Fold로 축소
N_SAMPLES = 1000  # 1000개 샘플만

# 모델 파라미터 (최소 iteration)
CATBOOST_PARAMS = {
    "iterations": 10,
    "learning_rate": 0.1,
    "depth": 4,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "task_type": "CPU",  # 테스트는 CPU로
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 10,
    "verbose": -1,
    "seed": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 10,
    "seed": RANDOM_STATE,
    "verbosity": 0,
}

NN_PARAMS = {
    "hidden_dims": [32, 16],
    "dropout": 0.3,
    "learning_rate": 0.01,
    "batch_size": 256,
    "epochs": 3,
    "early_stopping_patience": 2,
}

LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 100,
    "random_state": RANDOM_STATE,
}


def main():
    set_seed(RANDOM_STATE)
    logger = setup_logging(OUTPUT_DIR, "test_pipeline")

    logger.info("=" * 60)
    logger.info("파이프라인 테스트 (샘플 데이터)")
    logger.info("=" * 60)

    # 1. 데이터 로드 (샘플링)
    logger.info("\n[1/6] 데이터 로드")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # 샘플링
    train = train.sample(n=N_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
    test = test.sample(n=N_SAMPLES // 3, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info(f"  Train (sampled): {train.shape}")
    logger.info(f"  Test (sampled): {test.shape}")

    # ID 분리
    test_ids = test[ID_COL].values
    train = train.drop(columns=[ID_COL])
    test = test.drop(columns=[ID_COL])

    # 타겟 분리
    y = train[TARGET].values
    X = train.drop(columns=[TARGET])

    # 컬럼 타입 식별
    numerical_cols, categorical_cols = identify_column_types(X)
    logger.info(f"  수치형: {len(numerical_cols)}개, 범주형: {len(categorical_cols)}개")

    # 2. CV 실행
    logger.info(f"\n[2/6] {N_FOLDS}-Fold CV")

    fold_statistics = FoldStatistics()
    fold_models = {"catboost": [], "lightgbm": [], "xgboost": [], "neural_net": [], "logistic": []}
    fold_preprocessors = []  # 각 fold의 preprocessor stats 저장

    n_samples = len(y)
    oof = {name: np.zeros(n_samples) for name in fold_models.keys()}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # 전처리
        X_train_fold[TARGET] = y_train_fold
        preprocessor = Preprocessor(TARGET, numerical_cols, categorical_cols)
        preprocessor.fit(X_train_fold)
        X_train_fold = X_train_fold.drop(columns=[TARGET])

        fold_statistics.add_fold_stats(preprocessor.get_stats())
        # fold_preprocessors는 NN 학습 후에 저장 (onehot_columns 포함 위해)

        # CatBoost
        logger.info("  CatBoost...")
        X_train_cat, cat_features = preprocessor.transform_for_catboost(
            X_train_fold.assign(**{TARGET: y_train_fold})
        )
        X_val_cat, _ = preprocessor.transform_for_catboost(
            X_val_fold.assign(**{TARGET: y_val_fold})
        )
        X_train_cat = X_train_cat.drop(columns=[TARGET])
        X_val_cat = X_val_cat.drop(columns=[TARGET])

        model = CatBoostModel(CATBOOST_PARAMS.copy())
        model.fit(X_train_cat, y_train_fold, X_val_cat, y_val_fold, cat_features)
        oof["catboost"][val_idx] = model.predict_proba(X_val_cat)
        fold_models["catboost"].append(model)
        logger.info(f"    AUC: {roc_auc_score(y_val_fold, oof['catboost'][val_idx]):.4f}")

        # LightGBM
        logger.info("  LightGBM...")
        X_train_lgb = preprocessor.transform_for_lgbm_xgb(X_train_fold)
        X_val_lgb = preprocessor.transform_for_lgbm_xgb(X_val_fold)

        model = LightGBMModel(LIGHTGBM_PARAMS.copy())
        model.fit(X_train_lgb, y_train_fold, X_val_lgb, y_val_fold)
        oof["lightgbm"][val_idx] = model.predict_proba(X_val_lgb)
        fold_models["lightgbm"].append(model)
        logger.info(f"    AUC: {roc_auc_score(y_val_fold, oof['lightgbm'][val_idx]):.4f}")

        # XGBoost
        logger.info("  XGBoost...")
        model = XGBoostModel(XGBOOST_PARAMS.copy())
        model.fit(X_train_lgb, y_train_fold, X_val_lgb, y_val_fold)
        oof["xgboost"][val_idx] = model.predict_proba(X_val_lgb)
        fold_models["xgboost"].append(model)
        logger.info(f"    AUC: {roc_auc_score(y_val_fold, oof['xgboost'][val_idx]):.4f}")

        # Neural Network
        logger.info("  NeuralNet...")
        X_train_nn = preprocessor.transform_for_nn(X_train_fold, is_train=True)
        X_val_nn = preprocessor.transform_for_nn(X_val_fold, is_train=False)

        model = NeuralNetModel(NN_PARAMS.copy())
        model.fit(X_train_nn, y_train_fold, X_val_nn, y_val_fold)
        oof["neural_net"][val_idx] = model.predict_proba(X_val_nn)
        fold_models["neural_net"].append(model)
        logger.info(f"    AUC: {roc_auc_score(y_val_fold, oof['neural_net'][val_idx]):.4f}")

        # Logistic
        logger.info("  Logistic...")
        model = LogisticModel(LOGISTIC_PARAMS.copy())
        model.fit(X_train_nn, y_train_fold)
        oof["logistic"][val_idx] = model.predict_proba(X_val_nn)
        fold_models["logistic"].append(model)
        logger.info(f"    AUC: {roc_auc_score(y_val_fold, oof['logistic'][val_idx]):.4f}")

        # fold_preprocessors 저장 (NN 학습 후 - onehot_columns 포함)
        fold_preprocessors.append(preprocessor.get_stats().copy())

        gc.collect()

    # 3. CV 결과
    logger.info("\n[3/6] CV 결과")
    for name, pred in oof.items():
        logger.info(f"  {name}: {roc_auc_score(y, pred):.4f}")

    # 4. Meta Model
    logger.info("\n[4/6] L2 Meta Model")
    meta_features = np.column_stack(list(oof.values()))
    meta_model = LogisticRegression(C=1.0, max_iter=100, random_state=RANDOM_STATE)
    meta_model.fit(meta_features, y)

    meta_pred = meta_model.predict_proba(meta_features)[:, 1]
    logger.info(f"  Meta AUC: {roc_auc_score(y, meta_pred):.4f}")
    logger.info(f"  가중치: {meta_model.coef_[0].round(4)}")

    # 5. Test 예측
    logger.info("\n[5/6] Test 예측")
    median_stats = fold_statistics.get_median_stats()
    preprocessor = Preprocessor(TARGET, numerical_cols, categorical_cols)
    preprocessor.fit_from_stats(median_stats)

    test_preds = {}

    # CatBoost
    test_cat, cat_features = preprocessor.transform_for_catboost(test)
    preds = [m.predict_proba(test_cat) for m in fold_models["catboost"]]
    test_preds["catboost"] = np.mean(preds, axis=0)

    # LightGBM, XGBoost
    test_lgb = preprocessor.transform_for_lgbm_xgb(test)
    test_preds["lightgbm"] = np.mean([m.predict_proba(test_lgb) for m in fold_models["lightgbm"]], axis=0)
    test_preds["xgboost"] = np.mean([m.predict_proba(test_lgb) for m in fold_models["xgboost"]], axis=0)

    # NN, Logistic - 각 fold의 preprocessor 사용
    nn_preds = []
    lr_preds = []
    for i, (nn_model, lr_model) in enumerate(zip(fold_models["neural_net"], fold_models["logistic"])):
        fold_prep = Preprocessor(TARGET, numerical_cols, categorical_cols)
        fold_prep.fit_from_stats(fold_preprocessors[i])
        test_nn_fold = fold_prep.transform_for_nn(test.copy(), is_train=False)
        nn_preds.append(nn_model.predict_proba(test_nn_fold))
        lr_preds.append(lr_model.predict_proba(test_nn_fold))
    test_preds["neural_net"] = np.mean(nn_preds, axis=0)
    test_preds["logistic"] = np.mean(lr_preds, axis=0)

    # Meta 앙상블
    test_meta = np.column_stack(list(test_preds.values()))
    final_pred = meta_model.predict_proba(test_meta)[:, 1]

    logger.info(f"  Final Mean: {final_pred.mean():.4f}")

    # 6. 제출 파일
    logger.info("\n[6/6] 제출 파일 저장")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"ID": test_ids, "probability": final_pred})
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission_test.csv"), index=False)
    logger.info(f"  저장: {OUTPUT_DIR}/submission_test.csv")

    logger.info("\n" + "=" * 60)
    logger.info("테스트 완료! 오류 없음.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
