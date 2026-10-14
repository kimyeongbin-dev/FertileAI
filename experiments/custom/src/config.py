# -*- coding: utf-8 -*-
"""
설정값 모음

모든 하이퍼파라미터와 경로를 한 곳에서 관리합니다.
"""

import os

# =============================================================================
# 경로 설정
# =============================================================================
# 프로젝트 루트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 로컬 개발용
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 제출용 (해커톤 환경)
# DATA_DIR = "/data"
# OUTPUT_DIR = "."

# =============================================================================
# 데이터 설정
# =============================================================================
TARGET = "임신 성공 여부"
ID_COL = "ID"

# =============================================================================
# 학습 설정
# =============================================================================
RANDOM_STATE = 42
N_FOLDS = 8
USE_BAGGING = True   # 각 모델을 여러 seed로 학습
N_BAG_SEEDS = 3      # Bagging 사용 시 seed 개수

# =============================================================================
# 모델별 하이퍼파라미터
# =============================================================================
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "task_type": "GPU",  # CPU로 변경 가능
    "devices": "0",
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_STATE,
    "n_estimators": 1000,
    "early_stopping_rounds": 100,
}

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": RANDOM_STATE,
    "n_estimators": 1000,
    "early_stopping_rounds": 100,
    "tree_method": "hist",  # GPU: "gpu_hist"
    "device": "cuda",       # CPU: "cpu"
}

NN_PARAMS = {
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "epochs": 100,
    "early_stopping_patience": 10,
}

LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

# =============================================================================
# L2 Meta Model 설정
# =============================================================================
META_MODEL_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}
