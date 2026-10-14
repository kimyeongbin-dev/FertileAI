# -*- coding: utf-8 -*-
"""
AutoGluon 개선 학습 스크립트

변경사항:
- num_bag_folds: 8 → 10 (더 안정적인 CV)
- time_limit: 12시간 → 14시간
- excluded_model_types: RF, XT 제외 (과적합 방지)
"""

import os
import sys
import gc
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
TIME_LIMIT = 50400      # 14시간 (초)
RANDOM_STATE = 42
NUM_BAG_FOLDS = 10      # 기본 8 → 10으로 증가
NUM_BAG_SETS = 1
NUM_STACK_LEVELS = 1

# 경로
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
MODEL_DIR = "./models"
SUBMISSION_DIR = "./submission"

TARGET = '임신 성공 여부'
AUTOGLUON_PRESETS = 'best_quality'
EVAL_METRIC = 'roc_auc'

# 과적합 모델 제외
EXCLUDED_MODEL_TYPES = ['RF', 'XT']


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(
        OUTPUT_DIR,
        f"autogluon_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("AutoGluon 개선 학습 시작")
    logger.info("=" * 60)
    logger.info(f"Time Limit: {TIME_LIMIT // 3600}시간")
    logger.info(f"Num Bag Folds: {NUM_BAG_FOLDS}")
    logger.info(f"Excluded Models: {EXCLUDED_MODEL_TYPES}")

    # 1. AutoGluon 로드
    logger.info("\n[1/5] AutoGluon 로드")
    try:
        from autogluon.tabular import TabularPredictor, TabularDataset
        logger.info("  OK")
    except ImportError as e:
        logger.error(f"  FAILED: {e}")
        sys.exit(1)

    # 2. 데이터 로드
    logger.info("\n[2/5] 데이터 로드")
    import pandas as pd

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    logger.info(f"  Train: {train.shape[0]:,} rows")
    logger.info(f"  Test: {test.shape[0]:,} rows")

    test_ids = test['ID'].copy()
    train = train.drop(columns=['ID'])
    test = test.drop(columns=['ID'])

    train_data = TabularDataset(train)
    test_data = TabularDataset(test)

    # 3. 학습
    logger.info("\n[3/5] 학습 시작")
    logger.info(f"  예상 종료: {datetime.now().strftime('%Y-%m-%d %H:%M')} + 14시간")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "autogluon_improved")

    predictor = TabularPredictor(
        label=TARGET,
        eval_metric=EVAL_METRIC,
        path=save_path,
        verbosity=2,
    )

    predictor.fit(
        train_data=train_data,
        time_limit=TIME_LIMIT,
        presets=AUTOGLUON_PRESETS,
        num_bag_folds=NUM_BAG_FOLDS,
        num_bag_sets=NUM_BAG_SETS,
        num_stack_levels=NUM_STACK_LEVELS,
        excluded_model_types=EXCLUDED_MODEL_TYPES,
    )

    logger.info("  학습 완료!")

    # 4. 평가
    logger.info("\n[4/5] 모델 평가")
    leaderboard = predictor.leaderboard(train_data, silent=True)
    logger.info(f"\n{leaderboard.to_string()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    leaderboard.to_csv(os.path.join(OUTPUT_DIR, "leaderboard_improved.csv"), index=False)

    del train_data
    gc.collect()

    # 5. 예측
    logger.info("\n[5/5] 예측 및 제출 파일 생성")
    predictions = predictor.predict_proba(test_data)

    if isinstance(predictions, pd.DataFrame):
        if 1 in predictions.columns:
            proba = predictions[1].values
        else:
            proba = predictions.iloc[:, 1].values
    else:
        proba = predictions

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({
        'ID': test_ids,
        'probability': proba
    })

    submission_path = os.path.join(SUBMISSION_DIR, "submission_improved.csv")
    submission.to_csv(submission_path, index=False)

    logger.info(f"  저장: {submission_path}")
    logger.info(f"  Mean: {proba.mean():.4f}")
    logger.info(f"  Std:  {proba.std():.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("완료!")
    logger.info(f"기존 LB: 0.7420")
    logger.info(f"모델: {save_path}")
    logger.info(f"제출: {submission_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
