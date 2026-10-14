# -*- coding: utf-8 -*-
"""
AutoGluon 학습 스크립트 — 난임 환자 임신 성공 여부 예측

=============================================================================
개발 환경
=============================================================================
- OS: Windows
- Python: 3.13.12
- AutoGluon: 1.5.0
- Pandas: 2.3.3
- NumPy: 2.1.3

=============================================================================
사용법
=============================================================================
    pip install -r requirements.txt
    python src/train.py

=============================================================================
예상 시간: 12시간
=============================================================================
"""

import os
import sys
import gc
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 설정 (Hyperparameters & Configuration)
# ============================================================================
TIME_LIMIT = 43200      # 학습 제한 시간: 12시간 (초 단위)
RANDOM_STATE = 42       # 재현성을 위한 랜덤 시드

# 데이터 입출력 경로 ('/data' 규정 준수)
DATA_DIR = "./data"           # 데이터 입력 경로
OUTPUT_DIR = "./output"       # 결과 출력 경로
MODEL_DIR = "./models"        # 모델 저장 경로
SUBMISSION_DIR = "./submission"  # 제출 파일 저장 경로

# 타겟 변수
TARGET = '임신 성공 여부'

# AutoGluon 설정
AUTOGLUON_PRESETS = 'best_quality'  # 최고 품질 프리셋
EVAL_METRIC = 'roc_auc'             # 평가 지표: ROC-AUC


# ============================================================================
# 로깅 설정
# ============================================================================
def setup_logging():
    """
    로깅 설정 함수
    - 콘솔과 파일 양쪽에 로그 출력
    - UTF-8 인코딩으로 한글 지원
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_file = os.path.join(
        OUTPUT_DIR,
        f"autogluon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


# ============================================================================
# 메인 학습 함수
# ============================================================================
def main():
    """
    AutoGluon 메인 학습 함수

    1. AutoGluon 설치 확인
    2. 데이터 로드 (train.csv, test.csv)
    3. AutoGluon 학습 (best_quality 프리셋, 12시간)
    4. 모델 평가 및 리더보드 저장
    5. 예측 및 제출 파일 생성
    """
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("AutoGluon 학습 시작")
    logger.info("=" * 60)
    logger.info(f"Time Limit: {TIME_LIMIT // 3600}시간")
    logger.info(f"Random State: {RANDOM_STATE}")
    logger.info(f"Presets: {AUTOGLUON_PRESETS}")
    logger.info(f"Eval Metric: {EVAL_METRIC}")

    # ========================================================================
    # 1. AutoGluon 설치 확인
    # ========================================================================
    logger.info("\n[1/5] AutoGluon 설치 확인")
    try:
        from autogluon.tabular import TabularPredictor, TabularDataset
        logger.info("  AutoGluon 설치 확인 완료")
    except ImportError as e:
        logger.error("  AutoGluon이 설치되지 않았습니다.")
        logger.error("  설치 명령어: pip install autogluon")
        logger.error(f"  에러: {e}")
        sys.exit(1)

    # ========================================================================
    # 2. 데이터 로드
    # ========================================================================
    logger.info("\n[2/5] 데이터 로드")
    try:
        import pandas as pd

        train_path = os.path.join(DATA_DIR, "train.csv")
        test_path = os.path.join(DATA_DIR, "test.csv")

        # 데이터 파일 존재 확인
        if not os.path.exists(train_path):
            logger.error(f"  train.csv를 찾을 수 없습니다: {train_path}")
            sys.exit(1)
        if not os.path.exists(test_path):
            logger.error(f"  test.csv를 찾을 수 없습니다: {test_path}")
            sys.exit(1)

        # 데이터 로드
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        logger.info(f"  Train: {train.shape[0]:,} rows, {train.shape[1]} cols")
        logger.info(f"  Test: {test.shape[0]:,} rows, {test.shape[1]} cols")

        # ID 저장 (제출용)
        test_ids = test['ID'].copy()

        # ID 컬럼 제거 (학습에 불필요)
        train = train.drop(columns=['ID'])
        test = test.drop(columns=['ID'])

        # AutoGluon TabularDataset으로 변환
        train_data = TabularDataset(train)
        test_data = TabularDataset(test)

        logger.info("  데이터 로드 완료")

    except Exception as e:
        logger.error(f"  데이터 로드 실패: {e}")
        sys.exit(1)

    # ========================================================================
    # 3. AutoGluon 학습
    # ========================================================================
    logger.info("\n[3/5] AutoGluon 학습 시작")
    logger.info(f"  예상 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M')} + 12시간")

    try:
        # 모델 저장 경로
        os.makedirs(MODEL_DIR, exist_ok=True)
        save_path = os.path.join(MODEL_DIR, "autogluon_models")

        # AutoGluon TabularPredictor 생성
        predictor = TabularPredictor(
            label=TARGET,
            eval_metric=EVAL_METRIC,
            path=save_path,
            verbosity=2,  # 상세 로그 출력
        )

        # 학습 실행
        # - presets='best_quality': 최고 품질 (앙상블, 스태킹 등 포함)
        # - time_limit: 학습 제한 시간 (초)
        predictor.fit(
            train_data=train_data,
            time_limit=TIME_LIMIT,
            presets=AUTOGLUON_PRESETS,
        )

        logger.info("  학습 완료!")

    except Exception as e:
        logger.error(f"  학습 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # ========================================================================
    # 4. 모델 평가 및 결과 저장
    # ========================================================================
    logger.info("\n[4/5] 모델 평가")

    try:
        # 리더보드 출력
        leaderboard = predictor.leaderboard(train_data, silent=True)
        logger.info("\n  === Leaderboard (모델별 성능) ===")
        logger.info(f"\n{leaderboard.to_string()}")

        # 리더보드 CSV 저장
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        leaderboard.to_csv(os.path.join(OUTPUT_DIR, "leaderboard.csv"), index=False)
        logger.info(f"  리더보드 저장: {OUTPUT_DIR}/leaderboard.csv")

        # 피처 중요도 추출 및 저장
        try:
            importance = predictor.feature_importance(train_data)
            importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))
            logger.info(f"  피처 중요도 저장: {OUTPUT_DIR}/feature_importance.csv")
        except Exception as e:
            logger.warning(f"  피처 중요도 추출 실패: {e}")

        # 메모리 정리
        del train_data
        gc.collect()

    except Exception as e:
        logger.error(f"  평가 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ========================================================================
    # 5. 예측 및 제출 파일 생성
    # ========================================================================
    logger.info("\n[5/5] 예측 및 제출 파일 생성")

    try:
        # 테스트 데이터 예측 (확률값)
        predictions = predictor.predict_proba(test_data)

        # 양성 클래스(1) 확률 추출
        if isinstance(predictions, pd.DataFrame):
            if 1 in predictions.columns:
                proba = predictions[1].values
            else:
                proba = predictions.iloc[:, 1].values
        else:
            proba = predictions

        # 제출 파일 생성
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        submission = pd.DataFrame({
            'ID': test_ids,
            'probability': proba
        })

        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        logger.info(f"  제출 파일 저장: {submission_path}")

        # 예측 통계 출력
        logger.info(f"  예측 확률 통계:")
        logger.info(f"    Mean: {proba.mean():.4f}")
        logger.info(f"    Std:  {proba.std():.4f}")
        logger.info(f"    Min:  {proba.min():.4f}")
        logger.info(f"    Max:  {proba.max():.4f}")

    except Exception as e:
        logger.error(f"  예측 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # ========================================================================
    # 완료
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("AutoGluon 학습 완료!")
    logger.info("=" * 60)
    logger.info(f"  모델 저장 위치: {save_path}")
    logger.info(f"  제출 파일: {submission_path}")
    logger.info(f"  완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    main()
