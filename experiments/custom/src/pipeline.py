# -*- coding: utf-8 -*-
"""
메인 파이프라인

8-Fold StratifiedKFold + L2 Stacking

핵심 흐름:
1. 데이터 로드
2. 8-Fold CV로 Base Models 학습 + OOF 생성
3. L2 Meta Model 학습
4. test.csv 예측 (8개 fold 통계 중앙값 적용)
5. submission.csv 저장
"""

import os
import gc
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import (
    DATA_DIR, OUTPUT_DIR, TARGET, ID_COL, RANDOM_STATE, N_FOLDS,
    USE_BAGGING, N_BAG_SEEDS,
    CATBOOST_PARAMS, LIGHTGBM_PARAMS, XGBOOST_PARAMS, NN_PARAMS, LOGISTIC_PARAMS,
    META_MODEL_PARAMS,
)
from utils import set_seed, setup_logging, reduce_memory_usage
from preprocessing import Preprocessor, FoldStatistics, identify_column_types
from models import (
    CatBoostModel, LightGBMModel, XGBoostModel, NeuralNetModel, LogisticModel
)


class Pipeline:
    """메인 학습 파이프라인"""

    def __init__(self, logger=None):
        self.logger = logger or setup_logging(OUTPUT_DIR)
        self.fold_statistics = FoldStatistics()
        self.fold_preprocessors: List[Dict[str, Any]] = []  # 각 fold의 preprocessor stats
        self.fold_models: Dict[str, List[Any]] = {
            "catboost": [],
            "lightgbm": [],
            "xgboost": [],
            "neural_net": [],
            "logistic": [],
        }
        self.oof_predictions: Dict[str, np.ndarray] = {}
        self.meta_model = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        self.logger.info("=" * 60)
        self.logger.info("데이터 로드")
        self.logger.info("=" * 60)

        train_path = os.path.join(DATA_DIR, "train.csv")
        test_path = os.path.join(DATA_DIR, "test.csv")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        self.logger.info(f"Train: {train.shape[0]:,} rows, {train.shape[1]} cols")
        self.logger.info(f"Test: {test.shape[0]:,} rows, {test.shape[1]} cols")

        # 메모리 최적화
        train = reduce_memory_usage(train, verbose=False)
        test = reduce_memory_usage(test, verbose=False)

        return train, test

    def run_cv(self, train: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        8-Fold CV 실행

        Returns:
            OOF 예측 딕셔너리 {model_name: oof_array}
        """
        self.logger.info("=" * 60)
        self.logger.info(f"{N_FOLDS}-Fold Cross Validation 시작")
        self.logger.info("=" * 60)

        # ID 분리
        train_ids = train[ID_COL].values
        train = train.drop(columns=[ID_COL])

        # 타겟 분리
        y = train[TARGET].values
        X = train.drop(columns=[TARGET])

        # 컬럼 타입 식별
        self.numerical_cols, self.categorical_cols = identify_column_types(X)
        self.logger.info(f"수치형 컬럼: {len(self.numerical_cols)}개")
        self.logger.info(f"범주형 컬럼: {len(self.categorical_cols)}개")

        # OOF 초기화
        n_samples = len(y)
        oof_catboost = np.zeros(n_samples)
        oof_lightgbm = np.zeros(n_samples)
        oof_xgboost = np.zeros(n_samples)
        oof_neural_net = np.zeros(n_samples)
        oof_logistic = np.zeros(n_samples)

        # K-Fold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Fold {fold + 1}/{N_FOLDS}")
            self.logger.info(f"{'='*60}")

            # Fold 데이터 분할
            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # 전처리기 생성 및 fit (fold_train으로만)
            X_train_fold[TARGET] = y_train_fold  # fit에 타겟 필요
            preprocessor = Preprocessor(TARGET, self.numerical_cols, self.categorical_cols)
            preprocessor.fit(X_train_fold)
            X_train_fold = X_train_fold.drop(columns=[TARGET])

            # Fold 통계 저장
            self.fold_statistics.add_fold_stats(preprocessor.get_stats())

            # ===== CatBoost =====
            self.logger.info("\n[CatBoost]")
            X_train_cat, cat_features = preprocessor.transform_for_catboost(
                X_train_fold.assign(**{TARGET: y_train_fold})
            )
            X_val_cat, _ = preprocessor.transform_for_catboost(
                X_val_fold.assign(**{TARGET: y_val_fold})
            )
            X_train_cat = X_train_cat.drop(columns=[TARGET])
            X_val_cat = X_val_cat.drop(columns=[TARGET])

            catboost_model = CatBoostModel(CATBOOST_PARAMS.copy(), f"CatBoost_fold{fold}")
            catboost_model.fit(X_train_cat, y_train_fold, X_val_cat, y_val_fold, cat_features)
            oof_catboost[val_idx] = catboost_model.predict_proba(X_val_cat)
            self.fold_models["catboost"].append(catboost_model)

            auc = roc_auc_score(y_val_fold, oof_catboost[val_idx])
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== LightGBM =====
            self.logger.info("\n[LightGBM]")
            X_train_lgb = preprocessor.transform_for_lgbm_xgb(X_train_fold)
            X_val_lgb = preprocessor.transform_for_lgbm_xgb(X_val_fold)

            lightgbm_model = LightGBMModel(LIGHTGBM_PARAMS.copy(), f"LightGBM_fold{fold}")
            lightgbm_model.fit(X_train_lgb, y_train_fold, X_val_lgb, y_val_fold)
            oof_lightgbm[val_idx] = lightgbm_model.predict_proba(X_val_lgb)
            self.fold_models["lightgbm"].append(lightgbm_model)

            auc = roc_auc_score(y_val_fold, oof_lightgbm[val_idx])
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== XGBoost =====
            self.logger.info("\n[XGBoost]")
            xgboost_model = XGBoostModel(XGBOOST_PARAMS.copy(), f"XGBoost_fold{fold}")
            xgboost_model.fit(X_train_lgb, y_train_fold, X_val_lgb, y_val_fold)
            oof_xgboost[val_idx] = xgboost_model.predict_proba(X_val_lgb)
            self.fold_models["xgboost"].append(xgboost_model)

            auc = roc_auc_score(y_val_fold, oof_xgboost[val_idx])
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== Neural Network =====
            self.logger.info("\n[NeuralNet]")
            X_train_nn = preprocessor.transform_for_nn(X_train_fold, is_train=True)
            X_val_nn = preprocessor.transform_for_nn(X_val_fold, is_train=False)

            nn_model = NeuralNetModel(NN_PARAMS.copy(), f"NeuralNet_fold{fold}")
            nn_model.fit(X_train_nn, y_train_fold, X_val_nn, y_val_fold)
            oof_neural_net[val_idx] = nn_model.predict_proba(X_val_nn)
            self.fold_models["neural_net"].append(nn_model)

            auc = roc_auc_score(y_val_fold, oof_neural_net[val_idx])
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== Logistic Regression =====
            self.logger.info("\n[LogisticRegression]")
            logistic_model = LogisticModel(LOGISTIC_PARAMS.copy(), f"Logistic_fold{fold}")
            logistic_model.fit(X_train_nn, y_train_fold)
            oof_logistic[val_idx] = logistic_model.predict_proba(X_val_nn)
            self.fold_models["logistic"].append(logistic_model)

            auc = roc_auc_score(y_val_fold, oof_logistic[val_idx])
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # fold_preprocessors 저장 (NN 학습 후 - onehot_columns 포함)
            self.fold_preprocessors.append(preprocessor.get_stats().copy())

            # 메모리 정리
            del X_train_fold, X_val_fold, X_train_cat, X_val_cat
            del X_train_lgb, X_val_lgb, X_train_nn, X_val_nn
            gc.collect()

        # OOF 저장
        self.oof_predictions = {
            "catboost": oof_catboost,
            "lightgbm": oof_lightgbm,
            "xgboost": oof_xgboost,
            "neural_net": oof_neural_net,
            "logistic": oof_logistic,
        }

        # CV 결과 출력
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CV Results (전체)")
        self.logger.info("=" * 60)
        for name, oof in self.oof_predictions.items():
            auc = roc_auc_score(y, oof)
            self.logger.info(f"  {name}: {auc:.6f}")

        return self.oof_predictions

    def train_meta_model(self, y: np.ndarray) -> None:
        """L2 Meta Model 학습"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("L2 Meta Model 학습")
        self.logger.info("=" * 60)

        # OOF 예측을 피처로 사용
        meta_features = np.column_stack([
            self.oof_predictions["catboost"],
            self.oof_predictions["lightgbm"],
            self.oof_predictions["xgboost"],
            self.oof_predictions["neural_net"],
            self.oof_predictions["logistic"],
        ])

        self.meta_model = LogisticRegression(**META_MODEL_PARAMS)
        self.meta_model.fit(meta_features, y)

        # Meta Model 예측
        meta_pred = self.meta_model.predict_proba(meta_features)[:, 1]
        meta_auc = roc_auc_score(y, meta_pred)

        self.logger.info(f"  Meta Model CV AUC: {meta_auc:.6f}")
        self.logger.info(f"  가중치: {self.meta_model.coef_[0]}")

    def predict_test(self, test: pd.DataFrame) -> np.ndarray:
        """test.csv 예측"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Test 예측")
        self.logger.info("=" * 60)

        # ID 분리
        test_ids = test[ID_COL].values
        test = test.drop(columns=[ID_COL])

        # 8개 fold 통계의 중앙값으로 전처리
        median_stats = self.fold_statistics.get_median_stats()
        preprocessor = Preprocessor(TARGET, self.numerical_cols, self.categorical_cols)
        preprocessor.fit_from_stats(median_stats)

        # 각 모델별 예측 (8개 fold 평균)
        test_preds = {}

        # CatBoost
        self.logger.info("\n[CatBoost 예측]")
        test_cat, cat_features = preprocessor.transform_for_catboost(test)
        preds = []
        for model in self.fold_models["catboost"]:
            model.cat_features = cat_features  # 업데이트
            preds.append(model.predict_proba(test_cat))
        test_preds["catboost"] = np.mean(preds, axis=0)
        self.logger.info(f"  Mean: {test_preds['catboost'].mean():.4f}")

        # LightGBM
        self.logger.info("\n[LightGBM 예측]")
        test_lgb = preprocessor.transform_for_lgbm_xgb(test)
        preds = []
        for model in self.fold_models["lightgbm"]:
            preds.append(model.predict_proba(test_lgb))
        test_preds["lightgbm"] = np.mean(preds, axis=0)
        self.logger.info(f"  Mean: {test_preds['lightgbm'].mean():.4f}")

        # XGBoost
        self.logger.info("\n[XGBoost 예측]")
        preds = []
        for model in self.fold_models["xgboost"]:
            preds.append(model.predict_proba(test_lgb))
        test_preds["xgboost"] = np.mean(preds, axis=0)
        self.logger.info(f"  Mean: {test_preds['xgboost'].mean():.4f}")

        # Neural Network & Logistic - 각 fold의 preprocessor 사용
        self.logger.info("\n[NeuralNet & Logistic 예측]")
        nn_preds = []
        lr_preds = []
        for i, (nn_model, lr_model) in enumerate(zip(
            self.fold_models["neural_net"], self.fold_models["logistic"]
        )):
            fold_prep = Preprocessor(TARGET, self.numerical_cols, self.categorical_cols)
            fold_prep.fit_from_stats(self.fold_preprocessors[i])
            test_nn_fold = fold_prep.transform_for_nn(test.copy(), is_train=False)
            nn_preds.append(nn_model.predict_proba(test_nn_fold))
            lr_preds.append(lr_model.predict_proba(test_nn_fold))
        test_preds["neural_net"] = np.mean(nn_preds, axis=0)
        test_preds["logistic"] = np.mean(lr_preds, axis=0)
        self.logger.info(f"  NeuralNet Mean: {test_preds['neural_net'].mean():.4f}")
        self.logger.info(f"  Logistic Mean: {test_preds['logistic'].mean():.4f}")

        # L2 Meta Model로 최종 앙상블
        self.logger.info("\n[L2 Meta Model 앙상블]")
        meta_features = np.column_stack([
            test_preds["catboost"],
            test_preds["lightgbm"],
            test_preds["xgboost"],
            test_preds["neural_net"],
            test_preds["logistic"],
        ])
        final_pred = self.meta_model.predict_proba(meta_features)[:, 1]

        self.logger.info(f"  Final Mean: {final_pred.mean():.4f}")
        self.logger.info(f"  Final Std: {final_pred.std():.4f}")

        return test_ids, final_pred

    def save_submission(self, test_ids: np.ndarray, predictions: np.ndarray, filename: str = "submission.csv"):
        """제출 파일 저장"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        submission = pd.DataFrame({
            "ID": test_ids,
            "probability": predictions,
        })

        output_path = os.path.join(OUTPUT_DIR, filename)
        submission.to_csv(output_path, index=False)

        self.logger.info(f"\n제출 파일 저장: {output_path}")
        self.logger.info(f"  Shape: {submission.shape}")

    def save_artifacts(self):
        """모델 및 통계 저장"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Fold 통계 저장
        stats_path = os.path.join(OUTPUT_DIR, "fold_statistics.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(self.fold_statistics, f)
        self.logger.info(f"Fold 통계 저장: {stats_path}")

        # OOF 예측 저장
        oof_path = os.path.join(OUTPUT_DIR, "oof_predictions.pkl")
        with open(oof_path, "wb") as f:
            pickle.dump(self.oof_predictions, f)
        self.logger.info(f"OOF 예측 저장: {oof_path}")

        # Meta Model 저장
        meta_path = os.path.join(OUTPUT_DIR, "meta_model.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta_model, f)
        self.logger.info(f"Meta Model 저장: {meta_path}")

    def run(self):
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        self.logger.info(f"파이프라인 시작: {start_time}")

        # 1. 데이터 로드
        train, test = self.load_data()

        # 2. CV 실행
        y = train[TARGET].values
        self.run_cv(train)

        # 3. Meta Model 학습
        self.train_meta_model(y)

        # 4. Test 예측
        test_ids, predictions = self.predict_test(test)

        # 5. 제출 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_submission(test_ids, predictions, f"submission_{timestamp}.csv")

        # 6. 아티팩트 저장
        self.save_artifacts()

        # 완료
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"파이프라인 완료!")
        self.logger.info(f"소요 시간: {duration}")
        self.logger.info(f"{'='*60}")


def main():
    set_seed(RANDOM_STATE)
    pipeline = Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
