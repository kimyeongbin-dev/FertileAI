# -*- coding: utf-8 -*-
"""
메인 파이프라인 (Bagging 적용)

8-Fold StratifiedKFold + L2 Stacking + Bagging

Bagging: 각 모델을 여러 seed로 학습하여 분산 감소
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


class PipelineWithBagging:
    """메인 학습 파이프라인 (Bagging 적용)"""

    def __init__(self, logger=None):
        self.logger = logger or setup_logging(OUTPUT_DIR)
        self.fold_statistics = FoldStatistics()
        self.fold_preprocessors: List[Dict[str, Any]] = []
        # Bagging: fold_models[model_name][fold_idx] = [seed별 모델 리스트]
        self.fold_models: Dict[str, List[List[Any]]] = {
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
        self.logger.info(f"Bagging: {USE_BAGGING}, Seeds: {N_BAG_SEEDS}")

        train = reduce_memory_usage(train, verbose=False)
        test = reduce_memory_usage(test, verbose=False)

        return train, test

    def _train_with_bagging(
        self,
        model_class,
        base_params: dict,
        model_name: str,
        X_train, y_train,
        X_val, y_val,
        cat_features=None
    ) -> Tuple[List[Any], np.ndarray]:
        """
        Bagging으로 모델 학습

        Returns:
            (모델 리스트, OOF 예측 평균)
        """
        models = []
        oof_preds = []

        seeds = [RANDOM_STATE + i * 100 for i in range(N_BAG_SEEDS)]

        for seed_idx, seed in enumerate(seeds):
            params = base_params.copy()

            # seed 설정 (모델별로 다른 파라미터명)
            if "random_seed" in params:  # CatBoost
                params["random_seed"] = seed
            elif "seed" in params:  # LightGBM, XGBoost
                params["seed"] = seed
            elif "random_state" in params:  # Logistic
                params["random_state"] = seed

            model = model_class(params, f"{model_name}_seed{seed_idx}")

            if cat_features is not None:
                model.fit(X_train, y_train, X_val, y_val, cat_features)
            else:
                model.fit(X_train, y_train, X_val, y_val)

            pred = model.predict_proba(X_val)
            oof_preds.append(pred)
            models.append(model)

        # OOF 평균
        oof_mean = np.mean(oof_preds, axis=0)

        return models, oof_mean

    def run_cv(self, train: pd.DataFrame) -> Dict[str, np.ndarray]:
        """8-Fold CV 실행 (Bagging 적용)"""
        self.logger.info("=" * 60)
        self.logger.info(f"{N_FOLDS}-Fold Cross Validation 시작 (Bagging: {N_BAG_SEEDS} seeds)")
        self.logger.info("=" * 60)

        train_ids = train[ID_COL].values
        train = train.drop(columns=[ID_COL])

        y = train[TARGET].values
        X = train.drop(columns=[TARGET])

        self.numerical_cols, self.categorical_cols = identify_column_types(X)
        self.logger.info(f"수치형 컬럼: {len(self.numerical_cols)}개")
        self.logger.info(f"범주형 컬럼: {len(self.categorical_cols)}개")

        n_samples = len(y)
        oof_catboost = np.zeros(n_samples)
        oof_lightgbm = np.zeros(n_samples)
        oof_xgboost = np.zeros(n_samples)
        oof_neural_net = np.zeros(n_samples)
        oof_logistic = np.zeros(n_samples)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Fold {fold + 1}/{N_FOLDS}")
            self.logger.info(f"{'='*60}")

            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # 전처리
            X_train_fold[TARGET] = y_train_fold
            preprocessor = Preprocessor(TARGET, self.numerical_cols, self.categorical_cols)
            preprocessor.fit(X_train_fold)
            X_train_fold = X_train_fold.drop(columns=[TARGET])

            self.fold_statistics.add_fold_stats(preprocessor.get_stats())

            # ===== CatBoost (Bagging) =====
            self.logger.info(f"\n[CatBoost] Bagging x{N_BAG_SEEDS}")
            X_train_cat, cat_features = preprocessor.transform_for_catboost(
                X_train_fold.assign(**{TARGET: y_train_fold})
            )
            X_val_cat, _ = preprocessor.transform_for_catboost(
                X_val_fold.assign(**{TARGET: y_val_fold})
            )
            X_train_cat = X_train_cat.drop(columns=[TARGET])
            X_val_cat = X_val_cat.drop(columns=[TARGET])

            models, oof_pred = self._train_with_bagging(
                CatBoostModel, CATBOOST_PARAMS.copy(), f"CatBoost_fold{fold}",
                X_train_cat, y_train_fold, X_val_cat, y_val_fold, cat_features
            )
            oof_catboost[val_idx] = oof_pred
            self.fold_models["catboost"].append(models)

            auc = roc_auc_score(y_val_fold, oof_pred)
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== LightGBM (Bagging) =====
            self.logger.info(f"\n[LightGBM] Bagging x{N_BAG_SEEDS}")
            X_train_lgb = preprocessor.transform_for_lgbm_xgb(X_train_fold)
            X_val_lgb = preprocessor.transform_for_lgbm_xgb(X_val_fold)

            models, oof_pred = self._train_with_bagging(
                LightGBMModel, LIGHTGBM_PARAMS.copy(), f"LightGBM_fold{fold}",
                X_train_lgb, y_train_fold, X_val_lgb, y_val_fold
            )
            oof_lightgbm[val_idx] = oof_pred
            self.fold_models["lightgbm"].append(models)

            auc = roc_auc_score(y_val_fold, oof_pred)
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== XGBoost (Bagging) =====
            self.logger.info(f"\n[XGBoost] Bagging x{N_BAG_SEEDS}")

            models, oof_pred = self._train_with_bagging(
                XGBoostModel, XGBOOST_PARAMS.copy(), f"XGBoost_fold{fold}",
                X_train_lgb, y_train_fold, X_val_lgb, y_val_fold
            )
            oof_xgboost[val_idx] = oof_pred
            self.fold_models["xgboost"].append(models)

            auc = roc_auc_score(y_val_fold, oof_pred)
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== Neural Network (Bagging) =====
            self.logger.info(f"\n[NeuralNet] Bagging x{N_BAG_SEEDS}")
            X_train_nn = preprocessor.transform_for_nn(X_train_fold, is_train=True)
            X_val_nn = preprocessor.transform_for_nn(X_val_fold, is_train=False)

            # NN은 seed 설정 방식이 다름
            nn_models = []
            nn_preds = []
            seeds = [RANDOM_STATE + i * 100 for i in range(N_BAG_SEEDS)]
            for seed_idx, seed in enumerate(seeds):
                set_seed(seed)  # 전역 시드 설정
                params = NN_PARAMS.copy()
                model = NeuralNetModel(params, f"NeuralNet_fold{fold}_seed{seed_idx}")
                model.fit(X_train_nn, y_train_fold, X_val_nn, y_val_fold)
                nn_preds.append(model.predict_proba(X_val_nn))
                nn_models.append(model)

            oof_neural_net[val_idx] = np.mean(nn_preds, axis=0)
            self.fold_models["neural_net"].append(nn_models)

            auc = roc_auc_score(y_val_fold, oof_neural_net[val_idx])
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # ===== Logistic Regression (Bagging) =====
            self.logger.info(f"\n[LogisticRegression] Bagging x{N_BAG_SEEDS}")

            models, oof_pred = self._train_with_bagging(
                LogisticModel, LOGISTIC_PARAMS.copy(), f"Logistic_fold{fold}",
                X_train_nn, y_train_fold, X_val_nn, y_val_fold  # OOF 예측용 X_val 필요
            )
            oof_logistic[val_idx] = oof_pred
            self.fold_models["logistic"].append(models)

            auc = roc_auc_score(y_val_fold, oof_pred)
            self.logger.info(f"  Fold {fold+1} AUC: {auc:.6f}")

            # fold_preprocessors 저장
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

        meta_features = np.column_stack([
            self.oof_predictions["catboost"],
            self.oof_predictions["lightgbm"],
            self.oof_predictions["xgboost"],
            self.oof_predictions["neural_net"],
            self.oof_predictions["logistic"],
        ])

        self.meta_model = LogisticRegression(**META_MODEL_PARAMS)
        self.meta_model.fit(meta_features, y)

        meta_pred = self.meta_model.predict_proba(meta_features)[:, 1]
        meta_auc = roc_auc_score(y, meta_pred)

        self.logger.info(f"  Meta Model CV AUC: {meta_auc:.6f}")
        self.logger.info(f"  가중치: {self.meta_model.coef_[0]}")

    def predict_test(self, test: pd.DataFrame) -> np.ndarray:
        """test.csv 예측 (Bagging 모델 평균)"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Test 예측")
        self.logger.info("=" * 60)

        test_ids = test[ID_COL].values
        test = test.drop(columns=[ID_COL])

        median_stats = self.fold_statistics.get_median_stats()
        preprocessor = Preprocessor(TARGET, self.numerical_cols, self.categorical_cols)
        preprocessor.fit_from_stats(median_stats)

        test_preds = {}

        # CatBoost - 모든 fold x 모든 seed 평균
        self.logger.info("\n[CatBoost 예측]")
        test_cat, cat_features = preprocessor.transform_for_catboost(test)
        preds = []
        for fold_models in self.fold_models["catboost"]:
            for model in fold_models:
                model.cat_features = cat_features
                preds.append(model.predict_proba(test_cat))
        test_preds["catboost"] = np.mean(preds, axis=0)
        self.logger.info(f"  Mean: {test_preds['catboost'].mean():.4f} ({len(preds)} models)")

        # LightGBM
        self.logger.info("\n[LightGBM 예측]")
        test_lgb = preprocessor.transform_for_lgbm_xgb(test)
        preds = []
        for fold_models in self.fold_models["lightgbm"]:
            for model in fold_models:
                preds.append(model.predict_proba(test_lgb))
        test_preds["lightgbm"] = np.mean(preds, axis=0)
        self.logger.info(f"  Mean: {test_preds['lightgbm'].mean():.4f} ({len(preds)} models)")

        # XGBoost
        self.logger.info("\n[XGBoost 예측]")
        preds = []
        for fold_models in self.fold_models["xgboost"]:
            for model in fold_models:
                preds.append(model.predict_proba(test_lgb))
        test_preds["xgboost"] = np.mean(preds, axis=0)
        self.logger.info(f"  Mean: {test_preds['xgboost'].mean():.4f} ({len(preds)} models)")

        # Neural Network & Logistic - 각 fold의 preprocessor 사용
        self.logger.info("\n[NeuralNet & Logistic 예측]")
        nn_preds = []
        lr_preds = []
        for fold_idx in range(N_FOLDS):
            fold_prep = Preprocessor(TARGET, self.numerical_cols, self.categorical_cols)
            fold_prep.fit_from_stats(self.fold_preprocessors[fold_idx])
            test_nn_fold = fold_prep.transform_for_nn(test.copy(), is_train=False)

            for model in self.fold_models["neural_net"][fold_idx]:
                nn_preds.append(model.predict_proba(test_nn_fold))
            for model in self.fold_models["logistic"][fold_idx]:
                lr_preds.append(model.predict_proba(test_nn_fold))

        test_preds["neural_net"] = np.mean(nn_preds, axis=0)
        test_preds["logistic"] = np.mean(lr_preds, axis=0)
        self.logger.info(f"  NeuralNet Mean: {test_preds['neural_net'].mean():.4f} ({len(nn_preds)} models)")
        self.logger.info(f"  Logistic Mean: {test_preds['logistic'].mean():.4f} ({len(lr_preds)} models)")

        # L2 Meta Model
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

        stats_path = os.path.join(OUTPUT_DIR, "fold_statistics_bagging.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(self.fold_statistics, f)
        self.logger.info(f"Fold 통계 저장: {stats_path}")

        oof_path = os.path.join(OUTPUT_DIR, "oof_predictions_bagging.pkl")
        with open(oof_path, "wb") as f:
            pickle.dump(self.oof_predictions, f)
        self.logger.info(f"OOF 예측 저장: {oof_path}")

        meta_path = os.path.join(OUTPUT_DIR, "meta_model_bagging.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta_model, f)
        self.logger.info(f"Meta Model 저장: {meta_path}")

    def run(self):
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        self.logger.info(f"파이프라인 시작 (Bagging): {start_time}")

        train, test = self.load_data()

        y = train[TARGET].values
        self.run_cv(train)

        self.train_meta_model(y)

        test_ids, predictions = self.predict_test(test)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_submission(test_ids, predictions, f"submission_bagging_{timestamp}.csv")

        self.save_artifacts()

        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"파이프라인 완료! (Bagging x{N_BAG_SEEDS})")
        self.logger.info(f"소요 시간: {duration}")
        self.logger.info(f"총 모델 수: {N_FOLDS * N_BAG_SEEDS * 5}")
        self.logger.info(f"{'='*60}")


def main():
    set_seed(RANDOM_STATE)
    pipeline = PipelineWithBagging()
    pipeline.run()


if __name__ == "__main__":
    main()
