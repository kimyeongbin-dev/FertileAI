# -*- coding: utf-8 -*-
"""
XGBoost 모델

특징:
- 강력한 정규화
- GPU 지원
- Label Encoding 필요
"""

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost 분류 모델"""

    def __init__(self, params: Dict[str, Any], name: str = "XGBoost"):
        super().__init__(params, name)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[list] = None,
    ) -> "XGBoostModel":
        """
        XGBoost 학습

        Note: X_train은 이미 Label Encoding이 적용되어 있어야 함
        """
        params = self.params.copy()

        # early_stopping_rounds 분리
        early_stopping = params.pop("early_stopping_rounds", None)
        n_estimators = params.pop("n_estimators", 1000)

        # 모델 생성
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            **params
        )

        # 학습
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=100,
        )

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """양성 클래스 확률 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def save(self, path: str) -> None:
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # XGBoost 네이티브 저장
        self.model.save_model(path)

        # 메타데이터 저장
        meta = {
            "params": self.params,
            "name": self.name,
        }
        with open(path + ".meta", "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: str) -> "XGBoostModel":
        """모델 로드"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)

        # 메타데이터 로드
        with open(path + ".meta", "rb") as f:
            meta = pickle.load(f)
        self.params = meta["params"]
        self.name = meta["name"]

        self.is_fitted = True
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """피처 중요도 반환"""
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names

        if feature_names is None:
            return None

        return dict(zip(feature_names, importance))
