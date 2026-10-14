# -*- coding: utf-8 -*-
"""
LightGBM 모델

특징:
- 빠른 학습 속도
- 메모리 효율적
- Label Encoding 필요
"""

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM 분류 모델"""

    def __init__(self, params: Dict[str, Any], name: str = "LightGBM"):
        super().__init__(params, name)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[list] = None,
    ) -> "LightGBMModel":
        """
        LightGBM 학습

        Note: X_train은 이미 Label Encoding이 적용되어 있어야 함
        """
        params = self.params.copy()

        # early_stopping_rounds 분리
        early_stopping = params.pop("early_stopping_rounds", None)
        n_estimators = params.pop("n_estimators", 1000)

        # 모델 생성
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params
        )

        # 콜백 설정
        callbacks = []
        if early_stopping and X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))
            callbacks.append(lgb.log_evaluation(period=0))  # 0 = 로그 출력 안함

        # 학습
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
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

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "params": self.params,
                "name": self.name,
            }, f)

    def load(self, path: str) -> "LightGBMModel":
        """모델 로드"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.params = data["params"]
        self.name = data["name"]
        self.is_fitted = True
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """피처 중요도 반환"""
        if not self.is_fitted:
            return None

        importance = self.model.feature_importances_
        feature_names = self.model.feature_name_

        return dict(zip(feature_names, importance))
