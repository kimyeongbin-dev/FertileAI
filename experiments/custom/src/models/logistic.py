# -*- coding: utf-8 -*-
"""
Logistic Regression 모델

특징:
- 단순하고 해석 가능
- 기준선 모델로 유용
- One-Hot Encoding + Scaling 필요
"""

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class LogisticModel(BaseModel):
    """Logistic Regression 분류 모델"""

    def __init__(self, params: Dict[str, Any], name: str = "LogisticRegression"):
        super().__init__(params, name)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[list] = None,
    ) -> "LogisticModel":
        """
        Logistic Regression 학습

        Note: X_train은 이미 One-Hot + Scaling이 적용되어 있어야 함
        """
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)

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

    def load(self, path: str) -> "LogisticModel":
        """모델 로드"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.params = data["params"]
        self.name = data["name"]
        self.is_fitted = True
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """피처 중요도 (계수) 반환"""
        if not self.is_fitted:
            return None

        # 계수의 절대값을 중요도로 사용
        coef = np.abs(self.model.coef_[0])
        return {"coef": coef.tolist()}
