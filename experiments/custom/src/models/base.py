# -*- coding: utf-8 -*-
"""
Base Model 추상 클래스

모든 모델은 이 클래스를 상속받아 일관된 인터페이스를 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    모델 추상 클래스

    모든 모델은 fit, predict_proba, save, load 메서드를 구현해야 합니다.
    """

    def __init__(self, params: Dict[str, Any], name: str = "BaseModel"):
        self.params = params
        self.name = name
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[list] = None,
    ) -> "BaseModel":
        """
        모델 학습

        Args:
            X_train: 학습 피처
            y_train: 학습 타겟
            X_val: 검증 피처 (early stopping용)
            y_val: 검증 타겟
            cat_features: 범주형 컬럼 인덱스 (CatBoost용)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        확률 예측

        Args:
            X: 예측할 피처

        Returns:
            양성 클래스(1) 확률 배열
        """
        pass

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        클래스 예측

        Args:
            X: 예측할 피처
            threshold: 분류 임계값

        Returns:
            예측 클래스 배열 (0 또는 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 저장"""
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        """모델 로드"""
        pass

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """피처 중요도 반환 (지원하는 모델만)"""
        return None

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"
