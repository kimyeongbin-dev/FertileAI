# -*- coding: utf-8 -*-
"""
CatBoost 모델

특징:
- 범주형 변수 네이티브 지원 (Label Encoding 불필요)
- GPU 지원
- 과적합에 강건
"""

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost 분류 모델"""

    def __init__(self, params: Dict[str, Any], name: str = "CatBoost"):
        super().__init__(params, name)
        self.cat_features: Optional[List[int]] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[List[int]] = None,
    ) -> "CatBoostModel":
        """
        CatBoost 학습

        Args:
            cat_features: 범주형 컬럼 인덱스 리스트
        """
        self.cat_features = cat_features

        # CatBoost Pool 생성
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features,
        )

        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=cat_features,
            )

        # 모델 생성 및 학습
        self.model = CatBoostClassifier(**self.params)

        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True if eval_pool else False,
        )

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """양성 클래스 확률 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        pool = Pool(data=X, cat_features=self.cat_features)
        proba = self.model.predict_proba(pool)

        # 양성 클래스(1) 확률만 반환
        return proba[:, 1]

    def save(self, path: str) -> None:
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # CatBoost 네이티브 저장
        self.model.save_model(path)

        # 메타데이터 저장
        meta = {
            "params": self.params,
            "name": self.name,
            "cat_features": self.cat_features,
        }
        with open(path + ".meta", "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: str) -> "CatBoostModel":
        """모델 로드"""
        self.model = CatBoostClassifier()
        self.model.load_model(path)

        # 메타데이터 로드
        with open(path + ".meta", "rb") as f:
            meta = pickle.load(f)
        self.params = meta["params"]
        self.name = meta["name"]
        self.cat_features = meta["cat_features"]

        self.is_fitted = True
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """피처 중요도 반환"""
        if not self.is_fitted:
            return None

        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_

        return dict(zip(feature_names, importance))
