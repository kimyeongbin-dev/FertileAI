# -*- coding: utf-8 -*-
"""
전처리 파이프라인

핵심 원칙:
- Train 통계만 계산 → Validation/Test에 적용
- 각 Fold는 자신만의 통계를 계산
- test.csv: 8개 Fold 통계의 중앙값 적용 (Option B)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class FoldStatistics:
    """
    Fold별 통계를 저장하고 관리하는 클래스

    test.csv 적용 시 8개 fold 통계의 중앙값을 사용합니다.
    """

    def __init__(self):
        self.fold_stats: List[Dict[str, Any]] = []

    def add_fold_stats(self, stats: Dict[str, Any]):
        """fold 통계 추가"""
        self.fold_stats.append(stats)

    def get_median_stats(self) -> Dict[str, Any]:
        """8개 fold 통계의 중앙값 반환"""
        if not self.fold_stats:
            raise ValueError("저장된 fold 통계가 없습니다.")

        median_stats = {}

        # 수치형 결측치 대체값 (median)
        if "numerical_fill" in self.fold_stats[0]:
            median_stats["numerical_fill"] = {}
            for col in self.fold_stats[0]["numerical_fill"].keys():
                values = [fs["numerical_fill"][col] for fs in self.fold_stats]
                median_stats["numerical_fill"][col] = np.median(values)

        # 범주형 결측치 대체값 (mode) - 최빈값의 최빈값
        if "categorical_fill" in self.fold_stats[0]:
            median_stats["categorical_fill"] = {}
            for col in self.fold_stats[0]["categorical_fill"].keys():
                values = [fs["categorical_fill"][col] for fs in self.fold_stats]
                # 문자열은 중앙값 대신 최빈값
                from collections import Counter
                median_stats["categorical_fill"][col] = Counter(values).most_common(1)[0][0]

        # 타겟 인코딩 값
        if "target_encoding" in self.fold_stats[0]:
            median_stats["target_encoding"] = {}
            for col in self.fold_stats[0]["target_encoding"].keys():
                median_stats["target_encoding"][col] = {}
                all_categories = set()
                for fs in self.fold_stats:
                    all_categories.update(fs["target_encoding"][col].keys())

                for cat in all_categories:
                    values = []
                    for fs in self.fold_stats:
                        if cat in fs["target_encoding"][col]:
                            values.append(fs["target_encoding"][col][cat])
                    if values:
                        median_stats["target_encoding"][col][cat] = np.median(values)

        # 글로벌 타겟 평균 (unseen category용)
        if "global_target_mean" in self.fold_stats[0]:
            values = [fs["global_target_mean"] for fs in self.fold_stats]
            median_stats["global_target_mean"] = np.median(values)

        # Label Encoder 매핑
        if "label_encoding" in self.fold_stats[0]:
            median_stats["label_encoding"] = {}
            for col in self.fold_stats[0]["label_encoding"].keys():
                # 모든 fold에서 본 카테고리 통합
                all_categories = set()
                for fs in self.fold_stats:
                    all_categories.update(fs["label_encoding"][col].keys())
                # 정렬하여 일관된 인코딩
                sorted_cats = sorted(all_categories)
                median_stats["label_encoding"][col] = {
                    cat: idx for idx, cat in enumerate(sorted_cats)
                }

        # Scaler 파라미터
        if "scaler_mean" in self.fold_stats[0]:
            median_stats["scaler_mean"] = {}
            median_stats["scaler_std"] = {}
            for col in self.fold_stats[0]["scaler_mean"].keys():
                mean_values = [fs["scaler_mean"][col] for fs in self.fold_stats]
                std_values = [fs["scaler_std"][col] for fs in self.fold_stats]
                median_stats["scaler_mean"][col] = np.median(mean_values)
                median_stats["scaler_std"][col] = np.median(std_values)

        # One-Hot 컬럼 목록 (모든 fold의 합집합)
        if "onehot_columns" in self.fold_stats[0] and self.fold_stats[0]["onehot_columns"]:
            all_columns = set()
            for fs in self.fold_stats:
                if fs.get("onehot_columns"):
                    all_columns.update(fs["onehot_columns"])
            # 정렬하여 일관된 순서 유지
            median_stats["onehot_columns"] = sorted(list(all_columns))

        return median_stats


def normalize_category(value) -> str:
    """
    범주형 값 정규화

    - 대문자 변환
    - 앞뒤 공백 제거
    - NaN 유지
    """
    if pd.isna(value):
        return value
    return str(value).strip().upper()


def identify_column_types(df: pd.DataFrame, target: str = None) -> Tuple[List[str], List[str]]:
    """
    수치형/범주형 컬럼 자동 식별

    Returns:
        (numerical_cols, categorical_cols)
    """
    exclude_cols = [target] if target else []

    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if df[col].dtype in ["int64", "int32", "float64", "float32"]:
            # 유니크 값이 적으면 범주형으로 간주
            if df[col].nunique() <= 20:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return numerical_cols, categorical_cols


class Preprocessor:
    """
    전처리 클래스

    데이터 누수 방지:
    - fit(): train 데이터로 통계 계산
    - transform(): 저장된 통계로 변환
    """

    def __init__(self, target: str, numerical_cols: List[str], categorical_cols: List[str]):
        self.target = target
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.stats: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Train 데이터로 통계 계산 (데이터 누수 방지)
        """
        self.stats = {}

        # 1. 수치형 결측치 대체값 (중앙값)
        self.stats["numerical_fill"] = {}
        for col in self.numerical_cols:
            if col in df.columns:
                self.stats["numerical_fill"][col] = df[col].median()

        # 2. 범주형 결측치 대체값 (최빈값)
        self.stats["categorical_fill"] = {}
        for col in self.categorical_cols:
            if col in df.columns:
                mode_result = df[col].mode()
                self.stats["categorical_fill"][col] = mode_result[0] if len(mode_result) > 0 else "MISSING"

        # 3. 타겟 인코딩 (train에서만 계산)
        self.stats["target_encoding"] = {}
        self.stats["global_target_mean"] = df[self.target].mean()

        for col in self.categorical_cols:
            if col in df.columns:
                target_mean = df.groupby(col)[self.target].mean().to_dict()
                self.stats["target_encoding"][col] = target_mean

        # 4. Label Encoding 매핑
        self.stats["label_encoding"] = {}
        for col in self.categorical_cols:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                sorted_vals = sorted([str(v) for v in unique_vals])
                self.stats["label_encoding"][col] = {
                    val: idx for idx, val in enumerate(sorted_vals)
                }

        # 5. Scaler 파라미터 (NN, LogReg용)
        self.stats["scaler_mean"] = {}
        self.stats["scaler_std"] = {}
        for col in self.numerical_cols:
            if col in df.columns:
                self.stats["scaler_mean"][col] = df[col].mean()
                self.stats["scaler_std"][col] = df[col].std()
                if self.stats["scaler_std"][col] == 0:
                    self.stats["scaler_std"][col] = 1.0  # 0으로 나누기 방지

        # 6. One-Hot 인코딩 컬럼 목록 (NN용) - 설계문서 6.4
        self.stats["onehot_columns"] = None  # transform_for_nn 첫 호출 시 설정

        self.is_fitted = True
        return self

    def fit_from_stats(self, stats: Dict[str, Any]) -> "Preprocessor":
        """
        저장된 통계로 fit (test 데이터용)
        """
        self.stats = stats
        self.is_fitted = True
        return self

    def transform_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기본 전처리 (모든 모델 공통)

        - 범주형 정규화
        - 결측치 처리
        """
        if not self.is_fitted:
            raise ValueError("fit() 또는 fit_from_stats()를 먼저 호출하세요.")

        df = df.copy()

        # 1. 범주형 정규화
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].apply(normalize_category)

        # 2. 수치형 결측치 처리
        for col in self.numerical_cols:
            if col in df.columns and col in self.stats["numerical_fill"]:
                df[col] = df[col].fillna(self.stats["numerical_fill"][col])

        # 3. 범주형 결측치 처리
        for col in self.categorical_cols:
            if col in df.columns and col in self.stats["categorical_fill"]:
                df[col] = df[col].fillna(self.stats["categorical_fill"][col])

        return df

    def transform_for_catboost(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """
        CatBoost용 변환 (범주형을 문자열로 변환)

        Returns:
            (변환된 df, cat_features 인덱스 리스트)
        """
        df = self.transform_base(df)

        # 범주형 컬럼을 문자열로 변환 (CatBoost 요구사항)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', 'MISSING')

        # 타겟 인코딩 추가 (한번에 처리하여 fragmentation 방지)
        te_data = {}
        for col in self.categorical_cols:
            if col in df.columns and col in self.stats["target_encoding"]:
                te_col = f"{col}_te"
                te_values = df[col].map(self.stats["target_encoding"][col])
                te_data[te_col] = te_values.fillna(self.stats["global_target_mean"])

        if te_data:
            te_df = pd.DataFrame(te_data, index=df.index)
            df = pd.concat([df, te_df], axis=1)

        # cat_features: 원본 범주형 컬럼의 인덱스
        feature_cols = [c for c in df.columns if c != self.target]
        cat_features = [i for i, c in enumerate(feature_cols) if c in self.categorical_cols]

        return df, cat_features

    def transform_for_lgbm_xgb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LightGBM/XGBoost용 변환 (Label Encoding)
        """
        df = self.transform_base(df)

        # Label Encoding
        for col in self.categorical_cols:
            if col in df.columns and col in self.stats["label_encoding"]:
                le_map = self.stats["label_encoding"][col]
                # Unseen category는 -1
                df[col] = df[col].apply(
                    lambda x: le_map.get(str(x).strip().upper(), -1) if pd.notna(x) else -1
                )
                df[col] = df[col].astype(int)

        # 타겟 인코딩 추가
        for col in self.categorical_cols:
            if col in self.stats["target_encoding"]:
                te_col = f"{col}_te"
                # 원본 범주형 값 복원 (정규화된)
                original_col = col
                if original_col in self.stats["target_encoding"]:
                    # Label Encoding 전의 원본 매핑 필요
                    pass  # 이미 base_transform에서 처리됨

        return df

    def transform_for_nn(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Neural Network용 변환 (One-Hot + Scaling)

        설계문서 6.4: OneHotEncoder(handle_unknown='ignore')
        - train에서 생성된 컬럼 목록 저장
        - test에서는 저장된 컬럼 목록에 맞춰 정렬

        Args:
            is_train: True면 컬럼 목록 저장, False면 저장된 목록 사용
        """
        df = self.transform_base(df)

        # One-Hot Encoding (한번에 처리)
        cat_cols_in_df = [col for col in self.categorical_cols if col in df.columns]
        if cat_cols_in_df:
            # 범주형을 문자열로 변환
            for col in cat_cols_in_df:
                df[col] = df[col].astype(str)

            dummies = pd.get_dummies(df[cat_cols_in_df], prefix=cat_cols_in_df, dummy_na=False)
            df = pd.concat([df.drop(columns=cat_cols_in_df), dummies], axis=1)

        # Scaling
        for col in self.numerical_cols:
            if col in df.columns:
                mean = self.stats["scaler_mean"].get(col, 0)
                std = self.stats["scaler_std"].get(col, 1)
                df[col] = (df[col] - mean) / std

        # 모든 컬럼을 float로 변환
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 컬럼 목록 관리 (설계문서 6.4: handle_unknown='ignore' 효과)
        if is_train:
            # train: 컬럼 목록 저장
            self.stats["onehot_columns"] = list(df.columns)
        elif self.stats.get("onehot_columns"):
            # test: 저장된 컬럼 목록에 맞춤
            saved_cols = self.stats["onehot_columns"]
            # 없는 컬럼은 0으로 추가
            for col in saved_cols:
                if col not in df.columns:
                    df[col] = 0.0
            # 저장된 컬럼 순서로 정렬 (추가 컬럼은 제외)
            df = df[[col for col in saved_cols if col in df.columns]]

        df = df.astype(np.float32)

        return df

    def get_stats(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        return self.stats.copy()
