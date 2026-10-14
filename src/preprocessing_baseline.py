"""
베이스라인 전처리 파이프라인 — 최소한의 전처리만 적용

목적:
    - 피처 엔지니어링 없이 각 모델의 raw performance 확인
    - 타입 변환, 상수 컬럼 제거 등 필수 전처리만 수행

사용법:
    from preprocessing_baseline import preprocess_baseline

    X_train, y_train, X_test = preprocess_baseline(train, test)
"""

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────────
TARGET = '임신 성공 여부'

# 제거 컬럼 (ID + 상수 컬럼만)
DROP_COLS = [
    'ID',                    # 식별자
    '난자 채취 경과일',        # 전체 0.0 상수
    '불임 원인 - 여성 요인',   # 전체 0.0 상수
]

# 횟수 문자열 컬럼 ('0회', '6회 이상' → 정수)
HOE_COLS = [
    '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수',
    '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
    '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수',
]

# 순서형 인코딩 매핑
AGE_MAP = {
    '만18-34세': 0, '만35-37세': 1, '만38-39세': 2,
    '만40-42세': 3, '만43-44세': 4, '만45-50세': 5,
    '알 수 없음': np.nan,
}
DONOR_AGE_MAP = {
    '만20세 이하': 1, '만21-25세': 2, '만26-30세': 3, '만31-35세': 4,
    '만36-40세': 5, '만41-45세': 6,
    '알 수 없음': np.nan,
}

# 데이터 명세 기반 범주 정의 (Label Encoding용)
# 복합 조합 컬럼은 제외: 특정 시술 유형, 배아 생성 주요 이유 (복수 선택 가능)
CATEGORY_MAP = {
    '시술 시기 코드': ['TRCMWS', 'TRDQAZ', 'TRJXFG', 'TRVNRY', 'TRXQMD', 'TRYBLT', 'TRZKPL'],
    '시술 유형': ['DI', 'IVF'],
    '배란 유도 유형': ['기록되지 않은 시행', '생식선 자극 호르몬', '세트로타이드 (억제제)', '알 수 없음'],
    '난자 출처': ['기증 제공', '본인 제공', '알 수 없음'],
    '정자 출처': ['기증 제공', '미할당', '배우자 및 기증 제공', '배우자 제공'],
}

# 범주형 컬럼 (CatBoost용)
CATEGORICAL_COLS = [
    '시술 유형',
    '배란 유도 유형',
    '난자 출처',
    '정자 출처',
    '특정 시술 유형',
    '시술 시기 코드',
    '배아 생성 주요 이유',
    '착상 전 유전 검사 사용 여부',
    'PGD 시술 여부',
    'PGS 시술 여부',
]


# ────────────────────────────────────────────────────────────────
# 내부 함수
# ────────────────────────────────────────────────────────────────
def _transform_minimal(df: pd.DataFrame, for_catboost: bool = False) -> pd.DataFrame:
    """
    최소 전처리 (타입 변환만)

    Parameters
    ----------
    df          : 원본 데이터프레임
    for_catboost: True면 범주형을 문자열로 유지, False면 Label Encoding
    """
    df = df.copy()

    # 0. 컬럼명 정리 (OpenFE 호환성)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' - ', '_', regex=False)  # 하이픈 → 언더스코어

    # 1. 불필요 컬럼 제거 (ID, 상수만)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 2. 횟수 문자열 → 정수 ('6회 이상' → 6)
    for c in HOE_COLS:
        if c in df.columns:
            df[c] = df[c].str.extract(r'(\d+)').astype(float)  # NaN 허용 위해 float

    # 3. 나이 순서형 인코딩
    if '시술 당시 나이' in df.columns:
        df['시술 당시 나이'] = df['시술 당시 나이'].map(AGE_MAP)
    if '난자 기증자 나이' in df.columns:
        df['난자 기증자 나이'] = df['난자 기증자 나이'].map(DONOR_AGE_MAP)
    if '정자 기증자 나이' in df.columns:
        df['정자 기증자 나이'] = df['정자 기증자 나이'].map(DONOR_AGE_MAP)

    # 4. 범주형 처리 (CatBoost용만 여기서 처리)
    if for_catboost:
        # CatBoost: 범주형을 문자열로 유지 (NaN → 'missing')
        for c in CATEGORICAL_COLS:
            if c in df.columns:
                df[c] = df[c].fillna('missing').astype(str)
    # LGBM/XGBoost Label Encoding은 preprocess_baseline에서 train 기반으로 처리

    return df


# ────────────────────────────────────────────────────────────────
# 메인 함수
# ────────────────────────────────────────────────────────────────
def preprocess_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = TARGET,
    for_catboost: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    베이스라인 전처리 (최소한의 변환만)

    Parameters
    ----------
    train       : 학습 데이터 (target_col 포함)
    test        : 테스트 데이터
    target_col  : 타겟 컬럼명
    for_catboost: CatBoost용 범주형 처리 여부

    Returns
    -------
    X_train : 전처리된 학습 피처
    y_train : 학습 타겟
    X_test  : 전처리된 테스트 피처

    Notes
    -----
    - 피처 엔지니어링 없음
    - Target Encoding 없음
    - 결측치: 모델이 자체 처리 (LGBM, CatBoost, XGBoost 모두 지원)
    """
    y_train = train[target_col].copy()
    X_train = train.drop(columns=[target_col])

    X_train = _transform_minimal(X_train, for_catboost=for_catboost)
    X_test = _transform_minimal(test, for_catboost=for_catboost)

    # Label Encoding (LGBM/XGBoost용) - 명세 기반 + 복합조합 자동처리
    if not for_catboost:
        for c in X_train.select_dtypes(include='object').columns:
            if c in CATEGORY_MAP:
                # 명세에 정의된 범주 사용
                categories = CATEGORY_MAP[c]
            else:
                # 명세에 없는 컬럼 (특정 시술 유형 등 복합조합)
                # train 기반으로 카테고리 정의 (Data Leakage 방지)
                categories = X_train[c].dropna().unique().tolist()

            cat_type = pd.CategoricalDtype(categories=categories, ordered=False)

            # train/test 동일한 매핑 적용 (명세에 없는 값은 -1 → NaN)
            X_train[c] = X_train[c].astype(cat_type).cat.codes.replace(-1, np.nan)
            X_test[c] = X_test[c].astype(cat_type).cat.codes.replace(-1, np.nan)

    # 컬럼 순서 일치
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0 if not for_catboost else 'missing')

    return X_train, y_train, X_test


def get_categorical_features(df: pd.DataFrame) -> list:
    """CatBoost용 범주형 컬럼 인덱스 반환"""
    cat_cols = []
    for i, c in enumerate(df.columns):
        if c in CATEGORICAL_COLS or df[c].dtype == 'object':
            cat_cols.append(i)
    return cat_cols
