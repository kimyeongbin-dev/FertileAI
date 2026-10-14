"""
피처 선택 — Permutation Importance 기반 피처 선택

사용법:
    from feature_selection import select_features, select_by_importance

    # 방법 1: 미리 정의된 저중요도 피처 제거
    X_train, X_test = select_features(X_train, X_test)

    # 방법 2: 학습 후 중요도 기반 선택 (자동 FE용)
    X_train, X_test = select_by_importance(X_train, y_train, X_test, top_k=100)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

# LightGBM 경고 메시지 완전 비활성화
class _SilentLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
lgb.register_logger(_SilentLogger())


# AutoGluon 분석 기반 저중요도 피처 (p_value > 0.05 또는 importance < 0.0005)
LOW_IMPORTANCE_FEATURES = [
    # importance = 0 또는 음수
    '불임 원인 - 자궁경부 문제',
    '시술 유형',
    # p_value > 0.1
    'DI 출산 횟수',
    '대리모 여부',
    '불임 원인 - 정자 운동성',
    '난자 해동 경과일',
    '저장된 신선 난자 수',
    '불임 원인 - 정자 형태',
    # 기존 제거 대상
    '불임 원인 - 정자 면역학적 요인',
]


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    drop_features: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    피처 선택 (저중요도 피처 제거)

    Parameters
    ----------
    X_train      : 학습 데이터
    X_test       : 테스트 데이터
    drop_features: 제거할 피처 목록 (기본값: LOW_IMPORTANCE_FEATURES)

    Returns
    -------
    X_train : 피처 제거된 학습 데이터
    X_test  : 피처 제거된 테스트 데이터
    """
    if drop_features is None:
        drop_features = LOW_IMPORTANCE_FEATURES

    # 실제 존재하는 컬럼만 제거
    existing_drops = [c for c in drop_features if c in X_train.columns]

    if existing_drops:
        X_train = X_train.drop(columns=existing_drops)
        X_test = X_test.drop(columns=existing_drops)
        print(f"  제거된 피처 ({len(existing_drops)}개): {existing_drops[:5]}...")

    return X_train, X_test


def select_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    top_k: int = 100,
    method: str = 'lgbm_importance',
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    학습 후 중요도 기반 피처 선택 (자동 FE 피처용)

    Parameters
    ----------
    X_train : 학습 데이터
    y_train : 타겟
    X_test  : 테스트 데이터
    top_k   : 선택할 피처 수
    method  : 'lgbm_importance' 또는 'permutation'

    Returns
    -------
    X_train_selected : 선택된 피처만 포함된 학습 데이터
    X_test_selected  : 선택된 피처만 포함된 테스트 데이터
    selected_features: 선택된 피처 목록
    """
    print(f"  피처 선택 시작: {X_train.shape[1]}개 → top {top_k}개")

    if method == 'lgbm_importance':
        selected_features = _select_by_lgbm_importance(X_train, y_train, top_k)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print(f"  선택 완료: {len(selected_features)}개 피처")

    return X_train_selected, X_test_selected, selected_features


def _select_by_lgbm_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
) -> list:
    """LightGBM feature importance 기반 선택"""
    # 빠른 학습을 위한 간소화된 파라미터
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        verbosity=-1,
        force_col_wise=True,
    )

    model.fit(X_train, y_train)

    # 중요도 추출
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 상위 k개 선택
    selected = importance_df.head(top_k)['feature'].tolist()

    # 상위 10개 출력
    print("  Top 10 피처:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['feature'][:30]:<30} {row['importance']:>8.0f}")

    return selected


def get_low_importance_features() -> list:
    """저중요도 피처 목록 반환"""
    return LOW_IMPORTANCE_FEATURES


def analyze_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """전체 피처 중요도 분석 (디버깅용)"""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df
