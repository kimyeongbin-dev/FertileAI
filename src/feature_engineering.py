"""
피처 엔지니어링 — 난임 환자 대상 임신 성공 여부 예측

사용법:
    from feature_engineering import add_features

    X_train, X_test = add_features(X_prep, X_test_prep)

주의:
    - Data Leakage 방지: 행(row) 단위 연산만 사용
    - train/test 동일한 변환 적용
"""

import numpy as np
import pandas as pd


def add_features(
    X_prep: pd.DataFrame,
    X_test_prep: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    피처 엔지니어링 메인 함수

    Parameters
    ----------
    X_prep      : 전처리된 학습 데이터
    X_test_prep : 전처리된 테스트 데이터

    Returns
    -------
    X_train : 피처 엔지니어링 완료된 학습 데이터
    X_test  : 피처 엔지니어링 완료된 테스트 데이터
    """
    X_train = _create_features(X_prep)
    X_test = _create_features(X_test_prep)

    # 컬럼 순서 일치 (혹시 모를 불일치 방지)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test


def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    개별 피처 생성 (행 단위 연산만)

    Data Leakage 방지:
    - 열(column) 전체 통계 사용 금지
    - 다른 행 참조 금지
    """
    df = df.copy()

    # ────────────────────────────────────────────────────────────
    # 1. 비율 피처 (효율/성공률)
    # ────────────────────────────────────────────────────────────

    # 배아 이식 효율: 이식된 배아 / 총 생성 배아
    if '이식된 배아 수' in df.columns and '총 생성 배아 수' in df.columns:
        df['배아_이식_효율'] = df['이식된 배아 수'] / (df['총 생성 배아 수'] + 1)

    # 난자 활용률: 혼합된 난자 / 수집된 신선 난자
    if '혼합된 난자 수' in df.columns and '수집된 신선 난자 수' in df.columns:
        df['난자_활용률'] = df['혼합된 난자 수'] / (df['수집된 신선 난자 수'] + 1)

    # ICSI 효율: ICSI 생성 배아 / 미세주입 난자
    if '미세주입에서 생성된 배아 수' in df.columns and '미세주입된 난자 수' in df.columns:
        df['ICSI_효율'] = df['미세주입에서 생성된 배아 수'] / (df['미세주입된 난자 수'] + 1)

    # 배아 저장 비율: 저장 배아 / 총 생성 배아
    if '저장된 배아 수' in df.columns and '총 생성 배아 수' in df.columns:
        df['배아_저장_비율'] = df['저장된 배아 수'] / (df['총 생성 배아 수'] + 1)

    # 임신 성공률 (과거 이력): 총 임신 횟수 / 총 시술 횟수
    if '총 임신 횟수' in df.columns and '총 시술 횟수' in df.columns:
        df['과거_임신_성공률'] = df['총 임신 횟수'] / (df['총 시술 횟수'] + 1)

    # 출산 성공률 (과거 이력): 총 출산 횟수 / 총 임신 횟수
    if '총 출산 횟수' in df.columns and '총 임신 횟수' in df.columns:
        df['과거_출산_성공률'] = df['총 출산 횟수'] / (df['총 임신 횟수'] + 1)

    # ────────────────────────────────────────────────────────────
    # 2. 합계/집계 피처
    # ────────────────────────────────────────────────────────────

    # 총 불임 원인 수
    infertility_cols = [
        '남성 주 불임 원인', '남성 부 불임 원인',
        '여성 주 불임 원인', '여성 부 불임 원인',
        '부부 주 불임 원인', '부부 부 불임 원인',
        '불명확 불임 원인',
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인',
        '불임 원인 - 배란 장애', '불임 원인 - 자궁경부 문제',
        '불임 원인 - 자궁내막증', '불임 원인 - 정자 농도',
        '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
        '불임 원인 - 정자 형태',
    ]
    existing_infertility_cols = [c for c in infertility_cols if c in df.columns]
    if existing_infertility_cols:
        df['총_불임_원인_수'] = df[existing_infertility_cols].sum(axis=1)

    # 남성 요인 수
    male_cols = [
        '남성 주 불임 원인', '남성 부 불임 원인', '불임 원인 - 남성 요인',
        '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인',
        '불임 원인 - 정자 운동성', '불임 원인 - 정자 형태',
    ]
    existing_male_cols = [c for c in male_cols if c in df.columns]
    if existing_male_cols:
        df['남성_요인_수'] = df[existing_male_cols].sum(axis=1)

    # 여성 요인 수
    female_cols = [
        '여성 주 불임 원인', '여성 부 불임 원인',
        '불임 원인 - 난관 질환', '불임 원인 - 배란 장애',
        '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증',
    ]
    existing_female_cols = [c for c in female_cols if c in df.columns]
    if existing_female_cols:
        df['여성_요인_수'] = df[existing_female_cols].sum(axis=1)

    # ────────────────────────────────────────────────────────────
    # 3. 플래그/이진 피처
    # ────────────────────────────────────────────────────────────

    # 첫 시술 여부
    if '총 시술 횟수' in df.columns:
        df['첫_시술'] = (df['총 시술 횟수'] == 0).astype(np.int8)

    # 고령 여부 (만40세 이상 = 인코딩값 3 이상)
    if '시술 당시 나이' in df.columns:
        df['고령'] = (df['시술 당시 나이'] >= 3).astype(np.int8)

    # 반복 실패 여부 (시술 > 임신)
    if '총 시술 횟수' in df.columns and '총 임신 횟수' in df.columns:
        df['반복_실패'] = (df['총 시술 횟수'] > df['총 임신 횟수']).astype(np.int8)

    # ────────────────────────────────────────────────────────────
    # 4. 특정 시술 유형 파싱 (Target Encoding 전 원본 기준)
    # ────────────────────────────────────────────────────────────

    # 주의: 전처리에서 Target Encoding이 적용되므로,
    # 이 피처들은 전처리 전에 생성하거나 별도 처리 필요
    # 여기서는 이미 인코딩된 경우 생략

    # ────────────────────────────────────────────────────────────
    # 5. 교차/상호작용 피처
    # ────────────────────────────────────────────────────────────

    # 나이 × 시술 횟수
    if '시술 당시 나이' in df.columns and '총 시술 횟수' in df.columns:
        df['나이_시술횟수'] = df['시술 당시 나이'] * df['총 시술 횟수']

    # 나이 × 배아 수
    if '시술 당시 나이' in df.columns and '이식된 배아 수' in df.columns:
        df['나이_배아수'] = df['시술 당시 나이'] * df['이식된 배아 수']

    # 클리닉 충성도
    if '클리닉 내 총 시술 횟수' in df.columns and '총 시술 횟수' in df.columns:
        df['클리닉_충성도'] = df['클리닉 내 총 시술 횟수'] / (df['총 시술 횟수'] + 1)

    # ────────────────────────────────────────────────────────────
    # 6. 배아/난자 파생 피처
    # ────────────────────────────────────────────────────────────

    # 잔여 배아 (생성 - 이식 - 저장)
    if all(c in df.columns for c in ['총 생성 배아 수', '이식된 배아 수', '저장된 배아 수']):
        df['잔여_배아'] = df['총 생성 배아 수'] - df['이식된 배아 수'] - df['저장된 배아 수']

    # 파트너 정자 사용 비율
    if '파트너 정자와 혼합된 난자 수' in df.columns and '혼합된 난자 수' in df.columns:
        df['파트너_정자_비율'] = df['파트너 정자와 혼합된 난자 수'] / (df['혼합된 난자 수'] + 1)

    return df


def get_new_feature_names() -> list:
    """생성되는 새 피처 이름 목록"""
    return [
        # 비율 피처
        '배아_이식_효율',
        '난자_활용률',
        'ICSI_효율',
        '배아_저장_비율',
        '과거_임신_성공률',
        '과거_출산_성공률',
        # 합계 피처
        '총_불임_원인_수',
        '남성_요인_수',
        '여성_요인_수',
        # 플래그 피처
        '첫_시술',
        '고령',
        '반복_실패',
        # 교차 피처
        '나이_시술횟수',
        '나이_배아수',
        '클리닉_충성도',
        # 파생 피처
        '잔여_배아',
        '파트너_정자_비율',
    ]
