"""
자동 피처 엔지니어링 — OpenFE 기반

OpenFE: GBDT 친화적 자동 피처 생성 라이브러리
- train 데이터로만 fit (Data Leakage 방지)
- 행 단위 연산 기반 (대회 규칙 준수)

설치:
    pip install openfe

사용법:
    from auto_feature_engineering import generate_openfe_features

    X_train, X_test, new_features = generate_openfe_features(
        X_train, y_train, X_test,
        n_features=30
    )
"""

import os
import uuid
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split

# 캐시 디렉토리
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache', 'openfe')


CACHE_VERSION = "v3"  # v3: 명시적 컬럼명 (fe_colA+colB 형태)

def _get_cache_key(X_train: pd.DataFrame, n_features: int, sample_size: int, model_type: str = 'default') -> str:
    """캐시 키 생성 (데이터 shape + 설정 기반)"""
    key_str = f"{CACHE_VERSION}_{X_train.shape}_{n_features}_{sample_size}_{model_type}_{list(X_train.columns)[:5]}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _get_cache_paths(cache_key: str) -> dict:
    """캐시 파일 경로들 반환"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return {
        'features': os.path.join(CACHE_DIR, f'{cache_key}_features.pkl'),
        'train': os.path.join(CACHE_DIR, f'{cache_key}_train.feather'),
        'test': os.path.join(CACHE_DIR, f'{cache_key}_test.feather'),
        'names': os.path.join(CACHE_DIR, f'{cache_key}_names.json'),
    }


def _load_cache(cache_key: str, verbose: bool = True) -> tuple:
    """캐시 로드 (없으면 None 반환)"""
    paths = _get_cache_paths(cache_key)

    if all(os.path.exists(p) for p in [paths['train'], paths['test'], paths['names']]):
        if verbose:
            print(f"  [Cache] 캐시 발견: {cache_key}")
        X_train_new = pd.read_feather(paths['train'])
        X_test_new = pd.read_feather(paths['test'])
        with open(paths['names'], 'r', encoding='utf-8') as f:
            new_feature_names = json.load(f)
        if verbose:
            print(f"  [Cache] 로드 완료: {X_train_new.shape[1]}개 피처")
        return X_train_new, X_test_new, new_feature_names

    return None, None, None


def _save_cache(cache_key: str, X_train_new: pd.DataFrame, X_test_new: pd.DataFrame,
                new_feature_names: list, features: list = None, verbose: bool = True):
    """캐시 저장"""
    paths = _get_cache_paths(cache_key)

    X_train_new.reset_index(drop=True).to_feather(paths['train'])
    X_test_new.reset_index(drop=True).to_feather(paths['test'])
    with open(paths['names'], 'w', encoding='utf-8') as f:
        json.dump(new_feature_names, f, ensure_ascii=False)

    if features is not None:
        with open(paths['features'], 'wb') as f:
            pickle.dump(features, f)

    if verbose:
        print(f"  [Cache] 저장 완료: {cache_key}")

# OpenFE import (설치 필요: pip install openfe)
try:
    from openfe import OpenFE, transform
    OPENFE_AVAILABLE = True
except ImportError:
    OPENFE_AVAILABLE = False
    print("  [경고] OpenFE가 설치되지 않았습니다. pip install openfe 실행 필요")


def generate_openfe_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_features: int = 30,
    n_jobs: int = 1,  # Windows 멀티프로세싱 불안정으로 싱글 프로세스 권장
    sample_size: int = 100000,
    verbose: bool = True,
    use_cache: bool = True,
    cache_key_suffix: str = '',
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    OpenFE를 사용한 자동 피처 생성

    Parameters
    ----------
    X_train : pd.DataFrame
        학습 데이터 (피처만)
    y_train : pd.Series
        타겟 변수
    X_test : pd.DataFrame
        테스트 데이터
    n_features : int
        생성할 새 피처 수 (기본값: 30)
    n_jobs : int
        병렬 처리 수 (기본값: 1)
    sample_size : int
        피처 발견용 샘플 크기 (기본값: 100000, None이면 전체 사용)
    verbose : bool
        상세 출력 여부
    use_cache : bool
        캐시 사용 여부 (기본값: True)
    cache_key_suffix : str
        캐시 키 접미사 (모델별 구분용, 예: 'lgbm', 'catboost')

    Returns
    -------
    X_train_new : pd.DataFrame
        새 피처가 추가된 학습 데이터
    X_test_new : pd.DataFrame
        새 피처가 추가된 테스트 데이터
    new_feature_names : list
        생성된 새 피처 이름 목록
    """
    if not OPENFE_AVAILABLE:
        print("  [오류] OpenFE가 설치되지 않았습니다.")
        print("  pip install openfe 실행 후 다시 시도하세요.")
        return X_train, X_test, []

    # 컬럼명 정리 (OpenFE 호환성)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.columns = X_train.columns.str.strip().str.replace(' - ', '_', regex=False)
    X_test.columns = X_test.columns.str.strip().str.replace(' - ', '_', regex=False)

    # 캐시 확인
    if use_cache:
        cache_key = _get_cache_key(X_train, n_features, sample_size, cache_key_suffix)
        X_train_cached, X_test_cached, names_cached = _load_cache(cache_key, verbose)
        if X_train_cached is not None:
            return X_train_cached, X_test_cached, names_cached

    if verbose:
        print(f"  [OpenFE] 자동 피처 생성 시작")
        print(f"    - 입력 피처 수: {X_train.shape[1]}")
        print(f"    - 생성할 피처 수: {n_features}")

    # 수치형 컬럼만 선택 (OpenFE는 수치형에 최적화)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if verbose:
        print(f"    - 수치형 피처 수: {len(numeric_cols)}")

    # 샘플링 (피처 발견 속도 향상)
    if sample_size is not None and len(X_train) > sample_size:
        X_sample, _, y_sample, _ = train_test_split(
            X_train[numeric_cols], y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=42
        )
        if verbose:
            print(f"    - 샘플링: {len(X_train):,} → {len(X_sample):,} rows (Stratified)")
    else:
        X_sample = X_train[numeric_cols]
        y_sample = y_train

    # OpenFE 초기화 및 fit (train 데이터만 사용 - Data Leakage 방지)
    ofe = OpenFE()

    # 임시 파일 경로 (절대 경로 + UUID로 충돌 방지)
    tmp_path = os.path.abspath(f'./openfe_tmp_{uuid.uuid4().hex[:8]}.feather')

    # 피처 생성 (샘플 데이터 기반 - 발견 후 전체에 적용)
    features = ofe.fit(
        data=X_sample,
        label=y_sample,
        n_jobs=n_jobs,
        verbose=verbose,
        tmp_save_path=tmp_path,
        seed=42,  # 재현성을 위한 시드 고정
    )

    # ⚠️ Cross-row 연산 및 불확실한 연산 필터링 (Data Leakage 방지)
    # - GroupByThen*: 그룹별 통계 (cross-row)
    # - Freq, Rank, Count: 전체 데이터 통계 (cross-row)
    # - Combine: 카테고리 조합 후 인코딩 불확실
    UNSAFE_OPS = ['GroupByThen', 'Freq', 'Rank', 'Count', 'Combine']

    safe_features = []
    filtered_count = 0
    for feat in features:
        feat_name = str(feat.name) if hasattr(feat, 'name') else str(feat)
        if any(op in feat_name for op in UNSAFE_OPS):
            filtered_count += 1
            continue
        safe_features.append(feat)

    if verbose and filtered_count > 0:
        print(f"\n  [OpenFE] ⚠️ Cross-row 연산 {filtered_count}개 필터링 (Data Leakage 방지)")

    # 상위 n_features개 선택 (안전한 피처만)
    top_features = safe_features[:n_features]

    if verbose:
        print(f"  [OpenFE] 상위 {len(top_features)}개 피처 선택 완료 (행 단위 연산만)")
        print("  Top 10 생성 피처:")
        for i, feat in enumerate(top_features[:10]):
            op_name = feat.name
            children = feat.children if hasattr(feat, 'children') else []
            child_names = [c.name if hasattr(c, 'name') else str(c) for c in children]
            print(f"    {i+1}. {op_name}({', '.join(child_names)})")

    # train/test 변환 (행 단위 연산만 적용 - 전체 데이터에 적용)
    X_train_new, X_test_new = transform(
        X_train, X_test,
        top_features,
        n_jobs=n_jobs
    )

    # 임시 파일 정리
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # autoFE_f_X → 실제 연산명으로 컬럼명 변경
    original_cols = set(X_train.columns)
    auto_cols = [c for c in X_train_new.columns if c not in original_cols]

    rename_map = {}
    for i, col in enumerate(auto_cols):
        if i < len(top_features):
            feat = top_features[i]
            op_name = feat.name  # 연산자: *, +, -, / 등

            # children에서 컬럼명 추출
            children = feat.children if hasattr(feat, 'children') else []
            child_names = []
            for c in children:
                if hasattr(c, 'name'):
                    child_names.append(c.name.replace(' ', ''))
                else:
                    child_names.append(str(c).replace(' ', ''))

            # 연산자 변환
            op_map = {'*': 'mul', '/': 'div', '+': 'add', '-': 'sub', 'max': 'max', 'min': 'min'}
            op_str = op_map.get(op_name, op_name)

            # 피처명 생성: col1_op_col2
            if len(child_names) >= 2:
                new_name = f"{child_names[0]}_{op_str}_{child_names[1]}"
            elif len(child_names) == 1:
                new_name = f"{op_str}_{child_names[0]}"
            else:
                new_name = f"{op_str}_{i}"

            # 너무 길면 자르기 (max 60자)
            if len(new_name) > 60:
                new_name = new_name[:60]

            rename_map[col] = new_name

    X_train_new = X_train_new.rename(columns=rename_map)
    X_test_new = X_test_new.rename(columns=rename_map)

    if verbose:
        print("  생성된 피처명:")
        for new in list(rename_map.values())[:5]:
            print(f"    {new}")

    # OpenFE 생성 피처 중 category 타입을 숫자형으로 변환 (CatBoost 호환)
    # train 기반 매핑을 test에 적용 (Data Leakage 방지)
    for col in X_train_new.columns:
        if X_train_new[col].dtype.name == 'category':
            # train의 카테고리 타입을 test에도 동일하게 적용
            cat_type = pd.CategoricalDtype(categories=X_train_new[col].cat.categories)
            X_test_new[col] = X_test_new[col].astype(cat_type)

            # codes 변환 (train에 없는 카테고리는 -1 → NaN)
            X_train_new[col] = X_train_new[col].cat.codes.replace(-1, np.nan).astype(float)
            X_test_new[col] = X_test_new[col].cat.codes.replace(-1, np.nan).astype(float)

    # 새로 생성된 피처 이름 (rename 후)
    new_feature_names = list(rename_map.values())

    if verbose:
        print(f"\n  [OpenFE] 완료")
        print(f"    - 최종 피처 수: {X_train_new.shape[1]}")
        print(f"    - 새 피처 수: {len(new_feature_names)}")

    # 캐시 저장
    if use_cache:
        _save_cache(cache_key, X_train_new, X_test_new, new_feature_names, top_features, verbose)

    return X_train_new, X_test_new, new_feature_names


def generate_openfe_features_simple(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_features: int = 20,
    sample_size: int = 100000,
    use_cache: bool = True,
    cache_key_suffix: str = '',
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    OpenFE 간소화 버전 (기본 설정)
    """
    return generate_openfe_features(
        X_train, y_train, X_test,
        n_features=n_features,
        n_jobs=1,
        sample_size=sample_size,
        verbose=True,
        use_cache=use_cache,
        cache_key_suffix=cache_key_suffix,
    )


# ────────────────────────────────────────────────────────────────
# Fallback: OpenFE 미설치 시 수동 조합 사용
# ────────────────────────────────────────────────────────────────

# AutoGluon Feature Importance 기반 상위 피처
TOP_FEATURES = [
    '이식된 배아 수',
    '시술 당시 나이',
    '배아 이식 경과일',
    '저장된 배아 수',
    '총 생성 배아 수',
    '수집된 신선 난자 수',
    '시술 시기 코드',
    'IVF 시술 횟수',
    '혼합된 난자 수',
    '클리닉 내 총 시술 횟수',
]


def generate_manual_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    수동 피처 조합 (OpenFE 미설치 시 fallback)

    행 단위 연산만 사용 (Data Leakage 방지)
    """
    from itertools import combinations

    print(f"  [Manual FE] 수동 피처 조합 생성")

    available = [f for f in TOP_FEATURES[:top_k] if f in X_train.columns]
    print(f"    - 사용 피처: {len(available)}개")

    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    new_features = []

    # 사칙연산 조합
    for a, b in combinations(available, 2):
        # 곱셈
        name = f"{a}__mul__{b}"
        X_train_new[name] = X_train[a] * X_train[b]
        X_test_new[name] = X_test[a] * X_test[b]
        new_features.append(name)

        # 나눗셈
        name = f"{a}__div__{b}"
        X_train_new[name] = X_train[a] / (X_train[b] + 1)
        X_test_new[name] = X_test[a] / (X_test[b] + 1)
        new_features.append(name)

    print(f"    - 생성된 피처: {len(new_features)}개")
    print(f"    - 총 피처 수: {X_train_new.shape[1]}개")

    return X_train_new, X_test_new, new_features


def generate_auto_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_features: int = 30,
    sample_size: int = 100000,
    use_openfe: bool = True,
    use_cache: bool = True,
    cache_key_suffix: str = '',
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    자동 피처 생성 (OpenFE 우선, 미설치 시 수동 fallback)
    """
    if use_openfe and OPENFE_AVAILABLE:
        return generate_openfe_features(
            X_train, y_train, X_test, n_features,
            sample_size=sample_size,
            use_cache=use_cache,
            cache_key_suffix=cache_key_suffix,
        )
    else:
        if use_openfe and not OPENFE_AVAILABLE:
            print("  [경고] OpenFE 미설치, 수동 조합으로 대체")
        return generate_manual_features(X_train, X_test)


def clear_cache(verbose: bool = True):
    """OpenFE 캐시 전체 삭제"""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        if verbose:
            print(f"  [Cache] 캐시 삭제 완료: {CACHE_DIR}")
    else:
        if verbose:
            print(f"  [Cache] 캐시 없음")
