"""
베이스라인 학습 파이프라인 — Option A 실험용

목적:
    - 피처 엔지니어링/선택 없이 순수 모델 성능 비교
    - CV-LB gap 확인용 제출 파일 생성
    - OpenFE 자동 피처 엔지니어링 실험

Option A 제출 명령어:
    # 제출 1: CB 6:4 LGBM 앙상블 (baseline)
    python train_baseline.py --model ensemble --submit

    # 제출 2: OpenFE 자동 FE + 앙상블
    python train_baseline.py --model ensemble --auto-fe --submit

    # 제출 3: OpenFE + 피처 선택 (기본 80개) + 앙상블
    python train_baseline.py --model ensemble --auto-fe --feature-select --submit

    # 제출 4: OpenFE + 피처 선택 (상위 50개) + 앙상블
    python train_baseline.py --model ensemble --auto-fe --feature-select 50 --submit

기타 사용법:
    python train_baseline.py                     # 전체 모델 비교 (제출 없음)
    python train_baseline.py --model lgbm        # LGBM만 실행
    python train_baseline.py --model catboost    # CatBoost만 실행
    python train_baseline.py --fe                # 수동 FE 적용
"""

import argparse
import os
import sys
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# LightGBM C++ 라이브러리 로그 억제
logging.getLogger('lightgbm').setLevel(logging.ERROR)
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

import random
import numpy as np
import pandas as pd

def set_seed(seed: int = 42):
    """재현성을 위한 전역 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (Windows)
font_path = 'C:/Windows/Fonts/malgun.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from preprocessing_baseline import preprocess_baseline, get_categorical_features
from feature_engineering import add_features
from auto_feature_engineering import generate_auto_features
from feature_selection import select_by_importance

# OOF 저장 경로
OOF_DIR = "../outputs/oof_cache"

# OpenFE 설정
OPENFE_N_FEATURES = 30      # OpenFE로 생성할 피처 수
FEATURE_SELECT_TOP_K = 80   # 피처 선택 시 상위 80개


# ────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────
DATA_DIR = "../datasets"
OUTPUT_DIR = "../outputs"
N_SPLITS = 5
RANDOM_STATE = 42


# ────────────────────────────────────────────────────────────────
# 모델 기본 파라미터 (최소한의 튜닝)
# ────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'n_estimators': 1000,
    'is_unbalance': True,
    'verbosity': -1,
    'random_state': RANDOM_STATE,
}

CATBOOST_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 1000,
    'auto_class_weights': 'Balanced',
    'random_seed': RANDOM_STATE,
    'verbose': False,
}

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 1000,
    'scale_pos_weight': 2.87,  # 클래스 불균형 보정
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'verbosity': 0,
}


# ────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ────────────────────────────────────────────────────────────────
def load_data():
    """데이터 로드"""
    print("  Loading train.csv...", end=" ", flush=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    print(f"({train.shape[0]:,} rows, {train.shape[1]} cols)")

    print("  Loading test.csv...", end=" ", flush=True)
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"({test.shape[0]:,} rows, {test.shape[1]} cols)")

    return train, test


def create_run_dir(prefix: str = "baseline") -> str:
    """실행별 출력 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_submission(test_ids: pd.Series, preds: np.ndarray, run_dir: str, model_name: str):
    """submission.csv 저장"""
    filename = f"submission_{model_name}.csv"
    filepath = os.path.join(run_dir, filename)

    submission = pd.DataFrame({
        "ID": test_ids,
        "probability": preds
    })
    submission.to_csv(filepath, index=False)
    print(f"    저장: {filepath}")
    return filepath


def print_header(title: str):
    """헤더 출력"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ────────────────────────────────────────────────────────────────
# CV 학습 함수
# ────────────────────────────────────────────────────────────────
def cv_train_lgbm(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> dict:
    """LightGBM CV 학습"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, valid_idx) in enumerate(tqdm(skf.split(X, y), total=N_SPLITS, desc="    LGBM CV", ncols=80)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
        scores.append(fold_auc)

    oof_auc = roc_auc_score(y, oof_preds)
    return {
        'model': 'LGBM',
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'oof_auc': oof_auc,
        'fold_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }


def cv_train_catboost(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> dict:
    """CatBoost CV 학습"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    # 범주형 컬럼 인덱스
    cat_features = get_categorical_features(X)

    for fold, (train_idx, valid_idx) in enumerate(tqdm(skf.split(X, y), total=N_SPLITS, desc="    CatBoost CV", ncols=80)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=100,
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
        scores.append(fold_auc)

    oof_auc = roc_auc_score(y, oof_preds)
    return {
        'model': 'CatBoost',
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'oof_auc': oof_auc,
        'fold_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }


def cv_train_xgb(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> dict:
    """XGBoost CV 학습"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, valid_idx) in enumerate(tqdm(skf.split(X, y), total=N_SPLITS, desc="    XGBoost CV", ncols=80)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS, enable_categorical=True)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
        scores.append(fold_auc)

    oof_auc = roc_auc_score(y, oof_preds)
    return {
        'model': 'XGBoost',
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'oof_auc': oof_auc,
        'fold_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
    }


# ────────────────────────────────────────────────────────────────
# OOF 저장/로드 함수
# ────────────────────────────────────────────────────────────────
def save_oof(oof_dict: dict, test_dict: dict, y_train: pd.Series, suffix: str = "base"):
    """OOF 및 test predictions 저장"""
    save_dir = os.path.join(OOF_DIR, suffix)
    os.makedirs(save_dir, exist_ok=True)

    for model_name, oof in oof_dict.items():
        np.save(os.path.join(save_dir, f"oof_{model_name}_{suffix}.npy"), oof)

    for model_name, test_pred in test_dict.items():
        np.save(os.path.join(save_dir, f"test_{model_name}_{suffix}.npy"), test_pred)

    np.save(os.path.join(save_dir, f"y_train_{suffix}.npy"), y_train.values)

    print(f"\n  OOF 저장 완료: {save_dir}")


def load_oof(suffix: str = "base") -> tuple:
    """저장된 OOF 및 test predictions 로드"""
    load_dir = os.path.join(OOF_DIR, suffix)

    oof_dict = {}
    test_dict = {}

    for model_name in ['lgbm', 'catboost', 'xgb']:
        oof_path = os.path.join(load_dir, f"oof_{model_name}_{suffix}.npy")
        test_path = os.path.join(load_dir, f"test_{model_name}_{suffix}.npy")

        if os.path.exists(oof_path):
            oof_dict[model_name] = np.load(oof_path)
            print(f"    로드: {suffix}/oof_{model_name}_{suffix}.npy")
        if os.path.exists(test_path):
            test_dict[model_name] = np.load(test_path)

    y_train = np.load(os.path.join(load_dir, f"y_train_{suffix}.npy"))

    return oof_dict, test_dict, y_train


def blend_predictions(oof_dict: dict, test_dict: dict, y_train: np.ndarray, weights: dict) -> dict:
    """주어진 가중치로 블렌딩 계산"""
    oof_blend = np.zeros(len(y_train))
    test_blend = np.zeros(len(list(test_dict.values())[0]))

    total_weight = sum(weights.values())

    for model_name, w in weights.items():
        if model_name in oof_dict and w > 0:
            oof_blend += oof_dict[model_name] * (w / total_weight)
            test_blend += test_dict[model_name] * (w / total_weight)

    oof_auc = roc_auc_score(y_train, oof_blend)

    return {
        'oof_preds': oof_blend,
        'test_preds': test_blend,
        'oof_auc': oof_auc,
        'weights': weights,
    }


# ────────────────────────────────────────────────────────────────
# 앙상블 학습 함수
# ────────────────────────────────────────────────────────────────
# 앙상블 가중치 (6:4 최적 비율)
ENSEMBLE_WEIGHTS = {
    'cb_lgbm': {'lgbm': 0.4, 'catboost': 0.6, 'xgb': 0.0},  # CB 6:4 LGBM (최적)
    'full': {'lgbm': 0.35, 'catboost': 0.50, 'xgb': 0.15},  # 전체
}


def cv_train_ensemble(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    apply_fe: bool = False,
    apply_auto_fe: bool = False,
    apply_feature_select: bool = False,
    top_k: int = 80,
    ensemble_type: str = 'cb_lgbm',
    save_oof_flag: bool = False,
) -> dict:
    """앙상블 CV 학습 (모델별 전처리 개별 적용)"""

    weights = ENSEMBLE_WEIGHTS[ensemble_type]
    model_results = {}

    # 1. LGBM
    if weights['lgbm'] > 0:
        print(f"\n  [LGBM] (weight: {weights['lgbm']})")
        X_train, y_train, X_test = preprocess_baseline(train_raw, test_raw, for_catboost=False)

        # 수동 피처 엔지니어링
        if apply_fe:
            X_train, X_test = add_features(X_train, X_test)
            print(f"    수동 FE 적용: {X_train.shape[1]}개 피처")

        # OpenFE 자동 피처 엔지니어링
        if apply_auto_fe:
            print(f"    OpenFE 자동 FE 시작...")
            X_train, X_test, new_features = generate_auto_features(
                X_train, y_train, X_test,
                n_features=OPENFE_N_FEATURES,
                use_openfe=True,
                cache_key_suffix='lgbm'
            )
            print(f"    자동 FE 완료: {X_train.shape[1]}개 피처 (+{len(new_features)}개)")

        # 피처 선택
        if apply_feature_select:
            print(f"    피처 선택 시작 (top_k={top_k})...")
            X_train, X_test, selected = select_by_importance(
                X_train, y_train, X_test,
                top_k=top_k
            )
            print(f"    피처 선택 완료: {X_train.shape[1]}개 피처")

        print(f"    최종 피처 수: {X_train.shape[1]}")
        model_results['lgbm'] = cv_train_lgbm(X_train, y_train, X_test)
        print(f"    OOF AUC: {model_results['lgbm']['oof_auc']:.5f}")

    # 2. CatBoost
    if weights['catboost'] > 0:
        print(f"\n  [CatBoost] (weight: {weights['catboost']})")
        X_train_cb, y_train, X_test_cb = preprocess_baseline(train_raw, test_raw, for_catboost=True)

        # 수동 피처 엔지니어링
        if apply_fe:
            X_train_cb, X_test_cb = add_features(X_train_cb, X_test_cb)

        # OpenFE 자동 피처 엔지니어링
        # 주의: CatBoost 전처리는 범주형을 문자열로 유지 → 수치형 피처 수가 LGBM과 다름
        # 따라서 별도 캐시 사용 (시간 2배 소요되지만 정확한 결과)
        if apply_auto_fe:
            X_train_cb, X_test_cb, _ = generate_auto_features(
                X_train_cb, y_train, X_test_cb,
                n_features=OPENFE_N_FEATURES,
                use_openfe=True,
                cache_key_suffix='catboost'
            )

        # 피처 선택 (LGBM에서 선택된 피처 사용)
        if apply_feature_select:
            # LGBM에서 선택된 피처와 동일하게 적용
            common_cols = [c for c in selected if c in X_train_cb.columns]
            X_train_cb = X_train_cb[common_cols]
            X_test_cb = X_test_cb[common_cols]

        print(f"    최종 피처 수: {X_train_cb.shape[1]}")
        model_results['catboost'] = cv_train_catboost(X_train_cb, y_train, X_test_cb)
        print(f"    OOF AUC: {model_results['catboost']['oof_auc']:.5f}")

    # 3. XGBoost (가중치 > 0일 때만)
    if weights['xgb'] > 0:
        print(f"\n  [XGBoost] (weight: {weights['xgb']})")
        if 'X_train' not in dir():
            X_train, y_train, X_test = preprocess_baseline(train_raw, test_raw, for_catboost=False)
            if apply_fe:
                X_train, X_test = add_features(X_train, X_test)
            if apply_auto_fe:
                X_train, X_test, _ = generate_auto_features(
                    X_train, y_train, X_test,
                    n_features=OPENFE_N_FEATURES,
                    use_openfe=True,
                    cache_key_suffix='xgb'
                )
            if apply_feature_select:
                X_train, X_test, _ = select_by_importance(
                    X_train, y_train, X_test,
                    top_k=top_k
                )
        model_results['xgb'] = cv_train_xgb(X_train, y_train, X_test)
        print(f"    OOF AUC: {model_results['xgb']['oof_auc']:.5f}")

    # 4. 앙상블 OOF 계산
    oof_ensemble = np.zeros(len(y_train))
    test_ensemble = np.zeros(len(X_test_cb) if 'X_test_cb' in dir() else len(X_test))

    for model_name, w in weights.items():
        if w > 0 and model_name in model_results:
            oof_ensemble += model_results[model_name]['oof_preds'] * w
            test_ensemble += model_results[model_name]['test_preds'] * w

    oof_auc = roc_auc_score(y_train, oof_ensemble)

    print(f"\n  [Ensemble: {ensemble_type}]")
    active_weights = {k: v for k, v in weights.items() if v > 0}
    print(f"    가중치: {active_weights}")
    print(f"    ★ OOF AUC: {oof_auc:.5f}")

    # 개별 모델들의 std 평균 계산
    std_scores = [r['std_score'] for r in model_results.values() if 'std_score' in r]
    mean_std = np.mean(std_scores) if std_scores else 0.0

    # OOF 저장
    if save_oof_flag:
        oof_dict = {name: r['oof_preds'] for name, r in model_results.items()}
        test_dict = {name: r['test_preds'] for name, r in model_results.items()}
        suffix = "fe" if apply_fe else "base"
        save_oof(oof_dict, test_dict, y_train, suffix=suffix)

    return {
        'model': 'Ensemble',
        'oof_preds': oof_ensemble,
        'test_preds': test_ensemble,
        'oof_auc': oof_auc,
        'std_score': mean_std,
        'model_results': model_results,
    }


# ────────────────────────────────────────────────────────────────
# 결과 출력 함수
# ────────────────────────────────────────────────────────────────
def print_result(result: dict):
    """단일 모델 결과 출력"""
    print(f"    OOF AUC: {result['oof_auc']:.5f} (mean: {result['mean_score']:.5f} ± {result['std_score']:.5f})")


def print_comparison(results: list):
    """모델 비교 테이블 출력"""
    print_header("모델 비교 결과")

    print(f"\n    {'Model':<12} {'OOF AUC':>10} {'Mean±Std':>18}")
    print(f"    {'─'*12} {'─'*10} {'─'*18}")

    sorted_results = sorted(results, key=lambda x: x['oof_auc'], reverse=True)
    for r in sorted_results:
        print(f"    {r['model']:<12} {r['oof_auc']:>10.5f} {r['mean_score']:.5f}±{r['std_score']:.5f}")

    best = sorted_results[0]
    print(f"\n    ★ Best: {best['model']} (OOF AUC: {best['oof_auc']:.5f})")


def save_comparison_plot(results: list, save_path: str):
    """모델 비교 차트 저장"""
    models = [r['model'] for r in results]
    scores = [r['oof_auc'] for r in results]

    plt.figure(figsize=(8, 5))
    colors = ['#4e79a7', '#f28e2c', '#59a14f']
    bars = plt.bar(models, scores, color=colors[:len(models)], edgecolor='white')

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{score:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('OOF AUC')
    plt.title('Baseline Model Comparison')
    plt.ylim(min(scores) - 0.01, max(scores) + 0.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    차트 저장: {save_path}")


def save_run_info(results: list, run_dir: str, X_shape: tuple):
    """실행 정보 저장"""
    filepath = os.path.join(run_dir, "baseline_info.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("베이스라인 실험 결과\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"피처 수: {X_shape[1]}\n")
        f.write(f"학습 샘플 수: {X_shape[0]:,}\n")
        f.write(f"CV Folds: {N_SPLITS}\n\n")

        f.write("─" * 40 + "\n")
        f.write("전처리 설정\n")
        f.write("─" * 40 + "\n")
        f.write("  - 피처 엔지니어링: 없음\n")
        f.write("  - 피처 선택: 없음\n")
        f.write("  - Target Encoding: 없음\n")
        f.write("  - 결측치: 모델 자체 처리\n\n")

        f.write("─" * 40 + "\n")
        f.write("모델별 OOF AUC\n")
        f.write("─" * 40 + "\n")
        for r in sorted(results, key=lambda x: x['oof_auc'], reverse=True):
            std = r.get('std_score', 0.0)
            f.write(f"  {r['model']:<12}: {r['oof_auc']:.5f} (±{std:.5f})\n")
        f.write("\n")

    print(f"    정보 저장: {filepath}")


# ────────────────────────────────────────────────────────────────
# 블렌드 모드 함수
# ────────────────────────────────────────────────────────────────
def run_blend_mode(args):
    """저장된 OOF로 다양한 비율 블렌딩 (재학습 없음)"""

    # 그리드 서치 모드
    if args.grid_search:
        run_blend_grid_search(args)
        return

    print("\n" + "=" * 60)
    print("  블렌드 모드 — 비율 실험 (재학습 없음)")
    print("=" * 60)

    # OOF 로드
    print(f"\n  OOF 로드 (suffix: {args.oof_suffix})")
    oof_dict, test_dict, y_train = load_oof(suffix=args.oof_suffix)

    if not oof_dict:
        print("\n  [오류] 저장된 OOF가 없습니다. 먼저 --save-oof로 학습하세요.")
        return

    # 가중치 파싱
    weight_values = [float(w) for w in args.weights.split(',')]

    if len(weight_values) == 2:
        weights = {'catboost': weight_values[0], 'lgbm': weight_values[1]}
    elif len(weight_values) == 3:
        weights = {'catboost': weight_values[0], 'lgbm': weight_values[1], 'xgb': weight_values[2]}
    else:
        print("\n  [오류] 가중치는 2개(CB,LGBM) 또는 3개(CB,LGBM,XGB)여야 합니다.")
        return

    print(f"\n  가중치: {weights}")

    # 블렌딩 계산
    result = blend_predictions(oof_dict, test_dict, y_train, weights)

    print(f"\n  ★ Blend OOF AUC: {result['oof_auc']:.5f}")

    # 제출 파일 생성
    if args.submit:
        run_dir = create_run_dir("blend")
        test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        test_ids = test_raw['ID']
        weight_str = "_".join([f"{k}{int(v*100)}" for k, v in weights.items()])
        save_submission(test_ids, result['test_preds'], run_dir, f"blend_{weight_str}")

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60 + "\n")


def run_blend_grid_search(args):
    """그리드 서치로 최적 블렌딩 비율 탐색"""
    print("\n" + "=" * 60)
    print("  블렌드 그리드 서치 — 최적 비율 탐색")
    print("=" * 60)

    # OOF 로드
    print(f"\n  OOF 로드 (suffix: {args.oof_suffix})")
    oof_dict, test_dict, y_train = load_oof(suffix=args.oof_suffix)

    if not oof_dict:
        print("\n  [오류] 저장된 OOF가 없습니다. 먼저 --save-oof로 학습하세요.")
        return

    available_models = list(oof_dict.keys())
    print(f"  사용 가능 모델: {available_models}")

    # 그리드 생성
    step = args.grid_step
    results = []

    if len(available_models) == 2:
        # 2개 모델: CB, LGBM
        print(f"\n  탐색 범위: 0.0 ~ 1.0 (step: {step})")
        print(f"  {'CB':<6} {'LGBM':<6} {'OOF AUC':<10}")
        print(f"  {'-'*6} {'-'*6} {'-'*10}")

        cb_weights = np.arange(0.0, 1.0 + step, step)
        for cb_w in cb_weights:
            lgbm_w = 1.0 - cb_w
            if lgbm_w < 0:
                continue

            weights = {'catboost': cb_w, 'lgbm': lgbm_w}
            result = blend_predictions(oof_dict, test_dict, y_train, weights)
            results.append({
                'catboost': cb_w,
                'lgbm': lgbm_w,
                'oof_auc': result['oof_auc'],
                'test_preds': result['test_preds'],
            })
            print(f"  {cb_w:<6.2f} {lgbm_w:<6.2f} {result['oof_auc']:.5f}")

    elif len(available_models) == 3:
        # 3개 모델: CB, LGBM, XGB
        print(f"\n  탐색 범위: 0.0 ~ 1.0 (step: {step})")
        print(f"  {'CB':<6} {'LGBM':<6} {'XGB':<6} {'OOF AUC':<10}")
        print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*10}")

        for cb_w in np.arange(0.0, 1.0 + step, step):
            for lgbm_w in np.arange(0.0, 1.0 - cb_w + step, step):
                xgb_w = 1.0 - cb_w - lgbm_w
                if xgb_w < -0.001 or xgb_w > 1.001:
                    continue
                xgb_w = max(0, min(1, xgb_w))  # 부동소수점 오차 보정

                weights = {'catboost': cb_w, 'lgbm': lgbm_w, 'xgb': xgb_w}
                result = blend_predictions(oof_dict, test_dict, y_train, weights)
                results.append({
                    'catboost': cb_w,
                    'lgbm': lgbm_w,
                    'xgb': xgb_w,
                    'oof_auc': result['oof_auc'],
                    'test_preds': result['test_preds'],
                })
                print(f"  {cb_w:<6.2f} {lgbm_w:<6.2f} {xgb_w:<6.2f} {result['oof_auc']:.5f}")

    # 정렬 및 Top 결과
    results_sorted = sorted(results, key=lambda x: x['oof_auc'], reverse=True)

    print(f"\n  {'='*50}")
    print(f"  Top 10 결과")
    print(f"  {'='*50}")

    if len(available_models) == 2:
        print(f"  {'Rank':<6} {'CB':<6} {'LGBM':<6} {'OOF AUC':<10}")
        print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
        for i, r in enumerate(results_sorted[:10], 1):
            marker = "★" if i == 1 else " "
            print(f"  {marker}{i:<5} {r['catboost']:<6.2f} {r['lgbm']:<6.2f} {r['oof_auc']:.5f}")
    else:
        print(f"  {'Rank':<6} {'CB':<6} {'LGBM':<6} {'XGB':<6} {'OOF AUC':<10}")
        print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
        for i, r in enumerate(results_sorted[:10], 1):
            marker = "★" if i == 1 else " "
            print(f"  {marker}{i:<5} {r['catboost']:<6.2f} {r['lgbm']:<6.2f} {r['xgb']:<6.2f} {r['oof_auc']:.5f}")

    # 최적 결과
    best = results_sorted[0]
    print(f"\n  ★ 최적 비율: CB={best['catboost']:.2f}, LGBM={best['lgbm']:.2f}", end="")
    if 'xgb' in best:
        print(f", XGB={best['xgb']:.2f}")
    else:
        print()
    print(f"  ★ 최적 OOF AUC: {best['oof_auc']:.5f}")

    # 제출 파일 생성
    if args.submit:
        run_dir = create_run_dir("blend_grid")
        test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        test_ids = test_raw['ID']

        weight_str = f"CB{int(best['catboost']*100)}_LG{int(best['lgbm']*100)}"
        if 'xgb' in best:
            weight_str += f"_XG{int(best['xgb']*100)}"

        save_submission(test_ids, best['test_preds'], run_dir, f"blend_best_{weight_str}")

        # 결과 저장
        results_df = pd.DataFrame(results_sorted)
        results_df.to_csv(os.path.join(run_dir, "grid_search_results.csv"), index=False)
        print(f"  결과 저장: {run_dir}/grid_search_results.csv")

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60 + "\n")


# ────────────────────────────────────────────────────────────────
# 메인 함수
# ────────────────────────────────────────────────────────────────
def main(args):
    set_seed(RANDOM_STATE)

    # 블렌드 모드 (재학습 없이 비율만 변경)
    if args.blend:
        run_blend_mode(args)
        return

    # 실험 이름 생성
    exp_parts = ["baseline"]
    if args.fe:
        exp_parts.append("fe")
    if args.auto_fe:
        exp_parts.append("autofe")
    if args.feature_select is not None:
        exp_parts.append(f"fs{args.feature_select}")
    exp_name = "+".join(exp_parts)

    print("\n" + "=" * 60)
    print(f"  실험: {exp_name} — 모델 성능 비교")
    print("=" * 60)

    # 출력 디렉토리 생성
    run_dir = create_run_dir(exp_name)
    print(f"  출력 디렉토리: {run_dir}")

    # 1. 데이터 로드
    print_header("1. 데이터 로드")
    train_raw, test_raw = load_data()
    test_ids = test_raw['ID']

    # 2. 전처리 (모델별로 다르게 처리)
    print_header("2. 전처리")
    results = []

    # 실행할 모델 결정
    if args.model == 'ensemble':
        # 앙상블 모드
        print_header("3. 앙상블 학습")
        print(f"    타겟 분포: 0={int((train_raw['임신 성공 여부'] == 0).sum()):,}, "
              f"1={int((train_raw['임신 성공 여부'] == 1).sum()):,}")

        result = cv_train_ensemble(
            train_raw, test_raw,
            apply_fe=args.fe,
            apply_auto_fe=args.auto_fe,
            apply_feature_select=args.feature_select is not None,
            top_k=args.feature_select or 80,
            ensemble_type=args.ensemble_type,
            save_oof_flag=args.save_oof
        )
        results = [result]

        if args.submit:
            save_submission(test_ids, result['test_preds'], run_dir, 'ensemble')

        # 정보 저장
        save_run_info(results, run_dir, (len(train_raw), 65))

        print("\n" + "=" * 60)
        print("  완료!")
        print("=" * 60 + "\n")
        return

    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = ['lgbm', 'catboost', 'xgb']

    # 3. 모델별 학습
    print_header("3. 모델 학습")

    for model_name in models_to_run:
        print(f"\n  [{model_name.upper()}]")

        # 모델별 전처리
        if model_name == 'catboost':
            X_train, y_train, X_test = preprocess_baseline(train_raw, test_raw, for_catboost=True)
        else:
            X_train, y_train, X_test = preprocess_baseline(train_raw, test_raw, for_catboost=False)

        # 피처 엔지니어링 적용
        if args.fe:
            n_before = X_train.shape[1]
            X_train, X_test = add_features(X_train, X_test)
            print(f"    피처 엔지니어링: +{X_train.shape[1] - n_before}개 피처")

        print(f"    피처 수: {X_train.shape[1]}")
        print(f"    타겟 분포: 0={int((y_train == 0).sum()):,} ({(y_train == 0).mean()*100:.1f}%), "
              f"1={int((y_train == 1).sum()):,} ({(y_train == 1).mean()*100:.1f}%)")

        # CV 학습
        if model_name == 'lgbm':
            result = cv_train_lgbm(X_train, y_train, X_test)
        elif model_name == 'catboost':
            result = cv_train_catboost(X_train, y_train, X_test)
        elif model_name == 'xgb':
            result = cv_train_xgb(X_train, y_train, X_test)

        print_result(result)
        results.append(result)

        # Submission 저장
        if args.submit:
            save_submission(test_ids, result['test_preds'], run_dir, model_name)

    # 4. 결과 비교
    if len(results) > 1:
        print_comparison(results)
        save_comparison_plot(results, os.path.join(run_dir, "model_comparison.png"))

    # 5. 정보 저장
    save_run_info(results, run_dir, X_train.shape)

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="베이스라인 모델 성능 비교")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "catboost", "xgb", "ensemble"],
        help="특정 모델만 실행 (미지정시 전체 비교, ensemble=가중 앙상블)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submission 파일 생성"
    )
    parser.add_argument(
        "--fe",
        action="store_true",
        help="수동 피처 엔지니어링 적용"
    )
    parser.add_argument(
        "--auto-fe",
        action="store_true",
        help="OpenFE 자동 피처 엔지니어링 적용"
    )
    parser.add_argument(
        "--feature-select",
        type=int,
        nargs='?',
        const=80,
        default=None,
        metavar='K',
        help="피처 선택 적용 (기본: 80개, 예: --feature-select 50)"
    )
    parser.add_argument(
        "--ensemble-type",
        type=str,
        default="cb_lgbm",
        choices=["cb_lgbm", "full"],
        help="앙상블 유형: cb_lgbm(CB+LGBM), full(전체)"
    )
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help="OOF predictions 저장 (비율 실험용)"
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="저장된 OOF로 블렌딩 (재학습 없음)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.5,0.5",
        help="블렌딩 가중치 (CB,LGBM 순서, 예: 0.6,0.4)"
    )
    parser.add_argument(
        "--oof-suffix",
        type=str,
        default="base",
        help="OOF 파일 suffix (base 또는 fe)"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="그리드 서치로 최적 블렌딩 비율 탐색"
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="그리드 서치 step (기본값: 0.05)"
    )

    args = parser.parse_args()
    main(args)
