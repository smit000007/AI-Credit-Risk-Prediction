"""Stratified data splitting, scaling, and class imbalance handling (SMOTE & class weights)."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.feature_engineering import CreditFeatureEngineer


@dataclass
class PreprocessedData:
    """Container for train/test splits, feature engineer, and scaler."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    df_train_full: pd.DataFrame
    df_test_full: pd.DataFrame
    feature_engineer: CreditFeatureEngineer
    scaler: StandardScaler
    scale_pos_weight: float


def prepare_train_test_data(
    df_raw: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> PreprocessedData:
    """
    Execute stratified train/test split, fit feature engineering without data leakage,
    scale features, and handle class imbalance via SMOTE or class weights.
    """
    target_col = "SeriousDlqin2yrs"
    feature_cols = config.engineered_feature_columns

    # 1. Stratified split on raw data first to avoid any imputation leakage
    df_train_raw, df_test_raw = train_test_split(
        df_raw,
        test_size=config.test_size,
        stratify=df_raw[target_col],
        random_state=config.random_state,
    )

    # 2. Fit feature engineer on training set, transform both train and test
    fe = CreditFeatureEngineer()
    df_train_clean = fe.fit_transform(df_train_raw)
    df_test_clean = fe.transform(df_test_raw)

    X_train = df_train_clean[feature_cols].copy()
    y_train = df_train_clean[target_col].astype(int).copy()
    X_test = df_test_clean[feature_cols].copy()
    y_test = df_test_clean[target_col].astype(int).copy()

    # Compute positive class weight ratio (negatives / positives)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / n_pos)

    # 3. Handle class imbalance on the training split
    if config.imbalance_strategy.lower() == "smote":
        smote = SMOTE(sampling_strategy=0.35, random_state=config.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_train_resampled = pd.DataFrame(X_res, columns=feature_cols)
        y_train_resampled = pd.Series(y_res, name=target_col)
    else:
        X_train_resampled = X_train.copy()
        y_train_resampled = y_train.copy()

    # 4. Fit StandardScaler for linear models (Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_resampled),
        columns=feature_cols,
        index=X_train_resampled.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        df_train_full=df_train_clean,
        df_test_full=df_test_clean,
        feature_engineer=fe,
        scaler=scaler,
        scale_pos_weight=scale_pos_weight,
    )
