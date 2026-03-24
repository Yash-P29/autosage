"""
AutoSage — Preprocessing Pipeline
Automatic imputation, encoding, low-variance removal, feature selection, and train/test split.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from config import VARIANCE_THRESHOLD, K_BEST_DEFAULT, TEST_SIZE, RANDOM_STATE, COLORS
from utils import section_header


def _identify_column_types(df: pd.DataFrame, target_col: str):
    """Identify numeric, categorical, datetime, and ID-like columns."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = []
    categorical_cols = []
    id_cols = []
    date_cols = []

    for col in feature_cols:
        # Check datetime first
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
        elif df[col].dtype == "object":
            # Heuristic for string dates
            sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            if len(sample) >= 10 and sample.count("-") >= 2 and sample[0:4].isdigit():
                try:
                    # check first 5 rows
                    pd.to_datetime(df[col].dropna().head(5), errors="raise")
                    date_cols.append(col)
                    continue
                except:
                    pass
                    
        # Detect ID-like columns: high cardinality + (integer or object with unique values)
        if df[col].nunique() == len(df) and (df[col].dtype in ["int64", "int32"] or df[col].dtype == "object"):
            id_cols.append(col)
        elif df[col].dtype in ["object", "category", "bool"]:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, categorical_cols, id_cols, date_cols


def _impute_missing(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "UNKNOWN")
    return df


def _encode_categoricals(df: pd.DataFrame, categorical_cols: list) -> tuple:
    """One-hot encode categorical features. Returns (df, encoded_col_names)."""
    if not categorical_cols:
        return df, []
    # Only OHE low cardinality features to avoid explosion
    low_card = [c for c in categorical_cols if df[c].nunique() <= 10]
    high_card = [c for c in categorical_cols if c not in low_card]
    
    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=True, dtype=int)
        
    encoded_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in low_card)]
    return df, encoded_cols, high_card


def _remove_low_variance(df: pd.DataFrame, feature_cols: list, threshold: float = VARIANCE_THRESHOLD) -> tuple:
    """Remove features with variance below threshold. Returns (df, removed_cols)."""
    removed = []
    for col in feature_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, np.float32, np.int32, int, float]:
            if df[col].var() < threshold:
                removed.append(col)
    if removed:
        df = df.drop(columns=removed)
    return df, removed


def _select_k_best(X_train, X_test, y_train, feature_names: list, k: int = K_BEST_DEFAULT) -> tuple:
    """Select top-k features using statistical tests."""
    k = min(k, X_train.shape[1])
    if k <= 0 or X_train.shape[1] <= k:
        return X_train, X_test, feature_names

    try:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)
        mask = selector.get_support()
        selected_features = [f for f, m in zip(feature_names, mask) if m]
        return X_train_new, X_test_new, selected_features
    except Exception:
        # Fallback: try mutual information
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_train_new = selector.fit_transform(X_train, y_train)
            X_test_new = selector.transform(X_test)
            mask = selector.get_support()
            selected_features = [f for f, m in zip(feature_names, mask) if m]
            return X_train_new, X_test_new, selected_features
        except Exception:
            return X_train, X_test, feature_names


def preprocess(df: pd.DataFrame, target_col: str, k_features: int = K_BEST_DEFAULT, task_type: str = "Classification", enable_scaling: bool = True) -> dict:
    """
    Full preprocessing pipeline.
    """
    pipeline_info = {"steps": [], "warnings": []}

    # 1. Drop rows where target is null
    initial_rows = len(df)
    df = df.dropna(subset=[target_col])
    dropped_target = initial_rows - len(df)
    if dropped_target > 0:
        pipeline_info["steps"].append(f"Dropped {dropped_target} rows with missing target values")

    # 2. Identify column types
    numeric_cols, categorical_cols, id_cols, date_cols = _identify_column_types(df, target_col)
    pipeline_info["steps"].append(
        f"Identified {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(date_cols)} datetime, {len(id_cols)} ID-like columns"
    )

    # 2.5 Extract Datetime Features
    if date_cols:
        extracted = 0
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            extracted += 4
            numeric_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])
        df = df.drop(columns=date_cols)
        pipeline_info["steps"].append(f"Extracted {extracted} features from {len(date_cols)} datetime columns")

    # 3. Drop ID-like columns
    if id_cols:
        df = df.drop(columns=id_cols)
        pipeline_info["steps"].append(f"Removed ID-like columns: {', '.join(id_cols)}")

    # 4. Encode target variable
    le = None
    if task_type == "Classification":
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str))
        pipeline_info["steps"].append(
            f"Encoded target '{target_col}' → {len(le.classes_)} classes: {list(le.classes_)}"
        )
    else:
        y = df[target_col].values
        pipeline_info["steps"].append(f"Target '{target_col}' used for Regression")

    # 5. Impute missing values
    df = _impute_missing(df, numeric_cols, categorical_cols)
    pipeline_info["steps"].append("Imputed missing values (median for numeric, mode for categorical)")

    # 6. One-hot encode categoricals & Identify High-Cardinality
    df, encoded_cols, high_card_cols = _encode_categoricals(df, categorical_cols)
    if encoded_cols:
        pipeline_info["steps"].append(f"One-hot encoded low-cardinality categoricals → {len(encoded_cols)} new columns")

    # 7. Prepare feature matrix
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    feature_names = feature_cols

    # 8. Train/test split
    if task_type == "Classification":
        unique_classes, counts = np.unique(y, return_counts=True)
        stratify_target = y if (len(unique_classes) > 1 and np.min(counts) >= 2) else None
    else:
        stratify_target = None
        
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_target
    )
    pipeline_info["steps"].append(
        f"Split data: {X_train_df.shape[0]} train / {X_test_df.shape[0]} test ({int(TEST_SIZE*100)}% test)"
    )

    # 9. Advanced: Target Encoding
    if high_card_cols:
        if task_type == "Classification":
            global_mean = np.mean(y_train)
            for col in high_card_cols:
                # Calculate mean target per category on TRAIN set only
                means = pd.Series(y_train).groupby(X_train_df[col].values).mean()
                X_train_df[col] = X_train_df[col].map(means).fillna(global_mean)
                X_test_df[col] = X_test_df[col].map(means).fillna(global_mean)
        else:
            # Fallback to frequency encoding for regression for simplicity
            for col in high_card_cols:
                freq = X_train_df[col].value_counts() / len(X_train_df)
                X_train_df[col] = X_train_df[col].map(freq).fillna(0)
                X_test_df[col] = X_test_df[col].map(freq).fillna(0)
        pipeline_info["steps"].append(f"Target/Frequency encoded {len(high_card_cols)} high-cardinality categoricals")
        
    X_train = X_train_df.values.astype(float)
    X_test = X_test_df.values.astype(float)

    # 10. Advanced: Scaling
    if enable_scaling:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pipeline_info["steps"].append("Standardized numeric features using StandardScaler")

    # 11. Advanced: Polynomial Features (only if few features)
    if X_train.shape[1] > 0 and X_train.shape[1] <= 5:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        feature_names = poly.get_feature_names_out(feature_names).tolist()
        pipeline_info["steps"].append(f"Added Polynomial Features (degree=2), total features now {len(feature_names)}")

    # 12. Remove low-variance features
    df_train_features = pd.DataFrame(X_train, columns=feature_names)
    df_train_features, removed_cols = _remove_low_variance(df_train_features, feature_names)
    if removed_cols:
        pipeline_info["steps"].append(f"Removed {len(removed_cols)} low-variance features")
    X_train = df_train_features.values
    
    # Apply same removal to test
    df_test_features = pd.DataFrame(X_test, columns=feature_names)
    df_test_features = df_test_features.drop(columns=removed_cols, errors="ignore")
    X_test = df_test_features.values
    feature_names = list(df_train_features.columns)

    # 13. Feature selection (SelectKBest)
    if k_features > 0 and X_train.shape[1] > k_features and task_type == "Classification":
        X_train, X_test, feature_names = _select_k_best(X_train, X_test, y_train, feature_names, k_features)
        pipeline_info["steps"].append(f"Selected top {len(feature_names)} features using SelectKBest")

    pipeline_info["final_shape"] = {"train": X_train.shape, "test": X_test.shape}

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": le,
        "pipeline_info": pipeline_info,
    }


def render_pipeline_report(pipeline_info: dict):
    """Display the preprocessing pipeline steps in the UI."""
    section_header("Preprocessing Pipeline", "Steps applied to your dataset", "⚙️")

    for i, step in enumerate(pipeline_info["steps"], 1):
        st.markdown(
            f"""
            <div style="
                display:flex; align-items:center; gap:12px;
                padding:10px 16px; margin-bottom:8px;
                background:{COLORS['bg_card']}; border-radius:10px;
                border-left:3px solid {COLORS['primary']};
            ">
                <span style="
                    background:linear-gradient(135deg,{COLORS['gradient_start']},{COLORS['gradient_end']});
                    color:#fff; width:28px; height:28px; border-radius:50%;
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.8rem; font-weight:700; flex-shrink:0;
                ">{i}</span>
                <span style="color:{COLORS['text_primary']};font-size:0.92rem;">{step}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if pipeline_info.get("final_shape"):
        tr = pipeline_info["final_shape"]["train"]
        te = pipeline_info["final_shape"]["test"]
        st.markdown(
            f"""
            <div style="
                margin-top:16px; padding:14px 20px;
                background:linear-gradient(135deg, {COLORS['primary']}22, {COLORS['secondary']}22);
                border-radius:12px; border:1px solid {COLORS['primary']}33;
            ">
                <span style="color:{COLORS['text_primary']};font-weight:600;">📐 Final Shape:</span>
                <span style="color:{COLORS['text_secondary']};">
                    Train: {tr[0]} × {tr[1]} &nbsp;|&nbsp; Test: {te[0]} × {te[1]}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
