"""
AutoSage — Data Analysis Module
Dataset preview, statistics, missing-value heatmap, correlations, distributions, and health score.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import COLORS
from utils import metric_card, section_header, empty_state


def _compute_health_score(df: pd.DataFrame) -> dict:
    """Compute a dataset health score (0–100) based on quality metrics."""
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = df.isnull().sum().sum() / total_cells * 100 if total_cells > 0 else 0
    duplicate_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0

    # Constant columns (zero variance for numeric, single unique for all)
    constant_cols = 0
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols += 1
    constant_pct = constant_cols / len(df.columns) * 100 if len(df.columns) > 0 else 0

    # Outlier detection (IQR method)
    outlier_cols = 0
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
        outlier_cols = int((outliers > 0).sum())

    # Score calculation: start from 100, deduct penalties
    score = 100.0
    score -= missing_pct * 1.5   # heavy penalty for missing values
    score -= duplicate_pct * 0.5
    score -= constant_pct * 1.0
    score -= (outlier_cols / len(df.columns) * 5.0) if len(df.columns) > 0 else 0
    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "missing_pct": round(missing_pct, 2),
        "duplicate_pct": round(duplicate_pct, 2),
        "constant_cols": constant_cols,
        "outlier_cols": outlier_cols,
        "total_rows": df.shape[0],
        "total_cols": df.shape[1],
    }


def _render_overview(df: pd.DataFrame, health: dict):
    """Render top-level metric cards."""
    cols = st.columns(6)
    with cols[0]:
        color = COLORS["success"] if health["score"] >= 70 else (COLORS["warning"] if health["score"] >= 40 else COLORS["danger"])
        score_display = f'<span style="color:{color}">{health["score"]}%</span>'
        metric_card("Health Score", score_display, icon="💚")
    with cols[1]:
        metric_card("Rows", f'{health["total_rows"]:,}', icon="📏")
    with cols[2]:
        metric_card("Columns", health["total_cols"], icon="📐")
    with cols[3]:
        missing_color = COLORS["danger"] if health["missing_pct"] > 5 else COLORS["success"]
        missing_display = f'<span style="color:{missing_color}">{health["missing_pct"]}%</span>'
        metric_card("Missing", missing_display, icon="🕳️")
    with cols[4]:
        dup_color = COLORS["warning"] if health["duplicate_pct"] > 0 else COLORS["success"]
        dup_display = f'<span style="color:{dup_color}">{health["duplicate_pct"]}%</span>'
        metric_card("Duplicates", dup_display, icon="♻️")
    with cols[5]:
        out_col_color = COLORS["warning"] if health["outlier_cols"] > 0 else COLORS["success"]
        out_display = f'<span style="color:{out_col_color}">{health["outlier_cols"]}</span>'
        metric_card("Outliers", out_display, icon="⚠️")


def _render_preview(df: pd.DataFrame):
    """Dataset head and dtypes."""
    section_header("Dataset Preview", "First rows of your dataset", "")
    st.dataframe(df.head(50), use_container_width=True, height=380)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Feature Types**")
        type_counts = df.dtypes.value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        type_counts["Type"] = type_counts["Type"].astype(str)
        fig = px.pie(
            type_counts, names="Type", values="Count",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]],
            hole=0.4,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"], margin=dict(t=20, b=20, l=20, r=20),
            height=280, legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Column Info**")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Unique": df.nunique().values,
            "Missing %": (df.isnull().sum() / len(df) * 100).round(1).values,
        })
        st.dataframe(info_df, use_container_width=True, height=280, hide_index=True)


def _render_statistics(df: pd.DataFrame):
    """Descriptive statistics for numeric columns."""
    section_header("Descriptive Statistics", "Summary statistics for numerical features", "")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        empty_state("No numerical columns found.", "🔢")
        return
    st.dataframe(numeric_df.describe().T.round(3), use_container_width=True, height=350)


def _render_missing_values(df: pd.DataFrame):
    """Missing value heatmap."""
    section_header("Missing Values", "Heatmap of null values across the dataset", "")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("No missing values detected. Your dataset is complete.")
        return

    # Bar chart of missing counts
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=missing.index, y=missing.values,
        marker=dict(
            color=missing.values,
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["accent"]]],
            line=dict(width=0),
        ),
        text=[f"{v} ({v/len(df)*100:.1f}%)" for v in missing.values],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_secondary"]),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text_secondary"],
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Count"),
        margin=dict(t=30, b=80, l=50, r=20), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Missing value heatmap (matrix view)
    missing_matrix = df[missing.index].isnull().astype(int)
    sample_size = min(200, len(missing_matrix))
    sample = missing_matrix.sample(sample_size, random_state=42) if len(missing_matrix) > sample_size else missing_matrix
    fig2 = px.imshow(
        sample.T, color_continuous_scale=["#1A1E2E", COLORS["accent"]],
        labels=dict(x="Row Index", y="Feature", color="Missing"),
        aspect="auto",
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text_secondary"],
        margin=dict(t=20, b=40, l=20, r=20), height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_correlations(df: pd.DataFrame):
    """Correlation heatmap for numeric features."""
    section_header("Correlation Matrix", "Pairwise correlations between numerical features", "")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        empty_state("Need at least 2 numerical columns for correlation analysis.", "🔢")
        return

    corr = numeric_df.corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=["#6C63FF", "#0E1117", "#FF6584"],
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text_secondary"],
        margin=dict(t=20, b=20, l=20, r=20),
        height=max(350, min(600, numeric_df.shape[1] * 35)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_distributions(df: pd.DataFrame):
    """Distribution plots for numeric and categorical features."""
    section_header("Feature Distributions", "Explore the distribution of individual features", "")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        empty_state("No columns to visualize.", "📉")
        return

    col_options = numeric_cols + categorical_cols
    selected = st.selectbox("Select a feature", col_options, key="dist_feature")

    if selected in numeric_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[selected].dropna(), nbinsx=40,
            marker=dict(color=COLORS["primary"], line=dict(width=0.5, color=COLORS["bg_dark"])),
            opacity=0.85,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"],
            xaxis=dict(showgrid=False, title=selected),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Count"),
            margin=dict(t=20, b=50, l=50, r=20), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        vc = df[selected].value_counts().head(20)
        fig = px.bar(
            x=vc.index.astype(str), y=vc.values,
            color=vc.values, color_continuous_scale=[COLORS["primary"], COLORS["secondary"]],
            labels={"x": selected, "y": "Count"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"],
            margin=dict(t=20, b=50, l=50, r=20), height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_data_explorer(df: pd.DataFrame):
    """Main entrypoint: render the full Data Explorer tab."""
    if df is None or df.empty:
        empty_state("Upload a dataset to get started.", "📂")
        return

    health = _compute_health_score(df)
    _render_overview(df, health)

    st.markdown("<br>", unsafe_allow_html=True)

    subtabs = st.tabs(["Preview", "Statistics", "Missing Values", "Correlations", "Distributions"])
    with subtabs[0]:
        _render_preview(df)
    with subtabs[1]:
        _render_statistics(df)
    with subtabs[2]:
        _render_missing_values(df)
    with subtabs[3]:
        _render_correlations(df)
    with subtabs[4]:
        _render_distributions(df)
