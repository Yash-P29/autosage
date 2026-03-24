"""
AutoSage — Model Evaluation Module
Leaderboard, confusion matrix, classification report, ROC curves.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from config import COLORS, MODEL_REGISTRY
from utils import section_header, metric_card, empty_state, badge


def _render_leaderboard(results: dict, X_test, y_test):
    """Render model leaderboard with test-set metrics."""
    section_header("Model Leaderboard", "Ranked by test accuracy", "🏆")

    rows = []
    for name, res in results.items():
        if res.get("model") is None:
            rows.append({
                "Rank": 0, "Model": name, "Test Accuracy": 0,
                "Precision": 0, "Recall": 0, "F1 Score": 0,
                "CV Score": res["cv_score"], "Train Time (s)": res["train_time"],
                "Status": "❌ Failed",
            })
            continue

        y_pred = res["model"].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        rows.append({
            "Rank": 0, "Model": name, "Test Accuracy": round(acc, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "F1 Score": round(f1, 4), "CV Score": res["cv_score"],
            "Train Time (s)": res["train_time"], "Status": "✅",
        })

    df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    # Top 3 podium
    valid_models = df[df["Status"] == "✅"]
    if len(valid_models) >= 1:
        podium_cols = st.columns(min(3, len(valid_models)))
        medals = ["🥇", "🥈", "🥉"]
        for i, col in enumerate(podium_cols):
            if i < len(valid_models):
                row = valid_models.iloc[i]
                icon = MODEL_REGISTRY.get(row["Model"], {}).get("icon", "🤖")
                with col:
                    metric_card(
                        f'{medals[i]} {row["Model"]}',
                        f'{row["Test Accuracy"]:.2%}',
                        icon=icon,
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Full leaderboard table
    display_df = df[["Rank", "Model", "Test Accuracy", "Precision", "Recall", "F1 Score", "CV Score", "Train Time (s)", "Status"]]
    st.dataframe(
        display_df.style.format({
            "Test Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1 Score": "{:.4f}", "CV Score": "{:.4f}",
        }).background_gradient(subset=["Test Accuracy"], cmap="viridis"),
        use_container_width=True, hide_index=True, height=min(400, 45 * len(df) + 40),
    )

    return df


def _render_confusion_matrices(results: dict, X_test, y_test, label_encoder):
    """Render confusion matrix for each model."""
    section_header("Confusion Matrices", "Per-model prediction breakdown", "🔢")

    valid_models = {k: v for k, v in results.items() if v.get("model") is not None}
    if not valid_models:
        empty_state("No trained models available.", "🤖")
        return

    selected_model = st.selectbox("Select model", list(valid_models.keys()), key="cm_model")
    model = valid_models[selected_model]["model"]
    y_pred = model.predict(X_test)

    class_names = list(label_encoder.classes_) if label_encoder else [str(i) for i in range(len(np.unique(y_test)))]
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm, text_auto=True, x=class_names, y=class_names,
        color_continuous_scale=[[0, "#1A1E2E"], [0.5, COLORS["primary"]], [1, COLORS["secondary"]]],
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text_secondary"],
        margin=dict(t=30, b=50, l=50, r=20), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.markdown(f"**Classification Report — {selected_model}**")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)


def _render_roc_curves(results: dict, X_test, y_test, label_encoder):
    """Render ROC curves (binary or one-vs-rest for multi-class)."""
    section_header("ROC Curves", "Receiver Operating Characteristic", "📉")

    n_classes = len(np.unique(y_test))
    valid_models = {k: v for k, v in results.items() if v.get("model") is not None}

    if not valid_models:
        empty_state("No trained models available.", "🤖")
        return

    if n_classes == 2:
        # Binary ROC
        fig = go.Figure()
        color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                       COLORS["success"], COLORS["warning"], "#E040FB", "#00BCD4", "#FF9800"]

        for i, (name, res) in enumerate(valid_models.items()):
            model = res["model"]
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_prob = model.decision_function(X_test)
                else:
                    continue
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={roc_auc:.3f})",
                    line=dict(color=color_cycle[i % len(color_cycle)], width=2.5),
                ))
            except Exception:
                continue

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color="gray", dash="dash", width=1),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"],
            xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=30, b=50, l=50, r=20), height=450,
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"ROC curves for multi-class ({n_classes} classes) — showing per-model aggregate AUC.")
        # Show AUC bar chart instead for multi-class
        aucs = {}
        for name, res in valid_models.items():
            model = res["model"]
            try:
                if hasattr(model, "predict_proba"):
                    from sklearn.metrics import roc_auc_score
                    y_prob = model.predict_proba(X_test)
                    auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                    aucs[name] = round(auc_val, 4)
            except Exception:
                continue

        if aucs:
            fig = px.bar(
                x=list(aucs.keys()), y=list(aucs.values()),
                color=list(aucs.values()),
                color_continuous_scale=[COLORS["primary"], COLORS["secondary"]],
                labels={"x": "Model", "y": "Weighted AUC"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=COLORS["text_secondary"],
                margin=dict(t=30, b=50, l=50, r=20), height=380,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("Could not compute multi-class AUC for any models.", "📉")


def _render_model_details(results: dict):
    """Show best hyperparameters and training details per model."""
    section_header("Model Details", "Best hyperparameters and training info", "🔧")

    valid_models = {k: v for k, v in results.items() if v.get("model") is not None}
    if not valid_models:
        empty_state("No trained models available.", "🤖")
        return

    for name, res in valid_models.items():
        icon = MODEL_REGISTRY.get(name, {}).get("icon", "🤖")
        with st.expander(f"{icon} {name} — CV Score: {res['cv_score']:.4f} | Time: {res['train_time']}s"):
            st.markdown("**Best Hyperparameters:**")
            params = res.get("best_params", {})
            # Filter out default params for cleaner display
            default_keys = MODEL_REGISTRY.get(name, {}).get("default_params", {}).keys()
            tuned_params = {k: v for k, v in params.items() if k not in default_keys}
            if tuned_params:
                param_df = pd.DataFrame(
                    [{"Parameter": k, "Value": str(v)} for k, v in tuned_params.items()]
                )
                st.dataframe(param_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Default parameters used.")

            if res.get("warning"):
                st.warning(res["warning"])
            if res.get("error"):
                st.error(res["error"])


def _render_model_comparison(df: pd.DataFrame):
    """Render advanced model comparison visualizations."""
    section_header("Model Comparison", "Multi-metric comparison and optimal selection", "📊")
    
    valid_models = df[df["Status"] == "✅"].copy()
    if valid_models.empty:
        empty_state("No successful models to compare.", "🤖")
        return

    opt_goal = st.session_state.get("opt_goal", "Accuracy")
    
    # Map opt_goal to column name
    col_map = {
        "Accuracy": "Test Accuracy",
        "Recall": "Recall",
        "F1 Score": "F1 Score"
    }
    target_metric = col_map.get(opt_goal, "Test Accuracy")
    
    best_row = valid_models.sort_values(target_metric, ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_val = best_row[target_metric]
    
    # Render Best Model Recommendation
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_hover']});
            border:2px solid {COLORS['primary']}66; border-radius:16px;
            padding:24px; text-align:center; margin-bottom: 24px;
            box-shadow:0 8px 32px rgba(108,99,255,0.15);
        ">
            <div style="font-size:0.9rem;color:{COLORS['text_secondary']};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                Recommended Model (Optimized for {opt_goal})
            </div>
            <div style="font-size:2.2rem;font-weight:800;color:{COLORS['primary']};margin-bottom:8px;">
                {best_name}
            </div>
            <div style="font-size:1.1rem;font-weight:600;color:{COLORS['success']};">
                {target_metric}: {best_val:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Metric Comparison (Bar Chart)**")
        melted = valid_models.melt(
            id_vars=["Model"], 
            value_vars=["Test Accuracy", "Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score"
        )
        
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"], margin=dict(t=20, b=50, l=40, r=20),
            legend=dict(font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Multi-Metric Overview (Radar Chart)**")
        top_models = valid_models.head(5)
        
        radar_fig = go.Figure()
        categories = ["Test Accuracy", "Precision", "Recall", "F1 Score"]
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"], COLORS["warning"]]
        
        for i, row in top_models.iterrows():
            values = [row[c] for c in categories]
            radar_fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=row["Model"],
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.6
            ))
            
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(linecolor="rgba(255,255,255,0.1)")
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"], margin=dict(t=30, b=30, l=30, r=30),
            height=350,
            showlegend=True,
            legend=dict(font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(radar_fig, use_container_width=True)


def render_evaluation(results: dict, X_test, y_test, label_encoder):
    """Main entrypoint: render the full AutoML Results tab."""
    if not results:
        empty_state("Train models first to see evaluation results.", "🤖")
        return

    subtabs = st.tabs(["Leaderboard", "Comparison Dashboard", "Confusion Matrix", "ROC Curves", "Model Details"])
    with subtabs[0]:
        df = _render_leaderboard(results, X_test, y_test)
    with subtabs[1]:
        _render_model_comparison(df)
    with subtabs[2]:
        _render_confusion_matrices(results, X_test, y_test, label_encoder)
    with subtabs[3]:
        _render_roc_curves(results, X_test, y_test, label_encoder)
    with subtabs[4]:
        _render_model_details(results)
