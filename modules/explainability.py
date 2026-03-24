"""
AutoSage — Explainable AI Module
SHAP-based feature importance and beeswarm plots.
"""

import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
from config import COLORS
from utils import section_header, empty_state

matplotlib.use("Agg")


def _get_shap_explainer(model, X_sample):
    """Create appropriate SHAP explainer based on model type."""
    model_class_name = type(model).__name__

    try:
        # Tree-based models
        if model_class_name in [
            "RandomForestClassifier", "GradientBoostingClassifier",
            "ExtraTreesClassifier", "DecisionTreeClassifier", "XGBClassifier",
        ]:
            explainer = shap.TreeExplainer(model)
            return explainer
    except Exception:
        pass

    try:
        # Linear models
        if model_class_name in ["LogisticRegression"]:
            explainer = shap.LinearExplainer(model, X_sample)
            return explainer
    except Exception:
        pass

    # Fallback: KernelExplainer (works for any model but is slower)
    try:
        # Use a small background sample for speed
        bg_size = min(50, len(X_sample))
        background = shap.sample(X_sample, bg_size) if len(X_sample) > bg_size else X_sample
        explainer = shap.KernelExplainer(model.predict_proba, background)
        return explainer
    except Exception:
        try:
            background = shap.sample(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
            return explainer
        except Exception:
            return None


def _compute_shap_values(explainer, X_sample):
    """Compute SHAP values, handling multi-class gracefully."""
    try:
        shap_values = explainer.shap_values(X_sample)
        return shap_values
    except Exception:
        try:
            shap_values = explainer(X_sample)
            return shap_values
        except Exception:
            return None


def render_shap(model, X_test, feature_names, model_name, label_encoder=None):
    """Render SHAP explainability plots."""
    if model is None:
        empty_state("No model available for explanation.", "🔍")
        return

    section_header("SHAP Explainability", f"Understanding {model_name}'s predictions", "🔍")

    # Use a sample for performance
    max_samples = 200
    if len(X_test) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[indices] if isinstance(X_test, np.ndarray) else X_test.iloc[indices]
    else:
        X_sample = X_test

    X_df = pd.DataFrame(X_sample, columns=feature_names) if not isinstance(X_sample, pd.DataFrame) else X_sample

    with st.spinner("🔍 Computing SHAP values... This may take a moment."):
        explainer = _get_shap_explainer(model, X_df.values if hasattr(X_df, 'values') else X_df)

        if explainer is None:
            st.error("⚠️ Could not create SHAP explainer for this model type.")
            return

        shap_values = _compute_shap_values(explainer, X_df.values if hasattr(X_df, 'values') else X_df)

        if shap_values is None:
            st.error("⚠️ Could not compute SHAP values for this model.")
            return

    # Handle different SHAP value formats
    if isinstance(shap_values, shap.Explanation):
        shap_vals_array = shap_values.values
    elif isinstance(shap_values, list):
        # Multi-class: list of arrays
        shap_vals_array = shap_values
    else:
        shap_vals_array = shap_values

    # Create sub-tabs for different SHAP views
    shap_tabs = st.tabs(["Feature Importance", "Beeswarm Plot", "Partial Dependence Plot"])

    with shap_tabs[0]:
        _render_feature_importance(shap_vals_array, feature_names, X_df)

    with shap_tabs[1]:
        _render_beeswarm(shap_vals_array, feature_names, X_df)
        
    with shap_tabs[2]:
        _render_pdp(model, X_df, feature_names)


def _render_feature_importance(shap_values, feature_names, X_df):
    """Render global feature importance bar chart."""
    st.markdown(f"##### Global Feature Importance")
    st.caption("Mean absolute SHAP value — shows each feature's overall impact on predictions")

    try:
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            if shap_values.ndim == 3:
                abs_vals = np.mean(np.abs(shap_values), axis=(0, 2))
            else:
                abs_vals = np.mean(np.abs(shap_values), axis=0)

        if abs_vals.ndim > 1:
            abs_vals = np.mean(abs_vals, axis=-1)

        importance = pd.DataFrame({
            "Feature": feature_names[:len(abs_vals)],
            "Importance": abs_vals[:len(feature_names)],
        }).sort_values("Importance", ascending=True).tail(20)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance["Importance"].values,
            y=importance["Feature"].values,
            orientation="h",
            marker=dict(
                color=importance["Importance"].values,
                colorscale=[[0, COLORS["primary"]], [1, COLORS["secondary"]]],
                line=dict(width=0),
            ),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text_secondary"],
            xaxis=dict(title="Mean |SHAP Value|", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=False),
            margin=dict(t=20, b=40, l=20, r=20),
            height=max(350, min(600, len(importance) * 28)),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render feature importance: {str(e)[:100]}")


def _render_beeswarm(shap_values, feature_names, X_df):
    """Render SHAP beeswarm plot using matplotlib."""
    st.markdown(f"##### Beeswarm Plot")
    st.caption("Each dot is a data point — color shows feature value, position shows SHAP impact")

    try:
        fig, ax = plt.subplots(figsize=(10, max(4, min(8, len(feature_names) * 0.35))))

        # Style the matplotlib figure for dark theme
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="#A0AEC0")
        ax.xaxis.label.set_color("#A0AEC0")
        ax.yaxis.label.set_color("#A0AEC0")
        for spine in ax.spines.values():
            spine.set_color("#2D3748")

        if isinstance(shap_values, list):
            # Multi-class: use the first class or average
            sv = shap_values[0] if len(shap_values) > 0 else shap_values
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            sv = shap_values[:, :, 0]
        else:
            sv = shap_values

        explanation = shap.Explanation(
            values=sv[:, :len(feature_names)] if sv.shape[1] > len(feature_names) else sv,
            data=X_df.values[:sv.shape[0], :sv.shape[1]],
            feature_names=feature_names[:sv.shape[1]],
        )

        shap.plots.beeswarm(explanation, max_display=15, show=False, plot_size=None)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Could not render beeswarm plot: {str(e)[:120]}")
        st.info("Tip: Beeswarm plots work best with tree-based models (Random Forest, XGBoost, etc.).")


def _render_pdp(model, X_df, feature_names):
    """Render Partial Dependence Plot (PDP) for selected features."""
    st.markdown("##### Partial Dependence Plot (PDP)")
    st.caption("Shows the marginal effect of a feature on the predicted outcome.")
    
    top_features = st.multiselect(
        "Select features to analyze",
        options=feature_names,
        default=[feature_names[0]] if len(feature_names) > 0 else [],
        max_selections=2,
        help="Select up to 2 features to analyze their partial dependence."
    )
    
    if top_features:
        try:
            from sklearn.inspection import PartialDependenceDisplay
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.tick_params(colors="#A0AEC0")
            for spine in ax.spines.values():
                spine.set_color("#2D3748")
                
            PartialDependenceDisplay.from_estimator(
                model, X_df.values if hasattr(X_df, 'values') else X_df, 
                features=[feature_names.index(f) for f in top_features], 
                feature_names=feature_names,
                ax=ax, grid_resolution=20
            )
            
            # Make text white for dark theme
            ax.xaxis.label.set_color("#A0AEC0")
            ax.yaxis.label.set_color("#A0AEC0")
            
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"Could not generate PDP for this model/feature combination. {e}")
