"""
AutoSage — Prediction Module
Interactive single-row prediction with confidence display.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import COLORS
from utils import section_header, empty_state


def render_prediction(best_model, feature_names: list, label_encoder, model_name: str = "Best Model", X_train=None):
    """Render real-time prediction interface."""
    if best_model is None:
        empty_state("Train models first to make predictions.", "🎯")
        return

    section_header("Real-Time Prediction", f"Using {model_name}", "🎯")

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg, {COLORS['primary']}15, {COLORS['secondary']}15);
            border:1px solid {COLORS['primary']}33; border-radius:12px;
            padding:14px 20px; margin-bottom:20px;
        ">
            <span style="color:{COLORS['text_secondary']};font-size:0.9rem;">
                Enter feature values below and click <b>Predict</b> to get a classification result with confidence scores.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic input form
    n_cols = min(3, len(feature_names))
    input_values = {}

    with st.form("prediction_form"):
        cols = st.columns(n_cols)
        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                input_values[feat] = st.number_input(
                    feat, value=0.0, format="%.4f", key=f"pred_{feat}"
                )

        submitted = st.form_submit_button(
            "🚀 Predict",
            use_container_width=True,
        )

    if submitted:
        # Prepare input
        input_array = np.array([[input_values[f] for f in feature_names]])

        try:
            prediction = best_model.predict(input_array)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0] if label_encoder else str(prediction)

            # Confidence
            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(input_array)[0]
                confidence = float(np.max(proba))
                class_probas = {
                    (label_encoder.inverse_transform([i])[0] if label_encoder else str(i)): round(float(p), 4)
                    for i, p in enumerate(proba)
                }
            else:
                confidence = None
                class_probas = None

            # Render result
            st.markdown("<br>", unsafe_allow_html=True)

            result_cols = st.columns([2, 3])

            with result_cols[0]:
                conf_color = COLORS["success"] if (confidence and confidence >= 0.7) else (COLORS["warning"] if confidence else COLORS["text_secondary"])
                st.markdown(
                    f"""
                    <div style="
                        background:linear-gradient(135deg, {COLORS['bg_card']}, {COLORS['bg_card_hover']});
                        border:2px solid {COLORS['primary']}66; border-radius:16px;
                        padding:30px; text-align:center;
                        box-shadow:0 8px 32px rgba(108,99,255,0.15);
                    ">
                        <div style="font-size:0.85rem;color:{COLORS['text_secondary']};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">Predicted Class</div>
                        <div style="font-size:2rem;font-weight:800;color:{COLORS['primary']};margin-bottom:12px;">{predicted_label}</div>
                        {"" if confidence is None else f'<div style="font-size:0.85rem;color:{COLORS["text_secondary"]};">Confidence</div><div style="font-size:1.4rem;font-weight:700;color:{conf_color};">{confidence:.1%}</div>'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_cols[1]:
                if class_probas:
                    st.markdown(f"**Class Probabilities**")
                    sorted_probas = sorted(class_probas.items(), key=lambda x: x[1], reverse=True)
                    labels = [p[0] for p in sorted_probas]
                    values = [p[1] for p in sorted_probas]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=values, y=labels, orientation="h",
                        marker=dict(
                            color=values,
                            colorscale=[[0, COLORS["bg_card_hover"]], [0.5, COLORS["primary"]], [1, COLORS["secondary"]]],
                            line=dict(width=0),
                        ),
                        text=[f"{v:.1%}" for v in values],
                        textposition="inside",
                        textfont=dict(color="white", size=12),
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color=COLORS["text_secondary"],
                        xaxis=dict(showgrid=False, range=[0, 1], title="Probability"),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                        margin=dict(t=10, b=40, l=20, r=20), height=max(200, len(labels) * 40 + 60),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Local Explainability
            st.markdown("---")
            if st.button("🔍 Explain this prediction (SHAP Waterfall)", key="shap_local_btn"):
                from modules.explainability import _get_shap_explainer
                with st.spinner("Generating local explanation..."):
                    explainer = _get_shap_explainer(best_model, X_train if X_train is not None else input_array)
                    if explainer:
                        try:
                            import shap
                            import matplotlib.pyplot as plt
                            
                            shap_val = explainer(input_array)
                            sv = shap_val[0] if isinstance(shap_val, list) else shap_val
                            if hasattr(sv, 'shape') and len(sv.shape) == 3:
                                sv = sv[:,:,0] # Take first class
                                
                            fig, ax = plt.subplots(figsize=(8, 4))
                            fig.patch.set_facecolor("#0E1117")
                            
                            shap.plots.waterfall(sv[0], show=False)
                            st.pyplot(fig, use_container_width=False)
                            plt.close()
                        except Exception as e:
                            st.warning(f"Could not generate waterfall plot: {e}. Tip: Local explanation works best with tree-based models.")
                    else:
                        st.warning("Model type not supported for local SHAP explanation.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Make sure your input values are consistent with the training data format.")
