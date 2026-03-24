"""
AutoSage Utilities
Shared helpers for model export, styled components, and reusable UI elements.
"""

import io
import joblib
import streamlit as st
from config import COLORS


def download_model(model, model_name: str):
    """Provide a download button for a trained model as .pkl."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    safe_name = model_name.lower().replace(" ", "_")
    st.download_button(
        label=f"⬇️ Download {model_name} (.pkl)",
        data=buffer,
        file_name=f"autosage_{safe_name}.pkl",
        mime="application/octet-stream",
        use_container_width=True,
    )


def metric_card(label: str, value, icon: str = "📊", delta=None):
    """Render a styled metric card with glassmorphism effect."""
    delta_html = ""
    if delta is not None:
        color = COLORS["success"] if delta >= 0 else COLORS["danger"]
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div style="color:{color};font-size:0.85rem;margin-top:4px;">{arrow} {abs(delta):.2f}</div>'

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['bg_card']}cc, {COLORS['bg_card_hover']}cc);
            border: 1px solid {COLORS['primary']}33;
            border-radius: 16px;
            padding: 20px 24px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 24px rgba(108,99,255,0.10);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        ">
            <div style="font-size:1.8rem;margin-bottom:4px;">{icon}</div>
            <div style="color:{COLORS['text_secondary']};font-size:0.82rem;text-transform:uppercase;letter-spacing:1.2px;">{label}</div>
            <div style="color:{COLORS['text_primary']};font-size:1.7rem;font-weight:700;margin-top:4px;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled section header with gradient underline."""
    sub_html = (
        f'<p style="color:{COLORS["text_secondary"]};font-size:0.95rem;margin:4px 0 0 0;">{subtitle}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom:24px;">
            <h2 style="
                color:{COLORS['text_primary']};
                font-weight:700;
                margin-bottom:0;
                font-size:1.6rem;
            ">{icon} {title}</h2>
            {sub_html}
            <div style="
                height:3px;
                width:80px;
                background: linear-gradient(90deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
                border-radius:2px;
                margin-top:8px;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = None):
    """Render a small colored badge."""
    bg = color or COLORS["primary"]
    return f'<span style="background:{bg};color:#fff;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600;">{text}</span>'


def empty_state(message: str, icon: str = "📂"):
    """Render a styled empty-state placeholder."""
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:60px 20px;
            color:{COLORS['text_secondary']};
        ">
            <div style="font-size:3rem;margin-bottom:12px;">{icon}</div>
            <div style="font-size:1.1rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
