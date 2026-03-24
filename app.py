"""
AutoSage — Intelligent AutoML with Explainable AI
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoSage — Intelligent AutoML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Imports (after page config)
# ──────────────────────────────────────────────
from config import COLORS, MODEL_REGISTRY
from utils import download_model, section_header, empty_state
from modules.data_analysis import render_data_explorer
from modules.preprocessing import preprocess, render_pipeline_report
from modules.automl_engine import train_models, get_best_model
from modules.evaluation import render_evaluation
from modules.explainability import render_shap
from modules.prediction import render_prediction

# ──────────────────────────────────────────────
# Custom CSS — Clean Dark Theme
# ──────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ═══════════════════════════════════════
           GLOBAL
           ═══════════════════════════════════════ */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {COLORS['text_primary']};
        }}
        .stApp {{
            background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, #0B0E14 100%);
        }}
        p, span, label, .stMarkdown {{
            color: {COLORS['text_primary']};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS['text_primary']} !important;
        }}

        /* Hide sidebar completely */
        section[data-testid="stSidebar"] {{ display: none !important; }}
        button[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

        /* ═══════════════════════════════════════
           TABS
           ═══════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {COLORS['bg_card']};
            border: 1px solid rgba(108,99,255,0.12);
            border-radius: 14px;
            padding: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            color: {COLORS['text_secondary']};
            font-weight: 500;
            padding: 10px 22px;
            transition: all 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {COLORS['text_primary']};
            background: rgba(108,99,255,0.08);
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']}) !important;
            color: white !important;
            font-weight: 600;
            box-shadow: 0 2px 12px rgba(108,99,255,0.3);
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 20px;
        }}

        /* ═══════════════════════════════════════
           BUTTONS
           ═══════════════════════════════════════ */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
            color: white !important;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 12px 28px;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(108,99,255,0.2);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(108,99,255,0.4);
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}
        .stButton > button:disabled {{
            opacity: 0.4;
            transform: none;
            box-shadow: none;
        }}
        .stDownloadButton > button {{
            background: linear-gradient(135deg, {COLORS['success']}, #00C853);
            color: white !important;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            box-shadow: 0 2px 10px rgba(0,230,118,0.2);
        }}
        .stDownloadButton > button:hover {{
            box-shadow: 0 6px 24px rgba(0,230,118,0.35);
        }}

        /* ═══════════════════════════════════════
           FILE UPLOADER
           ═══════════════════════════════════════ */
        [data-testid="stFileUploader"] {{
            background: {COLORS['bg_card']};
            border: 2px dashed rgba(108,99,255,0.3);
            border-radius: 14px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: rgba(108,99,255,0.5);
            background: {COLORS['bg_card_hover']};
        }}
        [data-testid="stFileUploader"] section {{
            padding: 0 !important;
        }}
        [data-testid="stFileUploader"] section > button {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']}) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 8px 20px !important;
        }}
        [data-testid="stFileUploader"] small {{
            color: {COLORS['text_secondary']} !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: transparent !important;
            color: {COLORS['text_secondary']} !important;
        }}
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] small {{
            color: {COLORS['text_secondary']} !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] span {{
            color: {COLORS['text_secondary']} !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] div {{
            color: {COLORS['text_secondary']} !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] small {{
            color: {COLORS['text_secondary']} !important;
        }}

        /* ═══════════════════════════════════════
           DATAFRAMES & TABLES (GlideDataEditor)
           ═══════════════════════════════════════ */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(108,99,255,0.15);
        }}

        /* ═══════════════════════════════════════
           BROWSE FILES & FORM SUBMIT BUTTONS
           ═══════════════════════════════════════ */
        /* Browse files button inside file uploader (extremely specific to override streamlit's inline styles) */
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button,
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"],
        .stFileUploader button {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']}) !important;
            background-color: transparent !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 8px 20px !important;
            box-shadow: 0 2px 8px rgba(108,99,255,0.2) !important;
        }}
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploaderDropzone"] button * {{
            color: white !important;
        }}
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover,
        button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {{
            box-shadow: 0 4px 16px rgba(108,99,255,0.35) !important;
            border: none !important;
            color: white !important;
        }}
        /* Form submit button & Predict button */
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFormSubmitButton"] button *,
        .stForm button[type="submit"],
        .stForm .stButton > button,
        button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']}) !important;
            background-color: transparent !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }}
        /* Any remaining secondary/tertiary buttons */
        button[kind="tertiary"],
        [data-testid="baseButton-minimal"] {{
            background: {COLORS['bg_card_hover']} !important;
            background-color: {COLORS['bg_card_hover']} !important;
            color: {COLORS['text_primary']} !important;
            border: 1px solid rgba(108,99,255,0.25) !important;
        }}

        /* ═══════════════════════════════════════
           GLOBAL WHITE BACKGROUND KILLER
           ═══════════════════════════════════════ */
        /* Catch any Streamlit widget that still renders white */
        .element-container > div > div {{
            color: {COLORS['text_primary']};
        }}
        /* Markdown bold text */
        strong, b {{
            color: {COLORS['text_primary']};
        }}
        /* Markdown paragraphs */
        .stMarkdown p {{
            color: {COLORS['text_primary']};
        }}
        /* Links */
        a {{
            color: {COLORS['secondary']} !important;
        }}
        a:hover {{
            color: {COLORS['primary']} !important;
        }}
        /* Full-width bottom toolbar/bar */
        [data-testid="stBottomBlockContainer"] {{
            background: {COLORS['bg_dark']} !important;
        }}
        /* Main block container */
        [data-testid="stMainBlockContainer"] {{
            color: {COLORS['text_primary']};
        }}
        /* Column containers */
        [data-testid="column"] {{
            color: {COLORS['text_primary']};
        }}

        /* ═══════════════════════════════════════
           SELECTBOX / DROPDOWN
           ═══════════════════════════════════════ */
        [data-baseweb="select"] > div {{
            background: {COLORS['bg_card']} !important;
            border: 1px solid rgba(108,99,255,0.2) !important;
            border-radius: 10px !important;
            color: {COLORS['text_primary']} !important;
        }}
        [data-baseweb="select"] > div:hover {{
            border-color: rgba(108,99,255,0.4) !important;
        }}
        [data-baseweb="select"] span {{
            color: {COLORS['text_primary']} !important;
        }}
        /* Dropdown menu */
        [data-baseweb="popover"] {{
            background: {COLORS['bg_card']} !important;
            border: 1px solid rgba(108,99,255,0.2) !important;
            border-radius: 10px !important;
        }}
        [data-baseweb="popover"] ul {{
            background: {COLORS['bg_card']} !important;
        }}
        [data-baseweb="popover"] li {{
            color: {COLORS['text_primary']} !important;
            background: transparent !important;
        }}
        [data-baseweb="popover"] li:hover {{
            background: rgba(108,99,255,0.15) !important;
        }}
        [role="listbox"] {{
            background: {COLORS['bg_card']} !important;
        }}
        [role="option"] {{
            color: {COLORS['text_primary']} !important;
        }}
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {{
            background: rgba(108,99,255,0.15) !important;
        }}

        /* ═══════════════════════════════════════
           CHECKBOX
           ═══════════════════════════════════════ */
        [data-testid="stCheckbox"] label span {{
            color: {COLORS['text_primary']} !important;
        }}
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] {{
            background: {COLORS['bg_card']} !important;
            border-color: rgba(108,99,255,0.3) !important;
            border-radius: 6px !important;
        }}
        [data-testid="stCheckbox"] [data-baseweb="checkbox"]:hover {{
            border-color: {COLORS['primary']} !important;
        }}
        [data-testid="stCheckbox"] [data-baseweb="checkbox"][aria-checked="true"] {{
            background: {COLORS['primary']} !important;
            border-color: {COLORS['primary']} !important;
        }}

        /* ═══════════════════════════════════════
           SLIDER
           ═══════════════════════════════════════ */
        [data-testid="stSlider"] label {{
            color: {COLORS['text_primary']} !important;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
            background: {COLORS['primary']} !important;
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 8px rgba(108,99,255,0.4);
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] {{
            background: rgba(108,99,255,0.15) !important;
        }}
        .stSlider > div > div > div {{
            color: {COLORS['text_primary']} !important;
        }}

        /* ═══════════════════════════════════════
           NUMBER INPUT
           ═══════════════════════════════════════ */
        [data-testid="stNumberInput"] input {{
            background: {COLORS['bg_card']} !important;
            color: {COLORS['text_primary']} !important;
            border: 1px solid rgba(108,99,255,0.2) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stNumberInput"] input:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
        }}
        [data-testid="stNumberInput"] button {{
            background: {COLORS['bg_card']} !important;
            color: {COLORS['text_primary']} !important;
            border-color: rgba(108,99,255,0.2) !important;
        }}

        /* ═══════════════════════════════════════
           TEXT INPUT / TEXT AREA
           ═══════════════════════════════════════ */
        .stTextInput input, .stTextArea textarea {{
            background: {COLORS['bg_card']} !important;
            color: {COLORS['text_primary']} !important;
            border: 1px solid rgba(108,99,255,0.2) !important;
            border-radius: 10px !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
        }}

        /* ═══════════════════════════════════════
           FORM
           ═══════════════════════════════════════ */
        .stForm {{
            background: {COLORS['bg_card']};
            border: 1px solid rgba(108,99,255,0.15);
            border-radius: 14px;
            padding: 24px;
        }}

        /* ═══════════════════════════════════════
           EXPANDER
           ═══════════════════════════════════════ */
        [data-testid="stExpander"] {{
            background: {COLORS['bg_card']};
            border: 1px solid rgba(108,99,255,0.12);
            border-radius: 12px;
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{
            color: {COLORS['text_primary']} !important;
        }}
        [data-testid="stExpander"] summary span {{
            color: {COLORS['text_primary']} !important;
        }}
        .streamlit-expanderHeader {{
            background: {COLORS['bg_card']};
            border-radius: 12px;
            color: {COLORS['text_primary']} !important;
        }}

        /* ═══════════════════════════════════════
           PROGRESS BAR
           ═══════════════════════════════════════ */
        .stProgress > div > div {{
            background: rgba(108,99,255,0.15) !important;
            border-radius: 8px;
        }}
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {COLORS['gradient_start']}, {COLORS['gradient_end']}) !important;
            border-radius: 8px;
        }}

        /* ═══════════════════════════════════════
           ALERTS (success, warning, error, info)
           ═══════════════════════════════════════ */
        [data-testid="stAlert"] {{
            background: {COLORS['bg_card']} !important;
            border-radius: 10px !important;
            color: {COLORS['text_primary']} !important;
        }}
        .stSuccess {{
            border-left: 4px solid {COLORS['success']} !important;
        }}
        .stWarning {{
            border-left: 4px solid {COLORS['warning']} !important;
        }}
        .stError {{
            border-left: 4px solid {COLORS['danger']} !important;
        }}
        [data-testid="stAlert"] p {{
            color: {COLORS['text_primary']} !important;
        }}

        /* ═══════════════════════════════════════
           CAPTIONS & SMALL TEXT
           ═══════════════════════════════════════ */
        .stCaption, [data-testid="stCaption"] {{
            color: {COLORS['text_secondary']} !important;
        }}
        small {{
            color: {COLORS['text_secondary']} !important;
        }}

        /* ═══════════════════════════════════════
           TOOLTIP
           ═══════════════════════════════════════ */
        [data-baseweb="tooltip"] {{
            background: {COLORS['bg_card']} !important;
            border: 1px solid rgba(108,99,255,0.2) !important;
            border-radius: 8px !important;
            color: {COLORS['text_primary']} !important;
        }}

        /* ═══════════════════════════════════════
           SPINNER
           ═══════════════════════════════════════ */
        [data-testid="stSpinner"] {{
            color: {COLORS['text_primary']} !important;
        }}

        /* ═══════════════════════════════════════
           HERO HEADER
           ═══════════════════════════════════════ */
        .hero-header {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}14, {COLORS['gradient_end']}14);
            border: 1px solid rgba(108,99,255,0.18);
            border-radius: 18px;
            padding: 28px 36px;
            margin-bottom: 28px;
            display: flex;
            align-items: center;
            gap: 20px;
            position: relative;
            overflow: hidden;
        }}
        .hero-header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 20% 50%, rgba(108,99,255,0.06), transparent 60%),
                        radial-gradient(ellipse at 80% 50%, rgba(0,210,255,0.06), transparent 60%);
            pointer-events: none;
        }}
        .hero-header h1 {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0;
            position: relative;
        }}
        .hero-header p {{
            color: {COLORS['text_secondary']};
            font-size: 0.9rem;
            margin: 4px 0 0 0;
            position: relative;
        }}

        /* ═══════════════════════════════════════
           ANIMATIONS
           ═══════════════════════════════════════ */
        @keyframes pulse {{
            0% {{ opacity: 0.6; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.6; }}
        }}
        .training-pulse {{
            animation: pulse 1.5s ease-in-out infinite;
        }}
        @keyframes glow {{
            0%, 100% {{ box-shadow: 0 0 8px rgba(108,99,255,0.15); }}
            50% {{ box-shadow: 0 0 20px rgba(108,99,255,0.25); }}
        }}

        /* ═══════════════════════════════════════
           CARDS
           ═══════════════════════════════════════ */
        .info-card {{
            background: linear-gradient(145deg, {COLORS['bg_card']}, {COLORS['bg_card_hover']});
            border: 1px solid rgba(108,99,255,0.12);
            border-radius: 14px;
            padding: 20px 24px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }}
        .info-card:hover {{
            border-color: rgba(108,99,255,0.25);
            box-shadow: 0 4px 20px rgba(108,99,255,0.1);
            transform: translateY(-2px);
        }}

        /* ═══════════════════════════════════════
           SCROLLBAR
           ═══════════════════════════════════════ */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: {COLORS['bg_dark']}; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(108,99,255,0.3); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(108,99,255,0.5); }}

        /* ═══════════════════════════════════════
           METRIC / LABEL OVERRIDES
           ═══════════════════════════════════════ */
        [data-testid="stMetricValue"] {{
            color: {COLORS['text_primary']} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_secondary']} !important;
        }}
        label[data-testid="stWidgetLabel"] {{
            color: {COLORS['text_primary']} !important;
        }}
        .stSelectbox label, .stSlider label, .stCheckbox label,
        .stNumberInput label, .stTextInput label, .stTextArea label,
        .stFileUploader label {{
            color: {COLORS['text_primary']} !important;
        }}

        /* ═══════════════════════════════════════
           DIVIDERS
           ═══════════════════════════════════════ */
        hr {{
            border-color: rgba(108,99,255,0.1) !important;
        }}
        [data-testid="stDivider"] {{
            border-color: rgba(108,99,255,0.1) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
defaults = {
    "df": None,
    "preprocessed": None,
    "results": None,
    "best_model_name": None,
    "training_complete": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

col_logo, col_title = st.columns([0.06, 0.94])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=52)
with col_title:
    st.markdown(
        f"""
        <div style="padding-top:4px;">
            <span style="
                background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.6rem;
                font-weight: 800;
            ">AutoSage</span>
            <span style="color:{COLORS['text_secondary']};font-size:0.85rem;margin-left:12px;">
                Intelligent AutoML · Explainable AI
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────
if st.session_state.df is None:
    # Before dataset upload — show upload area + feature cards
    st.markdown(
        f"""
        <div class="hero-header">
            <div>
                <h1>AutoSage</h1>
                <p>Automated Machine Learning · Model Selection · Hyperparameter Tuning · Explainable AI</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    features = [
        ("Smart Analysis", "Automated dataset profiling with health scores and rich visualizations"),
        ("AutoML Engine", "Train and compare 8 models with hyperparameter tuning"),
        ("Explainable AI", "SHAP-powered model transparency and feature insights"),
        ("Live Predictions", "Real-time inference with confidence scores"),
    ]
    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="info-card" style="text-align:center;height:140px;display:flex;flex-direction:column;justify-content:center;">
                    <div style="color:{COLORS['text_primary']};font-weight:600;font-size:0.95rem;margin-bottom:6px;">{title}</div>
                    <div style="color:{COLORS['text_secondary']};font-size:0.8rem;line-height:1.4;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload area
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:12px;">
            <span style="color:{COLORS['text_primary']};font-size:1.1rem;font-weight:600;">Upload a dataset to get started</span><br>
            <span style="color:{COLORS['text_secondary']};font-size:0.85rem;">Supports CSV files up to 200MB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")

else:
    df = st.session_state.df

    # Dataset info bar
    st.markdown(
        f"""
        <div style="
            background:{COLORS['bg_card']};border:1px solid {COLORS['primary']}15;
            border-radius:10px;padding:8px 16px;margin-bottom:12px;
            display:flex;align-items:center;gap:16px;
        ">
            <span style="color:{COLORS['text_primary']};font-weight:600;font-size:0.88rem;">
                Dataset loaded
            </span>
            <span style="color:{COLORS['text_secondary']};font-size:0.82rem;">
                {df.shape[0]:,} rows · {df.shape[1]} columns
            </span>
            <span style="color:{COLORS['text_secondary']};font-size:0.82rem;">
                {df.select_dtypes(include=[np.number]).shape[1]} numeric · {df.select_dtypes(exclude=[np.number]).shape[1]} categorical
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_data, tab_train, tab_results, tab_xai, tab_predict, tab_experiments = st.tabs([
        "Data Explorer",
        "Train",
        "Results",
        "Explainability",
        "Predict",
        "Experiments"
    ])

    # ──────────────────────────────────
    # TAB 1: Data Explorer
    # ──────────────────────────────────
    with tab_data:
        render_data_explorer(df)
        if st.session_state.preprocessed:
            st.markdown("<br>", unsafe_allow_html=True)
            render_pipeline_report(st.session_state.preprocessed["pipeline_info"])

    # ──────────────────────────────────
    # TAB 2: Train
    # ──────────────────────────────────
    with tab_train:
        section_header("Configure Training", "Select your target, features, and models", "")

        # Upload a new dataset
        with st.expander("Upload a different dataset"):
            new_file = st.file_uploader("Upload CSV", type=["csv"], key="retrain_upload", label_visibility="collapsed")
            if new_file is not None:
                try:
                    df = pd.read_csv(new_file)
                    st.session_state.df = df
                    st.session_state.preprocessed = None
                    st.session_state.results = None
                    st.session_state.best_model_name = None
                    st.session_state.training_complete = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # Layout: two columns — left for target/features, right for model selection
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### Target Variable")
            target_col = st.selectbox(
                "Select the column you want to predict",
                options=df.columns.tolist(),
                index=len(df.columns) - 1,
            )

            # ── Target Validation ──
            target_valid = True
            target_series = df[target_col]
            n_unique = target_series.nunique()
            n_rows = len(df)
            missing_pct = target_series.isnull().sum() / n_rows * 100

            is_unique_id = (
                n_unique == n_rows
                or (n_unique / n_rows > 0.95 and target_series.dtype in ["int64", "int32", "object"])
            )
            is_constant = n_unique <= 1
            is_high_missing = missing_pct > 50

            if is_unique_id:
                target_valid = False
                st.markdown(
                    f"""
                    <div style="
                        background:{COLORS['danger']}15;border:1px solid {COLORS['danger']}35;
                        border-radius:10px;padding:12px 16px;margin:8px 0;
                    ">
                        <span style="color:{COLORS['danger']};font-weight:600;">Invalid Target</span><br>
                        <span style="color:{COLORS['text_secondary']};font-size:0.85rem;">
                            <b>{target_col}</b> looks like a unique identifier
                            ({n_unique:,} unique values out of {n_rows:,} rows).
                            Choose a column that represents the label you want to predict.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif is_constant:
                target_valid = False
                st.markdown(
                    f"""
                    <div style="
                        background:{COLORS['danger']}15;border:1px solid {COLORS['danger']}35;
                        border-radius:10px;padding:12px 16px;margin:8px 0;
                    ">
                        <span style="color:{COLORS['danger']};font-weight:600;">Invalid Target</span><br>
                        <span style="color:{COLORS['text_secondary']};font-size:0.85rem;">
                            <b>{target_col}</b> has only {n_unique} unique value(s).
                            A target needs at least 2 distinct classes.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif is_high_missing:
                target_valid = False
                st.markdown(
                    f"""
                    <div style="
                        background:{COLORS['warning']}15;border:1px solid {COLORS['warning']}35;
                        border-radius:10px;padding:12px 16px;margin:8px 0;
                    ">
                        <span style="color:{COLORS['warning']};font-weight:600;">High Missing Rate</span><br>
                        <span style="color:{COLORS['text_secondary']};font-size:0.85rem;">
                            <b>{target_col}</b> has {missing_pct:.1f}% missing values.
                            This is too high for reliable training.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # ── Task Detection ──
                auto_task = "Classification" if (target_series.dtype in ["object", "category", "bool"] or n_unique <= 20) else "Regression"
                
                st.markdown("#### Problem Type")
                task_type = st.selectbox(
                    "Override detected problem type if incorrect",
                    ["Classification", "Regression"],
                    index=0 if auto_task == "Classification" else 1,
                    label_visibility="collapsed"
                )
                task_color = COLORS["primary"] if task_type == "Classification" else COLORS["secondary"]

                st.markdown(
                    f"""
                    <div style="
                        background:{task_color}12;border:1px solid {task_color}25;
                        border-radius:10px;padding:12px 16px;margin:8px 0;
                    ">
                        <span style="color:{COLORS['text_secondary']};font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Task Type</span><br>
                        <span style="color:{COLORS['text_primary']};font-weight:600;">
                            Configured as a <span style="color:{task_color};">{task_type}</span> problem.
                        </span><br>
                        <span style="color:{COLORS['text_secondary']};font-size:0.82rem;">
                            {n_unique} unique values · {n_rows - target_series.isnull().sum():,} valid samples
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Class Imbalance Check
                if task_type == "Classification":
                    val_counts = target_series.value_counts(normalize=True)
                    if len(val_counts) > 0:
                        min_pct = val_counts.min() * 100
                        maj_pct = val_counts.max() * 100
                        if min_pct < 10 or maj_pct > 90:
                            st.markdown(
                                f"""
                                <div style="
                                    background:{COLORS['warning']}15;border:1px solid {COLORS['warning']}35;
                                    border-radius:10px;padding:12px 16px;margin:8px 0;
                                ">
                                    <span style="color:{COLORS['warning']};font-weight:600;">Class Imbalance Detected</span><br>
                                    <span style="color:{COLORS['text_secondary']};font-size:0.85rem;">
                                        The majority class makes up {maj_pct:.1f}% of the target. Metrics like Accuracy may be misleading. Consider prioritizing F1-Score or Recall for model selection.
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                elif task_type == "Regression":
                    st.caption(
                        f"Note: AutoSage is primarily designed for classification scoring in its UI, "
                        f"but you can proceed with algorithms that may support it."
                    )
                
                # Data Leakage Check
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                leakage_cols = []
                if pd.api.types.is_numeric_dtype(target_series):
                    for col in numeric_cols:
                        if col != target_col:
                            corr = abs(target_series.corr(df[col]))
                            if pd.notna(corr) and corr > 0.90:
                                leakage_cols.append(col)
                if leakage_cols:
                    st.markdown(
                        f"""
                        <div style="
                            background:{COLORS['danger']}15;border:1px solid {COLORS['danger']}35;
                            border-radius:10px;padding:12px 16px;margin:8px 0;
                        ">
                            <span style="color:{COLORS['danger']};font-weight:600;">Potential Data Leakage</span><br>
                            <span style="color:{COLORS['text_secondary']};font-size:0.85rem;">
                                Features <b>{', '.join(leakage_cols)}</b> have extremely high correlation (>0.90) with the target. Ensure these aren't providing the answer directly (e.g. knowing 'survived' beforehand).
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Feature Selection")
            max_features = len([c for c in df.columns if c != target_col])
            k_features = st.slider(
                "Number of top features to keep (0 = all)",
                min_value=0,
                max_value=max_features,
                value=min(10, max_features),
                help="Uses SelectKBest to keep only the most relevant features.",
            )

            st.markdown("#### Optimization Goal")
            st.session_state.opt_goal = st.selectbox(
                "Select prioritization metric:",
                ["Accuracy", "Recall", "F1 Score"],
                index=0,
                help="We use this to recommend the Best Model."
            )

        with col_right:
            st.markdown("#### Models")
            st.caption("Select which algorithms to train and compare.")

            # Model checkboxes in a clean two-column grid
            model_names = list(MODEL_REGISTRY.keys())
            mcol1, mcol2 = st.columns(2)
            selected_models = []
            for i, model_name in enumerate(model_names):
                info = MODEL_REGISTRY[model_name]
                target_col_widget = mcol1 if i % 2 == 0 else mcol2
                with target_col_widget:
                    if st.checkbox(f'{info["icon"]} {model_name}', value=False, key=f"chk_{model_name}"):
                        selected_models.append(model_name)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("#### Tuning")
            tcol1, tcol2, tcol3 = st.columns([2, 1, 1])
            with tcol1:
                tuning_strategy_label = st.selectbox(
                    "Strategy",
                    ["Fast (RandomSearch)", "Balanced (Optuna)", "Exhaustive (GridSearch)"],
                    index=0,
                    label_visibility="collapsed"
                )
                strategy_map = {
                    "Fast (RandomSearch)": "random",
                    "Balanced (Optuna)": "optuna",
                    "Exhaustive (GridSearch)": "grid"
                }
                tuning_strategy = strategy_map[tuning_strategy_label]
            with tcol2:
                n_iter = st.slider("Iterations", 5, 50, 20, help="For Random/Optuna")
            with tcol3:
                cv_folds = st.slider("CV folds", 2, 10, 5)

        # ── Train Button ──
        st.markdown("<br>", unsafe_allow_html=True)

        bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
        with bcol2:
            train_btn = st.button(
                "Train Models",
                use_container_width=True,
                type="primary",
                disabled=not target_valid,
            )

        if not target_valid and train_btn:
            st.error("Cannot train with the selected target variable. Please choose a valid target column.")

        elif train_btn and selected_models:
            try:
                with st.spinner("Preprocessing data..."):
                    preprocessed = preprocess(df, target_col, k_features, task_type=task_type, enable_scaling=True)
                    st.session_state.preprocessed = preprocessed

                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_cb(model_name, step, total):
                    progress_bar.progress((step + 1) / total)
                    icon = MODEL_REGISTRY.get(model_name, {}).get("icon", "")
                    status_text.markdown(
                        f'<div class="training-pulse" style="color:{COLORS["secondary"]};font-size:0.9rem;">'
                        f'Training {icon} {model_name}... ({step+1}/{total})</div>',
                        unsafe_allow_html=True,
                    )

                results = train_models(
                    preprocessed["X_train"],
                    preprocessed["y_train"],
                    selected_models,
                    n_iter=n_iter,
                    cv_folds=cv_folds,
                    tuning_strategy=tuning_strategy,
                    progress_callback=progress_cb,
                )

                st.session_state.results = results
                st.session_state.training_complete = True

                best_name, best_result = get_best_model(results)
                st.session_state.best_model_name = best_name

                # Evaluate best model for tracking
                if best_result and best_result.get("model"):
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_pred_test = best_result["model"].predict(st.session_state.preprocessed["X_test"])
                    y_test = st.session_state.preprocessed["y_test"]
                    test_metrics = {
                        "accuracy": accuracy_score(y_test, y_pred_test),
                        "precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
                        "recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
                        "f1_score": f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
                    }

                    # Log the experiment
                    from modules.experiment_tracking import log_experiment, hash_dataset
                    log_experiment(
                        hash_dataset(df), target_col, best_name, 
                        best_result["cv_score"], test_metrics, best_result.get("best_params", {})
                    )

                progress_bar.progress(1.0)
                if best_name:
                    status_text.success(f"Training complete — Best model: {best_name}")
                else:
                    status_text.error("All models failed to train. Check your data and try again.")

            except ValueError as e:
                st.error(f"**Data Error:** {str(e)}")
                st.info("Make sure your target column has enough samples per class (at least 2), "
                        "and that your features contain valid data.")
            except Exception as e:
                st.error(f"**Error:** {str(e)}")
                st.info("Try selecting a different target variable or check your dataset for issues.")

        elif train_btn and not selected_models:
            st.warning("Please select at least one model to train.")

        # Download model
        if st.session_state.training_complete and st.session_state.best_model_name:
            st.markdown("---")
            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                best_name = st.session_state.best_model_name
                best_model = st.session_state.results[best_name]["model"]
                download_model(best_model, best_name)
            with dcol2:
                from modules.export import generate_fastapi_script
                script1 = generate_fastapi_script("model.pkl", st.session_state.preprocessed["feature_names"])
                st.download_button("📥 Export FastAPI API", script1, file_name="fastapi_app.py", use_container_width=True)
            with dcol3:
                from modules.export import generate_streamlit_script
                script2 = generate_streamlit_script("model.pkl", st.session_state.preprocessed["feature_names"])
                st.download_button("📥 Export Streamlit App", script2, file_name="streamlit_app.py", use_container_width=True)

    # ──────────────────────────────────
    # TAB 3: Results
    # ──────────────────────────────────
    with tab_results:
        if st.session_state.results and st.session_state.preprocessed:
            render_evaluation(
                st.session_state.results,
                st.session_state.preprocessed["X_test"],
                st.session_state.preprocessed["y_test"],
                st.session_state.preprocessed["label_encoder"],
            )
        else:
            empty_state("Train models first to see evaluation results.", "")

    # ──────────────────────────────────
    # TAB 4: Explainability
    # ──────────────────────────────────
    with tab_xai:
        if st.session_state.results and st.session_state.preprocessed and st.session_state.best_model_name:
            valid_models = {k: v for k, v in st.session_state.results.items() if v.get("model") is not None}
            shap_model_name = st.selectbox(
                "Select model for SHAP analysis",
                list(valid_models.keys()),
                index=list(valid_models.keys()).index(st.session_state.best_model_name)
                    if st.session_state.best_model_name in valid_models else 0,
                key="shap_model_select",
            )
            render_shap(
                valid_models[shap_model_name]["model"],
                st.session_state.preprocessed["X_test"],
                st.session_state.preprocessed["feature_names"],
                shap_model_name,
                st.session_state.preprocessed["label_encoder"],
            )
        else:
            empty_state("Train models first to see explainability analysis.", "")

    # ──────────────────────────────────
    # TAB 5: Predict
    # ──────────────────────────────────
    with tab_predict:
        if st.session_state.results and st.session_state.preprocessed and st.session_state.best_model_name:
            best_name = st.session_state.best_model_name
            render_prediction(
                st.session_state.results[best_name]["model"],
                st.session_state.preprocessed["feature_names"],
                st.session_state.preprocessed["label_encoder"],
                best_name,
                st.session_state.preprocessed["X_train"],
            )
        else:
            empty_state("Train models first to make predictions.", "")

    # ──────────────────────────────────
    # TAB 6: Experiments
    # ──────────────────────────────────
    with tab_experiments:
        from modules.experiment_tracking import load_experiments
        try:
            exp_df = load_experiments()
            if exp_df.empty:
                empty_state("No experiments tracked yet. Train a model to see history.", "📊")
            else:
                section_header("Experiment History", "Tracked model runs and dataset versions", "📊")
                st.dataframe(exp_df, use_container_width=True, hide_index=True)
        except Exception:
            empty_state("No experiments tracked yet.", "📊")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:40px 20px 20px;
        color:{COLORS['text_secondary']};
        font-size:0.78rem;
        border-top:1px solid {COLORS['primary']}10;
        margin-top:60px;
    ">
        AutoSage — Intelligent AutoML with Explainable AI
    </div>
    """,
    unsafe_allow_html=True,
)
