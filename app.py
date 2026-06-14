import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import pickle

# Import modular project functions
from src.data_pipeline import fetch_ticker_data
from src.database import cache_ticker_data, get_cached_ticker_data
from src.features import compute_basic_features, compute_ml_features, compute_correlation_matrix
from src.models import (
    fit_and_forecast_prophet,
    predict_next_day_xgb,
    define_risk_regimes,
    predict_next_day_risk
)
from src.train_models import run_training_pipeline
from src.business_logic import get_procurement_advice

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="EcoGrid AI — Green Asset Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────
# GLOBAL DESIGN SYSTEM + CSS INJECTION
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ─── BASE & LAYOUT ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #080a10 !important;
    color: #cbd5e1 !important;
}
.main { background-color: #080a10 !important; }
.block-container {
    padding: 2rem 2.5rem 4rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {
    background-color: #04060a !important;
    border-right: 1px solid #1e293b !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ─── SIDEBAR BRAND STRIP ─── */
.sb-brand {
    background: linear-gradient(135deg, #090d16 0%, #0c1220 100%);
    border-bottom: 1px solid #1e293b;
    padding: 20px 16px 18px;
    margin-bottom: 8px;
}
.sb-brand-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.sb-brand-sub {
    font-size: 0.72rem;
    color: #64748b !important;
    letter-spacing: 0.03em;
}

/* ─── SIDEBAR SECTION LABELS ─── */
.sb-section {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3b82f6 !important;
    padding: 16px 0 6px 0;
    margin-left: 2px;
}
.sb-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 6px 0 10px;
}
.sb-meta-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #64748b !important;
    padding: 3px 0;
}

/* ─── HERO SECTION ─── */
.hero-wrapper {
    background: linear-gradient(135deg, #0e1322 0%, #121829 60%, #0e1628 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #60a5fa !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}
.live-dot {
    width: 7px; height: 7px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 1; }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.05rem;
    font-weight: 700;
    color: #fff !important;
    line-height: 1.2;
    letter-spacing: -0.03em;
    margin: 0 0 8px 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8 !important;
    line-height: 1.6;
    max-width: 600px;
    margin: 0;
}
.hero-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 24px;
}
.hero-pill {
    background: rgba(255,255,255,0.02);
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 8px 16px;
    display: flex;
    flex-direction: column;
}
.hp-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 3px;
}
.hp-value {
    font-size: 1rem;
    font-weight: 700;
    color: #fff !important;
    line-height: 1;
}

/* ─── KPI CARDS ─── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #121620;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 20px 16px;
    transition: border-color 0.2s, transform 0.2s, background-color 0.2s;
    cursor: default;
}
.kpi-card:hover {
    border-color: rgba(59,130,246,0.3);
    background-color: #161c2b;
    transform: translateY(-2px);
}
.kpi-icon { font-size: 1.4rem; margin-bottom: 10px; display: block; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-delta {
    margin-top: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}
.kpi-context {
    font-size: 0.72rem;
    color: #64748b !important;
    margin-top: 4px;
}

/* ─── SECTION HEADERS ─── */
.section-header {
    margin: 36px 0 4px 0;
}
.section-tag {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3b82f6 !important;
    margin-bottom: 4px;
    display: block;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}
.section-desc {
    font-size: 0.85rem;
    color: #64748b !important;
    margin: 0 0 20px 0;
}

/* ─── CHART WRAPPERS ─── */
.chart-card {
    background: #121620;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 16px 8px;
    margin-bottom: 16px;
}
.chart-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e1 !important;
    margin-bottom: 2px;
}
.chart-sub {
    font-size: 0.72rem;
    color: #64748b !important;
    margin-bottom: 12px;
}

/* ─── RISK ENGINE PANEL ─── */
.risk-panel {
    background: linear-gradient(135deg, #121620 0%, #151c2d 100%);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 20px;
}
.risk-factors-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-top: 16px;
}
.risk-factor-card {
    background: rgba(0,0,0,0.4);
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.rf-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
    margin-bottom: 6px;
}
.rf-value {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.25rem;
    font-weight: 700;
}
.rf-context { font-size: 0.68rem; color: #64748b !important; margin-top: 3px; }

/* ─── INSIGHT BOX ─── */
.insight-box {
    background: rgba(59,130,246,0.03);
    border-left: 3px solid #3b82f6;
    border-top: 1px solid #1e293b;
    border-bottom: 1px solid #1e293b;
    border-right: 1px solid #1e293b;
    border-radius: 0 12px 12px 0;
    padding: 18px 22px;
    margin: 20px 0 24px;
}
.insight-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.insight-badge {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #60a5fa !important;
}
.insight-row {
    font-size: 0.85rem;
    line-height: 1.6;
    color: #94a3b8 !important;
    margin-bottom: 5px;
}
.insight-row strong { color: #f8fafc !important; }

/* ─── ADVISORY PANEL ─── */
.advisory-panel {
    border-radius: 14px;
    padding: 24px;
    border: 1px solid #1e293b;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.advisory-action-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #fff !important;
    margin-bottom: 12px;
}
.advisory-headline {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.25rem;
    font-weight: 700;
    color: #fff !important;
    margin-bottom: 6px;
    line-height: 1.3;
}
.advisory-sub {
    font-size: 0.82rem;
    color: #94a3b8 !important;
}

/* ─── LEADERBOARD TABLE ─── */
.lb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    background: #121620;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1e293b;
}
.lb-table th {
    background: #181e2e;
    color: #94a3b8 !important;
    text-align: left;
    padding: 11px 14px;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    border-bottom: 1px solid #1e293b;
}
.lb-table td {
    padding: 13px 14px;
    color: #cbd5e1 !important;
    border-bottom: 1px solid #1e293b;
}
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr:hover td { background: rgba(255,255,255,0.02); }

/* ─── TABS ─── */
div[data-testid="stTabs"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 20px !important;
}
button[data-baseweb="tab"] {
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: color 0.2s !important;
}
button[data-baseweb="tab"]:hover { color: #f8fafc !important; }
button[aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}
div[data-baseweb="tab-list"] {
    border-bottom: 1px solid #1e293b !important;
    gap: 4px !important;
    background: transparent !important;
}

/* ─── STREAMLIT NATIVE OVERRIDES ─── */
div[data-testid="stMetric"] {
    background: #121620 !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #fff !important;
}
.stExpander {
    background: #121620 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

/* ─── FORM CONTROLS & INPUTS (SELECTBOX, SLIDERS, BUTTONS) ─── */
/* Selectboxes */
div[data-baseweb="select"] > div {
    background-color: #121620 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e293b !important;
}
div[data-baseweb="select"] * {
    color: #e2e8f0 !important;
}
/* Popover menus / Dropdowns */
div[data-baseweb="popover"] ul, ul[role="listbox"] {
    background-color: #121620 !important;
    border: 1px solid #1e293b !important;
}
li[role="option"] {
    background-color: #121620 !important;
    color: #cbd5e1 !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: #1e293b !important;
    color: #fff !important;
}

/* Sliders */
div[data-testid="stSlider"] > div {
    color: #e2e8f0 !important;
}
div[data-testid="stSlider"] [data-absolute="true"] {
    background-color: #1e293b !important;
}

/* Container borders and widgets */
div[data-testid="stForm"] {
    background-color: #121620 !important;
    border: 1px solid #1e293b !important;
}

/* Info Box */
.st-info-box {
    background: rgba(59,130,246,0.05) !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
div[data-testid="stAlert"] {
    background-color: #121620 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
}

/* ─── BUTTONS ─── */
.stDownloadButton button, .stButton button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
}
.stDownloadButton button:hover, .stButton button:hover {
    background: #1e40af !important;
}

/* ─── MISC ─── */
hr { border-color: #1e293b !important; }

/* ─── PROGRESS BAR ─── */
.risk-bar-track {
    height: 6px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0 4px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* ─── FOOTER ─── */
.footer {
    text-align: center;
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid #1e293b;
    font-size: 0.78rem;
    color: #475569 !important;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────
def apply_chart_theme(fig, title="", subtitle=""):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup style='color:#475569;font-size:11px'>{subtitle}</sup>" if subtitle else f"<b>{title}</b>",
            x=0.0, xanchor="left",
            font=dict(family="Space Grotesk, Inter, sans-serif", size=14, color="#e2e8f0"),
            pad=dict(b=10)
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#64748b", size=11),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor="rgba(10,12,21,0.7)",
            bordercolor="rgba(255,255,255,0.06)",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=8, r=8, t=55, b=8),
        hoverlabel=dict(
            bgcolor="#0f1422",
            bordercolor="rgba(59,130,246,0.3)",
            font=dict(family="Inter", size=12, color="#e2e8f0")
        )
    )
    return fig


# ──────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────
def render_insight_box(why_matters: str, what_changed: str, what_to_do: str):
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-header">
            <span style="font-size:1.1rem">🧠</span>
            <span class="insight-badge">AI Decision Intelligence</span>
        </div>
        <div class="insight-row"><strong>Why it matters:</strong> {why_matters}</div>
        <div class="insight-row"><strong>What changed:</strong> {what_changed}</div>
        <div class="insight-row"><strong>Recommended action:</strong> {what_to_do}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(icon, label, value, delta_text, delta_positive, context=""):
    delta_color = "#10b981" if delta_positive else "#ef4444"
    delta_icon = "↑" if delta_positive else "↓"
    st.markdown(f"""
    <div class="kpi-card">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{("#3b82f6" if delta_positive else "#f59e0b")}">{value}</div>
        <div class="kpi-delta" style="color:{delta_color}">{delta_icon} {delta_text}</div>
        <div class="kpi-context">{context}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(tag: str, title: str, desc: str = ""):
    st.markdown(f"""
    <div class="section-header">
        <span class="section-tag">{tag}</span>
        <div class="section-title">{title}</div>
        <div class="section-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def chart_card_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="chart-title">{title}</div>
    <div class="chart-sub">{subtitle}</div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# DATA LOADERS (CACHED)
# ──────────────────────────────────────────
@st.cache_data(ttl=300)
def load_and_preprocess_data(ticker: str, period: str) -> tuple:
    offline_mode = False
    try:
        df_raw = fetch_ticker_data(ticker, period)
        cache_ticker_data(ticker, df_raw)
    except Exception as e:
        df_raw = get_cached_ticker_data(ticker)
        if df_raw.empty:
            raise e
        offline_mode = True
    df_basic = compute_basic_features(df_raw)
    df_labeled, risk_meta = define_risk_regimes(df_basic)
    return df_labeled, risk_meta, offline_mode


@st.cache_data(ttl=300)
def get_ml_features(df_labeled: pd.DataFrame) -> tuple:
    return compute_ml_features(df_labeled)


@st.cache_resource(ttl=600)
def run_prophet_model(_df_labeled: pd.DataFrame, forecast_days: int) -> tuple:
    return fit_and_forecast_prophet(_df_labeled, forecast_days)


def load_serialized_models(ticker: str) -> tuple:
    xgb_path = f"models/{ticker}_xgb.pkl"
    risk_clf_path = f"models/{ticker}_risk_clf.pkl"
    if not os.path.exists(xgb_path) or not os.path.exists(risk_clf_path):
        with st.spinner("Initializing first-time model training bootstrap…"):
            run_training_pipeline()
    with open(f"models/{ticker}_xgb.pkl", "rb") as f:      xgb_model = pickle.load(f)
    with open(f"models/{ticker}_risk_clf.pkl", "rb") as f:  rf_clf = pickle.load(f)
    with open(f"models/{ticker}_risk_le.pkl", "rb") as f:   label_encoder = pickle.load(f)
    with open(f"models/{ticker}_xgb_metrics.pkl", "rb") as f: xgb_metrics = pickle.load(f)
    with open(f"models/{ticker}_xgb_results.pkl", "rb") as f: test_results = pickle.load(f)
    with open(f"models/{ticker}_xgb_importance.pkl", "rb") as f: importance_df = pickle.load(f)
    with open(f"models/{ticker}_risk_metrics.pkl", "rb") as f: rf_metrics = pickle.load(f)
    return xgb_model, rf_clf, label_encoder, xgb_metrics, test_results, importance_df, rf_metrics


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    # Brand strip
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">⚡ EcoGrid AI</div>
        <div class="sb-brand-sub">Green Asset Volatility Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Asset Selection ──
    st.markdown('<div class="sb-section">Asset</div>', unsafe_allow_html=True)
    ticker_option = st.selectbox(
        "Target Asset",
        options=["ICLN", "XLU", "CEG"],
        index=0,
        format_func=lambda x: {
            "ICLN": "🌱 ICLN — Clean Energy ETF",
            "XLU":  "🔋 XLU  — Utilities ETF",
            "CEG":  "⚡ CEG  — Clean Energy Producer"
        }[x],
        label_visibility="collapsed"
    )
    lookback_option = st.selectbox(
        "Historical Range",
        options=["1y", "2y", "5y"],
        index=2,
        format_func=lambda x: {"1y": "1 Year", "2y": "2 Years", "5y": "5 Years"}[x]
    )

    # ── View Controls ──
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Chart View</div>', unsafe_allow_html=True)
    view_window = st.slider(
        "Historical Window (Days)",
        min_value=30, max_value=365, value=75, step=5,
        help="Controls days shown in all historical charts."
    )

    # ── Forecast Settings ──
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Forecast</div>', unsafe_allow_html=True)
    horizon_option = st.slider(
        "Forecast Horizon (Days)",
        min_value=30, max_value=90, value=60, step=15,
        help="How many days ahead Prophet + GARCH will project."
    )
    model_selector = st.selectbox(
        "Model Engine",
        options=["Prophet + GARCH (Macro)", "XGBoost (Daily Returns)", "Ensemble (All Signals)"],
        index=2
    )

    # ── Advanced ──
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    with st.expander("⚙️ Advanced Controls", expanded=False):
        st.caption("Cache & Pipeline")
        if st.button("🔄 Force Refresh & Retrain", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            with st.spinner("Retraining ML models…"):
                run_training_pipeline()
            st.success("Retrained & cache cleared!")
            st.rerun()

    # ── Meta ──
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-meta-row">🕐 Synced: {datetime.now().strftime('%b %d, %Y %H:%M')}</div>
    <div class="sb-meta-row">📡 Source: Yahoo Finance API</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════
with st.spinner("Fetching market data and aligning indicators…"):
    try:
        df_labeled, risk_meta, offline_flag = load_and_preprocess_data(ticker_option, lookback_option)
        train_df, inference_row = get_ml_features(df_labeled)
        xgb_model, rf_clf, label_encoder, xgb_metrics, test_results, importance_df, rf_metrics = load_serialized_models(ticker_option)

        all_assets = {}
        for tick in ["ICLN", "XLU", "CEG"]:
            if tick == ticker_option:
                all_assets[tick] = df_labeled
            else:
                try:
                    raw = fetch_ticker_data(tick, lookback_option)
                    cache_ticker_data(tick, raw)
                except Exception:
                    raw = get_cached_ticker_data(tick)
                    if raw.empty:
                        st.error(f"Critical: local cache missing for {tick}.")
                        st.stop()
                basic = compute_basic_features(raw)
                all_assets[tick] = basic

        correlation_matrix = compute_correlation_matrix(all_assets)

    except Exception as e:
        st.error(f"Unable to load market data: {e}")
        st.stop()

if offline_flag:
    st.warning("⚠️ Yahoo Finance API offline — running on cached data. Prices reflect last successful sync.")


# ══════════════════════════════════════════
# COMPUTE METRICS
# ══════════════════════════════════════════
latest_row    = df_labeled.iloc[-1]
prev_row      = df_labeled.iloc[-2]
latest_close  = latest_row["Close"]  if not pd.isna(latest_row["Close"])  else 0.0
latest_vol    = latest_row["Roll_Vol_30d"] if not pd.isna(latest_row["Roll_Vol_30d"]) else 0.0
price_pct_chg = (latest_close - prev_row["Close"]) / max(prev_row["Close"], 1e-9) * 100
vol_change    = latest_vol - (prev_row["Roll_Vol_30d"] if not pd.isna(prev_row["Roll_Vol_30d"]) else latest_vol)
current_risk  = latest_row["Risk_Regime"]
current_dd    = latest_row["Drawdown"] if not pd.isna(latest_row["Drawdown"]) else 0.0

vol_median   = df_labeled["Roll_Vol_30d"].median()
max_price    = df_labeled["Close"].max()
min_drawdown = df_labeled["Drawdown"].min()
vol_75       = risk_meta.get("vol_75", df_labeled["Roll_Vol_30d"].quantile(0.75))

for v in [vol_median, max_price, min_drawdown, vol_75]:
    if pd.isna(v): v = 0.0

rsi = latest_row.get("RSI_14", 50.0)
if pd.isna(rsi): rsi = 50.0

vol_risk_score      = min(100.0, (latest_vol / max(0.01, vol_75)) * 75.0)
drawdown_risk_score = min(100.0, abs(current_dd) / 0.20 * 100.0)
momentum_risk_score = max(0.0, min(100.0, (50.0 - rsi) * 2.0 + 50.0))
risk_score = max(0.0, min(100.0, 0.4 * vol_risk_score + 0.4 * drawdown_risk_score + 0.2 * momentum_risk_score))

risk_category = (
    "Extreme" if risk_score >= 85 else
    "High"    if risk_score >= 60 else
    "Medium"  if risk_score >= 30 else "Low"
)
market_regime = "Volatile" if latest_vol > vol_75 else ("Bullish" if latest_row.get("SMA_50_Ratio", 0) > 0 else "Bearish")
regime_color = {"Volatile": "#ef4444", "Bullish": "#10b981", "Bearish": "#f59e0b"}[market_regime]
risk_color   = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Extreme": "#b91c1c"}[risk_category]
risk_state_color = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}.get(current_risk, "#f59e0b")

df_visual = df_labeled.iloc[-view_window:].copy()


# ══════════════════════════════════════════
# 1. HERO SECTION
# ══════════════════════════════════════════
st.markdown(f"""
<div class="hero-wrapper">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:24px;">
        <div>
            <div class="hero-eyebrow">
                <span class="live-dot"></span> Live Market Intelligence
            </div>
            <div class="hero-title">EcoGrid AI <span style="color:#3b82f6;">Volatility System</span></div>
            <p class="hero-sub">
                Predictive green asset volatility analysis and data-center energy procurement intelligence,
                powered by Prophet, XGBoost, and GARCH risk models.
            </p>
            <div class="hero-pills">
                <div class="hero-pill">
                    <span class="hp-label">Tracking</span>
                    <span class="hp-value">{ticker_option}</span>
                </div>
                <div class="hero-pill">
                    <span class="hp-label">Regime</span>
                    <span class="hp-value" style="color:{regime_color}">{market_regime}</span>
                </div>
                <div class="hero-pill">
                    <span class="hp-label">Risk Level</span>
                    <span class="hp-value" style="color:{risk_color}">{risk_category}</span>
                </div>
                <div class="hero-pill">
                    <span class="hp-label">Risk Score</span>
                    <span class="hp-value" style="color:{risk_color}">{risk_score:.0f} / 100</span>
                </div>
                <div class="hero-pill">
                    <span class="hp-label">As of</span>
                    <span class="hp-value" style="font-size:0.85rem">{datetime.now().strftime('%b %d, %Y')}</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# 2. KPI STRIP
# ══════════════════════════════════════════
kpi_cols = st.columns(5)

price_avg_pos = latest_close >= latest_row["SMA_50"]
dd_ok = current_dd > -0.05

with kpi_cols[0]:
    render_kpi(
        icon="💵",
        label="Current Price",
        value=f"${latest_close:.2f}",
        delta_text=f"{abs(price_pct_chg):.2f}% today",
        delta_positive=price_pct_chg >= 0,
        context=f"vs. 50-Day SMA ${latest_row['SMA_50']:.2f}"
    )

with kpi_cols[1]:
    render_kpi(
        icon="📊",
        label="30-Day Volatility",
        value=f"{latest_vol:.1%}",
        delta_text=f"{abs(vol_change)*100:.2f}% vs prev",
        delta_positive=vol_change <= 0,
        context=f"1Y Median: {vol_median:.1%}"
    )

with kpi_cols[2]:
    render_kpi(
        icon="📉",
        label="Peak Drawdown",
        value=f"{current_dd:.2%}",
        delta_text="within limits" if dd_ok else "above threshold",
        delta_positive=dd_ok,
        context=f"Hist. Low: {min_drawdown:.2%}"
    )

with kpi_cols[3]:
    render_kpi(
        icon="🎯",
        label="RSI (14-Day)",
        value=f"{rsi:.1f}",
        delta_text="Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
        delta_positive=30 <= rsi <= 70,
        context="50 = neutral baseline"
    )

with kpi_cols[4]:
    render_kpi(
        icon="🛡️",
        label="Risk State",
        value=current_risk.upper(),
        delta_text=f"Score: {risk_score:.0f}/100",
        delta_positive=risk_category in ["Low", "Medium"],
        context="Dynamic threshold engine"
    )


# ══════════════════════════════════════════
# 3. MAIN TABS
# ══════════════════════════════════════════
tabs = st.tabs([
    "📋  Overview",
    "📈  Market Intelligence",
    "🔮  Forecasting Engine",
    "⚠️  Risk & Strategy"
])


# ─────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────
with tabs[0]:

    # ── Risk Engine Panel ──
    section_header("Real-time Diagnostics", "Risk Engine Status",
                   "Multivariate threat indicators powered by volatility, drawdown, and momentum signals.")

    st.markdown(f"""
    <div class="risk-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div>
                <div style="font-size:1rem; font-weight:700; color:#e2e8f0;">Overall Risk Score</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">Composite of volatility, drawdown & momentum</div>
            </div>
            <div style="text-align:right;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:2.4rem; font-weight:800; color:{risk_color};">{risk_score:.0f}</span>
                <span style="font-size:1rem; color:#475569;">/ 100</span>
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:{risk_color}; margin-top:3px;">{risk_category} Risk</div>
            </div>
        </div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{risk_score}%; background:{risk_color};"></div>
        </div>
        <div class="risk-factors-grid">
            <div class="risk-factor-card">
                <div class="rf-label">Volatility Pressure</div>
                <div class="rf-value" style="color:{'#ef4444' if latest_vol > vol_median else '#10b981'};">
                    {latest_vol:.1%} {'↑' if latest_vol > vol_median else '↓'}
                </div>
                <div class="rf-context">Median: {vol_median:.1%}</div>
            </div>
            <div class="risk-factor-card">
                <div class="rf-label">Drawdown Impact</div>
                <div class="rf-value" style="color:{'#ef4444' if current_dd < -0.05 else '#10b981'};">
                    {current_dd:.2%} {'↑' if current_dd < -0.05 else '↓'}
                </div>
                <div class="rf-context">Hedge trigger: −5.0%</div>
            </div>
            <div class="risk-factor-card">
                <div class="rf-label">Trend Momentum</div>
                <div class="rf-value" style="color:{'#10b981' if latest_row.get('SMA_50_Ratio', 0) > 0 else '#ef4444'};">
                    {market_regime} {'↑' if latest_row.get('SMA_50_Ratio', 0) > 0 else '↓'}
                </div>
                <div class="rf-context">SMA-50 ratio: {latest_row.get('SMA_50_Ratio', 0):.1%}</div>
            </div>
            <div class="risk-factor-card">
                <div class="rf-label">RSI Signal</div>
                <div class="rf-value" style="color:{'#f59e0b' if rsi > 70 or rsi < 30 else '#10b981'};">
                    {rsi:.1f} {'⚠' if rsi > 70 or rsi < 30 else '✓'}
                </div>
                <div class="rf-context">{'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral zone')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Price overview chart ──
    section_header("Market Snapshot", f"{ticker_option} Price & Trend Overview",
                   f"Closing price over the last {view_window} days with SMA overlays.")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_visual.index, y=df_visual["Close"],
        name="Close Price", fill="tozeroy",
        fillcolor="rgba(59,130,246,0.06)",
        line=dict(color="#3b82f6", width=2.5)
    ))
    fig_price.add_trace(go.Scatter(
        x=df_visual.index, y=df_visual["SMA_50"],
        name="50-Day SMA", line=dict(color="#10b981", width=1.5, dash="dash")
    ))
    fig_price.add_trace(go.Scatter(
        x=df_visual.index, y=df_visual["SMA_200"],
        name="200-Day SMA", line=dict(color="#ec4899", width=1.5, dash="dot")
    ))
    apply_chart_theme(fig_price, f"{ticker_option} — Price & Moving Averages",
                      f"{view_window}-day window · SMA-50 and SMA-200 overlays")
    st.plotly_chart(fig_price, use_container_width=True)

    # ── AI Insight ──
    render_insight_box(
        why_matters=f"The {ticker_option} market is currently classified as {current_risk.upper()} risk. Knowing this state helps optimize energy contract mix ratios between spot purchases and defensive utility hedges.",
        what_changed=f"Volatility has reached {latest_vol:.1%} (median: {vol_median:.1%}), with {market_regime.lower()} momentum and a current drawdown of {current_dd:.2%}.",
        what_to_do=f"Based on the {current_risk.upper()} classification, consult the Risk & Strategy tab for dynamic volume allocation recommendations."
    )


# ─────────────────────────────────────────
# TAB 2 — MARKET INTELLIGENCE
# ─────────────────────────────────────────
with tabs[1]:
    section_header("Why Is It Happening?", "Volatility & Pricing Drivers",
                   "Contextual insights, historical correlations, and asset distribution relationships.")

    # Context cards
    ctx_col1, ctx_col2 = st.columns(2)
    with ctx_col1:
        with st.container(border=True):
            st.markdown("**⚡ Volatility Context**")
            st.markdown(
                f"The 30-day annualized rolling volatility for **{ticker_option}** is currently **{latest_vol:.1%}**, "
                f"against a historic median of **{vol_median:.1%}**. "
                f"Daily return swings are **{'wider' if latest_vol > vol_median else 'tighter'}** than average, driven by "
                "weather events disrupting renewable generation capacity, fuel index price spikes, and power grid reserve margin changes."
            )
    with ctx_col2:
        with st.container(border=True):
            st.markdown("**📉 Drawdown Context**")
            st.markdown(
                f"Peak-to-trough drawdown shows the price decline from the all-time high of **${max_price:.2f}**. "
                f"The current correction is **{current_dd:.2%}** against a historical extreme of **{min_drawdown:.2%}**. "
                "Severe drawdowns mark optimal long-term entry points or warn of rising default risk in forward contracts."
            )

    st.markdown("")

    # 2×2 chart grid
    chart_r1c1, chart_r1c2 = st.columns(2)
    with chart_r1c1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df_visual.index, y=df_visual["Roll_Vol_7d"],
            name="7-Day Vol", line=dict(color="#34d399", width=1.5)
        ))
        fig_vol.add_trace(go.Scatter(
            x=df_visual.index, y=df_visual["Roll_Vol_30d"],
            name="30-Day Vol", fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
            line=dict(color="#f59e0b", width=2.5)
        ))
        apply_chart_theme(fig_vol, "Annualized Rolling Volatility",
                          "Short-term (7d) vs. medium-term (30d)")
        st.plotly_chart(fig_vol, use_container_width=True)

    with chart_r1c2:
        fig_dd = px.area(df_visual, x=df_visual.index, y="Drawdown")
        fig_dd.update_traces(line_color="#ef4444", fillcolor="rgba(239,68,68,0.1)")
        apply_chart_theme(fig_dd, "Peak-to-Trough Drawdown",
                          "Percentage decline from rolling high")
        st.plotly_chart(fig_dd, use_container_width=True)

    chart_r2c1, chart_r2c2 = st.columns(2)
    with chart_r2c1:
        fig_rsi = go.Figure()
        if "RSI_14" in df_visual.columns:
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.5)",
                              annotation_text="Overbought (70)", annotation_position="top right")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="rgba(16,185,129,0.5)",
                              annotation_text="Oversold (30)", annotation_position="bottom right")
            fig_rsi.add_trace(go.Scatter(
                x=df_visual.index, y=df_visual["RSI_14"],
                name="RSI-14", line=dict(color="#a78bfa", width=2)
            ))
        apply_chart_theme(fig_rsi, "RSI — Relative Strength Index (14-Day)",
                          "Momentum oscillator: 30–70 = neutral zone")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with chart_r2c2:
        fig_corr = px.imshow(
            correlation_matrix, text_auto=".2f",
            aspect="auto", color_continuous_scale="Blues"
        )
        apply_chart_theme(fig_corr, "Asset Correlation Heatmap",
                          "Pairwise return correlations: ICLN · XLU · CEG")
        st.plotly_chart(fig_corr, use_container_width=True)

    max_corr_partner = correlation_matrix[ticker_option].drop(ticker_option).idxmax()
    max_corr_val = correlation_matrix[ticker_option].drop(ticker_option).max()
    render_insight_box(
        why_matters="Assets exhibit high daily return correlations during broad macro shifts, while decoupling signals localized supply/demand dynamics.",
        what_changed=f"{ticker_option} currently has its strongest correlation with {max_corr_partner} at {max_corr_val:.2f}.",
        what_to_do="If correlation with defensive XLU exceeds 0.70, a systemic utilities-driven regime is active — consider diversifying supply chain locations."
    )


# ─────────────────────────────────────────
# TAB 3 — FORECASTING ENGINE
# ─────────────────────────────────────────
with tabs[2]:
    section_header("What Is the Projection?", "Predictive Machine Learning Engine",
                   f"Prophet macro trend · XGBoost short-term returns · GARCH dynamic volatility bands · {horizon_option}-day horizon")

    with st.spinner("Fitting Prophet + GARCH residuals…"):
        model, forecast, prophet_metrics = run_prophet_model(df_labeled, horizon_option)

    # ── Model performance strip ──
    r2 = xgb_metrics["R2"]
    da = xgb_metrics["Directional_Accuracy"]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="XGBoost R² Score",
            value=f"{r2:.4f}",
            delta="High variance explained" if r2 >= 0.8 else ("Moderate" if r2 >= 0.5 else "Low – warning"),
            delta_color="normal" if r2 >= 0.5 else "inverse"
        )
    with m_col2:
        st.metric(
            label="Directional Accuracy",
            value=f"{da:.1f}%",
            delta="Satisfactory" if da >= 50 else "Needs improvement",
            delta_color="normal" if da >= 50 else "inverse"
        )
    with m_col3:
        st.metric(
            label="Prophet MAE",
            value=f"${prophet_metrics['MAE']:.2f}",
            delta="Mean absolute forecast error"
        )
    with m_col4:
        next_day_price = predict_next_day_xgb(xgb_model, inference_row, latest_close)
        direction = "↑" if next_day_price > latest_close else "↓"
        st.metric(
            label="XGB Tomorrow Target",
            value=f"${next_day_price:.2f}",
            delta=f"{direction} {abs(next_day_price - latest_close):.2f} vs today",
            delta_color="normal" if next_day_price > latest_close else "inverse"
        )

    st.markdown("")

    # ── Prophet + GARCH main chart ──
    section_header("Trend Projection", f"Prophet + GARCH {horizon_option}-Day Forecast",
                   "Actual price history with forward trend line and dynamic uncertainty band.")

    future_forecast = forecast.iloc[-horizon_option:]
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df_visual.index, y=df_visual["Close"],
        name="Actual Close", line=dict(color="#3b82f6", width=2.5)
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_forecast["ds"], y=future_forecast["yhat"],
        name="Forecast (Prophet)", line=dict(color="#10b981", width=2.5)
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([future_forecast["ds"], future_forecast["ds"][::-1]]),
        y=pd.concat([future_forecast["yhat_upper"], future_forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(16,185,129,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", name="95% GARCH Band"
    ))
    apply_chart_theme(fig_fc, f"{ticker_option} — {horizon_option}-Day Dynamic Forecast",
                      "Green band = dynamic GARCH uncertainty interval")
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── XGBoost walk-forward + feature importance ──
    section_header("Model Validation", "XGBoost Walk-Forward Validation",
                   "Reconstructed price predictions vs. actual prices on held-out test data.")

    xgb_c1, xgb_c2 = st.columns([2, 1])
    with xgb_c1:
        fig_xgb = go.Figure()
        fig_xgb.add_trace(go.Scatter(
            x=test_results.index, y=test_results["Actual"],
            name="Actual Price", line=dict(color="#3b82f6", width=2)
        ))
        fig_xgb.add_trace(go.Scatter(
            x=test_results.index, y=test_results["Predicted"],
            name="XGBoost Predicted", line=dict(color="#ec4899", width=1.5, dash="dash")
        ))
        apply_chart_theme(fig_xgb, "Walk-Forward Validation",
                          "Predictions vs. actuals on test set")
        st.plotly_chart(fig_xgb, use_container_width=True)

    with xgb_c2:
        fig_imp = px.bar(
            importance_df.sort_values("Importance", ascending=True),
            y="Feature", x="Importance", orientation="h"
        )
        fig_imp.update_traces(marker_color="#a78bfa", marker_line_width=0)
        apply_chart_theme(fig_imp, "Feature Importance",
                          "Top drivers of XGBoost predictions")
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Seasonality ──
    with st.expander("📅 Seasonality Component Breakdown", expanded=False):
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            weekly = forecast.groupby(forecast["ds"].dt.dayofweek)["weekly"].mean().reset_index()
            weekly["Day"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig_week = px.bar(weekly, x="Day", y="weekly")
            fig_week.update_traces(marker_color="#10b981", marker_line_width=0)
            apply_chart_theme(fig_week, "Weekly Seasonality Effect",
                              "Average price effect by day of week")
            st.plotly_chart(fig_week, use_container_width=True)
        with s_col2:
            yearly = forecast.groupby(forecast["ds"].dt.month)["yearly"].mean().reset_index()
            yearly["Month"] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig_year = px.line(yearly, x="Month", y="yearly", markers=True)
            fig_year.update_traces(line_color="#f59e0b", line_width=2.5, marker_color="#f59e0b")
            apply_chart_theme(fig_year, "Annual Seasonality Effect",
                              "Average price effect by month of year")
            st.plotly_chart(fig_year, use_container_width=True)

    render_insight_box(
        why_matters="Prophet isolates seasonal and annual load trends while GARCH scales uncertainty bands — revealing price-extreme periods ideal for spot-buying lock-ins.",
        what_changed=f"Prophet projects a price target of ${future_forecast['yhat'].iloc[-1]:.2f} in {horizon_option} days. XGBoost next-day target: ${next_day_price:.2f}.",
        what_to_do=f"The {horizon_option}-day uncertainty band spans ${future_forecast['yhat_lower'].iloc[-1]:.2f}–${future_forecast['yhat_upper'].iloc[-1]:.2f}. Adjust hedge ratios if the spread widens by more than 15%."
    )


# ─────────────────────────────────────────
# TAB 4 — RISK & STRATEGY
# ─────────────────────────────────────────
with tabs[3]:
    section_header("What Should You Do?", "Actionable Strategy & Procurement Decisions",
                   "Rule-based procurement advisory, energy allocation sandbox, and hedging guidance.")

    # ── Advisory Board ──
    predicted_risk_tomorrow = predict_next_day_risk(rf_clf, label_encoder, inference_row)
    trend_state = "Bullish" if latest_row["SMA_50_Ratio"] > 0 else "Bearish"
    latest_fc = forecast[forecast["ds"] <= df_labeled.index[-1]].iloc[-1]
    uncertainty_ratio = (latest_fc["yhat_upper"] - latest_fc["yhat_lower"]) / latest_fc["yhat"]
    next_day_price_tab4 = predict_next_day_xgb(xgb_model, inference_row, latest_close)

    procurement_advice = get_procurement_advice(
        ticker=ticker_option,
        current_price=latest_close,
        rolling_vol_30d=latest_vol,
        current_drawdown=current_dd,
        predicted_risk_regime=predicted_risk_tomorrow,
        trend_state=trend_state,
        forecast_uncertainty_ratio=uncertainty_ratio
    )

    badge_color = procurement_advice["badge_color"]
    st.markdown(f"""
    <div class="advisory-panel" style="background:{badge_color}09; border-color:{badge_color}40;">
        <span class="advisory-action-badge" style="background:{badge_color};">
            {procurement_advice['action']}
        </span>
        <div class="advisory-headline">{procurement_advice['headline']}</div>
        <div class="advisory-sub">Automated corporate energy procurement signals based on real-time risk classification models.</div>
    </div>
    """, unsafe_allow_html=True)

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        with st.container(border=True):
            st.markdown("**🎯 Decision Justification**")
            for j in procurement_advice["justifications"]:
                st.markdown(f"- {j}")

            st.markdown("")
            st.markdown("**📊 Forecast Signal Summary**")
            signal_rows = [
                ("Asset Ticker",           f"`{ticker_option}`"),
                ("Current Volatility",     f"`{latest_vol:.2%}`"),
                ("Yesterday → Today",      f"`${prev_row['Close']:.2f}` → `${latest_close:.2f}`"),
                ("XGB Tomorrow Target",    f"`${next_day_price_tab4:.2f}`"),
                ("Uncertainty Ratio",      f"`{uncertainty_ratio:.2%}`"),
                ("Tomorrow Risk Forecast", f"`{predicted_risk_tomorrow.upper()} RISK`"),
            ]
            for label, val in signal_rows:
                st.markdown(f"- **{label}:** {val}")

    with adv_col2:
        with st.container(border=True):
            st.markdown("**✅ Tactical Execution Steps**")
            for tactic in procurement_advice["tactics"]:
                st.markdown(f"- {tactic}")

    st.markdown("---")

    # ── Procurement Sandbox ──
    section_header("Strategy Simulator", "Data Center Procurement Sandbox",
                   "Experiment with CFE (Carbon-Free Energy) targets and utility hedges over the past 60-day window.")

    sandbox_col1, sandbox_col2 = st.columns([1, 2])
    with sandbox_col1:
        with st.container(border=True):
            st.markdown("**⚙️ Simulator Settings**")
            st.caption(
                "Your data center runs at a constant **100 MW** load, 24/7. "
                "Adjust the clean spot allocation to see how it affects total spend, volatility, and CFE score."
            )
            custom_spot_pct = st.slider(
                "Clean Spot Allocation (%)",
                min_value=0, max_value=100, value=50, step=5
            )
            custom_utility_pct = 100 - custom_spot_pct
            st.caption(f"Utility Hedge: **{custom_utility_pct}%** | Spot: **{custom_spot_pct}%**")

    # Simulation engine
    sim_days = min(60, len(df_labeled))
    sim_df = df_labeled.iloc[-sim_days:].copy()
    spot_prices    = sim_df["Close"].values
    utility_prices = all_assets["XLU"].iloc[-sim_days:]["Close"].values
    regimes        = sim_df["Risk_Regime"].values

    costs_ai = []; cfe_ai = []
    costs_spot = []; cfe_spot = []
    costs_utility = []; cfe_utility = []
    costs_custom = []; cfe_custom = []

    for i in range(sim_days):
        sp, up = spot_prices[i], utility_prices[i]
        costs_spot.append(sp * 100 * 24);      cfe_spot.append(100.0)
        costs_utility.append(up * 100 * 24);   cfe_utility.append(20.0)
        costs_custom.append((sp*(custom_spot_pct/100) + up*(custom_utility_pct/100))*100*24)
        cfe_custom.append(100.0*(custom_spot_pct/100) + 20.0*(custom_utility_pct/100))
        regime = regimes[i]
        ai_s = 0.20 if regime == "High" else (0.50 if regime == "Medium" else 0.90)
        ai_u = 1 - ai_s
        costs_ai.append((sp*ai_s + up*ai_u)*100*24)
        cfe_ai.append(100.0*ai_s + 20.0*ai_u)

    tot_ai,   vol_ai,   cfe_ai_avg   = sum(costs_ai),   np.std(costs_ai),   np.mean(cfe_ai)
    tot_spot, vol_spot, cfe_spot_avg = sum(costs_spot), np.std(costs_spot), np.mean(cfe_spot)
    tot_util, vol_util, cfe_util_avg = sum(costs_utility), np.std(costs_utility), np.mean(cfe_utility)
    tot_cust, vol_cust, cfe_cust_avg = sum(costs_custom), np.std(costs_custom), np.mean(cfe_custom)

    leaderboard_data = sorted([
        {"Strategy": "🤖 EcoGrid AI",     "Total Cost": tot_ai,   "Volatility (SD)": vol_ai,   "Avg CFE": cfe_ai_avg},
        {"Strategy": "🌱 100% Green Spot","Total Cost": tot_spot, "Volatility (SD)": vol_spot, "Avg CFE": cfe_spot_avg},
        {"Strategy": "🏭 100% Utility",   "Total Cost": tot_util, "Volatility (SD)": vol_util, "Avg CFE": cfe_util_avg},
        {"Strategy": "🎨 Your Custom Mix","Total Cost": tot_cust, "Volatility (SD)": vol_cust, "Avg CFE": cfe_cust_avg},
    ], key=lambda x: x["Total Cost"])
    medals = ["🥇", "🥈", "🥉", "🎗️"]

    with sandbox_col2:
        st.markdown("**🏆 Strategy Leaderboard** — Ranked by total spend")
        rows_html = ""
        for idx, row in enumerate(leaderboard_data):
            rows_html += f"""
            <tr>
                <td>{medals[idx]} {idx+1}</td>
                <td><strong>{row['Strategy']}</strong></td>
                <td>${row['Total Cost']:,.0f}</td>
                <td>${row['Volatility (SD)']:,.0f}</td>
                <td>{row['Avg CFE']:.1f}%</td>
            </tr>"""
        st.markdown(f"""
        <table class="lb-table">
            <thead><tr>
                <th>Rank</th><th>Strategy</th><th>Total Spend</th>
                <th>Spend Volatility</th><th>Avg CFE Score</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Cumulative cost chart
    fig_cum = go.Figure()
    dates = sim_df.index
    fig_cum.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_ai),      name="EcoGrid AI",     line=dict(color="#10b981", width=3)))
    fig_cum.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_spot),    name="100% Spot",       line=dict(color="#3b82f6", width=2)))
    fig_cum.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_utility), name="100% Utility",    line=dict(color="#ef4444", width=2)))
    fig_cum.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_custom),  name="Your Custom Mix", line=dict(color="#f59e0b", width=2, dash="dash")))
    apply_chart_theme(fig_cum, "Cumulative Budget Expenditure Comparison",
                      f"{sim_days}-day window — EcoGrid AI vs. baseline strategies")
    st.plotly_chart(fig_cum, use_container_width=True)

    # Side-by-side bars
    perf_col1, perf_col2 = st.columns(2)
    ld_df = pd.DataFrame(leaderboard_data)
    color_map = {
        "🤖 EcoGrid AI": "#10b981", "🌱 100% Green Spot": "#3b82f6",
        "🏭 100% Utility": "#ef4444", "🎨 Your Custom Mix": "#f59e0b"
    }
    with perf_col1:
        fig_bc = px.bar(ld_df, x="Strategy", y="Total Cost", color="Strategy",
                        color_discrete_map=color_map)
        fig_bc.update_traces(marker_line_width=0)
        apply_chart_theme(fig_bc, "Total Spend ($)", "Lower is better")
        st.plotly_chart(fig_bc, use_container_width=True)
    with perf_col2:
        fig_bcfe = px.bar(ld_df, x="Strategy", y="Avg CFE", color="Strategy",
                          color_discrete_map=color_map)
        fig_bcfe.update_traces(marker_line_width=0)
        apply_chart_theme(fig_bcfe, "Average CFE Score (%)", "Higher is better")
        st.plotly_chart(fig_bcfe, use_container_width=True)

    # AI savings callout
    saved_amount    = tot_spot - tot_ai
    risk_reduction  = (vol_spot - vol_ai) / max(vol_spot, 1e-9) * 100
    st.info(
        f"💡 **EcoGrid AI saved ${saved_amount:,.0f}** in procurement costs vs. 100% green spot buying, "
        f"reduced daily spending volatility by **{risk_reduction:.1f}%**, "
        f"and maintained an average CFE match of **{cfe_ai_avg:.1f}%**."
    )

    st.markdown("---")

    # Export section
    section_header("Data Export", "Download Processed Dataset",
                   "Export the full feature-engineered time series for external analytics or database logging.")

    with st.expander("📥 Export Options", expanded=False):
        export_df = train_df.copy().reset_index()
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {ticker_option} Feature Dataset (.csv)",
            data=csv_data,
            file_name=f"ecogrid_ai_{ticker_option}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    render_insight_box(
        why_matters="Execution tactics translate statistical signals into physical grid operations, shielding margins from sudden spot-rate spikes.",
        what_changed=f"The optimization simulator runs over the past {sim_days} days of {ticker_option} returns, accounting for full volatility regime shifts.",
        what_to_do="Purchase compliance blocks when prices touch the lower 95% GARCH band. Suspend incremental spot buys if the Risk Score breaches 75."
    )


# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown("""
<div class="footer">
    <strong>EcoGrid AI</strong> · Green Asset Volatility Intelligence Platform<br>
    Built with Python · Streamlit · Prophet · XGBoost · GARCH · Plotly<br>
    <span style="color:#1e293b;">Production-grade financial intelligence for sustainable energy procurement</span>
</div>
""", unsafe_allow_html=True)
