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

# Set Page Config
st.set_page_config(
    page_title="EcoGrid AI: Real-Time Green Asset Volatility System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode Glassmorphism Theme Styling Injection
st.markdown(
    """
    <style>
    /* Main body styles */
    .main {
        background-color: #0d0f12;
        color: #f3f4f6;
    }
    
    /* Sidebar styling & white text override */
    section[data-testid="stSidebar"] {
        background-color: #15181e;
        border-right: 1px solid #232833;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Header card */
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #065f46 100%);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        color: #ffffff;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .header-subtitle {
        color: #93c5fd;
        font-size: 1.1rem;
        margin-top: 10px;
        font-weight: 400;
    }
    
    /* Custom KPI Cards */
    .kpi-card {
        background: #1c202a;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2b3242;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Recommendation Room */
    .rec-box {
        border-radius: 12px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    
    /* Alert badge inside card headers */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    
    /* Block container alignment */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# STREAMLIT CACHED DATA & MODEL WRAPPERS
# ==========================================

@st.cache_data(ttl=300)
def load_and_preprocess_data(ticker: str, period: str) -> tuple:
    """
    Fetches live data (caching it locally) or falls back to SQLite DB if API fails.
    """
    offline_mode = False
    try:
        # 1. Fetch live data
        df_raw = fetch_ticker_data(ticker, period)
        # 2. Write to SQLite cache
        cache_ticker_data(ticker, df_raw)
    except Exception as e:
        # Fallback to local DB cache
        df_raw = get_cached_ticker_data(ticker)
        if df_raw.empty:
            raise e
        offline_mode = True
        
    # Generate technical indicators
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
    """
    Decoupled model loader. Pulls trained weights from disk (models/ folder).
    Triggers batch training pipeline as a bootstrap step if files are missing.
    """
    xgb_path = f"models/{ticker}_xgb.pkl"
    risk_clf_path = f"models/{ticker}_risk_clf.pkl"
    
    if not os.path.exists(xgb_path) or not os.path.exists(risk_clf_path):
        with st.spinner("Initializing first-time model training bootstrap..."):
            run_training_pipeline()
            
    with open(f"models/{ticker}_xgb.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(f"models/{ticker}_risk_clf.pkl", "rb") as f:
        rf_clf = pickle.load(f)
    with open(f"models/{ticker}_risk_le.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(f"models/{ticker}_xgb_metrics.pkl", "rb") as f:
        xgb_metrics = pickle.load(f)
    with open(f"models/{ticker}_xgb_results.pkl", "rb") as f:
        test_results = pickle.load(f)
    with open(f"models/{ticker}_xgb_importance.pkl", "rb") as f:
        importance_df = pickle.load(f)
    with open(f"models/{ticker}_risk_metrics.pkl", "rb") as f:
        rf_metrics = pickle.load(f)
        
    return xgb_model, rf_clf, label_encoder, xgb_metrics, test_results, importance_df, rf_metrics


# ==========================================
# SIDEBAR FILTERS & SETTINGS
# ==========================================

st.sidebar.markdown("### 🛠️ System Controller")

ticker_option = st.sidebar.selectbox(
    "Select Target Asset / Index",
    options=["ICLN", "XLU", "CEG"],
    index=0,
    help="ICLN (Clean Energy ETF), XLU (Utilities ETF), CEG (Clean Energy Producer)"
)

lookback_option = st.sidebar.selectbox(
    "Historical Data Range",
    options=["1y", "2y", "5y"],
    index=2,
    help="Longer history improves 200-day moving averages and Prophet's annual seasonality fit."
)

horizon_option = st.sidebar.slider(
    "Prophet Forecast Horizon (Days)",
    min_value=30,
    max_value=90,
    value=60,
    step=15
)

# Clear Cache / Force Reload button
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Cache Control")
if st.sidebar.button("Force Pipeline Refresh"):
    st.cache_data.clear()
    st.cache_resource.clear()
    with st.spinner("Retraining batch ML models..."):
        run_training_pipeline()
    st.success("API Cache Cleared & Models Retrained!")
    st.rerun()

# Metadata display
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Last Sync Date:** {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.markdown("**Source:** Live Yahoo Finance API")


# ==========================================
# APP CONTAINER HEADER
# ==========================================

st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">⚡ EcoGrid AI</h1>
        <div class="header-subtitle">Real-Time Renewable Asset Risk Volatility & Forecasting Engine for Tech Data Centers</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Load data with loading spinner
with st.spinner("Loading pricing series and aligning indicators..."):
    try:
        df_labeled, risk_meta, offline_flag = load_and_preprocess_data(ticker_option, lookback_option)
        train_df, inference_row = get_ml_features(df_labeled)
        
        # Pull serialised pickles for inference
        xgb_model, rf_clf, label_encoder, xgb_metrics, test_results, importance_df, rf_metrics = load_serialized_models(ticker_option)
        
        # Load other assets for correlation calculations
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
                        st.error(f"Critical local cache missing for {tick}.")
                        st.stop()
                basic = compute_basic_features(raw)
                all_assets[tick] = basic
        
        correlation_matrix = compute_correlation_matrix(all_assets)
        
    except Exception as e:
        st.error(f"Error establishing connection: {e}")
        st.stop()

# Display offline status banner if API failed
if offline_flag:
    st.warning("⚠️ Connection to Yahoo Finance API failed. Running in offline fallback mode using cached database records.")

# ==========================================
# KPI PANEL (METRIC MATRIX)
# ==========================================

latest_row = df_labeled.iloc[-1]
prev_row = df_labeled.iloc[-2]

latest_close = latest_row["Close"]
price_pct_change = (latest_close - prev_row["Close"]) / prev_row["Close"] * 100

latest_vol = latest_row["Roll_Vol_30d"]
vol_change = latest_vol - prev_row["Roll_Vol_30d"]

current_risk = latest_row["Risk_Regime"]
current_drawdown = latest_row["Drawdown"]

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Current Market Price</div>
            <div class="kpi-value" style="color: #3b82f6;">${latest_close:.2f}</div>
            <div class="kpi-delta" style="color: {'#10b981' if price_pct_change >=0 else '#ef4444'};">
                {'▲' if price_pct_change >= 0 else '▼'} {abs(price_pct_change):.2f}% (Daily)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_cols[1]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Annualized Volatility (30D)</div>
            <div class="kpi-value" style="color: #f59e0b;">{latest_vol:.1%}</div>
            <div class="kpi-delta" style="color: {'#ef4444' if vol_change >=0 else '#10b981'};">
                {'▲' if vol_change >= 0 else '▼'} {abs(vol_change)*100:.2f}% vs. Yesterday
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_cols[2]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Maximum Drawdown</div>
            <div class="kpi-value" style="color: #ec4899;">{current_drawdown:.2%}</div>
            <div class="kpi-delta" style="color: #9ca3af;">Peak: ${df_labeled['Close'].max():.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_cols[3]:
    risk_color = "#10B981" if current_risk == "Low" else ("#F59E0B" if current_risk == "Medium" else "#EF4444")
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Current Risk State</div>
            <div class="kpi-value" style="color: {risk_color};">{current_risk.upper()} RISK</div>
            <div class="kpi-delta" style="color: #9ca3af;">Dynamic Threshold Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# ==========================================
# TABS INTEGRATION
# ==========================================

tabs = st.tabs([
    "🎮 Procurement Sandbox Game",
    "🧠 1-Minute Energy Primer",
    "📊 Real-Time Market EDA",
    "📈 Time Series Forecasting (Prophet+GARCH)",
    "🤖 Price Predictor (XGBoost Returns)",
    "⚠️ Risk Regime Classifier",
    "💡 Procurement Decision Room"
])

# ------------------------------------------
# TAB 1: 🎮 PROCUREMENT SANDBOX GAME
# ------------------------------------------
with tabs[0]:
    st.markdown("### 🎮 Data Center Procurement Sandbox Simulator")
    st.caption("Step into the role of a Tech Data Center Energy Procurement Director. Experiment with hedging strategies using actual live market rates over the past 60 days.")
    
    with st.expander("📖 Read Simulator Rules & Strategic Objectives"):
        st.markdown(
            """
            **The Strategic Objective:**
            Your data center consumes a constant load of **100 MW** daily (around the clock). 
            You must secure energy contracts to balance three core performance vectors:
            1. **Minimize Budget Cost**: Keep total energy procurement expenses low.
            2. **Minimize Budget Volatility**: Reduce standard deviation of daily energy spending.
            3. **Maximize Carbon-Free Energy (CFE) Match**: Power your operations using clean energy.
            
            **Available Contract Vectors:**
            - **Clean Spot Energy** (linked to your selected Clean Asset price e.g., `ICLN`/`CEG`): Matches 100% CFE, but prices shift daily.
            - **Regulated Utility Grid** (tracked by defensive utilities ETF `XLU`): Stable prices, but carbon-heavy (only 20% CFE matching credit).
            
            **Strategies Compared:**
            * 🤖 **EcoGrid AI Optimizer**: Dynamic allocation shifting between Spot and Utility depending on model risk regimes.
            * 🔵 **Green-At-All-Costs**: 100% Clean Spot Energy allocation.
            * 🔴 **Fossil Regulated Hedger**: 100% Regulated Utility Grid allocation.
            * 🟡 **Your Custom Allocation**: A fixed compliance ratio set by your slider choice.
            """
        )
        
    st.markdown("#### ⚙️ Set Custom Strategy Mix")
    custom_spot_pct = st.slider("Custom Clean Spot Allocation (%)", min_value=0, max_value=100, value=50, step=5)
    custom_utility_pct = 100 - custom_spot_pct
    
    if st.button("⚡ Run Live Market Simulation", key="run_sim"):
        # Run over last 60 days of historical data
        sim_days = min(60, len(df_labeled))
        sim_df = df_labeled.iloc[-sim_days:].copy()
        
        # Pull Prices
        spot_prices = sim_df["Close"].values
        utility_prices = all_assets["XLU"].iloc[-sim_days:]["Close"].values
        regimes = sim_df["Risk_Regime"].values
        
        # Storage trackers
        costs_ai, cfe_ai = [], []
        costs_spot, cfe_spot = [], []
        costs_utility, cfe_utility = [], []
        costs_custom, cfe_custom = [], []
        
        for i in range(sim_days):
            sp = spot_prices[i]
            up = utility_prices[i]
            
            # 1. 100% Spot (Green)
            c_spot = sp * 100 * 24
            costs_spot.append(c_spot)
            cfe_spot.append(100.0)
            
            # 2. 100% Utility (Hedge)
            c_util = up * 100 * 24
            costs_utility.append(c_util)
            cfe_utility.append(20.0)
            
            # 3. Custom Mix
            c_cust = (sp * (custom_spot_pct / 100) + up * (custom_utility_pct / 100)) * 100 * 24
            costs_custom.append(c_cust)
            cfe_custom.append((100.0 * (custom_spot_pct / 100)) + (20.0 * (custom_utility_pct / 100)))
            
            # 4. EcoGrid AI Dynamic Allocation
            regime = regimes[i]
            if regime == "High":
                ai_spot_allocation = 0.20
                ai_utility_allocation = 0.80
            elif regime == "Medium":
                ai_spot_allocation = 0.50
                ai_utility_allocation = 0.50
            else: # Low Risk
                ai_spot_allocation = 0.90
                ai_utility_allocation = 0.10
                
            c_ai = (sp * ai_spot_allocation + up * ai_utility_allocation) * 100 * 24
            costs_ai.append(c_ai)
            cfe_ai.append((100.0 * ai_spot_allocation) + (20.0 * ai_utility_allocation))
            
        # Metrics compiled
        tot_ai, vol_ai, cfe_ai_avg = sum(costs_ai), np.std(costs_ai), np.mean(cfe_ai)
        tot_spot, vol_spot, cfe_spot_avg = sum(costs_spot), np.std(costs_spot), np.mean(cfe_spot)
        tot_util, vol_util, cfe_util_avg = sum(costs_utility), np.std(costs_utility), np.mean(cfe_utility)
        tot_cust, vol_cust, cfe_cust_avg = sum(costs_custom), np.std(costs_custom), np.mean(cfe_custom)
        
        # Leaderboard compilation
        leaderboard_data = [
            {"Strategy": "🤖 EcoGrid AI Optimizer", "Total Cost": tot_ai, "Cost Volatility (SD)": vol_ai, "Avg CFE Match": cfe_ai_avg},
            {"Strategy": "🔵 Green Spot (100% Clean)", "Total Cost": tot_spot, "Cost Volatility (SD)": vol_spot, "Avg CFE Match": cfe_spot_avg},
            {"Strategy": "🔴 Utility Hedge (100% Hedge)", "Total Cost": tot_util, "Cost Volatility (SD)": vol_util, "Avg CFE Match": cfe_util_avg},
            {"Strategy": "🟡 Your Custom Mix", "Total Cost": tot_cust, "Cost Volatility (SD)": vol_cust, "Avg CFE Match": cfe_cust_avg}
        ]
        
        # Sort leaderboard by total cost ascending
        leaderboard_data = sorted(leaderboard_data, key=lambda x: x["Total Cost"])
        
        # Assign rank and medals
        ranks = [1, 2, 3, 4]
        medals = ["🥇", "🥈", "🥉", "🎗️"]
        for idx, item in enumerate(leaderboard_data):
            item["Rank"] = ranks[idx]
            item["Medal"] = medals[idx]
            
        ld_df = pd.DataFrame(leaderboard_data)
        
        st.markdown("### 🏆 Performance Leaderboard")
        st.table(ld_df[["Rank", "Medal", "Strategy", "Total Cost", "Cost Volatility (SD)", "Avg CFE Match"]].style.format({
            "Total Cost": "${:,.2f}",
            "Cost Volatility (SD)": "${:,.2f}",
            "Avg CFE Match": "{:.1f}%"
        }))
        
        # Visualizing cumulative cost
        fig_cum_cost = go.Figure()
        dates = sim_df.index
        fig_cum_cost.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_ai), name="EcoGrid AI Dynamic", line=dict(color="#10b981", width=2.5)))
        fig_cum_cost.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_spot), name="100% Spot (Green)", line=dict(color="#3b82f6", width=2)))
        fig_cum_cost.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_utility), name="100% Utility (Hedge)", line=dict(color="#ef4444", width=2)))
        fig_cum_cost.add_trace(go.Scatter(x=dates, y=np.cumsum(costs_custom), name="Your Custom Mix", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig_cum_cost.update_layout(
            title="Cumulative Budget Expenditure Comparison (60-Day Window)",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Total Cost ($)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_cum_cost, use_container_width=True)
        
        # Visualizing Cost vs CFE Match
        st.markdown("#### Cost vs CFE Match Side-by-Side")
        plot_cols = st.columns(2)
        with plot_cols[0]:
            fig_bar_cost = px.bar(
                ld_df, x="Strategy", y="Total Cost", 
                title="Total Cost ($)", 
                color="Strategy",
                color_discrete_map={
                    "🤖 EcoGrid AI Optimizer": "#10b981",
                    "🔵 Green Spot (100% Clean)": "#3b82f6",
                    "🔴 Utility Hedge (100% Hedge)": "#ef4444",
                    "🟡 Your Custom Mix": "#f59e0b"
                }
            )
            fig_bar_cost.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bar_cost, use_container_width=True)
        with plot_cols[1]:
            fig_bar_cfe = px.bar(
                ld_df, x="Strategy", y="Avg CFE Match", 
                title="Carbon-Free Energy Matching (%)", 
                color="Strategy",
                color_discrete_map={
                    "🤖 EcoGrid AI Optimizer": "#10b981",
                    "🔵 Green Spot (100% Clean)": "#3b82f6",
                    "🔴 Utility Hedge (100% Hedge)": "#ef4444",
                    "🟡 Your Custom Mix": "#f59e0b"
                }
            )
            fig_bar_cfe.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bar_cfe, use_container_width=True)
            
        # Dynamic AI commentary
        saved_amount = tot_spot - tot_ai
        risk_reduction = (vol_spot - vol_ai) / (vol_spot + 1e-9) * 100
        
        st.info(
            f"💡 **Dynamic AI Insight:** The **EcoGrid AI Dynamic Strategy** saved your data center **${saved_amount:,.2f}** "
            f"in procurement costs relative to the Green Spot strategy and reduced daily budget volatility "
            f"by **{risk_reduction:.1f}%**, all while establishing a healthy CFE match score of **{cfe_ai_avg:.1f}%**."
        )


# ------------------------------------------
# TAB 2: 🧠 1-MINUTE ENERGY PRIMER
# ------------------------------------------
with tabs[1]:
    st.markdown("### 🧠 1-Minute Energy Portfolio Primer")
    st.caption("New to energy markets? Play with interactive sliders below to understand the core financial concepts used in this platform.")
    
    concept_choice = st.selectbox(
        "Select a Concept to Visualize",
        options=[
            "📈 Volatility (Market Turbulence)",
            "📉 Drawdown (Peak-to-Trough Decline)",
            "🛡️ Hedging (Risk Mitigation)",
            "🌿 24/7 Carbon-Free Energy (CFE Match)"
        ]
    )
    
    primer_cols = st.columns([1, 2])
    
    with primer_cols[0]:
        if "Volatility" in concept_choice:
            st.markdown("#### What is Volatility?")
            st.markdown(
                """
                Volatility measures how wildly an asset price swings up and down. 
                * **High Volatility**: Prices bounce around rapidly (e.g. Clean Energy ICLN during weather extremes).
                * **Low Volatility**: Prices change slowly and steadily (e.g. Regulated Utilities XLU).
                
                **✈️ The Analogy:**
                Think of volatility as flight turbulence. Regulated utilities are a smooth cruise; clean energy is flying through a storm.
                """
            )
            vol_slider = st.slider("Set Turbulence Level (Volatility)", min_value=5, max_value=80, value=25, step=5)
            
        elif "Drawdown" in concept_choice:
            st.markdown("#### What is Drawdown?")
            st.markdown(
                """
                Drawdown measures the peak-to-trough decline of an asset. It tells energy managers: 
                *\"If I bought at the absolute peak, how much money did I lose at the bottom?\"*
                
                **📉 Why it matters:**
                Data center budgets cannot sustain sudden 30% drops in asset values. Tracking drawdown helps set risk limits.
                """
            )
            dd_slider = st.slider("Simulate Peak Drop (%)", min_value=0, max_value=50, value=20, step=5)
            
        elif "Hedging" in concept_choice:
            st.markdown("#### What is Hedging?")
            st.markdown(
                """
                Hedging means buying a defensive asset (like utilities) to offset losses in your primary asset (clean energy).
                
                **🛡️ The Strategy:**
                When clean energy volatility is high, we dynamically buy utility power to smooth out our total cost curve.
                """
            )
            hedge_ratio = st.slider("Hedge Ratio (Utility Percentage)", min_value=0, max_value=100, value=40, step=10)
            
        else: # CFE
            st.markdown("#### What is 24/7 CFE Match?")
            st.markdown(
                """
                24/7 Carbon-Free Energy (CFE) matching means sourcing zero-carbon electricity to power your load every single hour of the day.
                
                * **Spot solar/wind**: 100% CFE match.
                * **Utility Grid mix**: 20% CFE match (grid has coal/gas baseline).
                
                **🌿 The Goal:**
                Maximize CFE matching while keeping energy costs stable.
                """
            )
            cfe_slider = st.slider("Clean Energy Mix Ratio (%)", min_value=0, max_value=100, value=70, step=10)
            
    with primer_cols[1]:
        if "Volatility" in concept_choice:
            np.random.seed(42)
            days = 60
            noise = np.random.normal(0, vol_slider / 100, days)
            prices = 100 * np.exp(np.cumsum(noise - 0.5 * (vol_slider / 100) ** 2))
            
            fig_vol_sim = px.line(
                x=pd.date_range(end=datetime.now(), periods=days),
                y=prices,
                title=f"Synthetic Asset Path (Volatility: {vol_slider}%)",
                labels={"x": "Day", "y": "Mock Price ($)"}
            )
            fig_vol_sim.update_traces(line_color="#f59e0b", line_width=2)
            fig_vol_sim.update_layout(template="plotly_dark")
            st.plotly_chart(fig_vol_sim, use_container_width=True)
            
        elif "Drawdown" in concept_choice:
            np.random.seed(10)
            days = 30
            base = np.linspace(100, 100 - dd_slider, days)
            prices = base + np.random.normal(0, 1.5, days)
            prices[0] = 100
            prices[-1] = 100 - dd_slider
            
            fig_dd_sim = go.Figure()
            dates = pd.date_range(end=datetime.now(), periods=days)
            fig_dd_sim.add_trace(go.Scatter(x=dates, y=prices, name="Price Path", line=dict(color="#ec4899", width=2)))
            fig_dd_sim.add_trace(go.Scatter(x=[dates[0]], y=[prices[0]], name="Peak ($100)", mode="markers", marker=dict(size=12, color="#10b981")))
            fig_dd_sim.add_trace(go.Scatter(x=[dates[-1]], y=[prices[-1]], name=f"Trough (${prices[-1]:.1f})", mode="markers", marker=dict(size=12, color="#ef4444")))
            fig_dd_sim.update_layout(
                title=f"Drawdown Simulation (Max Drop: -{dd_slider}%)",
                template="plotly_dark",
                xaxis_title="Day",
                yaxis_title="Mock Price ($)"
            )
            st.plotly_chart(fig_dd_sim, use_container_width=True)
            
        elif "Hedging" in concept_choice:
            np.random.seed(100)
            days = 60
            spot_noise = np.random.normal(0, 0.05, days)
            util_noise = np.random.normal(0, 0.015, days)
            
            spot_prices = 100 * np.exp(np.cumsum(spot_noise - 0.5 * 0.05**2))
            util_prices = 100 * np.exp(np.cumsum(util_noise - 0.5 * 0.015**2))
            
            hedged_prices = (spot_prices * (1 - hedge_ratio/100)) + (util_prices * (hedge_ratio/100))
            
            fig_hedge_sim = go.Figure()
            dates = pd.date_range(end=datetime.now(), periods=days)
            fig_hedge_sim.add_trace(go.Scatter(x=dates, y=spot_prices, name="Clean Spot (Volatile)", line=dict(color="#3b82f6", width=1.5, dash="dot")))
            fig_hedge_sim.add_trace(go.Scatter(x=dates, y=util_prices, name="Utility (Stable)", line=dict(color="#ef4444", width=1.5, dash="dot")))
            fig_hedge_sim.add_trace(go.Scatter(x=dates, y=hedged_prices, name=f"Hedged Mix ({hedge_ratio}% Utility)", line=dict(color="#10b981", width=3)))
            fig_hedge_sim.update_layout(
                title="Smoothing the Price Path using Hedging",
                template="plotly_dark",
                xaxis_title="Day",
                yaxis_title="Mock Price ($)"
            )
            st.plotly_chart(fig_hedge_sim, use_container_width=True)
            
        else: # CFE
            clean_contrib = cfe_slider
            fossil_contrib = 100 - cfe_slider
            cfe_percentage = (clean_contrib * 1.0) + (fossil_contrib * 0.2)
            
            fig_cfe_sim = px.pie(
                names=["Carbon-Free Energy Source", "Fossil Grid Energy Source"],
                values=[cfe_percentage, 100 - cfe_percentage],
                title=f"Total Hourly Energy Mix (CFE Score: {cfe_percentage:.1f}%)",
                color_discrete_sequence=["#10b981", "#374151"]
            )
            fig_cfe_sim.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cfe_sim, use_container_width=True)


# ------------------------------------------
# TAB 3: REAL-TIME MARKET EDA
# ------------------------------------------
with tabs[2]:
    st.markdown("### Market Intelligence Exploratory Analysis")
    st.caption("All visual analytics update dynamically with every incoming data refresh.")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        # 1. Price Trends (Closing Prices & Moving Averages)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_labeled.index, y=df_labeled["Close"], name="Closing Price", line=dict(color="#3b82f6", width=2)))
        fig_price.add_trace(go.Scatter(x=df_labeled.index, y=df_labeled["SMA_50"], name="50-day SMA", line=dict(color="#f59e0b", width=1.5, dash="dash")))
        fig_price.add_trace(go.Scatter(x=df_labeled.index, y=df_labeled["SMA_200"], name="200-day SMA", line=dict(color="#ec4899", width=1.5, dash="dot")))
        fig_price.update_layout(
            title=f"{ticker_option} Historical Price and Moving Averages",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # 2. Drawdowns
        fig_dd = px.area(df_labeled, x=df_labeled.index, y="Drawdown", title=f"{ticker_option} Peak-to-Trough Drawdown Curve")
        fig_dd.update_traces(line_color="#ef4444", fillcolor="rgba(239, 68, 68, 0.2)")
        fig_dd.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)
        
    with eda_col2:
        # 3. Volatility Windows comparison
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df_labeled.index, y=df_labeled["Roll_Vol_7d"], name="7-day Vol (Short)", line=dict(color="#34d399", width=1.5)))
        fig_vol.add_trace(go.Scatter(x=df_labeled.index, y=df_labeled["Roll_Vol_30d"], name="30-day Vol (Long)", line=dict(color="#f59e0b", width=2)))
        fig_vol.update_layout(
            title=f"{ticker_option} Annualized Rolling Volatility",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # 4. Correlation Heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Dynamic Asset Correlation Heatmap (Daily Returns)"
        )
        fig_corr.update_layout(template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)


# ------------------------------------------
# TAB 4: TIME SERIES FORECASTING (PROPHET+GARCH)
# ------------------------------------------
with tabs[3]:
    st.markdown("### Prophet + GARCH(1,1) Dynamic Volatility Forecasting")
    st.caption("Hybrid forecasting: Prophet models the additive trend, while a custom maximum-likelihood GARCH(1,1) solver scales forecast uncertainty bands dynamically.")
    
    with st.spinner("Fitting Prophet + GARCH residuals..."):
        model, forecast, prophet_metrics = run_prophet_model(df_labeled, horizon_option)
    
    # Showcase Error Metrics
    met_cols = st.columns(3)
    met_cols[0].metric("In-Sample MAE", f"${prophet_metrics['MAE']:.2f}", help="Mean Absolute Error in dollar terms.")
    met_cols[1].metric("In-Sample RMSE", f"${prophet_metrics['RMSE']:.2f}", help="Root Mean Squared Error.")
    met_cols[2].metric("In-Sample MAPE", f"{prophet_metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error.")
    
    # 1. Plot Forecast with confidence bands
    fig_fc = go.Figure()
    # Actuals
    fig_fc.add_trace(go.Scatter(x=df_labeled.index, y=df_labeled["Close"], name="Actual Close", line=dict(color="#3b82f6", width=2)))
    # Forecasted
    future_forecast = forecast.iloc[-horizon_option:]
    fig_fc.add_trace(go.Scatter(x=future_forecast["ds"], y=future_forecast["yhat"], name="Forecasted Trend", line=dict(color="#10b981", width=2)))
    # Confidence interval (Shaded)
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([future_forecast["ds"], future_forecast["ds"][::-1]]),
        y=pd.concat([future_forecast["yhat_upper"], future_forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Dynamic GARCH Band"
    ))
    
    fig_fc.update_layout(
        title=f"Prophet + GARCH {horizon_option}-Day Dynamic Forecast Projection for {ticker_option}",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # 2. Seasonality Components
    st.markdown("#### Seasonality Breakdowns")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        weekly = forecast.groupby(forecast["ds"].dt.dayofweek)["weekly"].mean().reset_index()
        weekly["Day"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig_week = px.bar(weekly, x="Day", y="weekly", title="Weekly Seasonality Component Effect", color_discrete_sequence=["#10b981"])
        fig_week.update_layout(template="plotly_dark", yaxis_title="Additive Adjustment ($)")
        st.plotly_chart(fig_week, use_container_width=True)
        
    with comp_col2:
        yearly = forecast.groupby(forecast["ds"].dt.month)["yearly"].mean().reset_index()
        yearly["Month"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig_year = px.line(yearly, x="Month", y="yearly", title="Annual Seasonality Component Effect", color_discrete_sequence=["#f59e0b"])
        fig_year.update_layout(template="plotly_dark", yaxis_title="Additive Adjustment ($)")
        st.plotly_chart(fig_year, use_container_width=True)


# ------------------------------------------
# TAB 5: PRICE PREDICTOR (XGBOOST RETURNS)
# ------------------------------------------
with tabs[4]:
    st.markdown("### Next-Day Price Forecasting via XGBoost (Log Returns)")
    st.caption("Decoupled supervised model. Trained on stationary log returns to extrapolate new pricing regimes correctly.")
    
    xgb_cols = st.columns(4)
    xgb_cols[0].metric("Validation Set MAE", f"${xgb_metrics['MAE']:.2f}")
    xgb_cols[1].metric("Validation Set RMSE", f"${xgb_metrics['RMSE']:.2f}")
    xgb_cols[2].metric("Validation Set R² Score", f"{xgb_metrics['R2']:.4f}")
    xgb_cols[3].metric("Directional Accuracy (Sign)", f"{xgb_metrics['Directional_Accuracy']:.1f}%")
    
    st.markdown("---")
    
    xgb_viz1, xgb_viz2 = st.columns([2, 1])
    
    with xgb_viz1:
        fig_xgb = go.Figure()
        fig_xgb.add_trace(go.Scatter(x=test_results.index, y=test_results["Actual"], name="Actual Price", line=dict(color="#3b82f6", width=2)))
        fig_xgb.add_trace(go.Scatter(x=test_results.index, y=test_results["Predicted"], name="XGBoost Reconstructed Price", line=dict(color="#ec4899", width=1.5, dash="dash")))
        fig_xgb.update_layout(
            title="XGBoost Walk-Forward Validation Predictions vs. Actual Closing Prices (Unseen Test Window)",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_xgb, use_container_width=True)
        
    with xgb_viz2:
        fig_imp = px.bar(
            importance_df.sort_values(by="Importance", ascending=True),
            y="Feature",
            x="Importance",
            orientation="h",
            title="XGBoost Feature Importance Ranking",
            color_discrete_sequence=["#ec4899"]
        )
        fig_imp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)


# ------------------------------------------
# TAB 6: RISK REGIME CLASSIFIER
# ------------------------------------------
with tabs[5]:
    st.markdown("### Risk Regime Classification Engine")
    st.caption("Supervised Random Forest classifier mapping multivariate pricing states to Low, Medium, and High volatility risk bands.")
    
    st.markdown(f"#### Classifier Accuracy: `{rf_metrics['Accuracy']:.1%}`")
    
    # Showcase classification report metrics
    st.markdown("##### Out-of-Sample Validation Report (Test Set)")
    report_data = []
    for label, metrics in rf_metrics["Report"].items():
        if label in ["Low", "Medium", "High"]:
            report_data.append({
                "Risk Regime": label,
                "Precision": f"{metrics['precision']:.1%}",
                "Recall": f"{metrics['recall']:.1%}",
                "F1-Score": f"{metrics['f1-score']:.1%}",
                "Support": int(metrics['support'])
            })
    st.table(pd.DataFrame(report_data))
    
    # Visualizing Volatility Regimes historically
    fig_regimes = px.scatter(
        df_labeled,
        x=df_labeled.index,
        y="Close",
        color="Risk_Regime",
        color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
        title="Historical Risk Volatility Regimes Across Assets",
        labels={"color": "Risk Level"}
    )
    fig_regimes.update_traces(marker=dict(size=4))
    fig_regimes.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig_regimes, use_container_width=True)


# ------------------------------------------
# TAB 7: PROCUREMENT DECISION ROOM
# ------------------------------------------
with tabs[6]:
    st.markdown("### Automated Energy Procurement Advisory Board")
    st.caption("Rule-based business logic converting predictive modeling states into corporate hedging strategies.")
    
    # 1. Fetch Tomorrow predictions using our true (latest) inference row and latest close price scale
    next_day_price = predict_next_day_xgb(xgb_model, inference_row, latest_close)
    predicted_risk_tomorrow = predict_next_day_risk(rf_clf, label_encoder, inference_row)
    
    # 2. Determine short-term trend direction from SMA ratio
    trend_state = "Bullish" if latest_row["SMA_50_Ratio"] > 0 else "Bearish"
    
    # 3. Determine uncertainty ratio from Prophet forecast GARCH intervals
    latest_fc = forecast[forecast["ds"] <= df_labeled.index[-1]].iloc[-1]
    uncertainty_ratio = (latest_fc["yhat_upper"] - latest_fc["yhat_lower"]) / latest_fc["yhat"]
    
    # Generate advice
    procurement_advice = get_procurement_advice(
        ticker=ticker_option,
        current_price=latest_close,
        rolling_vol_30d=latest_vol,
        current_drawdown=current_drawdown,
        predicted_risk_regime=predicted_risk_tomorrow,
        trend_state=trend_state,
        forecast_uncertainty_ratio=uncertainty_ratio
    )
    
    # Render advisory room
    st.markdown(
        f"""
        <div class="rec-box" style="background: {procurement_advice['badge_color']}1a; border-color: {procurement_advice['badge_color']};">
            <span class="badge" style="background-color: {procurement_advice['badge_color']}; color: #ffffff;">
                RECOMMENDED ACTION: {procurement_advice['action']}
            </span>
            <h3 style="margin: 10px 0; color: #ffffff;">{procurement_advice['headline']}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("#### 🎯 Core Decision Justification")
        for justification in procurement_advice["justifications"]:
            st.markdown(f"- {justification}")
            
        st.markdown("#### 🔮 Forecast Signals Summary")
        st.markdown(f"- **Current Asset Ticker:** `{ticker_option}`")
        st.markdown(f"- **Current Volatility:** `{latest_vol:.2%}`")
        st.markdown(f"- **Yesterday Close:** `${prev_row['Close']:.2f}` → **Today Close:** `${latest_close:.2f}`")
        st.markdown(f"- **XGBoost Tomorrow Price Target:** `${next_day_price:.2f}`")
        st.markdown(f"- **Prophet + GARCH Uncertainty Ratio:** `{uncertainty_ratio:.2%}`")
        st.markdown(f"- **Classification Engine Risk Prediction for Tomorrow:** `{predicted_risk_tomorrow.upper()} RISK`")
        
    with rec_col2:
        st.markdown("#### 🛠️ Tactical Execution Steps")
        for tactic in procurement_advice["tactics"]:
            st.markdown(f"✅ **{tactic}**")


# ==========================================
# REPORT DOWNLOAD SECTION
# ==========================================

st.markdown("---")
st.markdown("### 📥 Export Analytical Report")
st.caption("Generate and download a snapshot of the processed data vectors for external tooling or databases.")

# Compile downloadable table
export_df = train_df.copy()
export_df = export_df.reset_index()

# Download actions
csv_data = export_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Processed Dataset as CSV",
    data=csv_data,
    file_name=f"ecogrid_ai_{ticker_option}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: #4b5563; font-size: 0.85rem;">
        EcoGrid AI Platform • Engineered as a Production-Ready Financial Intelligence System • Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
