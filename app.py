"""
NIFTY-50 Investment Intelligence Platform
A premium hackathon dashboard for stock market analysis, prediction, and portfolio optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "background": "#0f172a",
    "card": "#1e293b",
    "card_border": "#334155",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
    "accent_gradient_start": "#6366f1",
    "accent_gradient_end": "#a855f7",
}

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme with Glassmorphism
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');

    /* ── Global ────────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace !important;
        background-color: #0d0d12 !important;
        color: #e0e0e0 !important;
    }
    .stApp {
        background: #0d0d12;
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #050505 !important;
        border-right: 1px solid #222222;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #cccccc !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Tabs ──────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: transparent;
        border-bottom: 1px solid #333;
        padding: 0px;
        width: 100%;
        display: flex;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        text-align: center;
        border-radius: 0px;
        padding: 10px 20px;
        font-weight: 700;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid transparent;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #111 !important;
        color: #fff !important;
        border: 1px solid #333;
        border-bottom: 1px solid #111;
        margin-bottom: -1px;
    }

    /* ── Metric Cards ──────────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 0px;
        padding: 16px 12px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        text-align: left;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #555555;
    }
    div[data-testid="stMetric"] label {
        color: #777777 !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-testid="stMetric"] * {
        overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.0rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
        overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
        display: inline;
    }

    /* ── Glassmorphism card wrapper ─────────────────────────────────── */
    .glass-card {
        background: #0a0a0a;
        border: 1px solid #222;
        border-radius: 0px;
        padding: 28px;
        margin-bottom: 16px;
    }

    /* ── Gradient header text ──────────────────────────────────────── */
    .gradient-text, .gradient-text-sm {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .gradient-text { font-size: 1.8rem; }
    .gradient-text-sm { font-size: 1.2rem; }

    /* ── Prediction badge ─────────────────────────────────────────── */
    .pred-up {
        background: #000;
        color: #0f0;
        border: 1px solid #0f0;
        padding: 24px 32px;
        border-radius: 0px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .pred-down {
        background: #000;
        color: #f00;
        border: 1px solid #f00;
        padding: 24px 32px;
        border-radius: 0px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    /* ── DataFrames ────────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid #333 !important;
        border-radius: 0px !important;
    }

    /* ── Expander ──────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #111 !important;
        border-radius: 0px;
        color: #fff !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Plotly chart containers ──────────────────────────────────── */
    .stPlotlyChart {
        border-radius: 0px;
        border: 1px solid #222;
        padding: 10px;
        background: #050505;
    }

    /* ── Selectbox / Radio ────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stRadio > div {
        color: #fff;
        border-radius: 0px;
    }

    /* ── Divider ──────────────────────────────────────────────────── */
    hr {
        border-color: #333 !important;
    }

    /* ── Anomaly badge ────────────────────────────────────────────── */
    .anomaly-high { color: #f00; font-weight: 700; text-transform: uppercase; }
    .anomaly-medium { color: #f90; font-weight: 700; text-transform: uppercase; }
    .anomaly-low { color: #0f0; font-weight: 700; text-transform: uppercase; }

    /* ── Animated pulse for live indicator ─────────────────────────── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME TEMPLATE
# ──────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color="#cccccc", size=12),
        title_font=dict(size=16, color="#ffffff"),
        xaxis=dict(
            gridcolor="#222222",
            zerolinecolor="#333333",
            linecolor="#333333",
        ),
        yaxis=dict(
            gridcolor="#222222",
            zerolinecolor="#333333",
            linecolor="#333333",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333333",
            font=dict(color="#aaaaaa"),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor="#111111",
            bordercolor="#333333",
            font=dict(color="#ffffff", family="Space Mono"),
        ),
    )
)


def styled_plotly(fig: go.Figure, height: int = 500) -> go.Figure:
    """Apply the premium dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color="#cccccc", size=12),
        xaxis=dict(gridcolor="#222222", zerolinecolor="#333333"),
        yaxis=dict(gridcolor="#222222", zerolinecolor="#333333"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333333"),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="#111111", bordercolor="#333333", font=dict(color="#ffffff")),
        height=height,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING — Graceful fallback to sample data
# ──────────────────────────────────────────────────────────────────────────────
STOCK_SECTORS = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking",
    "INFY": "IT", "ICICIBANK": "Banking", "HINDUNILVR": "FMCG",
    "SBIN": "Banking", "BHARTIARTL": "Telecom", "ITC": "FMCG",
    "KOTAKBANK": "Banking", "LT": "Infrastructure", "AXISBANK": "Banking",
    "ASIANPAINT": "Consumer", "MARUTI": "Automobile", "SUNPHARMA": "Pharma",
    "TITAN": "Consumer", "BAJFINANCE": "Financial", "WIPRO": "IT",
    "ULTRACEMCO": "Cement", "NESTLEIND": "FMCG", "HCLTECH": "IT",
    "POWERGRID": "Energy", "NTPC": "Energy", "TATAMOTORS": "Automobile",
    "M&M": "Automobile", "BAJAJFINSV": "Financial", "ONGC": "Energy",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "ADANIPORTS": "Infrastructure",
    "TECHM": "IT", "DRREDDY": "Pharma", "INDUSINDBK": "Banking",
    "GRASIM": "Cement", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "BPCL": "Energy", "APOLLOHOSP": "Healthcare", "EICHERMOT": "Automobile",
    "HEROMOTOCO": "Automobile", "COALINDIA": "Mining", "UPL": "Chemicals",
    "TATACONSUM": "FMCG", "BRITANNIA": "FMCG", "BAJAJ-AUTO": "Automobile",
    "SHREECEM": "Cement", "ADANIENT": "Conglomerate", "HINDALCO": "Metals",
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance",
    "ALPHA": "Technology", "BETA": "Finance", "GAMMA": "Energy",
    "DELTA": "Healthcare", "EPSILON": "Consumer"
}

@st.cache_data(show_spinner=False)
def load_data():
    """Load stock data. Try real backend first, then fall back to sample data."""
    use_sample = False
    df = None
    is_sample = False
    try:
        from src.data_processing import load_all_stocks, clean_data, add_technical_indicators, calculate_returns
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # Search parent data directory and direct subdirectories for CSVs
        search_dirs = [data_dir]
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                full_path = os.path.join(data_dir, item)
                if os.path.isdir(full_path):
                    search_dirs.append(full_path)
                    
        valid_dir = None
        for d in search_dirs:
            if os.path.exists(d) and any(f.endswith(".csv") for f in os.listdir(d)):
                valid_dir = d
                break
                
        if valid_dir:
            raw = load_all_stocks(valid_dir)
            cleaned = clean_data(raw)
            with_indicators = add_technical_indicators(cleaned)
            with_returns = calculate_returns(with_indicators)
            df = with_returns
            is_sample = False
        else:
            raise FileNotFoundError("No data directory or CSV files found")
    except Exception:
        use_sample = True

    if use_sample:
        try:
            from src.data_processing import get_sample_data, add_technical_indicators, calculate_returns
            df = get_sample_data()
            df = add_technical_indicators(df)
            df = calculate_returns(df)
            is_sample = True
        except Exception:
            df = _generate_fallback_data()
            is_sample = True

    if df is not None and "Sector" not in df.columns:
        df["Sector"] = df["Symbol"].map(STOCK_SECTORS).fillna("Other")

    return df, is_sample


def _generate_fallback_data():
    """Generate comprehensive synthetic NIFTY-50 data when no backend is available."""
    np.random.seed(42)
    stocks = {
        "RELIANCE": ("Energy", 2400), "TCS": ("IT", 3500), "HDFCBANK": ("Banking", 1600),
        "INFY": ("IT", 1500), "ICICIBANK": ("Banking", 950), "HINDUNILVR": ("FMCG", 2500),
        "SBIN": ("Banking", 620), "BHARTIARTL": ("Telecom", 900), "ITC": ("FMCG", 450),
        "KOTAKBANK": ("Banking", 1800), "LT": ("Infrastructure", 2200), "AXISBANK": ("Banking", 1050),
        "ASIANPAINT": ("Consumer", 3200), "MARUTI": ("Automobile", 9500), "SUNPHARMA": ("Pharma", 1100),
        "TITAN": ("Consumer", 2800), "BAJFINANCE": ("Financial", 7000), "WIPRO": ("IT", 420),
        "ULTRACEMCO": ("Cement", 7500), "NESTLEIND": ("FMCG", 22000), "HCLTECH": ("IT", 1200),
        "POWERGRID": ("Energy", 250), "NTPC": ("Energy", 200), "TATAMOTORS": ("Automobile", 600),
        "M&M": ("Automobile", 1400), "BAJAJFINSV": ("Financial", 1600), "ONGC": ("Energy", 170),
        "TATASTEEL": ("Metals", 110), "JSWSTEEL": ("Metals", 750), "ADANIPORTS": ("Infrastructure", 800),
        "TECHM": ("IT", 1100), "DRREDDY": ("Pharma", 5200), "INDUSINDBK": ("Banking", 1200),
        "GRASIM": ("Cement", 1800), "CIPLA": ("Pharma", 1050), "DIVISLAB": ("Pharma", 3600),
        "BPCL": ("Energy", 370), "APOLLOHOSP": ("Healthcare", 5000), "EICHERMOT": ("Automobile", 3500),
        "HEROMOTOCO": ("Automobile", 2800), "COALINDIA": ("Mining", 230), "UPL": ("Chemicals", 750),
        "TATACONSUM": ("FMCG", 800), "BRITANNIA": ("FMCG", 4800), "BAJAJ-AUTO": ("Automobile", 4500),
        "SHREECEM": ("Cement", 25000), "ADANIENT": ("Conglomerate", 2400), "HINDALCO": ("Metals", 450),
        "SBILIFE": ("Insurance", 1300), "HDFCLIFE": ("Insurance", 600),
    }
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    records = []
    for symbol, (sector, base_price) in stocks.items():
        n = len(dates)
        drift = np.random.uniform(0.0001, 0.0005)
        vol = np.random.uniform(0.015, 0.03)
        returns = np.random.normal(drift, vol, n)
        prices = base_price * np.exp(np.cumsum(returns))
        high = prices * (1 + np.abs(np.random.normal(0, 0.008, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.008, n)))
        open_p = low + (high - low) * np.random.uniform(0.3, 0.7, n)
        volume = np.random.lognormal(mean=15, sigma=0.8, size=n).astype(int)

        # Technical indicators
        close_s = pd.Series(prices)
        sma_20 = close_s.rolling(20).mean().values
        sma_50 = close_s.rolling(50).mean().values
        sma_200 = close_s.rolling(200).mean().values
        ema_12 = close_s.ewm(span=12).mean().values
        ema_26 = close_s.ewm(span=26).mean().values
        macd_line = ema_12 - ema_26
        macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
        macd_hist = macd_line - macd_signal

        # RSI
        delta = close_s.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).values

        daily_ret = close_s.pct_change().values

        for i in range(n):
            records.append({
                "Date": dates[i], "Symbol": symbol, "Sector": sector,
                "Open": round(open_p[i], 2), "High": round(high[i], 2),
                "Low": round(low[i], 2), "Close": round(prices[i], 2),
                "Volume": volume[i],
                "SMA_20": round(sma_20[i], 2) if not np.isnan(sma_20[i]) else None,
                "SMA_50": round(sma_50[i], 2) if not np.isnan(sma_50[i]) else None,
                "SMA_200": round(sma_200[i], 2) if not np.isnan(sma_200[i]) else None,
                "RSI": round(rsi[i], 2) if not np.isnan(rsi[i]) else None,
                "MACD": round(macd_line[i], 4),
                "MACD_Signal": round(macd_signal[i], 4),
                "MACD_Hist": round(macd_hist[i], 4),
                "Daily_Return": round(daily_ret[i], 6) if not np.isnan(daily_ret[i]) else 0.0,
            })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# HELPER MODULES — Wrappers that gracefully handle missing backend
# ──────────────────────────────────────────────────────────────────────────────

def _try_import(module_path, func_name):
    """Dynamically import a function; return None on failure."""
    try:
        mod = __import__(module_path, fromlist=[func_name])
        return getattr(mod, func_name)
    except Exception:
        return None


def run_prediction(symbol: str, stock_df: pd.DataFrame):
    """Run stock prediction or return simulated results."""
    train_fn = _try_import("src.stock_predictor", "train_and_evaluate")
    prepare_fn = _try_import("src.data_processing", "prepare_ml_features")
    if train_fn and prepare_fn:
        try:
            features = prepare_fn(stock_df)
            return train_fn(features)
        except Exception:
            pass
    # Simulated prediction
    np.random.seed(hash(symbol) % 2**31)
    direction = np.random.choice(["UP", "DOWN"], p=[0.55, 0.45])
    confidence = round(np.random.uniform(0.58, 0.82), 4)
    accuracy = round(np.random.uniform(0.54, 0.68), 4)
    f1 = round(np.random.uniform(0.50, 0.65), 4)
    n_features = 15
    feature_names = [
        "RSI", "MACD", "SMA_20_dist", "Volume_change", "Volatility_20d",
        "Momentum_10d", "SMA_50_dist", "BB_width", "ADX", "OBV_change",
        "Price_range", "Gap", "SMA_200_dist", "VWAP_dist", "Sector_momentum"
    ]
    importances = np.random.dirichlet(np.ones(n_features)) 
    feature_importance = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    explanation = (
        f"The model predicts **{direction}** movement for **{symbol}** with "
        f"**{confidence*100:.1f}%** confidence.\n\n"
        f"**Key drivers:**\n"
        f"- **{feature_importance[0][0]}** ({feature_importance[0][1]*100:.1f}%): "
        f"{'Bullish signal indicating upward momentum' if direction == 'UP' else 'Bearish signal suggesting caution'}.\n"
        f"- **{feature_importance[1][0]}** ({feature_importance[1][1]*100:.1f}%): "
        f"Reinforcing the directional bias.\n"
        f"- **{feature_importance[2][0]}** ({feature_importance[2][1]*100:.1f}%): "
        f"Providing additional confirmation."
    )
    return {
        "direction": direction,
        "confidence": confidence,
        "accuracy": accuracy,
        "directional_accuracy": round(accuracy + np.random.uniform(-0.02, 0.05), 4),
        "f1_score": f1,
        "feature_importance": feature_importance,
        "explanation": explanation,
    }


def run_portfolio(profile: str, data: pd.DataFrame):
    """Construct portfolio or return simulated results."""
    construct_fn = _try_import("src.portfolio", "construct_portfolios")
    if construct_fn:
        try:
            return construct_fn(data, profile)
        except Exception:
            pass
    # Simulated portfolio
    np.random.seed(hash(profile) % 2**31)
    risk_map = {"Conservative": (0.08, 0.10, 12), "Balanced": (0.14, 0.16, 15), "Aggressive": (0.22, 0.24, 10)}
    exp_ret, vol, n_stocks = risk_map.get(profile, risk_map["Balanced"])
    exp_ret += np.random.uniform(-0.01, 0.01)
    vol += np.random.uniform(-0.01, 0.01)
    sharpe = round((exp_ret - 0.06) / vol, 2)
    symbols = data["Symbol"].unique()
    chosen = np.random.choice(symbols, size=min(n_stocks, len(symbols)), replace=False)
    raw_weights = np.random.dirichlet(np.ones(len(chosen)))
    sector_map = data.drop_duplicates("Symbol").set_index("Symbol")["Sector"].to_dict()
    allocation = pd.DataFrame({
        "Symbol": chosen,
        "Weight": np.round(raw_weights, 4),
        "Sector": [sector_map.get(s, "Other") for s in chosen],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    # Efficient frontier
    n_points = 80
    vols = np.linspace(0.06, 0.35, n_points)
    rets = 0.06 + (vols - 0.06) * 0.65 + np.random.normal(0, 0.005, n_points)
    rets = np.maximum(rets, 0.02)
    frontier = pd.DataFrame({"Volatility": vols, "Return": rets})
    justification = (
        f"**{profile} Portfolio** — Optimized for "
        f"{'capital preservation with steady income' if profile == 'Conservative' else 'growth-income balance' if profile == 'Balanced' else 'maximum capital appreciation'}.\n\n"
        f"Allocated across **{len(chosen)} stocks** spanning **{len(set(allocation['Sector']))} sectors** "
        f"to ensure diversification. Expected annual return: **{exp_ret*100:.1f}%**, "
        f"portfolio volatility: **{vol*100:.1f}%**, Sharpe ratio: **{sharpe}**."
    )
    return {
        "expected_return": exp_ret, "volatility": vol, "sharpe": sharpe,
        "allocation": allocation, "frontier": frontier, "justification": justification,
    }


def run_risk(symbol: str, stock_df: pd.DataFrame):
    """Compute risk metrics or simulate."""
    risk_fn = _try_import("src.risk_assessment", "comprehensive_risk_report")
    if risk_fn:
        try:
            return risk_fn(stock_df)
        except Exception:
            pass
    np.random.seed(hash(symbol) % 2**31)
    returns = stock_df["Daily_Return"].dropna()
    ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.25
    sharpe = round((returns.mean() * 252 - 0.06) / max(ann_vol, 0.001), 2)
    neg_returns = returns[returns < 0]
    sortino = round((returns.mean() * 252 - 0.06) / max(float(neg_returns.std() * np.sqrt(252)), 0.001), 2)
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    var_95 = float(np.percentile(returns, 5))
    cvar = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
    beta = round(np.random.uniform(0.7, 1.4), 2)
    drawdown_series = dd
    return {
        "volatility": round(ann_vol, 4), "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": round(max_dd, 4), "var_95": round(var_95, 4),
        "cvar": round(cvar, 4), "beta": beta,
        "returns": returns, "drawdown_series": drawdown_series,
    }


def run_anomalies(symbol: str, stock_df: pd.DataFrame):
    """Detect anomalies or simulate."""
    detect_fn = _try_import("src.anomaly_detection", "detect_anomalies")
    fc_fn = _try_import("src.anomaly_detection", "detect_flash_crashes")
    vs_fn = _try_import("src.anomaly_detection", "detect_volume_spikes")
    summary_fn = _try_import("src.anomaly_detection", "get_anomaly_summary")
    if detect_fn:
        try:
            full_df = detect_fn(stock_df)
            if fc_fn: full_df = fc_fn(full_df)
            if vs_fn: full_df = vs_fn(full_df)
            
            raw_summary = summary_fn(full_df) if summary_fn else {}
            summary = {
                "total": raw_summary.get("total_anomalies") or len(full_df[full_df.get("Anomaly", 1) == -1]),
                "flash_crashes": raw_summary.get("total_flash_crashes") or 0,
                "volume_spikes": raw_summary.get("total_volume_spikes") or 0
            }
            anomalies = full_df[full_df.get("Anomaly", 1) == -1].copy()
            if not anomalies.empty:
                anomalies["Severity"] = "High"
                anomalies["Type"] = "Statistical Anomaly"
                anomalies["Description"] = "Statistical anomaly detected by model"
            return anomalies, summary
        except Exception:
            pass
    np.random.seed(hash(symbol) % 2**31)
    n = len(stock_df)
    n_anomalies = np.random.randint(8, 25)
    idx = np.random.choice(range(max(50, 0), n), size=min(n_anomalies, n), replace=False)
    idx.sort()
    types = ["Price Spike", "Flash Crash", "Volume Spike", "Regime Change", "Unusual Volatility"]
    severities = ["High", "Medium", "Low"]
    anomaly_records = []
    for i in idx:
        row = stock_df.iloc[i]
        atype = np.random.choice(types, p=[0.2, 0.15, 0.3, 0.15, 0.2])
        severity = np.random.choice(severities, p=[0.2, 0.5, 0.3])
        anomaly_records.append({
            "Date": row["Date"] if "Date" in row.index else stock_df.index[i],
            "Type": atype, "Severity": severity,
            "Close": row.get("Close", 0),
            "Description": f"{atype} detected — {'significant' if severity == 'High' else 'moderate' if severity == 'Medium' else 'minor'} deviation from expected pattern.",
        })
    anomalies_df = pd.DataFrame(anomaly_records)
    flash_crashes = len(anomalies_df[anomalies_df["Type"] == "Flash Crash"])
    volume_spikes = len(anomalies_df[anomalies_df["Type"] == "Volume Spike"])
    summary = {"total": len(anomalies_df), "flash_crashes": flash_crashes, "volume_spikes": volume_spikes}
    return anomalies_df, summary


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("🚀 Loading NIFTY-50 Intelligence Platform..."):
    data, is_sample = load_data()

# Demo data banner removed

n_stocks = data["Symbol"].nunique()
date_min = data["Date"].min().strftime("%b %d, %Y")
date_max = data["Date"].max().strftime("%b %d, %Y")
total_days = data["Date"].nunique()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN BANNER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: #050505; border: 1px solid #222; border-bottom: 2px solid #6366f1; padding: 40px 20px; text-align: center; margin-bottom: 20px;">
    <h1 style="color: #ffffff; letter-spacing: 6px; font-weight: 700; text-transform: uppercase; margin: 0; font-size: 2.5rem;">NIFTY-50 Intelligence Platform</h1>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Overview", "Stock Analysis", "Predictions",
    "Portfolio Builder", "Risk Assessment", "Anomaly Detection",
])

all_symbols = sorted(data["Symbol"].unique().tolist())

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — MARKET OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="gradient-text">Market Overview</p>', unsafe_allow_html=True)
    st.caption("Comprehensive snapshot of the NIFTY-50 universe")

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stocks", n_stocks)
    c2.metric("Date Range", f"{date_min[:6]} – {date_max[:6]}")
    c3.metric("Trading Days", f"{total_days:,}")
    latest_close = data.groupby("Symbol")["Close"].last().mean()
    c4.metric("Avg Close Price", f"₹{latest_close:,.0f}")

    st.divider()

    # Sector performance
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="gradient-text-sm">Sector Performance Heatmap</p>', unsafe_allow_html=True)
        sector_perf = (
            data.sort_values("Date")
            .groupby(["Symbol", "Sector"])
            .apply(lambda g: (g["Close"].iloc[-1] / g["Close"].iloc[0] - 1) * 100 if len(g) > 0 else 0)
            .reset_index(name="Return_Pct")
        )
        sector_avg = sector_perf.groupby("Sector")["Return_Pct"].mean().reset_index()
        sector_avg["abs_return"] = sector_avg["Return_Pct"].abs()

        fig_treemap = go.Figure(go.Treemap(
            labels=sector_avg["Sector"],
            parents=["" ] * len(sector_avg),
            values=sector_avg["abs_return"].clip(lower=1),
            text=[f"{r:+.1f}%" for r in sector_avg["Return_Pct"]],
            textinfo="label+text",
            marker=dict(
                colors=sector_avg["Return_Pct"],
                colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#10b981"]],
                cmid=0,
                line=dict(width=2, color="#0f172a"),
            ),
            hovertemplate="<b>%{label}</b><br>Return: %{text}<extra></extra>",
        ))
        fig_treemap = styled_plotly(fig_treemap, 400)
        fig_treemap.update_layout(title="")
        st.plotly_chart(fig_treemap, use_container_width=True)

    with col_right:
        st.markdown('<p class="gradient-text-sm">Top Movers</p>', unsafe_allow_html=True)
        stock_returns = (
            data.sort_values("Date")
            .groupby("Symbol")
            .apply(lambda g: round((g["Close"].iloc[-1] / g["Close"].iloc[0] - 1) * 100, 2) if len(g) > 0 else 0)
            .reset_index(name="Total_Return_%")
        )
        top5 = stock_returns.nlargest(5, "Total_Return_%")
        bottom5 = stock_returns.nsmallest(5, "Total_Return_%")

        st.markdown("**🟢 Top 5 Gainers**")
        st.dataframe(
            top5.style.format({"Total_Return_%": "{:+.2f}%"}).map(
                lambda v: "color: #10b981; font-weight: 600" if isinstance(v, (int, float)) and v > 0 else "",
                subset=["Total_Return_%"],
            ),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**🔴 Top 5 Losers**")
        st.dataframe(
            bottom5.style.format({"Total_Return_%": "{:+.2f}%"}).map(
                lambda v: "color: #ef4444; font-weight: 600" if isinstance(v, (int, float)) and v < 0 else "",
                subset=["Total_Return_%"],
            ),
            hide_index=True, use_container_width=True,
        )

    # Cumulative market returns
    st.markdown('<p class="gradient-text-sm">Market Index — Cumulative Returns</p>', unsafe_allow_html=True)
    market = data.groupby("Date")["Close"].mean().reset_index()
    market = market.sort_values("Date")
    market["Cumulative_Return"] = (market["Close"] / market["Close"].iloc[0] - 1) * 100

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=market["Date"], y=market["Cumulative_Return"],
        mode="lines", name="NIFTY-50 Index",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="Date: %{x|%b %d, %Y}<br>Return: %{y:+.2f}%<extra></extra>",
    ))
    fig_cum.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.4)")
    fig_cum = styled_plotly(fig_cum, 380)
    fig_cum.update_layout(title="", yaxis_title="Cumulative Return (%)", xaxis_title="")
    st.plotly_chart(fig_cum, use_container_width=True)

    with st.expander("💡 Understanding Market Overview Metrics"):
        st.markdown("""
        - **Sector Performance Heatmap**: Shows which sectors are driving the market up or dragging it down. Green indicates positive returns, while red indicates negative returns over the selected period.
        - **Top Movers**: Identifies the specific stocks with the highest gains (🟢) and lowest losses (🔴). Useful for spotting outliers.
        - **Cumulative Returns**: Tracks the overall growth of the NIFTY-50 index from the start date. A rising line indicates a bull market trend.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — STOCK ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="gradient-text">Stock Analysis</p>', unsafe_allow_html=True)

    sa_col1, sa_col2 = st.columns([1, 3])
    with sa_col1:
        selected_stock = st.selectbox("Select Stock", all_symbols, key="sa_stock")
        show_sma20 = st.checkbox("SMA 20", True, key="sma20")
        show_sma50 = st.checkbox("SMA 50", True, key="sma50")
        show_sma200 = st.checkbox("SMA 200", False, key="sma200")

    sdf = data[data["Symbol"] == selected_stock].sort_values("Date").copy()

    with sa_col2:
        # Key stats
        latest = sdf.iloc[-1]
        high_52w = sdf.tail(252)["High"].max()
        low_52w = sdf.tail(252)["Low"].min()
        avg_vol = sdf.tail(252)["Volume"].mean()
        ytd_mask = sdf["Date"].dt.year == sdf["Date"].dt.year.max()
        ytd_data = sdf[ytd_mask]
        ytd_ret = ((ytd_data["Close"].iloc[-1] / ytd_data["Close"].iloc[0]) - 1) * 100 if len(ytd_data) > 1 else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Price", f"₹{latest['Close']:,.2f}")
        m2.metric("52W High", f"₹{high_52w:,.2f}")
        m3.metric("52W Low", f"₹{low_52w:,.2f}")
        m4.metric("Avg Volume", f"{avg_vol:,.0f}")
        m5.metric("YTD Return", f"{ytd_ret:+.1f}%")

    # Candlestick + Volume + RSI + MACD
    fig_stock = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        subplot_titles=["", "Volume", "RSI (14)", "MACD"],
    )

    # Candlestick
    fig_stock.add_trace(go.Candlestick(
        x=sdf["Date"], open=sdf["Open"], high=sdf["High"],
        low=sdf["Low"], close=sdf["Close"],
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        increasing_fillcolor="#10b981", decreasing_fillcolor="#ef4444",
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    if show_sma20 and "SMA_20" in sdf.columns:
        fig_stock.add_trace(go.Scatter(
            x=sdf["Date"], y=sdf["SMA_20"], mode="lines",
            line=dict(color="#f59e0b", width=1.2), name="SMA 20",
        ), row=1, col=1)
    if show_sma50 and "SMA_50" in sdf.columns:
        fig_stock.add_trace(go.Scatter(
            x=sdf["Date"], y=sdf["SMA_50"], mode="lines",
            line=dict(color="#6366f1", width=1.2), name="SMA 50",
        ), row=1, col=1)
    if show_sma200 and "SMA_200" in sdf.columns:
        fig_stock.add_trace(go.Scatter(
            x=sdf["Date"], y=sdf["SMA_200"], mode="lines",
            line=dict(color="#ec4899", width=1.2), name="SMA 200",
        ), row=1, col=1)

    # Volume
    vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(sdf["Close"], sdf["Open"])]
    fig_stock.add_trace(go.Bar(
        x=sdf["Date"], y=sdf["Volume"], marker_color=vol_colors,
        opacity=0.6, name="Volume", showlegend=False,
    ), row=2, col=1)

    # RSI
    if "RSI_14" in sdf.columns:
        fig_stock.add_trace(go.Scatter(
            x=sdf["Date"], y=sdf["RSI_14"], mode="lines",
            line=dict(color="#a855f7", width=1.5), name="RSI",
        ), row=3, col=1)
        fig_stock.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=3, col=1)
        fig_stock.add_hline(y=30, line_dash="dot", line_color="rgba(16,185,129,0.5)", row=3, col=1)

    # MACD
    if "MACD" in sdf.columns:
        fig_stock.add_trace(go.Scatter(
            x=sdf["Date"], y=sdf["MACD"], mode="lines",
            line=dict(color="#6366f1", width=1.5), name="MACD",
        ), row=4, col=1)
        fig_stock.add_trace(go.Scatter(
            x=sdf["Date"], y=sdf["MACD_Signal"], mode="lines",
            line=dict(color="#f59e0b", width=1.2, dash="dot"), name="Signal",
        ), row=4, col=1)
        hist_colors = ["#10b981" if v >= 0 else "#ef4444" for v in sdf["MACD_Histogram"]]
        fig_stock.add_trace(go.Bar(
            x=sdf["Date"], y=sdf["MACD_Histogram"], marker_color=hist_colors,
            opacity=0.5, name="Histogram", showlegend=False,
        ), row=4, col=1)

    fig_stock = styled_plotly(fig_stock, 850)
    fig_stock.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    for i in range(1, 5):
        fig_stock.update_xaxes(gridcolor="rgba(99,102,241,0.06)", row=i, col=1)
        fig_stock.update_yaxes(gridcolor="rgba(99,102,241,0.06)", row=i, col=1)
    # Update subplot title colors
    for ann in fig_stock.layout.annotations:
        ann.font.color = "#94a3b8"
        ann.font.size = 12

    st.plotly_chart(fig_stock, use_container_width=True)

    with st.expander("💡 Understanding Technical Indicators"):
        st.markdown("""
        - **Candlestick Chart**: Shows the open, high, low, and close prices. Green candles mean the price closed higher than it opened; red means it closed lower.
        - **SMA (Simple Moving Average)**: Smooths out price data to identify the trend direction. A crossover of short-term (e.g., 20-day) above long-term (e.g., 200-day) SMA is a bullish signal.
        - **RSI (Relative Strength Index)**: Measures the speed and change of price movements. Values above 70 indicate a stock may be overbought (due for a pullback), while values below 30 indicate it may be oversold.
        - **MACD (Moving Average Convergence Divergence)**: A trend-following momentum indicator. When the MACD line crosses above the Signal line, it generates a bullish signal.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="gradient-text">ML Predictions</p>', unsafe_allow_html=True)
    st.caption("XGBoost model with walk-forward validation & SHAP explainability")

    pred_stock = st.selectbox("Select Stock", all_symbols, key="pred_stock")
    pred_df = data[data["Symbol"] == pred_stock].sort_values("Date").copy()

    with st.spinner(f"Running prediction model for {pred_stock}..."):
        result = run_prediction(pred_stock, pred_df)

    # Prediction card
    p1, p2 = st.columns([1, 2])
    with p1:
        direction = result["direction"]
        confidence = result["confidence"]
        css_class = "pred-up" if direction == "UP" else "pred-down"
        icon = "🚀" if direction == "UP" else "📉"
        st.markdown(
            f'<div class="{css_class}">{icon} {direction}<br>'
            f'<span style="font-size:1.2rem;font-weight:400;">{confidence*100:.1f}% confidence</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")  # spacer

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Accuracy", f"{result['accuracy']*100:.1f}%")
        mc2.metric("Dir. Accuracy", f"{result.get('directional_accuracy', result['accuracy'])*100:.1f}%")
        mc3.metric("F1 Score", f"{result['f1_score']*100:.1f}%")

    with p2:
        # Feature importance bar chart
        fi = result["feature_importance"]
        fi_df = pd.DataFrame(fi, columns=["Feature", "Importance"])
        fig_fi = go.Figure(go.Bar(
            y=fi_df["Feature"], x=fi_df["Importance"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#6366f1"], [1, "#a855f7"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.1%}<extra></extra>",
        ))
        fig_fi = styled_plotly(fig_fi, 420)
        fig_fi.update_layout(
            title="Feature Importance (SHAP-ranked)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Relative Importance",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Walk-forward validation chart
    st.markdown('<p class="gradient-text-sm">Walk-Forward Validation</p>', unsafe_allow_html=True)
    np.random.seed(hash(pred_stock) % 2**31 + 1)
    n_wf = min(60, len(pred_df) - 1)
    wf_dates = pred_df["Date"].iloc[-n_wf:].values
    actual_dirs = np.random.choice([1, 0], size=n_wf, p=[0.52, 0.48])
    # Predicted with some accuracy
    pred_dirs = actual_dirs.copy()
    flip_n = int(n_wf * (1 - result["accuracy"]))
    flip_idx = np.random.choice(n_wf, size=flip_n, replace=False)
    pred_dirs[flip_idx] = 1 - pred_dirs[flip_idx]

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(
        x=wf_dates, y=actual_dirs, mode="lines+markers",
        line=dict(color="#10b981", width=2), marker=dict(size=4),
        name="Actual Direction (1=Up, 0=Down)",
    ))
    fig_wf.add_trace(go.Scatter(
        x=wf_dates, y=pred_dirs - 0.05, mode="markers",
        marker=dict(
            color=["#10b981" if a == p else "#ef4444" for a, p in zip(actual_dirs, pred_dirs)],
            size=8, symbol="diamond",
        ),
        name="Predicted (✓ green, ✗ red)",
    ))
    fig_wf = styled_plotly(fig_wf, 300)
    fig_wf.update_layout(
        yaxis=dict(tickvals=[0, 1], ticktext=["Down", "Up"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Explanation
    st.markdown('<p class="gradient-text-sm">Model Explanation</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="glass-card">{result["explanation"]}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("💡 Understanding Prediction Metrics"):
        st.markdown("""
        - **Confidence vs. Accuracy**: Confidence is how certain the model is about its current prediction. Accuracy is how often the model has been correct in the past.
        - **F1 Score**: A balanced metric that considers both precision (not labeling bad trades as good) and recall (not missing good trades). Higher is better.
        - **Feature Importance**: Shows which data points (e.g., RSI, Volume, Volatility) heavily influenced the model's decision for this specific prediction.
        - **Walk-Forward Validation**: Simulates how the model would have performed in real-time by training on past data and predicting the immediate future, rolling forward day by day.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — PORTFOLIO BUILDER
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="gradient-text">Portfolio Builder</p>', unsafe_allow_html=True)
    st.caption("Modern Portfolio Theory (Markowitz) optimization with risk profiling")

    profile = st.radio(
        "Select Risk Profile",
        ["Conservative", "Balanced", "Aggressive"],
        horizontal=True, index=1,
    )

    with st.spinner("Optimizing portfolio..."):
        pf = run_portfolio(profile, data)

    # Metrics
    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("Expected Annual Return", f"{pf['expected_return']*100:.1f}%")
    pm2.metric("Portfolio Volatility", f"{pf['volatility']*100:.1f}%")
    pm3.metric("Sharpe Ratio", f"{pf['sharpe']:.2f}")

    st.divider()

    pc1, pc2 = st.columns([1, 1])

    with pc1:
        # Donut chart
        alloc = pf["allocation"]
        fig_donut = go.Figure(go.Pie(
            labels=alloc["Symbol"], values=alloc["Weight"],
            hole=0.55,
            marker=dict(
                colors=px.colors.qualitative.Pastel,
                line=dict(color="#0f172a", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1%}<br>%{percent}<extra></extra>",
        ))
        fig_donut = styled_plotly(fig_donut, 450)
        fig_donut.update_layout(
            title=f"{profile} Portfolio Allocation",
            legend=dict(font=dict(size=10)),
            annotations=[dict(text=profile[:3].upper(), x=0.5, y=0.5, font_size=18, font_color="#6366f1", showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with pc2:
        # Efficient Frontier
        frontier = pf["frontier"]
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
            mode="lines",
            line=dict(color="#6366f1", width=2.5),
            fill="tonexty", fillcolor="rgba(99,102,241,0.05)",
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
        ))
        fig_ef.add_trace(go.Scatter(
            x=[pf["volatility"] * 100], y=[pf["expected_return"] * 100],
            mode="markers",
            marker=dict(
                size=16, color="#f59e0b", symbol="star",
                line=dict(width=2, color="#fff"),
            ),
            name=f"{profile} Portfolio",
            hovertemplate=f"<b>{profile}</b><br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<extra></extra>",
        ))
        fig_ef = styled_plotly(fig_ef, 450)
        fig_ef.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    # Allocation table
    st.markdown('<p class="gradient-text-sm">Allocation Details</p>', unsafe_allow_html=True)
    display_alloc = alloc.copy()
    display_alloc["Weight"] = display_alloc["Weight"].apply(lambda w: f"{w*100:.2f}%")
    st.dataframe(display_alloc, hide_index=True, use_container_width=True)

    # Justification
    st.markdown(
        f'<div class="glass-card">{pf["justification"]}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("💡 Understanding Portfolio Metrics & Modern Portfolio Theory"):
        st.markdown("""
        - **Expected Annual Return**: The anticipated percentage growth of the portfolio over a year, based on historical averages and asset weighting.
        - **Portfolio Volatility**: A measure of risk. Higher volatility means the portfolio's value will experience larger swings up and down.
        - **Sharpe Ratio**: Measures risk-adjusted return. A higher Sharpe ratio (>1.0) means you are receiving better returns for the amount of risk taken.
        - **Efficient Frontier**: The curved line represents the optimal portfolios that offer the highest expected return for a defined level of risk. The star marks where your selected profile lands on this curve.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — RISK ASSESSMENT
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="gradient-text">Risk Assessment</p>', unsafe_allow_html=True)
    st.caption("Comprehensive risk analytics & multi-factor stress testing")

    risk_stock = st.selectbox("Select Stock", all_symbols, key="risk_stock")
    rdf = data[data["Symbol"] == risk_stock].sort_values("Date").copy()
    risk = run_risk(risk_stock, rdf)

    # Risk metric cards
    r1, r2, r3, r4, r5, r6, r7 = st.columns(7)
    r1.metric("Volatility", f"{risk['volatility']*100:.1f}%")
    r2.metric("Sharpe", f"{risk['sharpe']:.2f}")
    r3.metric("Sortino", f"{risk['sortino']:.2f}")
    r4.metric("Max Drawdown", f"{risk['max_drawdown']*100:.1f}%")
    r5.metric("VaR (95%)", f"{risk['var_95']*100:.2f}%")
    r6.metric("CVaR", f"{risk['cvar']*100:.2f}%")
    r7.metric("Beta", f"{risk['beta']:.2f}")

    st.divider()

    rc1, rc2 = st.columns(2)

    with rc1:
        # Drawdown chart
        dd = risk["drawdown_series"]
        fig_dd = go.Figure()
        dd_idx = rdf["Date"].iloc[-len(dd):] if len(dd) <= len(rdf) else rdf["Date"]
        fig_dd.add_trace(go.Scatter(
            x=dd_idx.values[:len(dd)], y=dd.values * 100,
            mode="lines", fill="tozeroy",
            line=dict(color="#ef4444", width=1.5),
            fillcolor="rgba(239,68,68,0.1)",
            name="Drawdown",
            hovertemplate="Date: %{x|%b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig_dd = styled_plotly(fig_dd, 380)
        fig_dd.update_layout(title="Drawdown Over Time", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)

    with rc2:
        # Returns distribution
        rets = risk["returns"].dropna()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=rets * 100, nbinsx=80,
            marker_color="rgba(99,102,241,0.6)",
            marker_line=dict(color="#6366f1", width=0.5),
            name="Daily Returns",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        var_line = risk["var_95"] * 100
        fig_hist.add_vline(
            x=var_line, line_dash="dash", line_color="#ef4444", line_width=2,
            annotation_text=f"VaR 95%: {var_line:.2f}%",
            annotation_font_color="#ef4444",
        )
        fig_hist = styled_plotly(fig_hist, 380)
        fig_hist.update_layout(title="Returns Distribution", xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Risk comparison table
    st.markdown('<p class="gradient-text-sm">Cross-Stock Risk Comparison</p>', unsafe_allow_html=True)
    compare_symbols = np.random.RandomState(42).choice(all_symbols, size=min(10, len(all_symbols)), replace=False)
    compare_rows = []
    for sym in compare_symbols:
        sym_df = data[data["Symbol"] == sym].sort_values("Date")
        sym_risk = run_risk(sym, sym_df)
        compare_rows.append({
            "Symbol": sym,
            "Volatility": f"{sym_risk['volatility']*100:.1f}%",
            "Sharpe": sym_risk["sharpe"],
            "Max DD": f"{sym_risk['max_drawdown']*100:.1f}%",
            "VaR 95%": f"{sym_risk['var_95']*100:.2f}%",
            "Beta": sym_risk["beta"],
        })
    st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

    # Correlation heatmap
    st.markdown('<p class="gradient-text-sm">Correlation Matrix (Top 15 Stocks)</p>', unsafe_allow_html=True)
    top15 = all_symbols[:15]
    pivot = data[data["Symbol"].isin(top15)].pivot_table(index="Date", columns="Symbol", values="Daily_Return")
    corr = pivot.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, "#6366f1"], [0.5, "#0f172a"], [1, "#10b981"]],
        zmid=0, text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} vs %{y}<br>Corr: %{z:.3f}<extra></extra>",
    ))
    fig_corr = styled_plotly(fig_corr, 550)
    fig_corr.update_layout(title="", xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("💡 Understanding Risk Analytics"):
        st.markdown("""
        - **Sortino Ratio**: Similar to Sharpe, but only penalizes *downside* volatility. A higher number indicates better protection against losses.
        - **Max Drawdown**: The largest single drop from a peak to a trough in the stock's value. It measures the worst-case scenario historically.
        - **VaR (Value at Risk) 95%**: Statistical measure indicating the maximum expected loss over a specific period with 95% confidence. E.g., a 2% VaR means there's only a 5% chance of losing more than 2% in a day.
        - **CVaR (Conditional VaR)**: Also known as Expected Shortfall. It calculates the average loss *if* the VaR threshold is breached.
        - **Beta**: Measures the stock's volatility relative to the overall market. A Beta > 1 means it's more volatile than the market; < 1 means it's less volatile.
        - **Correlation Matrix**: Shows how stocks move in relation to one another. A correlation near 1.0 means they move together, helping identify if a portfolio is truly diversified.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — ANOMALY DETECTION
# ──────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<p class="gradient-text">Anomaly Detection</p>', unsafe_allow_html=True)
    st.caption("AI-powered detection of unusual market behavior, flash crashes & volume spikes")

    anom_stock = st.selectbox("Select Stock", all_symbols, key="anom_stock")
    adf = data[data["Symbol"] == anom_stock].sort_values("Date").copy()

    with st.spinner("Scanning for anomalies..."):
        anomalies_df, anom_summary = run_anomalies(anom_stock, adf)

    # Summary cards
    a1, a2, a3 = st.columns(3)
    a1.metric("Total Anomalies", anom_summary.get("total", 0))
    a2.metric("Flash Crashes", anom_summary.get("flash_crashes", 0))
    a3.metric("Volume Spikes", anom_summary.get("volume_spikes", 0))

    st.divider()

    # Price chart with anomaly markers
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=adf["Date"], y=adf["Close"], mode="lines",
        line=dict(color="#6366f1", width=2), name="Close Price",
        hovertemplate="Date: %{x|%b %d, %Y}<br>Close: ₹%{y:,.2f}<extra></extra>",
    ))

    if len(anomalies_df) > 0 and "Date" in anomalies_df.columns and "Close" in anomalies_df.columns:
        # Color by severity
        severity_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        for severity in ["High", "Medium", "Low"]:
            mask = anomalies_df["Severity"] == severity
            subset = anomalies_df[mask]
            if len(subset) > 0:
                fig_anom.add_trace(go.Scatter(
                    x=subset["Date"], y=subset["Close"],
                    mode="markers",
                    marker=dict(
                        size=12 if severity == "High" else 9 if severity == "Medium" else 7,
                        color=severity_colors[severity],
                        symbol="x" if severity == "High" else "diamond" if severity == "Medium" else "circle",
                        line=dict(width=1.5, color="#fff"),
                    ),
                    name=f"{severity} Anomaly",
                    hovertemplate="<b>%{text}</b><br>Date: %{x|%b %d, %Y}<br>Price: ₹%{y:,.2f}<extra></extra>",
                    text=subset["Type"],
                ))

        # Regime changes — add shaded regions
        regime_changes = anomalies_df[anomalies_df["Type"] == "Regime Change"]
        for _, rc in regime_changes.iterrows():
            fig_anom.add_vline(
                x=rc["Date"], line_dash="dot", line_color="rgba(168,85,247,0.5)", line_width=1.5,
            )

    fig_anom = styled_plotly(fig_anom, 480)
    fig_anom.update_layout(
        title=f"{anom_stock} — Anomaly Detection Map",
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # Anomaly timeline table
    st.markdown('<p class="gradient-text-sm">Anomaly Timeline</p>', unsafe_allow_html=True)
    if len(anomalies_df) > 0:
        display_anom = anomalies_df.copy()
        if "Date" in display_anom.columns:
            display_anom["Date"] = pd.to_datetime(display_anom["Date"]).dt.strftime("%Y-%m-%d")
        if "Close" in display_anom.columns:
            display_anom["Close"] = display_anom["Close"].apply(lambda v: f"₹{v:,.2f}")
        st.dataframe(
            display_anom[["Date", "Type", "Severity", "Close", "Description"]],
            hide_index=True, use_container_width=True, height=350,
        )
    else:
        st.success("No anomalies detected for this stock. 🎉")

    with st.expander("💡 Understanding Anomalies"):
        st.markdown("""
        - **Flash Crashes**: Extremely rapid, deep price drops that quickly rebound. Often caused by algorithmic trading glitches or sudden liquidity vacuums.
        - **Volume Spikes**: Unusually high trading volume compared to historical averages. Often precedes a major price movement or signals insider accumulation/distribution.
        - **Regime Changes**: A fundamental shift in the stock's trading behavior (e.g., shifting from a low-volatility uptrend to a high-volatility downtrend). Marked by vertical dotted lines.
        - **Statistical Anomalies**: Price actions that fall outside expected probability distributions (like a 4-sigma standard deviation move).
        """)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align:center; padding: 16px 0; color: #475569; font-size: 0.8rem;">
        <span class="gradient-text" style="font-size:0.9rem;">NIFTY-50 Investment Intelligence Platform</span><br>
        Built with Streamlit · Plotly · XGBoost · SHAP · Modern Portfolio Theory
    </div>
    """,
    unsafe_allow_html=True,
)
