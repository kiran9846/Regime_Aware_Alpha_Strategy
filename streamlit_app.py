
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Regime-Aware Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }

    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    hr {
        border-color: #334155;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.6rem !important;
    }

    [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_pct(x):
    try:
        if x is None or np.isnan(float(x)):
            return "N/A"
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"

def format_num(x):
    try:
        if x is None or np.isnan(float(x)):
            return "N/A"
        return f"{float(x):.3f}"
    except Exception:
        return "N/A"

def signal_label(pred):
    return "▲ UP" if int(pred) == 1 else "▼ DOWN"

def regime_color(regime):
    colors = {
        "Bull": "#16a34a",
        "Bear": "#dc2626",
        "High_Vol": "#f97316",
        "High_vol": "#f97316"
    }
    return colors.get(regime, "#6b7280")

def drawdown_series(cum_returns):
    rolling_max = cum_returns.cummax()
    return (cum_returns - rolling_max) / rolling_max

def sharpe_ratio(returns, trading_days=252):
    returns = returns.dropna()
    if returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(trading_days)

def safe_get(summary_dict, key, default=np.nan):
    return summary_dict.get(key, default)

def get_col(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        return df
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        st.stop()

@st.cache_data
def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        st.stop()

original_df = load_csv("final_backtest_original.csv")
best_df = load_csv("final_backtest_best.csv")
buyhold_df = load_csv("final_backtest_buyhold.csv")
summary = load_json("performance_summary.json")

# ============================================================
# COLUMN SAFETY
# ============================================================

orig_cum_col = get_col(original_df, ["Original_XGB_GARCH_cum_net", "cumulative_strategy_net"])
orig_ret_col = get_col(original_df, ["Original_XGB_GARCH_net_return", "strategy_return_net"])

best_cum_col = get_col(best_df, ["Best_XGB_GARCH_cum_net", "cumulative_strategy_net"])
best_ret_col = get_col(best_df, ["Best_XGB_GARCH_net_return", "best_strategy_return", "strategy_return_net"])

buyhold_cum_col = get_col(buyhold_df, ["Buy_Hold_cum_gross", "cumulative_buy_hold"])
buyhold_ret_col = get_col(buyhold_df, ["Buy_Hold_gross_return", "buy_hold_return"])

missing = []
for name, col in {
    "Original cumulative return": orig_cum_col,
    "Original return": orig_ret_col,
    "Best cumulative return": best_cum_col,
    "Best return": best_ret_col,
    "Buy-hold cumulative return": buyhold_cum_col,
    "Buy-hold return": buyhold_ret_col
}.items():
    if col is None:
        missing.append(name)

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ============================================================
# LATEST VALUES
# ============================================================

latest = best_df.iloc[-1]
latest_date = best_df.index[-1]

current_regime = latest.get("regime", "Unknown")

latest_pred = int(latest.get("predicted_direction", 0))
latest_prob_up = float(latest.get("prob_up", 0.5))

# Confidence should match predicted direction
latest_confidence = latest_prob_up if latest_pred == 1 else (1 - latest_prob_up)

latest_vol = float(latest.get("garch_vol_forecast", 0.0))

# IMPORTANT:
# position_size = GARCH-only risk-adjusted allocation
# best_position = active final position after XGBoost threshold signal
latest_position_size = float(latest.get("position_size", 0.0))
latest_active_position = float(latest.get("best_position", 0.0))

signal_color = "#16a34a" if latest_pred == 1 else "#dc2626"
signal_text = signal_label(latest_pred)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧭 Navigation")

    page = st.radio(
        "Go to",
        [
            "Overview",
            "Performance",
            "Regime Analysis",
            "GARCH Volatility",
            "Signal Explorer",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📅 Data Info")
    st.markdown(f"**Period:** {best_df.index[0].date()} → {latest_date.date()}")
    st.markdown(f"**Trading Days:** {len(best_df)}")
    st.markdown("**Strategy:** Best XGB + GARCH")
    st.markdown("**Threshold:** prob_up > 0.52")

    st.markdown("---")
    st.markdown("### ⚙️ Strategy Parameters")
    st.markdown("- **HMM States:** Bull / Bear / High_Vol")
    st.markdown("- **XGBoost:** Walk-forward CV")
    st.markdown("- **GARCH:** GARCH(1,1), Student-t")
    st.markdown("- **Transaction Cost:** 0.1% per trade")

    st.markdown("---")
    st.caption("Capstone Project")
    st.caption("Managerial Economics & Data Science")

# ============================================================
# TITLE
# ============================================================

st.title("📈 Regime-Aware Stock Return Prediction")
st.caption(
    "HMM Regime Detection · XGBoost Direction Classification · "
    "GARCH Volatility-Adjusted Position Sizing"
)

st.markdown("---")

# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<p class="section-header">Current Regime</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:{regime_color(current_regime)};
            padding:20px;border-radius:12px;text-align:center;color:white;
            font-size:28px;font-weight:700;letter-spacing:0.05em;">
            {current_regime}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"As of {latest_date.date()}")

    with c2:
        st.markdown('<p class="section-header">Next-Day Signal</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:{signal_color};
            padding:20px;border-radius:12px;text-align:center;color:white;
            font-size:28px;font-weight:700;">
            {signal_text}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"Confidence: {latest_confidence:.1%} | P(Up): {latest_prob_up:.1%}")

    with c3:
        st.markdown('<p class="section-header">GARCH Position Size</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:#1d4ed8;
            padding:20px;border-radius:12px;text-align:center;color:white;
            font-size:28px;font-weight:700;">
            {latest_position_size:.1%}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"Active position after signal: {latest_active_position:.1%}")

    with c4:
        st.markdown('<p class="section-header">GARCH Forecast</p>', unsafe_allow_html=True)

        vol_color = (
            "#16a34a" if latest_vol < 1.0
            else "#f97316" if latest_vol < 2.0
            else "#dc2626"
        )

        st.markdown(
            f"""
            <div style="background:{vol_color};
            padding:20px;border-radius:12px;text-align:center;color:white;
            font-size:28px;font-weight:700;">
            {latest_vol:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Forecasted daily volatility")

    st.markdown("---")

    st.markdown("### 📊 Key Performance Metrics")

    best_s = summary.get("best", {})
    orig_s = summary.get("original", {})
    bh_s = summary.get("buyhold", {})

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.metric(
            "Gross Sharpe",
            format_num(safe_get(best_s, "Gross Sharpe")),
            f"BH: {format_num(safe_get(bh_s, 'Gross Sharpe'))}"
        )

    with m2:
        st.metric(
            "Net Sharpe",
            format_num(safe_get(best_s, "Net Sharpe")),
            f"BH: {format_num(safe_get(bh_s, 'Net Sharpe'))}"
        )

    with m3:
        st.metric(
            "Gross Return",
            format_pct(safe_get(best_s, "Gross Return")),
            f"BH: {format_pct(safe_get(bh_s, 'Gross Return'))}"
        )

    with m4:
        st.metric(
            "Net Return",
            format_pct(safe_get(best_s, "Net Return")),
            f"BH: {format_pct(safe_get(bh_s, 'Net Return'))}"
        )

    with m5:
        st.metric(
            "Max Drawdown",
            format_pct(safe_get(best_s, "Net Max Drawdown")),
            f"BH: {format_pct(safe_get(bh_s, 'Net Max Drawdown'))}"
        )

    with m6:
        st.metric(
            "Win Rate",
            format_pct(safe_get(best_s, "Win Rate")),
            f"Trades: {int(safe_get(best_s, 'Days Traded', 0))}"
        )

    st.markdown("---")

    st.markdown("### 📈 Cumulative Returns")

    fig_cum = go.Figure()

    fig_cum.add_trace(go.Scatter(
        x=original_df.index,
        y=original_df[orig_cum_col],
        name=f"Original Net ({format_pct(safe_get(orig_s, 'Net Return'))})",
        line=dict(color="#38bdf8", width=2)
    ))

    fig_cum.add_trace(go.Scatter(
        x=best_df.index,
        y=best_df[best_cum_col],
        name=f"Best Net ({format_pct(safe_get(best_s, 'Net Return'))})",
        line=dict(color="#818cf8", width=2, dash="dash")
    ))

    fig_cum.add_trace(go.Scatter(
        x=buyhold_df.index,
        y=buyhold_df[buyhold_cum_col],
        name=f"Buy & Hold ({format_pct(safe_get(bh_s, 'Gross Return'))})",
        line=dict(color="#94a3b8", width=2, dash="dot")
    ))

    fig_cum.add_hline(y=1.0, line_color="#475569", line_dash="dash", opacity=0.5)

    fig_cum.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=420,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=20, b=60)
    )

    st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("### 📉 Drawdown Curve")

    best_dd = drawdown_series(best_df[best_cum_col])
    bh_dd = drawdown_series(buyhold_df[buyhold_cum_col])

    fig_dd = go.Figure()

    fig_dd.add_trace(go.Scatter(
        x=best_df.index,
        y=best_dd,
        fill="tozeroy",
        name="Best Strategy",
        line=dict(color="#818cf8"),
        fillcolor="rgba(129,140,248,0.2)"
    ))

    fig_dd.add_trace(go.Scatter(
        x=buyhold_df.index,
        y=bh_dd,
        fill="tozeroy",
        name="Buy & Hold",
        line=dict(color="#94a3b8", dash="dash"),
        fillcolor="rgba(148,163,184,0.1)"
    ))

    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=320,
        xaxis_title="Date",
        yaxis_title="Drawdown from Peak",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=20, b=60)
    )

    st.plotly_chart(fig_dd, use_container_width=True)

# ============================================================
# PAGE: PERFORMANCE
# ============================================================

elif page == "Performance":

    st.markdown("### 📋 Full Performance Comparison")

    best_s = summary.get("best", {})
    orig_s = summary.get("original", {})
    bh_s = summary.get("buyhold", {})

    metrics_table = pd.DataFrame({
        "Metric": [
            "Gross Return",
            "Net Return",
            "Gross Sharpe",
            "Net Sharpe",
            "Sortino Ratio",
            "Calmar Ratio",
            "Gross Max Drawdown",
            "Net Max Drawdown",
            "Win Rate",
            "Days Traded",
            "Days Flat",
            "Total Turnover",
            "Total Cost"
        ],
        "Original XGB+GARCH": [
            format_pct(safe_get(orig_s, "Gross Return")),
            format_pct(safe_get(orig_s, "Net Return")),
            format_num(safe_get(orig_s, "Gross Sharpe")),
            format_num(safe_get(orig_s, "Net Sharpe")),
            format_num(safe_get(orig_s, "Sortino Ratio")),
            format_num(safe_get(orig_s, "Calmar Ratio")),
            format_pct(safe_get(orig_s, "Gross Max Drawdown")),
            format_pct(safe_get(orig_s, "Net Max Drawdown")),
            format_pct(safe_get(orig_s, "Win Rate")),
            str(int(safe_get(orig_s, "Days Traded", 0))),
            str(int(safe_get(orig_s, "Days Flat", 0))),
            f"{safe_get(orig_s, 'Total Turnover', 0):.2f}",
            f"{safe_get(orig_s, 'Total Cost', 0):.4f}"
        ],
        "Best XGB+GARCH": [
            format_pct(safe_get(best_s, "Gross Return")),
            format_pct(safe_get(best_s, "Net Return")),
            format_num(safe_get(best_s, "Gross Sharpe")),
            format_num(safe_get(best_s, "Net Sharpe")),
            format_num(safe_get(best_s, "Sortino Ratio")),
            format_num(safe_get(best_s, "Calmar Ratio")),
            format_pct(safe_get(best_s, "Gross Max Drawdown")),
            format_pct(safe_get(best_s, "Net Max Drawdown")),
            format_pct(safe_get(best_s, "Win Rate")),
            str(int(safe_get(best_s, "Days Traded", 0))),
            str(int(safe_get(best_s, "Days Flat", 0))),
            f"{safe_get(best_s, 'Total Turnover', 0):.2f}",
            f"{safe_get(best_s, 'Total Cost', 0):.4f}"
        ],
        "Buy & Hold": [
            format_pct(safe_get(bh_s, "Gross Return")),
            format_pct(safe_get(bh_s, "Net Return")),
            format_num(safe_get(bh_s, "Gross Sharpe")),
            format_num(safe_get(bh_s, "Net Sharpe")),
            format_num(safe_get(bh_s, "Sortino Ratio")),
            format_num(safe_get(bh_s, "Calmar Ratio")),
            format_pct(safe_get(bh_s, "Gross Max Drawdown")),
            format_pct(safe_get(bh_s, "Net Max Drawdown")),
            format_pct(safe_get(bh_s, "Win Rate")),
            str(int(safe_get(bh_s, "Days Traded", len(buyhold_df)))),
            str(int(safe_get(bh_s, "Days Flat", 0))),
            f"{safe_get(bh_s, 'Total Turnover', 1):.2f}",
            f"{safe_get(bh_s, 'Total Cost', 0):.4f}"
        ]
    })

    st.table(metrics_table.astype(str))

    st.markdown("---")

    st.markdown("### 📐 Rolling 60-Day Sharpe Ratio")

    window = 60

    roll_best = (
        best_df[best_ret_col].rolling(window).mean() /
        best_df[best_ret_col].rolling(window).std()
    ) * np.sqrt(252)

    roll_orig = (
        original_df[orig_ret_col].rolling(window).mean() /
        original_df[orig_ret_col].rolling(window).std()
    ) * np.sqrt(252)

    roll_bh = (
        buyhold_df[buyhold_ret_col].rolling(window).mean() /
        buyhold_df[buyhold_ret_col].rolling(window).std()
    ) * np.sqrt(252)

    fig_rs = go.Figure()

    fig_rs.add_trace(go.Scatter(
        x=best_df.index,
        y=roll_best,
        name="Best Strategy",
        line=dict(color="#818cf8", width=1.5)
    ))

    fig_rs.add_trace(go.Scatter(
        x=original_df.index,
        y=roll_orig,
        name="Original",
        line=dict(color="#38bdf8", width=1.5, dash="dash")
    ))

    fig_rs.add_trace(go.Scatter(
        x=buyhold_df.index,
        y=roll_bh,
        name="Buy & Hold",
        line=dict(color="#94a3b8", width=1.5, dash="dot")
    ))

    fig_rs.add_hline(y=0, line_color="#ef4444", line_dash="dash", opacity=0.6)
    fig_rs.add_hline(y=1, line_color="#22c55e", line_dash="dash", opacity=0.4)

    fig_rs.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=400,
        xaxis_title="Date",
        yaxis_title="Rolling Sharpe",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=60, t=20, b=60)
    )

    st.plotly_chart(fig_rs, use_container_width=True)

# ============================================================
# PAGE: REGIME ANALYSIS
# ============================================================

elif page == "Regime Analysis":

    st.markdown("### 🗺️ Regime Distribution")

    regime_counts = best_df["regime"].value_counts().reset_index()
    regime_counts.columns = ["Regime", "Days"]
    regime_counts["Color"] = regime_counts["Regime"].apply(regime_color)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_bar = go.Figure(go.Bar(
            x=regime_counts["Regime"],
            y=regime_counts["Days"],
            marker_color=regime_counts["Color"],
            text=regime_counts["Days"],
            textposition="outside"
        ))

        fig_bar.update_layout(
            title="Days per Regime",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            height=380,
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_pie = go.Figure(go.Pie(
            labels=regime_counts["Regime"],
            values=regime_counts["Days"],
            marker_colors=regime_counts["Color"],
            hole=0.4,
            textinfo="label+percent"
        ))

        fig_pie.update_layout(
            title="Regime Breakdown",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📅 Regime Timeline")

    fig_reg = go.Figure()

    for regime, color in [
        ("Bull", "#16a34a"),
        ("Bear", "#dc2626"),
        ("High_Vol", "#f97316")
    ]:
        mask = best_df["regime"] == regime

        y_values = (
            best_df.loc[mask, "adj_close"]
            if "adj_close" in best_df.columns
            else np.arange(mask.sum())
        )

        fig_reg.add_trace(go.Scatter(
            x=best_df.index[mask],
            y=y_values,
            mode="markers",
            name=regime,
            marker=dict(color=color, size=5, opacity=0.8)
        ))

    fig_reg.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=380,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=20, b=60)
    )

    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📊 Performance by Regime")

    regime_data = []

    for regime in ["Bull", "Bear", "High_Vol"]:
        subset = best_df[best_df["regime"] == regime]

        if len(subset) == 0:
            continue

        avg_ret = subset[best_ret_col].mean()
        total_ret = (1 + subset[best_ret_col]).prod() - 1
        sr = sharpe_ratio(subset[best_ret_col])
        wr = (subset[best_ret_col] > 0).mean()

        regime_data.append({
            "Regime": regime,
            "Days": len(subset),
            "Avg Daily Return": format_pct(avg_ret),
            "Total Return": format_pct(total_ret),
            "Net Sharpe": format_num(sr),
            "Win Rate": format_pct(wr)
        })

    if regime_data:
        st.table(pd.DataFrame(regime_data).astype(str))

# ============================================================
# PAGE: GARCH VOLATILITY
# ============================================================

elif page == "GARCH Volatility":

    st.markdown("### ⚡ GARCH Volatility Analysis")

    max_vol = max(5, float(best_df["garch_vol_forecast"].max()) * 1.1)
    avg_vol = float(best_df["garch_vol_forecast"].mean())

    # FIXED GAUGE:
    # Do NOT put "suffix" inside delta.
    # Suffix belongs in number or axis.
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=latest_vol,
        number={
            "suffix": "%",
            "valueformat": ".2f"
        },
        delta={
            "reference": avg_vol,
            "valueformat": ".2f",
            "relative": False
        },
        title={"text": "Current Daily Volatility Forecast"},
        gauge={
            "axis": {
                "range": [0, max_vol],
                "ticksuffix": "%"
            },
            "bar": {"color": "#38bdf8"},
            "steps": [
                {"range": [0, 1.0], "color": "#166534"},
                {"range": [1.0, 2.0], "color": "#854d0e"},
                {"range": [2.0, max_vol], "color": "#7f1d1d"}
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 3},
                "thickness": 0.75,
                "value": avg_vol
            }
        }
    ))

    fig_gauge.update_layout(
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=360,
        margin=dict(l=40, r=40, t=40, b=20)
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption(
        f"Delta compares current volatility forecast with average volatility "
        f"({avg_vol:.2f}%)."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Vol", f"{latest_vol:.3f}%")

    with col2:
        st.metric("Average Vol", f"{avg_vol:.3f}%")

    with col3:
        st.metric("GARCH Position Size", f"{latest_position_size:.1%}")

    with col4:
        st.metric("Active Position", f"{latest_active_position:.1%}")

    st.markdown("---")

    st.markdown("### 📉 Volatility and Position Size Over Time")

    fig_vol = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=[
            "GARCH Volatility Forecast",
            "GARCH Position Size vs Active Position"
        ]
    )

    fig_vol.add_trace(go.Scatter(
        x=best_df.index,
        y=best_df["garch_vol_forecast"],
        name="GARCH Vol",
        line=dict(color="#f97316", width=1.5)
    ), row=1, col=1)

    fig_vol.add_hline(
        y=avg_vol,
        line_color="#94a3b8",
        line_dash="dash",
        annotation_text="Average Vol",
        row=1,
        col=1
    )

    if "position_size" in best_df.columns:
        fig_vol.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df["position_size"],
            name="GARCH Position Size",
            line=dict(color="#38bdf8", width=1.5)
        ), row=2, col=1)

    if "best_position" in best_df.columns:
        fig_vol.add_trace(go.Scatter(
            x=best_df.index,
            y=best_df["best_position"],
            name="Active Position",
            line=dict(color="#818cf8", width=1.5, dash="dash")
        ), row=2, col=1)

    fig_vol.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=520,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=40, r=20, t=40, b=60)
    )

    fig_vol.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    fig_vol.update_yaxes(title_text="Position", row=2, col=1)

    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📊 Volatility Distribution")

    fig_hist = go.Figure(go.Histogram(
        x=best_df["garch_vol_forecast"],
        nbinsx=40,
        marker_color="#38bdf8",
        opacity=0.8,
        name="GARCH Vol"
    ))

    fig_hist.add_vline(
        x=avg_vol,
        line_color="#f97316",
        line_dash="dash",
        annotation_text=f"Mean: {avg_vol:.2f}%"
    )

    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=330,
        xaxis_title="Daily Vol Forecast (%)",
        yaxis_title="Frequency",
        margin=dict(l=40, r=20, t=20, b=40)
    )

    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# PAGE: SIGNAL EXPLORER
# ============================================================

elif page == "Signal Explorer":

    st.markdown("### 🔍 Signal Explorer")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        start_date = st.date_input(
            "From",
            value=best_df.index[0].date(),
            min_value=best_df.index[0].date(),
            max_value=best_df.index[-1].date()
        )

    with col_f2:
        end_date = st.date_input(
            "To",
            value=best_df.index[-1].date(),
            min_value=best_df.index[0].date(),
            max_value=best_df.index[-1].date()
        )

    filtered = best_df.loc[str(start_date):str(end_date)].copy()

    st.markdown(f"Showing **{len(filtered)}** trading days")

    if "adj_close" in filtered.columns:
        fig_sig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            subplot_titles=[
                "Price + Signals",
                "Prediction Probability"
            ]
        )

        fig_sig.add_trace(go.Scatter(
            x=filtered.index,
            y=filtered["adj_close"],
            name="S&P 500",
            line=dict(color="#e2e8f0", width=1.5)
        ), row=1, col=1)

        up_mask = filtered["predicted_direction"] == 1
        down_mask = filtered["predicted_direction"] == 0

        fig_sig.add_trace(go.Scatter(
            x=filtered.index[up_mask],
            y=filtered.loc[up_mask, "adj_close"],
            mode="markers",
            name="Up Signal",
            marker=dict(color="#16a34a", size=7, symbol="triangle-up")
        ), row=1, col=1)

        fig_sig.add_trace(go.Scatter(
            x=filtered.index[down_mask],
            y=filtered.loc[down_mask, "adj_close"],
            mode="markers",
            name="Down Signal",
            marker=dict(color="#dc2626", size=7, symbol="triangle-down")
        ), row=1, col=1)

        fig_sig.add_trace(go.Bar(
            x=filtered.index,
            y=filtered["prob_up"],
            name="P(Up)",
            marker_color=[
                "#16a34a" if p > 0.52 else "#dc2626"
                for p in filtered["prob_up"]
            ],
            opacity=0.7
        ), row=2, col=1)

        fig_sig.add_hline(
            y=0.52,
            line_color="#f97316",
            line_dash="dash",
            annotation_text="Threshold 0.52",
            row=2,
            col=1
        )

        fig_sig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            height=560,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=40, r=20, t=40, b=60)
        )

        st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📋 Recent Model Signals")

    display_cols = [
        col for col in [
            "regime",
            "predicted_direction",
            "prob_up",
            "garch_vol_forecast",
            "position_size",
            "best_position",
            best_ret_col
        ] if col in best_df.columns
    ]

    recent = best_df[display_cols].tail(15).copy()
    recent.index = recent.index.date

    if "predicted_direction" in recent.columns:
        recent["Signal"] = recent["predicted_direction"].apply(signal_label)

    st.table(recent.astype(str))

    st.markdown("---")

    st.markdown("### 📊 Model Confidence Distribution")

    fig_conf = go.Figure()

    fig_conf.add_trace(go.Histogram(
        x=best_df[best_df["predicted_direction"] == 1]["prob_up"],
        name="Up Predictions",
        marker_color="#16a34a",
        opacity=0.7,
        nbinsx=30
    ))

    fig_conf.add_trace(go.Histogram(
        x=best_df[best_df["predicted_direction"] == 0]["prob_up"],
        name="Down Predictions",
        marker_color="#dc2626",
        opacity=0.7,
        nbinsx=30
    ))

    fig_conf.add_vline(
        x=0.52,
        line_color="#f97316",
        line_dash="dash",
        annotation_text="Threshold 0.52"
    )

    fig_conf.update_layout(
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=340,
        xaxis_title="Probability of Up Move",
        yaxis_title="Count",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=20, b=60)
    )

    st.plotly_chart(fig_conf, use_container_width=True)

# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "About":

    st.markdown("### 📚 About This Project")

    st.markdown("""
    **Multi-Factor Regime-Aware Stock Return Prediction and Risk-Adjusted Trading Strategy**

    This dashboard presents the results of an end-to-end machine learning and quantitative finance project. 
    The framework combines HMM market regime detection, XGBoost next-day direction classification, 
    and GARCH volatility-adjusted position sizing.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 Methodology")
        st.markdown("""
        **Stage 1 — Feature Engineering**
        - Daily S&P 500 price data
        - Macro features from FRED
        - Technical indicators including RSI, MACD, Bollinger width, ATR, and volatility

        **Stage 2 — Regime Detection**
        - Gaussian HMM with 3 market states
        - Bull, Bear, and High-Volatility regimes

        **Stage 3 — XGBoost Classifier**
        - Target: next-day return direction
        - Walk-forward cross-validation
        - Optuna hyperparameter tuning

        **Stage 4 — GARCH Position Sizing**
        - GARCH(1,1) volatility forecast
        - Inverse-volatility position sizing
        - Position capped between 0% and 100%

        **Stage 5 — Backtesting**
        - Out-of-sample 2025 evaluation
        - Transaction cost: 0.1% per trade
        - Metrics: return, Sharpe, Sortino, Calmar, drawdown, and win rate
        """)

    with col2:
        st.markdown("#### 📊 Key Results")

        best_s = summary.get("best", {})
        bh_s = summary.get("buyhold", {})

        st.markdown(f"""
        | Metric | Best Strategy | Buy & Hold |
        |--------|--------------|------------|
        | Gross Sharpe | {format_num(safe_get(best_s, 'Gross Sharpe'))} | {format_num(safe_get(bh_s, 'Gross Sharpe'))} |
        | Net Sharpe | {format_num(safe_get(best_s, 'Net Sharpe'))} | {format_num(safe_get(bh_s, 'Net Sharpe'))} |
        | Gross Return | {format_pct(safe_get(best_s, 'Gross Return'))} | {format_pct(safe_get(bh_s, 'Gross Return'))} |
        | Net Return | {format_pct(safe_get(best_s, 'Net Return'))} | {format_pct(safe_get(bh_s, 'Net Return'))} |
        | Max Drawdown | {format_pct(safe_get(best_s, 'Net Max Drawdown'))} | {format_pct(safe_get(bh_s, 'Net Max Drawdown'))} |
        | Win Rate | {format_pct(safe_get(best_s, 'Win Rate'))} | {format_pct(safe_get(bh_s, 'Win Rate'))} |
        | Days Traded | {int(safe_get(best_s, 'Days Traded', 0))} | {int(safe_get(bh_s, 'Days Traded', len(buyhold_df)))} |
        """)

        st.markdown("#### 🔑 Interpretation")
        st.markdown("""
        - Buy-and-hold delivered the strongest total return in 2025.
        - The XGBoost + GARCH strategy reduced maximum drawdown.
        - The best strategy improved over the original model after transaction costs.
        - The framework is more useful for risk management than pure return maximization.
        """)

        st.markdown("#### 📦 Tech Stack")
        st.markdown("""
        `Python` · `pandas` · `numpy` · `scikit-learn` · `hmmlearn` · 
        `xgboost` · `optuna` · `arch` · `streamlit` · `plotly`
        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Out-of-sample backtest period: Jan 2025 – Dec 2025 · "
    "Model trained on: Jan 2015 – Dec 2024 · "
    "Transaction cost: 0.1% per trade"
)
