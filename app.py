import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Analytics", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM UI/UX CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Theme Base */
    [data-testid="stAppViewContainer"] { 
        background: linear-gradient(180deg, #090B10 0%, #111520 100%);
        color: #F3F4F6; 
        font-family: 'Outfit', sans-serif; 
    }
    [data-testid="stSidebar"] { 
        background-color: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
    }
    
    div.block-container { 
        padding-top: 1.5rem; 
        padding-bottom: 3rem;
    }
    
    /* Metric Cards with Glassmorphism & Hover */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Typography within cards */
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .text-green { color: #10B981; font-size: 0.95rem; font-weight: 500; }
    .text-red { color: #EF4444; font-size: 0.95rem; font-weight: 500; }
    .text-neutral { color: #94A3B8; font-size: 0.95rem; font-weight: 500; }
    
    /* Stress testing explicit cards */
    .stress-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 4px solid #3B82F6;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    .stress-card:hover {
        transform: translateX(4px);
    }
    .stress-red { border-left-color: #EF4444; }
    .stress-green { border-left-color: #10B981; }
    .stress-orange { border-left-color: #F59E0B; }
    
    /* Customizing Tabs for a sleeker look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        background-color: transparent;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        padding-top: 12px;
        padding-bottom: 12px;
        color: #94A3B8;
        font-weight: 500;
        font-size: 1.05rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8;
        background-color: transparent;
        border-bottom: 2px solid #38BDF8 !important;
    }
    
    /* Clean up native Streamlit elements */
    [data-testid="stMarkdownContainer"] h3 {
        font-weight: 600;
        color: #F1F5F9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Custom Table header styling */
    th {
        background-color: #0F172A !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    td {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border-bottom: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to inject common plotting layouts
def apply_premium_layout(fig, title="", x_title="", y_title=""):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(family="Outfit", size=18, color="#F8FAFC")),
        xaxis=dict(title=x_title, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", title_font=dict(color="#94A3B8")),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", title_font=dict(color="#94A3B8")),
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Outfit", color="#CBD5E1"),
        hovermode="x unified"
    )
    return fig

# --- DATA FETCHING (CACHED) ---
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_data(tickers_str, period):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return pd.DataFrame(), [], []
    
    fetch_list = tickers.copy()
    if "^GSPC" not in fetch_list:
        fetch_list.append("^GSPC")
        
    try:
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            try:
                data = yf.download(fetch_list, period=period, progress=False)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to fetch data: {e}")
                    return pd.DataFrame(), tickers, []
                time.sleep(2 ** attempt)
        
        if data is None or data.empty:
            return pd.DataFrame(), tickers, []
            
        if "Adj Close" in data:
            prices = data["Adj Close"]
        elif "Close" in data:
            prices = data["Close"]
        else:
            return pd.DataFrame(), tickers, []
            
        if isinstance(prices, pd.Series): 
            prices = prices.to_frame(name=fetch_list[0])
            
        prices.dropna(how='all', inplace=True)
        prices.ffill(inplace=True)
        prices.bfill(inplace=True)
        
        valid_tickers = [t for t in tickers if t in prices.columns and not prices[t].isna().all()]
        
        return prices, valid_tickers, fetch_list
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame(), tickers, []
        
# --- METRICS CALCULATIONS ---
def get_portfolio_returns(prices, weights_dict, active_tickers):
    if prices.empty or not active_tickers:
        return pd.Series(dtype=float)
    daily_returns = prices[active_tickers].pct_change().dropna()
    weights_array = np.array([weights_dict[t] for t in active_tickers])
    port_returns = daily_returns.dot(weights_array)
    return port_returns

def calculate_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    return drawdown

def historical_var(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

def parametric_var(returns, confidence_level=0.95):
    mu = np.mean(returns)
    std = np.std(returns)
    return norm.ppf(1 - confidence_level, mu, std)

def conditional_var(returns, confidence_level=0.95):
    var_threshold = historical_var(returns, confidence_level)
    return np.mean(returns[returns <= var_threshold])

# --- UI COMPONENTS ---
def render_metric_card(title, value, prev_value_str="", color_class="text-neutral"):
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="{color_class}">{prev_value_str}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- APP START ---
st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #F8FAFC;'>🏦 Configuration</h2>", unsafe_allow_html=True)

ticker_input = st.sidebar.text_input("Assets (comma separated)", "AAPL, MSFT, GOOGL, AMZN")
period_dict = {"1 Year": "1y", "3 Years": "3y", "5 Years": "5y", "Max": "max"}
period_label = st.sidebar.selectbox("Historical Window", list(period_dict.keys()), index=0)
period = period_dict[period_label]

rebalance = st.sidebar.selectbox("Rebalance Strategy", ["Buy and Hold", "Monthly Rebalance", "Quarterly Rebalance"])

with st.spinner("Synchronizing market feeds..."):
    prices, valid_tickers, all_fetched = fetch_data(ticker_input, period)

if prices.empty or not valid_tickers:
    st.warning("⚠️ No valid data found for the given assets. Please adjust your inputs.")
    st.stop()

st.sidebar.markdown("### Portfolio Weightings")
weighting_scheme = st.sidebar.radio("Allocation", ["Equal Weight", "Custom Weights"])

weights = {}
if weighting_scheme == "Equal Weight":
    w = 1.0 / len(valid_tickers)
    for t in valid_tickers:
        weights[t] = w
    st.sidebar.info(f"Target: {w*100:.1f}% distribution")
else:
    raw_weights = []
    for t in valid_tickers:
        w = st.sidebar.number_input(f"{t} Weight (%)", min_value=0.0, max_value=100.0, value=100.0/len(valid_tickers), step=1.0)
        raw_weights.append(w)
    
    total = sum(raw_weights)
    if total == 0:
        st.sidebar.error("Valid configuration required.")
        weights = {t: 1.0/len(valid_tickers) for t in valid_tickers}
    else:
        for t, w in zip(valid_tickers, raw_weights):
            weights[t] = w / total
        
        if abs(total - 100.0) > 0.1:
            st.sidebar.warning(f"Weights normalized to 100%.")

port_returns = get_portfolio_returns(prices, weights, valid_tickers)
port_cum_returns = (1 + port_returns).cumprod() - 1

if "^GSPC" in prices.columns:
    sp500_ret = prices["^GSPC"].pct_change().dropna()
    sp500_cum = (1 + sp500_ret).cumprod() - 1
else:
    sp500_ret = pd.Series(dtype=float)
    sp500_cum = pd.Series(dtype=float)

if not sp500_ret.empty:
    common_idx = port_returns.index.intersection(sp500_ret.index)
    port_returns = port_returns.loc[common_idx]
    sp500_ret = sp500_ret.loc[common_idx]
    port_cum_returns = (1 + port_returns).cumprod() - 1
    sp500_cum = (1 + sp500_ret).cumprod() - 1

# Top Metrics Calculations
ann_return = port_returns.mean() * 252
ann_vol = port_returns.std() * np.sqrt(252)
risk_free = 0.04 
sharpe = (ann_return - risk_free) / ann_vol if ann_vol != 0 else 0

downside_returns = port_returns[port_returns < 0]
downside_vol = downside_returns.std() * np.sqrt(252)
sortino = (ann_return - risk_free) / downside_vol if (not downside_returns.empty and downside_vol != 0) else 0

cov_matrix = np.cov(port_returns, sp500_ret) if (not sp500_ret.empty and len(port_returns) > 1) else None
beta = cov_matrix[0, 1] / cov_matrix[1, 1] if (cov_matrix is not None and cov_matrix.shape == (2,2) and cov_matrix[1, 1] != 0) else 1.0
treynor = (ann_return - risk_free) / beta if beta != 0 else 0

tracking_diff = port_returns - (sp500_ret if not sp500_ret.empty else 0)
info_ratio = (ann_return - (sp500_ret.mean() * 252)) / (tracking_diff.std() * np.sqrt(252)) if (not sp500_ret.empty and tracking_diff.std() != 0) else 0

port_skew = skew(port_returns)
port_kurtosis = kurtosis(port_returns)

st.markdown("<h1 style='font-family: Outfit; font-weight: 700; font-size: 2.5rem; margin-bottom: 2rem;'>Portfolio Analytics Terminal</h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    color = "text-green" if ann_return >= 0 else "text-red"
    # Added plus sign appropriately
    sgn = "+" if ann_return > 0 else ""
    render_metric_card("Annualized Return", f"{sgn}{ann_return*100:.2f}%", f"Market Beta: {beta:.2f}", color)
with col2:
    render_metric_card("Portfolio Volatility", f"{ann_vol*100:.2f}%", f"Sortino Ratio: {sortino:.2f}", "text-neutral")
with col3:
    color = "text-green" if sharpe >= 1.0 else "text-orange" if sharpe > 0.5 else "text-red"
    render_metric_card("Sharpe Ratio", f"{sharpe:.2f}", f"Treynor Ratio: {treynor:.2f}", color)
with col4:
    dd_series = calculate_drawdown(port_returns)
    max_dd = dd_series.min() if not dd_series.empty else 0
    color = "text-red" if max_dd < -0.2 else "text-neutral"
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    render_metric_card("Maximum Drawdown", f"{max_dd*100:.2f}%", f"Calmar Ratio: {calmar:.2f}", color)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview", 
    "⚠️  Risk Metrics", 
    "💥  Stress Tests", 
    "🔗  Correlations", 
    "🎲  Simulations"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    col_chart, col_stats = st.columns([3, 1])
    with col_chart:
        st.markdown("### Cumulative Historical Growth")
        fig = go.Figure()
        
        # Adding gradient fill to active portfolio line for aesthetics
        fig.add_trace(go.Scatter(
            x=port_cum_returns.index, y=port_cum_returns*100, 
            mode='lines', name='Your Portfolio', 
            line=dict(color='#38BDF8', width=2.5),
            fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)'
        ))
        
        if not sp500_cum.empty:
            fig.add_trace(go.Scatter(
                x=sp500_cum.index, y=sp500_cum*100, 
                mode='lines', name='S&P 500', 
                line=dict(color='#64748B', width=2, dash='dot')
            ))
            
        fig = apply_premium_layout(fig, y_title="Cumulative Return (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("### Allocation Summary")
        weights_df = pd.DataFrame({"Ticker": list(weights.keys()), "Weight": list(weights.values())})
        
        fig_pie = px.pie(
            weights_df, values='Weight', names='Ticker', hole=0.7,
            color_discrete_sequence=['#38BDF8', '#818CF8', '#C084FC', '#F472B6', '#FB923C', '#FCD34D']
        )
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0), showlegend=True, 
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_pie, use_container_width=True, height=250)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 15px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="color:#94A3B8; margin:0; font-size:0.85em; text-transform:uppercase;">Statistical Profile</p>
            <div style="margin-top:10px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#E2E8F0;">Information Ratio</span>
                    <strong style="color:#38BDF8;">{:.2f}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#E2E8F0;">Skewness</span>
                    <strong style="color:#38BDF8;">{:.2f}</strong>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#E2E8F0;">Kurtosis</span>
                    <strong style="color:#38BDF8;">{:.2f}</strong>
                </div>
            </div>
        </div>
        """.format(info_ratio, port_skew, port_kurtosis), unsafe_allow_html=True)

# --- TAB 2: RISK METRICS ---
with tab2:
    col_var, col_dd = st.columns(2)
    
    with col_var:
        st.markdown("### Daily Value at Risk (VaR)")
        if len(port_returns) > 0:
            var_95_h = historical_var(port_returns, 0.95) * 100
            var_99_h = historical_var(port_returns, 0.99) * 100
            var_95_p = parametric_var(port_returns, 0.95) * 100
            cvar_95 = conditional_var(port_returns, 0.95) * 100
            
            var_data = {
                "Calculation Model": ["Historical (95%)", "Historical (99%)", "Parametric (95%)", "Expected Shortfall (95%)"],
                "Daily VaR (%)": [f"{var_95_h:.2f}%", f"{var_99_h:.2f}%", f"{var_95_p:.2f}%", f"{cvar_95:.2f}%"],
                "Loss per $10k Exposure": [f"${abs(var_95_h)*100:.2f}", f"${abs(var_99_h)*100:.2f}", f"${abs(var_95_p)*100:.2f}", f"${abs(cvar_95)*100:.2f}"]
            }
            # Custom styled dataframe presentation
            st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)
            st.info(f"💡 Interpret: On 95% of trading days, the maximum daily loss is expected to be no worse than **{var_95_h:.2f}%**.")
        else:
            st.write("Insufficient data to compute distributions.")
            
    with col_dd:
        st.markdown("### Drawdown Profile")
        if not dd_series.empty:
            fig_dd = px.area(dd_series * 100, x=dd_series.index, y=dd_series.values*100)
            fig_dd.update_traces(line_color='#EF4444', fillcolor='rgba(239, 68, 68, 0.2)')
            fig_dd = apply_premium_layout(fig_dd, y_title="Drawdown Depth (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

# --- TAB 3: STRESS TESTING ---
with tab3:
    st.markdown("### Event Scenario Simulation")
    st.write("Predictive portfolio adjustments modeling sudden, sharp broad market shocks based on underlying asset beta metrics.")
    
    scenarios = {
        "Global Pandemic Crash": {"shock": -0.30, "desc": "Market drops 30%. High impact on travel/retail, slight buffer for big tech."},
        "Dot-Com / Tech Crash": {"shock": -0.25, "desc": "Market drops 25%. Growth & Tech multiples contract severely."},
        "2008 Financial Crisis": {"shock": -0.40, "desc": "Broad systemic liquidity crash (-40%). Financials take steepest hit."},
        "Inflation / Rate Hike Shock": {"shock": -0.15, "desc": "Sudden rate jumps (-15%). Growth stocks discount heavily, Energy/Banks buffer."}
    }
    
    if not sp500_ret.empty:
        asset_betas = {}
        var_m = sp500_ret.var()
        for t in valid_tickers:
            asset_ret = prices[t].pct_change().dropna()
            a_ret, m_ret = asset_ret.align(sp500_ret, join='inner')
            cov = np.cov(a_ret, m_ret)[0, 1] if len(a_ret) > 1 else 0
            asset_betas[t] = cov / var_m if var_m != 0 else 1.0
            
        c1, c2 = st.columns(2)
        cols = [c1, c2]
        
        for i, (s_name, details) in enumerate(scenarios.items()):
            s_shock = details["shock"]
            port_shock = 0
            for t in valid_tickers:
                b = asset_betas[t]
                if "Tech" in s_name and t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]:
                    b *= 1.5 
                if "Financial" in s_name and t in ["JPM", "BAC", "V", "MA", "GS", "MS"]:
                    b *= 1.4
                if "Rate" in s_name and t in ["XOM", "CVX", "JPM", "BAC"]:
                    b *= 0.5 
                port_shock += weights[t] * (b * s_shock)
                
            loss_class = "stress-red" if port_shock < -0.20 else ("stress-orange" if port_shock < -0.10 else "stress-green")
            with cols[i % 2]:
                st.markdown(f"""
                <div class="stress-card {loss_class}">
                    <h4 style="margin-top:0; color: #F8FAFC; font-weight:600; letter-spacing:0.02em;">{s_name}</h4>
                    <p style="color: #94A3B8; font-size: 0.9em; margin-bottom: 12px;">{details['desc']}</p>
                    <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                        <span style="color:#64748B; font-size:0.8em; text-transform:uppercase;">Estimated Impact</span>
                        <h2 style="margin:0; font-size:1.8rem;">{port_shock*100:.2f}%</h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Needs S&P 500 data for beta calculations in stress testing.")

    st.markdown("### Custom Stress Horizon")
    # Custom interactive shock gauge
    c_shock = st.slider("Assume a broad market shock of (%)", -50, 50, -10)
    if not sp500_ret.empty:
        c_impact = sum([weights[t] * (asset_betas[t] * (c_shock/100)) for t in valid_tickers])
        color = "#10B981" if c_impact > 0 else "#EF4444"
        sgn = "+" if c_impact > 0 else ""
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 30px; border-radius: 12px; text-align:center; border: 1px solid rgba(255,255,255,0.05);">
            <p style="margin-bottom:5px; color:#94A3B8; text-transform:uppercase; font-size:0.9em; letter-spacing:0.1em;">Resulting Portfolio Movement</p>
            <h1 style="color:{color}; font-size:3.5rem; margin:0; font-weight:700;">{sgn}{c_impact*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: CORRELATIONS ---
with tab4:
    col_heat, col_roll = st.columns([1, 1])
    
    daily_returns_all = prices[valid_tickers].pct_change().dropna()
    
    with col_heat:
        st.markdown("### Co-Movement Heatmap")
        if not daily_returns_all.empty and len(valid_tickers) > 1:
            corr_matrix = daily_returns_all.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                           z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                           colorscale='RdBu_r', zmin=-1, zmax=1, texttemplate="%{z:.2f}",
                           hoverinfo="z", showscale=False))
            
            fig_corr.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0), height=400,
                font=dict(family="Outfit", color="#CBD5E1")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # HHI Calculation beautifully formatted
            hhi = sum([w**2 for w in weights.values()]) * 10000
            hhi_text = "Highly Diversified" if hhi < 1500 else ("Moderately Concentrated" if hhi < 2500 else "Highly Concentrated")
            bar_color = "#10B981" if hhi < 1500 else ("#F59E0B" if hhi < 2500 else "#EF4444")
            
            st.markdown(f"""
            <div style="background:rgba(30, 41, 59, 0.4); padding:20px; border-radius:8px; border:1px solid rgba(255,255,255,0.05); margin-top:20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <span style="color:#94A3B8; text-transform:uppercase; font-size:0.85em;">Herfindahl-Hirschman Index (Waitings)</span>
                    <strong style="color:#F8FAFC;">{hhi:.0f}</strong>
                </div>
                <!-- Mini Progress Bar for HHI context -->
                <div style="background:#0F172A; height:8px; border-radius:4px; width:100%;">
                    <div style="background:{bar_color}; height:8px; border-radius:4px; width:{min((hhi/10000)*100, 100)}%;"></div>
                </div>
                <p style="text-align:right; margin-top:8px; margin-bottom:0; font-size:0.9em; color:{bar_color}; font-weight:600;">{hhi_text}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Requires at least 2 assets to evaluate correlations.")

    with col_roll:
        st.markdown("### 60-Day Rolling Beta vs Market")
        if not sp500_ret.empty and not daily_returns_all.empty:
            fig_roll = go.Figure()
            for t in valid_tickers:
                # Calculate rolling correlation dynamically
                roll_corr = daily_returns_all[t].rolling(window=60).corr(sp500_ret)
                fig_roll.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode='lines', name=t, line=dict(width=2)))
                
            fig_roll = apply_premium_layout(fig_roll, y_title="Correlation Coefficient")
            fig_roll.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("Requires S&P 500 equivalent data to render rolling trends.")


# --- TAB 5: MONTE CARLO ---
with tab5:
    st.markdown("### Predictive Path Modeling")
    st.write("Generates 10,000 distinct probability trajectories to gauge the next 252 trading days.")
    
    col_mc_btn, col_mc_empty = st.columns([1, 4])
    with col_mc_btn:
        run_mc = st.button("Simulate", type="primary", use_container_width=True)
    
    if run_mc:
        if not port_returns.empty:
            days = 252
            simulations = 10000
            
            mu = port_returns.mean()
            sigma = port_returns.std()
            drift = mu - (0.5 * sigma**2)
            
            np.random.seed(42)
            Z = np.random.normal(0, 1, (days, simulations))
            daily_sim_returns = np.exp(drift + sigma * Z)
            
            price_paths = np.zeros_like(daily_sim_returns)
            price_paths[0] = 1.0
            for t in range(1, days):
                price_paths[t] = price_paths[t-1] * daily_sim_returns[t]
                
            final_returns = price_paths[-1] - 1
            
            # Store results in session state to persist
            st.session_state['mc_paths'] = price_paths
            st.session_state['mc_final'] = final_returns
            st.session_state['mc_animating'] = True
        else:
            st.error("Historical variance structure insufficient to power simulator.")

    if 'mc_paths' in st.session_state:
        price_paths = st.session_state['mc_paths']
        final_returns = st.session_state['mc_final']
        days = price_paths.shape[0]
        
        col_mc1, col_mc2 = st.columns([2, 1])
        
        with col_mc1:
            chart_placeholder = st.empty()
            
            total_displayed_paths = 150
            global_median = np.median(price_paths, axis=1) * 100
            x_axis = np.arange(days)
            y_max = np.max(price_paths[:, :total_displayed_paths] * 100) * 1.05
            y_min = np.min(price_paths[:, :total_displayed_paths] * 100) * 0.95
            
            if st.session_state.get('mc_animating', False):
                progress_bar = st.progress(0)
                frames = 30
                paths_per_frame = total_displayed_paths // frames
                
                for step in range(1, frames + 1):
                    current_paths_count = step * paths_per_frame
                    
                    fig_mc = go.Figure()
                    for i in range(current_paths_count):
                        fig_mc.add_trace(go.Scatter(
                            x=x_axis, y=price_paths[:, i]*100, mode='lines', 
                            line=dict(width=1, color='rgba(56, 189, 248, 0.08)'), 
                            showlegend=False, hoverinfo='skip'
                        ))
                    
                    fig_mc.add_trace(go.Scatter(
                        x=x_axis, y=global_median, mode='lines', 
                        line=dict(width=3, color='#FCD34D'), 
                        name='50th Pctl Path'
                    ))
                    
                    fig_mc = apply_premium_layout(fig_mc, y_title="Portfolio Terminal Value (Base 100)")
                    fig_mc.update_layout(
                        title=dict(text=f"Rendering Path Configuration: {current_paths_count} / {total_displayed_paths}", font=dict(color="#38BDF8")),
                        xaxis=dict(range=[0, days]), yaxis=dict(range=[y_min, y_max])
                    )
                    
                    chart_placeholder.plotly_chart(fig_mc, use_container_width=True)
                    progress_bar.progress(step / frames)
                    time.sleep(0.08)
                
                progress_bar.empty()
                st.session_state['mc_animating'] = False
            
            # Final Chart Rendering
            fig_mc = go.Figure()
            for i in range(total_displayed_paths):
                fig_mc.add_trace(go.Scatter(
                    x=x_axis, y=price_paths[:, i]*100, mode='lines', 
                    line=dict(width=1, color='rgba(56, 189, 248, 0.08)'), 
                    showlegend=False, hoverinfo='skip'
                ))
            fig_mc.add_trace(go.Scatter(
                x=x_axis, y=global_median, mode='lines', 
                line=dict(width=3, color='#FCD34D'), 
                name='50th Pctl Path'
            ))
            fig_mc = apply_premium_layout(fig_mc, y_title="Portfolio Terminal Value (Base 100)")
            fig_mc.update_layout(title=dict(text="System Compilation Complete (10,000 Iterations)", font=dict(color="#10B981")))
            chart_placeholder.plotly_chart(fig_mc, use_container_width=True)

        with col_mc2:
            perc_5 = np.percentile(final_returns, 5) * 100
            perc_50 = np.percentile(final_returns, 50) * 100
            perc_95 = np.percentile(final_returns, 95) * 100
            prob_loss = np.mean(final_returns < 0) * 100
            
            sign_95 = "+" if perc_95 > 0 else ""
            sign_50 = "+" if perc_50 > 0 else ""
            sign_5 = "+" if perc_5 > 0 else ""
            st.markdown(f"<div style='background: rgba(30, 41, 59, 0.4); padding: 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); height: 100%;'><h3 style='margin-top:0; color:#F8FAFC; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:15px; margin-bottom:20px;'>Outcome Matrix</h3><p style='color:#94A3B8; margin-bottom:5px; font-size:0.9em; text-transform:uppercase;'>95th Percentile (Bull)</p><h2 style='color:#10B981; margin-top:0; margin-bottom:20px;'>{sign_95}{perc_95:.2f}%</h2><p style='color:#94A3B8; margin-bottom:5px; font-size:0.9em; text-transform:uppercase;'>50th Percentile (Median)</p><h2 style='color:#FCD34D; margin-top:0; margin-bottom:20px;'>{sign_50}{perc_50:.2f}%</h2><p style='color:#94A3B8; margin-bottom:5px; font-size:0.9em; text-transform:uppercase;'>5th Percentile (Bear)</p><h2 style='color:#EF4444; margin-top:0; margin-bottom:30px;'>{sign_5}{perc_5:.2f}%</h2><div style='background: rgba(15, 23, 42, 0.8); padding: 15px; border-radius: 8px;'><p style='color:#CBD5E1; margin:0; font-size:0.9em;'>Total Loss Probability</p><h2 style='color:#EF4444; margin:0;'>{prob_loss:.1f}%</h2></div></div>", unsafe_allow_html=True)