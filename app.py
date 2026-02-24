"""
app.py — Streamlit dashboard for the short-term trading assistant.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data as data_module
import indicators as ind_module
import ai_analysis
import watchlist as wl_module
import os

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Short-Term Trading Assistant")
st.caption("Signals for 1-3 day trades · Not financial advice · Data via Yahoo Finance")

# ── Session state ─────────────────────────────────────────────────────────

if "watchlist" not in st.session_state:
    st.session_state.watchlist = wl_module.load()

# ── Sidebar — watchlist management ────────────────────────────────────────

with st.sidebar:
    st.header("Watchlist")

    # Add ticker
    with st.form("add_ticker", clear_on_submit=True):
        new_ticker = st.text_input("Add ticker (e.g. NOVO-B.CO)", placeholder="NOVO-B.CO")
        if st.form_submit_button("Add"):
            if new_ticker.strip():
                st.session_state.watchlist = wl_module.add(
                    new_ticker.strip().upper(), st.session_state.watchlist
                )
                st.rerun()

    st.divider()

    # List tickers with remove buttons
    for ticker in list(st.session_state.watchlist):
        col1, col2 = st.columns([4, 1])
        col1.write(ticker)
        if col2.button("✕", key=f"rm_{ticker}", help=f"Remove {ticker}"):
            st.session_state.watchlist = wl_module.remove(ticker, st.session_state.watchlist)
            st.rerun()

    st.divider()
    st.subheader("Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Required for AI analysis. Set OPENAI_API_KEY env var or paste here.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        ai_analysis._CLIENT = None  # reset client when key changes

    lookback = st.selectbox("Price history", ["3mo", "6mo", "1mo"], index=0)
    st.caption("Data refreshes each time you open a ticker.")

# ── Main area: ticker selector + analysis ─────────────────────────────────

if not st.session_state.watchlist:
    st.info("Add tickers in the sidebar to get started.")
    st.stop()

selected = st.selectbox(
    "Select ticker to analyse",
    st.session_state.watchlist,
    label_visibility="collapsed",
)

# ── Fetch data ────────────────────────────────────────────────────────────

with st.spinner(f"Fetching data for {selected}…"):
    df     = data_module.fetch_ohlcv(selected, period=lookback)
    info   = data_module.fetch_info(selected)
    news   = data_module.fetch_news(selected)
    result = ind_module.compute_all(df)

if df.empty or not result:
    st.error(
        f"Could not fetch data for **{selected}**. "
        "Check the ticker symbol (e.g. `NOVO-B.CO` for Danish stocks)."
    )
    st.stop()

vals   = result["values"]
interp = result["interp"]
series = result["series"]

# ── Header row ────────────────────────────────────────────────────────────

st.subheader(f"{info.get('name', selected)}  ·  {selected}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"{vals['close']} {info.get('currency','')}")
col2.metric("RSI (14)", f"{vals['rsi']}")
col3.metric("ATR (14)", f"{vals['atr']}")

prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else vals["close"]
pct_change = (vals["close"] - prev_close) / prev_close * 100
col4.metric("1d change", f"{pct_change:+.2f}%", delta_color="normal")

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.55, 0.25, 0.20],
    subplot_titles=("Price + Bollinger Bands + EMAs", "MACD", "RSI"),
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"],
    name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=series["bb_upper"], name="BB Upper",
    line=dict(color="rgba(100,100,255,0.4)", dash="dot"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=series["bb_mid"], name="BB Mid",
    line=dict(color="rgba(100,100,255,0.6)", dash="dash"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=series["bb_lower"], name="BB Lower",
    line=dict(color="rgba(100,100,255,0.4)", dash="dot"),
    fill="tonexty", fillcolor="rgba(100,100,255,0.05)", showlegend=False), row=1, col=1)

# EMAs
fig.add_trace(go.Scatter(x=df.index, y=series["ema_short"], name="EMA 9",
    line=dict(color="#ff9800", width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=series["ema_long"], name="EMA 21",
    line=dict(color="#9c27b0", width=1.2)), row=1, col=1)

# MACD
colors = ["#26a69a" if v >= 0 else "#ef5350" for v in series["hist"]]
fig.add_trace(go.Bar(x=df.index, y=series["hist"], name="MACD Hist",
    marker_color=colors, showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=series["macd_line"], name="MACD",
    line=dict(color="#2196f3", width=1.2)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=series["sig_line"], name="Signal",
    line=dict(color="#ff5722", width=1.2)), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=series["rsi"], name="RSI",
    line=dict(color="#00bcd4", width=1.5)), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red",   opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)

fig.update_layout(
    height=640,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
    template="plotly_dark",
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="MACD",  row=2, col=1)
fig.update_yaxes(title_text="RSI",   row=3, col=1, range=[0, 100])

st.plotly_chart(fig, use_container_width=True)

# ── Indicator table ───────────────────────────────────────────────────────

st.subheader("Indicator Summary")

rows = [
    ("RSI (14)",           interp.get("rsi", "")),
    ("MACD",               interp.get("macd", "")),
    ("EMA Crossover 9/21", interp.get("ema_cross", "")),
    ("Bollinger Bands",    interp.get("bb", "")),
    ("Stochastic",         interp.get("stoch", "")),
    ("Volume",             interp.get("volume", "")),
]

ind_df = pd.DataFrame(rows, columns=["Indicator", "Reading"])
st.dataframe(ind_df, use_container_width=True, hide_index=True)

# Composite signal badge
signal = result["signal"]
score  = result["score"]
color  = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(signal, "gray")
st.markdown(
    f"**Technical composite signal:** "
    f"<span style='color:{color}; font-size:1.3em; font-weight:bold;'>{signal}</span>"
    f"  (score {score:+d}/3)",
    unsafe_allow_html=True,
)

st.divider()

# ── AI Analysis ───────────────────────────────────────────────────────────

st.subheader("AI Analysis (Claude)")

run_ai = st.button("Run AI Analysis", type="primary")

if run_ai:
    with st.spinner("Asking Claude for a 1-3 day recommendation…"):
        ai = ai_analysis.analyse(selected, info, result, news)

    action     = ai.get("action", "HOLD")
    confidence = ai.get("confidence", "Low")
    reasoning  = ai.get("reasoning", "")
    risks      = ai.get("risks", "")
    entry_tip  = ai.get("entry_tip", "")
    available  = ai.get("available", False)

    if not available:
        st.warning("AI layer not available — set your Anthropic API key in the sidebar.")

    action_color = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#ff9800"}.get(action, "#aaa")
    st.markdown(
        f"### Recommendation: "
        f"<span style='color:{action_color}; font-size:1.5em; font-weight:bold;'>{action}</span>"
        f"  · Confidence: **{confidence}**",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Reasoning**")
        st.write(reasoning)
        if entry_tip:
            st.markdown("**Entry tip**")
            st.info(entry_tip)
    with col_b:
        st.markdown("**Risks**")
        st.warning(risks)
else:
    st.info("Click **Run AI Analysis** to get Claude's 1-3 day recommendation for this ticker.")

st.divider()

# ── News ──────────────────────────────────────────────────────────────────

st.subheader("Recent News")

if news:
    for item in news:
        st.markdown(
            f"**[{item['title']}]({item['link']})**  \n"
            f"_{item['publisher']}_ · {item['published']}"
        )
        st.divider()
else:
    st.caption("No recent news found for this ticker.")

# ── Watchlist overview ────────────────────────────────────────────────────

st.subheader("Watchlist Quick Scan")
st.caption("Live snapshot of all tickers. Click a ticker above to deep-dive.")

if st.button("Scan all tickers"):
    scan_rows = []
    prog = st.progress(0)
    for i, tk in enumerate(st.session_state.watchlist):
        prog.progress((i + 1) / len(st.session_state.watchlist), text=f"Scanning {tk}…")
        d = data_module.fetch_ohlcv(tk, period="1mo")
        r = ind_module.compute_all(d)
        if r:
            v = r["values"]
            prev = float(d["Close"].iloc[-2]) if len(d) > 1 else v["close"]
            chg  = (v["close"] - prev) / prev * 100
            scan_rows.append({
                "Ticker":  tk,
                "Price":   v["close"],
                "1d %":    f"{chg:+.2f}%",
                "RSI":     v["rsi"],
                "Signal":  r["signal"],
                "Score":   f"{r['score']:+d}",
            })
        else:
            scan_rows.append({"Ticker": tk, "Price": "—", "1d %": "—",
                               "RSI": "—", "Signal": "N/A", "Score": "—"})
    prog.empty()

    def colour_signal(val):
        if val == "BUY":  return "color: #26a69a; font-weight: bold"
        if val == "SELL": return "color: #ef5350; font-weight: bold"
        return "color: #ff9800"

    scan_df = pd.DataFrame(scan_rows)
    st.dataframe(
        scan_df.style.applymap(colour_signal, subset=["Signal"]),
        use_container_width=True,
        hide_index=True,
    )
