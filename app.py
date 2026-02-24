"""
app.py — Streamlit trading watchlist dashboard (table-first redesign).
Run with:  streamlit run app.py
"""

import streamlit as st
import data as data_module
import indicators as ind_module
import ai_analysis
import watchlist as wl_module
import os

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── URL key gate ──────────────────────────────────────────────────────────────

if st.query_params.get("key") != os.environ.get("APP_KEY", ""):
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

_defaults = {
    "watchlist": None,
    "scan_cache": {},       # ticker -> computed row data
    "detail_ticker": None,  # ticker currently shown in AI detail panel
    "ai_detail_cache": {},  # ticker -> ai analysis result
    "suggestions": None,    # list[dict] from AI suggest
    "show_suggestions": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.watchlist is None:
    st.session_state.watchlist = wl_module.load()

# ── Helper functions ──────────────────────────────────────────────────────────

def _stars_n(score: int) -> int:
    """Map composite score (-3..+3) to 1-5 stars."""
    if score <= -2:  return 1
    elif score == -1: return 2
    elif score == 0:  return 3
    elif score == 1:  return 4
    else:             return 5


def _speed_n(result: dict) -> int:
    """
    Compute movement speed indicator 1-3 from indicator strength.
    1 = slow, 2 = medium, 3 = fast.
    """
    vals = result.get("values", {})
    vol_ratio  = vals.get("volume", 1) / max(vals.get("volume_avg", 1), 1)
    rsi_dist   = abs(vals.get("rsi", 50) - 50)   # 0-50
    macd_hist  = abs(vals.get("macd_hist", 0))
    price      = vals.get("close", 1) or 1
    macd_rel   = macd_hist / price * 100

    pts = 0
    if vol_ratio > 1.5:   pts += 2
    elif vol_ratio > 1.0: pts += 1
    if rsi_dist > 25:     pts += 2
    elif rsi_dist > 12:   pts += 1
    if macd_rel > 0.3:    pts += 1

    if pts >= 4: return 3
    if pts >= 2: return 2
    return 1


def _quick_context(result: dict) -> str:
    """Generate a ≤25-word plain-English summary of the technical setup."""
    if not result:
        return "Data unavailable."
    vals   = result.get("values", {})
    interp = result.get("interp", {})
    signal = result.get("signal", "HOLD")
    rsi    = vals.get("rsi", 50)

    parts = []
    if rsi < 35:    parts.append("oversold RSI")
    elif rsi > 65:  parts.append("overbought RSI")

    macd_t = interp.get("macd", "")
    if "Bullish"  in macd_t: parts.append("bullish MACD")
    elif "Bearish" in macd_t: parts.append("bearish MACD")
    elif "Crossing" in macd_t: parts.append("MACD crossover imminent")

    ema_t = interp.get("ema_cross", "")
    if "Bullish" in ema_t:  parts.append("EMA uptrend")
    elif "Bearish" in ema_t: parts.append("EMA downtrend")

    bb_t = interp.get("bb", "")
    if "above upper" in bb_t: parts.append("broke above upper band")
    elif "below lower" in bb_t: parts.append("broke below lower band")

    vol_t = interp.get("volume", "")
    if "High volume" in vol_t:  parts.append("high conviction volume")
    elif "Low volume" in vol_t: parts.append("low conviction volume")

    base = (", ".join(parts[:4]) + ".") if parts else "Neutral setup."
    suffix = f" Signal: {signal}."
    full = base + suffix
    # Trim to ~25 words
    words = full.split()
    if len(words) > 25:
        full = " ".join(words[:25]) + "…"
    return full


_STAR_COLORS = ["#ef5350", "#ff7043", "#ff9800", "#66bb6a", "#26a69a"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📋 Watchlist")

    with st.form("add_ticker", clear_on_submit=True):
        new_ticker = st.text_input("Add ticker", placeholder="e.g. AAPL")
        if st.form_submit_button("➕ Add", use_container_width=True):
            if new_ticker.strip():
                t = new_ticker.strip().upper()
                st.session_state.watchlist = wl_module.add(t, st.session_state.watchlist)
                st.rerun()

    st.divider()

    for ticker in list(st.session_state.watchlist):
        c1, c2 = st.columns([4, 1])
        c1.write(ticker)
        if c2.button("✕", key=f"rm_{ticker}", help=f"Remove {ticker}"):
            st.session_state.watchlist = wl_module.remove(ticker, st.session_state.watchlist)
            st.session_state.scan_cache.pop(ticker, None)
            if st.session_state.detail_ticker == ticker:
                st.session_state.detail_ticker = None
            st.rerun()

    st.divider()

    if st.button("🤖 AI Suggest Values", use_container_width=True, type="primary"):
        with st.spinner("Asking AI for suggestions…"):
            st.session_state.suggestions = ai_analysis.suggest_tickers(
                existing=st.session_state.watchlist
            )
        st.session_state.show_suggestions = True
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📈 Trading Watchlist")

if not st.session_state.watchlist:
    st.info("Add tickers in the sidebar to get started.")
    st.stop()

# Refresh button
_, btn_col = st.columns([8, 1])
with btn_col:
    if st.button("🔄 Refresh"):
        st.session_state.scan_cache = {}
        st.session_state.ai_detail_cache = {}
        st.rerun()

# ── Auto-scan tickers not yet cached ─────────────────────────────────────────

missing = [t for t in st.session_state.watchlist if t not in st.session_state.scan_cache]
if missing:
    prog = st.progress(0, text="Scanning tickers…")
    for i, tk in enumerate(missing):
        prog.progress((i + 1) / len(missing), text=f"Scanning {tk}…")
        df   = data_module.fetch_ohlcv(tk, period="1mo")
        res  = ind_module.compute_all(df)
        inf  = data_module.fetch_info(tk)
        if res:
            sc = res["score"]
            st.session_state.scan_cache[tk] = {
                "name":    inf.get("name", tk),
                "stars_n": _stars_n(sc),
                "speed_n": _speed_n(res),
                "context": _quick_context(res),
                "signal":  res["signal"],
                "result":  res,
                "info":    inf,
            }
        else:
            st.session_state.scan_cache[tk] = {
                "name": inf.get("name", tk), "stars_n": 3, "speed_n": 1,
                "context": "Could not fetch data.", "signal": "N/A",
                "result": {}, "info": inf,
            }
    prog.empty()

# ── Table header ─────────────────────────────────────────────────────────────

h1, h2, h3, h4, h5, h6 = st.columns([1, 2.5, 1.5, 1, 3.5, 1.5])
h1.markdown("**Code**")
h2.markdown("**Company**")
h3.markdown("**Outlook**")
h4.markdown("**Speed**")
h5.markdown("**Context**")
h6.markdown("")
st.divider()

# ── Table rows ────────────────────────────────────────────────────────────────

for tk in st.session_state.watchlist:
    cache = st.session_state.scan_cache.get(tk, {})
    sn    = cache.get("stars_n", 3)
    spd   = cache.get("speed_n", 1)

    stars_html = (
        f"<span style='color:{_STAR_COLORS[sn-1]}; font-size:1.15em'>"
        + "★" * sn + "☆" * (5 - sn)
        + "</span>"
    )
    speed_label = {1: "● slow", 2: "●● mid", 3: "●●● fast"}.get(spd, "—")

    c1, c2, c3, c4, c5, c6 = st.columns([1, 2.5, 1.5, 1, 3.5, 1.5])
    c1.markdown(f"**{tk}**")
    c2.write(cache.get("name", tk))
    c3.markdown(stars_html, unsafe_allow_html=True)
    c4.caption(speed_label)
    c5.caption(cache.get("context", ""))
    if c6.button("🔍 AI Analysis", key=f"ai_{tk}"):
        st.session_state.detail_ticker = tk
        st.rerun()

st.divider()

# ── AI Detail panel ───────────────────────────────────────────────────────────

if st.session_state.detail_ticker:
    tk    = st.session_state.detail_ticker
    cache = st.session_state.scan_cache.get(tk, {})

    with st.container(border=True):
        col_title, col_close = st.columns([9, 1])
        col_title.subheader(f"🔍 AI Analysis — {tk}  ·  {cache.get('name', tk)}")
        if col_close.button("✕ Close", key="close_detail"):
            st.session_state.detail_ticker = None
            st.rerun()

        if tk not in st.session_state.ai_detail_cache:
            with st.spinner(f"Analysing {tk}…"):
                news = data_module.fetch_news(tk)
                ai_result = ai_analysis.analyse(
                    tk, cache.get("info", {}), cache.get("result", {}), news
                )
                st.session_state.ai_detail_cache[tk] = ai_result

        ai = st.session_state.ai_detail_cache[tk]
        action     = ai.get("action", "HOLD")
        confidence = ai.get("confidence", "Low")
        color      = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#ff9800"}.get(action, "#aaa")

        if not ai.get("available", True):
            st.warning("AI unavailable — set OPENAI_API_KEY environment variable.")

        st.markdown(
            f"**Recommendation:** "
            f"<span style='color:{color}; font-size:1.3em; font-weight:bold'>{action}</span>"
            f" · Confidence: **{confidence}**",
            unsafe_allow_html=True,
        )
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Reasoning**")
            st.write(ai.get("reasoning", "—"))
            if ai.get("entry_tip"):
                st.info(ai["entry_tip"])
        with cb:
            st.markdown("**Risks**")
            st.warning(ai.get("risks", "—"))

    st.divider()

# ── AI Suggestions panel ──────────────────────────────────────────────────────

if st.session_state.show_suggestions:
    sugs = st.session_state.suggestions or []

    with st.container(border=True):
        col_hdr, col_cls = st.columns([9, 1])
        col_hdr.subheader("🤖 AI Suggested Values")
        if col_cls.button("✕ Close", key="close_sugs"):
            st.session_state.show_suggestions = False
            st.session_state.suggestions = None
            st.rerun()

        if not sugs:
            st.warning("AI could not generate suggestions — check your OPENAI_API_KEY.")
        else:
            st.caption("Click ➕ to add any ticker to your watchlist.")
            sh1, sh2, sh3, sh4 = st.columns([1, 2, 4.5, 0.7])
            sh1.markdown("**Ticker**"); sh2.markdown("**Name**")
            sh3.markdown("**Why watch it**"); sh4.markdown("")
            for sug in sugs:
                stk    = sug.get("ticker", "")
                sname  = sug.get("name", stk)
                sreason = sug.get("reason", "")
                sc1, sc2, sc3, sc4 = st.columns([1, 2, 4.5, 0.7])
                sc1.write(f"**{stk}**")
                sc2.write(sname)
                sc3.caption(sreason)
                already = stk in st.session_state.watchlist
                if already:
                    sc4.caption("✓ added")
                elif sc4.button("➕", key=f"add_sug_{stk}"):
                    st.session_state.watchlist = wl_module.add(stk, st.session_state.watchlist)
                    st.rerun()
