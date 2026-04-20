import concurrent.futures
import logging
import os
from datetime import datetime, timezone

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import db
import tracker
import oanda_feed
from alpaca_trader import trader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")

REFRESH_MS = 60 * 1000          # full chart + confluence refresh (60 s)
TICKER_MS  = 5  * 1000          # live price ticker (5 s)

PERIOD_OPTIONS = [
    {"label": "5 Days (1h)",   "value": "5d|1h"},
    {"label": "1 Month (1h)",  "value": "1mo|1h"},
    {"label": "3 Months (1d)", "value": "3mo|1d"},
    {"label": "6 Months (1d)", "value": "6mo|1d"},
    {"label": "1 Year (1d)",   "value": "1y|1d"},
]

SESSION_ZONES = [
    (0,    7,   "#263238", "Asian",      "Low volatility — reduce size"),
    (7,    9,   "#1b5e20", "London KZ",  "Kill zone — highest probability"),
    (9,    12,  "#0277bd", "London",     "Active session"),
    (12,   13.5,"#0d47a1", "Overlap",    "London/NY transition"),
    (13.5, 16,  "#2e7d32", "NY Open KZ", "Kill zone — high probability"),
    (16,   21,  "#1565c0", "NY",         "Active session"),
]

# ── Theme ──────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
DIM    = "#8b949e"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
BLUE   = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"
TEAL   = "#39d353"


def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════


def calculate_indicators(df):
    c = df["Close"]
    # RSI (Wilder)
    delta = c.diff()
    avg_g = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_l = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    # Stochastic RSI (14, 14, 3, 3)
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    stoch   = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)
    df["StochK"] = stoch.rolling(3).mean() * 100
    df["StochD"] = df["StochK"].rolling(3).mean()
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    # Bollinger Bands
    df["BB_mid"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    # EMA ribbon (scalp structure)
    df["EMA8"]   = c.ewm(span=8,   adjust=False).mean()
    df["EMA21"]  = c.ewm(span=21,  adjust=False).mean()
    df["EMA55"]  = c.ewm(span=55,  adjust=False).mean()
    # Legacy MAs (kept for chart + golden/death cross)
    df["SMA9"]   = c.rolling(9).mean()
    df["SMA50"]  = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["EMA20"]  = c.ewm(span=20, adjust=False).mean()
    # ATR (14) — for dynamic SL/TP
    hl  = df["High"] - df["Low"]
    hpc = abs(df["High"] - df["Close"].shift(1))
    lpc = abs(df["Low"]  - df["Close"].shift(1))
    df["ATR14"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1) \
                    .ewm(span=14, min_periods=14, adjust=False).mean()
    return df


def fetch_all(period, interval):
    """Fetch all OHLCV data from Oanda — XAU_USD + EUR_USD (DXY proxy)."""
    raw_main, raw_15m, raw_1h, raw_4h, raw_1d, df_eurusd = \
        oanda_feed.fetch_all_oanda(f"{period}|{interval}")
    return (
        calculate_indicators(raw_main),
        calculate_indicators(raw_15m),
        calculate_indicators(raw_1h),
        calculate_indicators(raw_4h),
        calculate_indicators(raw_1d),
        df_eurusd,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def _rsi_sig(row):
    v = row["RSI"]
    if pd.isna(v):  return "N/A",  DIM,    "–"
    if v < 35:      return "BULL", GREEN,  f"{v:.1f}"
    if v > 65:      return "BEAR", RED,    f"{v:.1f}"
    return                  "NEUT", YELLOW, f"{v:.1f}"


def _macd_sig(row):
    m, s = row["MACD"], row["MACD_signal"]
    if pd.isna(m) or pd.isna(s): return "N/A",  DIM,   "–"
    d = m - s
    return ("BULL", GREEN, f"{d:+.2f}") if d > 0 else ("BEAR", RED, f"{d:+.2f}")


def _ma_sig(row):
    c, s50, e20 = row["Close"], row["SMA50"], row["EMA20"]
    ref = s50 if not pd.isna(s50) else e20
    if pd.isna(ref): return "N/A",  DIM,    "–"
    if c > ref and c > e20: return "BULL", GREEN,  "Above"
    if c < ref and c < e20: return "BEAR", RED,    "Below"
    return                          "NEUT", YELLOW, "Mixed"


def mtf_signals(df_15m, df_1h, df_4h, df_1d):
    return {
        tf: (_rsi_sig(df.iloc[-1]), _macd_sig(df.iloc[-1]), _ma_sig(df.iloc[-1]))
        for tf, df in [("15M", df_15m), ("1H", df_1h), ("4H", df_4h), ("1D", df_1d)]
    }


def ma_crossover_signal(df):
    if len(df) < 2: return "N/A", DIM
    r, r1 = df.iloc[-1], df.iloc[-2]
    s, e, s1, e1 = r["SMA9"], r["EMA20"], r1["SMA9"], r1["EMA20"]
    if any(pd.isna(v) for v in [s, e, s1, e1]): return "N/A", DIM
    if s > e and s1 <= e1: return "BULL CROSS ▲", GREEN
    if s < e and s1 >= e1: return "BEAR CROSS ▼", RED
    return ("BULL", GREEN) if s > e else ("BEAR", RED)


def _stochrsi_sig(row):
    k = row.get("StochK") if isinstance(row, dict) else row["StochK"]
    if pd.isna(k): return "N/A", DIM, "–"
    if k < 20:     return "BULL", GREEN,  f"K={k:.1f} oversold"
    if k > 80:     return "BEAR", RED,    f"K={k:.1f} overbought"
    return                 "NEUT", YELLOW, f"K={k:.1f}"


def _ema_ribbon_sig(row):
    e8, e21, e55 = row["EMA8"], row["EMA21"], row["EMA55"]
    if any(pd.isna(v) for v in [e8, e21, e55]): return "N/A", DIM, "–"
    if e8 > e21 > e55: return "BULL", GREEN,  "Aligned ↑"
    if e8 < e21 < e55: return "BEAR", RED,    "Aligned ↓"
    return                     "NEUT", YELLOW, "Mixed"


def _macd_hist_sig(df):
    if len(df) < 2: return "N/A", DIM, "–"
    h, h1 = df["MACD_hist"].iloc[-1], df["MACD_hist"].iloc[-2]
    if pd.isna(h): return "N/A", DIM, "–"
    if h > 0 and h > h1: return "BULL", GREEN,  f"{h:+.2f} rising"
    if h < 0 and h < h1: return "BEAR", RED,    f"{h:+.2f} falling"
    return                       "NEUT", YELLOW, f"{h:+.2f} flat"


def _bb_pos_sig(row):
    c, upper, lower = row["Close"], row["BB_upper"], row["BB_lower"]
    if any(pd.isna(v) for v in [upper, lower]): return "N/A", DIM, "–"
    rng = upper - lower
    if rng == 0: return "NEUT", YELLOW, "–"
    pos = (c - lower) / rng
    if pos < 0.25: return "BULL", GREEN,  f"Near lower {pos:.0%}"
    if pos > 0.75: return "BEAR", RED,    f"Near upper {pos:.0%}"
    return                 "NEUT", YELLOW, f"Mid {pos:.0%}"


def _kill_zone_sig():
    _, name, _, col, active = get_session()
    if name in ("London KZ", "NY Open KZ"):
        return "ACTIVE",   GREEN,  name
    if name == "Asian":
        return "AVOID",    RED,    "Asian session"
    if active:
        return "OPEN",     YELLOW, name
    return     "INACTIVE", DIM,    "Off hours"


def dxy_signal(df_eurusd):
    """EUR/USD is the primary DXY component (57.6% weight) and its inverse.
    EUR/USD rising → DXY falling → bullish for gold.
    EUR/USD falling → DXY rising → bearish for gold.
    """
    if len(df_eurusd) < 6: return "N/A", DIM, 0.0
    ret5 = (df_eurusd["Close"].iloc[-1] / df_eurusd["Close"].iloc[-6] - 1) * 100
    if ret5 >  0.15: return "BULLISH FOR GOLD", GREEN,  ret5
    if ret5 < -0.15: return "BEARISH FOR GOLD", RED,    ret5
    return                   "NEUTRAL",          YELLOW, ret5


def get_session():
    now = datetime.now(timezone.utc)
    h   = now.hour + now.minute / 60.0
    for start, end, color, name, desc in SESSION_ZONES:
        if start <= h < end:
            return now, name, desc, color, True
    return now, "Off Hours", "No major session active", DIM, False


def liquidity_levels(df_1d):
    if len(df_1d) < 2: return None, None
    prev = df_1d.iloc[-2]
    return float(prev["High"]), float(prev["Low"])


def compute_confluence(df_15m, df_1h, df_4h, df_eurusd, weights=None):
    """
    7-signal ICT-style confluence:
      1. StochRSI 15M     — entry timing (oversold/overbought)
      2. EMA Ribbon 4H    — trend structure (8/21/55 alignment)
      3. MACD Hist 1H     — momentum (rising/falling histogram)
      4. BB Position 1H   — mean reversion zone (near band edge)
      5. EUR/USD (DXY)    — macro correlation via Oanda
      6. Kill Zone        — London KZ / NY Open KZ = active
      7. HTF Bias 4H      — price above/below EMA55 on 4H
    Score normalised to 0–5.
    """
    if weights is None:
        weights = db.load_weights()

    results = []

    # 1. StochRSI 15M
    lbl, col, sub = _stochrsi_sig(df_15m.iloc[-1])
    vote = 1 if lbl == "BULL" else (-1 if lbl == "BEAR" else 0)
    results.append(("StochRSI 15M", lbl, col, vote, weights.get("StochRSI 15M", 1.0)))

    # 2. EMA Ribbon 4H
    lbl, col, sub = _ema_ribbon_sig(df_4h.iloc[-1])
    vote = 1 if lbl == "BULL" else (-1 if lbl == "BEAR" else 0)
    results.append(("EMA Ribbon 4H", lbl, col, vote, weights.get("EMA Ribbon 4H", 1.0)))

    # 3. MACD Histogram 1H
    lbl, col, sub = _macd_hist_sig(df_1h)
    vote = 1 if lbl == "BULL" else (-1 if lbl == "BEAR" else 0)
    results.append(("MACD Hist 1H", lbl, col, vote, weights.get("MACD Hist 1H", 1.0)))

    # 4. BB Position 1H
    lbl, col, sub = _bb_pos_sig(df_1h.iloc[-1])
    vote = 1 if lbl == "BULL" else (-1 if lbl == "BEAR" else 0)
    results.append(("BB Position 1H", lbl, col, vote, weights.get("BB Position 1H", 1.0)))

    # 5. EUR/USD macro (DXY inverse proxy via Oanda)
    lbl, col, _ = dxy_signal(df_eurusd)
    vote = 1 if "BULL" in lbl else (-1 if "BEAR" in lbl else 0)
    results.append(("EUR/USD DXY", lbl, col, vote, weights.get("EUR/USD DXY", 1.0)))

    # 6. Kill Zone session
    lbl, col, sub = _kill_zone_sig()
    if lbl == "ACTIVE":
        vote = 1
    elif lbl == "AVOID":
        vote = -1
    else:
        vote = 0
    results.append(("Kill Zone", lbl, col, vote, weights.get("Kill Zone", 1.0)))

    # 7. HTF Bias — price vs EMA55 on 4H
    r4 = df_4h.iloc[-1]
    e55, price_4h = r4["EMA55"], r4["Close"]
    if not pd.isna(e55):
        htf_vote = 1 if price_4h > e55 else -1
        htf_lbl  = "BULL" if htf_vote == 1 else "BEAR"
        htf_col  = GREEN  if htf_vote == 1 else RED
    else:
        htf_vote, htf_lbl, htf_col = 0, "N/A", DIM
    results.append(("HTF Bias 4H", htf_lbl, htf_col, htf_vote,
                     weights.get("HTF Bias 4H", 1.0)))

    # Weighted scoring
    bull_w  = sum(w for *_, v, w in results if v == 1)
    bear_w  = sum(w for *_, v, w in results if v == -1)
    total_w = sum(w for *_, v, w in results) or 1.0
    direction = "BULL" if bull_w >= bear_w else "BEAR"
    score = min(5, max(0, round(max(bull_w, bear_w) / total_w * 5)))
    ratio = max(bull_w, bear_w) / total_w

    if ratio >= 0.75 and score >= 4:
        verdict = ("STRONG BUY"  if direction == "BULL" else "STRONG SELL",
                   GREEN         if direction == "BULL" else RED)
    elif ratio >= 0.58:
        verdict = ("BUY"  if direction == "BULL" else "SELL",
                   GREEN  if direction == "BULL" else RED)
    else:
        verdict = ("NEUTRAL", YELLOW)

    display = [(n, l, c, v) for n, l, c, v, _ in results]
    return score, direction, verdict, display


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS TAB — UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def _panel(children, extra=None):
    s = {"background": PANEL, "border": f"1px solid {BORDER}",
         "borderRadius": "8px", "padding": "14px 18px"}
    if extra: s.update(extra)
    return html.Div(children, style=s)


def _label(t):
    return html.P(t, style={"margin": "0 0 8px", "color": DIM, "fontSize": "10px",
                              "letterSpacing": "0.1em", "fontFamily": "monospace",
                              "textTransform": "uppercase"})


def build_session_panel():
    now, name, desc, col, _ = get_session()
    h_frac = (now.hour + now.minute / 60.0) / 24.0

    segs = []
    for s, e, c, *_ in SESSION_ZONES:
        segs.append(html.Div(style={
            "position": "absolute", "left": f"{s/24*100:.1f}%",
            "width": f"{(e-s)/24*100:.1f}%", "height": "100%",
            "backgroundColor": c, "opacity": "0.85",
        }))
    segs.append(html.Div(style={
        "position": "absolute", "left": f"{h_frac*100:.1f}%",
        "width": "2px", "height": "130%", "top": "-15%",
        "backgroundColor": "white", "zIndex": "10", "borderRadius": "2px",
    }))
    timeline = html.Div([
        html.Div(segs, style={"position": "relative", "height": "12px",
                               "borderRadius": "4px", "overflow": "visible",
                               "backgroundColor": BORDER}),
        html.Div([
            html.Span(h, style={"position": "absolute",
                                 "left": f"{int(h)*100//24}%",
                                 "fontSize": "9px", "color": DIM,
                                 "fontFamily": "monospace"})
            for h in ["0", "6", "12", "18", "24"]
        ], style={"position": "relative", "height": "14px", "marginTop": "3px"}),
    ])
    legend = html.Div([
        item
        for _, _, c, n, _ in SESSION_ZONES
        for item in [
            html.Span("▌", style={"color": c, "fontSize": "14px", "marginRight": "3px"}),
            html.Span(n, style={"color": DIM, "fontSize": "10px",
                                 "fontFamily": "monospace", "marginRight": "10px"}),
        ]
    ], style={"marginTop": "8px"})

    return _panel([
        _label("Session Clock"),
        html.Div([
            html.Span(now.strftime("%H:%M") + " UTC",
                      style={"color": TEXT, "fontSize": "20px", "fontWeight": "700",
                             "fontFamily": "monospace"}),
            html.Span(f"  {name}", style={"color": col, "fontSize": "14px",
                                           "fontWeight": "600", "fontFamily": "monospace"}),
        ], style={"marginBottom": "4px"}),
        html.P(desc, style={"margin": "0 0 10px", "color": DIM,
                              "fontSize": "11px", "fontFamily": "monospace"}),
        timeline, legend,
    ], {"flex": "2"})


def build_dxy_panel(df_eurusd):
    lbl, col, ret5 = dxy_signal(df_eurusd)
    current = df_eurusd["Close"].iloc[-1] if len(df_eurusd) else 0.0
    arrow   = "↑" if ret5 > 0 else "↓"
    pos_pct = (max(-1.0, min(1.0, ret5)) + 1.0) / 2.0 * 100

    meter = html.Div([
        html.Div(style={"position": "absolute", "inset": "0",
                         "background": f"linear-gradient(to right,{RED}44,{BORDER} 50%,{GREEN}44)",
                         "borderRadius": "4px"}),
        html.Div(style={"position": "absolute", "left": "50%", "width": "1px",
                         "height": "100%", "backgroundColor": BORDER}),
        html.Div(style={"position": "absolute", "left": f"{pos_pct:.1f}%",
                         "width": "3px", "height": "100%", "backgroundColor": col,
                         "borderRadius": "2px", "transform": "translateX(-50%)"}),
    ], style={"position": "relative", "height": "10px",
               "borderRadius": "4px", "margin": "8px 0"})

    return _panel([
        _label("DXY Correlation  (EUR/USD inverse proxy  ·  Oanda)"),
        html.Div([
            html.Span(f"EUR/USD  {current:.5f}", style={"color": TEXT, "fontSize": "18px",
                                                          "fontWeight": "700",
                                                          "fontFamily": "monospace"}),
            html.Span(f"  {arrow} {ret5:+.3f}%  (5d)", style={"color": col,
                                                                 "fontSize": "13px",
                                                                 "fontFamily": "monospace"}),
        ], style={"marginBottom": "4px"}),
        meter,
        html.Div([
            html.Span("Gold ↓", style={"color": RED,   "fontSize": "10px",
                                        "fontFamily": "monospace"}),
            html.Span("Gold ↑", style={"color": GREEN, "fontSize": "10px",
                                        "fontFamily": "monospace"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                   "marginBottom": "8px"}),
        html.P(f"EUR/USD {arrow}  →  {lbl}",
               style={"margin": 0, "color": col, "fontSize": "13px",
                      "fontWeight": "600", "fontFamily": "monospace"}),
        html.P("Rising EUR/USD = falling DXY = bullish for gold",
               style={"margin": "3px 0 0", "color": DIM, "fontSize": "10px",
                      "fontFamily": "monospace"}),
    ], {"flex": "1"})


def build_confluence_panel(score, direction, verdict, results):
    v_text, v_col = verdict
    w = db.load_weights()

    dots = [
        html.Span("●", style={"color": v_col if i < score else BORDER,
                                "fontSize": "22px", "margin": "0 2px"})
        for i in range(5)
    ]
    rows = []
    for name, lbl, col, vote in results:
        wt = w.get(name, 1.0)
        wt_str = f"×{wt:.2f}" if wt != 1.0 else ""
        rows.append(html.Div([
            html.Span(name, style={"color": DIM, "fontSize": "11px",
                                    "fontFamily": "monospace", "display": "inline-block",
                                    "width": "95px"}),
            html.Span(lbl, style={"color": col, "fontSize": "11px", "fontWeight": "600",
                                   "fontFamily": "monospace", "background": f"{col}22",
                                   "borderRadius": "4px", "padding": "1px 7px"}),
            html.Span(" ▲" if vote == 1 else (" ▼" if vote == -1 else " –"),
                      style={"color": col, "fontSize": "10px", "marginLeft": "4px",
                             "fontFamily": "monospace"}),
            html.Span(f"  {wt_str}", style={"color": BLUE if wt > 1 else (RED if wt < 1 else DIM),
                                             "fontSize": "10px", "fontFamily": "monospace"}),
        ], style={"marginBottom": "5px"}))

    return _panel([
        _label("Confluence Score"),
        html.Div([
            html.Span(f"{score}/5", style={"color": v_col, "fontSize": "34px",
                                            "fontWeight": "700", "fontFamily": "monospace",
                                            "marginRight": "10px", "lineHeight": "1"}),
            html.Div(dots, style={"display": "flex", "alignItems": "center"}),
        ], style={"display": "flex", "alignItems": "center", "margin": "6px 0 10px"}),
        html.Div(v_text, style={
            "display": "inline-block", "color": v_col, "fontSize": "15px",
            "fontWeight": "700", "fontFamily": "monospace",
            "border": f"1px solid {v_col}", "borderRadius": "6px",
            "padding": "3px 14px", "background": f"{v_col}18", "marginBottom": "10px",
        }),
        html.Div(rows),
    ], {"flex": "1", "border": f"2px solid {v_col}"})


def build_mtf_panel(sigs, df_4h):
    mac_lbl, mac_col = ma_crossover_signal(df_4h)

    th = lambda t: html.Th(t, style={"color": DIM, "fontFamily": "monospace",
                                      "fontSize": "11px", "padding": "6px 14px",
                                      "textAlign": "center", "fontWeight": "400",
                                      "borderBottom": f"1px solid {BORDER}"})
    td_label = lambda t: html.Td(t, style={"color": DIM, "fontFamily": "monospace",
                                            "fontSize": "11px", "padding": "7px 14px",
                                            "borderRight": f"1px solid {BORDER}"})
    def td_sig(lbl, col, sub=""):
        return html.Td([
            html.Div(lbl, style={"color": col, "fontWeight": "600", "fontSize": "12px",
                                  "fontFamily": "monospace", "background": f"{col}22",
                                  "borderRadius": "4px", "padding": "2px 8px",
                                  "textAlign": "center"}),
            html.Div(sub, style={"color": DIM, "fontSize": "10px", "textAlign": "center",
                                  "fontFamily": "monospace", "marginTop": "2px"}),
        ], style={"padding": "6px 14px"})
    td_dash = lambda: html.Td("–", style={"color": BORDER, "textAlign": "center",
                                           "fontFamily": "monospace", "padding": "7px 14px"})

    rows = [
        html.Tr([td_label(ind)] + [td_sig(*sigs[tf][i]) for tf in ["15M", "1H", "4H", "1D"]])
        for i, ind in enumerate(["RSI", "MACD", "MA"])
    ]
    rows.append(html.Tr([td_label("SMA9/EMA20"), td_dash(),
                          td_dash(), td_sig(mac_lbl, mac_col), td_dash()]))

    return _panel([
        _label("Multi-Timeframe Analysis"),
        html.Table([
            html.Thead(html.Tr([th("Indicator"), th("15M"), th("1H"), th("4H"), th("1D")])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ])


def build_summary(df_main, df_1d):
    r  = df_main.iloc[-1]
    r1 = df_main.iloc[-2] if len(df_main) > 1 else r
    price, change = r["Close"], r["Close"] - r1["Close"]
    pct  = change / r1["Close"] * 100 if r1["Close"] else 0
    sign = "+" if change >= 0 else ""
    pdh, pdl = liquidity_levels(df_1d)

    def card(title, value, color=TEXT, sub=None):
        return html.Div([
            html.P(title, style={"margin": 0, "color": DIM, "fontSize": "10px",
                                  "textTransform": "uppercase", "letterSpacing": "0.08em",
                                  "fontFamily": "monospace"}),
            html.P(value, style={"margin": "4px 0 0", "color": color, "fontSize": "18px",
                                  "fontWeight": "700", "fontFamily": "monospace"}),
            *([] if not sub else [html.P(sub, style={"margin": "2px 0 0", "color": DIM,
                                                       "fontSize": "10px",
                                                       "fontFamily": "monospace"})]),
        ], style={"background": PANEL, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "12px 16px",
                   "flex": "1", "minWidth": "130px"})

    bb   = ("Oversold"   if r["Close"] < r["BB_lower"] else
            "Overbought" if r["Close"] > r["BB_upper"] else "Inside bands")
    s50, s200 = r["SMA50"], r["SMA200"]
    if not pd.isna(s50) and not pd.isna(s200):
        cross = "Golden Cross" if s50 > s200 else "Death Cross"
        cc    = GREEN if s50 > s200 else RED
        csub  = f"SMA50 {s50:,.1f}  /  SMA200 {s200:,.1f}"
    else:
        cross, cc, csub = "N/A", DIM, "Insufficient history"

    return html.Div([
        html.Div([
            card("XAU/USD",   f"${price:,.2f}", TEXT,
                 sub=f"{sign}{change:,.2f}  ({sign}{pct:.2f}%)"),
            card("RSI (14)",  f"{r['RSI']:.1f}",
                 GREEN if r["RSI"] < 35 else RED if r["RSI"] > 65 else TEXT,
                 sub="<35 Oversold  /  >65 Overbought"),
            card("MACD",      "Bullish" if r["MACD"] > r["MACD_signal"] else "Bearish",
                 GREEN if r["MACD"] > r["MACD_signal"] else RED,
                 sub=f"{r['MACD'] - r['MACD_signal']:+.2f}  hist"),
            card("Bollinger", bb,
                 GREEN if "Oversold" in bb else RED if "Overbought" in bb else TEXT,
                 sub=f"U {r['BB_upper']:,.1f}  /  L {r['BB_lower']:,.1f}"),
            card("SMA Cross", cross, cc, sub=csub),
            card("EMA 20",    f"${r['EMA20']:,.2f}",
                 GREEN if r["Close"] > r["EMA20"] else RED,
                 sub="Price above" if r["Close"] > r["EMA20"] else "Price below"),
            card("Prev Day High", f"${pdh:,.2f}" if pdh else "N/A", YELLOW,
                 sub="Liquidity zone"),
            card("Prev Day Low",  f"${pdl:,.2f}" if pdl else "N/A", ORANGE,
                 sub="Liquidity zone"),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "10px",
                   "marginBottom": "8px"}),
        html.P(f"Last bar: {df_main.index[-1].strftime('%Y-%m-%d  %H:%M')}  ·  "
               "Auto-refresh every 5 min",
               style={"color": DIM, "fontSize": "10px", "margin": 0,
                      "fontFamily": "monospace"}),
    ])


def build_figure(df, df_1d):
    pdh, pdl = liquidity_levels(df_1d)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.52, 0.16, 0.16, 0.16],
                        subplot_titles=("XAU/USD  (GC=F)", "Volume", "RSI (14)", "MACD (12/26/9)"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED, line_width=1,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
        line=dict(color=PURPLE, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
        line=dict(color=PURPLE, width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(188,140,255,0.07)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"],  name="BB Mid",
        line=dict(color=PURPLE, width=1)), row=1, col=1)
    # EMA ribbon (scalp structure)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA8"],   name="EMA 8",
        line=dict(color=TEAL,   width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"],  name="EMA 21",
        line=dict(color=BLUE,   width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA55"],  name="EMA 55",
        line=dict(color=ORANGE, width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200",
        line=dict(color=RED,    width=1.5, dash="dash")), row=1, col=1)

    # EMA 8/21 crossover markers
    e8  = df["EMA8"].ffill()
    e21 = df["EMA21"]
    if not e8.isna().all():
        up   = (e8 > e21) & (e8.shift(1) <= e21.shift(1))
        down = (e8 < e21) & (e8.shift(1) >= e21.shift(1))
        if up.any():
            fig.add_trace(go.Scatter(x=df.index[up], y=df["Low"][up] * 0.9975,
                mode="markers", name="Bull Cross",
                marker=dict(symbol="triangle-up", size=13, color=GREEN,
                            line=dict(color=BG, width=1))), row=1, col=1)
        if down.any():
            fig.add_trace(go.Scatter(x=df.index[down], y=df["High"][down] * 1.0025,
                mode="markers", name="Bear Cross",
                marker=dict(symbol="triangle-down", size=13, color=RED,
                            line=dict(color=BG, width=1))), row=1, col=1)

    if pdh:
        fig.add_hline(y=pdh, line_color="#ffeb3b", line_dash="dash", line_width=1.5,
                      annotation_text=f"  PDH  ${pdh:,.1f}", row=1, col=1,
                      annotation_font_color="#ffeb3b", annotation_position="right")
    if pdl:
        fig.add_hline(y=pdl, line_color=ORANGE, line_dash="dash", line_width=1.5,
                      annotation_text=f"  PDL  ${pdl:,.1f}", row=1, col=1,
                      annotation_font_color=ORANGE, annotation_position="right")

    vol_col = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_col,
                          showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color=BLUE, width=1.5), showlegend=False), row=3, col=1)
    for lvl, c, d in [(70, RED, "dot"), (30, GREEN, "dot"), (50, DIM, "dash")]:
        fig.add_hline(y=lvl, line_color=c, line_dash=d, line_width=1, row=3, col=1)

    hcol = [GREEN if v >= 0 else RED for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hcol,
                          showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color=BLUE, width=1.5), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"],
        line=dict(color=ORANGE, width=1.5), showlegend=False), row=4, col=1)

    fig.update_layout(
        height=800, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace", size=11),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=80, t=40, b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    ax = dict(gridcolor=BORDER, zeroline=False,
              showspikes=True, spikecolor=DIM, spikedash="dot", spikethickness=1)
    for i in range(1, 5):
        fig.update_xaxes(ax, row=i, col=1)
        fig.update_yaxes(ax, row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    for ann in fig.layout.annotations:
        ann.font.color = DIM
        ann.font.size  = 11
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE TAB
# ══════════════════════════════════════════════════════════════════════════════

def _perf_chart(title, x, y, colors, text_vals=None, kind="bar"):
    fig = go.Figure()
    if kind == "bar":
        fig.add_trace(go.Bar(x=x, y=y, marker_color=colors,
                              text=text_vals, textposition="outside",
                              textfont=dict(color=DIM, size=10)))
        fig.add_hline(y=50, line_color=DIM, line_dash="dash", line_width=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=colors, width=2),
                                  fill="tozeroy", fillcolor=_hex_rgba(colors, 0.13)))
        fig.add_hline(y=10000, line_color=DIM, line_dash="dash", line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color=DIM, size=11,
                                          family="monospace"), x=0.02),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=DIM, family="monospace", size=10),
        margin=dict(l=40, r=20, t=36, b=30),
        showlegend=False, height=240,
        xaxis=dict(gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor=BORDER, zeroline=False),
    )
    return fig


def build_performance_tab():
    stats   = db.get_performance_stats()
    weights = db.load_weights()
    recent  = db.get_recent_signals(20)

    def win_pct(r):
        return r["wins"] / r["total"] * 100 if r.get("total") else 0

    def bar_colors(vals):
        return [GREEN if v >= 55 else RED if v < 45 else YELLOW for v in vals]

    # ── Session win-rate ───────────────────────────────────────────────────
    sess = stats["by_session"]
    sess_x = [r["session"] for r in sess]
    sess_y = [win_pct(r) for r in sess]
    sess_t = [f"n={r['total']}" for r in sess]
    fig_sess = _perf_chart("Win Rate by Session (%)", sess_x, sess_y,
                            bar_colors(sess_y), sess_t) if sess else None

    # ── Score win-rate ─────────────────────────────────────────────────────
    sc = stats["by_score"]
    sc_x = [str(r["score"]) for r in sc]
    sc_y = [win_pct(r) for r in sc]
    sc_t = [f"n={r['total']}" for r in sc]
    fig_sc = _perf_chart("Win Rate by Confluence Score (%)", sc_x, sc_y,
                          bar_colors(sc_y), sc_t) if sc else None

    # ── Indicator accuracy ─────────────────────────────────────────────────
    ind = stats["indicator_acc"]
    ind_x = list(ind.keys())
    ind_y = [d["pct"] or 0 for d in ind.values()]
    ind_t = [f"n={d['total']}" for d in ind.values()]
    fig_ind = _perf_chart("Indicator Accuracy (%)", ind_x, ind_y,
                           bar_colors(ind_y), ind_t)

    # ── Equity curve ───────────────────────────────────────────────────────
    eq_data = stats["equity"]
    if eq_data:
        eq = 10_000.0
        curve = [eq]
        for row in eq_data:
            eq *= 1 + (row["pnl_pct"] or 0) / 100
            curve.append(eq)
        final_ret = (curve[-1] / 10_000 - 1) * 100
        eq_col = GREEN if curve[-1] >= 10_000 else RED
        fig_eq = _perf_chart(f"Equity Curve  ({final_ret:+.1f}%)",
                              list(range(len(curve))), curve,
                              eq_col, kind="line")
    else:
        fig_eq = None

    # ── Adaptive weights bar ───────────────────────────────────────────────
    w_fig = go.Figure(go.Bar(
        x=list(weights.values()), y=list(weights.keys()),
        orientation="h",
        marker_color=[GREEN if v >= 1.0 else RED for v in weights.values()],
        text=[f"{v:.2f}" for v in weights.values()],
        textposition="outside",
    ))
    w_fig.add_vline(x=1.0, line_color=DIM, line_dash="dash", line_width=1)
    w_fig.update_layout(
        title=dict(text="Adaptive Indicator Weights", font=dict(color=DIM, size=11,
                                                                  family="monospace"), x=0.02),
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=DIM, family="monospace", size=10),
        margin=dict(l=10, r=60, t=36, b=30),
        xaxis=dict(range=[0, 2.4], gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor=BORDER, zeroline=False),
        showlegend=False, height=240,
    )

    def empty_chart(title):
        fig = go.Figure()
        fig.add_annotation(text="No data yet — signals will appear after first BUY/SELL",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=DIM, size=11))
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PANEL, height=240,
                          title=dict(text=title, font=dict(color=DIM, size=11,
                                                            family="monospace"), x=0.02),
                          margin=dict(l=40, r=20, t=36, b=30))
        return fig

    # ── Recent signals table ───────────────────────────────────────────────
    def outcome_cell(pnl, won):
        if won is None: return html.Td("–", style={"color": DIM, "textAlign": "center",
                                                     "fontFamily": "monospace", "fontSize": "11px",
                                                     "padding": "6px 10px"})
        c = GREEN if won else RED
        return html.Td(f"{'✓' if won else '✗'}  {pnl:+.2f}%",
                       style={"color": c, "textAlign": "center",
                               "fontFamily": "monospace", "fontSize": "11px",
                               "padding": "6px 10px"})

    header_style = {"color": DIM, "fontFamily": "monospace", "fontSize": "10px",
                     "padding": "6px 10px", "borderBottom": f"1px solid {BORDER}",
                     "textAlign": "center", "fontWeight": "400",
                     "textTransform": "uppercase", "letterSpacing": "0.06em"}
    sig_rows = []
    for row in recent:
        dir_col = GREEN if row["direction"] == "BUY" else RED
        sig_rows.append(html.Tr([
            html.Td(row["ts"][:16], style={"color": DIM, "fontFamily": "monospace",
                                            "fontSize": "11px", "padding": "6px 10px"}),
            html.Td(row["direction"],
                    style={"color": dir_col, "fontWeight": "700",
                           "fontFamily": "monospace", "fontSize": "11px",
                           "textAlign": "center", "padding": "6px 10px"}),
            html.Td(f"${row['price']:,.2f}", style={"color": TEXT, "fontFamily": "monospace",
                                                     "fontSize": "11px", "padding": "6px 10px",
                                                     "textAlign": "right"}),
            html.Td(f"{row['score']}/5", style={"color": BLUE, "textAlign": "center",
                                                  "fontFamily": "monospace", "fontSize": "11px",
                                                  "padding": "6px 10px"}),
            html.Td(row["session"], style={"color": DIM, "fontFamily": "monospace",
                                            "fontSize": "11px", "padding": "6px 10px"}),
            outcome_cell(row["pnl_1h"],  row["won_1h"]),
            outcome_cell(row["pnl_4h"],  row["won_4h"]),
            outcome_cell(row["pnl_24h"], row["won_24h"]),
        ], style={"borderBottom": f"1px solid {BORDER}22"}))

    table = _panel([
        _label("Signal History (last 20)"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Time",     style=header_style),
                html.Th("Signal",   style=header_style),
                html.Th("Price",    style=header_style),
                html.Th("Score",    style=header_style),
                html.Th("Session",  style=header_style),
                html.Th("+1h",      style=header_style),
                html.Th("+4h",      style=header_style),
                html.Th("+24h",     style=header_style),
            ])),
            html.Tbody(sig_rows if sig_rows else [
                html.Tr(html.Td("No signals logged yet", colSpan=8,
                                style={"color": DIM, "padding": "16px",
                                       "textAlign": "center",
                                       "fontFamily": "monospace", "fontSize": "12px"}))
            ]),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ])

    charts_row1 = html.Div([
        dcc.Graph(figure=fig_sess or empty_chart("Win Rate by Session (%)"),
                  config={"displayModeBar": False}, style={"flex": "1"}),
        dcc.Graph(figure=fig_sc or empty_chart("Win Rate by Confluence Score (%)"),
                  config={"displayModeBar": False}, style={"flex": "1"}),
        dcc.Graph(figure=fig_ind, config={"displayModeBar": False}, style={"flex": "1"}),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    charts_row2 = html.Div([
        dcc.Graph(figure=fig_eq or empty_chart("Equity Curve"),
                  config={"displayModeBar": False}, style={"flex": "2"}),
        dcc.Graph(figure=w_fig, config={"displayModeBar": False}, style={"flex": "1"}),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"})

    return html.Div([charts_row1, charts_row2, table])


# ══════════════════════════════════════════════════════════════════════════════
#  PAPER TRADES TAB
# ══════════════════════════════════════════════════════════════════════════════

def build_paper_trades_tab():
    if not trader.enabled:
        return _panel([
            html.P("Alpaca paper trading is not configured.",
                   style={"color": DIM, "fontFamily": "monospace", "fontSize": "13px",
                          "margin": "0 0 8px"}),
            html.P("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
                   "and restart the bot.",
                   style={"color": DIM, "fontFamily": "monospace", "fontSize": "12px",
                          "margin": 0}),
        ])

    # ── Fetch live data from Alpaca ────────────────────────────────────────
    try:
        acct       = trader.account()
        positions  = trader.positions()
        ph         = trader.portfolio_history(period="1M", timeframe="1D")
        gold_sym   = trader.gold_symbol()
    except Exception as exc:
        return _panel([html.P(f"Alpaca API error: {exc}",
                               style={"color": RED, "fontFamily": "monospace",
                                      "fontSize": "13px"})])

    equity       = float(acct.equity)
    cash         = float(acct.cash)
    buying_power = float(acct.buying_power)
    last_eq      = float(acct.last_equity)
    daily_pnl    = equity - last_eq
    daily_pnl_pct = daily_pnl / last_eq * 100 if last_eq else 0
    dp_col        = GREEN if daily_pnl >= 0 else RED

    # ── Trade history from DB ──────────────────────────────────────────────
    trades     = db.get_trades(50)
    trade_stats = db.get_trade_stats()

    # Compute stats
    wins   = next((r["n"]        for r in trade_stats if r["outcome"] == "win"),  0)
    losses = next((r["n"]        for r in trade_stats if r["outcome"] == "loss"), 0)
    total_closed = wins + losses
    win_rate = wins / total_closed * 100 if total_closed else None
    total_return = (equity / 100_000 - 1) * 100  # assume $100k paper start

    # Avg R:R  (avg_win / avg_loss)
    avg_win  = next((abs(r["avg_pnl"]) for r in trade_stats if r["outcome"] == "win"),  None)
    avg_loss = next((abs(r["avg_pnl"]) for r in trade_stats if r["outcome"] == "loss"), None)
    rr_ratio = avg_win / avg_loss if (avg_win and avg_loss) else None

    def stat_card(title, value, color=TEXT, sub=None):
        return html.Div([
            html.P(title, style={"margin": 0, "color": DIM, "fontSize": "10px",
                                  "textTransform": "uppercase", "letterSpacing": "0.08em",
                                  "fontFamily": "monospace"}),
            html.P(value, style={"margin": "4px 0 0", "color": color, "fontSize": "18px",
                                  "fontWeight": "700", "fontFamily": "monospace"}),
            *([] if not sub else [html.P(sub, style={"margin": "2px 0 0", "color": DIM,
                                                       "fontSize": "10px",
                                                       "fontFamily": "monospace"})]),
        ], style={"background": PANEL, "border": f"1px solid {BORDER}",
                   "borderRadius": "8px", "padding": "12px 16px",
                   "flex": "1", "minWidth": "130px"})

    stats_row = html.Div([
        stat_card("Equity",       f"${equity:,.2f}", TEXT),
        stat_card("Cash",         f"${cash:,.2f}",   TEXT),
        stat_card("Buying Power", f"${buying_power:,.2f}", TEXT),
        stat_card("Daily P&L",    f"{'+'if daily_pnl>=0 else ''}{daily_pnl:,.2f}",
                  dp_col, sub=f"{'+'if daily_pnl_pct>=0 else ''}{daily_pnl_pct:.2f}%"),
        stat_card("Total Return", f"{'+'if total_return>=0 else ''}{total_return:.2f}%",
                  GREEN if total_return >= 0 else RED),
        stat_card("Win Rate",
                  f"{win_rate:.1f}%" if win_rate is not None else "–",
                  GREEN if (win_rate or 0) >= 50 else RED,
                  sub=f"{wins}W / {losses}L  ({total_closed} closed)"),
        stat_card("Avg R:R",
                  f"1 : {rr_ratio:.2f}" if rr_ratio else "–",
                  GREEN if (rr_ratio or 0) >= 1.5 else YELLOW,
                  sub="Target: 1 : 2.0"),
        stat_card("Gold Symbol", gold_sym, BLUE,
                  sub="Paper trading"),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "10px",
               "marginBottom": "12px"})

    # ── Equity curve ───────────────────────────────────────────────────────
    eq_fig = go.Figure()
    if ph and ph.equity:
        from datetime import datetime as _dt
        dates = [_dt.utcfromtimestamp(t).strftime("%Y-%m-%d") for t in ph.timestamp]
        eq_col = GREEN if ph.equity[-1] >= ph.equity[0] else RED
        eq_fig.add_trace(go.Scatter(
            x=dates, y=ph.equity, name="Equity",
            line=dict(color=eq_col, width=2),
            fill="tozeroy", fillcolor=_hex_rgba(eq_col, 0.13),
        ))
        eq_fig.add_hline(y=ph.base_value, line_color=DIM,
                          line_dash="dash", line_width=1)
    eq_fig.update_layout(
        title=dict(text="Paper Account Equity", font=dict(color=DIM, size=11,
                                                            family="monospace"), x=0.01),
        paper_bgcolor=BG, plot_bgcolor=PANEL, height=220,
        font=dict(color=DIM, family="monospace", size=10),
        margin=dict(l=50, r=20, t=36, b=30), showlegend=False,
        xaxis=dict(gridcolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor=BORDER, zeroline=False, tickprefix="$"),
    )

    # ── Open positions ─────────────────────────────────────────────────────
    pos_th = lambda t: html.Th(t, style={"color": DIM, "fontFamily": "monospace",
                                          "fontSize": "10px", "padding": "6px 12px",
                                          "textTransform": "uppercase",
                                          "letterSpacing": "0.06em",
                                          "borderBottom": f"1px solid {BORDER}",
                                          "textAlign": "right" if t != "Symbol" else "left"})
    def pos_td(val, color=TEXT, align="right"):
        return html.Td(val, style={"color": color, "fontFamily": "monospace",
                                    "fontSize": "12px", "padding": "7px 12px",
                                    "textAlign": align})

    pos_rows = []
    for p in positions:
        pnl     = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        pc      = GREEN if pnl >= 0 else RED
        pos_rows.append(html.Tr([
            pos_td(p.symbol, TEXT, "left"),
            pos_td(p.side.upper(), GREEN if p.side == "long" else RED),
            pos_td(p.qty),
            pos_td(f"${float(p.avg_entry_price):,.2f}"),
            pos_td(f"${float(p.current_price):,.2f}"),
            pos_td(f"{'+'if pnl>=0 else ''}{pnl:,.2f}", pc),
            pos_td(f"{'+'if pnl_pct>=0 else ''}{pnl_pct:.2f}%", pc),
        ]))

    open_positions = _panel([
        _label("Open Positions"),
        html.Table([
            html.Thead(html.Tr([pos_th(h) for h in
                                ["Symbol", "Side", "Qty", "Entry", "Current",
                                 "Unreal. P&L", "P&L %"]])),
            html.Tbody(pos_rows if pos_rows else [
                html.Tr(html.Td("No open positions", colSpan=7,
                                style={"color": DIM, "padding": "14px",
                                       "textAlign": "center",
                                       "fontFamily": "monospace", "fontSize": "12px"}))
            ]),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ], {"marginBottom": "12px"})

    # ── Trade history ──────────────────────────────────────────────────────
    th_style = {"color": DIM, "fontFamily": "monospace", "fontSize": "10px",
                "padding": "6px 12px", "textTransform": "uppercase",
                "letterSpacing": "0.06em",
                "borderBottom": f"1px solid {BORDER}", "textAlign": "center"}

    hist_rows = []
    for t in trades:
        oc = t["outcome"] or "open"
        oc_col = (GREEN if oc == "win" else RED if oc == "loss"
                  else YELLOW if oc == "open" else DIM)
        dir_col = GREEN if t["direction"] == "BUY" else RED
        pnl_str = (f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "–")
        pnl_col = (GREEN if (t["pnl_pct"] or 0) > 0
                   else RED if (t["pnl_pct"] or 0) < 0 else DIM)
        hist_rows.append(html.Tr([
            html.Td(t["opened_at"][:16] if t["opened_at"] else "–",
                    style={"color": DIM, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px"}),
            html.Td(t["symbol"],
                    style={"color": TEXT, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px", "textAlign": "center"}),
            html.Td(t["direction"],
                    style={"color": dir_col, "fontFamily": "monospace",
                           "fontSize": "11px", "fontWeight": "700",
                           "padding": "6px 12px", "textAlign": "center"}),
            html.Td(f"${t['entry_price']:,.2f}" if t["entry_price"] else "–",
                    style={"color": TEXT, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px", "textAlign": "right"}),
            html.Td(f"${t['stop_loss']:,.2f}",
                    style={"color": RED, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px", "textAlign": "right"}),
            html.Td(f"${t['take_profit']:,.2f}",
                    style={"color": GREEN, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px", "textAlign": "right"}),
            html.Td(f"{t['score']}/5",
                    style={"color": BLUE, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px", "textAlign": "center"}),
            html.Td(t["session"],
                    style={"color": DIM, "fontFamily": "monospace", "fontSize": "11px",
                           "padding": "6px 12px"}),
            html.Td(pnl_str, style={"color": pnl_col, "fontFamily": "monospace",
                                     "fontSize": "11px", "fontWeight": "600",
                                     "padding": "6px 12px", "textAlign": "right"}),
            html.Td(oc.upper(), style={"color": oc_col, "fontFamily": "monospace",
                                        "fontSize": "11px", "fontWeight": "600",
                                        "padding": "6px 12px", "textAlign": "center"}),
        ], style={"borderBottom": f"1px solid {BORDER}22"}))

    trade_history = _panel([
        _label("Trade History (last 50)"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Opened",     style=th_style),
                html.Th("Symbol",     style=th_style),
                html.Th("Direction",  style=th_style),
                html.Th("Entry",      style=th_style),
                html.Th("Stop Loss",  style=th_style),
                html.Th("Take Profit",style=th_style),
                html.Th("Score",      style=th_style),
                html.Th("Session",    style=th_style),
                html.Th("P&L",        style=th_style),
                html.Th("Outcome",    style=th_style),
            ])),
            html.Tbody(hist_rows if hist_rows else [
                html.Tr(html.Td("No trades placed yet — score 5/5 required to execute",
                                colSpan=10,
                                style={"color": DIM, "padding": "16px",
                                       "textAlign": "center",
                                       "fontFamily": "monospace", "fontSize": "12px"}))
            ]),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ])

    return html.Div([
        stats_row,
        dcc.Graph(figure=eq_fig, config={"displayModeBar": False},
                  style={"marginBottom": "12px"}),
        open_positions,
        trade_history,
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE PRICE BAR
# ══════════════════════════════════════════════════════════════════════════════

def build_live_price_bar(price=None, bid=None, ask=None, prev_close=None, source="yfinance"):
    if price is None:
        return html.Div(style={"height": "44px"})

    change     = price - prev_close if prev_close else 0.0
    change_pct = change / prev_close * 100 if prev_close else 0.0
    up         = change >= 0
    col        = GREEN if up else RED
    arrow      = "▲" if up else "▼"
    spread     = f"  ·  Bid {bid:,.2f}  /  Ask {ask:,.2f}" if bid and ask else ""
    src_dot    = html.Span("● LIVE", style={
        "color": GREEN, "fontSize": "10px", "fontFamily": "monospace",
        "marginLeft": "12px", "letterSpacing": "0.08em",
    }) if source == "oanda" else html.Span(
        "yfinance", style={"color": DIM, "fontSize": "10px", "fontFamily": "monospace",
                           "marginLeft": "12px"})

    return html.Div([
        html.Span("XAU/USD", style={"color": DIM, "fontFamily": "monospace",
                                     "fontSize": "12px", "marginRight": "10px"}),
        html.Span(f"${price:,.2f}", style={"color": TEXT, "fontFamily": "monospace",
                                            "fontSize": "26px", "fontWeight": "700",
                                            "marginRight": "8px"}),
        html.Span(f"{arrow} {change:+,.2f}  ({change_pct:+.2f}%)",
                  style={"color": col, "fontFamily": "monospace", "fontSize": "14px",
                         "fontWeight": "600", "marginRight": "8px"}),
        html.Span(spread, style={"color": DIM, "fontFamily": "monospace", "fontSize": "12px"}),
        src_dot,
    ], style={
        "background": PANEL, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {col}",
        "borderRadius": "8px", "padding": "8px 20px",
        "display": "flex", "alignItems": "center",
        "marginBottom": "12px",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS TAB
# ══════════════════════════════════════════════════════════════════════════════

def _setting_row(label, hint, control):
    return html.Div([
        html.Div([
            html.P(label, style={"margin": 0, "color": TEXT, "fontFamily": "monospace",
                                  "fontSize": "13px", "fontWeight": "600"}),
            html.P(hint,  style={"margin": "2px 0 0", "color": DIM, "fontFamily": "monospace",
                                  "fontSize": "10px"}),
        ], style={"flex": "1"}),
        html.Div(control, style={"flex": "1", "display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "alignItems": "center", "padding": "10px 0",
               "borderBottom": f"1px solid {BORDER}"})


def build_settings_tab():
    s = db.load_settings()

    def toggle(id_, val):
        return dcc.RadioItems(
            id=id_, value="on" if val else "off",
            options=[{"label": "  ON",  "value": "on"},
                     {"label": "  OFF", "value": "off"}],
            inline=True,
            inputStyle={"marginRight": "4px"},
            labelStyle={"marginRight": "16px", "color": TEXT,
                        "fontFamily": "monospace", "fontSize": "13px"},
        )

    def slider(id_, val, mn, mx, step, marks=None):
        return dcc.Slider(
            id=id_, value=val, min=mn, max=mx, step=step,
            marks=marks or {i: {"label": str(i), "style": {"color": DIM, "fontSize": "10px"}}
                            for i in [mn, (mn+mx)//2, mx]},
            tooltip={"placement": "bottom", "always_visible": True},
        )

    rows = [
        _setting_row("Trading Enabled",
                     "Master switch — disables all order submission when OFF",
                     toggle("set-trading-enabled", s["trading_enabled"])),
        _setting_row("Score Threshold",
                     "Minimum confluence score (0–5) to execute a trade",
                     slider("set-score-threshold", s["score_threshold"], 1, 5, 1,
                            {i: {"label": str(i), "style": {"color": DIM, "fontSize": "10px"}}
                             for i in range(1, 6)})),
        _setting_row("Max Open Positions",
                     "Maximum concurrent trades at any time",
                     slider("set-max-positions", s["max_positions"], 1, 6, 1,
                            {i: {"label": str(i), "style": {"color": DIM, "fontSize": "10px"}}
                             for i in range(1, 7)})),
        _setting_row("Daily Loss Limit (%)",
                     "Halt trading when day's P&L drops below this % of equity",
                     slider("set-max-daily-loss", s["max_daily_loss"], 0.5, 10.0, 0.5,
                            {v: {"label": f"{v}%", "style": {"color": DIM, "fontSize": "10px"}}
                             for v in [0.5, 2.0, 5.0, 10.0]})),
        _setting_row("ATR Stop-Loss Multiplier",
                     "SL = this × ATR14 from current price",
                     slider("set-atr-sl-mult", s["atr_sl_mult"], 0.5, 4.0, 0.25,
                            {v: {"label": str(v), "style": {"color": DIM, "fontSize": "10px"}}
                             for v in [0.5, 1.5, 2.5, 4.0]})),
        _setting_row("ATR Take-Profit Multiplier",
                     "TP = this × ATR14  (keep ≥ 3× SL mult for positive R:R)",
                     slider("set-atr-tp-mult", s["atr_tp_mult"], 1.5, 12.0, 0.5,
                            {v: {"label": str(v), "style": {"color": DIM, "fontSize": "10px"}}
                             for v in [1.5, 4.5, 8.0, 12.0]})),
        _setting_row("Block Asian Session",
                     "Skip trade execution during Asian hours (00:00–07:00 UTC)",
                     toggle("set-block-asian", s["block_asian"])),
    ]

    save_btn = html.Button("Save Settings", id="btn-save-settings", n_clicks=0, style={
        "backgroundColor": BLUE, "color": BG, "border": "none",
        "borderRadius": "6px", "padding": "10px 28px",
        "fontFamily": "monospace", "fontSize": "13px", "fontWeight": "700",
        "cursor": "pointer", "marginTop": "16px",
    })
    save_msg = html.Div(id="settings-save-msg", style={"marginTop": "10px",
                                                         "fontFamily": "monospace",
                                                         "fontSize": "12px"})

    return html.Div([
        _panel([
            _label("Strategy Settings"),
            html.Div(rows),
            save_btn,
            save_msg,
        ]),
    ], style={"maxWidth": "860px"})


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════

TAB_STYLE = {
    "backgroundColor": PANEL, "color": DIM,
    "border": f"1px solid {BORDER}", "borderRadius": "6px 6px 0 0",
    "padding": "8px 20px", "fontFamily": "monospace", "fontSize": "13px",
}
TAB_SELECTED = {**TAB_STYLE, "color": TEXT, "borderBottom": f"1px solid {BG}",
                "backgroundColor": BG, "fontWeight": "600"}

app = dash.Dash(__name__, title="XAU/USD Pro Dashboard")

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("XAU/USD  ·  Gold Technical Analysis",
                    style={"margin": 0, "fontSize": "18px", "fontWeight": "700",
                           "color": TEXT, "fontFamily": "monospace"}),
            html.P(f"XAU_USD  ·  {'Oanda live' if oanda_feed.enabled() else 'yfinance'}  ·  Multi-timeframe  ·  Adaptive learning",
                   style={"margin": "4px 0 0", "color": DIM, "fontSize": "11px",
                          "fontFamily": "monospace"}),
        ]),
        html.Div([
            html.Label("Chart Timeframe", style={"color": DIM, "fontSize": "11px",
                                                  "fontFamily": "monospace",
                                                  "marginRight": "8px"}),
            dcc.Dropdown(id="period-select", options=PERIOD_OPTIONS, value="1mo|1h",
                         clearable=False,
                         style={"width": "200px", "backgroundColor": PANEL, "color": TEXT,
                                "border": f"1px solid {BORDER}",
                                "fontFamily": "monospace", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
               "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "8px",
               "padding": "14px 20px", "marginBottom": "12px"}),

    # Live price bar (updates every 5 s)
    html.Div(id="live-price-bar"),

    # Tabs
    dcc.Tabs(id="tabs", value="analysis", children=[
        dcc.Tab(label="Analysis",     value="analysis",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="Performance",  value="performance",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="Paper Trades", value="paper_trades",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
        dcc.Tab(label="Settings",     value="settings",
                style=TAB_STYLE, selected_style=TAB_SELECTED),
    ], style={"marginBottom": "0"}),

    html.Div(id="tab-content"),

    html.Div(id="error-banner"),
    dcc.Interval(id="timer",  interval=REFRESH_MS, n_intervals=0),
    dcc.Interval(id="ticker", interval=TICKER_MS,  n_intervals=0),

], style={"backgroundColor": BG, "minHeight": "100vh",
           "padding": "16px 20px", "boxSizing": "border-box"})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("live-price-bar", "children"),
    Input("ticker", "n_intervals"),
)
def update_live_price(_n):
    try:
        from oandapyV20.endpoints.pricing import PricingInfo
        from oandapyV20 import API as _OA
        client = _OA(access_token=oanda_feed.OANDA_API_KEY,
                     environment=oanda_feed.OANDA_ENV)
        r = PricingInfo(oanda_feed.OANDA_ACCOUNT_ID,
                        params={"instruments": oanda_feed.INSTRUMENT})
        client.request(r)
        p   = r.response["prices"][0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        mid = (bid + ask) / 2
        try:
            prev_df    = oanda_feed.fetch_candles("1d", 2)
            prev_close = float(prev_df["Close"].iloc[-2]) if len(prev_df) >= 2 else None
        except Exception:
            prev_close = None
        return build_live_price_bar(mid, bid, ask, prev_close, source="oanda")
    except Exception as exc:
        log.warning("Live price update failed: %s", exc)
        return build_live_price_bar()


@app.callback(
    Output("settings-save-msg", "children"),
    Output("settings-save-msg", "style"),
    Input("btn-save-settings", "n_clicks"),
    State("set-trading-enabled",  "value"),
    State("set-score-threshold",  "value"),
    State("set-max-positions",    "value"),
    State("set-max-daily-loss",   "value"),
    State("set-atr-sl-mult",      "value"),
    State("set-atr-tp-mult",      "value"),
    State("set-block-asian",      "value"),
    prevent_initial_call=True,
)
def save_settings_cb(n_clicks, trading_enabled, score_threshold,
                     max_positions, max_daily_loss, atr_sl_mult, atr_tp_mult, block_asian):
    base_style = {"marginTop": "10px", "fontFamily": "monospace", "fontSize": "12px"}
    try:
        db.save_settings({
            "trading_enabled": trading_enabled == "on",
            "score_threshold": int(score_threshold),
            "max_positions":   int(max_positions),
            "max_daily_loss":  float(max_daily_loss),
            "atr_sl_mult":     float(atr_sl_mult),
            "atr_tp_mult":     float(atr_tp_mult),
            "atr_sl_min":      db.load_settings().get("atr_sl_min", 0.3),
            "atr_sl_max":      db.load_settings().get("atr_sl_max", 1.5),
            "block_asian":     block_asian == "on",
        })
        return "✓ Settings saved — applied on next trade check", {**base_style, "color": GREEN}
    except Exception as exc:
        return f"✗ Save failed: {exc}", {**base_style, "color": RED}


@app.callback(
    Output("tab-content",  "children"),
    Output("error-banner", "children"),
    Input("tabs",          "value"),
    Input("timer",         "n_intervals"),
    Input("period-select", "value"),
)
def render_tab(tab, _n, period_value):
    if tab == "performance":
        try:
            return build_performance_tab(), ""
        except Exception as exc:
            return "", _err(exc)

    if tab == "paper_trades":
        try:
            return build_paper_trades_tab(), ""
        except Exception as exc:
            return "", _err(exc)

    if tab == "settings":
        try:
            return build_settings_tab(), ""
        except Exception as exc:
            return "", _err(exc)

    # ── Analysis tab ──────────────────────────────────────────────────────
    period, interval = period_value.split("|")
    try:
        df_main, df_15m, df_1h, df_4h, df_1d, df_uup = fetch_all(period, interval)
        if df_main.empty or len(df_main) < 5:
            raise ValueError("No data returned — market may be closed.")

        score, direction, verdict, conf_results = compute_confluence(df_15m, df_1h, df_4h, df_uup)
        sigs = mtf_signals(df_15m, df_1h, df_4h, df_1d)

        # ── Log signal to DB ───────────────────────────────────────────────
        try:
            r        = df_main.iloc[-1]
            _, sess_name, _, _, _ = get_session()
            votes    = {name: vote for name, _, _, vote in conf_results}
            db.log_signal(
                ts=df_main.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                price=float(r["Close"]),
                direction=verdict[0].replace("STRONG ", ""),
                score=score,
                session=sess_name,
                rsi=float(r["RSI"]) if not pd.isna(r["RSI"]) else None,
                votes=votes,
                interval=interval,
            )
        except Exception as log_exc:
            app.logger.warning("Signal log failed: %s", log_exc)

        # ── Auto-trade via Alpaca (score 5 only, non-Asian sessions) ──────
        try:
            _r = df_main.iloc[-1]
            trade_msg, order_id = trader.maybe_trade(
                score=score,
                direction=verdict[0],
                price=float(_r["Close"]),
                session=sess_name,
                atr=float(_r["ATR14"]) if not pd.isna(_r["ATR14"]) else None,
            )
            if order_id:
                app.logger.info("Trade executed: %s  order=%s", trade_msg, order_id)
            elif "keys not set" not in trade_msg:
                app.logger.info("Trade check: %s", trade_msg)
        except Exception as trade_exc:
            app.logger.warning("Trade execution failed: %s", trade_exc)

        content = html.Div([
            html.Div([build_session_panel(), build_dxy_panel(df_uup),
                      build_confluence_panel(score, direction, verdict, conf_results)],
                     style={"display": "flex", "gap": "12px",
                             "marginBottom": "12px", "flexWrap": "wrap"}),
            html.Div(build_mtf_panel(sigs, df_4h), style={"marginBottom": "12px"}),
            html.Div(build_summary(df_main, df_1d), style={"marginBottom": "12px"}),
            dcc.Graph(id="chart", figure=build_figure(df_main, df_1d),
                      config={"displayModeBar": True, "scrollZoom": True}),
        ])
        return content, ""
    except Exception as exc:
        return "", _err(exc)


def _err(exc):
    return html.Div(f"⚠  {exc}", style={
        "background": "#2d1b1b", "border": f"1px solid {RED}",
        "borderRadius": "6px", "padding": "10px 16px",
        "color": RED, "fontFamily": "monospace",
        "fontSize": "13px", "marginTop": "10px",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  TRADINGVIEW WEBHOOK
# ══════════════════════════════════════════════════════════════════════════════
#
#  TradingView alert → Webhook URL → POST /webhook
#
#  Expected JSON body (paste into TradingView alert "Message" field):
#  {
#    "secret":    "YOUR_WEBHOOK_SECRET",
#    "direction": "BUY",           ← or "SELL"
#    "price":     {{close}},
#    "score":     4,               ← optional, defaults to 4
#    "interval":  "{{interval}}"   ← optional, for logging
#  }
#
#  Set WEBHOOK_SECRET in ecosystem.config.js (same place as Alpaca keys).

import json as _json
from flask import request as _req, jsonify as _jsonify

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")


@app.server.route("/webhook", methods=["POST"])
def tradingview_webhook():
    try:
        data = _req.get_json(force=True, silent=True) or {}
    except Exception:
        return _jsonify({"error": "invalid JSON"}), 400

    # Validate secret
    if WEBHOOK_SECRET and data.get("secret") != WEBHOOK_SECRET:
        log.warning("Webhook: rejected — bad secret from %s", _req.remote_addr)
        return _jsonify({"error": "unauthorized"}), 403

    direction = str(data.get("direction", "")).upper().strip()
    if direction not in ("BUY", "SELL"):
        return _jsonify({"error": f"invalid direction '{direction}'"}), 400

    try:
        price = float(data["price"])
    except (KeyError, ValueError, TypeError):
        return _jsonify({"error": "missing or invalid price"}), 400

    score    = int(data.get("score", 4))
    interval = str(data.get("interval", "TV"))
    _, sess_name, _, _, _ = get_session()

    log.info("Webhook received: direction=%s  price=%.2f  score=%d  session=%s  interval=%s",
             direction, price, score, sess_name, interval)

    # Log to DB
    try:
        db.log_signal(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            price=price,
            direction=direction,
            score=score,
            session=sess_name,
            rsi=None,
            votes={"source": "TradingView"},
            interval=interval,
        )
    except Exception as exc:
        log.warning("Webhook signal log failed: %s", exc)

    # Execute trade
    trade_msg, order_id = trader.maybe_trade(
        score=score,
        direction=direction,
        price=price,
        session=sess_name,
    )
    log.info("Webhook trade result: %s  order_id=%s", trade_msg, order_id)

    return _jsonify({
        "status":   "ok",
        "trade":    trade_msg,
        "order_id": order_id,
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tracker.start()
    port = int(os.getenv("PORT", 8050))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Starting XAU/USD Pro Dashboard → http://{host}:{port}")
    app.run(debug=False, host=host, port=port)
