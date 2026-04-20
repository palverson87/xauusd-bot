"""
TradingView data feed via tvDatafeed.
Replaces the Oanda feed — no API key needed for anonymous access.
Optional: set TV_USERNAME + TV_PASSWORD env vars for a logged-in session
(gives access to more history and private indicators).

Symbol: XAUUSD on OANDA exchange (spot gold, same quotes as Oanda)
DXY proxy: EURUSD on OANDA (rising EUR = falling DXY = bullish gold)
"""
import logging
import os
import time

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

log = logging.getLogger(__name__)

TV_USERNAME = os.getenv("TV_USERNAME", "")
TV_PASSWORD = os.getenv("TV_PASSWORD", "")

SYMBOL   = "XAUUSD"
EXCHANGE = "OANDA"
EUR_SYM  = "EURUSD"

_INTERVAL = {
    "1m":  Interval.in_1_minute,
    "15m": Interval.in_15_minute,
    "1h":  Interval.in_1_hour,
    "4h":  Interval.in_4_hour,
    "1d":  Interval.in_daily,
}

_PERIOD_MAP = {
    "5d|1h":   ("1h",  120),
    "1mo|1h":  ("1h",  720),
    "3mo|1d":  ("1d",   92),
    "6mo|1d":  ("1d",  183),
    "1y|1d":   ("1d",  365),
}

_tv = None


def _client() -> TvDatafeed:
    global _tv
    if _tv is None:
        if TV_USERNAME and TV_PASSWORD:
            _tv = TvDatafeed(username=TV_USERNAME, password=TV_PASSWORD)
            log.info("TradingView: logged in as %s", TV_USERNAME)
        else:
            _tv = TvDatafeed()
            log.info("TradingView: anonymous session (data may be limited)")
    return _tv


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename tvDatafeed columns to match the rest of the bot."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={
        "open": "Open", "high": "High",
        "low":  "Low",  "close": "Close", "volume": "Volume",
    })
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_candles(interval: str, count: int,
                  symbol: str = SYMBOL, exchange: str = EXCHANGE,
                  _retries: int = 3) -> pd.DataFrame:
    """Fetch OHLCV bars from TradingView with retry on rate-limit."""
    global _tv
    tv_interval = _INTERVAL.get(interval)
    if tv_interval is None:
        raise ValueError(f"Unsupported interval '{interval}'. Use: {list(_INTERVAL)}")
    for attempt in range(_retries):
        try:
            df = _client().get_hist(symbol, exchange, interval=tv_interval, n_bars=count)
            if df is None or df.empty:
                raise ValueError(f"TradingView returned no data for {symbol} {interval}")
            return _normalise(df)
        except Exception as exc:
            if "429" in str(exc) or "Too Many" in str(exc):
                wait = 10 * (attempt + 1)
                log.warning("TV rate limit hit — waiting %ds (attempt %d/%d)",
                            wait, attempt + 1, _retries)
                _tv = None   # force new session on retry
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"TV fetch failed after {_retries} attempts ({symbol} {interval})")


def fetch_eurusd(count: int = 10) -> pd.DataFrame:
    """EUR/USD daily candles — inverse DXY proxy."""
    return fetch_candles("1d", count, symbol=EUR_SYM, exchange=EXCHANGE)


def live_price() -> float:
    """Current XAU/USD price — last close of a 1-minute bar."""
    df = fetch_candles("1m", 2)
    return float(df["Close"].iloc[-1])


_FETCH_DELAY = 1.2   # seconds between requests — avoids TradingView 429s


def fetch_all_tv(period_interval: str):
    """
    Sequential fetch of all timeframes from TradingView.
    Sequential (not parallel) to stay within TradingView's rate limits.
    Returns (df_main, df_15m, df_1h, df_4h, df_1d, df_eurusd).
    """
    gran, count = _PERIOD_MAP.get(period_interval, ("1h", 168))
    results = []
    fetches = [
        (gran,  count,  SYMBOL,  EXCHANGE),
        ("15m", 300,    SYMBOL,  EXCHANGE),
        ("1h",  168,    SYMBOL,  EXCHANGE),
        ("4h",  360,    SYMBOL,  EXCHANGE),
        ("1d",  365,    SYMBOL,  EXCHANGE),
        ("1d",  10,     EUR_SYM, EXCHANGE),
    ]
    for i, (iv, n, sym, exch) in enumerate(fetches):
        if i > 0:
            time.sleep(_FETCH_DELAY)
        results.append(fetch_candles(iv, n, symbol=sym, exchange=exch))
    return tuple(results)
