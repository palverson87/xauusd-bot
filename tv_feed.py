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
import concurrent.futures

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
                  symbol: str = SYMBOL, exchange: str = EXCHANGE) -> pd.DataFrame:
    """Fetch OHLCV bars from TradingView."""
    tv_interval = _INTERVAL.get(interval)
    if tv_interval is None:
        raise ValueError(f"Unsupported interval '{interval}'. Use: {list(_INTERVAL)}")
    df = _client().get_hist(symbol, exchange, interval=tv_interval, n_bars=count)
    if df is None or df.empty:
        raise ValueError(f"TradingView returned no data for {symbol} {interval}")
    return _normalise(df)


def fetch_eurusd(count: int = 10) -> pd.DataFrame:
    """EUR/USD daily candles — inverse DXY proxy."""
    return fetch_candles("1d", count, symbol=EUR_SYM, exchange=EXCHANGE)


def live_price() -> float:
    """Current XAU/USD price — last close of a 1-minute bar."""
    df = fetch_candles("1m", 2)
    return float(df["Close"].iloc[-1])


def fetch_all_tv(period_interval: str):
    """
    Parallel fetch of all timeframes from TradingView.
    Returns (df_main, df_15m, df_1h, df_4h, df_1d, df_eurusd).
    """
    gran, count = _PERIOD_MAP.get(period_interval, ("1h", 168))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        f_main   = ex.submit(fetch_candles, gran,  count)
        f_15m    = ex.submit(fetch_candles, "15m", 300)
        f_1h     = ex.submit(fetch_candles, "1h",  168)
        f_4h     = ex.submit(fetch_candles, "4h",  360)
        f_1d     = ex.submit(fetch_candles, "1d",  365)
        f_eurusd = ex.submit(fetch_eurusd,  10)

    return (
        f_main.result(),
        f_15m.result(),
        f_1h.result(),
        f_4h.result(),
        f_1d.result(),
        f_eurusd.result(),
    )
