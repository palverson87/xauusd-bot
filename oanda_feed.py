"""
Oanda v20 REST API data feed for XAU_USD.
Returns DataFrames in the same OHLCV format as yfinance so the rest of the
bot works without modification.

Env vars required:
  OANDA_API_KEY     — your Oanda API token
  OANDA_ACCOUNT_ID  — your Oanda account ID  (e.g. 101-001-12345678-001)
  OANDA_ENV         — "practice" (default) or "live"

Granularity map:
  15m → M15   1h → H1   4h → H4   1d → D
"""
import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

OANDA_API_KEY    = os.getenv("OANDA_API_KEY",    "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV        = os.getenv("OANDA_ENV",        "practice")
INSTRUMENT       = "XAU_USD"

_GRAN = {
    "15m": "M15",
    "1h":  "H1",
    "4h":  "H4",
    "1d":  "D",
}

# Period string → (granularity, candle count)
_PERIOD_MAP = {
    "5d|1h":   ("1h",  120),
    "1mo|1h":  ("1h",  720),
    "3mo|1d":  ("1d",   92),
    "6mo|1d":  ("1d",  183),
    "1y|1d":   ("1d",  365),
}


def enabled() -> bool:
    return bool(OANDA_API_KEY and OANDA_ACCOUNT_ID)


def _client():
    try:
        from oandapyV20 import API
        return API(access_token=OANDA_API_KEY, environment=OANDA_ENV)
    except ImportError:
        raise RuntimeError("oandapyV20 not installed — run: pip install oandapyV20")


def fetch_candles(interval: str, count: int) -> pd.DataFrame:
    """
    Fetch up to `count` completed midpoint candles from Oanda.
    `interval` must be one of: 15m, 1h, 4h, 1d
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    from oandapyV20.endpoints.instruments import InstrumentsCandles

    gran = _GRAN.get(interval)
    if gran is None:
        raise ValueError(f"Unsupported interval '{interval}'. Use: {list(_GRAN)}")

    client = _client()
    params = {"granularity": gran, "count": count, "price": "M"}
    r = InstrumentsCandles(INSTRUMENT, params=params)
    client.request(r)

    rows = []
    for c in r.response.get("candles", []):
        if not c.get("complete", False):
            continue
        mid = c["mid"]
        rows.append({
            "time":   pd.Timestamp(c["time"]),
            "Open":   float(mid["o"]),
            "High":   float(mid["h"]),
            "Low":    float(mid["l"]),
            "Close":  float(mid["c"]),
            "Volume": int(c.get("volume", 0)),
        })

    if not rows:
        raise ValueError(f"Oanda returned no candles for {INSTRUMENT} {gran}")

    df = pd.DataFrame(rows).set_index("time")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    log.debug("Oanda: fetched %d %s candles for %s", len(df), gran, INSTRUMENT)
    return df


def live_price() -> float:
    """Return the current mid price for XAU_USD."""
    from oandapyV20.endpoints.pricing import PricingInfo

    client = _client()
    r = PricingInfo(OANDA_ACCOUNT_ID, params={"instruments": INSTRUMENT})
    client.request(r)
    price_data = r.response["prices"][0]
    bid = float(price_data["bids"][0]["price"])
    ask = float(price_data["asks"][0]["price"])
    return round((bid + ask) / 2, 5)


def fetch_all_oanda(period_interval: str):
    """
    Fetch all required DataFrames from Oanda.
    Returns (df_main, df_15m, df_1h, df_4h, df_1d) — no df_uup (still from yfinance).
    """
    import concurrent.futures

    gran, count = _PERIOD_MAP.get(period_interval, ("1h", 168))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        f_main = ex.submit(fetch_candles, gran,  count)
        f_15m  = ex.submit(fetch_candles, "15m", 300)
        f_1h   = ex.submit(fetch_candles, "1h",  168)
        f_4h   = ex.submit(fetch_candles, "4h",  360)
        f_1d   = ex.submit(fetch_candles, "1d",  365)

    return (
        f_main.result(),
        f_15m.result(),
        f_1h.result(),
        f_4h.result(),
        f_1d.result(),
    )
