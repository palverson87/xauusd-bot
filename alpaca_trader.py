"""
Alpaca paper trading for XAU/USD bot.
Endpoint: https://paper-api.alpaca.markets
Keys:     ALPACA_API_KEY / ALPACA_SECRET_KEY env vars

Gold ticker detection order: XAUUSD → GLD (whichever is tradable).
Qty mapping: GLD  → 1 share  ≈ 0.1 oz
             XAUUSD → 0.1 qty = 0.1 oz
"""
import logging
import os

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

import db

log = logging.getLogger(__name__)

PAPER_URL       = "https://paper-api.alpaca.markets"
MAX_POSITIONS   = 2
MAX_DAILY_LOSS  = 2.0   # %
STOP_LOSS_PCT   = 0.8   # %
TAKE_PROFIT_PCT = 1.6   # %
SCORE_THRESHOLD = 4     # minimum score to actually execute


class AlpacaTrader:

    def __init__(self):
        self.key    = os.getenv("ALPACA_API_KEY",    "")
        self.secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._api    = None
        self._symbol = None

    # ── Connection ─────────────────────────────────────────────────────────
    @property
    def enabled(self):
        return bool(self.key and self.secret)

    def _get_api(self):
        if self._api is None:
            self._api = tradeapi.REST(
                key_id=self.key,
                secret_key=self.secret,
                base_url=PAPER_URL,
            )
        return self._api

    # ── Gold symbol detection ───────────────────────────────────────────────
    def gold_symbol(self):
        if self._symbol:
            return self._symbol
        api = self._get_api()
        for sym in ("XAUUSD", "GLD"):
            try:
                asset = api.get_asset(sym)
                if asset.tradable:
                    self._symbol = sym
                    log.info("Gold symbol resolved: %s", sym)
                    return sym
            except Exception:
                continue
        self._symbol = "GLD"
        log.warning("Could not verify symbol — defaulting to GLD")
        return "GLD"

    # ── Account / market data ───────────────────────────────────────────────
    def account(self):
        return self._get_api().get_account()

    def positions(self):
        return self._get_api().list_positions()

    def orders(self, status="all", limit=50):
        return self._get_api().list_orders(status=status, limit=limit)

    def get_order(self, order_id):
        return self._get_api().get_order(order_id)

    def portfolio_history(self, period="1M", timeframe="1D"):
        return self._get_api().get_portfolio_history(
            period=period, timeframe=timeframe
        )

    # ── Risk guard-rails ────────────────────────────────────────────────────
    def _daily_loss_ok(self):
        try:
            acct = self.account()
            eq, last_eq = float(acct.equity), float(acct.last_equity)
            if last_eq == 0:
                return True
            pnl_pct = (eq - last_eq) / last_eq * 100
            if pnl_pct <= -MAX_DAILY_LOSS:
                log.warning("Daily P&L %.2f%% — trading halted for today", pnl_pct)
                return False
        except Exception as exc:
            log.error("Daily loss check error: %s", exc)
        return True

    def _positions_ok(self):
        try:
            if len(self.positions()) >= MAX_POSITIONS:
                log.info("Max open positions (%d) reached", MAX_POSITIONS)
                return False
        except Exception as exc:
            log.error("Position count error: %s", exc)
        return True

    # ── Entry point from dashboard callback ────────────────────────────────
    def maybe_trade(self, score, direction, price, session):
        """
        Evaluate all conditions and place an order if everything passes.
        Returns (status_message: str, order_id: str | None).
        Score 3 = log only; score 4-5 = execute.
        """
        if not self.enabled:
            return "Alpaca keys not set — trading disabled", None

        v_direction = direction.replace("STRONG ", "")

        if v_direction not in ("BUY", "SELL"):
            return "Neutral — no trade", None

        if score < 3:
            return f"Score {score}/5 — below threshold", None

        if score == 3:
            log.info("Score 3/5: signal logged, not executed")
            return "Score 3/5: signal noted — below execution threshold", None

        # score >= 4 from here — execute
        if session == "Asian":
            return "Skipped: Asian session (low liquidity)", None

        if not self._daily_loss_ok():
            return "Halted: daily loss limit (2%) reached", None

        if not self._positions_ok():
            return f"Skipped: {MAX_POSITIONS} open positions already", None

        return self._submit(v_direction, price, score, session)

    # ── Order submission ────────────────────────────────────────────────────
    def _submit(self, direction, price, score, session):
        symbol = self.gold_symbol()
        side   = "buy" if direction == "BUY" else "sell"

        if direction == "BUY":
            sl = round(price * (1 - STOP_LOSS_PCT   / 100), 2)
            tp = round(price * (1 + TAKE_PROFIT_PCT / 100), 2)
        else:
            sl = round(price * (1 + STOP_LOSS_PCT   / 100), 2)
            tp = round(price * (1 - TAKE_PROFIT_PCT / 100), 2)

        qty = "1"   if symbol == "GLD"    else "0.1"
        tif = "day" if symbol == "GLD"    else "gtc"

        try:
            order = self._get_api().submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force=tif,
                order_class="bracket",
                stop_loss={"stop_price": str(sl)},
                take_profit={"limit_price": str(tp)},
            )
            db.log_trade(
                alpaca_order_id=order.id,
                symbol=symbol,
                direction=direction,
                qty=float(qty),
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                score=score,
                session=session,
            )
            msg = (f"✓ {side.upper()} {qty} {symbol} | "
                   f"SL ${sl:,.2f}  TP ${tp:,.2f}")
            log.info(msg)
            return msg, order.id

        except APIError as exc:
            msg = f"Order failed: {exc}"
            log.error(msg)
            return msg, None
        except Exception as exc:
            msg = f"Unexpected error: {exc}"
            log.error(msg)
            return msg, None

    # ── Outcome sync (called by tracker) ───────────────────────────────────
    def sync_open_trades(self):
        """Pull Alpaca order status for all open DB trades and update outcomes."""
        open_trades = db.get_open_trades()
        if not open_trades:
            return
        for trade in open_trades:
            try:
                order = self.get_order(trade["alpaca_order_id"])
                status = order.status  # filled | canceled | expired | held …
                if status in ("filled", "canceled", "expired", "done_for_day"):
                    ep = trade["entry_price"]
                    filled = float(order.filled_avg_price or ep)
                    if status == "filled":
                        pnl = (filled - ep) if trade["direction"] == "BUY" else (ep - filled)
                        pnl_pct = pnl / ep * 100
                        outcome = "win" if pnl > 0 else "loss"
                    else:
                        pnl_pct = 0.0
                        outcome = "cancelled"
                    db.update_trade_outcome(
                        trade_id=trade["id"],
                        exit_price=filled,
                        pnl_pct=pnl_pct,
                        outcome=outcome,
                        status=status,
                    )
            except Exception as exc:
                log.error("Trade sync %s: %s", trade["alpaca_order_id"], exc)


# Module-level singleton — imported by app.py and tracker.py
trader = AlpacaTrader()
