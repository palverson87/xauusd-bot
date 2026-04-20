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
STOP_LOSS_PCT   = 0.5   # % fallback when ATR unavailable
TAKE_PROFIT_PCT = 1.5   # % fallback (3:1 R:R)
ATR_SL_MULT     = 1.5   # SL = 1.5 × ATR
ATR_TP_MULT     = 4.5   # TP = 4.5 × ATR (3:1 R:R)
ATR_SL_MIN      = 0.3   # floor %
ATR_SL_MAX      = 1.5   # cap %
SCORE_THRESHOLD = 3     # minimum score to actually execute


class AlpacaTrader:

    def __init__(self):
        self.key       = os.getenv("ALPACA_API_KEY",    "")
        self.secret    = os.getenv("ALPACA_SECRET_KEY", "")
        self._api      = None
        self._symbol   = None
        self._shortable = None

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
                    self._shortable = getattr(asset, "shortable", False)
                    log.info("Gold symbol resolved: %s  shortable=%s", sym, self._shortable)
                    return sym
            except Exception:
                continue
        self._symbol = "GLD"
        self._shortable = False
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
    def _daily_loss_ok(self, limit_pct=MAX_DAILY_LOSS):
        try:
            acct = self.account()
            eq, last_eq = float(acct.equity), float(acct.last_equity)
            if last_eq == 0:
                return True
            pnl_pct = (eq - last_eq) / last_eq * 100
            if pnl_pct <= -abs(limit_pct):
                log.warning("Daily P&L %.2f%% — trading halted for today", pnl_pct)
                return False
        except Exception as exc:
            log.error("Daily loss check error: %s", exc)
        return True

    def _positions_ok(self, limit=MAX_POSITIONS):
        try:
            if len(self.positions()) >= limit:
                log.info("Max open positions (%d) reached", limit)
                return False
        except Exception as exc:
            log.error("Position count error: %s", exc)
        return True

    # ── Entry point from dashboard callback ────────────────────────────────
    def maybe_trade(self, score, direction, price, session, atr=None):
        """
        Evaluate all conditions and place an order if everything passes.
        Returns (status_message: str, order_id: str | None).
        Score 2 = log only; score 3-5 = execute.
        atr: current ATR14 value for dynamic SL/TP sizing.
        """
        if not self.enabled:
            return "Alpaca keys not set — trading disabled", None

        # Load live settings from file (updated via dashboard)
        settings = db.load_settings()

        if not settings.get("trading_enabled", True):
            return "Trading disabled in Settings tab", None

        v_direction = direction.replace("STRONG ", "")
        if v_direction not in ("BUY", "SELL"):
            return "Neutral — no trade", None

        threshold = int(settings.get("score_threshold", SCORE_THRESHOLD))
        log_only_score = threshold - 1

        if score < log_only_score:
            return f"Score {score}/5 — below threshold ({threshold})", None

        if score == log_only_score:
            log.info("Score %d/5: signal logged, not executed (threshold=%d)", score, threshold)
            return f"Score {score}/5: signal noted — threshold is {threshold}", None

        # score >= threshold — execute
        if settings.get("block_asian", True) and session == "Asian":
            return "Skipped: Asian session (low liquidity)", None

        if not self._daily_loss_ok(settings.get("max_daily_loss", MAX_DAILY_LOSS)):
            return f"Halted: daily loss limit ({settings.get('max_daily_loss', MAX_DAILY_LOSS)}%) reached", None

        max_pos = int(settings.get("max_positions", MAX_POSITIONS))
        if not self._positions_ok(max_pos):
            return f"Skipped: {max_pos} open positions already", None

        return self._submit(v_direction, price, score, session, atr, settings)

    # ── Order submission ────────────────────────────────────────────────────
    def _submit(self, direction, price, score, session, atr=None, settings=None):
        if settings is None:
            settings = db.load_settings()
        symbol = self.gold_symbol()
        side   = "buy" if direction == "BUY" else "sell"

        # ── Sell: check shortability and handle conflicting long ───────────
        if side == "sell":
            if self._shortable is None:
                self.gold_symbol()
            if not self._shortable:
                return f"Cannot short {symbol} — asset not shortable on this account", None
            try:
                positions = {p.symbol: p for p in self.positions()}
                if symbol in positions and float(positions[symbol].qty) > 0:
                    self._get_api().close_position(symbol)
                    log.info("Closed existing long %s before entering short", symbol)
            except Exception as exc:
                log.warning("Could not close conflicting long: %s", exc)

        # Dynamic SL/TP from ATR using live settings; fallback to fixed %
        sl_mult  = float(settings.get("atr_sl_mult", ATR_SL_MULT))
        tp_mult  = float(settings.get("atr_tp_mult", ATR_TP_MULT))
        sl_floor = float(settings.get("atr_sl_min",  ATR_SL_MIN))
        sl_cap   = float(settings.get("atr_sl_max",  ATR_SL_MAX))

        if atr and atr > 0:
            sl_pct = max(sl_floor, min(sl_cap, atr / price * 100 * sl_mult))
            tp_pct = sl_pct * (tp_mult / sl_mult)
            log.info("ATR-based sizing: SL=%.2f%%  TP=%.2f%%  (ATR=%.2f)", sl_pct, tp_pct, atr)
        else:
            sl_pct, tp_pct = STOP_LOSS_PCT, TAKE_PROFIT_PCT

        if direction == "BUY":
            sl = round(price * (1 - sl_pct / 100), 2)
            tp = round(price * (1 + tp_pct / 100), 2)
        else:
            sl = round(price * (1 + sl_pct / 100), 2)
            tp = round(price * (1 - tp_pct / 100), 2)

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
