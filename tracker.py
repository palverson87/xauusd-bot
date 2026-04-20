"""
Background scheduler:
  • Every 30 min  — fill in pending signal outcomes via live GC=F price
  • Every Sunday 23:59 UTC — generate weekly PDF report
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler

import db
import tv_feed
from alpaca_trader import trader

log = logging.getLogger(__name__)

_scheduler = None


def check_outcomes():
    pending = db.get_pending_outcomes()
    if not pending:
        return
    try:
        price = tv_feed.live_price()
    except Exception as exc:
        log.error("Outcome check — price fetch failed: %s", exc)
        return

    updated = 0
    for row in pending:
        try:
            db.record_outcome(row["id"], price, row["entry_price"], row["direction"])
            updated += 1
        except Exception as exc:
            log.error("record_outcome %s failed: %s", row["id"], exc)

    if updated:
        log.info("Outcomes recorded: %d  — recalculating weights…", updated)
        try:
            new_w = db.recalculate_weights()
            log.info("New weights: %s", new_w)
        except Exception as exc:
            log.error("Weight recalc failed: %s", exc)


def sync_trades():
    """Sync Alpaca order outcomes into the local trades table."""
    if not trader.enabled:
        return
    try:
        trader.sync_open_trades()
    except Exception as exc:
        log.error("Trade sync failed: %s", exc)


def weekly_report():
    try:
        import reporter
        path = reporter.generate_weekly_report()
        log.info("Weekly report saved → %s", path)
    except Exception as exc:
        log.error("Weekly report failed: %s", exc)


def start():
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    db.init_db()

    _scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    _scheduler.add_job(check_outcomes, "interval", minutes=30,   id="outcomes")
    _scheduler.add_job(sync_trades,    "interval", minutes=10,   id="trade_sync")
    _scheduler.add_job(weekly_report,  "cron",
                       day_of_week="sun", hour=23, minute=59,    id="weekly_pdf")
    _scheduler.start()
    log.info("Background scheduler started (outcome checks every 30 min, report every Sunday)")
    return _scheduler
