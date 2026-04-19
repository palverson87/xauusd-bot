"""
Signal logging, outcome tracking, adaptive weights — all via SQLite.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Use /app when it exists (Railway Volume mount); fall back to script directory locally
_RAILWAY = Path("/app")
BASE_DIR  = _RAILWAY if _RAILWAY.exists() else Path(__file__).parent

DB_PATH      = BASE_DIR / "signals.db"
WEIGHTS_PATH = BASE_DIR / "weights.json"

INDICATOR_COLS = {
    "RSI":        "rsi_vote",
    "MACD":       "macd_vote",
    "MA Cross 4H":"ma_vote",
    "DXY / UUP":  "dxy_vote",
    "Session":    "session_vote",
}

DEFAULT_WEIGHTS = {k: 1.0 for k in INDICATOR_COLS}


# ── Connection ─────────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Schema ─────────────────────────────────────────────────────────────────────
def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,
            price         REAL    NOT NULL,
            direction     TEXT    NOT NULL,   -- BUY | SELL | NEUTRAL
            score         INTEGER NOT NULL,
            session       TEXT    NOT NULL,
            rsi           REAL,
            rsi_vote      INTEGER DEFAULT 0,
            macd_vote     INTEGER DEFAULT 0,
            ma_vote       INTEGER DEFAULT 0,
            dxy_vote      INTEGER DEFAULT 0,
            session_vote  INTEGER DEFAULT 0,
            interval      TEXT,
            logged_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
        );
        CREATE TABLE IF NOT EXISTS outcomes (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id     INTEGER NOT NULL REFERENCES signals(id),
            horizon       TEXT    NOT NULL,   -- 1h | 4h | 24h
            price_at      REAL,
            pnl_pct       REAL,
            won           INTEGER,            -- 1 = correct, 0 = wrong
            due_at        TEXT    NOT NULL,
            recorded_at   TEXT,
            UNIQUE(signal_id, horizon)
        );
        """)


# ── Write ──────────────────────────────────────────────────────────────────────
def log_signal(ts, price, direction, score, session, rsi, votes, interval="1h"):
    """
    Persist a signal. Returns the new row id, or None if deduplicated
    (same direction already logged within the last 60 minutes).
    """
    if direction == "NEUTRAL":
        return None

    hour_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00")
    with _conn() as c:
        dup = c.execute(
            "SELECT id FROM signals WHERE direction=? AND ts >= ? LIMIT 1",
            (direction, hour_bucket),
        ).fetchone()
        if dup:
            return None

        cur = c.execute(
            """INSERT INTO signals
               (ts, price, direction, score, session, rsi,
                rsi_vote, macd_vote, ma_vote, dxy_vote, session_vote, interval)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts, price, direction, score, session, rsi,
                votes.get("RSI", 0), votes.get("MACD", 0),
                votes.get("MA Cross 4H", 0), votes.get("DXY / UUP", 0),
                votes.get("Session", 0), interval,
            ),
        )
        sid = cur.lastrowid
        now = datetime.now(timezone.utc)
        for hours in [1, 4, 24]:
            due = (now + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
            c.execute(
                "INSERT INTO outcomes (signal_id, horizon, due_at) VALUES (?,?,?)",
                (sid, f"{hours}h", due),
            )
        return sid


# ── Outcome tracking ───────────────────────────────────────────────────────────
def get_pending_outcomes():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        rows = c.execute(
            """SELECT o.id, o.signal_id, o.horizon,
                      s.price AS entry_price, s.direction
               FROM outcomes o
               JOIN signals s ON s.id = o.signal_id
               WHERE o.recorded_at IS NULL AND o.due_at <= ?""",
            (now,),
        ).fetchall()
    return [dict(r) for r in rows]


def record_outcome(outcome_id, price_at, entry_price, direction):
    pnl_pct = (price_at / entry_price - 1) * 100
    won = int(
        (direction == "BUY"  and pnl_pct > 0) or
        (direction == "SELL" and pnl_pct < 0)
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        c.execute(
            "UPDATE outcomes SET price_at=?, pnl_pct=?, won=?, recorded_at=? WHERE id=?",
            (price_at, pnl_pct, won, now, outcome_id),
        )


# ── Read / stats ───────────────────────────────────────────────────────────────
def get_recent_signals(n=20):
    with _conn() as c:
        rows = c.execute(
            """SELECT s.ts, s.price, s.direction, s.score, s.session, s.rsi,
                      o1.pnl_pct AS pnl_1h,  o1.won  AS won_1h,
                      o4.pnl_pct AS pnl_4h,  o4.won  AS won_4h,
                      o24.pnl_pct AS pnl_24h, o24.won AS won_24h
               FROM signals s
               LEFT JOIN outcomes o1  ON o1.signal_id  = s.id AND o1.horizon  = '1h'
               LEFT JOIN outcomes o4  ON o4.signal_id  = s.id AND o4.horizon  = '4h'
               LEFT JOIN outcomes o24 ON o24.signal_id = s.id AND o24.horizon = '24h'
               ORDER BY s.id DESC LIMIT ?""",
            (n,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_performance_stats():
    with _conn() as c:
        # ── win rate by session ────────────────────────────────────────────
        by_session = [dict(r) for r in c.execute(
            """SELECT s.session, COUNT(*) AS total,
                      SUM(o.won) AS wins
               FROM signals s
               JOIN outcomes o ON o.signal_id=s.id AND o.horizon='1h'
               WHERE o.recorded_at IS NOT NULL AND s.direction != 'NEUTRAL'
               GROUP BY s.session ORDER BY wins * 1.0 / total DESC"""
        ).fetchall()]

        # ── win rate by confluence score ───────────────────────────────────
        by_score = [dict(r) for r in c.execute(
            """SELECT s.score, COUNT(*) AS total, SUM(o.won) AS wins
               FROM signals s
               JOIN outcomes o ON o.signal_id=s.id AND o.horizon='1h'
               WHERE o.recorded_at IS NOT NULL AND s.direction != 'NEUTRAL'
               GROUP BY s.score ORDER BY s.score"""
        ).fetchall()]

        # ── equity curve (1h horizon) ──────────────────────────────────────
        equity_rows = [dict(r) for r in c.execute(
            """SELECT s.ts, s.direction, s.price, o.pnl_pct
               FROM signals s
               JOIN outcomes o ON o.signal_id=s.id AND o.horizon='1h'
               WHERE o.recorded_at IS NOT NULL AND s.direction != 'NEUTRAL'
               ORDER BY s.ts ASC"""
        ).fetchall()]

        # ── raw rows for indicator accuracy (last 50) ──────────────────────
        raw = [dict(r) for r in c.execute(
            """SELECT s.direction, s.rsi_vote, s.macd_vote, s.ma_vote,
                      s.dxy_vote, s.session_vote, o.won
               FROM signals s
               JOIN outcomes o ON o.signal_id=s.id AND o.horizon='1h'
               WHERE o.recorded_at IS NOT NULL AND s.direction != 'NEUTRAL'
               ORDER BY s.id DESC LIMIT 50"""
        ).fetchall()]

    # compute indicator accuracy in Python
    vote_map = [
        ("RSI",         "rsi_vote"),
        ("MACD",        "macd_vote"),
        ("MA Cross 4H", "ma_vote"),
        ("DXY / UUP",   "dxy_vote"),
        ("Session",     "session_vote"),
    ]
    indicator_acc = {}
    for name, col in vote_map:
        correct = total = 0
        for row in raw:
            v   = row.get(col) or 0
            won = row.get("won")
            sig = row.get("direction")
            if v == 0 or won is None:
                continue
            total += 1
            expected = 1 if (sig == "BUY" and won) or (sig == "SELL" and not won) else -1
            if v == expected:
                correct += 1
        indicator_acc[name] = {"correct": correct, "total": total,
                                "pct": correct / total * 100 if total else None}

    return {
        "by_session":    by_session,
        "by_score":      by_score,
        "equity":        equity_rows,
        "indicator_acc": indicator_acc,
    }


# ── Adaptive weights ───────────────────────────────────────────────────────────
def load_weights():
    if WEIGHTS_PATH.exists():
        try:
            return json.loads(WEIGHTS_PATH.read_text())
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()


def save_weights(w):
    WEIGHTS_PATH.write_text(json.dumps(w, indent=2))


def recalculate_weights():
    """Recompute weights from last-50 accuracy; save and return."""
    stats = get_performance_stats()
    w = load_weights()
    for name, data in stats["indicator_acc"].items():
        if data["total"] >= 10:
            acc = data["correct"] / data["total"]   # 0.0–1.0
            # 50 % accuracy → weight 1.0; 75 % → 1.5; 25 % → 0.5
            w[name] = round(max(0.3, min(2.0, acc * 2.0)), 3)
    save_weights(w)
    return w
