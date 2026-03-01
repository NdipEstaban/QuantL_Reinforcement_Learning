"""
api/state.py
=============
In-memory state management for the bridge.

Classes:
    TradeRecord       — single closed trade snapshot
    SessionStore      — tracks active EA sessions
    PerformanceTracker — rolling metrics: win rate, PF, equity curve, etc.
"""

from __future__ import annotations

import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone


# ══════════════════════════════════════════════════════════════════════════════
# TRADE RECORD
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TradeRecord:
    signal_id: str
    action: int  # -1 / 1
    risk_pct: float
    confidence: float
    symbol: str
    opened_at: str
    closed_at: str = ""
    pnl_pct: float = 0.0
    close_reason: str = "open"  # open / tp / sl / manual / timeout
    balance_at_open: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0

    @property
    def direction(self) -> str:
        return "LONG" if self.action == 1 else "SHORT"

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "direction": self.direction,
            "risk_pct": round(self.risk_pct * 100, 1),
            "confidence": round(self.confidence, 3),
            "symbol": self.symbol,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "pnl_pct": round(self.pnl_pct * 100, 4),
            "close_reason": self.close_reason,
            "result": "WIN" if self.is_win else "LOSS",
        }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STORE
# ══════════════════════════════════════════════════════════════════════════════


class SessionStore:
    """Tracks active EA connection sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}

    def register(self, session_id: str, meta: dict) -> None:
        self._sessions[session_id] = {**meta, "last_seen": time.time()}

    def touch(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["last_seen"] = time.time()

    def count(self) -> int:
        # Only count sessions seen in the last 5 minutes
        cutoff = time.time() - 300
        return sum(1 for s in self._sessions.values() if s.get("last_seen", 0) > cutoff)

    def all(self) -> list[dict]:
        return list(self._sessions.values())


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKER
# ══════════════════════════════════════════════════════════════════════════════


class PerformanceTracker:
    """
    Rolling performance metrics updated in real-time.

    Tracks:
        - Win rate, profit factor, expectancy
        - Equity curve (last 200 points)
        - Latency histogram
        - Direction breakdown (long vs short)
        - Confidence distribution
        - Recent signals queue
    """

    MAX_HISTORY = 1000  # max closed trades retained in memory
    MAX_EQUITY = 200  # equity curve points

    def __init__(self) -> None:
        self._open_signals: dict[str, TradeRecord] = {}
        self._closed_trades: deque[TradeRecord] = deque(maxlen=self.MAX_HISTORY)
        self._equity_curve: deque[dict] = deque(maxlen=self.MAX_EQUITY)
        self._latency_ms: deque[float] = deque(maxlen=500)
        self._signal_count: int = 0
        self._flat_count: int = 0
        self._start_time: float = time.time()
        self._peak_balance: float = 0.0
        self._current_balance: float = 0.0

    # ── Ingest ────────────────────────────────────────────────────────────────

    def register_signal(self, signal_id: str, meta: dict) -> None:
        """Register an open signal (trade entry)."""
        self._signal_count += 1
        self._current_balance = meta.get("balance", self._current_balance)
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance

        record = TradeRecord(
            signal_id=signal_id,
            action=meta["action"],
            risk_pct=meta["risk_pct"],
            confidence=meta["confidence"],
            symbol=meta.get("symbol", "EURUSD"),
            opened_at=meta["timestamp"],
            balance_at_open=meta.get("balance", 0),
        )
        self._open_signals[signal_id] = record

        # Log entry to equity curve
        self._equity_curve.append(
            {
                "t": meta["timestamp"],
                "balance": self._current_balance,
                "event": "entry",
            }
        )

    def record_close(self, signal_id: str, pnl_pct: float, close_reason: str) -> None:
        """Record a trade close. Updates all running metrics."""
        record = self._open_signals.pop(signal_id, None)
        if record is None:
            return

        record.pnl_pct = pnl_pct
        record.close_reason = close_reason
        record.closed_at = datetime.now(timezone.utc).isoformat()

        self._closed_trades.append(record)

        # Update balance estimate
        self._current_balance *= 1 + pnl_pct
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance

        self._equity_curve.append(
            {
                "t": record.closed_at,
                "balance": round(self._current_balance, 2),
                "event": close_reason,
                "pnl": round(pnl_pct * 100, 4),
            }
        )

    def register_flat(self) -> None:
        self._flat_count += 1

    def update_latency(self, ms: float) -> None:
        self._latency_ms.append(ms)

    # ── Computed Metrics ──────────────────────────────────────────────────────

    def _closed_list(self) -> list[TradeRecord]:
        return list(self._closed_trades)

    def win_rate(self) -> float:
        closed = self._closed_list()
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.is_win)
        return round(wins / len(closed), 4)

    def profit_factor(self) -> float:
        closed = self._closed_list()
        gross_win = sum(t.pnl_pct for t in closed if t.is_win)
        gross_loss = sum(abs(t.pnl_pct) for t in closed if not t.is_win)
        if gross_loss == 0:
            return 0.0 if gross_win == 0 else 99.0
        return round(gross_win / gross_loss, 3)

    def expectancy(self) -> float:
        closed = self._closed_list()
        if not closed:
            return 0.0
        return round(sum(t.pnl_pct for t in closed) / len(closed) * 100, 5)

    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self._closed_list() if t.is_win]
        return round(statistics.mean(wins) * 100, 4) if wins else 0.0

    def avg_loss(self) -> float:
        losses = [t.pnl_pct for t in self._closed_list() if not t.is_win]
        return round(statistics.mean(losses) * 100, 4) if losses else 0.0

    def max_drawdown(self) -> float:
        eq = [p["balance"] for p in self._equity_curve if "balance" in p]
        if len(eq) < 2:
            return 0.0
        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd * 100, 3)

    def direction_split(self) -> dict:
        closed = self._closed_list()
        longs = [t for t in closed if t.action == 1]
        shorts = [t for t in closed if t.action == -1]
        return {
            "long_count": len(longs),
            "short_count": len(shorts),
            "long_wr": round(
                sum(1 for t in longs if t.is_win) / len(longs) if longs else 0, 4
            ),
            "short_wr": round(
                sum(1 for t in shorts if t.is_win) / len(shorts) if shorts else 0, 4
            ),
        }

    def confidence_stats(self) -> dict:
        closed = self._closed_list()
        if not closed:
            return {"mean": 0, "min": 0, "max": 0}
        confs = [t.confidence for t in closed]
        return {
            "mean": round(statistics.mean(confs), 3),
            "min": round(min(confs), 3),
            "max": round(max(confs), 3),
        }

    def latency_stats(self) -> dict:
        if not self._latency_ms:
            return {"mean_ms": 0, "p99_ms": 0}
        lats = sorted(self._latency_ms)
        p99 = lats[int(len(lats) * 0.99)] if len(lats) > 1 else lats[-1]
        return {
            "mean_ms": round(statistics.mean(lats), 1),
            "p99_ms": round(p99, 1),
        }

    def close_reason_breakdown(self) -> dict:
        closed = self._closed_list()
        reasons: dict[str, int] = {}
        for t in closed:
            reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1
        return reasons

    def recent_trades(self, n: int = 20) -> list[dict]:
        return [t.to_dict() for t in list(self._closed_trades)[-n:]]

    def equity_curve_data(self) -> list[dict]:
        return list(self._equity_curve)

    def uptime_hours(self) -> float:
        return round((time.time() - self._start_time) / 3600, 2)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full metrics snapshot — sent to dashboard on every WS update."""
        closed = self._closed_list()
        return {
            "total_signals": self._signal_count,
            "total_trades": len(closed),
            "open_trades": len(self._open_signals),
            "flat_signals": self._flat_count,
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "expectancy_pct": self.expectancy(),
            "avg_win_pct": self.avg_win(),
            "avg_loss_pct": self.avg_loss(),
            "max_drawdown_pct": self.max_drawdown(),
            "current_balance": round(self._current_balance, 2),
            "peak_balance": round(self._peak_balance, 2),
            "direction": self.direction_split(),
            "confidence": self.confidence_stats(),
            "latency": self.latency_stats(),
            "close_reasons": self.close_reason_breakdown(),
            "equity_curve": self.equity_curve_data()[-50:],
            "recent_trades": self.recent_trades(10),
            "uptime_hours": self.uptime_hours(),
        }

    def trade_history(self, limit: int = 100, offset: int = 0) -> dict:
        all_trades = [t.to_dict() for t in reversed(list(self._closed_trades))]
        return {
            "total": len(all_trades),
            "limit": limit,
            "offset": offset,
            "trades": all_trades[offset : offset + limit],
        }
