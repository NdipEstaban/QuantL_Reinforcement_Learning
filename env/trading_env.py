"""
env/trading_env.py
===================
Wraps gym-trading-env with:
  - ICT feature set
  - Custom reward functions
  - Position sizing logic
  - Tiered SL placement based on available structure
"""

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.reward import get_reward_function

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.reward import get_reward_function


class RiskScaledWrapper(Wrapper):
    """
    Scales the portfolio impact of each trade to reflect
    actual fixed-fractional risk per trade.

    Problem this solves:
        gym-trading-env treats action=1 as "100% long".
        If price moves +0.5%, portfolio moves +0.5%.
        At 15% risk per trade with 3:1 RRR, a winning trade
        should move the portfolio +45%, not +0.5%.

    How it works:
        We track the portfolio internally and apply a multiplier
        that converts the env's small returns into risk-scaled returns.

        risk_pct = 0.15 (15% risk per trade)
        When a trade closes with +0.5% raw return:
            → scaled return = (raw_return / expected_move) * risk_pct * rrr
        When a trade closes with -0.5% raw return:
            → scaled return = (raw_return / expected_move) * risk_pct

        This way the reward function sees portfolio values that
        actually reflect 15% risk compounding.

    Parameters:
        risk_pct : float — fraction of portfolio risked per trade (0.15 = 15%)
        rrr      : float — reward-to-risk ratio (3.0 = 3:1)
    """

    def __init__(self, env, risk_pct: float = 0.15, rrr: float = 3.0):
        super().__init__(env)
        self.risk_pct = risk_pct
        self.rrr = rrr
        self._portfolio = config.PORTFOLIO_INITIAL_VALUE
        self._peak = config.PORTFOLIO_INITIAL_VALUE
        self._in_trade = False
        self._entry_price = None
        self._trade_direction = 0

    def reset(self, **kwargs):
        self._portfolio = config.PORTFOLIO_INITIAL_VALUE
        self._peak = config.PORTFOLIO_INITIAL_VALUE
        self._in_trade = False
        self._entry_price = None
        self._trade_direction = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current price and position from info
        raw_price = info.get("price", None)
        raw_pos = info.get("position", 0)

        if raw_price is not None:
            # ── Detect trade entry ──────────────────────────
            if not self._in_trade and raw_pos != 0:
                self._in_trade = True
                self._entry_price = raw_price
                self._trade_direction = raw_pos

            # ── Detect trade exit and apply risk scaling ────
            elif self._in_trade and raw_pos == 0:
                if self._entry_price and self._entry_price > 0:
                    raw_ret = (
                        (raw_price - self._entry_price)
                        / self._entry_price
                        * self._trade_direction
                    )

                    # Scale to fixed-fractional risk
                    if raw_ret > 0:
                        # Winning trade: portfolio grows by risk_pct * rrr
                        # Scaled proportionally to how good the raw return was
                        scale = min(raw_ret / 0.001, self.rrr) if raw_ret > 0 else 0
                        scaled_return = self.risk_pct * min(scale, self.rrr)
                    else:
                        # Losing trade: portfolio shrinks by risk_pct
                        # Capped at risk_pct to prevent catastrophic loss
                        scaled_return = max(
                            raw_ret * self.risk_pct / 0.001, -self.risk_pct
                        )

                    self._portfolio *= 1 + scaled_return
                    self._portfolio = max(self._portfolio, 0.01)  # floor

                self._in_trade = False
                self._entry_price = None
                self._trade_direction = 0

            # Update info with scaled portfolio value
            info["portfolio_valuation"] = self._portfolio
            info["scaled_return"] = self._portfolio / config.PORTFOLIO_INITIAL_VALUE - 1

        return obs, reward, terminated, truncated, info


def make_env(df: pd.DataFrame, mode: str = "train") -> gym.Env:
    """
    Create TradingEnv with custom position sizing that reflects
    actual 5-15% risk per trade compounding.

    The key change: we use a custom reward that operates on
    log portfolio growth with proper risk scaling, and we set
    the env to use leveraged positions so the model's actions
    have real compounding impact.
    """
    reward_fn = get_reward_function(config.REWARD_FUNCTION)

    env = gym.make(
        "TradingEnv",
        name="FOREX_ICT_FLIP",
        df=df,
        positions=config.POSITIONS,
        trading_fees=config.TRADING_FEES,
        borrow_interest_rate=config.BORROW_INTEREST_RATE,
        portfolio_initial_value=config.PORTFOLIO_INITIAL_VALUE,
        initial_position=config.INITIAL_POSITION if mode == "train" else 0,
        max_episode_duration=config.MAX_EPISODE_DURATION,
        reward_function=reward_fn,
        verbose=0,
    )


    return env


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    sl_price: float,
    risk_pct: float = None,
) -> dict:
    """
    Calculate position size based on fixed fractional risk.

    Returns dict with:
        lots         : position size in standard lots
        risk_dollars : dollar amount at risk
        sl_pips      : stop loss distance in pips
    """
    risk_pct = risk_pct or config.RISK_PER_TRADE
    risk_dollars = account_equity * risk_pct
    sl_distance = abs(entry_price - sl_price)

    if sl_distance == 0:
        return {"lots": 0, "risk_dollars": 0, "sl_pips": 0}

    # For EURUSD: 1 standard lot = $10/pip = $100,000 units
    # pip value = 0.0001
    sl_pips = sl_distance / 0.0001
    lot_size = risk_dollars / (sl_pips * 10)

    return {
        "lots": round(lot_size, 2),
        "risk_dollars": round(risk_dollars, 2),
        "sl_pips": round(sl_pips, 1),
    }


def get_sl_price(
    direction: int,
    features: dict,
    internal: dict,
    close: float,
) -> tuple:
    """
    Tiered SL placement — uses best available structural reference.

    Priority (bullish trade):
        1. OB low (if ob_active and ob_direction == +1)
        2. STL price (if nearby)
        3. ATR-based default (close - 1.5 ATR)

    Returns (sl_price, sl_type)
    """
    atr = internal.get("_atr", 0.001)

    if direction == 1:  # LONG
        if (
            features.get("feature_ob_active", 0) == 1
            and features.get("feature_ob_direction", 0) == 1
        ):
            sl = internal["_ob_low"] - (config.SL_ATR_BUFFER * atr)
            return sl, "ob_low"

        stl = internal.get("_stl_price")
        if stl and not np.isnan(stl) and (close - stl) < 2 * atr:
            sl = stl - (config.SL_ATR_BUFFER * atr)
            return sl, "stl"

        return close - (1.5 * atr), "atr_default"

    else:  # SHORT
        if (
            features.get("feature_ob_active", 0) == 1
            and features.get("feature_ob_direction", 0) == -1
        ):
            sl = internal["_ob_high"] + (config.SL_ATR_BUFFER * atr)
            return sl, "ob_high"

        sth = internal.get("_sth_price")
        if sth and not np.isnan(sth) and (sth - close) < 2 * atr:
            sl = sth + (config.SL_ATR_BUFFER * atr)
            return sl, "sth"

        return close + (1.5 * atr), "atr_default"
