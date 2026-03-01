"""
deploy/predictor.py
====================
RPPO / multi-model inference engine for the FastAPI bridge.

Critical design notes
─────────────────────
1. ACTION INDEX vs POSITION VALUE
   model.predict() returns an INTEGER INDEX into config.POSITIONS = [-1, 0, 1].
       index 0 → position -1  (SHORT)
       index 1 → position  0  (FLAT)
       index 2 → position +1  (LONG)
   We ALWAYS convert before sending to the EA:
       action_value = POSITIONS[action_index]

2. OBSERVATION SPACE — must match training exactly
   gym-trading-env:
     a) Silently appends feature_close_pct = close.pct_change() to the df
        BEFORE building the observation space.
     b) Uses DataFrame COLUMN ORDER (not sorted) for obs_features.
   Therefore this predictor:
     a) Adds feature_close_pct after running the feature pipeline.
     b) Reads obs_features from model_info.json (saved at training time) OR
        falls back to reading a temp TradingEnv at startup — never uses sorted().

3. LSTM STATE
   RecurrentPPO.predict() requires:
       state=lstm_states        # None for first step of new episode
       episode_start=np.array([True/False])
   States are stored per (model_key, session_id) and expire after TTL.
"""

from __future__ import annotations

import sys
import json
import time
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("predictor")


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────


class Predictor:
    """
    Wraps one specific trained model (algorithm + version).

    Path convention:  models/{algo}/{version}/final_model.zip
                      models/{algo}/{version}/model_info.json
    """

    LSTM_STATE_TTL = 3600  # seconds before LSTM state expires
    MIN_CANDLES = 100  # minimum bars for feature pipeline
    PIPS_PER_UNIT = 10_000  # EURUSD: 1 pip = 0.0001 = 1/10_000

    def __init__(self, bot_root: Path, algo: str, version: str) -> None:
        self.bot_root = bot_root
        self.algo = algo.lower()
        self.version = version.lower()
        self.model_dir = bot_root / "models" / self.algo / self.version
        self.model_path = self.model_dir / "final_model.zip"
        self.info_path = self.model_dir / "model_info.json"

        self._model: Any = None
        self._loaded: bool = False
        self._obs_features: list[str] = []
        self._positions: list[int] = []
        self._algorithm: str = ""

        self._lstm_states: dict[str, dict] = {}
        self._episode_starts: dict[str, np.ndarray] = {}

        sys.path.insert(0, str(bot_root))

    @property
    def key(self) -> str:
        return f"{self.algo}/{self.version}"

    def is_loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict:
        return {
            "algo": self.algo,
            "version": self.version,
            "model_path": str(self.model_path),
            "loaded": self._loaded,
            "n_features": len(self._obs_features),
            "positions": self._positions,
            "algorithm_class": self._algorithm,
        }

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run training first or copy model to models/{self.algo}/{self.version}/"
            )

        AlgorithmClass = self._get_algorithm_class()
        self._model = AlgorithmClass.load(str(self.model_path), device="cpu")
        self._loaded = True
        log.info(f"[{self.key}] Loaded from {self.model_path}")

        if self.info_path.exists():
            self._load_model_info()
        else:
            log.warning(
                f"[{self.key}] model_info.json missing — deriving obs features. "
                "Run: python deploy/save_model_info.py"
            )
            self._derive_obs_features()

        log.info(
            f"[{self.key}] {len(self._obs_features)} obs features, positions={self._positions}"
        )

    def _get_algorithm_class(self):
        tag = self.algo.upper().replace("-", "").replace("_", "")
        if tag in ("RECURRENTPPO", "RPPO"):
            try:
                from sb3_contrib import RecurrentPPO

                self._algorithm = "RecurrentPPO"
                return RecurrentPPO
            except ImportError:
                raise ImportError(
                    "sb3-contrib not installed. Run: pip install sb3-contrib"
                )
        from stable_baselines3 import PPO, A2C

        mapping = {"PPO": PPO, "A2C": A2C}
        cls = mapping.get(tag, PPO)
        self._algorithm = tag if tag in mapping else "PPO"
        return cls

    def _load_model_info(self) -> None:
        with open(self.info_path) as f:
            info = json.load(f)
        self._obs_features = info["obs_features"]
        self._positions = info.get("positions", [-1, 0, 1])
        log.info(
            f"[{self.key}] model_info.json loaded ({len(self._obs_features)} features)"
        )

    def _derive_obs_features(self) -> None:
        import config

        self._positions = list(config.POSITIONS)
        try:
            import gymnasium as gym
            import gym_trading_env  # noqa: F401
            from features.pipeline import load_processed

            df = load_processed()
            env = gym.make(
                "TradingEnv",
                name="INFO_PROBE",
                df=df,
                positions=config.POSITIONS,
                trading_fees=config.TRADING_FEES,
                borrow_interest_rate=config.BORROW_INTEREST_RATE,
                portfolio_initial_value=config.PORTFOLIO_INITIAL_VALUE,
                max_episode_duration=100,
                verbose=0,
            )
            raw = env.unwrapped
            if hasattr(raw, "_obs_features"):
                self._obs_features = list(raw._obs_features)
            elif hasattr(raw, "obs_features"):
                self._obs_features = list(raw.obs_features)
            else:
                self._obs_features = [
                    c for c in raw.df.columns if c.startswith("feature_")
                ]
            env.close()
            log.info(
                f"[{self.key}] Derived {len(self._obs_features)} features from TradingEnv"
            )
        except Exception as e:
            log.warning(
                f"[{self.key}] TradingEnv probe failed ({e}) — pipeline fallback"
            )
            self._fallback_obs_features()

    def _fallback_obs_features(self) -> None:
        try:
            from features.pipeline import load_processed

            df = load_processed()
            cols = [c for c in df.columns if c.startswith("feature_")]
            if "feature_close_pct" not in cols:
                cols.append("feature_close_pct")
            self._obs_features = cols
        except Exception:
            self._obs_features = [
                "feature_fvg_active",
                "feature_fvg_direction",
                "feature_fvg_rebalance_pct",
                "feature_fvg_position",
                "feature_dist_to_fvg",
                "feature_bos_bull_st",
                "feature_bos_bear_st",
                "feature_bos_bull_it",
                "feature_bos_bear_it",
                "feature_bos_bull_lt",
                "feature_bos_bear_lt",
                "feature_choch_bull",
                "feature_choch_bear",
                "feature_choch_bull_active",
                "feature_choch_bear_active",
                "feature_st_bias",
                "feature_it_bias",
                "feature_lt_bias",
                "feature_bias_aligned",
                "feature_dist_to_sth",
                "feature_dist_to_stl",
                "feature_dist_to_ith",
                "feature_dist_to_itl",
                "feature_dist_to_lth",
                "feature_dist_to_ltl",
                "feature_it_range_position",
                "feature_lt_range_position",
                "feature_ob_active",
                "feature_ob_direction",
                "feature_ob_position",
                "feature_ob_correct_side",
                "feature_ob_50_dist",
                "feature_bb_active",
                "feature_bb_direction",
                "feature_bb_position",
                "feature_bb_correct_side",
                "feature_bb_50_dist",
                "feature_session",
                "feature_killzone_active",
                "feature_close_pct",
            ]

    # ── LSTM ──────────────────────────────────────────────────────────────────

    def _get_lstm_state(self, session_id: str, reset: bool = False):
        now = time.time()
        stale = [
            k
            for k, v in self._lstm_states.items()
            if now - v["last_seen"] > self.LSTM_STATE_TTL
        ]
        for k in stale:
            del self._lstm_states[k]
            self._episode_starts.pop(k, None)

        if reset or session_id not in self._lstm_states:
            self._lstm_states[session_id] = {"states": None, "last_seen": now}
            self._episode_starts[session_id] = np.ones((1,), dtype=bool)
            return None, True

        self._lstm_states[session_id]["last_seen"] = now
        return self._lstm_states[session_id]["states"], False

    def _save_lstm_state(self, session_id: str, new_states) -> None:
        if session_id in self._lstm_states:
            self._lstm_states[session_id]["states"] = new_states
        self._episode_starts[session_id] = np.zeros((1,), dtype=bool)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full ICT feature pipeline — identical order to training.
        Also adds feature_close_pct (gym-trading-env adds this internally).
        """
        from features.structure import compute_atr, add_structure_features
        from features.zones import (
            add_fvg_features,
            add_order_block_features,
            add_breaker_block_features,
        )
        from features.session import add_session_features

        df["_atr"] = compute_atr(df)
        df = add_fvg_features(df)
        df = add_structure_features(df)
        df = add_order_block_features(df)
        df = add_breaker_block_features(df)
        df = add_session_features(df)

        # CRITICAL: gym-trading-env appends this before building obs space.
        # Model was trained with it — must be present or observation shape is wrong.
        if "feature_close_pct" not in df.columns:
            df["feature_close_pct"] = df["close"].pct_change().fillna(0)

        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df = df.dropna()
        return df

    # ── Obs ───────────────────────────────────────────────────────────────────

    def _build_obs(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract observation vector for the last row using the EXACT feature
        list and ORDER stored in model_info.json (or derived at load time).

        NEVER uses sorted() — column order must match training.
        """
        missing = [c for c in self._obs_features if c not in df.columns]
        if missing:
            raise ValueError(f"[{self.key}] Missing obs columns: {missing}")
        return df[self._obs_features].iloc[-1].values.astype(np.float32).reshape(1, -1)

    # ── Confidence ────────────────────────────────────────────────────────────

    def _get_confidence(self, obs: np.ndarray, lstm_states, episode_starts) -> float:
        try:
            import torch

            policy = self._model.policy
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                if self._algorithm == "RecurrentPPO" and lstm_states is not None:
                    lstm_t = (
                        torch.FloatTensor(lstm_states[0]),
                        torch.FloatTensor(lstm_states[1]),
                    )
                    ep_t = torch.BoolTensor(episode_starts)
                    feats, _ = policy.extract_features(obs_t)
                    latent_pi, _, _ = policy._process_sequence(feats, lstm_t, ep_t)
                else:
                    feats = policy.extract_features(obs_t)
                    if hasattr(policy, "mlp_extractor"):
                        latent_pi, _ = policy.mlp_extractor(feats)
                    else:
                        latent_pi = feats
                dist = policy.action_dist.proba_distribution(
                    policy.action_net(latent_pi)
                )
                probs = dist.distribution.probs.cpu().numpy().flatten()
            return float(probs.max())
        except Exception:
            return 0.5

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        session_id: str,
        df: pd.DataFrame,
        current_position: int = 0,
        reset_lstm: bool = False,
    ) -> dict[str, Any]:
        """
        Full inference pipeline for one live candle.

        Returns:
            action           : Position VALUE -1/0/+1 (EA uses this directly)
            action_index     : Raw model index 0/1/2 (debug only)
            confidence       : Softmax confidence 0–1
            sl_pips          : Stop-loss in pips
            tp_pips          : Take-profit in pips
            atr_pips         : Current ATR in pips
            sl_price         : Actual SL price level
            sl_type          : SL placement method used
            bias_aligned     : feature_bias_aligned value
            killzone_active  : feature_killzone_active value
            last_close       : Current close price
        """
        if not self._loaded:
            raise RuntimeError(f"[{self.key}] Not loaded")

        processed = self._run_pipeline(df.copy())
        if len(processed) == 0:
            raise ValueError(f"[{self.key}] Empty DataFrame after pipeline")

        obs = self._build_obs(processed)

        lstm_states, is_episode_start = self._get_lstm_state(
            session_id, reset=reset_lstm
        )
        episode_starts = np.array([is_episode_start], dtype=bool)

        action_array, new_lstm_states = self._model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        action_index = int(action_array[0])

        # ── ACTION INDEX → POSITION VALUE ─────────────────────────────────────
        # model.predict() returns index (0/1/2) into POSITIONS = [-1, 0, 1]
        # The EA ALWAYS expects position values: -1=SHORT, 0=FLAT, +1=LONG
        positions = self._positions if self._positions else [-1, 0, 1]
        action_value = int(positions[action_index])

        self._save_lstm_state(session_id, new_lstm_states)

        confidence = self._get_confidence(obs, lstm_states, episode_starts)

        last = processed.iloc[-1]
        atr_raw = (
            float(processed["_atr"].iloc[-1]) if "_atr" in processed.columns else 0.001
        )
        close_raw = float(last["close"])
        atr_pips = round(atr_raw * self.PIPS_PER_UNIT, 1)

        import config

        sl_atr = config.CATASTROPHIC_SL_ATR
        rr = config.MIN_RR_RATIO

        from env.trading_env import get_sl_price

        features_d = {c: float(last[c]) for c in last.index if c.startswith("feature_")}
        internal_d = {c: float(last[c]) for c in last.index if c.startswith("_")}

        if action_value != 0:
            sl_price, sl_type = get_sl_price(
                action_value, features_d, internal_d, close_raw
            )
            max_sl = sl_atr * atr_raw
            if abs(close_raw - sl_price) > max_sl:
                sl_price = (
                    close_raw - max_sl if action_value == 1 else close_raw + max_sl
                )
                sl_type = "atr_cap"
        else:
            sl_price = close_raw - (1.5 * atr_raw)
            sl_type = "flat"

        sl_pips = round(abs(close_raw - sl_price) * self.PIPS_PER_UNIT, 1)
        tp_pips = round(sl_pips * rr, 1)

        return {
            "action": action_value,
            "action_index": action_index,
            "confidence": confidence,
            "sl_pips": max(sl_pips, 5.0),
            "tp_pips": max(tp_pips, 15.0),
            "atr_pips": atr_pips,
            "sl_price": round(sl_price, 5),
            "sl_type": sl_type,
            "bias_aligned": float(features_d.get("feature_bias_aligned", 0)),
            "killzone_active": int(features_d.get("feature_killzone_active", 0)),
            "last_close": close_raw,
        }

    def active_sessions(self) -> int:
        return len(self._lstm_states)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY — manages multiple models
# ──────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """
    Discovers, loads, and routes predictions to multiple Predictor instances.

    Directory structure:
        models/
          rppo/v1/final_model.zip  +  model_info.json
          rppo/v2/final_model.zip  +  model_info.json
          ppo/v1/final_model.zip   +  model_info.json

    Legacy flat layout (models/forex_ppo/final_model.zip) is auto-detected
    and served as rppo/v1 for backward compatibility.
    """

    def __init__(self, bot_root: Path) -> None:
        self.bot_root = bot_root
        self._models: dict[str, Predictor] = {}
        self._default_key = ""

    def discover(self) -> list[str]:
        models_dir = self.bot_root / "models"
        found = []

        if not models_dir.exists():
            log.warning(f"models/ not found at {models_dir}")
            return found

        for algo_dir in sorted(models_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            # Legacy flat layout: models/forex_ppo/final_model.zip
            if (algo_dir / "final_model.zip").exists():
                key = "rppo/v1"
                if key not in self._models:
                    p = Predictor.__new__(Predictor)
                    p.bot_root = self.bot_root
                    p.algo = "rppo"
                    p.version = "v1"
                    p.model_dir = algo_dir
                    p.model_path = algo_dir / "final_model.zip"
                    p.info_path = algo_dir / "model_info.json"
                    p._model = None
                    p._loaded = False
                    p._obs_features = []
                    p._positions = []
                    p._algorithm = ""
                    p._lstm_states = {}
                    p._episode_starts = {}
                    sys.path.insert(0, str(self.bot_root))
                    self._models[key] = p
                    found.append(key)
                continue

            for version_dir in sorted(algo_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                if not (version_dir / "final_model.zip").exists():
                    continue
                key = f"{algo_dir.name}/{version_dir.name}"
                if key not in self._models:
                    self._models[key] = Predictor(
                        self.bot_root, algo_dir.name, version_dir.name
                    )
                    found.append(key)

        if found:
            self._default_key = found[0]
            log.info(f"Discovered models: {found}  (default: {self._default_key})")
        else:
            log.warning("No models found in models/")
        return found

    def load_all(self) -> None:
        for key, p in self._models.items():
            try:
                p.load()
                log.info(f"✅ {key} ready")
            except Exception as e:
                log.error(f"❌ {key} failed: {e}")

    def get(self, algo: str, version: str) -> Predictor:
        key = f"{algo.lower()}/{version.lower()}"
        if key not in self._models:
            raise KeyError(
                f"Model '{key}' not available. Loaded: {list(self._models.keys())}"
            )
        p = self._models[key]
        if not p.is_loaded():
            p.load()
        return p

    def predict(
        self,
        algo: str,
        version: str,
        session_id: str,
        df: pd.DataFrame,
        current_position: int = 0,
        reset_lstm: bool = False,
    ) -> dict[str, Any]:
        return self.get(algo, version).predict(
            session_id, df, current_position, reset_lstm
        )

    def default_predict(
        self,
        session_id: str,
        df: pd.DataFrame,
        current_position: int = 0,
        reset_lstm: bool = False,
    ) -> dict[str, Any]:
        if not self._default_key:
            raise RuntimeError("No models loaded")
        algo, version = self._default_key.split("/", 1)
        return self.predict(algo, version, session_id, df, current_position, reset_lstm)

    def list_models(self) -> list[dict]:
        return [p.info() for p in self._models.values()]

    def any_loaded(self) -> bool:
        return any(p.is_loaded() for p in self._models.values())

    @property
    def default_key(self) -> str:
        return self._default_key
