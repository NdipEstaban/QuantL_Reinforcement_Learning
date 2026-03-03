"""
config.py — FIXES:
  1. VOLUME_COLUMN "None" (string) → None (Python None)
  2. Phase 1 ent_coef 0.02 → 0.08  (comment said high entropy but value was lowest)
  3. DEFAULT_MODEL "rppo/v1" → "rppo/v1.1"  (must match MODEL_SAVE_PATH)
  4. CATASTROPHIC_SL_ATR 5.0 → 10.0  (model-exit mode needs wide black-swan guard)
  5. REWARD_FUNCTION "account_flip_aggressive_v2" now registered in reward.py
"""

# ── DATA ─────────────────────────────────────────────────────────
DATA_FILE = "data/raw/EURUSD_M15_2010-2016.csv"
DATE_COLUMN = "date"
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = None  # FIX 1: was "None" (string) — pipeline read a column named "None"

BROKER_TIMEZONE = "UTC"

# ── FEATURE ENGINEERING ──────────────────────────────────────────
SWING_LOOKBACK = 3
ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.5
FVG_QUALIFY_THRESHOLD = 0.80
CHOCH_ACTIVE_WINDOW = 20
BOS_ACTIVE_WINDOW = 10

LONDON_KZ_START = 7
LONDON_KZ_END = 10
NY_KZ_START = 12
NY_KZ_END = 15
ASIAN_START = 0
ASIAN_END = 8

# ── ENVIRONMENT ──────────────────────────────────────────────────
POSITIONS = [-1, 0, 1]
TRADING_FEES = 0.0003
BORROW_INTEREST_RATE = 0.000003
PORTFOLIO_INITIAL_VALUE = 10_000
MAX_EPISODE_DURATION = 2000
INITIAL_POSITION = "random"

# ── RISK ─────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.15
RISK_MIN = 0.05
RISK_MAX = 0.15
MIN_RR_RATIO = 3.0
CATASTROPHIC_SL_ATR = 5.0
SL_ATR_BUFFER = 0.1

# ── REWARD ───────────────────────────────────────────────────────
REWARD_FUNCTION = "custom"  # v4 — teaches model to use action=0

FLIP_RRR_MULTIPLIER = 4.0
FLIP_LOG_SCALE = 10.0  # scales the log return signal
FLIP_INACTION_PENALTY = -0.000001  # base per-step flat penalty (small)
FLIP_FLAT_WINDOW = 20
FLIP_DD_PENALTY_WEIGHT = 2.0  # soft — 5.0 is too harsh for early training
FLIP_DD_FREE_ZONE = 0.08  # wide — reduces noise during exploration

# ── V3 REWARD KEYS ──────────────────────────────────────────────────
FLIP_INTRA_DD_THRESHOLD = 0.80  # intra-trade loss threshold (80% of risk)
FLIP_INTRA_DD_WEIGHT = 2.0  # intra-trade DD penalty scale
FLIP_QUALITY_EXIT_BONUS = 0.15  # max bonus for profitable flat exit (RRR-scaled)
FLIP_CONSEC_LOSS_THRESHOLD = 2  # consecutive losses before penalty
FLIP_CONSEC_LOSS_PENALTY = -0.02  # per-step penalty during loss streak
FLIP_DIRECT_FLIP_PENALTY = -0.05  # penalty for +1->-1 or -1->+1 without flat


# ── PARALLELISATION ───────────────────────────────────────────────
N_ENVS = 6

# ── TRAINING ─────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
ALGORITHM = "RecurrentPPO"  # RecurrentPPO or PPO
CHECKPOINT_FREQ = 100_000
MODEL_SAVE_PATH = "models/rppo/v4.0"
LOG_DIR = "logs/"

# ── RECURRENT PPO ─────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 64
LSTM_N_LAYERS = 1

PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 384,
    "n_epochs": 5,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "ent_coef": 0.08,
    "learning_rate": 0.0001,
    "clip_range": 0.08,
}

# ── EVALUATION ───────────────────────────────────────────────────
BASE_TF = "M15"
TRAIN_TEST_SPLIT = 0.80
WFO_FOLDS = 5

# ── MULTI-MODEL / VERSIONING ──────────────────────────────────────
DEFAULT_MODEL = "rppo/v1.1"  # FIX 3: was "rppo/v1" — bridge loaded wrong model

# ── AUTOMATED 3-PHASE TRAINING ────────────────────────────────────
TRAINING_PHASES = [
    {
        "name": "phase1_foundation",
        "description": "Foundation — broad market structure on 2010-2016 data",
        "algorithm": "RecurrentPPO",
        "data_file": "data/raw/EURUSD_M15_2010-2016.csv",
        # "force_reprocess": True,  # ← reprocess even if cache is valid
        "timesteps": 5_000_000,
        "resume_from": "models/rppo/v4.0/final_model.zip",
        "save_to": "models/rppo/v4.0",
        "skip": False,
        "ppo_config": {
            "n_steps": 2048,
            "batch_size": 384,
            "n_epochs": 5,
            "gamma": 0.98,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "learning_rate": 0.00008,
            "clip_range": 0.2,
        },
    },
    {
        "name": "phase2_general",
        "description": "General fine-tune on 2022-2023 training set",
        "algorithm": "RecurrentPPO",
        "data_file": "data/raw/EURUSD_2022-2023_M15.csv",
        # "force_reprocess": True,  # ← reprocess even if cache is valid
        "timesteps": 1_000_000,
        "resume_from": "models/rppo/v4.0/final_model.zip",
        "save_to": "models/rppo/v4.0",
        "skip": True,
        "ppo_config": {
            "n_steps": 2048,
            "batch_size": 384,
            "n_epochs": 5,
            "gamma": 0.98,
            "gae_lambda": 0.95,
            "ent_coef": 0.04,
            "learning_rate": 0.00001,
            "clip_range": 0.05,
        },
    },
    {
        "name": "phase3_recent",
        "description": "Recent fine-tune on 2024 unseen data",
        "algorithm": "RecurrentPPO",
        "data_file": "data/unseen/EURUSD_2024_M15_fixed.csv",
        # "force_reprocess": True,  # ← reprocess even if cache is valid
        "timesteps": 500_000,
        "resume_from": "models/rppo/v4.0/final_model.zip",
        "save_to": "models/rppo/v4.0",
        "skip": True,
        "ppo_config": {
            "n_steps": 2048,
            "batch_size": 384,
            "n_epochs": 3,
            "gamma": 0.98,
            "gae_lambda": 0.95,
            "ent_coef": 0.02,
            "learning_rate": 0.000005,
            "clip_range": 0.03,
        },
    },
]
