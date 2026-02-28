"""
config.py
=========
All project settings. Edit this file before running anything.
"""

# ── DATA ─────────────────────────────────────────────────────────
DATA_FILE = "data/raw/EURUSD_2024_M15_fixed.csv"  # Phase 2 dataset
DATE_COLUMN = "date"
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "None"  # set to None if no volume

# Broker timezone → will be converted to UTC for session features
BROKER_TIMEZONE = "UTC"

# ── FEATURE ENGINEERING ──────────────────────────────────────────

SWING_LOOKBACK = 3
ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.5
FVG_QUALIFY_THRESHOLD = 0.80
CHOCH_ACTIVE_WINDOW = 20
BOS_ACTIVE_WINDOW = 10

# Session times in UTC hours
LONDON_KZ_START = 7
LONDON_KZ_END = 10
NY_KZ_START = 12
NY_KZ_END = 15
ASIAN_START = 0
ASIAN_END = 8

# ── ENVIRONMENT ──────────────────────────────────────────────────
POSITIONS = [-1, 0, 1]
TRADING_FEES = 0.0003  # 3 pips — DO NOT CHANGE
BORROW_INTEREST_RATE = 0.000003
PORTFOLIO_INITIAL_VALUE = 10_000
MAX_EPISODE_DURATION = 4000
INITIAL_POSITION = "random"

# ── RISK ─────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.15
RISK_MIN = 0.05
RISK_MAX = 0.15
MIN_RR_RATIO = 3.0
CATASTROPHIC_SL_ATR = 5.0
SL_ATR_BUFFER = 0.1

# ── REWARD ───────────────────────────────────────────────────────
REWARD_FUNCTION = "account_flip_aggressive"  # DO NOT CHANGE

FLIP_RRR_MULTIPLIER = 4.0  # DO NOT CHANGE
FLIP_LOG_SCALE = 10.0
FLIP_INACTION_PENALTY = -0.000025  # was -0.00005, halved for 4000-step episodes
FLIP_FLAT_WINDOW = 20
FLIP_FLAT_ESCALATION = 3.0
FLIP_FREQ_BONUS_WEIGHT = 0.0005
FLIP_WIN_BONUS_THRESHOLD = 0.02
FLIP_WIN_BONUS_SCALE = 5.0
FLIP_DD_PENALTY_WEIGHT = 2.0
FLIP_DD_FREE_ZONE = 0.08

# ── PARALLELISATION ───────────────────────────────────────────────
N_ENVS = 6

# ── TRAINING ─────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000  # Phase 2 fine-tune budget
ALGORITHM = "RecurrentPPO"
CHECKPOINT_FREQ = 100_000
MODEL_SAVE_PATH = "models/forex_ppo"
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
    "learning_rate": 0.00003,
    "clip_range": 0.08,
}

# ── EVALUATION ───────────────────────────────────────────────────
BASE_TF = "M15"
TRAIN_TEST_SPLIT = 0.80
WFO_FOLDS = 5
