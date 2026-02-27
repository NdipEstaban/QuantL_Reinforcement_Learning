"""
config.py
=========
All project settings. Edit this file before running anything.
"""

# ── DATA ─────────────────────────────────────────────────────────
DATA_FILE = "data/raw/EURUSD_2022_M15.csv"
DATE_COLUMN     = "date"          # name of datetime column in your CSV
OPEN_COLUMN     = "open"
HIGH_COLUMN     = "high"
LOW_COLUMN      = "low"
CLOSE_COLUMN    = "close"
VOLUME_COLUMN   = "volume"        # set to None if no volume

# Broker timezone → will be converted to UTC for session features
# "Etc/GMT-2" = UTC+2 | "Etc/GMT-3" = UTC+3 | "UTC" = already UTC
BROKER_TIMEZONE = "Etc/GMT-3" #TODO: CHANGE TO UTC as current data is in utc

# ── FEATURE ENGINEERING ──────────────────────────────────────────

# Candles each side required to confirm a swing point (STH/STL)
SWING_LOOKBACK = 3

# ATR period for all normalisation
ATR_PERIOD = 14

# ATR multiples a move must cover to qualify as displacement
DISPLACEMENT_ATR_MULT = 1.5

# FVG fill % required to promote a swing to ITH/ITL
FVG_QUALIFY_THRESHOLD = 0.80

# How many candles CHoCH / BOS active flags stay "hot" after firing
CHOCH_ACTIVE_WINDOW = 20
BOS_ACTIVE_WINDOW   = 10

# Session times in UTC hours
LONDON_KZ_START = 7    # London killzone open
LONDON_KZ_END   = 10   # London killzone close
NY_KZ_START     = 12   # NY killzone open
NY_KZ_END       = 15   # NY killzone close
ASIAN_START     = 0
ASIAN_END       = 8

# ── ENVIRONMENT ──────────────────────────────────────────────────
POSITIONS               = [-1, 0, 1]   # short / flat / long
TRADING_FEES            = 0.0001       # ~1 pip spread
BORROW_INTEREST_RATE    = 0.000003     # per step, for shorts
PORTFOLIO_INITIAL_VALUE = 10_000
MAX_EPISODE_DURATION    = 2000         # candles per episode
INITIAL_POSITION        = "random"

# ── RISK ─────────────────────────────────────────────────────────
RISK_PER_TRADE = 0.15  # 15% — maximum aggression
RISK_MIN = 0.05
RISK_MAX = 0.15
MIN_RR_RATIO = 3.0
CATASTROPHIC_SL_ATR = 5.0
SL_ATR_BUFFER = 0.1

# ── REWARD ───────────────────────────────────────────────────────
REWARD_FUNCTION = "account_flip_aggressive"

FLIP_RRR_MULTIPLIER = 3.0
FLIP_LOG_SCALE = 10.0
FLIP_INACTION_PENALTY = -0.0002
FLIP_FLAT_WINDOW = 20  # candles before tier escalates
FLIP_FLAT_ESCALATION = 3.0
FLIP_DD_FREE_ZONE = 0.05
FLIP_DD_PENALTY_WEIGHT = 5.0
FLIP_FREQ_BONUS_WEIGHT = 0.0005  # new — rewards trade frequency
FLIP_WIN_BONUS_THRESHOLD = 0.02
FLIP_WIN_BONUS_SCALE = 5.0

# ── PARALLELISATION ───────────────────────────────────────────────
N_ENVS = 6  # parallel environments
# increase to 12 if RAM allows (each env loads feature data)
# decrease to 4 if you get memory errors

# ── TRAINING ─────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 3_000_000
ALGORITHM = "PPO"
CHECKPOINT_FREQ = 100_000
MODEL_SAVE_PATH = "models/forex_ppo"
LOG_DIR = "logs/"

PPO_CONFIG = {
    "learning_rate": 0.0001,
    "n_steps": 2048,  # steps per env per iteration
    # total per iter = 2048 × 8 = 16,384
    "batch_size": 512,  # was 256 — scale with n_envs
    # rule of thumb: batch_size = n_envs × 64
    "n_epochs": 5,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.05,
}

# ── EVALUATION ───────────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.80   # first 80% train, last 20% test
WFO_FOLDS        = 5      # walk-forward folds
