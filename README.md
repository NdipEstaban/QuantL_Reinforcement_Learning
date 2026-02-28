# Forex RL Trading Bot — ICT/SMC Feature Based
## Built on gym-trading-env + Stable-Baselines3

A reinforcement learning trading bot using ICT/SMC market structure
concepts as features: STH/STL/ITH/ITL/LTH/LTL, Order Blocks,
Breaker Blocks, Fair Value Gaps, BOS, CHoCH.

---

## Project Structure

```
forex_rl_bot/
├── run.py                     ← SINGLE ENTRY POINT (start here)
├── config.py                  ← ALL settings — edit before running
├── requirements.txt
│
├── data/
│   ├── raw/                   ← DROP YOUR CSV HERE
│   └── processed/             ← Auto-generated after running features
│
├── features/
│   ├── pipeline.py            ← Master pipeline (calls all below)
│   ├── structure.py           ← STH/STL/ITH/ITL/LTH/LTL + BOS/CHoCH/Bias
│   ├── zones.py               ← FVG, Order Block, Breaker Block
│   └── session.py             ← Sessions + killzones
│
├── env/
│   └── trading_env.py         ← gym-trading-env wrapper + position sizing
│
├── training/
│   ├── train.py               ← Training script
│   ├── callbacks.py           ← SB3 callbacks
│   └── reward.py              ← Reward functions
│
├── evaluation/
│   ├── backtest.py            ← Backtesting + walk-forward validation
│   └── metrics.py             ← Sharpe, drawdown, win rate, etc.
│
├── visualization/
│   ├── dashboard.py           ← Interactive Dash dashboard
│   └── charts.py              ← Individual Plotly chart functions
│
├── models/                    ← Trained models saved here
└── logs/                      ← TensorBoard logs
```

---

## SETUP INSTRUCTIONS

### 1. Create virtual environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your data

Drop your EURUSD CSV into `data/raw/`. It must have these columns:

```
date, open, high, low, close
```

Volume is optional. Datetime format is flexible (pandas will parse it).

Example: `data/raw/EURUSD_M15.csv`

```
date,open,high,low,close
2021-01-04 00:00:00,1.22485,1.22532,1.22378,1.22451
2021-01-04 00:15:00,1.22451,1.22580,1.22430,1.22561
```

### 4. Configure

Open `config.py` and update at minimum:

```python
DATA_FILE       = "data/raw/YOUR_FILE.csv"
DATE_COLUMN     = "date"          # your datetime column name
BASE_TF         = "M15"           # or "H1"
BROKER_TIMEZONE = "Etc/GMT-3"     # your broker's server timezone
```

---

## RUNNING THE PROJECT

Everything runs through `run.py`:

### Step 1: Process features
```bash
python run.py features
```
Reads your CSV, computes all 39 ICT/SMC features, saves to
`data/processed/features.pkl`. Takes 1-5 minutes depending on data size.

### Step 2: Train the model
```bash
python run.py train
```
Trains PPO for 500k steps (configurable). Saves checkpoints to `models/`.
Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
```

### Step 3: Backtest
```bash
# Simple train/test split backtest
python run.py backtest

# Walk-forward validation (more rigorous)
python run.py backtest --wfo
```

### Step 4: Dashboard
```bash
python run.py dashboard
```
Opens interactive dashboard at http://127.0.0.1:8050

### Step 5: Forward test
```bash
python run.py forwardtest --data data\unseen\EURUSD_2024_M15_fixed.csv
```

### Run everything at once
```bash
python run.py all
```
### Render performance
# Step 1 — run and save logs
python run.py backtest --render
python run.py forwardtest --data 2025.csv --render

# Step 2 — open browser viewer
python run.py render --render-dir render_logs/test
python run.py render --render-dir render_logs/forwardtest
---

## FEATURE DEFINITIONS (exact ICT definitions used)

### Structure Levels

| Level | Definition |
|-------|-----------|
| STH | Swing high with NO other highs to its left OR right within lookback |
| STL | Swing low with NO other lows to its left OR right within lookback |
| ITH | Has a STH to its LEFT AND RIGHT — OR — rebalanced an FVG by ≥80% |
| ITL | Has a STL to its LEFT AND RIGHT — OR — rebalanced an FVG by ≥80% |
| LTH | Has an ITH to its LEFT AND RIGHT |
| LTL | Has an ITL to its LEFT AND RIGHT |

### Events

| Event | Definition |
|-------|-----------|
| BOS (bull) | Price closes ABOVE an ITH |
| BOS (bear) | Price closes BELOW an ITL |
| CHoCH (bull) | Price closes ABOVE an LTH |
| CHoCH (bear) | Price closes BELOW an LTL |

### Zones

| Zone | Definition |
|------|-----------|
| Bullish OB | Lowest down-close candle near ITL/LTL with best body-to-wick ratio before bullish displacement + BOS |
| Bearish OB | Highest up-close candle near ITH/LTH before bearish displacement + BOS |
| Breaker | Former OB that price has broken through after a stop run + structure shift |
| FVG | 3-candle imbalance (gap between C1 high and C3 low) |

---

## FEATURES FED TO MODEL (39 total — no raw prices)

```
# Distances (all normalised by ATR)
feature_dist_to_sth, feature_dist_to_stl
feature_dist_to_ith, feature_dist_to_itl
feature_dist_to_lth, feature_dist_to_ltl

# Range positions (0.0=at low, 1.0=at high)
feature_it_range_position, feature_lt_range_position

# FVG
feature_fvg_active, feature_fvg_direction
feature_fvg_rebalance_pct, feature_fvg_position, feature_dist_to_fvg

# Order Block
feature_ob_active, feature_ob_direction
feature_ob_position, feature_ob_50_dist, feature_ob_correct_side

# Breaker Block
feature_bb_active, feature_bb_direction
feature_bb_position, feature_bb_50_dist, feature_bb_correct_side

# BOS (6 directional+levelled)
feature_bos_bull_st, feature_bos_bear_st
feature_bos_bull_it, feature_bos_bear_it
feature_bos_bull_lt, feature_bos_bear_lt

# CHoCH + active flags
feature_choch_bull, feature_choch_bear
feature_choch_bull_active, feature_choch_bear_active

# Bias (persistent state)
feature_st_bias, feature_it_bias, feature_lt_bias, feature_bias_aligned

# Session
feature_session, feature_killzone_active
```

---

## ADDING NEW FEATURES

1. Add your calculation to the appropriate module in `features/`
2. Column name MUST start with `feature_`
3. Value MUST be relative (no raw prices):
   - Distances: `(level - close) / atr`
   - Zone positions: `(close - zone_low) / (zone_high - zone_low)` → 0.0→1.0
   - Binary flags: 0 or 1
4. gym-trading-env automatically picks up all `feature_` columns

---

## RECOMMENDED WORKFLOW

```
Week 1:   Run pipeline. Visually inspect structure chart.
          Does STH/STL/ITH/ITL detection match your chart?
          Tune SWING_LOOKBACK in config.py if needed.

Week 2:   First training run (100k steps — fast).
          Is reward increasing? Check TensorBoard.

Week 3:   Walk-forward backtest.
          Does it generalise out-of-sample?

Month 2+: Paper trade via MT5 socket bridge.
          Never live trade until 1+ month paper trade is clean.
```

---

## MT5 DEPLOYMENT (after validation)

The trained model outputs actions (-1/0/+1).
Connect to MT5 via a Flask REST API:

```python
# Python server
from flask import Flask, request, jsonify
from stable_baselines3 import PPO
app = Flask(__name__)
model = PPO.load("models/forex_ppo/final_model")

@app.route("/predict", methods=["POST"])
def predict():
    obs = request.json["observation"]
    action, _ = model.predict(obs, deterministic=True)
    return jsonify({"action": int(action)})
```

```mql5
// MQL5 EA
string response = HTTPPost("http://localhost:5000/predict", obs_json);
int action = ParseAction(response);  // -1, 0, or 1
```