//+------------------------------------------------------------------+
//|  ForexRL_EA.mq5                                                  |
//|  RPPO-powered Expert Advisor — v2.1                              |
//|  Model-exit mode: model controls all exits, catastrophic SL only |
//+------------------------------------------------------------------+
#property copyright "ForexRL"
#property version   "2.10"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+
input group "=== Bridge Connection ==="
input string   BridgeURL        = "http://127.0.0.1:8000";  // FastAPI URL
input string   SessionID        = "mt5_session_01";          // Unique session ID
input int      CandleHistory    = 300;                        // Candles to send per request
input int      RequestTimeoutMs = 5000;                       // HTTP timeout (ms)

input group "=== Model Selection ==="
input string   ModelAlgo        = "rppo";  // Algorithm: rppo, ppo, a2c
input string   ModelVersion     = "v1";    // Version: v1, v2, …

input group "=== Trading ==="
input string   BaseSymbol       = "EURUSD";      // Base symbol to search for (e.g. EURUSD, BTCUSD)
input ENUM_TIMEFRAMES Timeframe = PERIOD_M15;    // Timeframe
input double   StartingBalance  = 10000.0;       // Starting balance ($) for ratchet
input bool     TradeLongsOnly   = false;         // Only take long signals
input bool     TradeShortsOnly  = false;         // Only take short signals
input bool     KillzoneOnly     = false;         // Only trade in killzone

input group "=== Risk Override ==="
input bool     UseManualRisk    = false;         // Override API risk %
input double   ManualRiskPct    = 15.0;          // Manual risk % (if override)
input double   MaxSpreadPips    = 3.0;           // Skip trade if spread > this

input group "=== Session Control ==="
input int      LondonStart      = 7;             // London killzone start (UTC hr)
input int      LondonEnd        = 10;            // London killzone end   (UTC hr)
input int      NYStart          = 12;            // NY killzone start     (UTC hr)
input int      NYEnd            = 15;            // NY killzone end       (UTC hr)

input group "=== Exit Mode ==="
input bool     UseModelExit     = true;          // true=model controls exit (no TP, wide SL)
                                                  // false=fixed SL/TP from bridge
input double   CatastrophicATR  = 10.0;          // ATR multiplier for catastrophic SL
                                                  // (model-exit mode only — black swan guard)

input group "=== Debug ==="
input bool     VerboseLogging   = true;          // Print detailed logs
input bool     SignalOnly       = false;         // Signal log only, no execution

//+------------------------------------------------------------------+
//| GLOBALS                                                          |
//+------------------------------------------------------------------+
CTrade    Trade;
string    Symbol_       = "";   // Resolved at OnInit() — do not edit
string    LastSignalID  = "";
int       LastAction    = 0;
datetime  LastBarTime   = 0;
bool      IsReset       = true;

// Performance tracking
int    TotalTrades  = 0;
int    WinTrades    = 0;
double GrossWin     = 0.0;
double GrossLoss    = 0.0;

//+------------------------------------------------------------------+
//| EA INIT                                                          |
//+------------------------------------------------------------------+
int OnInit() {
   Trade.SetExpertMagicNumber(202501);
   Trade.SetDeviationInPoints(10);
   Trade.SetTypeFilling(ORDER_FILLING_IOC);
   Trade.LogLevel(LOG_LEVEL_ERRORS);

   IsReset = true;

   // ── Auto-detect broker symbol name ────────────────────────────
   // Brokers append suffixes: EURUSDz, EURUSDpro, EURUSD., EURUSD+, etc.
   // Scan Market Watch for any symbol containing BaseSymbol string.
   // Falls back to the current chart symbol if nothing found.
   Symbol_ = DetectSymbol(BaseSymbol);
   if (Symbol_ == "") {
      PrintFormat("[ForexRL] WARNING: Could not find '%s' in Market Watch — "
                  "using chart symbol '%s'", BaseSymbol, _Symbol);
      Symbol_ = _Symbol;
   }

   PrintFormat("[ForexRL] EA v2.0 | Session: %s | URL: %s | Model: %s/v%s | Symbol: %s",
               SessionID, BridgeURL, ModelAlgo, ModelVersion, Symbol_);

   if (!PingBridge()) {
      Alert("[ForexRL] WARNING: Bridge not reachable at " + BridgeURL +
            "\nEA loaded but will not trade until bridge is online.");
   }

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| AUTO-DETECT BROKER SYMBOL                                        |
//| Scans Market Watch for any symbol containing BaseSymbol string.  |
//| Priority: 1. exact match  2. shortest name (least broker suffix) |
//+------------------------------------------------------------------+
string DetectSymbol(string base) {
   string best    = "";
   int    bestLen = 999;
   int    total   = SymbolsTotal(true);   // true = Market Watch only

   for (int i = 0; i < total; i++) {
      string sym      = SymbolName(i, true);
      string symUpper = sym;
      string baseUpper= base;
      StringToUpper(symUpper);
      StringToUpper(baseUpper);

      if (StringFind(symUpper, baseUpper) >= 0) {
         if (symUpper == baseUpper) {
            PrintFormat("[ForexRL] Symbol exact match: '%s'", sym);
            return sym;   // exact match — done immediately
         }
         if (StringLen(sym) < bestLen) {
            best    = sym;
            bestLen = StringLen(sym);
         }
      }
   }

   if (best != "")
      PrintFormat("[ForexRL] Symbol fuzzy match: '%s' (base: '%s')", best, base);

   return best;
}

//+------------------------------------------------------------------+
//| EA DEINIT                                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   PrintFormat("[ForexRL] EA removed. Trades: %d | Wins: %d | "
               "GrossWin: %.4f%% | GrossLoss: %.4f%%",
               TotalTrades, WinTrades, GrossWin * 100, GrossLoss * 100);
}

//+------------------------------------------------------------------+
//| MAIN TICK                                                        |
//+------------------------------------------------------------------+
void OnTick() {
   datetime barTime = iTime(Symbol_, Timeframe, 0);
   if (barTime == LastBarTime) return;
   LastBarTime = barTime;

   CheckOpenPositions();

   if (KillzoneOnly && !IsInKillzone()) {
      if (VerboseLogging)
         PrintFormat("[ForexRL] Outside killzone — skipping %s",
                     TimeToString(barTime, TIME_DATE|TIME_MINUTES));
      return;
   }

   double spread = GetSpreadPips();
   if (spread > MaxSpreadPips) {
      if (VerboseLogging)
         PrintFormat("[ForexRL] Spread %.1f > max %.1f — skipping", spread, MaxSpreadPips);
      return;
   }

   string response = "";
   if (!RequestPrediction(response)) return;

   int    action            = 0;
   double riskPct           = 0.15;
   double slPips            = 30.0;
   double tpPips            = 90.0;
   double catSlPips         = 200.0;
   double catSlPrice        = 0.0;
   double atrPips           = 10.0;
   double confidence        = 0.5;
   int    killzoneActive    = 0;
   string signalID          = "";
   double slPrice           = 0.0;
   string slType            = "";

   if (!ParsePrediction(response, action, riskPct, slPips, tpPips,
                        catSlPips, catSlPrice, atrPips, confidence,
                        killzoneActive, signalID, slPrice, slType)) {
      Print("[ForexRL] Failed to parse bridge response");
      return;
   }

   if (VerboseLogging) {
      if (UseModelExit)
         PrintFormat("[ForexRL] Signal | Action: %+d | Risk: %.0f%% | "
                     "CatSL: %.1fpips(%.5f) | Conf: %.2f | KZ: %d | ID: %s [MODEL EXIT]",
                     action, riskPct * 100, catSlPips, catSlPrice,
                     confidence, killzoneActive, signalID);
      else
         PrintFormat("[ForexRL] Signal | Action: %+d | Risk: %.0f%% | "
                     "SL: %.1fpips(%.5f/%s) | TP: %.1f | Conf: %.2f | KZ: %d | ID: %s",
                     action, riskPct * 100, slPips, slPrice, slType,
                     tpPips, confidence, killzoneActive, signalID);
   }

   if (action ==  1 && TradeShortsOnly) return;
   if (action == -1 && TradeLongsOnly)  return;

   if (UseManualRisk) riskPct = ManualRiskPct / 100.0;

   if (!SignalOnly) {
      ExecuteSignal(action, riskPct, slPips, tpPips,
                    catSlPips, catSlPrice, signalID, slPrice);
   }

   LastAction   = action;
   LastSignalID = signalID;
   IsReset      = false;
}

//+------------------------------------------------------------------+
//| REQUEST PREDICTION FROM BRIDGE                                   |
//+------------------------------------------------------------------+
bool RequestPrediction(string &response) {
   string candlesJSON = BuildCandlesJSON(CandleHistory);
   if (candlesJSON == "") return false;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   int    pos     = GetCurrentPosition();

   string body = StringFormat(
      "{"
        "\"session_id\":\"%s\","
        "\"symbol\":\"%s\","
        "\"candles\":%s,"
        "\"current_position\":%d,"
        "\"account_balance\":%.2f,"
        "\"starting_balance\":%.2f,"
        "\"reset_lstm\":%s"
      "}",
      SessionID, Symbol_, candlesJSON, pos,
      balance, StartingBalance,
      IsReset ? "true" : "false"
   );

   // ── Versioned endpoint: /{algo}/v{version}/predict ────────────
   string url     = BridgeURL + "/" + ModelAlgo + "/v" + ModelVersion + "/predict";
   string headers = "Content-Type: application/json\r\n";
   char   bodyArr[]; StringToCharArray(body, bodyArr, 0, StringLen(body));
   char   resultArr[];
   string resultHeaders;

   ResetLastError();
   int code = WebRequest("POST", url, headers, RequestTimeoutMs,
                         bodyArr, resultArr, resultHeaders);

   if (code != 200) {
      if (VerboseLogging)
         PrintFormat("[ForexRL] Bridge HTTP %d @ %s | err=%d", code, url, GetLastError());
      return false;
   }

   response = CharArrayToString(resultArr);
   return true;
}

//+------------------------------------------------------------------+
//| BUILD OHLCV JSON ARRAY                                           |
//+------------------------------------------------------------------+
string BuildCandlesJSON(int count) {
   int bars = MathMin(count, iBars(Symbol_, Timeframe) - 1);
   if (bars < 50) {
      Print("[ForexRL] Not enough bars: ", bars);
      return "";
   }

   string json = "[";
   for (int i = bars; i >= 1; i--) {
      datetime t = iTime  (Symbol_, Timeframe, i);
      double   o = iOpen  (Symbol_, Timeframe, i);
      double   h = iHigh  (Symbol_, Timeframe, i);
      double   l = iLow   (Symbol_, Timeframe, i);
      double   c = iClose (Symbol_, Timeframe, i);
      long     v = iVolume(Symbol_, Timeframe, i);

      json += StringFormat(
         "{\"time\":%d,\"open\":%.5f,\"high\":%.5f,"
          "\"low\":%.5f,\"close\":%.5f,\"volume\":%d}",
         (int)t, o, h, l, c, (int)v
      );
      if (i > 1) json += ",";
   }
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| PARSE JSON RESPONSE                                              |
//+------------------------------------------------------------------+
bool ParsePrediction(const string &resp,
                     int    &action,    double &riskPct,
                     double &slPips,    double &tpPips,
                     double &catSlPips, double &catSlPrice,
                     double &atrPips,   double &confidence,
                     int    &killzone,  string &signalID,
                     double &slPrice,   string &slType) {
   action      = (int)ParseJSONInt   (resp, "action");
   riskPct     =      ParseJSONDouble(resp, "risk_pct");
   slPips      =      ParseJSONDouble(resp, "sl_pips");
   tpPips      =      ParseJSONDouble(resp, "tp_pips");
   catSlPips   =      ParseJSONDouble(resp, "catastrophic_sl_pips");
   catSlPrice  =      ParseJSONDouble(resp, "catastrophic_sl_price");
   atrPips     =      ParseJSONDouble(resp, "atr_pips");
   confidence  =      ParseJSONDouble(resp, "confidence");
   killzone    = (int)ParseJSONInt   (resp, "killzone_active");
   signalID    =      ParseJSONString(resp, "signal_id");
   slPrice     =      ParseJSONDouble(resp, "sl_price");
   slType      =      ParseJSONString(resp, "sl_type");
   return true;
}

//+------------------------------------------------------------------+
//| EXECUTE SIGNAL                                                   |
//|                                                                  |
//| UseModelExit=true  → No TP. Model exits by switching action.    |
//|   Only wide catastrophic SL set as black-swan guard.            |
//| UseModelExit=false → Fixed SL/TP from bridge (structure-based). |
//+------------------------------------------------------------------+
void ExecuteSignal(int action, double riskPct,
                   double slPips,    double tpPips,
                   double catSlPips, double catSlPrice,
                   string signalID,  double slPriceFixed = 0.0) {

   bool hasLong  = PositionSelect(Symbol_) &&
                   PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
   bool hasShort = PositionSelect(Symbol_) &&
                   PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL;

   double pip = GetPipSize();
   double price, sl, tp, lots;

   // ── MODEL EXIT MODE ────────────────────────────────────────────
   // The model was trained without SL/TP — it learned to hold positions
   // through drawdowns and exit when structure invalidates. Replicating
   // that here gives the model full control, matching training exactly.
   if (UseModelExit) {

      // action=0 → model says flat: close any open position
      if (action == 0) {
         if (hasLong || hasShort) {
            CloseAllPositions();
            PrintFormat("[ForexRL] CLOSE | Model exit (flat signal) | %s", signalID);
         }
         return;
      }

      // Direction flip: close existing before opening opposite
      if (action ==  1 && hasShort) {
         CloseAllPositions();
         PrintFormat("[ForexRL] FLIP SHORT→LONG | Model exit triggered close | %s", signalID);
         hasShort = false;
      }
      if (action == -1 && hasLong) {
         CloseAllPositions();
         PrintFormat("[ForexRL] FLIP LONG→SHORT | Model exit triggered close | %s", signalID);
         hasLong = false;
      }

      // Open LONG — no TP, catastrophic SL only
      if (action == 1 && !hasLong) {
         price = SymbolInfoDouble(Symbol_, SYMBOL_ASK);
         sl    = (catSlPrice > 0 && catSlPrice < price)
                  ? catSlPrice
                  : price - catSlPips * pip;
         tp    = 0;   // model controls exit — no TP
         lots  = CalculateLots(riskPct, MathAbs(price - sl) / pip);
         if (lots <= 0) return;
         if (Trade.Buy(lots, Symbol_, price, sl, tp, "ForexRL|" + signalID)) {
            TotalTrades++;
            PrintFormat("[ForexRL] BUY  [model-exit] | lots=%.2f | cat_sl=%.5f | no TP | %s",
                        lots, sl, signalID);
         }
      }

      // Open SHORT — no TP, catastrophic SL only
      if (action == -1 && !hasShort) {
         price = SymbolInfoDouble(Symbol_, SYMBOL_BID);
         sl    = (catSlPrice > 0 && catSlPrice > price)
                  ? catSlPrice
                  : price + catSlPips * pip;
         tp    = 0;   // model controls exit — no TP
         lots  = CalculateLots(riskPct, MathAbs(sl - price) / pip);
         if (lots <= 0) return;
         if (Trade.Sell(lots, Symbol_, price, sl, tp, "ForexRL|" + signalID)) {
            TotalTrades++;
            PrintFormat("[ForexRL] SELL [model-exit] | lots=%.2f | cat_sl=%.5f | no TP | %s",
                        lots, sl, signalID);
         }
      }

   // ── FIXED SL/TP MODE ───────────────────────────────────────────
   } else {

      if (action ==  1 && hasShort) CloseAllPositions();
      if (action == -1 && hasLong)  CloseAllPositions();

      if (action == 1 && !hasLong) {
         price = SymbolInfoDouble(Symbol_, SYMBOL_ASK);
         sl    = (slPriceFixed > 0 && slPriceFixed < price)
                  ? slPriceFixed : price - slPips * pip;
         tp    = price + tpPips * pip;
         lots  = CalculateLots(riskPct, MathAbs(price - sl) / pip);
         if (lots <= 0) return;
         if (Trade.Buy(lots, Symbol_, price, sl, tp, "ForexRL|" + signalID)) {
            TotalTrades++;
            PrintFormat("[ForexRL] BUY  | lots=%.2f | sl=%.5f | tp=%.5f | %s",
                        lots, sl, tp, signalID);
         }
      }

      if (action == -1 && !hasShort) {
         price = SymbolInfoDouble(Symbol_, SYMBOL_BID);
         sl    = (slPriceFixed > 0 && slPriceFixed > price)
                  ? slPriceFixed : price + slPips * pip;
         tp    = price - tpPips * pip;
         lots  = CalculateLots(riskPct, MathAbs(sl - price) / pip);
         if (lots <= 0) return;
         if (Trade.Sell(lots, Symbol_, price, sl, tp, "ForexRL|" + signalID)) {
            TotalTrades++;
            PrintFormat("[ForexRL] SELL | lots=%.2f | sl=%.5f | tp=%.5f | %s",
                        lots, sl, tp, signalID);
         }
      }

      if (action == 0) CloseAllPositions();
   }
}

//+------------------------------------------------------------------+
//| CHECK OPEN POSITIONS — HistorySelect() FIX                       |
//+------------------------------------------------------------------+
void CheckOpenPositions() {
   // CRITICAL: HistorySelect() MUST be called before HistoryDealsTotal().
   // Without it, MT5 returns empty history on most brokers and closed trades
   // are NEVER detected → bridge never gets /trade/close notifications.
   if (!HistorySelect(0, TimeCurrent())) {
      if (VerboseLogging) Print("[ForexRL] HistorySelect() failed");
      return;
   }

   int      histTotal    = HistoryDealsTotal();
   static datetime lastHistCheck = 0;

   for (int i = histTotal - 1; i >= 0; i--) {
      ulong ticket = HistoryDealGetTicket(i);
      if (!HistoryDealSelect(ticket)) continue;

      datetime closeTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      if (closeTime <= lastHistCheck) break;

      if (HistoryDealGetString(ticket, DEAL_SYMBOL) != Symbol_) continue;

      // Only process exit deals
      if ((ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      string comment = HistoryDealGetString(ticket, DEAL_COMMENT);
      if (StringFind(comment, "ForexRL|") < 0) continue;

      double profit  = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double pnlPct  = (balance > 0) ? profit / balance : 0;
      string sigID   = StringSubstr(comment, 8);

      string reason = "manual";
      if (profit > 0)  reason = "tp";
      else if (profit < 0) reason = "sl";

      if (profit > 0) { WinTrades++; GrossWin  += pnlPct; }
      else              GrossLoss += MathAbs(pnlPct);

      NotifyBridgeClose(sigID, pnlPct, reason);

      PrintFormat("[ForexRL] Trade closed | %s | P&L: %.4f%% | %s",
                  sigID, pnlPct * 100, reason);
   }

   if (histTotal > 0) {
      ulong last = HistoryDealGetTicket(histTotal - 1);
      if (HistoryDealSelect(last))
         lastHistCheck = (datetime)HistoryDealGetInteger(last, DEAL_TIME);
   }
}

//+------------------------------------------------------------------+
//| NOTIFY BRIDGE — TRADE CLOSE                                      |
//+------------------------------------------------------------------+
void NotifyBridgeClose(string signalID, double pnlPct, string reason) {
   string body = StringFormat(
      "{\"session_id\":\"%s\","
       "\"signal_id\":\"%s\","
       "\"pnl_pct\":%.6f,"
       "\"close_reason\":\"%s\"}",
      SessionID, signalID, pnlPct, reason
   );

   string url     = BridgeURL + "/trade/close";
   string headers = "Content-Type: application/json\r\n";
   char   bodyArr[]; StringToCharArray(body, bodyArr, 0, StringLen(body));
   char   result[];
   string resHeaders;

   WebRequest("POST", url, headers, RequestTimeoutMs, bodyArr, result, resHeaders);
}

void NotifyBridgeEntry(string signalID, int action, double riskPct) {
   // Bridge already registers the signal from /predict — no extra call needed.
}

//+------------------------------------------------------------------+
//| HELPER: CALCULATE LOT SIZE FROM RISK %                          |
//+------------------------------------------------------------------+
double CalculateLots(double riskPct, double slPips) {
   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * riskPct;
   double tickValue  = SymbolInfoDouble(Symbol_, SYMBOL_TRADE_TICK_VALUE);
   double tickSize   = SymbolInfoDouble(Symbol_, SYMBOL_TRADE_TICK_SIZE);
   double pip        = GetPipSize();

   if (slPips <= 0 || tickValue <= 0) return 0;

   double pipValue = tickValue * (pip / tickSize);
   double lots     = riskAmount / (slPips * pipValue);

   double minLot  = SymbolInfoDouble(Symbol_, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(Symbol_, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(Symbol_, SYMBOL_VOLUME_STEP);

   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| HELPER: CLOSE ALL POSITIONS                                      |
//+------------------------------------------------------------------+
void CloseAllPositions() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (PositionGetString(POSITION_SYMBOL) == Symbol_) {
         Trade.PositionClose(ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| HELPER: CURRENT POSITION (-1/0/1)                               |
//+------------------------------------------------------------------+
int GetCurrentPosition() {
   if (!PositionSelect(Symbol_)) return 0;
   int type = (int)PositionGetInteger(POSITION_TYPE);
   return type == POSITION_TYPE_BUY ? 1 : -1;
}

//+------------------------------------------------------------------+
//| HELPER: PIP SIZE                                                 |
//+------------------------------------------------------------------+
double GetPipSize() {
   int digits = (int)SymbolInfoInteger(Symbol_, SYMBOL_DIGITS);
   return (digits == 3 || digits == 5)
          ? 10 * SymbolInfoDouble(Symbol_, SYMBOL_POINT)
          : SymbolInfoDouble(Symbol_, SYMBOL_POINT);
}

//+------------------------------------------------------------------+
//| HELPER: SPREAD IN PIPS                                          |
//+------------------------------------------------------------------+
double GetSpreadPips() {
   double ask = SymbolInfoDouble(Symbol_, SYMBOL_ASK);
   double bid = SymbolInfoDouble(Symbol_, SYMBOL_BID);
   return (ask - bid) / GetPipSize();
}

//+------------------------------------------------------------------+
//| HELPER: KILLZONE CHECK                                           |
//+------------------------------------------------------------------+
bool IsInKillzone() {
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int h = dt.hour;
   return (h >= LondonStart && h < LondonEnd) ||
          (h >= NYStart      && h < NYEnd);
}

//+------------------------------------------------------------------+
//| HELPER: PING BRIDGE                                             |
//+------------------------------------------------------------------+
bool PingBridge() {
   string url  = BridgeURL + "/health";
   char   body[], result[];
   string headers, resHeaders;
   int code = WebRequest("GET", url, headers, 2000, body, result, resHeaders);
   return code == 200;
}

//+------------------------------------------------------------------+
//| HELPER: JSON PARSERS                                             |
//+------------------------------------------------------------------+
double ParseJSONDouble(const string &json, const string &key) {
   string search = "\"" + key + "\":";
   int    start  = StringFind(json, search);
   if (start < 0) return 0.0;
   start += StringLen(search);
   while (start < StringLen(json) &&
          (StringGetCharacter(json, start) == ' ' ||
           StringGetCharacter(json, start) == '\t')) start++;
   int end = start;
   while (end < StringLen(json)) {
      ushort c = StringGetCharacter(json, end);
      if (c == ',' || c == '}' || c == ']') break;
      end++;
   }
   return StringToDouble(StringSubstr(json, start, end - start));
}

int ParseJSONInt(const string &json, const string &key) {
   return (int)ParseJSONDouble(json, key);
}

string ParseJSONString(const string &json, const string &key) {
   string search = "\"" + key + "\":\"";
   int    start  = StringFind(json, search);
   if (start < 0) return "";
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if (end < 0) return "";
   return StringSubstr(json, start, end - start);
}
//+------------------------------------------------------------------+