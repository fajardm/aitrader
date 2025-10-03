"""
LLM + Backtest (Daily timeframe) — using investiny for data fetching
--------------------------------------------------------------------
- Fetch OHLCV from investiny (with yfinance fallback for compatibility).
- Compute features (EMA/RSI/ATR/Bollinger + regime classification).
- Ask an LLM (Groq) for entry/SL/TP in JSON **or** use rule-based fallback.
- Simulate next-day limit/market execution, SL/TP handling, equity curve.
- Report key metrics (trades, win rate, expectancy, profit factor, max DD).
- Optional matplotlib plots.
- **Includes offline unit tests** (no internet needed).

How to run (no internet):
  pip install pandas numpy matplotlib
  python main.py --equity 100000000 --risk 1.0 --plots 1

Optional (with internet):
  pip install investiny
  python main.py --tickers WIFI.JK --start 2020-01-01 --equity 100000000 --risk 1.0 --plots 1

Groq API (optional):
  # Script will call the LLM if available, otherwise fallback rules apply.

Run tests (offline):
  python main.py --run-tests
"""
from __future__ import annotations
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import argparse
import json
from investiny import search_assets, historical_data

class TickerParam:
    symbol: str
    name: str
    ema_short: int
    ema_long: int
    rsi: int
    investing_id: Optional[int]

    def __init__(self, symbol: str, name: str, ema_short: int, ema_long: int, rsi: int, investing_id: Optional[int] = None):
        self.symbol = symbol
        self.name = name
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi = rsi
        self.investing_id = investing_id

def load_ticker(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return [TickerParam(**item) for item in data]

# =========================
# Feature engineering utils
# =========================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(n).mean()
    roll_down = pd.Series(down, index=close.index).rolling(n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)  # neutralize early NaNs

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n).mean()

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    lower = mid - k * sd
    upper = mid + k * sd
    return mid, lower, upper


def classify_regime(df: pd.DataFrame) -> pd.Series:
    cond_up = (df['EMA_SHORT'] > df['EMA_LONG']) & (df['Close'] > df['EMA_LONG']) & (df['RSI'] >= 45)
    cond_range = (df['Close'].sub(df['BB_MID']).abs() / df['BB_MID'] < 0.05) & df['RSI'].between(35, 65)
    regime = np.where(cond_up, 'trend_up', np.where(cond_range, 'ranging', 'trend_down'))
    return pd.Series(regime, index=df.index)

# =========================
# Data loading (investiny primary, yfinance fallback)
# =========================

def ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        'date': 'Date', 'timestamp': 'Date',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'adj close': 'Adj Close', 'adj_close': 'Adj Close', 'volume': 'Volume'
    }
    df = df.copy()
    # Flatten multi-index columns if present
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [colmap.get(str(c).lower(), c) for c in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')

    if not {'Open','High','Low','Close','Volume'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: Date, Open, High, Low, Close, Volume")
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna()


def get_investiny_id(ticker: str) -> Optional[int]:
    """
    Search for a ticker symbol and return the best matching investiny ID.
    Prioritizes major exchanges like NASDAQ, NYSE, Jakarta for Indonesian stocks.
    """
    try:
        # Handle Indonesian stocks (.JK suffix)
        search_ticker = ticker.replace('.JK', '')
        results = search_assets(search_ticker)
        if not results:
            return None
        
        filtered = [r for r in results if r.get('exchange') == 'Jakarta']
        
        return int(filtered[0]['ticker'])
        
    except Exception as e:
        print(f"Error searching for ticker {ticker}: {e}")
        return None


def save_investing_id_to_cache(symbol: str, investing_id: int, issi_path: str = "issi.json"):
    """
    Save investing_id to issi.json for caching purposes.
    """
    try:
        # Load existing data
        with open(issi_path, 'r') as f:
            data = json.load(f)
        
        # Find and update the symbol
        for item in data:
            if item.get('symbol') == symbol:
                item['investing_id'] = investing_id
                break
        
        # Save back to file
        with open(issi_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved investing_id {investing_id} for {symbol}")
        
    except Exception as e:
        print(f"❌ Error saving investing_id for {symbol}: {e}")


def get_investing_id_from_cache(symbol: str, issi_path: str = "issi.json") -> Optional[int]:
    """
    Get investing_id from issi.json cache.
    """
    try:
        with open(issi_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            if item.get('symbol') == symbol and 'investing_id' in item:
                return item['investing_id']
        
        return None
        
    except Exception as e:
        print(f"❌ Error reading cache for {symbol}: {e}")
        return None


def get_or_fetch_investing_id(symbol: str, issi_path: str = "issi.json") -> Optional[int]:
    """
    Get investing_id from cache first, if not found then fetch from API.
    """
    # Try to get from cache first
    cached_id = get_investing_id_from_cache(symbol, issi_path)
    if cached_id:
        print(f"📦 Using cached investing_id {cached_id} for {symbol}")
        return cached_id
    
    # If not in cache, fetch from API
    print(f"🔍 Fetching investing_id for {symbol} from API...")
    investing_id = get_investiny_id(symbol)
    
    # Save to cache if found
    if investing_id:
        save_investing_id_to_cache(symbol, investing_id, issi_path)
    
    return investing_id


def init_investing_ids_for_tickers(tickers: List[str], issi_path: str = "issi.json"):
    """
    Initialize investing_id for specific tickers and save to cache.
    """
    print(f"🚀 Initializing investing_id for {len(tickers)} tickers...")
    
    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        investing_id = get_or_fetch_investing_id(ticker, issi_path)
        
        if investing_id:
            print(f"✅ {ticker} -> investing_id: {investing_id}")
        else:
            print(f"❌ Failed to get investing_id for {ticker}")
    
    print(f"\n🎉 Completed initialization for {len(tickers)} tickers!")


def load_ohlcv(ticker: str, start: Optional[str]) -> pd.DataFrame:
    """
    Load OHLCV data using investiny (primary) with caching support.
    """
    if not start:
        start = '2020-01-01'
    
    try:
        # Use cached investing_id system
        investiny_id = get_or_fetch_investing_id(ticker)
        if investiny_id:
            # Convert start date to investiny format (m/d/Y)
            start_date = pd.to_datetime(start).strftime('%m/%d/%Y')
            end_date = pd.Timestamp.now().strftime('%m/%d/%Y')
            
            data = historical_data(investiny_id, start_date, end_date)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to match expected schema
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open', 
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            df = df.sort_values('Date').set_index('Date')
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add Adj Close column (same as Close for investiny)
            df['Adj Close'] = df['Close']
            
            print(f"✓ Loaded {len(df)} records for {ticker} using investiny")
            return df.dropna()
            
    except Exception as e:
        print(f"Investiny failed for {ticker}: {e}")

# =========================
# LLM interface (pluggable)
# =========================

def render_prompt(row: pd.Series, ticker: str, risk_pct: float) -> str:
    return (
        f"Timeframe: 1D, Ticker: {ticker}, Date: {row.name.date()}\n"
        f"Close: {row.Close:.2f}\n"
        f"EMA_SHORT/LONG: {row.EMA_SHORT:.2f}, {row.EMA_LONG:.2f}\n"
        f"RSI: {row.RSI:.2f}, MACD_hist: {row.MACD_HIST:.4f}\n"
        f"ATR: {row.ATR:.2f}\n"
        f"Bollinger mid/low/up: {row.BB_MID:.2f}, {row.BB_LOW:.2f}, {row.BB_UP:.2f}\n"
        f"Regime: {row.Regime}\n"
        f"Kandidat: Pullback_EMA_SHORT {row.Pullback_EMA_SHORT:.2f}, Pullback_ATR {row.Pullback_ATR:.2f}\n"
        f"Risk per trade max: {risk_pct}%. Output JSON only with keys: regime, enter{{type,price}}, stop_loss, take_profit, position_size_pct, confidence, rationale."
    )

def safe_parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON blob if wrapped
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
        return None

def call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": "Bearer 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"}
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(url, headers=headers, json=data)
    return safe_parse_llm_json(resp.json().get('choices', [{}])[0].get('message', {}).get('content', ''))


def fallback_decision(row: pd.Series) -> Dict[str, Any]:
    regime = row.Regime
    if regime not in ("trend_up", "ranging"):
        return {"regime": "no_trade"}

    entry_price = round(min(row.Pullback_EMA_SHORT, row.Pullback_ATR))
    
    # Stop loss yang lebih ketat - berdasarkan persentase dari entry price
    if regime == "trend_up":
        # Trend up: SL 2-3% di bawah entry point
        sl_pct = 0.025  # 2.5%
    else:
        # Ranging: SL 1.5-2% di bawah entry point  
        sl_pct = 0.02   # 2%
    
    # SL minimum berdasarkan persentase, maksimum berdasarkan ATR
    sl_pct_based = round(entry_price * (1 - sl_pct))
    sl_atr_based = round(entry_price - 0.5 * row.ATR if not np.isnan(row.ATR) else entry_price * 0.98)
    
    # Gunakan yang lebih ketat (lebih tinggi) antara persentase dan ATR
    sl = max(sl_pct_based, sl_atr_based)
    
    r = max(entry_price - sl, 1e-6)
    # Single take profit untuk pemula - target 2R (risk-reward ratio 1:2)
    tp = round(entry_price + 2 * r)

    # position_size_pct: semakin kecil ATR, semakin besar size (maks 20%)
    base_size = 10.0
    atr_factor = max(1.0, min(2.0, row.ATR / row.Close * 100))  # ATR sebagai % dari harga
    position_size_pct = max(5.0, base_size * (2.0 - atr_factor / 2))  # range 5-20%

    # confidence: gabungan selisih EMA dan RSI
    ema_diff = row.EMA_SHORT - row.EMA_LONG
    confidence = 50 + min(40, max(0, ema_diff / row.Close * 100)) + min(10, max(0, (row.RSI - 50) / 5))

    # Percentage stop loss
    stop_loss_pct = round(100 * abs(entry_price - sl) / entry_price, 2) if entry_price != 0 else None

    # Percentage take profit
    take_profit_pct = round(100 * abs(tp - entry_price) / entry_price, 2) if entry_price != 0 else None

    return {
        "regime": regime,
        "enter": {"type": "limit", "price": entry_price},
        "stop_loss": sl,
        "stop_loss_pct": stop_loss_pct,
        "take_profit": tp,
        "take_profit_pct": take_profit_pct,
        "position_size_pct": round(position_size_pct, 2),
        "confidence": round(float(confidence), 2),
        "rationale": f"fallback {regime} pullback, 2R target"
    }

# =========================
# Backtest engine
# =========================

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_date: pd.Timestamp
    exit: float
    r_multiple: float


def position_qty(equity: float, entry: float, sl: float, risk_pct: float) -> float:
    risk_capital = equity * risk_pct / 100.0
    risk_per_share = max(entry - sl, 1e-8)
    qty = risk_capital / risk_per_share
    return max(qty, 0.0)


def simulate(df: pd.DataFrame, ticker: str, start_idx: int, init_equity: float, risk_pct: float,
             use_llm: bool = True) -> Dict[str, Any]:
    equity = init_equity
    equity_curve = []
    trades: List[Trade] = []

    in_pos = False
    entry_px = sl_px = tp = 0.0
    qty = 0.0
    entry_date: Optional[pd.Timestamp] = None

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]  # simulate decisions using next day's range

        # Update equity curve (mark-to-market) even when flat or in position
        mtm = equity if not in_pos else equity + qty * (row['Close'] - entry_px)
        equity_curve.append(mtm)

        # Manage open position exits on next day (priority SL first)
        if in_pos:
            exited = False
            # Check SL
            if nxt['Low'] <= sl_px:
                exit_px = float(sl_px)
                equity += qty * (exit_px - entry_px)
                trades.append(Trade(entry_date, entry_px, sl_px, tp, nxt.name, exit_px,
                                    (exit_px - entry_px) / max(entry_px - sl_px, 1e-8)))
                in_pos = False
                exited = True
            # Check single TP
            if not exited and nxt['High'] >= tp:
                exit_px = float(tp)
                equity += qty * (exit_px - entry_px)
                trades.append(Trade(entry_date, entry_px, sl_px, tp, nxt.name, exit_px,
                                    (exit_px - entry_px) / max(entry_px - sl_px, 1e-8)))
                in_pos = False
                exited = True
            if exited:
                qty = 0.0
                entry_date = None
                continue # proceed to next day after exit
            continue # still in position, skip to next day

        # If flat, ask LLM (or fallback) for a plan
        decision = None
        if use_llm:
            prompt = render_prompt(row, ticker=ticker, risk_pct=risk_pct)
            decision = call_llm(prompt)
        if not decision:
            decision = fallback_decision(row)

        if decision.get('regime') in ['trend_up', 'ranging'] and 'enter' in decision:
            entry_plan = decision['enter']
            if entry_plan.get('type') == 'market':
                # market: enter at next open
                entry_px = float(nxt['Open'])
                sl_px = float(decision['stop_loss'])
                tp = float(decision['take_profit'])
                qty = position_qty(equity, entry_px, sl_px, risk_pct)
                entry_date = nxt.name
                in_pos = qty > 0
            else:
                entry_px = entry_plan['price']
                if nxt['Low'] <= entry_px <= nxt['High']:
                    sl_px = float(decision['stop_loss'])
                    tp = float(decision['take_profit'])
                    qty = position_qty(equity, entry_px, sl_px, risk_pct)
                    entry_date = nxt.name
                    in_pos = qty > 0
                    if i==len(df)-2:
                        equity += qty * (exit_px - entry_px)
                        trades.append(Trade(entry_date, entry_px, sl_px, tp, None, None, 0))
                    
    # finalize equity curve with last close
    if len(df) > 0:
        last_close = df.iloc[-1]['Close']
        final_mtm = equity if not in_pos else equity + qty * (last_close - entry_px)
        equity_curve.append(final_mtm)

    # Metrics
    res = pd.DataFrame([t.__dict__ for t in trades])
    metrics: Dict[str, Any] = {
        'trades': int(len(res)),
        'final_equity': float(equity_curve[-1] if equity_curve else init_equity),
        'initial_equity': float(init_equity),
        'total_return_pct': float((equity_curve[-1] / init_equity - 1) * 100) if equity_curve else 0.0,
        'win_rate_pct': float((res['r_multiple'] > 0).mean() * 100) if len(res) else 0.0,
        'expectancy_R': float(res['r_multiple'].mean()) if len(res) else 0.0,
        'profit_factor': float(res.loc[res['r_multiple'] > 0, 'r_multiple'].sum() / max(1e-8, abs(res.loc[res['r_multiple'] <= 0, 'r_multiple'].sum()))) if len(res) else float('nan'),
    }

    # Max drawdown from equity curve
    eq = pd.Series(equity_curve)
    if not eq.empty:
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max
        metrics['max_drawdown_pct'] = float(dd.min() * 100)
    else:
        metrics['max_drawdown_pct'] = 0.0

    return {
        'metrics': metrics,
        'trades_df': res,
        'equity_curve': pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    }

# =========================
# Dataset builder
# =========================

def build_dataset(df: pd.DataFrame, ticker: TickerParam) -> pd.DataFrame:
    df = df.copy()
    # Indicators
    df['EMA_SHORT'] = ema(df['Close'], ticker.ema_short)
    df['EMA_LONG'] = ema(df['Close'], ticker.ema_long)
    df['RSI'] = rsi(df['Close'], ticker.rsi)

    # MACD histogram
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = macd - signal

    df['ATR'] = atr(df, 14)
    df['BB_MID'], df['BB_LOW'], df['BB_UP'] = bollinger(df['Close'], 20, 2)
    df['Regime'] = classify_regime(df)

    # Candidate levels
    df['Pullback_EMA_SHORT'] = df['EMA_SHORT']
    df['Pullback_ATR'] = df['Close'] - 0.5 * df['ATR']

    # Drop early NaNs
    df = df.dropna().copy()
    return df

# =========================
# CLI / Main
# =========================

def plot_results(ticker: str, df: pd.DataFrame, equity_curve: pd.Series, trades_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price and EMA plot
    ax1.plot(df.index, df['Close'], label='Close')
    ax1.plot(df.index, df['EMA_SHORT'], label='EMA_SHORT')
    ax1.plot(df.index, df['EMA_LONG'], label='EMA_LONG')

    cross_up = (df['EMA_SHORT'] > df['EMA_LONG']) & (df['EMA_SHORT'].shift(1) <= df['EMA_LONG'].shift(1))
    cross_down = (df['EMA_SHORT'] < df['EMA_LONG']) & (df['EMA_SHORT'].shift(1) >= df['EMA_LONG'].shift(1))

    ax1.scatter(df.index[cross_up], df['Close'][cross_up], marker='o', color='green', label='EMA Cross Up')
    ax1.scatter(df.index[cross_down], df['Close'][cross_down], marker='x', color='red', label='EMA Cross Down')

    if not trades_df.empty:
        ax1.scatter(trades_df['entry_date'], trades_df['entry'], marker='^', label='Entry')
        ax1.scatter(trades_df['exit_date'], trades_df['exit'], marker='v', label='Exit')
    ax1.set_title(f"{ticker} – Entries/Exits & EMA Crossings")
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # RSI plot
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)
    ax2.set_title('RSI Status')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Equity curve plot
    plt.figure(figsize=(12, 5))
    plt.plot(equity_curve.index, equity_curve.values, label='Strategy Equity')
    plt.title(f"{ticker} – Equity Curve")
    plt.xlabel('Date'); plt.ylabel('Equity'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

def optuna_objective(trial, raw_df, equity, risk_pct, use_llm):
    ema20 = trial.suggest_int('EMA_SHORT', 10, 30)
    ema50 = trial.suggest_int('EMA_LONG', 35, 70)
    rsi_n = trial.suggest_int('RSI', 7, 21)

    df = raw_df.copy()
    df['EMA_SHORT'] = ema(df['Close'], ema20)
    df['EMA_LONG'] = ema(df['Close'], ema50)
    df['RSI'] = rsi(df['Close'], rsi_n)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = macd - signal
    df['ATR'] = atr(df, 14)
    df['BB_MID'], df['BB_LOW'], df['BB_UP'] = bollinger(df['Close'], 20, 2)
    df['Regime'] = classify_regime(df)
    df['Pullback_EMA_SHORT'] = df['EMA_SHORT']
    df['Pullback_ATR'] = df['Close'] - 0.5 * df['ATR']
    df = df.dropna().copy()

    result = simulate(df, ticker="OPT", start_idx=0, init_equity=equity, risk_pct=risk_pct, use_llm=use_llm)
    metrics = result['metrics']

    # Extract key metrics
    total_return = metrics['total_return_pct']
    win_rate = metrics['win_rate_pct']
    max_drawdown = metrics['max_drawdown_pct']
    trades = metrics['trades']

    print(f"Trial {trial.number}: return={total_return:.2f}%, win_rate={win_rate:.2f}%, drawdown={max_drawdown:.2f}%, trades={trades}")
    
    # Constraint: Max drawdown must be better than -5%
    if max_drawdown < -5.0:
        # Penalty for excessive drawdown
        penalty = abs(max_drawdown + 5.0) * 10  # Heavy penalty
        score = total_return - penalty
        print(f"  → PENALIZED: DD={max_drawdown:.2f}% exceeds -5% limit, penalty={penalty:.2f}")
        return score
    
    # Minimum trade requirement (avoid overfitting on few trades)
    if trades < 10:
        penalty = (10 - trades) * 5  # Penalty for too few trades
        score = total_return - penalty
        print(f"  → PENALIZED: Only {trades} trades (min 10), penalty={penalty:.2f}")
        return score
    
    # Multi-objective optimization with ORIGINAL weights
    # Primary: Total return (60%) - Maximize profits!
    # Secondary: Win rate (25%) - Consistency matters  
    # Tertiary: Drawdown control (15%) - Risk management
    
    return_score = total_return * 0.6        # High weight on raw return
    win_rate_score = (win_rate - 50) * 0.5   # Reward for >50% win rate
    drawdown_score = (5 + max_drawdown) * 3  # Moderate reward for smaller drawdown
    
    final_score = return_score + win_rate_score + drawdown_score
    
    print(f"  → SCORE: {final_score:.2f} (return={return_score:.2f} + win_rate={win_rate_score:.2f} + dd_control={drawdown_score:.2f})")
    
    return final_score

def run_optuna(raw_df, equity, risk_pct, use_llm, n_trials):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, raw_df, equity, risk_pct, use_llm), n_trials=n_trials)
    
    print("\n" + "="*80)
    print("🎯 OPTIMIZATION RESULTS")
    print("="*80)
    print("Best parameters:", study.best_params)
    print(f"Best score: {study.best_value:.2f}")
    
    # Test best parameters to get detailed metrics
    best_trial = study.best_trial
    ema_short = best_trial.params['EMA_SHORT']
    ema_long = best_trial.params['EMA_LONG']
    rsi_period = best_trial.params['RSI']
    
    # Rebuild with best params
    df = raw_df.copy()
    df['EMA_SHORT'] = ema(df['Close'], ema_short)
    df['EMA_LONG'] = ema(df['Close'], ema_long)
    df['RSI'] = rsi(df['Close'], rsi_period)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = macd - signal
    df['ATR'] = atr(df, 14)
    df['BB_MID'], df['BB_LOW'], df['BB_UP'] = bollinger(df['Close'], 20, 2)
    df['Regime'] = classify_regime(df)
    df['Pullback_EMA_SHORT'] = df['EMA_SHORT']
    df['Pullback_ATR'] = df['Close'] - 0.5 * df['ATR']
    df = df.dropna().copy()
    
    result = simulate(df, ticker="OPTIMIZED", start_idx=0, init_equity=equity, risk_pct=risk_pct, use_llm=use_llm)
    best_metrics = result['metrics']
    
    print("\n📊 BEST STRATEGY PERFORMANCE:")
    print(f"  • Total Return: {best_metrics['total_return_pct']:.2f}%")
    print(f"  • Win Rate: {best_metrics['win_rate_pct']:.2f}%")
    print(f"  • Max Drawdown: {best_metrics['max_drawdown_pct']:.2f}%")
    print(f"  • Total Trades: {best_metrics['trades']}")
    print(f"  • Profit Factor: {best_metrics['profit_factor']:.2f}")
    print(f"  • Expectancy R: {best_metrics['expectancy_R']:.3f}")
    
    # Validate constraints
    dd_ok = best_metrics['max_drawdown_pct'] >= -5.0
    trades_ok = best_metrics['trades'] >= 10
    
    print("\n✅ CONSTRAINT VALIDATION:")
    print(f"  • Max DD ≥ -5%: {'✅ PASS' if dd_ok else '❌ FAIL'} ({best_metrics['max_drawdown_pct']:.2f}%)")
    print(f"  • Min Trades ≥ 10: {'✅ PASS' if trades_ok else '❌ FAIL'} ({best_metrics['trades']} trades)")
    
    if dd_ok and trades_ok:
        print("\n🎉 OPTIMIZATION SUCCESS: All constraints satisfied!")
    else:
        print("\n⚠️  WARNING: Some constraints not met. Consider running more trials.")
    
    print("="*80)
    
    return study

def backtest(args):
    # Load raw OHLCV
    list_tickers = load_ticker('./issi.json')
    raw = load_ohlcv(args.tickers[0], args.start)
    ticker = next((t for t in list_tickers if t.symbol == args.tickers[0]), None)
    df = build_dataset(raw, ticker)

    # Start index after indicators are warmed up
    start_idx = 0
    result = simulate(df, ticker=ticker.symbol, start_idx=start_idx, init_equity=args.equity, risk_pct=args.risk, use_llm=(not args.no_llm))

    metrics = result['metrics']
    trades_df = result['trades_df']
    equity_curve = result['equity_curve']

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if not trades_df.empty:
        print("\nSample trades (last 10):")
        print(trades_df.tail(10).to_string(index=False))
    else:
        print("No trades generated.")

    if args.plots:
        plot_results(ticker.symbol, df, equity_curve, trades_df)

def live_signal_loop(args):
    while True:
        all_tickers = load_ticker('./issi.json')
        tickers = [t for t in all_tickers if args.tickers is None or t.symbol in args.tickers]
        
        for ticker in tickers:
            raw = load_ohlcv(ticker.symbol, args.start)
            df = build_dataset(raw, ticker)
            prevday = df.iloc[-2]
            currentday = df.iloc[-1]

            valid_currentday = currentday.name.date() == pd.Timestamp.now().date()
            if not valid_currentday:
                prevday = currentday

            if not args.no_llm:
                prompt = render_prompt(prevday, ticker=ticker.symbol, risk_pct=args.risk)
                decision = call_llm(prompt)
            else:
                decision = fallback_decision(prevday)

            print(f"{ticker.symbol} [{prevday.name.date()}] Signal: {decision}")

        # Wait 15 minutes before checking again
        time.sleep(900)

def optimize(args):
    all_tickers = load_ticker('./issi.json')
    # Determine which tickers to optimize
    ticker_names = args.tickers if args.tickers is not None else [t.symbol for t in all_tickers]
    # Ensure all tickers in ticker_names exist in all_tickers
    symbol_to_ticker = {t.symbol: t for t in all_tickers}
    for ticker_name in ticker_names:
        ticker = symbol_to_ticker.get(ticker_name)
        if ticker is None:
            ticker = TickerParam(ticker_name, ticker_name, 20, 50, 14)
            all_tickers.append(ticker)
            symbol_to_ticker[ticker_name] = ticker
        df = load_ohlcv(ticker.symbol, args.start)
        study = run_optuna(df, args.equity, args.risk, not args.no_llm, n_trials=500)
        ticker.ema_short = study.best_params['EMA_SHORT']
        ticker.ema_long = study.best_params['EMA_LONG']
        ticker.rsi = study.best_params['RSI']

    with open('./issi.json', 'w', encoding='utf-8') as f:
        json.dump([t.__dict__ for t in all_tickers], f, indent=2)

def main():
    # set -start to a year from now if not provided
    one_year_ago = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', type=str, help='List of tickers to optimize (space separated)')
    parser.add_argument('--start', type=str, default=one_year_ago)
    parser.add_argument('--equity', type=float, default=100_000_000)
    parser.add_argument('--risk', type=float, default=1.0, help='Risk per trade in %% of equity')
    parser.add_argument('--plots', type=int, default=1, help='1=show plots, 0=no plots')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM and use fallback only')
    parser.add_argument('--optimize', action='store_true', help='Run Optuna parameter optimization')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--init-investing-id', action='store_true', help='Initialize investing_id for specified tickers and save to cache')
    args = parser.parse_args()

    if args.init_investing_id:
        if not args.tickers:
            print("❌ Error: --tickers is required when using --init-investing-id")
            print("Example: python main.py --tickers WIFI.JK BRMS.JK --init-investing-id")
            return
        init_investing_ids_for_tickers(args.tickers)
        return

    if args.optimize:
        optimize(args)
        return

    if args.backtest:
        backtest(args)
        return

    live_signal_loop(args)

if __name__ == '__main__':
    main()