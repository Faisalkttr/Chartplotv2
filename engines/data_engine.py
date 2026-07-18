import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any

# Complete Integration of your Multi-Tiered Portfolio Allocation Grid
PORTFOLIO_GRID = {
    "INFRA (14%)": {
        "Layer 1: Hard Assets (40%)": {
            "tickers": ["TPL", "ADPORTS.AE", "ICTEY", "CNI", "CP", "UNP"],
            "benchmark": "XLI"  # Industrials Baseline
        },
        "Layer 2: Grid & Utilities (40%)": {
            "tickers": ["LIN", "ABBN.SW", "SU.PA", "GEV", "ETN", "NVT", "CEG", "PWR", "CWCO", "XYL", "ECL", "WM", "RSG"],
            "benchmark": "XLU"  # Utilities/Power Baseline
        },
        "Layer 3: Tech-Adjacent (20%)": {
            "tickers": ["VRT", "BE", "ANET", "FTNT", "CHKP", "CRWD", "ZS"],
            "benchmark": "QQQ"  # Structural Tech
        }
    },
    "ENERGY & COMMODITY (18%)": {
        "Layer 1: Monetary Royalties (40%)": {
            "tickers": ["FNV", "WPM"],
            "benchmark": "GLD"  # Spot Gold Benchmark
        },
        "Layer 2: Baseload Energy (40%)": {
            "tickers": ["CCJ", "CNQ", "XOM", "SU", "EQT", "CVX"],
            "benchmark": "XLE"  # Energy Baseline
        },
        "Layer 3: Industrial Materials (20%)": {
            "tickers": ["FCX", "SCCO", "BHP", "NEM", "COP", "NUE", "PH", "CAT"],
            "benchmark": "XLB"  # Basic Materials Sector
        }
    },
    "AI / SEMIS (10%)": {
        "Layer 1: Physical Monopolies (60%)": {
            "tickers": ["TSM", "ASML", "SHECY", "6920.T"],
            "benchmark": "SMH"  # Semiconductor Index
        },
        "Layer 2: Architecture & Robotics (30%)": {
            "tickers": ["AVGO", "CDNS", "QCOM", "FANUY", "8035.T", "SNPS"],
            "benchmark": "SMH"
        },
        "Layer 3: Velocity Applications (10%)": {
            "tickers": ["NOW", "PANW", "STX"],
            "benchmark": "XLK"  # Technology Sector
        }
    },
    "EM (7%)": {
        "Layer 1: INDIA (40%)": {
            "tickers": ["ABB.NS", "SIEMENS.NS", "CGPOWER.NS", "PIIND.NS", "SUNPHARMA.NS", "HCLTECH.NS"],
            "benchmark": "INDA" # India ETF
        },
        "Layer 2: GCC (40%)": {
            "tickers": ["2222.SR", "ARAMCO", "ACWAPOWER.SR", "STC.SR"],
            "benchmark": "KSA"  # Saudi/Regional Benchmark
        },
        "Layer 3: Other Jurisdiction (20%)": {
            "tickers": ["DXJ", "TLK", "VALE", "CEO", "CHL"],
            "benchmark": "EEM"  # Emerging Markets Core
        }
    },
    "Business & Futuristic Overlay (6%)": {
        "Core Healthcare & Biotech (100%)": {
            "tickers": ["NVO", "AZN", "ISRG", "TMO"],
            "benchmark": "XLV"  # Healthcare Core Index
        }
    }
}

@st.cache_data(ttl=3600)
def get_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches core equity data paths while handling errors clean."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        df = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])
        return df.ffill().bfill()
    except Exception:
        return pd.DataFrame()

def generate_automated_scoring(df: pd.DataFrame, target_tickers: List[str], base_ticker: str) -> List[Dict[str, Any]]:
    """Generates complex trend metrics and composite health scores (0-100) for all tickers."""
    records = []
    if df.empty or base_ticker not in df.columns:
        return records

    base_prices = df[base_ticker]
    base_returns = base_prices.pct_change()

    for ticker in target_tickers:
        if ticker not in df.columns or ticker == base_ticker:
            continue
            
        t_price = df[ticker]
        t_returns = t_price.pct_change()
        ratio = t_price / base_prices
        
        # Performance Windows
        ret_1m = float((t_price.iloc[-1] / t_price.iloc[-21]) - 1) if len(t_price) > 21 else 0.0
        ret_3m = float((t_price.iloc[-1] / t_price.iloc[-63]) - 1) if len(t_price) > 63 else 0.0
        
        # Alpha and Tracking Calculations
        rolling_alpha = float(t_returns.tail(63).sum() - base_returns.tail(63).sum())
        tracking_err = (t_returns - base_returns).tail(63).std() * np.sqrt(252)
        vol_adj_rs = (rolling_alpha * np.sqrt(252)) / tracking_err if tracking_err > 0 else 0.0
        
        # Breakout Diagnostics
        ratio_ma20 = ratio.rolling(20).mean().iloc[-1]
        ratio_ma50 = ratio.rolling(50).mean().iloc[-1]
        is_breakout = ratio.iloc[-1] > ratio_ma20 > ratio_ma50
        
        # Trend Line OLS Slope fitting
        y_vals = ratio.tail(63).values.reshape(-1, 1)
        x_vals = np.arange(len(y_vals)).reshape(-1, 1)
        reg = LinearRegression().fit(x_vals, y_vals)
        r2 = float(reg.score(x_vals, y_vals))
        slope = float(reg.coef_[0][0])
        
        # Peak-to-trough Drawdown calculation
        max_dd = float(((t_price - t_price.cummax()) / t_price.cummax()).min())

        # Quantitative 0-100 Health Scoring Engine Algorithm
        score = 30.0  # Baseline
        if vol_adj_rs > 0: score += min(vol_adj_rs * 15, 30) # Vol-adjusted strength component (Max 30)
        if is_breakout: score += 20                          # Breakout state component (Max 20)
        if r2 > 0.4 and slope > 0: score += 20               # Structural trend alignment (Max 20)
        score = max(0.0, min(100.0, score))                  # Boundaries enforcement

        if score >= 75.0: status = "🚀 BUY"
        elif score >= 45.0: status = "✋ HOLD"
        else: status = "❌ WATCH"

        records.append({
            "Asset": ticker,
            "Health Score": round(score, 1),
            "Status": status,
            "1M Return": ret_1m,
            "3M Return": ret_3m,
            "63D Alpha vs BM": rolling_alpha,
            "Vol-Adjusted RS": vol_adj_rs,
            "Trend R²": r2,
            "Max Drawdown": max_dd
        })
        
    return records