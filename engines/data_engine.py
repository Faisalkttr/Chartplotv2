import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Tuple

PORTFOLIO_GRID = {
    "INFRA (14%)": {
        "Layer 1: Hard Assets (40%)": {
            "tickers": ["TPL", "ADPORTS.AD", "ICTEY", "CNI", "CP", "UNP"],
            "benchmark": "XLI"
        },
        "Layer 2: Grid & Utilities (40%)": {
            "tickers": ["LIN", "ABBN.SW", "SU.PA", "GEV", "ETN", "NVT", "CEG", "PWR", "CWCO", "XYL", "ECL", "WM", "RSG"],
            "benchmark": "XLU"
        },
        "Layer 3: Tech-Adjacent (20%)": {
            "tickers": ["VRT", "BE", "ANET", "FTNT", "CHKP", "CRWD", "ZS"],
            "benchmark": "QQQ"
        }
    },
    "ENERGY & COMMODITY (18%)": {
        "Layer 1: Monetary Royalties (40%)": {
            "tickers": ["FNV", "WPM"],
            "benchmark": "GLD"
        },
        "Layer 2: Baseload Energy (40%)": {
            "tickers": ["CCJ", "CNQ", "XOM", "SU", "EQT", "CVX"],
            "benchmark": "XLE"
        },
        "Layer 3: Industrial Materials (20%)": {
            "tickers": ["FCX", "SCCO", "BHP", "NEM", "COP", "NUE", "PH", "CAT"],
            "benchmark": "XLB"
        }
    },
    "AI / SEMIS (10%)": {
        "Layer 1: Physical Monopolies (60%)": {
            "tickers": ["TSM", "ASML", "SHECY", "6920.T"],
            "benchmark": "SMH"
        },
        "Layer 2: Architecture & Robotics (30%)": {
            "tickers": ["AVGO", "CDNS", "QCOM", "FANUY", "8035.T", "SNPS"],
            "benchmark": "SMH"
        },
        "Layer 3: Velocity Applications (10%)": {
            "tickers": ["NOW", "PANW", "STX"],
            "benchmark": "XLK"
        }
    },
    "EM (7%)": {
        "Layer 1: INDIA (40%)": {
            "tickers": ["ABB.NS", "SIEMENS.NS", "CGPOWER.NS", "PIIND.NS", "SUNPHARMA.NS", "HCLTECH.NS"],
            "benchmark": "INDA"
        },
        "Layer 2: GCC (40%)": {
            "tickers": ["2222.SR", "ACWAPOWER.SR", "STC.SR"],
            "benchmark": "KSA"
        },
        "Layer 3: Other Jurisdiction (20%)": {
            "tickers": ["DXJ", "TLK", "VALE", "CEO", "CHL"],
            "benchmark": "EEM"
        }
    },
    "Business & Futuristic Overlay (6%)": {
        "Core Healthcare & Biotech (100%)": {
            "tickers": ["NVO", "AZN", "ISRG", "TMO"],
            "benchmark": "XLV"
        }
    }
}

@st.cache_data(ttl=3600)
def get_stock_data_isolated(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Fetches ticker data individually. Isolates broken tickers so they don't corrupt 
    the rest of the structural layer dataset.
    
    Returns:
        DataFrame containing successful tickers
        List of working tickers
        List of failed tickers
    """
    if not tickers:
        return pd.DataFrame(), [], []
        
    successful_dfs = {}
    failed_tickers = []
    working_tickers = []
    
    for ticker in tickers:
        try:
            # Download individually with a strict timeout parameter
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
            
            if data.empty:
                failed_tickers.append(ticker)
                continue
                
            df_col = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            
            # Ensure series formatting is converted clean to a 1D column
            if isinstance(df_col, pd.DataFrame):
                df_col = df_col.iloc[:, 0]
                
            successful_dfs[ticker] = df_col.ffill().bfill()
            working_tickers.append(ticker)
            
        except Exception:
            failed_tickers.append(ticker)
            
    if not successful_dfs:
        return pd.DataFrame(), [], failed_tickers
        
    combined_df = pd.DataFrame(successful_dfs)
    return combined_df, working_tickers, failed_tickers

def generate_automated_scoring(df: pd.DataFrame, target_tickers: List[str], base_ticker: str) -> List[Dict[str, Any]]:
    """Generates trend metrics only for target tickers that exist inside the successfully loaded DataFrame."""
    records = []
    if df.empty or base_ticker not in df.columns:
        return records

    base_prices = df[base_ticker]
    
    # Fill any internal price gaps and calculate returns safely
    base_returns = base_prices.ffill().bfill().pct_change().fillna(0.0)

    for ticker in target_tickers:
        if ticker not in df.columns or ticker == base_ticker:
            continue
            
        t_price = df[ticker].ffill().bfill()
        t_returns = t_price.pct_change().fillna(0.0) # <-- Fix 1: Eliminate the first-row NaN from pct_change
        
        # Calculate raw relative strength ratio curve
        raw_ratio = t_price / base_prices
        
        # Convert any division anomalies into nulls and drop them
        clean_ratio = raw_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Ensure we have enough history left to run an analytical window
        if len(clean_ratio) < 10:
            continue
            
        # Extract the trailing 63 trading days from the cleaned ratio series
        ratio_window = clean_ratio.tail(63)
        tail_index = ratio_window.index
        
        # Re-align returns data arrays on the exact indices that contain clean ratios
        tail_returns = t_returns.loc[tail_index].fillna(0.0)
        tail_base_returns = base_returns.loc[tail_index].fillna(0.0)
        
        # Performance Windows
        ret_1m = float((t_price.iloc[-1] / t_price.iloc[-21]) - 1) if len(t_price) > 21 else 0.0
        ret_3m = float((t_price.iloc[-1] / t_price.iloc[-63]) - 1) if len(t_price) > 63 else 0.0
        
        # Volatility-Adjusted RS calculations
        rolling_alpha = float(tail_returns.sum() - tail_base_returns.sum())
        tracking_diff = tail_returns - tail_base_returns
        tracking_err = tracking_diff.std() * np.sqrt(252)
        vol_adj_rs = (rolling_alpha * np.sqrt(252)) / tracking_err if tracking_err > 0 else 0.0
        
        # Breakout Diagnostics
        ratio_ma20 = clean_ratio.rolling(20, min_periods=1).mean().iloc[-1]
        ratio_ma50 = clean_ratio.rolling(50, min_periods=1).mean().iloc[-1]
        is_breakout = clean_ratio.iloc[-1] > ratio_ma20 > ratio_ma50
        
        # Prepare inputs for linear regression modeling
        y_vals = ratio_window.values.reshape(-1, 1)
        x_vals = np.arange(len(y_vals)).reshape(-1, 1)
        
        # Double check that y_vals contains absolutely no infinite or missing blocks
        if not np.isfinite(y_vals).all():
            continue
            
        # Safe regression processing execution
        reg = LinearRegression().fit(x_vals, y_vals)
        r2 = float(reg.score(x_vals, y_vals))
        slope = float(reg.coef_[0][0])
        
        max_dd = float(((t_price - t_price.cummax()) / t_price.cummax()).min())

        score = 30.0
        if vol_adj_rs > 0: score += min(vol_adj_rs * 15, 30)
        if is_breakout: score += 20
        if r2 > 0.4 and slope > 0: score += 20
        score = max(0.0, min(100.0, score))

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
