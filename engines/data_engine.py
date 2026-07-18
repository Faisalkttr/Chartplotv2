import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Tuple

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
    base_returns = base_prices.pct_change()

    for ticker in target_tickers:
        # Gracefully skip if the ticker failed download and isn't present in the DataFrame columns
        if ticker not in df.columns or ticker == base_ticker:
            continue
            
        t_price = df[ticker]
        t_returns = t_price.pct_change()
        ratio = t_price / base_prices
        
        ret_1m = float((t_price.iloc[-1] / t_price.iloc[-21]) - 1) if len(t_price) > 21 else 0.0
        ret_3m = float((t_price.iloc[-1] / t_price.iloc[-63]) - 1) if len(t_price) > 63 else 0.0
        
        rolling_alpha = float(t_returns.tail(63).sum() - base_returns.tail(63).sum())
        tracking_err = (t_returns - base_returns).tail(63).std() * np.sqrt(252)
        vol_adj_rs = (rolling_alpha * np.sqrt(252)) / tracking_err if tracking_err > 0 else 0.0
        
        ratio_ma20 = ratio.rolling(20).mean().iloc[-1]
        ratio_ma50 = ratio.rolling(50).mean().iloc[-1]
        is_breakout = ratio.iloc[-1] > ratio_ma20 > ratio_ma50
        
        y_vals = ratio.tail(63).values.reshape(-1, 1)
        x_vals = np.arange(len(y_vals)).reshape(-1, 1)
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
