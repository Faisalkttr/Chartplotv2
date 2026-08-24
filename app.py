import streamlit as st
import datetime
import re
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Tuple

st.set_page_config(layout="wide", page_title="Portfolio Grid Intelligence Engine", page_icon="⚡")

# =============================================================================
# UNIFIED BUILD — data engine + chart engine + UI in one file.
# Integrates: (1) master-table photo sync of PORTFOLIO_GRID; (2) currency /
# IR / log-regression fixes; (3) Long-Term Mode (252D slope, 12-1 skip-month
# rel. momentum, capture ratios, crisis alpha, rank-based composite, breakout
# hysteresis); (4) Sizing & Stress anti-fragile panel (cash governor,
# risk-parity lens, concentration + liquidity audit).
# CAVEAT (unchanged): 1M/3M returns for India/GCC layers are local-currency.
# =============================================================================

PORTFOLIO_GRID = {
    "INFRA (14%)": {
        "Layer 1: Hard Assets & Global Freight (35%)": {
            "tickers": ["TPL", "ICTEY", "CNI", "CP", "UNP"],
            "benchmark": "XLI"},
        "Layer 2: Electrification, Grid & AI-Power (40%)": {
            # ABBN->ABBNY, SU.PA->SBGSY verified USD ADR swaps (currency fix)
            "tickers": ["ABBNY", "SBGSY", "ETN", "GEV", "CEG", "PWR", "PRY.MI", "FUWAY",
                        "VRT", "AGX", "LIN", "NVT", "PH", "BE"],
            "benchmark": "GRID"},
        "Layer 3: Water & Environmental (10%)": {
            "tickers": ["XYL", "ECL", "WM", "RSG", "CWCO", "BMI"],
            "benchmark": "XLU"},
        "Layer 4: Cyber, Networking & Edge (15%)": {
            "tickers": ["ANET", "CRWD", "FTNT", "PANW", "OKTA", "ADTN", "CALX", "HLIT", "ZS", "CHKP"],
            "benchmark": "QQQ"}},
    "ENERGY & COMMODITY (18%)": {
        "Layer 1: Monetary Royalties (30%)": {
            "tickers": ["FNV", "WPM"], "benchmark": "GLD"},
        "Layer 2: Baseload & Nuclear Fuel (40%)": {
            "tickers": ["CCJ", "UUUU", "CNQ", "XOM", "SU", "EQT", "CVX"], "benchmark": "XLE"},
        "Layer 3: Industrial & Critical Materials (30%)": {
            # COP kept here per master table; RIO removed per master table
            "tickers": ["FCX", "SCCO", "BHP", "NEM", "STLD", "CAT", "3750.HK", "HBM",
                        "AA", "ALB", "ALM", "LYSCF", "NUE", "COP"],
            "benchmark": "XLB"}},
    "AI / SEMIS (10%)": {
        "Layer 1: Physical Monopolies, Foundry (45%)": {
            # Lasertec 6920.T -> LSRCY verified ADR
            "tickers": ["TSM", "ASML", "SHECY", "ENTG", "GFS", "LSRCY", "AXTI", "ALMU"],
            "benchmark": "SMH"},
        "Layer 2: Architecture, Edge AI & Memory (30%)": {
            # Tokyo Electron 8035.T -> TOELY verified ADR; FANUY kept per thesis text
            "tickers": ["AVGO", "CDNS", "SNPS", "TOELY", "QCOM", "MRAM", "AMBA", "PENG", "FANUY"],
            "benchmark": "SMH"},
        "Layer 3: Velocity Applications (15%)": {
            "tickers": ["NOW", "STX"], "benchmark": "XLK"},
        "Layer 4: Vertical Software & Data Monopolies (10%)": {
            "tickers": ["VRSN", "MANH", "WTC.AX", "DSGX", "FDS", "KXS.TO", "TRMB", "IKTSY"],
            "benchmark": "IGV"}},
    "EM (7%)": {
        "Layer 1: India — Grid, Manufacturing, Consumption (50%)": {
            # .NS suffixes for NSE; ^NSEI benchmark cancels the INR term.
            # CAVEAT: 6501.T is JPY-listed -> JPY/INR FX term in its ratio.
            "tickers": ["6501.T", "CGPOWER.NS", "DIXON.NS", "KAYNES.NS", "HFCL.NS",
                        "CONCOR.NS", "SUNPHARMA.NS", "HCLTECH.NS", "ABB.NS", "SIEMENS.NS",
                        "PIIND.NS", "STLTECH.NS", "PRECWIRE.NS", "MTARTECH.NS",
                        "HINDCOPPER.NS", "DIACABS.NS", "POWERINDIA.NS"],
            "benchmark": "^NSEI"},
        "Layer 2: GCC (20%)": {
            "tickers": ["2222.SR", "2082.SR", "7010.SR"], "benchmark": "^TASI.SR"},
        "Layer 3: Other EM / China / LatAm (30%)": {
            "tickers": ["VALE", "TLK", "BABAF", "0941.HK", "CSUAY", "0883.HK", "YPF",
                        "INDO", "ISDE", "HIJP", "KAP.IL"],
            "benchmark": "EEM"}},
    "Biz & Futuristic Overlay (6%)": {
        "Health, Biotech & Longevity (100%)": {
            "tickers": ["NVO", "AZN", "ISRG", "TMO", "RHHBY"], "benchmark": "XLV"}},
    "BTC (25%)": {
        "Core / Satellite (90/10)": {
            # Cold-wallet BTC not fetchable; BTC-USD is the core baseline.
            "tickers": ["MSTR", "RIOT"], "benchmark": "BTC-USD"}},
    "GOLD (10%)": {
        "Physical": {
            # GLD = tradable proxy; SPY = opportunity-cost baseline.
            "tickers": ["GLD"], "benchmark": "SPY"}}}

SECTION_TARGETS = {"INFRA": 0.14, "ENERGY & COMMODITY": 0.18, "AI / SEMIS": 0.10, "EM": 0.07,
                   "Biz & Futuristic Overlay": 0.06, "BTC": 0.25, "GOLD": 0.10, "CASH": 0.10}
LIQUIDITY_WATCH = {"LYSCF", "IKTSY", "BABAF", "CSUAY", "0941.HK", "ALMU",
                   "KAP.IL", "FUWAY"}


# =============================================================================
# DATA ENGINE
# =============================================================================
# Suffix -> currency map for FX cross-term detection (best-effort, not exhaustive).
_SUFFIX_CCY = {
    ".T": "JPY", ".NS": "INR", ".BO": "INR", ".SR": "SAR", ".AX": "AUD", ".TO": "CAD",
    ".HK": "HKD", ".IL": "ILS", ".AD": "AED", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR",
    ".DE": "EUR", ".SW": "CHF", ".L": "GBP", ".KS": "KRW", ".SS": "CNY", ".SZ": "CNY"}


def infer_currency(ticker: str) -> str:
    """Best-effort currency inference from ticker suffix. US listings / ADRs and
    '-USD' pairs default to USD, which is correct for the vast majority of the
    grid but is a heuristic, not a data feed — treat FX flags as 'check this',
    not as ground truth."""
    if ticker.upper().endswith("-USD"):
        return "USD"
    for suf, ccy in _SUFFIX_CCY.items():
        if ticker.upper().endswith(suf):
            return ccy
    return "USD"


@st.cache_data(ttl=3600)
def get_stock_data_isolated(tickers: List[str], start_date: str, end_date: str
                            ) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Dict[str, float]]]:
    """Fetches tickers individually; isolates broken ones so they can't corrupt the set.

    Also returns a data_quality dict per working ticker with:
      - real_count: number of genuinely observed (non-NaN, pre-fill) price points
      - fill_ratio: fraction of the returned series that was ffill/bfill-padded
      - first_valid: date of the first real observation (None if unknown)
    This lets downstream code refuse to treat padded rows as real history —
    ffill/bfill make gaps *look* continuous, but a stock that IPO'd 4 months
    into a 3-year window will otherwise silently pass "252 days of history"
    checks on mostly synthetic flat-lined data.
    """
    if not tickers:
        return pd.DataFrame(), [], [], {}
    successful_dfs, failed_tickers, working_tickers, quality = {}, [], [], {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
            if data.empty:
                failed_tickers.append(ticker)
                continue
            df_col = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            if isinstance(df_col, pd.DataFrame):
                df_col = df_col.iloc[:, 0]
            raw_col = df_col
            real_count = int(raw_col.notna().sum())
            total_count = int(len(raw_col))
            first_valid = raw_col.first_valid_index()
            filled = raw_col.ffill().bfill()
            successful_dfs[ticker] = filled
            working_tickers.append(ticker)
            quality[ticker] = {
                "real_count": real_count,
                "total_count": total_count,
                "fill_ratio": float(1.0 - real_count / total_count) if total_count else 0.0,
                "first_valid": first_valid.strftime("%Y-%m-%d") if first_valid is not None else None}
        except Exception:
            failed_tickers.append(ticker)
    if not successful_dfs:
        return pd.DataFrame(), [], failed_tickers, {}
    return pd.DataFrame(successful_dfs), working_tickers, failed_tickers, quality


def _clip(v, lo=-5.0, hi=5.0):
    return None if v is None else float(np.clip(v, lo, hi))


def generate_automated_scoring(df: pd.DataFrame, target_tickers: List[str], base_ticker: str,
                               mode: str = "trading",
                               data_quality: Dict[str, Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Two-pass RS scoring. mode='trading' = 63D horizon score (original);
    mode='longterm' = rank-based 252D/12-1 composite with convexity & crisis alpha,
    where the convexity/crisis components are now computed on a genuine ~252D
    window instead of silently reusing the 63D trading-window numbers.

    data_quality (optional): per-ticker dict with 'real_count' (# of genuinely
    observed, pre-fill price points). Used to gate long-window calcs so that
    ffill/bfill padding on recently-listed tickers can't masquerade as real
    252-day history.
    """
    records = []
    if df.empty or base_ticker not in df.columns:
        return records
    base_prices = df[base_ticker]
    base_returns = base_prices.ffill().bfill().pct_change().fillna(0.0)
    dq = data_quality or {}
    base_real = dq.get(base_ticker, {}).get("real_count", len(base_prices))
    base_ccy = infer_currency(base_ticker)

    raw: Dict[str, Dict[str, Any]] = {}
    for ticker in target_tickers:
        if ticker not in df.columns or ticker == base_ticker:
            continue
        t_price = df[ticker].ffill().bfill()
        t_returns = t_price.pct_change().fillna(0.0)
        clean_ratio = (t_price / base_prices).replace([np.inf, -np.inf], np.nan).dropna()
        clean_ratio = clean_ratio[clean_ratio > 0]
        if len(clean_ratio) < 10:
            continue

        # Real (non-padded) history available for THIS pair — the binding
        # constraint is whichever leg (asset or benchmark) has less genuine data.
        t_real = dq.get(ticker, {}).get("real_count", len(t_price))
        eff_real = min(t_real, base_real)
        fill_ratio = float(dq.get(ticker, {}).get("fill_ratio", 0.0))
        fx_mismatch = infer_currency(ticker) != base_ccy

        ratio_window = clean_ratio.tail(63)
        tail_index = ratio_window.index
        tail_returns = t_returns.loc[tail_index].fillna(0.0)
        tail_base_returns = base_returns.loc[tail_index].fillna(0.0)

        ret_1m = float((t_price.iloc[-1] / t_price.iloc[-21]) - 1) if len(t_price) > 21 else 0.0
        ret_3m = float((t_price.iloc[-1] / t_price.iloc[-63]) - 1) if len(t_price) > 63 else 0.0

        # Information Ratio (consistent daily-frequency annualization)
        tracking_diff = tail_returns - tail_base_returns
        annualized_alpha = float(tracking_diff.mean() * 252)
        tracking_err = float(tracking_diff.std() * np.sqrt(252))
        vol_adj_rs = annualized_alpha / tracking_err if tracking_err > 0 else 0.0
        asset_compounded = float((1.0 + tail_returns).prod() - 1.0)
        base_compounded = float((1.0 + tail_base_returns).prod() - 1.0)
        rolling_alpha = asset_compounded - base_compounded

        # Breakout with LONG-TERM hysteresis: entry needs 3 consecutive sessions
        # above the MA20>MA50 stack; exit only below MA20*0.98 or MA20<MA50.
        ratio_ma20_series = clean_ratio.rolling(20, min_periods=20).mean()
        ratio_ma50_series = clean_ratio.rolling(50, min_periods=50).mean()
        state, streak = False, 0
        for r, m20, m50 in zip(clean_ratio.values[-252:], ratio_ma20_series.values[-252:], ratio_ma50_series.values[-252:]):
            if np.isnan(m20) or np.isnan(m50):
                continue
            if not state:
                streak = streak + 1 if (r > m20 > m50) else 0
                if streak >= 3:
                    state = True
            elif r < m20 * 0.98 or m20 < m50:
                state, streak = False, 0
        is_breakout = bool(state)

        # 63D log-ratio regression (scale-invariant daily relative return)
        y_vals = np.log(ratio_window.values).reshape(-1, 1)
        x_vals = np.arange(len(y_vals)).reshape(-1, 1)
        if not np.isfinite(y_vals).all():
            continue
        reg = LinearRegression().fit(x_vals, y_vals)
        r2 = float(reg.score(x_vals, y_vals))
        slope_63d = float(reg.coef_[0][0])

        slope_20d, accel = None, None
        if len(clean_ratio) >= 20:
            sy = np.log(clean_ratio.tail(20).values).reshape(-1, 1)
            sx = np.arange(len(sy)).reshape(-1, 1)
            if np.isfinite(sy).all():
                sreg = LinearRegression().fit(sx, sy)
                slope_20d = float(sreg.coef_[0][0])
                accel = slope_20d - slope_63d

        # Full-period drawdown (used for LT "inv_dd" component & display) —
        # kept separate from the 63D-matched drawdown used in the trading score.
        max_dd_full = float(((t_price - t_price.cummax()) / t_price.cummax()).min())
        tail_prices = t_price.loc[tail_index]
        max_dd_63 = float(((tail_prices - tail_prices.cummax()) / tail_prices.cummax()).min()) \
            if len(tail_prices) > 1 else 0.0
        # dd_efficiency now divides a 63D alpha by a 63D drawdown — same horizon
        # on both sides, instead of the old full-period/63D mismatch.
        dd_efficiency = rolling_alpha / max(abs(max_dd_63), 0.02)

        long_slope, regime = None, "N/A"
        if len(clean_ratio) >= 200:
            ly = np.log(clean_ratio.tail(200).values).reshape(-1, 1)
            lx = np.arange(len(ly)).reshape(-1, 1)
            if np.isfinite(ly).all():
                lreg0 = LinearRegression().fit(lx, ly)
                long_slope = float(lreg0.coef_[0][0])
                lr2 = float(lreg0.score(lx, ly))
                regime = "Bull" if (lr2 > 0.3 and long_slope > 0) else ("Bear" if lr2 > 0.3 else "Neutral")

        # --- LONG-TERM MODE metrics: 252D trend, 12-1 skip-month rel. momentum,
        # capture ratios (convexity) and crisis alpha (behaviour in stress).
        # Gated on EFFECTIVE (real, non-padded) history, not on post-fill row
        # count, so recently-listed tickers can't pass on synthetic flat data. ---
        slope_252d, rel_12_1 = None, None
        has_252_real = eff_real >= 253
        if len(clean_ratio) >= 253 and has_252_real:
            lw = np.log(clean_ratio.tail(252).values).reshape(-1, 1)
            lxw = np.arange(len(lw)).reshape(-1, 1)
            if np.isfinite(lw).all():
                lreg = LinearRegression().fit(lxw, lw)
                slope_252d = float(lreg.coef_[0][0])
            rel_12_1 = float(clean_ratio.iloc[-21] / clean_ratio.iloc[-252] - 1.0)

        # 63D (trading-horizon) capture ratios — kept for the trading score / display.
        up_cap, down_cap = None, None
        up_m, dn_m = tail_base_returns > 0, tail_base_returns < 0
        if up_m.sum() > 5 and tail_base_returns[up_m].mean() != 0:
            up_cap = _clip(tail_returns[up_m].mean() / tail_base_returns[up_m].mean())
        if dn_m.sum() > 5 and tail_base_returns[dn_m].mean() != 0:
            down_cap = _clip(tail_returns[dn_m].mean() / tail_base_returns[dn_m].mean())
        worst = tail_base_returns.nsmallest(10)
        crisis_alpha = float(tail_returns.loc[worst.index].mean() - worst.mean())

        # 252D (genuinely long-term) capture ratios & crisis alpha — these, not
        # the 63D numbers above, feed the Long-Term Mode composite. Requires
        # real (unpadded) history over the window, and more observations per
        # bucket since the window is ~4x longer.
        up_cap_lt, down_cap_lt, crisis_alpha_lt = None, None, None
        if has_252_real:
            lt_index = clean_ratio.tail(252).index
            lt_returns = t_returns.loc[lt_index].fillna(0.0)
            lt_base_returns = base_returns.loc[lt_index].fillna(0.0)
            up_lt, dn_lt = lt_base_returns > 0, lt_base_returns < 0
            if up_lt.sum() > 20 and lt_base_returns[up_lt].mean() != 0:
                up_cap_lt = _clip(lt_returns[up_lt].mean() / lt_base_returns[up_lt].mean())
            if dn_lt.sum() > 20 and lt_base_returns[dn_lt].mean() != 0:
                down_cap_lt = _clip(lt_returns[dn_lt].mean() / lt_base_returns[dn_lt].mean())
            worst_lt = lt_base_returns.nsmallest(20)
            crisis_alpha_lt = float(lt_returns.loc[worst_lt.index].mean() - worst_lt.mean())

        raw[ticker] = dict(
            ret_1m=ret_1m, ret_3m=ret_3m, rolling_alpha=rolling_alpha, vol_adj_rs=vol_adj_rs,
            is_breakout=is_breakout, r2=r2, slope_63d=slope_63d, accel=accel,
            max_dd=max_dd_full, max_dd_63=max_dd_63,
            dd_efficiency=dd_efficiency, long_slope=long_slope, regime=regime,
            slope_252d=slope_252d, rel_12_1=rel_12_1, up_cap=up_cap, down_cap=down_cap,
            crisis_alpha=crisis_alpha, up_cap_lt=up_cap_lt, down_cap_lt=down_cap_lt,
            crisis_alpha_lt=crisis_alpha_lt, fill_ratio=fill_ratio, real_days=t_real,
            fx_mismatch=fx_mismatch)

    if not raw:
        return records

    vol_adj_series = pd.Series({t: m["vol_adj_rs"] for t, m in raw.items()})
    percentile_rank = vol_adj_series.rank(pct=True) * 100.0
    universe_size = len(vol_adj_series)
    ordinal_rank = vol_adj_series.rank(ascending=False, method="min").astype(int)

    # --- LONG-TERM composite: mean of component percentile ranks (no magic caps).
    # All six components are now genuinely long-window (252D-based); none of
    # them silently fall back to the 63D trading-window statistics.
    #
    # NOTE: percentile ranking needs a peer group. With a single asset in `raw`
    # (e.g. a solo ad-hoc lookup), there's nothing to rank against, so we fall
    # back to an ABSOLUTE score built from the raw long-term metrics instead of
    # returning a fake-neutral 50 for everything. `lt_score_is_relative` records
    # which method was used so the UI can tell the user which one they're seeing. ---
    lt_score_is_relative = len(raw) >= 2

    def _pct(vals):
        present = {k: v for k, v in vals.items() if v is not None}
        if len(present) < 2:
            return {k: 50.0 for k in vals}
        s = pd.Series(present).rank(pct=True) * 100.0
        return {k: (float(s[k]) if k in s else 50.0) for k in vals}

    def _abs_score(v, lo, hi):
        """Map a raw metric onto 0-100 using fixed, asset-agnostic bounds — used
        only when there's no peer group to rank against."""
        if v is None:
            return 50.0
        return float(np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0))

    comps = ["slope_252d", "rel_12_1", "ir_252d", "inv_downcap_lt", "crisis_alpha_lt", "inv_dd"]
    src = {t: {"slope_252d": m["slope_252d"], "rel_12_1": m["rel_12_1"],
               "ir_252d": m["rel_12_1"],  # rel_12_1 already IS a 252D-horizon relative-strength figure
               "inv_downcap_lt": None if m["down_cap_lt"] is None else 1.0 - m["down_cap_lt"],
               "crisis_alpha_lt": m["crisis_alpha_lt"], "inv_dd": -abs(m["max_dd"])} for t, m in raw.items()}
    if lt_score_is_relative:
        ranks = {c: _pct({t: src[t][c] for t in raw}) for c in comps}
        lt_score = {t: float(np.mean([ranks[c][t] for c in comps])) for t in raw}
    else:
        # Fixed bounds chosen from typical ranges for each metric so a lone
        # ticker still gets a real, if rougher, absolute long-term read.
        bounds = {"slope_252d": (-0.003, 0.003), "rel_12_1": (-0.30, 0.30), "ir_252d": (-0.30, 0.30),
                  "inv_downcap_lt": (0.0, 2.0), "crisis_alpha_lt": (-0.03, 0.03), "inv_dd": (-0.60, 0.0)}
        lt_score = {t: float(np.mean([_abs_score(src[t][c], *bounds[c]) for c in comps])) for t in raw}


    for ticker, m in raw.items():
        score = 15.0
        if m["vol_adj_rs"] > 0:
            score += min(m["vol_adj_rs"] * 10, 18)
        if m["is_breakout"]:
            score += 12
        if m["r2"] > 0.4 and m["slope_63d"] > 0:
            score += 12
        if m["regime"] == "Bull": score += 5
        elif m["regime"] == "Bear": score -= 5
        if m["accel"] is not None:
            score += max(-8.0, min(m["accel"] * 800.0, 12.0))
        pct = float(percentile_rank[ticker])
        score += (pct - 50.0) / 50.0 * 10.0
        score -= min(abs(m["max_dd_63"]) * 25.0, 15.0)
        if m["dd_efficiency"] is not None:
            score += max(-6.0, min(m["dd_efficiency"] * 15.0, 10.0))
        trading_score = max(0.0, min(100.0, score))
        final_score = max(0.0, min(100.0, lt_score[ticker] if mode == "longterm" else trading_score))

        if final_score >= 80.0: status = "🏆 LEADER"
        elif final_score >= 60.0: status = "📈 IMPROVING"
        elif final_score >= 40.0: status = "➖ NEUTRAL"
        elif final_score >= 20.0: status = "📉 WEAKENING"
        else: status = "🔻 LAGGARD"

        records.append({
            "Asset": ticker, "Health Score": round(final_score, 1),
            "LT Score": round(lt_score[ticker], 1), "Status": status,
            "1M Return": m["ret_1m"], "3M Return": m["ret_3m"], "63D Alpha vs BM": m["rolling_alpha"],
            "Vol-Adjusted RS": m["vol_adj_rs"], "Sector Percentile": round(pct, 1),
            "Sector Rank": f"{int(ordinal_rank[ticker])} of {universe_size}",
            "Trend R²": m["r2"], "RS Acceleration": m["accel"],
            "Drawdown Efficiency": round(m["dd_efficiency"], 2) if m["dd_efficiency"] is not None else None,
            "200D Trend Slope": m["long_slope"], "252D Slope": m["slope_252d"],
            "12-1 Rel Mom": m["rel_12_1"],
            "Up Capture": m["up_cap"], "Down Capture": m["down_cap"],
            "Up Capture (LT)": m["up_cap_lt"], "Down Capture (LT)": m["down_cap_lt"],
            "Crisis Alpha": m["crisis_alpha"], "Crisis Alpha (LT)": m["crisis_alpha_lt"],
            "Regime": m["regime"], "Max Drawdown": m["max_dd"],
            "Real Days": m["real_days"], "Data Fill %": m["fill_ratio"],
            "FX Mismatch": "⚠️" if m["fx_mismatch"] else "",
            "LT Score Basis": "Peer-relative" if lt_score_is_relative else "Absolute (single-asset, no peers)"})
    return records


def _layer_weight(layer_key: str) -> float:
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", layer_key)
    return float(m.group(1)) / 100.0 if m else 1.0


def generate_sizing_stress(start_date: str, end_date: str) -> Dict[str, Any]:
    """Grid-wide anti-fragile lens: cash governor, risk-parity suggested weights
    (benchmark-vol proxy), concentration + liquidity audit."""
    bull, total, section_vols, concentration, liq = 0, 0, {}, [], []
    for sec_key, layers in PORTFOLIO_GRID.items():
        sec = sec_key.split(" (")[0]
        sec_target = SECTION_TARGETS.get(sec, 0.0)
        vols = []
        for lay_key, cfg in layers.items():
            lay_w = _layer_weight(lay_key)
            bm_df, _, _, _ = get_stock_data_isolated([cfg["benchmark"]], start_date, end_date)
            if not bm_df.empty:
                r = bm_df[cfg["benchmark"]].pct_change().dropna()
                if len(r) > 63:
                    vols.append(float(r.tail(63).std() * np.sqrt(252)))
            mem_df, _, _, mem_dq = get_stock_data_isolated(cfg["tickers"], start_date, end_date)
            for rec in generate_automated_scoring(mem_df, cfg["tickers"], cfg["benchmark"], data_quality=mem_dq):
                total += 1
                bull += rec["Regime"] == "Bull"
                eff = sec_target * lay_w / max(len(cfg["tickers"]), 1)
                if eff > 0.025:
                    concentration.append({"Ticker": rec["Asset"], "Section": sec, "Eff. Weight": eff})
                if rec["Asset"] in LIQUIDITY_WATCH:
                    liq.append({"Ticker": rec["Asset"], "Section": sec, "Note": "OTC/thin — cap ~1% eff."})
                elif rec["Data Fill %"] > 0.15:
                    liq.append({"Ticker": rec["Asset"], "Section": sec,
                                "Note": f"High data-fill ({rec['Data Fill %']:.0%}) — treat metrics with caution."})
        if vols:
            section_vols[sec] = float(np.mean(vols))
    breadth = bull / total if total else 0.5
    stress = 0.0
    spy_df, _, _, _ = get_stock_data_isolated(["SPY"], start_date, end_date)
    if not spy_df.empty:
        sr = spy_df["SPY"].pct_change().dropna()
        if len(sr) > 200:
            v63 = float(sr.tail(63).std() * np.sqrt(252))
            med = float((sr.tail(200).rolling(63).std() * np.sqrt(252)).median())
            stress = max(0.0, v63 / med - 1.0) if med > 0 else 0.0
    cash_target = float(np.clip(0.10 + 0.30 * (0.5 - breadth) + 0.10 * stress, 0.05, 0.35))
    inv = {s: 1.0 / v for s, v in section_vols.items() if v > 0}
    invested, tot_inv = 1.0 - cash_target, sum(inv.values())
    sizing = pd.DataFrame([
        {"Section": s, "Target": SECTION_TARGETS.get(s, 0.0),
         "Risk-Parity Suggest": invested * inv[s] / tot_inv if tot_inv else 0.0,
         "Vol Proxy (63D)": section_vols.get(s, np.nan)}
        for s in SECTION_TARGETS if s != "CASH" and s in section_vols])
    sizing["Delta vs Target"] = sizing["Risk-Parity Suggest"] - sizing["Target"]
    return {"breadth": breadth, "vol_stress": stress, "cash_target": cash_target, "sizing_df": sizing,
            "concentration_df": pd.DataFrame(concentration), "liquidity_df": pd.DataFrame(liq)}


# =============================================================================
# FULL UNIVERSE SCAN — every ticker in PORTFOLIO_GRID, scored against its own
# layer benchmark, tagged with Section/Layer/Benchmark for a flat, filterable,
# CSV-exportable master table (feeds other AI agents / external pipelines).
# =============================================================================
UNIVERSE_DISPLAY_COLS = [
    "Section", "Layer", "Asset", "Benchmark", "Status", "Health Score", "LT Score", "LT Score Basis",
    "Regime", "Sector Rank", "Sector Percentile", "1M Return", "3M Return", "63D Alpha vs BM",
    "Vol-Adjusted RS", "Trend R²", "RS Acceleration", "200D Trend Slope", "252D Slope", "12-1 Rel Mom",
    "Drawdown Efficiency", "Max Drawdown", "Up Capture", "Down Capture", "Up Capture (LT)",
    "Down Capture (LT)", "Crisis Alpha", "Crisis Alpha (LT)", "Real Days", "Data Fill %", "FX Mismatch"]


@st.cache_data(ttl=3600)
def generate_full_universe_scan(start_date: str, end_date: str, mode: str = "trading") -> pd.DataFrame:
    """Walks every Section > Layer > ticker in PORTFOLIO_GRID, fetches each layer's
    tickers + its own benchmark in isolation (so one broken layer can't sink the
    scan), scores each ticker against ITS OWN layer benchmark via the same engine
    used elsewhere, and returns one flat table tagged with Section/Layer/Benchmark.
    Tickers that fail to fetch or can't be scored still get a row (Status flags why),
    so the CSV output always accounts for 100% of the configured universe."""
    all_records: List[Dict[str, Any]] = []
    for sec_key, layers in PORTFOLIO_GRID.items():
        sec = sec_key.split(" (")[0]
        for lay_key, cfg in layers.items():
            layer_name = lay_key.split(" (")[0]
            tickers, benchmark = cfg["tickers"], cfg["benchmark"]
            all_req = sorted(set(tickers + [benchmark]))
            price_df, working_list, failed_list, dq = get_stock_data_isolated(all_req, start_date, end_date)
            if price_df.empty or benchmark not in price_df.columns:
                for t in tickers:
                    all_records.append({"Section": sec, "Layer": layer_name, "Asset": t, "Benchmark": benchmark,
                                        "Status": "🚫 BENCHMARK FAILED"})
                continue
            recs = generate_automated_scoring(price_df, tickers, benchmark, mode=mode, data_quality=dq)
            scored_tickers = set()
            for r in recs:
                r["Section"], r["Layer"], r["Benchmark"] = sec, layer_name, benchmark
                all_records.append(r)
                scored_tickers.add(r["Asset"])
            for t in tickers:
                if t == benchmark or t in scored_tickers:
                    continue
                reason = "⚠️ FETCH FAILED" if t in failed_list else "⚠️ SCORING SKIPPED (insufficient overlap)"
                all_records.append({"Section": sec, "Layer": layer_name, "Asset": t, "Benchmark": benchmark,
                                    "Status": reason})
    df = pd.DataFrame(all_records)
    if df.empty:
        return df
    ordered = [c for c in UNIVERSE_DISPLAY_COLS if c in df.columns] + \
              [c for c in df.columns if c not in UNIVERSE_DISPLAY_COLS]
    return df[ordered]


# =============================================================================
# CHART ENGINE
# =============================================================================
def generate_rotational_heatmap(metrics_df: pd.DataFrame) -> go.Figure:
    if metrics_df.empty:
        return go.Figure()
    heatmap_data = metrics_df.set_index("Asset")[["1M Return", "3M Return", "63D Rolling Alpha"]] * 100.0
    fig = px.imshow(heatmap_data, labels=dict(x="Performance Metric", y="Asset", color="Velocity Scale (%)"),
                    x=["1M Window", "3M Window", "63D Alpha"], color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0.0, text_auto=".1f")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white")
    return fig


def plot_intelligence_charts(df_indexed: pd.DataFrame, df_ratios: pd.DataFrame, base_ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])
    palette = ["#1abc9c", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#2ecc71"]
    for idx, col in enumerate(df_indexed.columns):
        if col == base_ticker:
            fig.add_trace(go.Scatter(x=df_indexed.index, y=df_indexed[col], mode='lines',
                        name=f"{col} Baseline", line=dict(color="#2c3e50", width=2.5, dash="dash")), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_indexed.index, y=df_indexed[col], mode='lines',
                        name=f"{col} (Indexed)", line=dict(color=palette[idx % len(palette)], width=1.5)), row=1, col=1)
    for idx, col in enumerate(df_ratios.columns):
        color = palette[idx % len(palette)]
        series = df_ratios[col]
        ma50 = series.rolling(50).mean()
        fig.add_trace(go.Scatter(x=df_ratios.index, y=series, mode='lines', name=f"{col} Ratio Line",
                    line=dict(color=color, width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ratios.index, y=ma50, mode='lines', name=f"{col} 50D MA Filter",
                    line=dict(color=color, width=1, dash="dot"), showlegend=False), row=2, col=1)
    fig.update_layout(template="plotly_white", height=650, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title_text="Base 100 Scale", row=1, col=1)
    fig.update_yaxes(title_text="Alpha Ratio Axis", row=2, col=1)
    return fig


# =============================================================================
# UI
# =============================================================================
st.title("⚡ Structural Allocation & Relative Strength Scoring Engine")
st.caption("Unified build — RS scoring, Long-Term Mode and anti-fragile sizing lens in one file.")

_KNOWN_TICKERS = sorted({t for section in PORTFOLIO_GRID.values() for layer in section.values()
                         for t in (layer["tickers"] + [layer["benchmark"]])})
today = datetime.date.today()
lookback_boundary = today - datetime.timedelta(days=3 * 365)


def run_pipeline(tickers, benchmark, start_date, end_date):
    all_requested = sorted(set(tickers + [benchmark]))
    return get_stock_data_isolated(all_requested, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))


def style_fill(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ''
    if v > 0.30: return 'background-color: #b71c1c; color: white; font-weight: bold;'
    if v > 0.15: return 'background-color: #ef6c00; color: white;'
    return ''


def style_signals(val):
    v = str(val)
    if "🏆 LEADER" in v: return 'background-color: #2e7d32; color: white; font-weight: bold;'
    if "📈 IMPROVING" in v: return 'background-color: #558b2f; color: white; font-weight: bold;'
    if "➖ NEUTRAL" in v: return 'background-color: #455a64; color: white;'
    if "📉 WEAKENING" in v: return 'background-color: #ef6c00; color: white; font-weight: bold;'
    return 'background-color: #b71c1c; color: white; font-weight: bold;'


def style_regime(val):
    if val == "Bull": return 'color: #2e7d32; font-weight: bold;'
    if val == "Bear": return 'color: #c62828; font-weight: bold;'
    return 'color: #607d8b;'


def render_results(price_df, working_list, failed_list, configured_tickers, benchmark_ticker, heading,
                   lt_mode=False, data_quality=None):
    if failed_list:
        failed_targets = [t for t in failed_list if t != benchmark_ticker]
        working_targets = [t for t in working_list if t != benchmark_ticker]
        if failed_targets:
            st.warning(f"⚠️ **Ticker Mismatch / Fetch Failure:** skipped `{', '.join(failed_targets)}`. "
                       f"Processed: `{', '.join(working_targets)}`")
    if benchmark_ticker in failed_list:
        st.error(f"🚨 **Critical Baseline Error:** benchmark `{benchmark_ticker}` failed to load. Execution halted.")
        return
    elif price_df.empty or len(working_list) <= 1:
        st.error("Engine Data Failure: No usable operational target assets were loaded for this request.")
        return
    scored_records = generate_automated_scoring(price_df, configured_tickers, benchmark_ticker,
                                                mode="longterm" if lt_mode else "trading",
                                                data_quality=data_quality)
    metrics_df = pd.DataFrame(scored_records)
    if metrics_df.empty:
        st.warning("No metrics compiled. Check tracking logs for verification.")
        return
    metrics_df = metrics_df.sort_values(by="Health Score", ascending=False)
    if lt_mode:
        st.caption("🛡️ Long-Term Mode active — ranking on the 252D/12-1 rank composite "
                   "(convexity & crisis alpha now computed on real ~252D history, not the 63D trading window).")
        if (metrics_df["LT Score Basis"] == "Absolute (single-asset, no peers)").any():
            st.caption("ℹ️ Only one asset in this run, so LT Score can't be peer-ranked — it's computed from "
                      "fixed absolute bounds on each metric instead. Treat it as a rough directional read, "
                      "not a precise percentile; add more tickers to this search for a proper relative score.")
    fx_flags = metrics_df.loc[metrics_df["FX Mismatch"] != "", "Asset"].tolist()
    stale_flags = metrics_df.loc[metrics_df["Data Fill %"] > 0.15, "Asset"].tolist()
    if fx_flags:
        st.caption(f"⚠️ FX cross-term: `{', '.join(fx_flags)}` are priced in a different currency than "
                   f"`{benchmark_ticker}` — their ratio/slope/momentum metrics embed an uncontrolled FX term.")
    if stale_flags:
        st.caption(f"⚠️ Data-fill warning: `{', '.join(stale_flags)}` have >15% of their price history "
                   f"forward/back-filled (thin trading, halts, or a short listing history) — treat their "
                   f"trend/momentum metrics with extra caution.")
    left_grid, right_heatmap = st.columns([0.65, 0.35])
    with left_grid:
        st.subheader(f"📊 {heading} Leadership Standings (vs {benchmark_ticker})")
        display_cols = [c for c in metrics_df.columns if c not in ("Up Capture (LT)", "Down Capture (LT)",
                                                                    "Crisis Alpha (LT)", "Real Days",
                                                                    "LT Score Basis")]
        st.dataframe(
            metrics_df[display_cols].style.map(style_signals, subset=["Status"])
            .map(style_regime, subset=["Regime"]).map(style_fill, subset=["Data Fill %"])
            .background_gradient(cmap="Blues", subset=["Health Score"])
            .format({
                "1M Return": "{:.2%}", "3M Return": "{:.2%}", "63D Alpha vs BM": "{:+.2%}",
                "Vol-Adjusted RS": "{:.2f}", "Sector Percentile": "{:.0f}", "Trend R²": "{:.2f}",
                "RS Acceleration": "{:+.5f}", "Drawdown Efficiency": "{:+.2f}",
                "200D Trend Slope": "{:.4f}", "Max Drawdown": "{:.2%}", "LT Score": "{:.1f}",
                "252D Slope": "{:+.4f}", "12-1 Rel Mom": "{:+.2%}", "Up Capture": "{:.2f}",
                "Down Capture": "{:.2f}", "Crisis Alpha": "{:+.4f}", "Data Fill %": "{:.1%}"
            }, na_rep="N/A"),
            hide_index=True, use_container_width=True, height=330)
        with st.expander("🔎 Long-Term convexity detail (252D-window capture ratios & crisis alpha)"):
            lt_cols = ["Asset", "Up Capture (LT)", "Down Capture (LT)", "Crisis Alpha (LT)", "Real Days"]
            st.dataframe(metrics_df[lt_cols].style.format(
                {"Up Capture (LT)": "{:.2f}", "Down Capture (LT)": "{:.2f}", "Crisis Alpha (LT)": "{:+.4f}"},
                na_rep="N/A — <253 real trading days"), hide_index=True, use_container_width=True)
    with right_heatmap:
        st.subheader("🔥 Layer Performance Heatmap")
        st.plotly_chart(generate_rotational_heatmap(
            metrics_df.rename(columns={"63D Alpha vs BM": "63D Rolling Alpha"})), use_container_width=True)
    st.markdown("---")
    st.subheader("📉 Historical Trend & Crossover Analytics")
    ratios_df = pd.DataFrame(index=price_df.index)
    for ticker in configured_tickers:
        if ticker in price_df.columns and ticker != benchmark_ticker:
            ratios_df[f"{ticker}/{benchmark_ticker}"] = price_df[ticker] / price_df[benchmark_ticker]
    indexed_df = (price_df[working_list] / price_df[working_list].iloc[0]) * 100
    st.plotly_chart(plot_intelligence_charts(indexed_df, ratios_df, benchmark_ticker), use_container_width=True)


# --- SIDEBAR: grid rails ---
st.sidebar.header("Allocation Framework Alignment")
selected_section = st.sidebar.selectbox("Target Core Section", list(PORTFOLIO_GRID.keys()))
selected_layer = st.sidebar.selectbox("Structural Sub-Layer", list(PORTFOLIO_GRID[selected_section].keys()))
layer_config = PORTFOLIO_GRID[selected_section][selected_layer]
configured_tickers = layer_config["tickers"]
benchmark_ticker = st.sidebar.text_input("Assigned Baseline Reference Asset", value=layer_config["benchmark"]).strip().upper()
lt_mode = st.sidebar.checkbox("🛡️ Long-Term Mode", value=False,
    help="Rank on the 252D/12-1 rank-based composite (with convexity & crisis alpha) instead of the 63D trading-horizon score.")
start_date = st.sidebar.date_input("Start Date Profile", value=lookback_boundary, key="grid_start")
end_date = st.sidebar.date_input("End Date Profile", value=today, key="grid_end")
if start_date >= end_date:
    st.sidebar.error("🚫 Start Date must be earlier than End Date.")
st.sidebar.markdown("---")
execute_run = st.sidebar.button("Run Analytics Engine", type="primary", use_container_width=True,
                                disabled=(start_date >= end_date))

# --- SIDEBAR: ad-hoc ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Ad-Hoc Ticker Search")
adhoc_selection = st.sidebar.multiselect("Search or type a ticker", options=_KNOWN_TICKERS, default=[],
    accept_new_options=True, max_selections=15, placeholder="e.g. AAPL, NVDA, BTC-USD…")
adhoc_tickers = sorted({t.strip().upper() for t in adhoc_selection if t and t.strip()})
adhoc_benchmark = st.sidebar.text_input("Ad-Hoc Baseline Reference Asset", value="SPY", key="adhoc_bm").strip().upper()
adhoc_start_date = st.sidebar.date_input("Ad-Hoc Start Date", value=lookback_boundary, key="adhoc_start")
adhoc_end_date = st.sidebar.date_input("Ad-Hoc End Date", value=today, key="adhoc_end")
if adhoc_start_date >= adhoc_end_date:
    st.sidebar.error("🚫 Start Date must be earlier than End Date.")
execute_adhoc = st.sidebar.button("Run Ad-Hoc Lookup", use_container_width=True,
                                  disabled=(adhoc_start_date >= adhoc_end_date or len(adhoc_tickers) == 0))

# --- MAIN PANEL ---
if execute_run:
    with st.spinner(f"Extracting historical data arrays for {selected_layer}..."):
        price_df, working_list, failed_list, data_quality = run_pipeline(
            configured_tickers, benchmark_ticker, start_date, end_date)
    st.session_state["grid_result"] = {"price_df": price_df, "working_list": working_list, "failed_list": failed_list,
                                       "configured_tickers": configured_tickers, "benchmark_ticker": benchmark_ticker,
                                       "heading": selected_layer, "data_quality": data_quality}
if execute_adhoc:
    with st.spinner(f"Extracting historical data arrays for {len(adhoc_tickers)} ad-hoc ticker(s)..."):
        price_df, working_list, failed_list, data_quality = run_pipeline(
            adhoc_tickers, adhoc_benchmark, adhoc_start_date, adhoc_end_date)
    st.session_state["adhoc_result"] = {"price_df": price_df, "working_list": working_list, "failed_list": failed_list,
                                        "data_quality": data_quality,
                                        "configured_tickers": adhoc_tickers, "benchmark_ticker": adhoc_benchmark,
                                        "heading": "Ad-Hoc Search"}

tab_grid, tab_adhoc, tab_universe, tab_sizing = st.tabs(
    ["📊 Portfolio Grid", "🔍 Ad-Hoc Search", "🗂️ Full Ticker Universe", "🛡️ Sizing & Stress"])
with tab_grid:
    result = st.session_state.get("grid_result")
    if result:
        render_results(result["price_df"], result["working_list"], result["failed_list"],
                       result["configured_tickers"], result["benchmark_ticker"], result["heading"],
                       lt_mode=lt_mode, data_quality=result.get("data_quality"))
    else:
        st.info("💡 Select a portfolio segment from the sidebar menus and run the engine.")
with tab_adhoc:
    result = st.session_state.get("adhoc_result")
    if result:
        render_results(result["price_df"], result["working_list"], result["failed_list"],
                       result["configured_tickers"], result["benchmark_ticker"], result["heading"],
                       lt_mode=lt_mode, data_quality=result.get("data_quality"))
    else:
        st.info("🔍 Search or type any ticker in the sidebar's Ad-Hoc box, set a baseline, and run the lookup.")
with tab_universe:
    st.subheader("🗂️ Full Ticker Universe")
    st.caption("Every ticker configured in `PORTFOLIO_GRID`, scored against its **own layer's benchmark** "
               "(not the sidebar layer) — one flat master table with status, regime and the full "
               "long-term technical set. Filter/sort here, or export the CSV for use in other AI agents.")
    run_universe = st.button("🗂️ Run Full Universe Scan", type="primary", use_container_width=True,
                             help="Scans every Section/Layer in the grid using the dates set in the sidebar's "
                                  "'Target Core Section' panel. Fetches each layer in isolation, so one broken "
                                  "ticker or benchmark can't sink the rest of the scan.")
    if run_universe:
        with st.spinner("Scanning the entire portfolio grid, layer by layer — this covers the full universe "
                        "and can take a minute..."):
            st.session_state["universe_result"] = generate_full_universe_scan(
                start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
                mode="longterm" if lt_mode else "trading")

    udf = st.session_state.get("universe_result")
    if udf is None or udf.empty:
        st.info("💡 Click **Run Full Universe Scan** to build the master table (uses the sidebar's grid "
               "Start/End Date and Long-Term Mode setting).")
    else:
        ok_mask = udf["Health Score"].notna() if "Health Score" in udf.columns else pd.Series(False, index=udf.index)
        st.caption(f"🛡️ Long-Term Mode: {'ON — ranked on 252D/12-1 composite' if lt_mode else 'OFF — 63D trading score'}"
                  f" · {int(ok_mask.sum())} of {len(udf)} tickers fully scored "
                  f"· {udf['Section'].nunique()} sections, {udf['Layer'].nunique()} layers.")

        f1, f2, f3, f4 = st.columns([1, 1, 1, 1.2])
        with f1:
            sec_pick = st.multiselect("Section", sorted(udf["Section"].dropna().unique()), key="uni_sec")
        with f2:
            status_pick = st.multiselect("Status", sorted(udf["Status"].dropna().unique()), key="uni_status")
        with f3:
            regime_opts = sorted(udf["Regime"].dropna().unique()) if "Regime" in udf.columns else []
            regime_pick = st.multiselect("Regime", regime_opts, key="uni_regime")
        with f4:
            ticker_search = st.text_input("Search ticker", "", key="uni_search",
                                          placeholder="e.g. NVDA, CCJ…").strip().upper()

        s1, s2, s3 = st.columns([1.4, 1, 1])
        sortable_cols = [c for c in UNIVERSE_DISPLAY_COLS if c in udf.columns
                         and c not in ("Section", "Layer", "Asset", "Benchmark", "Status")]
        with s1:
            sort_col = st.selectbox("Sort by", sortable_cols,
                                    index=sortable_cols.index("Health Score") if "Health Score" in sortable_cols else 0)
        with s2:
            sort_dir = st.radio("Order", ["Descending", "Ascending"], horizontal=True, key="uni_dir")
        with s3:
            hide_unscored = st.checkbox("Hide unscored/failed rows", value=False)

        filtered = udf.copy()
        if sec_pick: filtered = filtered[filtered["Section"].isin(sec_pick)]
        if status_pick: filtered = filtered[filtered["Status"].isin(status_pick)]
        if regime_pick: filtered = filtered[filtered["Regime"].isin(regime_pick)]
        if ticker_search: filtered = filtered[filtered["Asset"].str.upper().str.contains(ticker_search, na=False)]
        if hide_unscored and "Health Score" in filtered.columns:
            filtered = filtered[filtered["Health Score"].notna()]
        filtered = filtered.sort_values(by=sort_col, ascending=(sort_dir == "Ascending"), na_position="last")

        display_cols = [c for c in UNIVERSE_DISPLAY_COLS if c in filtered.columns]
        styler = filtered[display_cols].style
        if "Status" in display_cols:
            styler = styler.map(style_signals, subset=["Status"])
        if "Regime" in display_cols:
            styler = styler.map(style_regime, subset=["Regime"])
        if "Data Fill %" in display_cols:
            styler = styler.map(style_fill, subset=["Data Fill %"])
        if "Health Score" in display_cols:
            styler = styler.background_gradient(cmap="Blues", subset=["Health Score"])
        styler = styler.format({
            "1M Return": "{:.2%}", "3M Return": "{:.2%}", "63D Alpha vs BM": "{:+.2%}",
            "Vol-Adjusted RS": "{:.2f}", "Sector Percentile": "{:.0f}", "Trend R²": "{:.2f}",
            "RS Acceleration": "{:+.5f}", "Drawdown Efficiency": "{:+.2f}", "200D Trend Slope": "{:.4f}",
            "Max Drawdown": "{:.2%}", "Health Score": "{:.1f}", "LT Score": "{:.1f}", "252D Slope": "{:+.4f}",
            "12-1 Rel Mom": "{:+.2%}", "Up Capture": "{:.2f}", "Down Capture": "{:.2f}",
            "Up Capture (LT)": "{:.2f}", "Down Capture (LT)": "{:.2f}", "Crisis Alpha": "{:+.4f}",
            "Crisis Alpha (LT)": "{:+.4f}", "Data Fill %": "{:.1%}"
        }, na_rep="—")
        st.dataframe(styler, hide_index=True, use_container_width=True, height=560)
        st.caption(f"Showing {len(filtered)} of {len(udf)} tickers.")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️ Download filtered view (CSV)",
                               filtered[display_cols].to_csv(index=False).encode("utf-8"),
                               file_name="ticker_universe_filtered.csv", mime="text/csv",
                               use_container_width=True)
        with dl2:
            st.download_button("⬇️ Download full universe, all columns (CSV)",
                               udf.to_csv(index=False).encode("utf-8"),
                               file_name="ticker_universe_full.csv", mime="text/csv",
                               use_container_width=True,
                               help="Unfiltered, every scored/raw column — the file to hand to another AI "
                                    "agent or downstream pipeline.")
with tab_sizing:
    if st.button("🛡️ Run Sizing & Stress Scan", type="primary"):
        with st.spinner("Scanning full grid: breadth, vol stress, concentration, liquidity..."):
            st.session_state["sizing_result"] = generate_sizing_stress(
                start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    ss = st.session_state.get("sizing_result")
    if ss:
        c1, c2, c3 = st.columns(3)
        c1.metric("Grid Breadth (% Bull)", f"{ss['breadth']:.0%}")
        c2.metric("SPY Vol Stress", f"{ss['vol_stress']:+.2f}")
        c3.metric("Suggested Dry Powder", f"{ss['cash_target']:.1%}",
                  help="10% base + 30%*(50%-breadth) + 10%*vol-stress, clipped 5-35%. Rises in stress = systematic buying power.")
        st.subheader("Risk-Parity Suggested Weights vs Targets")
        st.dataframe(ss["sizing_df"].style.format({"Target": "{:.1%}", "Risk-Parity Suggest": "{:.1%}",
                     "Vol Proxy (63D)": "{:.1%}", "Delta vs Target": "{:+.1%}"}), hide_index=True, use_container_width=True)
        st.caption("Sizing lens assumes equal-weight within layers; BTC satellites (90/10 cold wallet) are intentionally oversized here to flag risk concentration.")
        st.subheader("Concentration Flags (eff. weight > 2.5%)")
        if not ss["concentration_df"].empty:
            st.dataframe(ss["concentration_df"].style.format({"Eff. Weight": "{:.2%}"}), hide_index=True)
        else:
            st.success("No single-name concentration above 2.5%.")
        st.subheader("Liquidity Watchlist")
        if not ss["liquidity_df"].empty:
            st.dataframe(ss["liquidity_df"], hide_index=True)
        else:
            st.success("No OTC/thin-liquidity names flagged.")
    else:
        st.info("🛡️ Run the scan to see the anti-fragile sizing lens (uses the sidebar grid dates).")
