# Chartplotv2 ⚡

**Structural Allocation & Relative Strength Scoring Engine**

A single-file [Streamlit](https://streamlit.io/) app that scores stocks, ETFs, and crypto against a chosen benchmark using relative-strength analytics, and helps track a predefined multi-section portfolio allocation ("Portfolio Grid").

## What it does

The app is built around three ideas:

1. **A predefined Portfolio Grid** — a nested structure of Sections (e.g. `INFRA`, `ENERGY & COMMODITY`, `AI / SEMIS`, `EM`, `BTC`, `GOLD`) → Layers (sub-themes with target weights) → Tickers, each layer mapped to its own benchmark.
2. **A relative-strength scoring engine** that pulls historical prices via `yfinance` and computes, per ticker vs. its benchmark:
   - 1M / 3M returns, annualized alpha, tracking error, and vol-adjusted RS (information ratio)
   - 63-day log-ratio regression (slope, R², acceleration) for a "trading" horizon score
   - A rank-based 252-day / 12-1 skip-month "Long-Term Mode" composite, including convexity and crisis-alpha components
   - Breakout detection with hysteresis (MA20/MA50 crossover logic)
   - Up/down capture ratios, max drawdown, and drawdown efficiency
   - Data-quality flags: real vs. forward/backward-filled data points, fill %, and FX currency-mismatch detection (via ticker suffix)
3. **A sizing & stress overlay** — a portfolio-wide "anti-fragile" lens with a cash-target governor (based on market breadth and volatility stress vs. SPY), risk-parity suggested section weights, a concentration audit, and a liquidity watchlist for thinly-traded tickers.

## App layout

The app opens with a sidebar and four tabs:

- **📊 Portfolio Grid** — pick a Section and Layer from the predefined grid, choose a benchmark and date range, and run the scoring engine on that layer's tickers.
- **🔍 Ad-Hoc Search** — score any ticker(s) you type or select against a benchmark of your choice, independent of the predefined grid.
- **🗂️ Full Ticker Universe** — a flat, filterable, CSV-exportable table scoring every ticker in the Portfolio Grid against its own layer's benchmark (cached for an hour).
- **🛡️ Sizing & Stress** — grid-wide cash-target suggestion, risk-parity weights vs. targets, concentration flags, and a liquidity watchlist.

Each results view shows leadership standings, a layer performance heatmap, and historical trend/crossover charts (Plotly).

## Installation

```bash
git clone https://github.com/Faisalkttr/Chartplotv2.git
cd Chartplotv2
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- [streamlit](https://streamlit.io/) >= 1.35.0
- [yfinance](https://github.com/ranaroussi/yfinance) >= 0.2.40
- [pandas](https://pandas.pydata.org/) >= 2.0.0
- [plotly](https://plotly.com/python/) >= 5.18.0
- [scikit-learn](https://scikit-learn.org/) >= 1.3.0
- [matplotlib](https://matplotlib.org/) >= 3.7.0

## Usage

```bash
streamlit run app.py
```

This launches the app in your browser (by default at `http://localhost:8501`). From there:

1. Use the sidebar to pick a portfolio Section/Layer (or switch to Ad-Hoc Search for custom tickers).
2. Optionally enable **Long-Term Mode** to switch the scoring basis from the 63-day trading window to the 252-day/12-1 long-term composite.
3. Set a date range and click **Run Analytics Engine** (or **Run Ad-Hoc Lookup**).
4. Review results in the tabs — standings tables, heatmaps, trend charts — and check the **Sizing & Stress** tab for portfolio-level risk signals.

## Notes & caveats

- Price data is fetched live from Yahoo Finance via `yfinance`; results depend on data availability and quality for each ticker.
- Currency inference is a best-effort heuristic based on ticker suffix (e.g. `.T`, `.NS`, `.SR`) — it flags likely FX mismatches but is not a live FX data feed.
- 1M/3M returns for some non-USD layers (e.g. India/GCC) are in local currency.
- The Full Ticker Universe scan is cached for 1 hour (`ttl=3600`) to reduce repeated API calls.

## Disclaimer

This tool is for informational and research purposes only and does not constitute financial advice. Always do your own due diligence before making investment decisions.
