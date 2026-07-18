import streamlit as st
import datetime
import pandas as pd
from engines import data_engine as de
from engines import chart_engine as ce

st.set_page_config(layout="wide", page_title="Portfolio Grid Intelligence Engine", page_icon="⚡")

st.title("⚡ Structural Allocation & Relative Strength Scoring Engine")
st.caption("Automated analytics parsing asset breakout scores and metrics matched against designated core sector benchmarks.")

# --- Structural Allocation Selection Rails ---
st.sidebar.header("Allocation Framework Alignment")

selected_section = st.sidebar.selectbox("Target Core Section", list(de.PORTFOLIO_GRID.keys()))
selected_layer = st.sidebar.selectbox("Structural Sub-Layer", list(de.PORTFOLIO_GRID[selected_section].keys()))

# Read configuration details out of the active block
layer_config = de.PORTFOLIO_GRID[selected_section][selected_layer]
configured_tickers = layer_config["tickers"]
default_benchmark = layer_config["benchmark"]

# Allow manual fine-tuning if necessary
benchmark_ticker = st.sidebar.text_input("Assigned Baseline Reference Asset", value=default_benchmark).strip().upper()

# Dynamic Time Horizon Setup
today = datetime.date.today()
lookback_boundary = today - datetime.timedelta(days=3*365)
start_date = st.sidebar.date_input("Start Date Profile", value=lookback_boundary)
end_date = st.sidebar.date_input("End Date Profile", value=today)

st.sidebar.markdown("---")
execute_run = st.sidebar.button("Run Analytics Engine", type="primary", use_container_width=True)

if execute_run:
    all_requested_tickers = list(set(configured_tickers + [benchmark_ticker]))
    
    with st.spinner(f"Extracting historical data arrays for {selected_layer}..."):
        price_df = de.get_stock_data(all_requested_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if price_df.empty or benchmark_ticker not in price_df.columns:
            st.error("Engine Data Error: Verify network availability or ticker properties.")
        else:
            # Run automated structural evaluation loops
            scored_records = de.generate_automated_scoring(price_df, configured_tickers, benchmark_ticker)
            metrics_df = pd.DataFrame(scored_records)
            
            if metrics_df.empty:
                st.warning("No metrics compiled. Check tracking logs for verification.")
            else:
                metrics_df = metrics_df.sort_values(by="Health Score", ascending=False)
                
                # Render Split Analysis Panel Setup
                left_grid, right_heatmap = st.columns([0.65, 0.35])
                
                with left_grid:
                    st.subheader(f"📊 {selected_layer} Leadership Standings (vs {benchmark_ticker})")
                    
                    def style_signals(val):
                        if "🚀 BUY" in str(val): return 'background-color: #2e7d32; color: white; font-weight: bold;'
                        if "✋ HOLD" in str(val): return 'background-color: #ef6c00; color: white; font-weight: bold;'
                        return 'background-color: #455a64; color: white;'

                    st.dataframe(
                        metrics_df.style.applymap(style_signals, subset=["Status"])
                        .background_gradient(cmap="Blues", subset=["Health Score"])
                        .format({
                            "1M Return": "{:.2%}", "3M Return": "{:.2%}", "63D Alpha vs BM": "{:+.2%}",
                            "Vol-Adjusted RS": "{:.2f}", "Trend R²": "{:.2f}", "Max Drawdown": "{:.2%}"
                        }),
                        hide_index=True, use_container_width=True, height=330
                    )
                    
                with right_heatmap:
                    st.subheader("🔥 Layer Performance Heatmap")
                    heatmap_fig = ce.generate_rotational_heatmap(metrics_df.rename(columns={"63D Alpha vs BM": "63D Rolling Alpha"}))
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                st.markdown("---")
                
                # Render Bottom Subplot Visualizations
                st.subheader("📉 Historical Trend & Crossover Analytics")
                ratios_df = pd.DataFrame(index=price_df.index)
                for ticker in configured_tickers:
                    if ticker in price_df.columns:
                        ratios_df[f"{ticker}/{benchmark_ticker}"] = price_df[ticker] / price_df[benchmark_ticker]
                
                indexed_df = (price_df / price_df.iloc[0]) * 100
                chart_fig = ce.plot_intelligence_charts(indexed_df, ratios_df, benchmark_ticker)
                st.plotly_chart(chart_fig, use_container_width=True)
else:
    st.info(f"💡 Select a portfolio segment from the sidebar menus. The system will automatically configure and pull variables matching your target assets.")
