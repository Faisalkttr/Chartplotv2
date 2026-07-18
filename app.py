# Locate your execution trigger loop inside app.py and replace with this configuration:
if execute_run:
    all_requested_tickers = list(set(configured_tickers + [benchmark_ticker]))
    
    with st.spinner(f"Extracting historical data arrays for {selected_layer}..."):
        # Call updated isolated processing function
        price_df, working_list, failed_list = de.get_stock_data_isolated(
            all_requested_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )
        
        # --- Notification Alert Framework ---
        if failed_list:
            # Highlight only the target ticker failures (filter out benchmark ticker if it failed separately)
            failed_targets = [t for t in failed_list if t != benchmark_ticker]
            working_targets = [t for t in working_list if t != benchmark_ticker]
            
            if failed_targets:
                st.warning(
                    f"⚠️ **Ticker Mismatch / Fetch Failure:** The following assets could not be retrieved and were skipped: "
                    f"`{', '.join(failed_targets)}`. \n\n"
                    f"**Operational layer assets processed successfully:** `{', '.join(working_targets)}`"
                )
        
        # Check if the baseline reference ticker failed entirely
        if benchmark_ticker in failed_list:
            st.error(f"🚨 **Critical Baseline Error:** The core reference benchmark `{benchmark_ticker}` failed to load. Execution halted.")
        elif price_df.empty or len(working_list) <= 1:
            st.error("Engine Data Failure: No usable operational target assets were loaded for this layer profile.")
        else:
            # Run calculations using the valid filtered asset lists
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
                        metrics_df.style.map(style_signals, subset=["Status"])
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
                
                # Render Bottom Subplot Visualizations using ONLY working tickers
                st.subheader("📉 Historical Trend & Crossover Analytics")
                ratios_df = pd.DataFrame(index=price_df.index)
                for ticker in configured_tickers:
                    if ticker in price_df.columns and ticker != benchmark_ticker:
                        ratios_df[f"{ticker}/{benchmark_ticker}"] = price_df[ticker] / price_df[benchmark_ticker]
                
                # Re-index only assets that safely populated data matrices
                indexed_df = (price_df[working_list] / price_df[working_list].iloc[0]) * 100
                chart_fig = ce.plot_intelligence_charts(indexed_df, ratios_df, benchmark_ticker)
                st.plotly_chart(chart_fig, use_container_width=True)
