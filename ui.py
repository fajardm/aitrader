"""
AI Trader - Web UI
===================
A Streamlit interface for the AI Trading backtesting and live signal system.
Run with: streamlit run ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta

# Import the main trading functions
from main import (
    load_ticker, load_ohlcv, build_dataset, simulate, run_optuna,
    fallback_decision, call_llm, render_prompt, TickerParam
)

# Import watchlist configuration
from watchlist_config import get_watchlist, get_watchlist_names

# Page configuration
st.set_page_config(
    page_title="AI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .warning-card {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    .error-card {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def load_tickers_safe():
    """Load tickers with error handling"""
    try:
        return load_ticker('./issi.json')
    except FileNotFoundError:
        st.error("issi.json file not found. Please ensure the file exists.")
        return []
    except Exception as e:
        st.error(f"Error loading tickers: {str(e)}")
        return []

def create_plotly_chart(df, ticker_symbol, trades_df=None):
    """Create interactive Plotly chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & EMAs', 'RSI', 'Volume'),
        row_width=[0.2, 0.2, 0.1]
    )
    
    # Price and EMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_SHORT'], name='EMA Short', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_LONG'], name='EMA Long', line=dict(color='#2ca02c')),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'BB_UP' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_LOW'], name='BB Lower', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
    
    # Trade markers
    if trades_df is not None and not trades_df.empty:
        fig.add_trace(
            go.Scatter(
                x=trades_df['entry_date'], y=trades_df['entry'],
                mode='markers', name='Entry', marker=dict(symbol='triangle-up', size=10, color='green')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=trades_df['exit_date'], y=trades_df['exit'],
                mode='markers', name='Exit', marker=dict(symbol='triangle-down', size=10, color='red')
            ),
            row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker=dict(color='lightblue')),
            row=3, col=1
        )
    
    fig.update_layout(
        title=f'{ticker_symbol} - Technical Analysis',
        xaxis_title='Date',
        height=800,
        showlegend=True
    )
    
    return fig

def display_metrics(metrics):
    """Display trading metrics in a nice format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", int(metrics['trades']))
        st.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
    
    with col2:
        st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        st.metric("Expectancy R", f"{metrics['expectancy_R']:.3f}")
    
    with col3:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    
    with col4:
        st.metric("Final Equity", f"${metrics['final_equity']:,.0f}")
        st.metric("Initial Equity", f"${metrics['initial_equity']:,.0f}")

def main():
    st.markdown('<h1 class="main-header">🤖 AI Trader Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📈 Backtest", 
        "🎯 Live Signals", 
        "⚙️ Optimization", 
        "📊 Portfolio Analysis",
        "🔧 Settings"
    ])
    
    # Load tickers
    tickers = load_tickers_safe()
    if not tickers:
        st.stop()
    
    ticker_symbols = [t.symbol for t in tickers]
    
    if page == "📈 Backtest":
        backtest_page(tickers, ticker_symbols)
    elif page == "🎯 Live Signals":
        live_signals_page(tickers, ticker_symbols)
    elif page == "⚙️ Optimization":
        optimization_page(tickers, ticker_symbols)
    elif page == "📊 Portfolio Analysis":
        portfolio_analysis_page(tickers, ticker_symbols)
    elif page == "🔧 Settings":
        settings_page(tickers)

def backtest_page(tickers, ticker_symbols):
    st.header("📈 Backtesting")
    
    # Parameters
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_ticker = st.selectbox("Select Ticker", ticker_symbols)
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        equity = st.number_input("Initial Equity", value=100_000_000, step=1000000)
    
    with col2:
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        use_llm = st.checkbox("Use LLM (instead of fallback)", value=False)
        show_plots = st.checkbox("Show Charts", value=True)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Find ticker params
                ticker_obj = next((t for t in tickers if t.symbol == selected_ticker), None)
                if not ticker_obj:
                    st.error("Ticker not found!")
                    return
                
                # Load data and run backtest
                raw_data = load_ohlcv(selected_ticker, start_date.strftime('%Y-%m-%d'))
                df = build_dataset(raw_data, ticker_obj)
                
                result = simulate(
                    df, 
                    ticker=selected_ticker, 
                    start_idx=0, 
                    init_equity=equity, 
                    risk_pct=risk_pct, 
                    use_llm=use_llm
                )
                
                # Display results
                st.success("✅ Backtest completed!")
                
                # Metrics
                st.subheader("📊 Performance Metrics")
                display_metrics(result['metrics'])
                
                # Charts
                if show_plots:
                    st.subheader("📈 Price Chart")
                    fig = create_plotly_chart(df, selected_ticker, result['trades_df'])
                    st.plotly_chart(fig, width='stretch')
                    
                    # Equity curve
                    st.subheader("💰 Equity Curve")
                    equity_fig = go.Figure()
                    equity_fig.add_trace(go.Scatter(
                        x=result['equity_curve'].index,
                        y=result['equity_curve'].values,
                        name='Equity',
                        line=dict(color='green', width=2)
                    ))
                    equity_fig.update_layout(
                        title='Portfolio Equity Over Time',
                        xaxis_title='Date',
                        yaxis_title='Equity ($)',
                        height=400
                    )
                    st.plotly_chart(equity_fig, width='stretch')
                
                # Trades table
                if not result['trades_df'].empty:
                    st.subheader("🔄 Trade History")
                    st.dataframe(
                        result['trades_df'].round(4),
                        width='stretch',
                        height=300
                    )
                    
                    # Download trades
                    csv = result['trades_df'].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Trades CSV",
                        data=csv,
                        file_name=f"{selected_ticker}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during backtesting: {str(e)}")

def live_signals_page(tickers, ticker_symbols):
    st.header("🎯 Live Trading Signals")
    
    # Watchlist selection
    st.subheader("📋 Watchlist Selection")
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        watchlist_name = st.selectbox("Choose Watchlist", get_watchlist_names(), index=0)
        default_tickers = get_watchlist(watchlist_name)
        # Filter to only include tickers that exist in our configuration
        default_tickers = [t for t in default_tickers if t in ticker_symbols]
    
    with col_w2:
        st.info(f"📊 {watchlist_name.title()} watchlist: {len(default_tickers)} tickers")
    
    # Parameters
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_tickers = st.multiselect(
            "Select Tickers", 
            ticker_symbols, 
            default=default_tickers
        )
    
    with col2:
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        use_llm = st.checkbox("Use LLM for Signals", value=False)
        auto_refresh = st.checkbox("Auto Refresh (15 min)", value=False)
    
    # Manual refresh button
    if st.button("🔄 Generate Signals", type="primary") or auto_refresh:
        signals_container = st.container()
        
        with signals_container:
            if not selected_tickers:
                st.warning("Please select at least one ticker.")
                return
                
            progress_bar = st.progress(0)
            signals_data = []
            
            for i, ticker_symbol in enumerate(selected_tickers):
                try:
                    # Find ticker params
                    ticker_obj = next((t for t in tickers if t.symbol == ticker_symbol), None)
                    if not ticker_obj:
                        continue
                    
                    # Load and process data
                    raw_data = load_ohlcv(ticker_symbol, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
                    df = build_dataset(raw_data, ticker_obj)
                    prevday = df.iloc[-2]
                    currentday = df.iloc[-1]

                    valid_currentday = currentday.name.date() == datetime.now().date()
                    if not valid_currentday:
                        prevday = currentday
            
                    if use_llm:
                        prompt = render_prompt(prevday, ticker=ticker_symbol, risk_pct=risk_pct)
                        decision = call_llm(prompt)
                    else:
                        decision = fallback_decision(prevday)

                    if decision['regime'] == 'no_trade':
                        continue
                        
                    zone_low = decision['zone']['low']
                    zone_high = decision['zone']['high']
                        
                    if currentday.High >= zone_low and currentday.Low <= zone_high:
                        entry_plan = decision['enter']
                        price = max(entry_plan['price'], currentday.Low)
                            
                        signals_data.append({
                            'Ticker': ticker_symbol,
                            'Last Date': currentday.name.date(),
                            'Regime': decision.get('regime', 'N/A'),
                            'Zone': f"{zone_low:.2f} - {zone_high:.2f}",
                            'Entry Type': entry_plan.get('type', 'N/A'),
                            'Entry Price': f"{price:.2f}",
                            'Stop Loss': f"{decision.get('stop_loss', 0):.2f} ({decision.get('stop_loss_pct', 0):.2f}%)",
                            'Take Profits': f"{decision.get('take_profits', [])}",
                            'Position Size': f"{decision.get('position_size_pct', 0):.2f}%",
                            'Confidence': f"{decision.get('confidence', 0):.2f}%",
                            'Current Price': f"{currentday.Close:.2f}",
                            'Valid Currentday': valid_currentday
                        })
                    
                    progress_bar.progress((i + 1) / len(selected_tickers))
                    
                except Exception as e:
                    st.error(f"Error processing {ticker_symbol}: {str(e)}")
            
            # Display signals
            if signals_data:
                st.subheader("🎯 Trading Signals")
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, width='stretch', height=400)
                
                # Export signals
                csv = signals_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Signals CSV",
                    data=csv,
                    file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No signals generated.")
        
        # Auto refresh
        if auto_refresh:
            time.sleep(900)  # 15 minutes
            st.rerun()

def optimization_page(tickers, ticker_symbols):
    st.header("⚙️ Parameter Optimization")
    
    # Watchlist selection
    st.subheader("📋 Watchlist Selection")
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        watchlist_name = st.selectbox("Choose Watchlist", get_watchlist_names(), index=0)
        default_tickers = get_watchlist(watchlist_name)
        # Filter to only include tickers that exist in our configuration
        default_tickers = [t for t in default_tickers if t in ticker_symbols]
    
    with col_w2:
        st.info(f"📊 {watchlist_name.title()} watchlist: {len(default_tickers)} tickers")
    
    # Parameters
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_tickers = st.multiselect(
            "Select Tickers to Optimize", 
            ticker_symbols, 
            default=default_tickers
        )
        n_trials = st.slider("Number of Trials", min_value=10, max_value=1000, value=100, step=10)
        equity = st.number_input("Equity for Testing", value=100_000_000, step=1000000)
    
    with col2:
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        use_llm = st.checkbox("Use LLM in Optimization", value=False)
        start_date = st.date_input("Data Start Date", value=datetime.now() - timedelta(days=365))
    
    if st.button("🚀 Start Optimization", type="primary"):
        if not selected_tickers:
            st.warning("Please select at least one ticker to optimize.")
            return
            
        progress_container = st.container()
        results_container = st.container()
        
        with progress_container:
            st.info("🔧 Starting optimization process...")
            overall_progress = st.progress(0)
            
            optimization_results = []
            
            for i, ticker_symbol in enumerate(selected_tickers):
                st.write(f"Optimizing {ticker_symbol}...")
                
                try:
                    # Load data
                    raw_data = load_ohlcv(ticker_symbol, start_date.strftime('%Y-%m-%d'))
                    
                    # Run optimization
                    study = run_optuna(raw_data, equity, risk_pct, use_llm, n_trials)
                    
                    # Store results
                    optimization_results.append({
                        'Ticker': ticker_symbol,
                        'Best Score': study.best_value,
                        'EMA Short': study.best_params['EMA_SHORT'],
                        'EMA Long': study.best_params['EMA_LONG'],
                        'RSI Period': study.best_params['RSI'],
                        'Trials': n_trials
                    })
                    
                    # Update ticker params
                    ticker_obj = next((t for t in tickers if t.symbol == ticker_symbol), None)
                    if ticker_obj:
                        ticker_obj.ema_short = study.best_params['EMA_SHORT']
                        ticker_obj.ema_long = study.best_params['EMA_LONG']
                        ticker_obj.rsi = study.best_params['RSI']
                    
                    overall_progress.progress((i + 1) / len(selected_tickers))
                    
                except Exception as e:
                    st.error(f"Error optimizing {ticker_symbol}: {str(e)}")
        
        # Display results
        with results_container:
            if optimization_results:
                st.success("✅ Optimization completed!")
                
                st.subheader("🎯 Optimization Results")
                results_df = pd.DataFrame(optimization_results)
                st.dataframe(results_df, width='stretch')
                
                # Save results
                if st.button("💾 Save Optimized Parameters"):
                    try:
                        with open('./issi.json', 'w', encoding='utf-8') as f:
                            json.dump([t.__dict__ for t in tickers], f, indent=2)
                        st.success("✅ Parameters saved to issi.json!")
                    except Exception as e:
                        st.error(f"❌ Error saving parameters: {str(e)}")
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Optimization Results",
                    data=csv,
                    file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

def portfolio_analysis_page(tickers, ticker_symbols):
    st.header("📊 Portfolio Analysis")
    
    # Watchlist selection
    st.subheader("📋 Watchlist Selection")
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        watchlist_name = st.selectbox("Choose Watchlist", get_watchlist_names(), index=0)
        default_tickers = get_watchlist(watchlist_name)
        # Filter to only include tickers that exist in our configuration
        default_tickers = [t for t in default_tickers if t in ticker_symbols]
    
    with col_w2:
        st.info(f"📊 {watchlist_name.title()} watchlist: {len(default_tickers)} tickers")
    
    # Parameters
    selected_tickers = st.multiselect(
        "Select Portfolio Tickers", 
        ticker_symbols, 
        default=default_tickers
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        start_date = st.date_input("Analysis Start Date", value=datetime.now() - timedelta(days=365))
        equity_per_ticker = st.number_input("Equity per Ticker", value=10_000_000, step=1000000)
    
    with col2:
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        use_llm = st.checkbox("Use LLM", value=False)
    
    if st.button("📈 Analyze Portfolio", type="primary") and selected_tickers:
        with st.spinner("Analyzing portfolio..."):
            portfolio_results = []
            total_metrics = {
                'trades': 0,
                'total_return_pct': 0,
                'win_rate_pct': 0,
                'final_equity': 0,
                'initial_equity': 0
            }
            
            progress_bar = st.progress(0)
            
            for i, ticker_symbol in enumerate(selected_tickers):
                try:
                    ticker_obj = next((t for t in tickers if t.symbol == ticker_symbol), None)
                    if not ticker_obj:
                        continue
                    
                    raw_data = load_ohlcv(ticker_symbol, start_date.strftime('%Y-%m-%d'))
                    df = build_dataset(raw_data, ticker_obj)
                    
                    result = simulate(
                        df, 
                        ticker=ticker_symbol, 
                        start_idx=0, 
                        init_equity=equity_per_ticker, 
                        risk_pct=risk_pct, 
                        use_llm=use_llm
                    )
                    
                    metrics = result['metrics']
                    portfolio_results.append({
                        'Ticker': ticker_symbol,
                        'Trades': metrics['trades'],
                        'Return %': metrics['total_return_pct'],
                        'Win Rate %': metrics['win_rate_pct'],
                        'Max DD %': metrics['max_drawdown_pct'],
                        'Profit Factor': metrics['profit_factor'],
                        'Final Equity': metrics['final_equity']
                    })
                    
                    # Aggregate metrics
                    total_metrics['trades'] += metrics['trades']
                    total_metrics['final_equity'] += metrics['final_equity']
                    total_metrics['initial_equity'] += metrics['initial_equity']
                    
                    progress_bar.progress((i + 1) / len(selected_tickers))
                    
                except Exception as e:
                    st.error(f"Error analyzing {ticker_symbol}: {str(e)}")
            
            if portfolio_results:
                # Portfolio summary
                st.subheader("🏆 Portfolio Summary")
                
                total_return = ((total_metrics['final_equity'] - total_metrics['initial_equity']) 
                              / total_metrics['initial_equity'] * 100)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tickers", len(portfolio_results))
                    st.metric("Total Trades", total_metrics['trades'])
                
                with col2:
                    st.metric("Portfolio Return", f"{total_return:.2f}%")
                    st.metric("Final Portfolio Value", f"${total_metrics['final_equity']:,.0f}")
                
                with col3:
                    avg_return = np.mean([r['Return %'] for r in portfolio_results])
                    st.metric("Average Return", f"{avg_return:.2f}%")
                    
                with col4:
                    avg_win_rate = np.mean([r['Win Rate %'] for r in portfolio_results])
                    st.metric("Average Win Rate", f"{avg_win_rate:.2f}%")
                
                # Detailed results
                st.subheader("📊 Individual Ticker Performance")
                portfolio_df = pd.DataFrame(portfolio_results)
                st.dataframe(portfolio_df.round(2), width='stretch')
                
                # Performance chart
                st.subheader("📈 Return Distribution")
                fig = go.Figure(data=[
                    go.Bar(x=portfolio_df['Ticker'], y=portfolio_df['Return %'])
                ])
                fig.update_layout(
                    title='Return % by Ticker',
                    xaxis_title='Ticker',
                    yaxis_title='Return %',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')

def settings_page(tickers):
    st.header("🔧 Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Ticker Management", "� Watchlist Management", "�🔑 API Settings", "📁 Data Management"])
    
    with tab2:
        st.subheader("Watchlist Management")
        
        # Import watchlist functions for management
        from watchlist_config import (
            get_all_watchlists, add_watchlist, update_watchlist, 
            delete_watchlist, add_to_watchlist, remove_from_watchlist
        )
        
        # Display current watchlists
        st.write("**Current Watchlists:**")
        all_watchlists = get_all_watchlists()
        
        for name, symbols in all_watchlists.items():
            with st.expander(f"📊 {name.title()} ({len(symbols)} tickers)"):
                st.write(", ".join(symbols))
                
                # Edit watchlist
                col_edit1, col_edit2 = st.columns([3, 1])
                with col_edit1:
                    new_symbols = st.text_area(
                        f"Edit {name} symbols (comma-separated):",
                        value=", ".join(symbols),
                        key=f"edit_{name}",
                        height=100
                    )
                with col_edit2:
                    if st.button(f"💾 Save", key=f"save_{name}"):
                        new_symbol_list = [s.strip() for s in new_symbols.split(",") if s.strip()]
                        update_watchlist(name, new_symbol_list)
                        st.success(f"Updated {name} watchlist!")
                        st.rerun()
                    
                    if name != "main" and st.button(f"🗑️ Delete", key=f"delete_{name}"):
                        delete_watchlist(name)
                        st.success(f"Deleted {name} watchlist!")
                        st.rerun()
        
        # Add new watchlist
        st.write("**Add New Watchlist:**")
        col_new1, col_new2 = st.columns([1, 1])
        
        with col_new1:
            new_watchlist_name = st.text_input("Watchlist Name")
        
        with col_new2:
            new_watchlist_symbols = st.text_input("Symbols (comma-separated)")
        
        if st.button("➕ Create Watchlist") and new_watchlist_name and new_watchlist_symbols:
            symbol_list = [s.strip() for s in new_watchlist_symbols.split(",") if s.strip()]
            add_watchlist(new_watchlist_name.lower(), symbol_list)
            st.success(f"Created {new_watchlist_name} watchlist!")
            st.rerun()
        
        # Export/Import watchlists
        st.write("**Export/Import Watchlists:**")
        col_exp1, col_exp2 = st.columns([1, 1])
        
        with col_exp1:
            if st.button("📥 Export Watchlists"):
                import json
                watchlist_json = json.dumps({"watchlists": all_watchlists}, indent=2)
                st.download_button(
                    label="Download watchlists.json",
                    data=watchlist_json,
                    file_name="watchlists_export.json",
                    mime="application/json"
                )
        
        with col_exp2:
            uploaded_watchlist = st.file_uploader("Import Watchlists", type=['json'])
            if uploaded_watchlist:
                try:
                    import json
                    watchlist_data = json.load(uploaded_watchlist)
                    if "watchlists" in watchlist_data:
                        from watchlist_config import save_watchlists
                        save_watchlists(watchlist_data)
                        st.success("Watchlists imported successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid watchlist file format.")
                except Exception as e:
                    st.error(f"Error importing watchlists: {str(e)}")
        
        st.info("💡 **Tip:** Watchlists are automatically saved to `watchlists.json` file.")
    
    
    with tab1:
        st.subheader("Ticker Parameters")
        
        # Add new ticker
        with st.expander("➕ Add New Ticker"):
            col1, col2 = st.columns([1, 1])
            with col1:
                new_symbol = st.text_input("Symbol")
                new_name = st.text_input("Name")
            with col2:
                new_ema_short = st.number_input("EMA Short", value=20, min_value=1)
                new_ema_long = st.number_input("EMA Long", value=50, min_value=1)
                new_rsi = st.number_input("RSI Period", value=14, min_value=1)
            
            if st.button("Add Ticker") and new_symbol:
                new_ticker = TickerParam(new_symbol, new_name, new_ema_short, new_ema_long, new_rsi)
                tickers.append(new_ticker)
                st.success(f"Added {new_symbol}")
        
        # Edit existing tickers
        st.subheader("Edit Existing Tickers")
        ticker_data = []
        for ticker in tickers:
            ticker_data.append({
                'Symbol': ticker.symbol,
                'Name': ticker.name,
                'EMA Short': ticker.ema_short,
                'EMA Long': ticker.ema_long,
                'RSI': ticker.rsi
            })
        
        if ticker_data:
            edited_df = st.data_editor(
                pd.DataFrame(ticker_data),
                width='stretch',
                num_rows="dynamic"
            )
            
            if st.button("💾 Save Changes"):
                try:
                    # Update ticker objects
                    updated_tickers = []
                    for _, row in edited_df.iterrows():
                        updated_tickers.append(TickerParam(
                            row['Symbol'], row['Name'], 
                            int(row['EMA Short']), int(row['EMA Long']), int(row['RSI'])
                        ))
                    
                    # Save to file
                    with open('./issi.json', 'w', encoding='utf-8') as f:
                        json.dump([t.__dict__ for t in updated_tickers], f, indent=2)
                    
                    st.success("✅ Ticker settings saved!")
                    
                except Exception as e:
                    st.error(f"❌ Error saving settings: {str(e)}")
    
    with tab2:
        st.subheader("API Configuration")
        st.info("🔑 LLM API settings are currently hardcoded. Future versions will support configuration.")
        
        st.text_area("Groq API Key", value="gsk_p8y...", disabled=True, help="API key is set in the code")
        st.selectbox("LLM Model", ["llama3-8b-8192"], disabled=True)
    
    with tab3:
        st.subheader("Data Management")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("📁 **Export Data**")
            if st.button("📥 Export Ticker Config"):
                ticker_json = json.dumps([t.__dict__ for t in tickers], indent=2)
                st.download_button(
                    label="Download issi.json",
                    data=ticker_json,
                    file_name=f"issi_backup_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.write("📁 **Import Data**")
            uploaded_file = st.file_uploader("Upload ticker config", type=['json'])
            if uploaded_file:
                try:
                    config_data = json.load(uploaded_file)
                    st.json(config_data)
                    if st.button("Import Configuration"):
                        with open('./issi.json', 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2)
                        st.success("✅ Configuration imported!")
                except Exception as e:
                    st.error(f"❌ Error importing: {str(e)}")

if __name__ == "__main__":
    main()
