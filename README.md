# AI Trader Dashboard

A comprehensive web-based UI for the AI Trading backtesting and live signal system.

## Features

### 📈 Backtesting
- Interactive parameter selection
- Real-time backtesting execution
- Performance metrics display
- Interactive charts with Plotly
- Trade history analysis
- Downloadable reports

### 🎯 Live Signals
- Multi-ticker signal generation
- LLM or fallback decision modes
- Actionable signal identification
- Auto-refresh capabilities
- Signal export functionality

### ⚙️ Optimization
- Multi-ticker parameter optimization
- Configurable trial counts
- Progress tracking
- Results visualization
- Automatic parameter saving

### 📊 Portfolio Analysis
- Multi-ticker portfolio backtesting
- Portfolio-level metrics
- Individual ticker performance
- Return distribution charts
- Export capabilities

### 🔧 Settings
- Ticker parameter management
- Add/edit/remove tickers
- Configuration import/export
- Data management tools

## Installation

### Option 1: Automatic Setup
```bash
./launch_ui.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Launch the UI
streamlit run ui.py
```

## Usage

1. **Start the Application**
   - Run `./launch_ui.sh` or `streamlit run ui.py`
   - The app will open at `http://localhost:8501`

2. **Backtesting**
   - Navigate to "📈 Backtest" page
   - Select ticker, date range, and parameters
   - Click "Run Backtest" to see results

3. **Live Signals**
   - Go to "🎯 Live Signals" page
   - Select tickers and date
   - Generate signals for trading decisions

4. **Optimization**
   - Use "⚙️ Optimization" page
   - Select tickers to optimize
   - Set trial count and run optimization

5. **Portfolio Analysis**
   - Visit "📊 Portfolio Analysis" page
   - Select multiple tickers for portfolio view
   - Analyze combined performance

6. **Configuration**
   - Manage tickers in "🔧 Settings" page
   - Import/export configurations
   - Modify ticker parameters

## Requirements

- Python 3.8+
- Streamlit 1.28+
- Plotly 5.15+
- All dependencies in `requirements.txt`

## File Structure

```
aitrader/
├── main.py              # Core trading logic
├── ui.py                # Streamlit web interface
├── issi.json           # Ticker configurations
├── requirements.txt     # Python dependencies
├── launch_ui.sh        # Launcher script
└── README.md           # This file
```

## Key Components

### Interactive Charts
- Price action with EMAs and Bollinger Bands
- RSI indicator with overbought/oversold levels
- Volume analysis
- Trade entry/exit markers
- Equity curve visualization

### Performance Metrics
- Total trades and win rate
- Return percentage and expectancy
- Profit factor and maximum drawdown
- Risk-adjusted metrics

### Data Management
- Automatic data fetching from yfinance
- Technical indicator calculations
- Signal generation (LLM or rule-based)
- Portfolio-level aggregation

## Tips

1. **First Time Setup**
   - Ensure `issi.json` exists with ticker configurations
   - Test with a small date range first
   - Verify internet connection for data fetching

2. **Performance**
   - Use smaller trial counts for faster optimization
   - Limit number of tickers for quicker analysis
   - Close unused browser tabs for better performance

3. **Customization**
   - Modify ticker parameters in Settings
   - Adjust risk percentages based on strategy
   - Export configurations for backup

## Troubleshooting

**Port Already in Use**
```bash
streamlit run ui.py --server.port 8502
```

**Missing Dependencies**
```bash
pip3 install --upgrade -r requirements.txt
```

**Data Loading Issues**
- Check internet connection
- Verify ticker symbols are correct
- Ensure sufficient historical data exists

**Memory Issues**
- Reduce number of tickers
- Use shorter date ranges
- Close other applications

## Advanced Usage

### Custom Parameters
Modify `issi.json` to add custom tickers:
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "ema_short": 20,
    "ema_long": 50,
    "rsi": 14
  }
]
```

### Batch Operations
- Use multi-select for bulk operations
- Export results for further analysis
- Import optimized parameters

### Integration
- Results can be exported as CSV
- Compatible with external analysis tools
- API endpoints can be added for automation

## Support

For issues or feature requests:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure data files exist and are readable
4. Review console output for error messages

## License

This project is for educational and research purposes. Use at your own risk for actual trading decisions.
