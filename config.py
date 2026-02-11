"""
Configuration file for the multi-timeframe trading strategy
"""

# Data download settings
TICKERS_FILE = 'tickers.txt'
START_DATE = '2025-11-01'
END_DATE = '2026-02-03'

# Timeframes
TIMEFRAME_4H = '4h'
TIMEFRAME_1H = '1h'
TIMEFRAME_15M = '15m'

# Support/Resistance settings (4H)
SR_LOOKBACK_PERIOD = 20  # Number of bars to look back for swing highs/lows
SR_PROXIMITY_PERCENT = 2.0  # Percentage proximity to consider "near" a level
SR_MIN_TOUCHES = 2  # Minimum touches to confirm a level

# MACD settings (1H)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Divergence detection settings
DIVERGENCE_LOOKBACK = 50  # Bars to look back for divergence
MIN_PIVOT_DISTANCE = 5  # Minimum bars between pivots

# Entry settings (15M)
BREAKOUT_CONFIRMATION_BARS = 2  # Bars to confirm breakout

# Backtest settings
INITIAL_CASH = 100000.0
COMMISSION = 0.001  # 0.1%
POSITION_SIZE_PERCENT = 0.1  # 10% of portfolio per trade
STOP_LOSS_PERCENT = 2.0  # 2% stop loss
TAKE_PROFIT_PERCENT = 6.0  # 6% take profit (3:1 R:R)

# Output settings
RESULTS_DIR = 'results'
SIGNALS_DIR = 'signals'  # Directory for signal charts
PLOT_RESULTS = True
SAVE_SIGNALS = True
SAVE_SIGNAL_CHARTS = True  # Toggle for automatic chart generation
CHART_LOOKBACK_BARS = 100  # Number of bars to show in signal charts
