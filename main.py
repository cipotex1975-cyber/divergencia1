"""
Main execution script for running the backtest
"""

import backtrader as bt
import pandas as pd
from datetime import datetime
import os
import sys

from data_loader import DataLoader, load_tickers
from strategy import MultiTimeframeStrategy
import config


class PandasData_15M(bt.feeds.PandasData):
    """Custom data feed for 15-minute data"""
    pass


class PandasData_1H(bt.feeds.PandasData):
    """Custom data feed for 1-hour data"""
    pass


class PandasData_4H(bt.feeds.PandasData):
    """Custom data feed for 4-hour data"""
    pass


def prepare_data_for_backtrader(df):
    """
    Prepare DataFrame for Backtrader
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame formatted for Backtrader
    """
    if df is None or df.empty:
        return None
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Rename columns to match Backtrader expectations
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing column {col}")
            return None
    
    return df[required_cols]


def run_backtest(ticker, data_dict):
    """
    Run backtest for a single ticker
    
    Args:
        ticker: Stock symbol
        data_dict: Dictionary with timeframe data
        
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*60}")
    print(f"Running backtest for {ticker}")
    print(f"{'='*60}")
    
    # Prepare data for Backtrader
    data_15m = prepare_data_for_backtrader(data_dict['15m'])
    data_1h = prepare_data_for_backtrader(data_dict['1h'])
    data_4h = prepare_data_for_backtrader(data_dict['4h'])
    
    if data_15m is None or data_1h is None or data_4h is None:
        print(f"Error: Missing data for {ticker}")
        return None
    
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(MultiTimeframeStrategy)
    
    # Add data feeds (order matters: 15M, 1H, 4H)
    feed_15m = PandasData_15M(dataname=data_15m, name=ticker)
    feed_1h = PandasData_1H(dataname=data_1h, name=ticker)
    feed_4h = PandasData_4H(dataname=data_4h, name=ticker)
    
    cerebro.adddata(feed_15m)
    cerebro.adddata(feed_1h)
    cerebro.adddata(feed_4h)
    
    # Set initial cash
    cerebro.broker.setcash(config.INITIAL_CASH)
    
    # Set commission
    cerebro.broker.setcommission(commission=config.COMMISSION)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Print starting conditions
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Run backtest
    results = cerebro.run()
    strat = results[0]
    
    # Print ending conditions
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # Extract analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    # Print analysis
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS FOR {ticker}")
    print(f"{'='*60}")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")
    print(f"Total Return: {returns.get('rtot', 'N/A'):.2%}")
    
    if 'total' in trades:
        print(f"\nTotal Trades: {trades['total'].get('total', 0)}")
        print(f"Won Trades: {trades['won'].get('total', 0)}")
        print(f"Lost Trades: {trades['lost'].get('total', 0)}")
        
        if trades['total'].get('total', 0) > 0:
            win_rate = trades['won'].get('total', 0) / trades['total']['total'] * 100
            print(f"Win Rate: {win_rate:.2f}%")
    
    # Plot if configured
    if config.PLOT_RESULTS:
        print(f"\nGenerating plot for {ticker}...")
        try:
            cerebro.plot(style='candlestick', barup='green', bardown='red')
        except Exception as e:
            print(f"Error plotting: {e}")
    
    # Return results
    return {
        'ticker': ticker,
        'final_value': cerebro.broker.getvalue(),
        'sharpe': sharpe.get('sharperatio', None),
        'max_drawdown': drawdown.get('max', {}).get('drawdown', None),
        'total_return': returns.get('rtot', None),
        'total_trades': trades.get('total', {}).get('total', 0),
        'won_trades': trades.get('won', {}).get('total', 0),
        'lost_trades': trades.get('lost', {}).get('total', 0)
    }


def main():
    """Main execution function"""
    print("="*60)
    print("MULTI-TIMEFRAME DIVERGENCE TRADING STRATEGY")
    print("="*60)
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load tickers
    tickers = load_tickers()
    
    if not tickers:
        print("Error: No tickers loaded")
        return
    
    print(f"\nLoaded {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Store all results
    all_results = []
    
    # Iterate through each ticker
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        
        try:
            # Download data
            loader = DataLoader(ticker)
            data_dict = loader.download_all_timeframes()
            
            # Check if we have all required data
            #if not all(data_dict.values()):
            if any(df is None or df.empty for df in data_dict.values()):
                print(f"Skipping {ticker} - incomplete data")
                continue
            
            # Run backtest
            result = run_backtest(ticker, data_dict)
            
            if result:
                all_results.append(result)
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    # Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{config.RESULTS_DIR}/backtest_summary_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL BACKTESTS")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
        print(f"\nResults saved to {filename}")
    else:
        print("\nNo results to save")


if __name__ == "__main__":
    main()
