"""
Chart generation module for divergence signals
Creates visual charts showing price action, MACD, and divergence markers
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from datetime import datetime
import os
import config


class SignalChartGenerator:
    """Generates charts for divergence signals"""
    
    def __init__(self, ticker):
        """
        Initialize chart generator
        
        Args:
            ticker: Stock symbol
        """
        self.ticker = ticker
        self.signals_dir = config.SIGNALS_DIR
        
        # Create signals directory if it doesn't exist
        os.makedirs(self.signals_dir, exist_ok=True)
    
    def generate_divergence_chart(self, price_data, macd_data, signal_type, 
                                  divergence_indices, sr_levels=None, 
                                  entry_level=None):
        """
        Generate a chart showing divergence signal
        
        Args:
            price_data: DataFrame with OHLCV data
            macd_data: DataFrame with MACD values
            signal_type: 'bullish' or 'bearish'
            divergence_indices: Tuple of (prev_idx, curr_idx) where divergence detected
            sr_levels: Dict with 'support' and 'resistance' lists (optional)
            entry_level: Entry price level (optional)
            
        Returns:
            Path to saved chart file
        """
        # Limit data to lookback period
        lookback = config.CHART_LOOKBACK_BARS
        if len(price_data) > lookback:
            price_data = price_data.iloc[-lookback:]
            macd_data = macd_data.iloc[-lookback:]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       sharex=True)
        
        # Plot price panel
        self._plot_price_panel(ax1, price_data, signal_type, 
                              divergence_indices, sr_levels, entry_level)
        
        # Plot MACD panel
        self._plot_macd_panel(ax2, price_data, macd_data, signal_type, 
                             divergence_indices)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')
        
        # Add title
        title = f"{self.ticker} - {signal_type.upper()} DIVERGENCE DETECTED"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save chart
        filepath = self._save_chart(fig, signal_type)
        
        # Close figure to free memory
        plt.close(fig)
        
        return filepath
    
    def _plot_price_panel(self, ax, price_data, signal_type, 
                         divergence_indices, sr_levels, entry_level):
        """Plot price candlestick chart with markers"""
        
        # Plot candlesticks
        for idx in range(len(price_data)):
            date = price_data.index[idx]
            open_price = price_data['Open'].iloc[idx]
            high = price_data['High'].iloc[idx]
            low = price_data['Low'].iloc[idx]
            close = price_data['Close'].iloc[idx]
            
            # Determine color
            color = 'green' if close >= open_price else 'red'
            
            # Draw high-low line
            ax.plot([date, date], [low, high], color='black', linewidth=0.5)
            
            # Draw body
            height = abs(close - open_price)
            bottom = min(open_price, close)
            width = pd.Timedelta(hours=0.5)  # Adjust based on timeframe
            
            rect = Rectangle((date - width/2, bottom), width, height,
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Plot close price line for clarity
        ax.plot(price_data.index, price_data['Close'], 
               color='blue', linewidth=1, alpha=0.5, label='Close Price')
        
        # Add support/resistance levels
        if sr_levels:
            for level in sr_levels.get('support', []):
                ax.axhline(y=level, color='green', linestyle='--', 
                          linewidth=1, alpha=0.5, label='Support')
            
            for level in sr_levels.get('resistance', []):
                ax.axhline(y=level, color='red', linestyle='--', 
                          linewidth=1, alpha=0.5, label='Resistance')
        
        # Add entry level
        if entry_level:
            ax.axhline(y=entry_level, color='orange', linestyle='-', 
                      linewidth=2, alpha=0.7, label=f'Entry: {entry_level:.2f}')
        
        # Mark divergence points
        if divergence_indices:
            prev_idx, curr_idx = divergence_indices
            
            if signal_type == 'bullish':
                # Mark the lows
                if prev_idx in price_data.index:
                    ax.scatter(prev_idx, price_data.loc[prev_idx, 'Low'], 
                             color='lime', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('Low 1', xy=(prev_idx, price_data.loc[prev_idx, 'Low']),
                              xytext=(10, -20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
                
                if curr_idx in price_data.index:
                    ax.scatter(curr_idx, price_data.loc[curr_idx, 'Low'], 
                             color='lime', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('Low 2 (Lower)', xy=(curr_idx, price_data.loc[curr_idx, 'Low']),
                              xytext=(10, -20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
                
                # Draw line connecting the lows
                if prev_idx in price_data.index and curr_idx in price_data.index:
                    ax.plot([prev_idx, curr_idx], 
                           [price_data.loc[prev_idx, 'Low'], 
                            price_data.loc[curr_idx, 'Low']],
                           color='lime', linewidth=2, linestyle='--', alpha=0.7)
            
            elif signal_type == 'bearish':
                # Mark the highs
                if prev_idx in price_data.index:
                    ax.scatter(prev_idx, price_data.loc[prev_idx, 'High'], 
                             color='red', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('High 1', xy=(prev_idx, price_data.loc[prev_idx, 'High']),
                              xytext=(10, 20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                
                if curr_idx in price_data.index:
                    ax.scatter(curr_idx, price_data.loc[curr_idx, 'High'], 
                             color='red', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('High 2 (Higher)', xy=(curr_idx, price_data.loc[curr_idx, 'High']),
                              xytext=(10, 20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                
                # Draw line connecting the highs
                if prev_idx in price_data.index and curr_idx in price_data.index:
                    ax.plot([prev_idx, curr_idx], 
                           [price_data.loc[prev_idx, 'High'], 
                            price_data.loc[curr_idx, 'High']],
                           color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title('Price Action with Divergence Points', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    def _plot_macd_panel(self, ax, price_data, macd_data, signal_type, 
                        divergence_indices):
        """Plot MACD indicator with divergence lines"""
        
        # Plot MACD line
        ax.plot(macd_data.index, macd_data['MACD'], 
               color='blue', linewidth=1.5, label='MACD')
        
        # Plot Signal line
        ax.plot(macd_data.index, macd_data['Signal'], 
               color='red', linewidth=1.5, label='Signal')
        
        # Plot Histogram
        colors = ['green' if val >= 0 else 'red' for val in macd_data['Histogram']]
        ax.bar(macd_data.index, macd_data['Histogram'], 
              color=colors, alpha=0.3, label='Histogram')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Mark divergence points on MACD
        if divergence_indices:
            prev_idx, curr_idx = divergence_indices
            
            if signal_type == 'bullish':
                # Mark MACD lows (should be higher)
                if prev_idx in macd_data.index:
                    ax.scatter(prev_idx, macd_data.loc[prev_idx, 'MACD'], 
                             color='lime', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('MACD Low 1', xy=(prev_idx, macd_data.loc[prev_idx, 'MACD']),
                              xytext=(10, -20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
                
                if curr_idx in macd_data.index:
                    ax.scatter(curr_idx, macd_data.loc[curr_idx, 'MACD'], 
                             color='lime', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('MACD Low 2 (Higher)', xy=(curr_idx, macd_data.loc[curr_idx, 'MACD']),
                              xytext=(10, 20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7))
                
                # Draw line connecting MACD lows
                if prev_idx in macd_data.index and curr_idx in macd_data.index:
                    ax.plot([prev_idx, curr_idx], 
                           [macd_data.loc[prev_idx, 'MACD'], 
                            macd_data.loc[curr_idx, 'MACD']],
                           color='lime', linewidth=2, linestyle='--', alpha=0.7)
            
            elif signal_type == 'bearish':
                # Mark MACD highs (should be lower)
                if prev_idx in macd_data.index:
                    ax.scatter(prev_idx, macd_data.loc[prev_idx, 'MACD'], 
                             color='red', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('MACD High 1', xy=(prev_idx, macd_data.loc[prev_idx, 'MACD']),
                              xytext=(10, 20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                
                if curr_idx in macd_data.index:
                    ax.scatter(curr_idx, macd_data.loc[curr_idx, 'MACD'], 
                             color='red', s=200, marker='o', zorder=5,
                             edgecolors='black', linewidths=2)
                    ax.annotate('MACD High 2 (Lower)', xy=(curr_idx, macd_data.loc[curr_idx, 'MACD']),
                              xytext=(10, -20), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
                
                # Draw line connecting MACD highs
                if prev_idx in macd_data.index and curr_idx in macd_data.index:
                    ax.plot([prev_idx, curr_idx], 
                           [macd_data.loc[prev_idx, 'MACD'], 
                            macd_data.loc[curr_idx, 'MACD']],
                           color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_ylabel('MACD', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('MACD Indicator with Divergence', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    def _save_chart(self, fig, signal_type):
        """
        Save chart to file
        
        Args:
            fig: Matplotlib figure
            signal_type: 'bullish' or 'bearish'
            
        Returns:
            Path to saved file
        """
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_ticker = self.ticker.replace('/', '_').replace('\\', '_')
        filename = f"{clean_ticker}_{signal_type}_divergence_{timestamp}.png"
        filepath = os.path.join(self.signals_dir, filename)
        
        # Save figure
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        
        print(f"  Chart saved to: {filepath}")
        
        return filepath


if __name__ == "__main__":
    # Test chart generation with sample data
    print("Testing chart generator...")
    
    # Create sample data
    dates = pd.date_range(start='2026-01-01', periods=100, freq='1H')
    
    # Sample price data
    np.random.seed(42)
    price_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'High': 101 + np.cumsum(np.random.randn(100) * 0.5),
        'Low': 99 + np.cumsum(np.random.randn(100) * 0.5),
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Adjust High/Low to be consistent
    price_data['High'] = price_data[['Open', 'High', 'Close']].max(axis=1) + 0.5
    price_data['Low'] = price_data[['Open', 'Low', 'Close']].min(axis=1) - 0.5
    
    # Sample MACD data
    macd_data = pd.DataFrame({
        'MACD': np.sin(np.arange(100) * 0.1) * 2,
        'Signal': np.sin(np.arange(100) * 0.1 - 0.2) * 2,
        'Histogram': np.random.randn(100) * 0.5
    }, index=dates)
    
    # Create chart generator
    generator = SignalChartGenerator('TEST')
    
    # Generate bullish divergence chart
    divergence_indices = (dates[30], dates[70])
    sr_levels = {'support': [98, 95], 'resistance': [105, 108]}
    
    filepath = generator.generate_divergence_chart(
        price_data=price_data,
        macd_data=macd_data,
        signal_type='bullish',
        divergence_indices=divergence_indices,
        sr_levels=sr_levels,
        entry_level=102.5
    )
    
    print(f"\nTest chart generated successfully: {filepath}")
