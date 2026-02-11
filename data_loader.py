"""
Data loading module using yfinance
Downloads historical data for multiple timeframes
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import config


class DataLoader:
    """Handles downloading and preparing data from Yahoo Finance"""
    
    def __init__(self, ticker, start_date=None, end_date=None):
        """
        Initialize data loader for a specific ticker
        
        Args:
            ticker: Stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        self.ticker = ticker
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        
    def download_data(self, interval='1h', save_to_file=True):
        """
        Download data for specified interval
        
        Args:
            interval: Data interval ('15m', '1h', '4h', '1d', etc.)
            save_to_file: If True, save downloaded data to CSV file in download folder
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        print(f"Downloading {interval} data for {self.ticker}...")
        
        try:
            # For 15-minute data, Yahoo Finance only allows max 60 days
            # Automatically adjust the date range for 15m interval
            start_date = self.start_date
            end_date = self.end_date
            
            if interval == '15m':
                from datetime import datetime, timedelta
                
                # Parse end_date if it's a string
                if isinstance(end_date, str):
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                else:
                    end_dt = end_date
                
                # Calculate 60 days before end_date
                start_dt = end_dt - timedelta(days=60)
                
                # Format back to string
                start_date = start_dt.strftime('%Y-%m-%d')
                end_date = end_dt.strftime('%Y-%m-%d')
                
                print(f"  Note: 15m data limited to last 60 days ({start_date} to {end_date})")
            
            # Download data from yfinance
            data = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                multi_level_index =False
            )
            
            if data.empty:
                print(f"Warning: No data downloaded for {self.ticker} at {interval}")
                return None
                
            # Clean column names (remove multi-level index if present)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                print(f"Warning: Missing required columns for {self.ticker}")
                return None
            
            # Remove any NaN values
            data = data.dropna()
            
            print(f"Downloaded {len(data)} bars for {self.ticker} ({interval})")
            
            # Save to file if requested
            if save_to_file:
                self._save_data_to_file(data, interval)
            
            return data
            
        except Exception as e:
            print(f"Error downloading data for {self.ticker}: {e}")
            return None
    
    def _save_data_to_file(self, data, interval):
        """
        Save downloaded data to CSV file in download folder
        
        Args:
            data: DataFrame with downloaded data
            interval: Timeframe interval (e.g., '1h', '15m', '4h')
        """
        try:
            # Create download directory if it doesn't exist
            #download_dir = 'download'
            download_dir = 'data'
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            # Generate filename with ticker, interval, and timestamp
            # Format: TICKER_INTERVAL_YYYYMMDD_HHMMSS.csv
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Clean ticker name (remove special characters)
            clean_ticker = self.ticker.replace('/', '_').replace('\\', '_')
            filename = f"{clean_ticker}_{interval}_{timestamp}.csv"
            filepath = os.path.join(download_dir, filename)
            
            # Save to CSV
            data.to_csv(filepath)
            print(f"  Saved to: {filepath}")
            
        except Exception as e:
            print(f"  Warning: Could not save file: {e}")
    
    def download_all_timeframes(self):
        """
        Download data for all required timeframes
        
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        timeframes = {
            '4h': self.download_data('1h'),  # yfinance doesn't have 4h, we'll resample
            '1h': self.download_data('1h'),
            '15m': self.download_data('15m')
        }
        
        # Resample 1h to 4h
        if timeframes['4h'] is not None:
            timeframes['4h'] = self._resample_to_4h(timeframes['4h'])
        
        return timeframes
    
    def _resample_to_4h(self, df_1h):
        """
        Resample 1-hour data to 4-hour bars
        
        Args:
            df_1h: DataFrame with 1-hour data
            
        Returns:
            DataFrame with 4-hour data
        """
        if df_1h is None or df_1h.empty:
            return None
            
        df_4h = df_1h.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return df_4h


def load_tickers(filename=None):
    """
    Load ticker symbols from file
    
    Args:
        filename: Path to file containing tickers (one per line)
        
    Returns:
        List of ticker symbols
    """
    filename = filename or config.TICKERS_FILE
    
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {filename}")
        return tickers
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading tickers: {e}")
        return []


if __name__ == "__main__":
    # Test the data loader
    tickers = load_tickers()
    if tickers:
        loader = DataLoader(tickers[0])
        data = loader.download_all_timeframes()
        
        for tf, df in data.items():
            if df is not None:
                print(f"\n{tf} timeframe:")
                print(df.head())
                print(f"Shape: {df.shape}")
