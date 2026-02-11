"""
Technical indicators and analysis functions
Includes support/resistance detection and MACD divergence detection
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import config


class SupportResistance:
    """Detects support and resistance levels using swing highs/lows"""
    
    def __init__(self, lookback=None):
        """
        Initialize S/R detector
        
        Args:
            lookback: Number of bars to look back for swing points
        """
        self.lookback = lookback or config.SR_LOOKBACK_PERIOD
    
    def find_swing_highs(self, data):
        """
        Find swing high points in the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with swing high prices (NaN where no swing high)
        """
        highs = data['High'].values
        
        # Find local maxima
        swing_highs = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.lookback, len(data) - self.lookback):
            if highs[i] == max(highs[i - self.lookback:i + self.lookback + 1]):
                swing_highs.iloc[i] = highs[i]
        
        return swing_highs
    
    def find_swing_lows(self, data):
        """
        Find swing low points in the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with swing low prices (NaN where no swing low)
        """
        lows = data['Low'].values
        
        # Find local minima
        swing_lows = pd.Series(index=data.index, dtype=float)
        
        for i in range(self.lookback, len(data) - self.lookback):
            if lows[i] == min(lows[i - self.lookback:i + self.lookback + 1]):
                swing_lows.iloc[i] = lows[i]
        
        return swing_lows
    
    def get_levels(self, data):
        """
        Get all support and resistance levels
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with 'resistance' and 'support' lists
        """
        swing_highs = self.find_swing_highs(data)
        swing_lows = self.find_swing_lows(data)
        
        # Get unique levels (cluster nearby levels)
        resistance_levels = self._cluster_levels(swing_highs.dropna().values)
        support_levels = self._cluster_levels(swing_lows.dropna().values)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _cluster_levels(self, levels, threshold_percent=1.0):
        """
        Cluster nearby levels together
        
        Args:
            levels: Array of price levels
            threshold_percent: Percentage threshold for clustering
            
        Returns:
            List of clustered levels
        """
        if len(levels) == 0:
            return []
        
        levels = np.sort(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] * 100 < threshold_percent:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    def is_near_level(self, price, level, threshold_percent=None):
        """
        Check if price is near a support/resistance level
        
        Args:
            price: Current price
            level: S/R level to check
            threshold_percent: Percentage threshold
            
        Returns:
            Boolean indicating if price is near level
        """
        threshold = threshold_percent or config.SR_PROXIMITY_PERCENT
        distance_percent = abs(price - level) / level * 100
        return distance_percent <= threshold


class MACDDivergence:
    """Detects MACD divergences"""
    
    def __init__(self, fast=None, slow=None, signal=None):
        """
        Initialize MACD divergence detector
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        self.fast = fast or config.MACD_FAST
        self.slow = slow or config.MACD_SLOW
        self.signal = signal or config.MACD_SIGNAL
    
    def calculate_macd(self, data):
        """
        Calculate MACD indicator
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with MACD, Signal, and Histogram columns
        """
        close = data['Close']
        
        # Calculate EMAs
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        
        # Histogram
        histogram = macd - signal
        
        result = pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': histogram
        }, index=data.index)
        
        return result
    
    def find_price_pivots(self, data, lookback=None):
        """
        Find price pivot points (highs and lows)
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Lookback period for pivot detection
            
        Returns:
            Dictionary with 'highs' and 'lows' DataFrames
        """
        lookback = lookback or config.MIN_PIVOT_DISTANCE
        
        highs = pd.Series(index=data.index, dtype=float)
        lows = pd.Series(index=data.index, dtype=float)
        
        high_vals = data['High'].values
        low_vals = data['Low'].values
        
        for i in range(lookback, len(data) - lookback):
            # Pivot high
            if high_vals[i] == max(high_vals[i - lookback:i + lookback + 1]):
                highs.iloc[i] = high_vals[i]
            
            # Pivot low
            if low_vals[i] == min(low_vals[i - lookback:i + lookback + 1]):
                lows.iloc[i] = low_vals[i]
        
        return {'highs': highs, 'lows': lows}
    
    def detect_bullish_divergence(self, data, macd_data, lookback=None):
        """
        Detect bullish divergence (price lower lows, MACD higher lows)
        
        Args:
            data: DataFrame with OHLCV data
            macd_data: DataFrame with MACD values
            lookback: Lookback period
            
        Returns:
            Series with True where bullish divergence detected
        """
        lookback = lookback or config.DIVERGENCE_LOOKBACK
        
        pivots = self.find_price_pivots(data)
        price_lows = pivots['lows'].dropna()
        
        divergences = pd.Series(False, index=data.index)
        
        if len(price_lows) < 2:
            return divergences
        
        # Check last few pivot lows
        price_low_indices = price_lows.index[-min(5, len(price_lows)):]
        
        for i in range(1, len(price_low_indices)):
            idx_prev = price_low_indices[i - 1]
            idx_curr = price_low_indices[i]
            
            # Price making lower low
            if data.loc[idx_curr, 'Low'] < data.loc[idx_prev, 'Low']:
                # MACD making higher low
                if macd_data.loc[idx_curr, 'MACD'] > macd_data.loc[idx_prev, 'MACD']:
                    divergences.loc[idx_curr] = True
        
        return divergences
    
    def detect_bearish_divergence(self, data, macd_data, lookback=None):
        """
        Detect bearish divergence (price higher highs, MACD lower highs)
        
        Args:
            data: DataFrame with OHLCV data
            macd_data: DataFrame with MACD values
            lookback: Lookback period
            
        Returns:
            Series with True where bearish divergence detected
        """
        lookback = lookback or config.DIVERGENCE_LOOKBACK
        
        pivots = self.find_price_pivots(data)
        price_highs = pivots['highs'].dropna()
        
        divergences = pd.Series(False, index=data.index)
        
        if len(price_highs) < 2:
            return divergences
        
        # Check last few pivot highs
        price_high_indices = price_highs.index[-min(5, len(price_highs)):]
        
        for i in range(1, len(price_high_indices)):
            idx_prev = price_high_indices[i - 1]
            idx_curr = price_high_indices[i]
            
            # Price making higher high
            if data.loc[idx_curr, 'High'] > data.loc[idx_prev, 'High']:
                # MACD making lower high
                if macd_data.loc[idx_curr, 'MACD'] < macd_data.loc[idx_prev, 'MACD']:
                    divergences.loc[idx_curr] = True
        
        return divergences


if __name__ == "__main__":
    # Test the indicators
    from data_loader import DataLoader, load_tickers
    
    tickers = load_tickers()
    if tickers:
        loader = DataLoader(tickers[0])
        data = loader.download_all_timeframes()
        
        # Test S/R detection
        sr = SupportResistance()
        levels = sr.get_levels(data['4h'])
        print(f"\nSupport levels: {levels['support']}")
        print(f"Resistance levels: {levels['resistance']}")
        
        # Test MACD divergence
        macd_detector = MACDDivergence()
        macd_data = macd_detector.calculate_macd(data['1h'])
        print(f"\nMACD data:\n{macd_data.tail()}")
