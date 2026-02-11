"""
Main Backtrader strategy implementing multi-timeframe divergence trading
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import config
from chart_generator import SignalChartGenerator


class MultiTimeframeStrategy(bt.Strategy):
    """
    Multi-timeframe trading strategy:
    - 4H: Identify support/resistance levels
    - 1H: Detect MACD divergences near S/R levels
    - 15M: Execute entries on breakout confirmation
    """
    
    params = (
        ('sr_lookback', config.SR_LOOKBACK_PERIOD),
        ('sr_proximity', config.SR_PROXIMITY_PERCENT),
        ('macd_fast', config.MACD_FAST),
        ('macd_slow', config.MACD_SLOW),
        ('macd_signal', config.MACD_SIGNAL),
        ('stop_loss_pct', config.STOP_LOSS_PERCENT),
        ('take_profit_pct', config.TAKE_PROFIT_PERCENT),
        ('position_size_pct', config.POSITION_SIZE_PERCENT),
        ('printlog', True),
    )
    
    def __init__(self):
        """Initialize strategy and indicators"""
        
        # Keep reference to data feeds
        # data0 = 15M, data1 = 1H, data2 = 4H
        self.data_15m = self.datas[0]
        self.data_1h = self.datas[1] if len(self.datas) > 1 else None
        self.data_4h = self.datas[2] if len(self.datas) > 2 else None
        
        # MACD indicator on 1H data
        if self.data_1h is not None:
            self.macd = bt.indicators.MACD(
                self.data_1h.close,
                period_me1=self.params.macd_fast,
                period_me2=self.params.macd_slow,
                period_signal=self.params.macd_signal
            )
        
        # Track signals and levels
        self.support_levels = []
        self.resistance_levels = []
        self.entry_level = None
        self.signal_type = None  # 'bullish' or 'bearish'
        self.divergence_detected = False
        
        # Order tracking
        self.order = None
        self.buy_price = None
        self.stop_loss = None
        self.take_profit = None
        
        # Signal log
        self.signals = []
        
        # Track processed divergences to prevent duplicate chart generation
        # Stores tuples of (signal_type, pivot_timestamp) for already processed divergences
        self.processed_divergences = set()
        
        # Chart generator (get ticker from data feed name if available)
        ticker = getattr(self.data_15m, '_name', 'UNKNOWN')
        self.chart_generator = SignalChartGenerator(ticker) if config.SAVE_SIGNAL_CHARTS else None
        
    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.data_15m.datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """Notification of order status"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                
                # Set stop loss and take profit
                self.stop_loss = self.buy_price * (1 - self.params.stop_loss_pct / 100)
                self.take_profit = self.buy_price * (1 + self.params.take_profit_pct / 100)
                
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
            
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification of trade status"""
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def prenext(self):
        """Called before minimum period is met"""
        pass
    
    def next(self):
        """Main strategy logic called on each bar"""
        
        # Don't trade if we have a pending order
        if self.order:
            return
        
        # Check if we have a position
        if self.position:
            self._manage_position()
            return
        
        # Look for new trading opportunities
        self._analyze_market()
    
    def _manage_position(self):
        """Manage open positions with stop loss and take profit"""
        current_price = self.data_15m.close[0]
        
        # Check stop loss
        if current_price <= self.stop_loss:
            self.log(f'STOP LOSS HIT at {current_price:.2f}')
            self.order = self.sell()
            self._reset_signals()
            return
        
        # Check take profit
        if current_price >= self.take_profit:
            self.log(f'TAKE PROFIT HIT at {current_price:.2f}')
            self.order = self.sell()
            self._reset_signals()
            return
    
    def _analyze_market(self):
        """
        Multi-timeframe market analysis:
        1. Check 4H for S/R levels
        2. Check if price is near S/R
        3. Check 1H for MACD divergence
        4. Identify entry level
        5. Execute on 15M breakout
        """
        
        # Step 1 & 2: Check 4H support/resistance
        if self.data_4h is not None and len(self.data_4h) > self.params.sr_lookback:
            self._update_sr_levels()
            near_support, near_resistance = self._check_proximity_to_sr()
        else:
            return
        
        # Step 3: Check for MACD divergence on 1H
        if self.data_1h is not None and len(self.data_1h) > 50:
            if near_support:
                bullish_div, pivot_points = self._detect_bullish_divergence()
                if bullish_div and not self.divergence_detected:
                    # Create unique identifier for this divergence using pivot timestamps
                    if len(pivot_points) >= 2:
                        # Convert relative indices to absolute timestamps
                        prev_idx_rel = pivot_points[1][0]  # Older point
                        curr_idx_rel = pivot_points[0][0]  # Newer point
                        prev_timestamp = self.data_1h.datetime.datetime(-prev_idx_rel)
                        curr_timestamp = self.data_1h.datetime.datetime(-curr_idx_rel)
                        divergence_id = ('bullish', prev_timestamp, curr_timestamp)
                        
                        # Only process if we haven't seen this exact divergence before
                        if divergence_id not in self.processed_divergences:
                            self.log('BULLISH DIVERGENCE DETECTED near SUPPORT')
                            self.signal_type = 'bullish'
                            self.divergence_detected = True
                            self._identify_entry_level_bullish()
                            
                            # Generate chart only once for this specific divergence
                            if self.chart_generator:
                                self._generate_signal_chart('bullish', pivot_points)
                            
                            # Mark this divergence as processed
                            self.processed_divergences.add(divergence_id)
            
            if near_resistance:
                bearish_div, pivot_points = self._detect_bearish_divergence()
                if bearish_div and not self.divergence_detected:
                    # Create unique identifier for this divergence using pivot timestamps
                    if len(pivot_points) >= 2:
                        # Convert relative indices to absolute timestamps
                        prev_idx_rel = pivot_points[1][0]  # Older point
                        curr_idx_rel = pivot_points[0][0]  # Newer point
                        prev_timestamp = self.data_1h.datetime.datetime(-prev_idx_rel)
                        curr_timestamp = self.data_1h.datetime.datetime(-curr_idx_rel)
                        divergence_id = ('bearish', prev_timestamp, curr_timestamp)
                        
                        # Only process if we haven't seen this exact divergence before
                        if divergence_id not in self.processed_divergences:
                            self.log('BEARISH DIVERGENCE DETECTED near RESISTANCE')
                            self.signal_type = 'bearish'
                            self.divergence_detected = True
                            self._identify_entry_level_bearish()
                            
                            # Generate chart only once for this specific divergence
                            if self.chart_generator:
                                self._generate_signal_chart('bearish', pivot_points)
                            
                            # Mark this divergence as processed
                            self.processed_divergences.add(divergence_id)
        
        # Step 5: Execute entry on 15M if we have a signal
        if self.divergence_detected and self.entry_level:
            self._check_entry_breakout()
    
    def _update_sr_levels(self):
        """Update support and resistance levels from 4H data"""
        lookback = self.params.sr_lookback
        
        if len(self.data_4h) < lookback * 2:
            return
        
        # Get recent highs and lows
        highs = [self.data_4h.high[-i] for i in range(lookback)]
        lows = [self.data_4h.low[-i] for i in range(lookback)]
        
        # Find swing highs (resistance)
        self.resistance_levels = []
        for i in range(2, len(highs) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                self.resistance_levels.append(highs[i])
        
        # Find swing lows (support)
        self.support_levels = []
        for i in range(2, len(lows) - 2):
            if lows[i] == min(lows[i-2:i+3]):
                self.support_levels.append(lows[i])
        
        # Keep only unique levels (cluster nearby ones)
        self.resistance_levels = self._cluster_levels(self.resistance_levels)
        self.support_levels = self._cluster_levels(self.support_levels)
    
    def _cluster_levels(self, levels):
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] * 100 < 1.0:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    def _check_proximity_to_sr(self):
        """Check if current price is near support or resistance"""
        current_price = self.data_4h.close[0]
        threshold = self.params.sr_proximity
        
        near_support = False
        near_resistance = False
        
        for level in self.support_levels:
            if abs(current_price - level) / level * 100 <= threshold:
                near_support = True
                break
        
        for level in self.resistance_levels:
            if abs(current_price - level) / level * 100 <= threshold:
                near_resistance = True
                break
        
        return near_support, near_resistance
    
    def _detect_bullish_divergence(self):
        """Detect bullish MACD divergence on 1H"""
        if len(self.data_1h) < 50:
            return False
        
        # Get recent price lows and MACD lows
        lookback = 20
        price_lows = []
        macd_lows = []
        
        for i in range(5, lookback):
            # Check for pivot low
            if (self.data_1h.low[-i] < self.data_1h.low[-i-1] and 
                self.data_1h.low[-i] < self.data_1h.low[-i+1]):
                price_lows.append((i, self.data_1h.low[-i]))
                macd_lows.append((i, self.macd.macd[-i]))
        
        # Need at least 2 lows to compare
        if len(price_lows) < 2:
            return False, []
        
        # Check if price making lower low but MACD making higher low
        if (price_lows[0][1] < price_lows[1][1] and 
            macd_lows[0][1] > macd_lows[1][1]):
            return True, price_lows
        
        return False, []
    
    def _detect_bearish_divergence(self):
        """Detect bearish MACD divergence on 1H"""
        if len(self.data_1h) < 50:
            return False
        
        # Get recent price highs and MACD highs
        lookback = 20
        price_highs = []
        macd_highs = []
        
        for i in range(5, lookback):
            # Check for pivot high
            if (self.data_1h.high[-i] > self.data_1h.high[-i-1] and 
                self.data_1h.high[-i] > self.data_1h.high[-i+1]):
                price_highs.append((i, self.data_1h.high[-i]))
                macd_highs.append((i, self.macd.macd[-i]))
        
        # Need at least 2 highs to compare
        if len(price_highs) < 2:
            return False, []
        
        # Check if price making higher high but MACD making lower high
        if (price_highs[0][1] > price_highs[1][1] and 
            macd_highs[0][1] < macd_highs[1][1]):
            return True, price_highs
        
        return False, []
    
    def _identify_entry_level_bullish(self):
        """Identify entry level for bullish setup (break above last high)"""
        if len(self.data_1h) < 20:
            return
        
        # Find the last significant high
        recent_highs = [self.data_1h.high[-i] for i in range(10)]
        self.entry_level = max(recent_highs)
        
        self.log(f'BULLISH ENTRY LEVEL identified at {self.entry_level:.2f}')
    
    def _identify_entry_level_bearish(self):
        """Identify entry level for bearish setup (break below last low)"""
        if len(self.data_1h) < 20:
            return
        
        # Find the last significant low
        recent_lows = [self.data_1h.low[-i] for i in range(10)]
        self.entry_level = min(recent_lows)
        
        self.log(f'BEARISH ENTRY LEVEL identified at {self.entry_level:.2f}')
    
    def _check_entry_breakout(self):
        """Check for breakout on 15M timeframe and execute entry"""
        current_price = self.data_15m.close[0]
        
        if self.signal_type == 'bullish':
            # Check for breakout above entry level
            if current_price > self.entry_level:
                self.log(f'BULLISH BREAKOUT at {current_price:.2f}')
                
                # Calculate position size
                size = self._calculate_position_size()
                
                # Execute buy order
                self.order = self.buy(size=size)
                
                # Log signal
                self.signals.append({
                    'date': self.data_15m.datetime.date(0),
                    'type': 'BUY',
                    'price': current_price,
                    'entry_level': self.entry_level
                })
                
                # Reset divergence flag (keep entry level for position management)
                self.divergence_detected = False
        
        elif self.signal_type == 'bearish':
            # For bearish signals, we would short (not implemented in this basic version)
            # In a complete implementation, you would add short selling logic here
            self.log(f'BEARISH SIGNAL - Short selling not implemented')
            self._reset_signals()
    
    def _calculate_position_size(self):
        """Calculate position size based on portfolio percentage"""
        cash = self.broker.getcash()
        price = self.data_15m.close[0]
        
        # Calculate size based on position_size_pct
        value = cash * (self.params.position_size_pct / 100)
        size = int(value / price)
        
        return size
    
    def _reset_signals(self):
        """Reset signal tracking variables"""
        self.entry_level = None
        self.signal_type = None
        self.divergence_detected = False
    
    def _generate_signal_chart(self, signal_type, pivot_points):
        """
        Generate and save a chart for the detected divergence signal
        
        Args:
            signal_type: 'bullish' or 'bearish'
            pivot_points: List of (index, price) tuples for pivot points
        """
        try:
            # Convert Backtrader data to pandas DataFrames
            lookback = config.CHART_LOOKBACK_BARS
            
            # Get 1H price data
            price_data = []
            macd_data = []
            
            for i in range(min(lookback, len(self.data_1h))):
                idx = -i
                date = self.data_1h.datetime.datetime(idx)
                
                price_data.append({
                    'Date': date,
                    'Open': self.data_1h.open[idx],
                    'High': self.data_1h.high[idx],
                    'Low': self.data_1h.low[idx],
                    'Close': self.data_1h.close[idx],
                    'Volume': self.data_1h.volume[idx]
                })
                
                macd_data.append({
                    'Date': date,
                    'MACD': self.macd.macd[idx],
                    'Signal': self.macd.signal[idx],
                    'Histogram': self.macd.macd[idx] - self.macd.signal[idx]
                })
            
            # Reverse to get chronological order
            price_data.reverse()
            macd_data.reverse()
            
            # Convert to DataFrames
            price_df = pd.DataFrame(price_data).set_index('Date')
            macd_df = pd.DataFrame(macd_data).set_index('Date')
            
            # Get divergence indices (convert from relative to absolute dates)
            if len(pivot_points) >= 2:
                # pivot_points are (relative_index, price) tuples
                prev_idx_rel, _ = pivot_points[1]  # Older point
                curr_idx_rel, _ = pivot_points[0]  # Newer point
                
                # Convert to datetime
                prev_date = self.data_1h.datetime.datetime(-prev_idx_rel)
                curr_date = self.data_1h.datetime.datetime(-curr_idx_rel)
                
                divergence_indices = (prev_date, curr_date)
            else:
                divergence_indices = None
            
            # Prepare S/R levels
            sr_levels = {
                'support': self.support_levels,
                'resistance': self.resistance_levels
            }
            
            # Generate chart
            self.chart_generator.generate_divergence_chart(
                price_data=price_df,
                macd_data=macd_df,
                signal_type=signal_type,
                divergence_indices=divergence_indices,
                sr_levels=sr_levels,
                entry_level=self.entry_level
            )
            
        except Exception as e:
            self.log(f'Error generating chart: {e}')
    
    def stop(self):
        """Called at the end of the backtest"""
        self.log(f'Final Portfolio Value: {self.broker.getvalue():.2f}', 
                dt=self.data_15m.datetime.date(0))
        
        # Save signals if configured
        if config.SAVE_SIGNALS and self.signals:
            self._save_signals()
    
    def _save_signals(self):
        """Save detected signals to file"""
        import os
        
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        signals_df = pd.DataFrame(self.signals)
        filename = f"{config.RESULTS_DIR}/signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        signals_df.to_csv(filename, index=False)
        
        print(f"\nSignals saved to {filename}")
