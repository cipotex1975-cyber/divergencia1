"""
Quick test script to verify the strategy components
"""

import sys
from data_loader import DataLoader, load_tickers
from indicators import SupportResistance, MACDDivergence


def test_data_loading():
    """Test data loading functionality"""
    print("="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    tickers = load_tickers()
    if not tickers:
        print("❌ Failed to load tickers")
        return False
    
    print(f"✓ Loaded {len(tickers)} tickers")
    
    # Test with first ticker
    ticker = tickers[0]
    print(f"\nTesting data download for {ticker}...")
    
    loader = DataLoader(ticker, start_date='2024-01-01', end_date='2024-12-31')
    data = loader.download_all_timeframes()
    
    for tf, df in data.items():
        if df is not None and not df.empty:
            print(f"✓ {tf}: {len(df)} bars")
        else:
            print(f"❌ {tf}: No data")
            return False
    
    return True


def test_indicators():
    """Test indicator calculations"""
    print("\n" + "="*60)
    print("TEST 2: Technical Indicators")
    print("="*60)
    
    tickers = load_tickers()
    ticker = tickers[0]
    
    loader = DataLoader(ticker, start_date='2024-01-01', end_date='2024-12-31')
    data = loader.download_all_timeframes()
    
    # Test Support/Resistance
    print("\nTesting Support/Resistance detection...")
    sr = SupportResistance()
    levels = sr.get_levels(data['4h'])
    
    print(f"✓ Found {len(levels['support'])} support levels")
    print(f"✓ Found {len(levels['resistance'])} resistance levels")
    
    if levels['support']:
        print(f"  Support levels: {[f'{x:.2f}' for x in levels['support'][:3]]}")
    if levels['resistance']:
        print(f"  Resistance levels: {[f'{x:.2f}' for x in levels['resistance'][:3]]}")
    
    # Test MACD
    print("\nTesting MACD calculation...")
    macd_detector = MACDDivergence()
    macd_data = macd_detector.calculate_macd(data['1h'])
    
    if macd_data is not None and not macd_data.empty:
        print(f"✓ MACD calculated for {len(macd_data)} bars")
        print(f"  Latest MACD: {macd_data['MACD'].iloc[-1]:.4f}")
        print(f"  Latest Signal: {macd_data['Signal'].iloc[-1]:.4f}")
    else:
        print("❌ MACD calculation failed")
        return False
    
    # Test divergence detection
    print("\nTesting divergence detection...")
    bullish_div = macd_detector.detect_bullish_divergence(data['1h'], macd_data)
    bearish_div = macd_detector.detect_bearish_divergence(data['1h'], macd_data)
    
    print(f"✓ Bullish divergences detected: {bullish_div.sum()}")
    print(f"✓ Bearish divergences detected: {bearish_div.sum()}")
    
    return True


def test_configuration():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 3: Configuration")
    print("="*60)
    
    import config
    
    print(f"✓ Initial Cash: ${config.INITIAL_CASH:,.2f}")
    print(f"✓ Commission: {config.COMMISSION*100}%")
    print(f"✓ Position Size: {config.POSITION_SIZE_PERCENT}%")
    print(f"✓ Stop Loss: {config.STOP_LOSS_PERCENT}%")
    print(f"✓ Take Profit: {config.TAKE_PROFIT_PERCENT}%")
    print(f"✓ MACD Settings: {config.MACD_FAST}/{config.MACD_SLOW}/{config.MACD_SIGNAL}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("STRATEGY COMPONENT TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Loading", test_data_loading),
        ("Indicators", test_indicators),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run backtest.")
        print("\nTo run the full backtest, execute:")
        print("  python main.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
