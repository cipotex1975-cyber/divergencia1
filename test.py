import yfinance as yf
data = yf.Ticker("USDJPY=X").history(period="3mo")
print(data.head())

print(yf.download("AAPL", period="1mo"))
