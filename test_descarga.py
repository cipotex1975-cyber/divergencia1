import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime, time

# 🔹 Paso 1: Descargar datos (already done in previous cells)
symbol = 'USDJPY=X'
start_date = "2025-05-27"
end_date = "2025-07-21"

# Datos de 15 minutos (already loaded)
# data_15m = yf.download(symbol, start=start_date, end=end_date, interval="15m", progress=False, multi_level_index=False)
# data_15m.index = pd.to_datetime(data_15m.index)

# Datos diarios
#data_daily = yf.download(symbol, start=start_date, end=end_date, interval="1d")

#data_daily = yf.download("USDJPY=X", period="2mo", interval="1d")
#print(data_daily.head())

data_daily = yf.download(
    "USDJPY=X",
    start="2025-05-27",
    end="2025-07-21",
    interval="1d",
    auto_adjust=True
)

