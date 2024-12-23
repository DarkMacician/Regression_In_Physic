import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import numpy as np
# Calculate ATR


# Load your dataset
df = pd.read_csv('D:/Python Project/RecommendationSystem/tradingbot/BTCUSDT_2024-12-29.csv')

# Ensure the dataset is sorted by Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

# Calculate RSI (default period is 14)
rsi_indicator = RSIIndicator(close=df['Close Price'], window=14)
df['RSI'] = rsi_indicator.rsi()



# Calculate Bollinger Bands
bollinger = BollingerBands(close=df['Close Price'], window=20, window_dev=2)
df['Bollinger_Upper'] = bollinger.bollinger_hband()
df['Bollinger_Lower'] = bollinger.bollinger_lband()



# Calculate MACD
macd = MACD(close=df['Close Price'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Diff'] = macd.macd_diff()

atr = AverageTrueRange(high=df['High Price'], low=df['Low Price'], close=df['Close Price'], window=14)
df['ATR'] = atr.average_true_range()


# Placeholder sentiment score
df['Sentiment_Score'] = np.random.uniform(-1, 1, len(df))

df['Volatility'] = (df['High Price'] - df['Low Price']) / df['Low Price']
df['VWAP'] = (df['Volume'] * (df['High Price'] + df['Low Price'] + df['Close Price']) / 3).cumsum() / df['Volume'].cumsum()
df.to_csv('updated_data.csv', index=False)

