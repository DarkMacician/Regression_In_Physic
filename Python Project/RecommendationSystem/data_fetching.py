import requests
import csv
from datetime import datetime, timedelta

def fetch_all_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    response.raise_for_status()  # Raise error if the request fails
    data = response.json()
    symbols = [symbol['symbol'] for symbol in data['symbols'] if symbol['status'] == 'TRADING']
    return symbols

def fetch_historical_1m_data(symbol, start_date="2024-12-01", end_date="2024-12-26"):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if the request fails
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1  # Move to the next interval

    return all_data

def export_to_csv(data, symbol):
    headers = ["Timestamp", "Open Price", "High Price", "Low Price", "Close Price", "Volume"]
    # Save each symbol to its own CSV file
    filename = f"data/{symbol}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')  # Convert ms to readable time
            open_price = entry[1]
            high_price = entry[2]
            low_price = entry[3]
            close_price = entry[4]
            volume = entry[5]  # Extract volume
            writer.writerow([timestamp, open_price, high_price, low_price, close_price, volume])
    print(f"Data for {symbol} successfully exported to {filename}.")

# Main process to fetch and export data for all symbols
def fetch_and_export_all(symbols, start_date="2024-12-01", end_date="2024-12-26"):
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        data = fetch_historical_1m_data(symbol, start_date, end_date)
        export_to_csv(data, symbol)

# Fetch all available symbols
symbols = fetch_all_symbols()

# Fetch and export data for all symbols (may take time depending on the number of symbols)
fetch_and_export_all(symbols)
