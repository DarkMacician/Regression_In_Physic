import requests
import csv
from datetime import datetime

def fetch_historical_1m_data(symbol, start_time, end_time):
    """
    Fetch historical data for a given symbol at 1-minute intervals.
    """
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000
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
    """
    Export historical data to a CSV file.
    """
    headers = ["Timestamp", "Open Price", "High Price", "Low Price", "Close Price", "Volume"]
    filename = f"{symbol}_2024-12-29.csv"
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

# Main process to fetch BTC data
def fetch_btc_data():
    symbol = "BTCUSDT"  # Symbol for BTC/USDT trading pair
    start_time = int(datetime.strptime("2024-12-29 11:23:36", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    end_time = int(datetime.strptime("2024-12-29 12:23:36", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

    print(f"Fetching data for {symbol} from {start_time} to {end_time}...")
    data = fetch_historical_1m_data(symbol, start_time, end_time)
    export_to_csv(data, symbol)

if __name__ == "__main__":
    fetch_btc_data()
