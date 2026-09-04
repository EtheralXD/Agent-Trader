import ccxt
import time
import mysql.connector
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from datetime import datetime

load_dotenv()

# Config
DAYS_AGO = 350
TIMEFRAME = '5m'   # 5m, 15m, 1h, etc
LIMIT = 1000      # max per request (MEXC allows ~1000)


candles_per_day = (60 // 5) * 24 
total_candles_expected = DAYS_AGO * candles_per_day

# DB Connection
mysqlcon = mysql.connector.connect(
    host=os.getenv('DATABASE_HOST'),
    user=os.getenv('DATABASE_USER'),
    password=os.getenv('DATABASE_PASS'),
    database=os.getenv('DATABASE_NAME')
)
cursor = mysqlcon.cursor(dictionary=True)

# Exchange 
exchange = ccxt.mexc({
    'enableRateLimit': True
})


# Helpers
def get_since(days_ago):
    now = int(time.time() * 1000)
    return now - (days_ago * 24 * 60 * 60 * 1000)

# Get symbols
symbols = [
    'SOL/USDT',
    'XRP/USDT'
]

print(f"Loaded {len(symbols)} symbols")
def ensure_symbol(symbol):
    cursor.execute("SELECT id FROM symbols WHERE symbol=%s", (symbol,))
    row = cursor.fetchone()

    if row:
        return row['id']

    cursor.execute("INSERT INTO symbols (symbol) VALUES (%s)", (symbol,))
    mysqlcon.commit()

    return cursor.lastrowid

# Main fetch loop
for sym in symbols:
    symbol = sym
    symbol_id = ensure_symbol(symbol)

    print(f"\nFetching {symbol}...")

    start_since = get_since(DAYS_AGO)
    current_since = start_since
    now = int(time.time() * 1000)

    total_groupings = 0
    progress = 0
    fetched_candles = 0

    pbar = tqdm(total=100, desc=symbol)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current_since, limit=LIMIT)

            if not ohlcv:
                pbar.close()
                print("No more data.")
                break
            fetched_candles += len(ohlcv)
            for candle in ohlcv:
                ts, open_, high, low, close, volume = candle

                dt = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts / 1000))

                cursor.execute("""
                    INSERT INTO raw_candles 
                    (symbol_id, timestamp, open, high, low, close, volume, timeframe)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (symbol_id, dt, open_, high, low, close, volume, TIMEFRAME))

            mysqlcon.commit()

            current_since = ohlcv[-1][0] + 1
            
            progress = (current_since - start_since) / (now - start_since) * 100

            pbar.n = progress
            pbar.refresh()

            #tqdm.write(f"Inserted {len(ohlcv)} candles... next batch. Current insert {total_groupings}")

            # prevent rate limit
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)
            continue
    start = start_since / 1000
    end = current_since / 1000
    print(f'{symbol}', 'Started at:', datetime.fromtimestamp(start), 'And ended at:', datetime.fromtimestamp(end))


print("\nDONE. Total inserts:", fetched_candles)