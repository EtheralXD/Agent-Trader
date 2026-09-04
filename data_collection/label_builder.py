import mysql.connector
import os
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HORIZON = 3

conn = mysql.connector.connect(
    host=os.getenv('DATABASE_HOST'),
    user=os.getenv('DATABASE_USER'),
    password=os.getenv('DATABASE_PASS'),
    database=os.getenv('DATABASE_NAME')
)

cursor = conn.cursor(dictionary=True)

# Load groups
cursor.execute("""
    SELECT 
        pg.id AS group_id,
        pg.symbol_id,
        pg.start_time,

        cd.candle_index,
        cd.open_price,
        cd.high_price,
        cd.low_price,
        cd.close_price

    FROM pattern_groups pg
    JOIN candle_differences cd 
        ON cd.total_dif_id = (
            SELECT id FROM total_group_differences 
            WHERE group_id = pg.id
        )

    ORDER BY pg.symbol_id, pg.start_time ASC, cd.candle_index ASC;
""")

rows = cursor.fetchall()

# Group into memory
groups = {}

for r in rows:
    gid = r['group_id']

    if gid not in groups:
        groups[gid] = {
            "symbol_id": r["symbol_id"],
            "candles": []
        }

    groups[gid]["candles"].append(r)

group_list = list(groups.items())

print("Groups loaded:", len(group_list))
pbar = tqdm(total=len(group_list), desc='Building labels' )

# ATR function
def calc_atr(candles):
    trs = []

    for i in range(len(candles)):
        high = float(candles[i]["high_price"])
        low = float(candles[i]["low_price"])
        close = float(candles[i]["close_price"])

        if i == 0:
            prev_close = close
        else:
            prev_close = float(candles[i-1]["close_price"])

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        trs.append(tr)

    return np.mean(trs)

# Label building
for i in range(len(group_list) - HORIZON):

    current_id, current = group_list[i]
    future_id, future = group_list[i + HORIZON]

    # prevent cross-symbol leakage
    if current["symbol_id"] != future["symbol_id"]:
        continue

    current_candles = current["candles"]
    future_candles = future["candles"]

    current_close = float(current_candles[-1]["close_price"])
    future_high = max(float(c["high_price"]) for c in future_candles)
    future_low = min(float(c["low_price"]) for c in future_candles)
    future_close = float(future_candles[-1]["close_price"])


    excursion_up = (future_high - current_close) / current_close
    excursion_down = (current_close - future_low) / current_close

    excursion = max(excursion_up, excursion_down)

    # direction
    direction = 1 if future_close > current_close else 0

    raw_return = (future_close - current_close) / current_close

    vol = np.std([float(c["close_price"]) for c in current_candles]) + 1e-8
    vol_adj_return = raw_return / vol

    atr = calc_atr(current_candles) + 1e-8
    atr_move = (future_close - current_close) / atr

    cursor.execute("""
        SELECT id FROM group_labels
        WHERE group_id = %s
    """, (current_id,))

    pbar.update(1)

    if cursor.fetchone():
        continue

    cursor.execute("""
        INSERT INTO group_labels
        (group_id, future_group_id, future_return, direction)
        VALUES (%s, %s, %s, %s)
    """, (
        current_id,
        future_id,
        atr_move, 
        direction
    ))

conn.commit()
pbar.close()
print("DONE: advanced labels created")