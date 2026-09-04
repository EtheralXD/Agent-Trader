print('Agent is now booting up!')
#region Imports
import os
import mysql.connector
import numpy as np
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
#endregion

#region Variables
take_prc = 0.033
loss_prc = 0.016
balance = 1000
earned = 0
finished_set = 0
wins = 0
losses = 0
total = 0

position = None
stop_training = False
in_trade = False

#lists
results = []

#dicts
bin_winrates = {}
#endregion

#region Connections 
current_model = 'model.npz'

mysqlcon = mysql.connector.connect(
    host = os.getenv('DATABASE_HOST'),
    user = os.getenv('DATABASE_USER'),
    password = os.getenv('DATABASE_PASS'),
    database = os.getenv('DATABASE_NAME')
)
cursor = mysqlcon.cursor(dictionary=True)
#endregion

#region Layers
input_layer = 17
hidden_layer_1 = 34
hidden_layer_2 = 24
output_layer = 3
#endregion

#region Rand Gen
W1 = np.random.randn(input_layer, hidden_layer_1) * np.sqrt(2 / input_layer)
B1 = np.zeros((1, hidden_layer_1))

W2 = np.random.randn(hidden_layer_1, hidden_layer_2) * np.sqrt(2 / hidden_layer_1)
B2 = np.zeros((1, hidden_layer_2))

W3 = np.random.randn(hidden_layer_2, output_layer) * np.sqrt(2 / hidden_layer_2)
B3 = np.zeros((1, output_layer))
#endregion

#region Data Fetch
def split(X, y):
    split_idx = int(len(X) * 0.3)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

query = """
SELECT 
    s.symbol,
    pg.id AS group_id,
    pg.start_time,
    pg.group_index,

    tgd.total_value,

    cd.candle_index,
    cd.difference_value,
    cd.open_price,
    cd.high_price,
    cd.low_price,
    cd.close_price

FROM pattern_groups pg
JOIN symbols s ON s.id = pg.symbol_id
JOIN total_group_differences tgd 
    ON tgd.group_id = pg.id
JOIN candle_differences cd 
    ON cd.total_dif_id = tgd.id

ORDER BY pg.start_time ASC, cd.candle_index ASC;
"""

cursor.execute(query)
rows = cursor.fetchall()

symbol_groups = {}
for row in tqdm(rows, desc='Fetching data'):
    gid = row['group_id']
    symbol = row['symbol']  

    if symbol not in symbol_groups:
        symbol_groups[symbol] = {}

    if gid not in symbol_groups[symbol]:
        symbol_groups[symbol][gid] = {
            'start_time': row['start_time'],
            'group_index': row['group_index'],
            'total_dif': float(row['total_value']),
            'candles': []
        }
    
    candle = {
        'dif': float(row['difference_value']),
        'open': float(row['open_price']),
        'high': float(row['high_price']),
        'low': float(row['low_price']),
        'close': float(row['close_price'])
    }
    symbol_groups[symbol][gid]['candles'].append(candle)

for symbol in symbol_groups:
    symbol_groups[symbol] = list(symbol_groups[symbol].values())
#endregion

#region Build
def build_features(group):
    features = []
    for c in group['candles']:
        features.append(c['dif'])

        candle_range = c['high'] - c['close']
        upper_wick = c['high'] - max(c['open'], c['close'])
        lower_wick = min(c['close'], c['open']) - c['low']

        features.append(candle_range)
        features.append(upper_wick)
        features.append(lower_wick)
    
    features.append(group['total_dif'])
    return features
#endregion

#region Build Dataset
X = []
y = []
all_groups = []
for groups in symbol_groups.values():
    all_groups.extend(groups)

for i in tqdm(range(len(all_groups) - 1), desc='Building dataset'):
    current = all_groups[i]

    if len(current['candles']) != 4:
        continue
    
    entry = current['candles'][-1]['close']

    long_tp = entry * (1 + take_prc)
    long_sl = entry * (1 - loss_prc)

    short_tp = entry * (1 - take_prc)
    short_sl = entry * (1 + loss_prc)

    long_hit_tp = long_hit_sl = False
    short_hit_tp = short_hit_sl = False

    label = None

    max_lookahead = 3

    for j in range(i + 1, min(i + 1 + max_lookahead, len(all_groups))):
        future_group = all_groups[j]

        for c in future_group['candles']:
            if c['high'] >= long_tp:
                label = [1, 0, 0]
                break

            if c['low'] >= short_tp:
                label = [0, 1, 0]
                break
            
            if c['low'] <= long_sl:
                label = [0, 0, 1]
                break
            
            if c['high'] >= short_sl:
                label = [0, 0, 1]

        if label is not None:
            break

    if label is None:
        continue
    X.append(build_features(current))
    y.append(label)

X = np.array(X)
y = np.array(y)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-8
X = (X - mean) / std

print('Dataset size:', len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Sample X:", X[0])
print("Sample y:", y[0])

print("Long win rate:", np.mean(y[:,0]))
print("Short win rate:", np.mean(y[:,1]))
#endregion

# region Computing
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(X):
    Z1 = np.dot(X, W1) + B1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + B2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + B3
    A3 = softmax(Z3)

    cache = (X, Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def compute_loss(A3, y):
    return -np.mean(np.sum(y * np.log(A3 + 1e-8), axis=1))

def backward(cache, y):
    global W1, B1, W2, B2, W3, B3
    X, Z1, A1, Z2, A2, Z3, A3 = cache
    m  = X.shape[0]

    dZ3 = (A3 - y) / m
    dW3 = np.dot(A2.T, dZ3) / m
    dB3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, W3.T) / m
    dZ2 = dA2 * relu_derivative(Z2) / m
    dW2 = np.dot(A1.T, dZ2) / m
    dB2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T) / m
    dZ1 = dA1 * relu_derivative(Z1) / m
    dW1 = np.dot(X.T, dZ1) / m
    dB1 = np.sum(dZ1, axis=0, keepdims=True) / m

    lr = 0.01
    W1 -= lr * dW1
    B1 -= lr * dB1
    W2 -= lr * dW2
    B2 -= lr * dB2
    W3 -= lr * dW3
    B3 -= lr * dB3

def predict(X):
    A3, _ = forward(X)
    return np.argmax(A3, axis=1)

def accuracy(X, y):
    A3, _ = forward(X)
    preds = np.argmax(A3, axis=1)
    true = np.argmax(y, axis=1)
    return np.mean(preds == true)
#endregion

# region Trading
def take_long(group, prediction):
    global position, total
    entry_price = group['candles'][-1]['close']

    position = {
        'side': 'long',
        'entry': entry_price,
        'take': entry_price * (1 + take_prc),
        'stop': entry_price * (1 - loss_prc)
    }

    total += 1

    print(f"""
    LONG TRADE
    Time: {group['start_time']}
    Entry: {entry_price}
    Balance: {balance:.2f}
    Prediction: {prediction}
    """)

def take_short(group, prediction):
    global position, total
    entry_price = group['candles'][-1]['close']

    position = {
        'side': 'short',
        'entry': entry_price,
        'take': entry_price * (1 - take_prc),
        'stop': entry_price * (1 + loss_prc)
    }
    total += 1


    print(f"""
    SHORT TRADE
    Time: {group['start_time']}
    Entry: {entry_price}
    Balance: {balance:.2f}
    Prediction: {prediction}
    """)
    
def close_position(group):
    global position, balance, wins, losses

    if position is None:
        return

    direction = position['side']
    take = position['take']
    stop = position['stop']

    for c in group['candles']:
        if direction == 'short':
            if c['low'] <= take:
                balance = balance + (balance * 0.30)
                wins += 1
                print('SHORT TP HIT', group['start_time'])
                position = None
                return

            elif c['high'] >= stop:
                balance = balance - (balance * 0.30)
                losses += 1
                print('SHORT SL HIT', group['start_time'])
                position = None
                return

        if direction == 'long':
            if c['high'] >= take:
                balance = balance + (balance * 0.30)
                wins += 1
                print('LONG TP HIT', group['start_time'])
                position = None
                return

            elif c['low'] <= stop:
                balance = balance - (balance * 0.30)
                losses += 1
                print('LONG SL HIT', group['start_time'])
                position = None
                return
#endregion

# region Testing Trading
def training_buffer_update(X, y):
    X = np.array([X])
    y = np.array([y])

    A3, cache = forward(X)
    loss = compute_loss(A3, y)
    backward(cache, y)

def train_take_long(group, prediction):
    global position, position_sample, total

    if not in_trade:
        entry_price = group['candles'][-1]['close']
        position_sample = build_features(group)

        position = {
            'side': 'long',
            'entry': entry_price,
            'take': entry_price * (1 + take_prc),
            'stop': entry_price * (1 - loss_prc)
        }

        total += 1

        tqdm.write(f"""
        LONG TRADE
        Time: {group['start_time']}
        Entry: {entry_price}
        Balance: {balance:.2f}
        Prediction: {prediction}
        """)
    else:
        print('still in trade')

def train_take_short(group, prediction):
    global position, position_sample, total

    if not in_trade:
        entry_price = group['candles'][-1]['close']

        position_sample = build_features(group) 

        position = {
            'side': 'short',
            'entry': entry_price,
            'take': entry_price * (1 - take_prc),
            'stop': entry_price * (1 + loss_prc)
        }
        total += 1

        tqdm.write(f"""
        SHORT TRADE
        Time: {group['start_time']}
        Entry: {entry_price}
        Balance: {balance:.2f}
        Prediction: {prediction}
        """)
    else:
        print('In trade')
    
def train_close_position(group):
    global position, position_sample, balance, wins, losses, in_trade

    if position is None:
        return

    direction = position['side']
    take = position['take']
    stop = position['stop']

    for c in group['candles']:
        start_time = group['start_time']
        if direction == 'short':
            if c['low'] <= take:
                balance = balance + (balance * 0.30)
                wins += 1

                X = np.array(position_sample).reshape(1, -1)
                y = np.array([[0, 1, 0]]) 

                A3, cache = forward(X)
                backward(cache, y)

                tqdm.write(f'SHORT TP HIT {start_time}')
                position = None
                in_trade = False
                return

            elif c['high'] >= stop:
                balance = balance - (balance * 0.30)
                losses += 1

                X = np.array(position_sample).reshape(1, -1)
                y = np.array([[0, 0, 1]])  

                A3, cache = forward(X)
                backward(cache, y)

                tqdm.write(f'SHORT SL HIT {start_time}')
                position = None
                in_trade = False
                return

        if direction == 'long':
            if c['high'] >= take:
                balance = balance + (balance * 0.30)
                wins += 1

                X = np.array(position_sample).reshape(1, -1)
                y = np.array([[1, 0, 0]])

                A3, cache = forward(X)
                backward(cache, y)

                tqdm.write(f'LONG TP HIT {start_time}')
                position = None
                in_trade = False
                return

            elif c['low'] <= stop:
                balance = balance - (balance * 0.30)
                losses += 1

                X = np.array(position_sample).reshape(1, -1)
                y = np.array([[0, 0, 1]])

                A3, cache = forward(X)
                backward(cache, y)

                tqdm.write(f'LONG SL HIT {start_time}')
                position = None
                in_trade = False
                return
#endregion

# region Save & Load Model
def save_model():
    np.savez('model.npz', W1=W1, B1=B1, W2=W2, B2=B2, W3=W3, B3=B3, mean=mean, std=std)

def load_model():
    global W1, B1, W2, B2, W3, B3, mean, std
    data = np.load('model.npz')

    W1 = data['W1']
    B1 = data['B1']
    W2 = data['W2']
    B2 = data['B2']
    W3 = data['W3']
    B3 = data['B3']
    mean = data['mean']
    std = data['std']
#endregion

# region Training 
def end_train():
    global stop_training
    if stop_training:
        return

    stop_training = True

    if wins and losses >= 1:
        average_win = (wins / (wins + losses)) * 100
    else:
        average_win = 0
    total_pnl = (balance - 275) / 275 * 100
    print("Train:", accuracy(X_train, y_train))
    print("Test:", accuracy(X_test, y_test) * 100,'%')
    print("\nTraining Has Ended")
    print("⏹⏹--- Summary ---⏹⏹")
    print(f"Total Trades: {total}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Total PnL (%): {total_pnl:.2f}%")
    print(f'Total Finished Sets: {finished_set:.2f}')
    print(f'Earned: {earned}')
    print(f"Final Balance: {balance:.2f}")
    print(f"Average Win (%): {average_win:.2f}%") 

def train(X, y, epochs=1000, batch_size=64):
    m = X.shape[0]
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0

        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            A3, cache= forward(X_batch)
            loss = compute_loss(A3, y_batch)
            epoch_loss += loss * len(X_batch)
            backward(cache, y_batch)
        
        if epoch % 100 == 0:
            acc = accuracy(X_test, y_test) * 100
            tqdm.write(f'Epoch {epoch}, Loss {epoch_loss}, Accuracy: {acc:.4f}%')

X_train, y_train, X_test, y_test = split(X, y)

if os.path.exists(current_model):
    load_model()
    print("Model loaded", current_model)
else:
    print('No Model saved training new one')
    train(X_train, y_train, epochs=1000)
    save_model()
#endregion

# region Bins
def get_bin(confidence):
    if confidence < 0.45:
        return "0.45-0.50"
    elif confidence < 0.55:
        return "0.50-0.55"
    elif confidence < 0.65:
        return "0.55-0.65"
    else:
        return "0.65+"

bin_stats = {
    "0.45-0.50": {"wins": 0, "total": 0},
    "0.50-0.55": {"wins": 0, "total": 0},
    "0.50-0.55": {"wins": 0, "total": 0},
    "0.55-0.65": {"wins": 0, "total": 0},
    "0.65+": {"wins": 0, "total": 0}
}

for i in tqdm(range(len(X_test)), desc="Building bins"):
    X_sample = X_test[i:i+1]
    pred = forward(X_sample)[0]

    confidence = np.max(pred)
    pred_class = np.argmax(pred)
    true_class = np.argmax(y_test[i])
    
    bin_name = get_bin(confidence)

    if pred_class == true_class:
        bin_stats[bin_name]["wins"] += 1

    bin_stats[bin_name]["total"] += 1

for k, v in bin_stats.items():
    if v["total"] > 0:
        winrate = v["wins"] / v["total"]
        bin_winrates[k] = winrate
        print(k, f"{winrate*100:.2f}%")

with open("bin_winrates.json", "w") as f:
    json.dump(bin_winrates, f)
    print('Bin data saved to bin_winrates.json')

#endregion

# region run Training
def run_training(mode='train'):
    global balance, wins, losses, total, bin_winrates, results, finished_set, earned

    first_symbol = list(symbol_groups.keys())[0]
    split_idx = int(len(symbol_groups[first_symbol]) * 0.3)

    pbar = tqdm(range(split_idx, len(symbol_groups[first_symbol]) - 3), desc="Backtesting")
    for i in pbar: 
        candidates = []

        for symbol, groups in symbol_groups.items():

            if i >= len(groups) - 3:
                continue

            current = groups[i]

            if balance >= 100000: 
                finished_set += 1
                earned += balance
                balance = 275

            if balance <= 20:
                end_train()
                break

            if position is not None:
                train_close_position(current)
                continue

            X_live = np.array([build_features(current)])
            X_live = (X_live - mean) / std

            A3, _ = forward(X_live)
            pred = A3[0]
            
            long_conf = pred[0]
            short_conf = pred[1]

            confidence_threshold = 0.55

            if long_conf > short_conf:
                direction = 'long'
                confidence = long_conf
            else:
                direction = 'short'
                confidence = short_conf

            if confidence < confidence_threshold:
                continue

            tqdm.write(f"Direction: {direction}")
            tqdm.write(f"L:{long_conf:.3f} S:{short_conf:.3f}")

            confidence_bin = get_bin(confidence)
            if confidence_bin not in bin_winrates:
                tqdm.write(f"SKIPPED | Bin: {confidence_bin} has no data")
                continue
            winrate = bin_winrates[confidence_bin]
            
            aggression_threshhold = 0.613

            candidates.append({
                "symbol": symbol,
                "group": current,
                "direction": direction,
                "confidence": confidence,
                "winrate": winrate
            })

        if not candidates:
            continue

        best = max(candidates, key=lambda x: (x['confidence'] ** 2) * x['winrate'])

        tqdm.write(f"BEST: {best['symbol']} | Dir: {best['direction']} | Conf: {best['confidence']:.3f} | Winrate: {best['winrate']:.2f}")

        best_group = best['group']
        best_conf = best['confidence']
        best_winrate = best['winrate']
        best_dir = best['direction']

        if best_winrate >= aggression_threshhold:
            if best_dir == 'long' and best_conf > 0.55:
                train_take_long(best_group, best_conf)

            elif best_dir == 'short' and best_conf > 0.55:
                train_take_short(best_group, best_conf)
            
            tqdm.write(f"TRADE ALLOWED | Bin: {confidence_bin} | Winrate: {winrate:.2f}")
        else:
            tqdm.write(f"SKIPPED | Bin: {confidence_bin} | Winrate too low: {winrate:.2f}")


    end_train()
# endregion

# region run Live Trading
def run_live(mode='live'):
    pass

#endregion

run_training(mode='train')
#run_live(mode='live')