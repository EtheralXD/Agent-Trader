# Step 1
--@block
DROP TABLE IF EXISTS group_labels;
DROP TABLE IF EXISTS raw_candles;
DROP TABLE IF EXISTS candle_differences;
DROP TABLE IF EXISTS total_group_differences;
DROP TABLE IF EXISTS pattern_groups;
DROP TABLE IF EXISTS chart_patterns;
DROP TABLE IF EXISTS symbols;


# Step 2
--@block 
CREATE TABLE symbols (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE
);

CREATE TABLE chart_patterns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol_id INT,
    pattern_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

CREATE TABLE pattern_groups (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol_id INT,
    pattern_id INT,
    group_index INT,
    start_time DATETIME,
    FOREIGN KEY (pattern_id) REFERENCES chart_patterns(id)
);

CREATE TABLE total_group_differences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    group_id INT,
    total_value DECIMAL(10,5),
    FOREIGN KEY (group_id) REFERENCES pattern_groups(id)
);

CREATE TABLE candle_differences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    total_dif_id INT,
    candle_index INT,
    difference_value DECIMAL(10,5),
    FOREIGN KEY (total_dif_id) REFERENCES total_group_differences(id)
);

CREATE TABLE raw_candles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol_id INT,
    timestamp DATETIME,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT,
    timeframe VARCHAR(10),

    UNIQUE KEY unique_candle (symbol_id, timestamp, timeframe),

    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

CREATE TABLE group_labels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    group_id INT,
    future_group_id INT,
    future_return FLOAT,
    direction INT,
    FOREIGN KEY (group_id) REFERENCES pattern_groups(id)
);

# Step 3
--@block
ALTER TABLE candle_differences
ADD COLUMN open_price FLOAT,
ADD COLUMN close_price FLOAT,
ADD COLUMN high_price FLOAT,
ADD COLUMN low_price FLOAT;

CREATE INDEX idx_total_value ON total_group_differences(total_value);
CREATE INDEX idx_group_time ON pattern_groups(start_time);
CREATE INDEX idx_group_id ON candle_differences(total_dif_id);

--@block
SELECT * FROM symbols;