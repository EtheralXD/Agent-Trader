# Agent Trader #
This is a free to use trading neural network. It uses ccxt to pull candle data on selected tokens and groups them into 4s and trades based on past candle group similaritys. 

### Required Dependencies 
Use requirements.txt to install these dependencies. 
- tqdm
- numpy
- ccxt 
- my-sql-connector-python
- dotenv

# Latest Release #
## [1.0.0] - 2026-09-4
Initial Release
### Notes
This is the initial release if Agent Trader

### Added
- data_collection scripts | data_collecter.py | label_builder.py | pattern_builder.py
- reset.py Resets the database and model 
- master.py Runs the full data collection suite
- agent.py This is the neural network
