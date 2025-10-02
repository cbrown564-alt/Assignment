import mltester

tickers = ["TLT","GLD","XLP","XLU","XLV"]

mltester.run_mltester(model_spec="student.py:Student", 
tickers=tickers, 
data_file="data/prices.csv", 
start="2020-01-01", end="2024-12-31", 
horizon=10, step=1, 
out_dir="outputs")