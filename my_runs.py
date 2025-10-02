import student
import mltester 
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ["TLT","GLD","XLP","XLU","XLV"]


'''# This was the initial data download and processing step. It is quoted out to avoid
# using up yfinance's API quota.

# Download data
df = yf.download(tickers, group_by='Ticker', start="2020-01-01", end="2024-12-31")

# Reshape data to create one row per date + ticker combination
df = df.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker'])

# Save the data to a parquet file
df.to_parquet("data/prices.parquet")'''

# Read the data from a parquet file
df = pd.read_parquet("data/prices.parquet")

# Create a matrix of features X and the target vector y
X = df # close gets dropped in the student class
y = df["Close"]

# Run the student
student = student.Student()
student.fit(X, y)
y_pred = student.predict(df)

# Create a new dataframe with the actual and predicted values
new_df = df.join(y_pred.rename('y_pred'), how = 'inner')
new_df['error'] = new_df['Close'] - new_df['y_pred']

# Create wide dataframes with the actual, predicted, and error values

actual = (
    new_df.reset_index()
      .pivot(index='Date', columns='Ticker', values='Close')  # or 'Adj Close'
      .sort_index()
)

predicted = (
    new_df.reset_index()
      .pivot(index='Date', columns='Ticker', values='y_pred')  # or 'Adj Close'
      .sort_index()
)

error = (
    new_df.reset_index()
      .pivot(index='Date', columns='Ticker', values='error')  # or 'Adj Close'
      .sort_index()
)

negative_error = error[error < 0]
positive_error = error[error > 0]

# Calculate metrics
mae = [np.mean(np.abs(error[ticker])) for ticker in tickers]
rmse = [np.sqrt(np.mean(error[ticker]**2)) for ticker in tickers]

# Plot the actual, predicted, and error values for each ticker in a 3x1 subplot 
# with each subplot displaying each ticker

fig, ax = plt.subplots(5, 1, sharex=False)
fig.set_size_inches(5, 15)
plt.subplots_adjust(hspace=0.5)  # increase space between rows

for i in range(len(tickers)):
    ax[i].plot(actual.index, actual[tickers[i]], label='Actual', color='black')
    ax[i].plot(predicted.index, predicted[tickers[i]], label='Predicted')
    ax[i].bar(negative_error.index, negative_error[tickers[i]], label='Error', color='red')
    ax[i].bar(positive_error.index, positive_error[tickers[i]], color='green')
    ax[i].plot(actual.index, np.zeros(len(actual.index)), color='grey', alpha=0.5, linewidth=0.1)
    ax[i].set_ylabel('Close')
    ax[i].set_title(f'{tickers[i]} (MAE: {mae[i]:.1f}, RMSE: {rmse[i]:.1f})')
    
ax[0].legend()
ax[4].set_xlabel('Date')


# Save the plot to a file
plt.savefig('plots/my_predictions.png')