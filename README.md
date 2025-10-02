# Predict the Stock Market

## Purpose

The goal is to build a model that can predict the next n-day prices for individual stocks. It should minimise the loss on three metrics: directional accuracy, MAE and RMSE. The main evaluation method is the walk-forward evaluation, so it is critical to avoid data leakage.

## Workflow

### Typical Workflow

The typical machine learning workflow is:

- Collect data
- Perform some EDA
- Clean / wrangle it
- Conduct feature selection and engineering
- Split the data for test/train/validation
- Train the model
- Evaluate the model
- Recycle the last two steps - tune hyperparameters, etc.

### Specific Workflow

For us that means:

- Extract data from the yfinance package (yahoo finance)
- Analyse the raw outputs; volume, close price, trends, distributions, correlations
- Reshape the data to one observation per (date, ticker) and review the data coverage
- Explore ways to turn raw data into features e.g. lagged values, ranges, summary statistics
- Assess how to split the data in a time-appropriate way to avoid data leakage / 'snooping'
- Create a benchmark with the existing ridge regression, then explore SVMs, GBMs, LSTMs and potentially transformers to train the models
  - Need to explore the correct cost function to train on
- Assess the metrics for directional accuracy, MAE and RMSE
- Iterate testing for hyperparameter selection (e.g. leaf size for GBMs) and feature selection
- Test against the walk-forward evaluation

Throughout the process I need to keep reproducibility in mind, so test in Colab and Jupyter, test on Mac Mini and Windows laptop. Clear documentation as I go along would be heplful too...

The Python module we'll be building on top of for the evaluation is in `student.py` and it will be tested against `mltester.py`.

## Implementation Notes

I am using uv instead of pip.

To create the pip-compatible requirements.txt file I need to use the following command:

> uv export --without-hashes > requirements.txt

