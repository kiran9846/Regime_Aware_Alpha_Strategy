# Multi-Factor Regime-Aware Stock Return Prediction and Risk-Adjusted Trading Strategy

## Project Overview

This project builds an end-to-end quantitative finance and data science pipeline for predicting short-term stock market direction and evaluating a risk-adjusted trading strategy.

The main research question is:

**Can a regime-aware machine learning model improve short-term stock-market prediction and reduce downside risk compared with a simple buy-and-hold benchmark?**

Instead of only building a regular prediction model, this project combines:

- Market data and macroeconomic data
- Feature engineering
- Hidden Markov Model regime detection
- XGBoost next-day return direction classification
- GARCH volatility forecasting
- Volatility-adjusted position sizing
- Transaction-cost backtesting
- Streamlit dashboard deployment

The final result shows that the model did not outperform buy-and-hold in total return, but it reduced maximum drawdown substantially. This makes the framework more useful as a risk-management system than a pure return-maximization model.

---

## Research Question

The project focuses on the following question:

**Can combining HMM market regimes, XGBoost prediction, and GARCH volatility-based position sizing create a more risk-controlled trading strategy than a standard buy-and-hold benchmark?**

This question is important because financial markets change across different environments. A model that works in a calm Bull market may not work the same way during a High-Volatility market. Therefore, this project does not only predict market direction. It also identifies market regime, forecasts volatility, adjusts position size, and evaluates performance after transaction costs.

---

## Expected Output

The expected output of this project is a complete quant finance pipeline that can:

1. Collect and process market and macroeconomic data.
2. Engineer technical and macro-financial features.
3. Detect market regimes using a Hidden Markov Model.
4. Predict next-day return direction using XGBoost.
5. Forecast volatility using GARCH.
6. Adjust position size based on predicted volatility.
7. Backtest the strategy against buy-and-hold.
8. Display the final results in a Streamlit dashboard.

---

## Project Pipeline

The project is divided into five main stages:

```text
Stage 1 — Data Collection and Feature Engineering
Stage 2 — HMM Regime Detection
Stage 3 — XGBoost Return Direction Classification
Stage 4 — GARCH Volatility Forecasting and Position Sizing
Stage 5 — Backtesting and Performance Evaluation
```

A Streamlit dashboard is also included for final visualization and presentation.

---

# Stage 1 — Data Collection and Feature Engineering

## Goal

The goal of Stage 1 is to collect historical market and macroeconomic data, clean it, merge it, and create the final modeling dataset.

## Data Sources

The project uses:

- `yfinance` for S&P 500 daily market data
- FRED API for macroeconomic indicators

The market data includes:

- Open
- High
- Low
- Close
- Volume

The macroeconomic data includes:

- VIX
- Federal Funds Rate
- CPI
- 10Y minus 2Y Treasury yield spread

## Date Range

The dataset covers:

```text
Training period: January 2015 – December 2024
Testing period:  January 2025 – December 2025
```

The model is trained on 2015–2024 data and tested on 2025 data.

## Feature Engineering

The project creates technical and macro-financial features including:

- Daily return
- Lagged returns
- Rolling volatility
- RSI
- MACD
- Bollinger Band width
- ATR
- Moving averages
- VIX change
- VIX 20-day moving average
- CPI year-over-year inflation
- Real-rate proxy
- Yield-spread change

## Target Variable

The target variable is next-day return direction:

```python
target = 1 if next_day_return > 0 else 0
```

This means:

```text
1 = market goes up tomorrow
0 = market goes down or stays flat tomorrow
```

The final target distribution was approximately:

```text
Up days:   54.5%
Down days: 45.5%
```

This makes the classification problem reasonably balanced, but still difficult because daily stock-market returns are noisy.

## Stage 1 Output

The main output from Stage 1 is:

```text
full_data.csv
final_modeling_dataset.csv
```

These files contain the processed market data, macro data, engineered features, and target variable.

---

# Stage 2 — HMM Regime Detection

## Goal

The goal of Stage 2 is to identify hidden market regimes using a Gaussian Hidden Markov Model.

The model classifies each trading day into one of three regimes:

```text
Bull
Bear
High_Vol
```

## Why HMM?

A Hidden Markov Model is useful because market regimes are not directly observed. We observe market indicators such as VIX, returns, volatility, and yield spread, but we do not directly observe the true market state. HMM estimates these hidden states from the data.

## HMM Features

The HMM was trained using macro and market features:

```text
vix
vix_ma_20
yield_spread
fedfunds
cpi_yoy
real_rate
return_1d
volatility_20d
```

These features were selected because they capture:

- Market fear
- Volatility conditions
- Monetary policy environment
- Inflation environment
- Yield curve conditions
- Market momentum
- Market risk

## Model Setup

The HMM used:

```text
Model: GaussianHMM
Number of states: 3
Covariance type: full
Initialization: KMeans
Training period: 2015–2024
Testing period: 2025
```

KMeans was used to initialize the HMM states more reliably.

## Regime Labeling

The raw HMM states were mapped into meaningful labels based on regime summaries:

```text
High VIX + high volatility = High_Vol
Positive return + low volatility = Bull
Weaker or defensive market environment = Bear
```

## Stage 2 Output

The main output from Stage 2 is:

```text
processed_data_with_regimes.csv
```

This file includes the original features plus the regime labels.

---

# Stage 3 — XGBoost Return Direction Classifier

## Goal

The goal of Stage 3 is to predict next-day market direction using XGBoost.

The model predicts:

```text
1 = next-day return is positive
0 = next-day return is negative or flat
```

## Why XGBoost?

XGBoost is a strong machine learning model for structured tabular data. It can capture nonlinear relationships and interactions between financial features. This is useful because stock-market behavior is complex and may depend on technical signals, macro variables, volatility, and regime conditions at the same time.

## Features Used

The XGBoost model used:

- Technical indicators
- Macro indicators
- Rolling volatility features
- Lagged returns
- HMM regime information
- GARCH-related risk features where applicable

## Train-Test Split

The split was:

```text
Training: January 2015 – December 2024
Testing:  January 2025 – December 2025
```

The model was trained only on past data and tested on unseen 2025 data.

## Walk-Forward Cross-Validation

Walk-forward cross-validation was used to avoid lookahead bias.

This means the model was trained on earlier data and validated on later data in each fold. This better simulates real trading because future data is never used to train the model before the prediction date.

## Optuna Tuning

Optuna was used to tune XGBoost hyperparameters such as:

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- gamma
- reg_alpha
- reg_lambda

## Final XGBoost Test Results

The 2025 test performance was:

| Metric | Value |
|---|---:|
| Accuracy | 51.61% |
| AUC | 0.533 |
| Precision for Down Class | 0.43 |
| Precision for Up Class | 0.59 |

The model showed modest predictive power, which is realistic for next-day stock-market direction prediction.

## Stage 3 Output

The main output from Stage 3 is:

```text
XGBoost_results.csv
```

This file includes:

- Target variable
- Predicted direction
- Probability of up move
- Regime features
- Return data
- Model prediction results

---

# Stage 4 — GARCH Volatility Forecasting and Position Sizing

## Goal

The goal of Stage 4 is to forecast volatility using GARCH and use that forecast to size trading positions.

This stage answers:

```text
If the model decides to trade, how large should the position be?
```

## Why GARCH?

GARCH is used because financial returns often show volatility clustering. High-volatility periods tend to be followed by high-volatility periods, and low-volatility periods tend to stay calm for some time.

GARCH helps estimate time-varying market risk.

## Model Setup

The model used:

```text
Model: GARCH(1,1)
Distribution: Student-t
Input: daily returns in percentage form
Training diagnostic period: 2015–2024
Forecasting method: expanding-window forecast for 2025
```

Student-t was used because financial returns often have fat tails.

## GARCH Sanity Check

The GARCH model produced a realistic one-step-ahead daily volatility forecast:

```text
Forecasted daily volatility: about 1.04%
Expected normal range: about 0.5% to 2.5%
```

The full 2025 GARCH forecast summary was approximately:

| Metric | Value |
|---|---:|
| Mean Forecasted Volatility | 1.04% |
| Median Forecasted Volatility | 0.84% |
| Minimum Forecasted Volatility | 0.46% |
| Maximum Forecasted Volatility | 4.64% |

## Position Sizing

The GARCH volatility forecast was converted into a position size using inverse volatility sizing:

```python
position_size = target_daily_vol / garch_vol_forecast
position_size = position_size.clip(0, 1)
```

This means:

```text
Low predicted volatility  → larger position
High predicted volatility → smaller position
```

The position size was capped at 100%.

## Active Position

The final active position combines the XGBoost signal and GARCH position size:

```python
best_position = XGBoost_signal * position_size
```

This means:

```text
If XGBoost predicts Up:   take the GARCH-sized position
If XGBoost predicts Down: stay flat
```

Example:

```text
GARCH position size = 100%
XGBoost signal = Down
Active position = 0%
```

## Stage 4 Output

The main output from Stage 4 is:

```text
backtest_data.csv
```

This file includes:

- Predicted direction
- Probability of up move
- Regime label
- GARCH volatility forecast
- GARCH position size
- Final active position
- Strategy return
- Buy-and-hold return

---

# Stage 5 — Backtesting and Performance Evaluation

## Goal

The goal of Stage 5 is to evaluate whether the model strategy performs better than buy-and-hold.

The backtest compares:

```text
Original XGB + GARCH strategy
Best XGB + GARCH strategy
Buy-and-hold benchmark
```

## Backtest Formula

The strategy return is calculated as:

```python
strategy_return = position * next_day_return
```

Transaction costs are included:

```python
turnover = abs(position_today - position_yesterday)
cost = 0.001 * turnover
net_return = gross_return - cost
```

A 0.1% transaction cost is used to represent realistic trading friction.

## Performance Metrics

The backtest evaluates:

- Gross return
- Net return
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Win rate
- Days traded
- Days flat
- Total turnover
- Total transaction cost

## Final Backtest Results

| Metric | Original XGB+GARCH | Best XGB+GARCH | Buy & Hold |
|---|---:|---:|---:|
| Gross Return | 8.83% | 8.66% | 17.51% |
| Net Return | 1.89% | 2.40% | 17.51% |
| Gross Sharpe | 1.020 | 1.036 | 0.965 |
| Net Sharpe | 0.260 | 0.326 | 0.965 |
| Sortino Ratio | 0.257 | 0.309 | 1.210 |
| Calmar Ratio | 0.188 | 0.251 | 0.943 |
| Gross Max Drawdown | -9.57% | -8.91% | -18.90% |
| Net Max Drawdown | -10.22% | -9.74% | -18.90% |
| Win Rate | 59.32% | 58.33% | 58.06% |
| Days Traded | 118 | 108 | 248 |
| Days Flat | 130 | 140 | 0 |
| Total Turnover | 65.86 | 59.25 | 1.00 |

## Key Backtest Interpretation

Buy-and-hold produced the highest total return:

```text
Buy & Hold Net Return: 17.51%
Best Strategy Net Return: 2.40%
```

However, the model strategy reduced drawdown significantly:

```text
Buy & Hold Max Drawdown: -18.90%
Best Strategy Max Drawdown: -9.74%
```

This means the model did not beat buy-and-hold in total return, but it reduced downside risk. The final framework is therefore stronger as a risk-management strategy than as a pure return-maximization strategy.

---

# Regime-Level Performance

The strategy was also evaluated by market regime.

## Original Strategy by Regime

| Regime | Strategy Net | Buy & Hold | Advantage |
|---|---:|---:|---:|
| Bull | 2.12% | 11.06% | -8.93% |
| Bear | 0.90% | 1.26% | -0.36% |
| High_Vol | -1.12% | 4.50% | -5.62% |

## Best Strategy by Regime

| Regime | Strategy Net | Buy & Hold | Advantage |
|---|---:|---:|---:|
| Bull | 2.38% | 11.06% | -8.68% |
| Bear | 0.90% | 1.26% | -0.36% |
| High_Vol | -0.87% | 4.50% | -5.37% |

## Regime Interpretation

The best strategy improved slightly over the original strategy in both Bull and High-Volatility regimes.

However, buy-and-hold still outperformed in every regime. The model missed upside during Bull periods and struggled during High-Volatility periods.

The Bear regime had only one observation in the 2025 test set, so it should not be strongly interpreted.

---

# Visual Outputs

The project includes several visual outputs:

- Cumulative return chart
- Drawdown curve
- Rolling Sharpe ratio
- Daily return bar chart
- Performance metrics heatmap
- Regime distribution chart
- GARCH volatility chart
- Streamlit dashboard

Important saved visuals include:

```text
backtest_performance.png
performance_metrics_heatmap.png
strategy_comparison.png
```

---

# Streamlit Dashboard

The project includes a Streamlit dashboard for final presentation.

The dashboard includes:

- Current regime badge
- Latest prediction signal
- Prediction confidence
- GARCH volatility forecast
- GARCH position size
- Active trading position
- Cumulative return chart
- Drawdown chart
- Performance metrics table
- Regime distribution bar chart
- Regime distribution pie chart
- GARCH volatility gauge
- Signal explorer
- Model confidence distribution

The dashboard uses these files:

```text
final_backtest_original.csv
final_backtest_best.csv
final_backtest_buyhold.csv
performance_summary.json
streamlit_app.py
```

---

# Project Files

## Main Notebooks

| File | Description |
|---|---|
| `feature_engineering_dataset.ipynb` | Data collection and feature engineering |
| `Regime_Detection_HMM.ipynb` | HMM regime detection |
| `XGBoost_return.ipynb` | XGBoost return direction classifier |
| `GARCH_modeling.ipynb` | GARCH volatility forecasting and position sizing |
| `backtest_performance_evaluation.ipynb` | Backtesting and final evaluation |

## Main Data Files

| File | Description |
|---|---|
| `full_data.csv` | Full historical processed dataset |
| `processed_data_with_regimes.csv` | Dataset with HMM regime labels |
| `final_modeling_dataset.csv` | Final modeling dataset |
| `XGBoost_results.csv` | XGBoost predictions and probabilities |
| `backtest_data.csv` | Stage 4 output for backtesting |
| `final_backtest_original.csv` | Original strategy backtest results |
| `final_backtest_best.csv` | Best strategy backtest results |
| `final_backtest_buyhold.csv` | Buy-and-hold benchmark results |
| `performance_summary.json` | Summary metrics used by Streamlit |
| `streamlit_app.py` | Streamlit dashboard code |

## Main Visual Files

| File | Description |
|---|---|
| `backtest_performance.png` | Main backtest performance chart |
| `performance_metrics_heatmap.png` | Heatmap of strategy performance metrics |
| `strategy_comparison.png` | Strategy comparison visualization |

---

# How to Run the Project

## 1. Install Required Packages

Run the following command in Terminal:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost optuna yfinance fredapi hmmlearn arch streamlit plotly
```

If using Anaconda, you can install packages inside your base environment or a separate project environment.

---

## 2. Run the Notebooks in Order

Run the notebooks in this order:

```text
1. feature_engineering_dataset.ipynb
2. Regime_Detection_HMM.ipynb
3. XGBoost_return.ipynb
4. GARCH_modeling.ipynb
5. backtest_performance_evaluation.ipynb
```

The order matters because each notebook creates output files used by the next stage.

---

## 3. Create the Streamlit App

If `streamlit_app.py` does not already exist, create it from a notebook cell using:

```python
%%writefile streamlit_app.py

# paste the Streamlit dashboard code here
```

Then run that cell once.

---

## 4. Run the Streamlit Dashboard

Make sure these files are in the same folder:

```text
streamlit_app.py
final_backtest_original.csv
final_backtest_best.csv
final_backtest_buyhold.csv
performance_summary.json
```

Then open Terminal in that folder and run:

```bash
python -m streamlit run streamlit_app.py
```

If the browser does not open automatically, copy the local URL from Terminal. It usually looks like:

```text
http://localhost:8501
```

Paste it into your browser.

---

# How to Reproduce Final Results

To reproduce the full project:

1. Run the feature engineering notebook.
2. Run the HMM regime detection notebook.
3. Run the XGBoost modeling notebook.
4. Run the GARCH volatility and position sizing notebook.
5. Run the backtesting notebook.
6. Run the Streamlit dashboard.

Each notebook saves intermediate output files used by the next stage.

---

# Final Interpretation

This project shows that a regular prediction model is not enough for quantitative finance. A strong financial modeling workflow needs prediction, regime awareness, risk control, transaction-cost modeling, and benchmark comparison.

The XGBoost model alone had modest predictive power, with about 51.6% accuracy and 0.533 AUC. However, when combined with HMM regimes and GARCH volatility-based position sizing, it became a complete risk-aware trading framework.

The final model did not outperform buy-and-hold in total return, but it reduced maximum drawdown from -18.90% to -9.74%. This shows that the model is more valuable as a risk-managed trading framework than as a pure return-maximization strategy.

---

# Key Takeaways

- A regular prediction model only predicts direction.
- This project builds a full quantitative finance system.
- HMM identifies market regimes.
- XGBoost predicts next-day direction.
- GARCH forecasts volatility and controls position size.
- Backtesting includes realistic transaction costs.
- Buy-and-hold produced higher total return.
- The model strategy reduced maximum drawdown.
- The final result is strongest as a risk-management framework.

---

# References

- Hamilton, J. D. Regime-switching models and economic time series.
- Engle, R. F. ARCH models for conditional heteroskedasticity.
- Bollerslev, T. Generalized ARCH/GARCH volatility modeling.
- Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System.
- FRED Economic Data, Federal Reserve Bank of St. Louis.
- yfinance Python package.
- Python `arch` package documentation.
- Streamlit and Plotly documentation.

---

# Author

Kiran Thapa Chhetri  
Managerial Economics and Data Science  
Virginia Tech  

