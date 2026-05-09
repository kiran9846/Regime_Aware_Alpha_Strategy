# Multi-Factor Regime-Aware Stock Return Prediction and Risk-Adjusted Trading Strategy

## Overview

This project builds an end-to-end quantitative finance and machine learning pipeline for short-term stock market direction prediction and risk-adjusted trading strategy evaluation.

The goal is not only to predict whether the market will go up or down, but also to identify the market regime, forecast volatility, adjust position size, and evaluate the strategy against a passive buy-and-hold benchmark.

The project combines:

- Market and macroeconomic feature engineering
- Hidden Markov Model regime detection
- XGBoost next-day return direction classification
- GARCH volatility forecasting
- Volatility-adjusted position sizing
- Transaction-cost backtesting
- Streamlit dashboard deployment

The final results show that the model did not outperform buy-and-hold in total return, but it significantly reduced maximum drawdown. This makes the framework more useful as a risk-management system than a pure return-maximization strategy.

---

## Research Question

Can a regime-aware machine learning model improve short-term stock-market prediction and reduce downside risk compared with a simple buy-and-hold benchmark?

This project addresses that question by combining supervised prediction, unsupervised regime detection, volatility forecasting, realistic backtesting, and dashboard visualization.

---

## Project Pipeline

Stage 1 — Data Collection and Feature Engineering
Stage 2 — HMM Regime Detection
Stage 3 — XGBoost Return Direction Classification
Stage 4 — GARCH Volatility Forecasting and Position Sizing
Stage 5 — Backtesting and Performance Evaluation
Stage 6 — Streamlit Dashboard

## Stage 1 — Data Collection and Feature Engineering

Market data was collected using yfinance, and macroeconomic data was collected from FRED.

Data Sources
Source	Data
yfinance	S&P 500 daily OHLCV data
FRED	VIX, Federal Funds Rate, CPI, 10Y–2Y Treasury Yield Spread
Time Period
Training period: January 2015 – December 2024
Testing period:  January 2025 – December 2025
Engineered Features

The dataset includes technical, macroeconomic, and volatility-based features:

Daily returns
Lagged returns
Rolling volatility
RSI
MACD
Bollinger Band width
ATR
Moving averages
VIX change
VIX 20-day moving average
CPI year-over-year inflation
Real-rate proxy
Yield-spread change

The target variable is next-day return direction:

target = 1 if next_day_return > 0 else 0

This means:

1 = market goes up tomorrow
0 = market goes down or stays flat tomorrow

The final target distribution was approximately:

Up days:   54.5%
Down days: 45.5%

This makes the classification problem reasonably balanced, but still difficult because daily stock-market returns are noisy.

## Stage 2 — HMM Regime Detection

A Gaussian Hidden Markov Model was used to classify each trading day into one of three regimes:

Bull
Bear
High_Vol

The HMM used macro and market features such as:

VIX
VIX moving average
Yield spread
Federal Funds Rate
CPI year-over-year inflation
Real-rate proxy
Daily return
20-day volatility

KMeans initialization was used to improve regime stability before fitting the HMM.

Why HMM?

Market regimes are not directly observable. Investors can observe returns, volatility, VIX, interest rates, and macroeconomic variables, but the true market state is hidden. A Hidden Markov Model helps infer these hidden states from observable data.

## Stage 3 — XGBoost Return Direction Classifier

An XGBoost classifier was trained to predict whether the next trading day return would be positive or negative.

The model used:

Technical indicators
Macro indicators
Volatility features
HMM regime information

Walk-forward cross-validation was used to avoid lookahead bias, and Optuna was used for hyperparameter tuning.

Final 2025 Test Results
Metric	Value
Accuracy	51.61%
AUC	0.533
Precision for Down Class	0.43
Precision for Up Class	0.59

The model showed modest predictive power, which is realistic for next-day market direction forecasting.

## Stage 4 — GARCH Volatility Forecasting and Position Sizing

A GARCH(1,1) model with Student-t errors was used to forecast next-day volatility.

The GARCH forecast was then used for inverse-volatility position sizing:

position_size = target_daily_vol / garch_vol_forecast
position_size = position_size.clip(0, 1)

This means:

Lower predicted volatility  → larger position
Higher predicted volatility → smaller position

The final active trading position combines the XGBoost signal with GARCH sizing:

best_position = XGBoost_signal * position_size

So if GARCH allows a 100% position but XGBoost predicts Down, the active position becomes 0%.

GARCH Forecast Summary
Metric	Value
Mean Forecasted Daily Volatility	1.04%
Median Forecasted Daily Volatility	0.84%
Minimum Forecasted Daily Volatility	0.46%
Maximum Forecasted Daily Volatility	4.64%

## Stage 5 — Backtesting and Performance Evaluation

A vectorized backtest was used to evaluate the strategy on the 2025 out-of-sample period.

The strategy return was calculated as:

strategy_return = position * next_day_return

Transaction costs were included:

turnover = abs(position_today - position_yesterday)
cost = 0.001 * turnover
net_return = gross_return - cost

A 0.1% transaction cost was used to represent realistic trading friction.

Final Backtest Results
Metric	Original XGB+GARCH	Best XGB+GARCH	Buy & Hold

Gross Return	8.83%	8.66%	17.51%

Net Return	1.89%	2.40%	17.51%

Gross Sharpe	1.020	1.036	0.965

Net Sharpe	0.260	0.326	0.965

Sortino Ratio	0.257	0.309	1.210

Calmar Ratio	0.188	0.251	0.943

Gross Max Drawdown	-9.57%	-8.91%	-18.90%

Net Max Drawdown	-10.22%	-9.74%	-18.90%

Win Rate	59.32%	58.33%	58.06%

Days Traded	118	108	248

Days Flat	130	140	0

Total Turnover	65.86	59.25	1.00

Key Findings

The best XGBoost + GARCH strategy improved over the original strategy after transaction costs.

The best strategy achieved:

2.40% net return
0.326 net Sharpe ratio
-9.74% maximum drawdown
108 traded days
Lower turnover than the original strategy

Buy-and-hold achieved the highest total return at 17.51%, but it also had a much larger drawdown of -18.90%.

The main conclusion is:

The model did not outperform buy-and-hold in total return, but it reduced downside risk meaningfully.

This suggests that the framework is more useful as a risk-management strategy than a pure return-maximization strategy.

Regime-Level Performance
Original Strategy by Regime
Regime	Strategy Net	Buy & Hold	Advantage
Bull	2.12%	11.06%	-8.93%
Bear	0.90%	1.26%	-0.36%
High_Vol	-1.12%	4.50%	-5.62%
Best Strategy by Regime
Regime	Strategy Net	Buy & Hold	Advantage
Bull	2.38%	11.06%	-8.68%
Bear	0.90%	1.26%	-0.36%
High_Vol	-0.87%	4.50%	-5.37%

The best strategy improved slightly over the original strategy in Bull and High-Volatility regimes, but buy-and-hold still outperformed in total return.

The Bear regime had only one observation in the 2025 test period, so it should not be strongly interpreted.

Streamlit Dashboard

The project includes a Streamlit dashboard with:

Current regime badge
Latest prediction signal
Prediction confidence
GARCH volatility forecast
GARCH position size
Active trading position
Cumulative return chart
Drawdown chart
Performance metrics table
Regime distribution charts
GARCH volatility gauge
Signal explorer
Project Structure

## DS Quant Fin/
│
├── feature_engineering_dataset.ipynb
├── Regime_Detection_HMM.ipynb
├── XGBoost_return.ipynb
├── GARCH_modeling.ipynb
├── backtest_performance_evaluation.ipynb
│
├── full_data.csv
├── processed_data_with_regimes.csv
├── final_modeling_dataset.csv
├── XGBoost_results.csv
├── backtest_data.csv
│
├── final_backtest_original.csv
├── final_backtest_best.csv
├── final_backtest_buyhold.csv
├── performance_summary.json
│
├── backtest_performance.png
├── performance_metrics_heatmap.png
├── strategy_comparison.png
│
├── streamlit_app.py
└── README.md
## How to Run the Project
1. Clone the Repository
git clone <your-repo-link>
cd <your-repo-folder>
2. Install Required Packages
pip install pandas numpy matplotlib scikit-learn xgboost optuna yfinance fredapi hmmlearn arch streamlit plotly

If using Anaconda, install the packages inside your preferred environment.

3. Run the Notebooks in Order

Run the notebooks in this order:

1. feature_engineering_dataset.ipynb
2. Regime_Detection_HMM.ipynb
3. XGBoost_return.ipynb
4. GARCH_modeling.ipynb
5. backtest_performance_evaluation.ipynb

The order matters because each notebook creates files used by the next stage.

4. Run the Streamlit Dashboard

Make sure these files are in the same folder as streamlit_app.py:

final_backtest_original.csv
final_backtest_best.csv
final_backtest_buyhold.csv
performance_summary.json

Then run:

python -m streamlit run streamlit_app.py

If the browser does not open automatically, copy the local URL from Terminal. It usually looks like:

http://localhost:8501
How to Reproduce the Final Results

To reproduce the full pipeline:

Run data collection and feature engineering.
Run HMM regime detection.
Run XGBoost modeling.
Run GARCH volatility forecasting and position sizing.
Run the backtesting notebook.
Launch the Streamlit dashboard.
Technologies Used
Python
pandas
NumPy
scikit-learn
XGBoost
Optuna
hmmlearn
arch
yfinance
fredapi
Matplotlib
Plotly
Streamlit
Final Interpretation

This project shows that a regular prediction model is not enough for quantitative finance. A stronger workflow needs prediction, regime awareness, volatility forecasting, risk management, transaction-cost modeling, and benchmark comparison.

The XGBoost model alone had modest predictive power, with about 51.6% accuracy and 0.533 AUC. However, when combined with HMM regimes and GARCH volatility-based position sizing, it became a complete risk-aware trading framework.

The final model did not outperform buy-and-hold in total return, but it reduced maximum drawdown from -18.90% to -9.74%. This shows that the model is more valuable as a risk-managed trading framework than as a pure return-maximization strategy.

Author

Kiran Thapa Chhetri
Managerial Economics and Data Science
Virginia Tech
