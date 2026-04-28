# Multi-Factor Regime-Aware Stock Return Prediction and Risk-Adjusted Trading Strategy

## Overview

This project develops an end-to-end machine learning framework for predicting short-horizon stock return direction and building a risk-adjusted trading strategy. The main idea is that stock return predictability changes across market environments, so a single model trained across all time periods may miss important regime-specific patterns.

To address this, the project combines unsupervised regime detection with supervised return prediction. A Hidden Markov Model is used to classify market conditions into regimes such as Bull, Bear, and High-Volatility states. These regime labels are then used as features inside an XGBoost model that predicts next-day stock return direction. The final strategy uses volatility-aware position sizing and transaction cost assumptions to evaluate whether the model can outperform a passive S&P 500 benchmark on a risk-adjusted basis.

## Project Goal

The goal of this project is to test whether conditioning machine learning predictions on macroeconomic market regimes can improve trading performance compared to a traditional passive S&P 500 buy-and-hold strategy.

The central hypothesis is:

> Equity return predictability is not constant over time. Instead, it varies across different macroeconomic and volatility regimes. By identifying these regimes and conditioning predictions on them, a trading strategy can achieve stronger risk-adjusted performance.

## Key Features

- Regime-aware machine learning pipeline
- Hidden Markov Model for market state detection
- XGBoost classifier for next-day return direction prediction
- Macroeconomic, technical, and price-based feature engineering
- Walk-forward validation to reduce look-ahead bias
- GARCH-based volatility forecasting for dynamic position sizing
- Backtesting with realistic transaction costs
- Feature importance analysis by market regime
- Streamlit dashboard for model outputs and strategy visualization

## Data Sources

This project uses three main groups of data.

### 1. Stock Market Data

Daily price and volume data are collected using the `yfinance` Python library. The planned sample period covers:

- Training period: January 2015 to December 2024
- Testing period: January 2025 to December 2025

The project focuses on S&P 500 constituent stocks and benchmark comparison against the S&P 500.

### 2. Macroeconomic Data

Macroeconomic variables are collected from the Federal Reserve Economic Data API, including:

- Volatility Index
- Federal Funds Effective Rate
- Consumer Price Index
- 10-Year Treasury Yield
- 2-Year Treasury Yield
- 10-Year minus 2-Year Treasury yield spread

### 3. Technical Indicators

Technical indicators are engineered directly from price data, including:

- Relative Strength Index
- MACD
- Bollinger Band width
- Average True Range
- Rolling returns
- Rolling volatility
- Momentum indicators
- Macro-technical interaction terms

## Methodology

The project follows a four-stage workflow.

### 1. Feature Engineering

The first stage creates more than 30 predictive features from stock prices, volume, macroeconomic variables, and technical indicators. These features are designed to capture momentum, volatility, trend strength, and broader macroeconomic conditions.

Examples include:

- 1-day, 5-day, and 20-day rolling returns
- Rolling volatility
- RSI
- MACD
- Bollinger Band width
- Treasury yield spread
- CPI changes
- VIX changes
- Interaction terms between macro variables and technical signals

### 2. Market Regime Detection

A Gaussian Hidden Markov Model is trained on macroeconomic and volatility features to classify each trading period into one of three hidden regimes:

- Bull Market Regime
- Bear Market Regime
- High-Volatility Regime

The model uses only historical information available at the time of prediction to avoid look-ahead bias.

### 3. Return Direction Prediction

An XGBoost classifier predicts whether a stock's next-day return will be positive or negative. The detected market regime is included as an input feature so the model can adjust its predictions depending on the current market state.

Bayesian optimization is used to tune model hyperparameters.

### 4. Risk-Adjusted Strategy Backtesting

The trading strategy is backtested using walk-forward validation and rolling retraining windows. Position sizes are adjusted using volatility forecasts so the strategy reduces exposure during periods of high predicted risk.

The backtest includes:

- Out-of-sample testing
- Transaction cost modeling
- 0.1% cost per trade
- Benchmark comparison against S&P 500 buy-and-hold
- Performance evaluation across market subperiods

## Model Evaluation

The strategy will be evaluated using both predictive and financial performance metrics.

### Prediction Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Directional accuracy by regime

### Trading Performance Metrics

- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Alpha versus S&P 500
- Transaction-cost-adjusted return

## Expected Outputs

The final project will produce:

- A trained regime-aware stock return prediction model
- A complete backtesting framework
- Out-of-sample performance results
- Feature importance analysis
- Regime classification plots
- Cumulative return comparison against the S&P 500
- Robustness tests across different market periods
- Transaction cost sensitivity analysis
- Streamlit dashboard for interactive visualization
- Experiment logs for reproducibility

## Expected Project Structure

```text
multi-factor-regime-aware-trading/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_regime_detection.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_backtesting_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── regime_model.py
│   ├── train_model.py
│   ├── backtest.py
│   └── utils.py
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/multi-factor-regime-aware-trading.git
cd multi-factor-regime-aware-trading
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Main Python Libraries

This project uses the following major Python libraries:

```text
pandas
numpy
yfinance
fredapi
scikit-learn
xgboost
hmmlearn
arch
ta
matplotlib
seaborn
plotly
streamlit
optuna
mlflow
```

## Usage

### 1. Collect Data

```bash
python src/data_loader.py
```

### 2. Engineer Features

```bash
python src/feature_engineering.py
```

### 3. Detect Market Regimes

```bash
python src/regime_model.py
```

### 4. Train Prediction Model

```bash
python src/train_model.py
```

### 5. Run Backtest

```bash
python src/backtest.py
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Dashboard

The Streamlit dashboard is designed to display:

- Current predicted market regime
- Next-day return signal
- Model confidence score
- Cumulative strategy performance
- S&P 500 benchmark comparison
- Feature importance
- Drawdown chart
- Regime timeline

## Research Motivation

Financial markets are often modeled as if relationships between variables are stable over time. However, empirical evidence suggests that market behavior changes across different economic environments. A model that works during a bull market may perform poorly during a bear market or high-volatility period.

This project addresses that issue by allowing the model to recognize changing market states before making predictions. The combination of Hidden Markov Models, XGBoost, walk-forward validation, and volatility-based risk management creates a more realistic machine learning framework for financial prediction.

## Limitations

This project is for educational and research purposes only. It does not provide financial advice or guarantee trading profits.

Potential limitations include:

- Historical relationships may not persist in future markets
- Model performance may be sensitive to feature selection
- Transaction costs and slippage may be higher in real trading
- S&P 500 constituent changes may introduce survivorship bias
- Macroeconomic data may have reporting delays or revisions
- Short-horizon stock return prediction is inherently noisy

## Future Improvements

Potential future extensions include:

- Adding earnings call sentiment data
- Including news sentiment or Reddit sentiment
- Testing alternative regime models
- Comparing XGBoost with neural networks or random forests
- Expanding to multi-asset portfolios
- Adding Black-Litterman portfolio optimization
- Improving transaction cost and slippage modeling
- Deploying the dashboard with live data updates

## Disclaimer

This project is intended for academic and portfolio purposes. It should not be used as financial advice. Trading strategies involve risk, and past performance does not guarantee future results.

## Author

Kiran Thapa Chhetri

B.S. Managerial Economics and Data Science  
Virginia Tech
