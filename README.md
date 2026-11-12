# Financial Econometrics Dashboard

**An interactive Python application for analysing stock-return dynamics, modelling volatility, and estimating market risk.**

[**Launch the dashboard**](https://financialeconometrics-schneider-turon.streamlit.app/) · [**Source code**](https://github.com/Leo-s13/financial_econometrics)

## Overview

Financial returns often exhibit heavy tails, volatility clustering, and asymmetric responses to market shocks. These properties matter when estimating market risk: a constant-volatility model with normally distributed innovations may fail to capture changing conditions and extreme observations.

Developed as a university financial econometrics project, this dashboard investigates these properties using historical equity prices for **Goldman Sachs, Bank of America, and MetLife** from **2007 to 2023**, sourced from **Yahoo Finance**. It combines statistical analysis, conditional-volatility modelling, and risk estimation in an interactive Streamlit interface.

The application is organised into four sections:

| Section | What it covers |
| --- | --- |
| **Market overview** | Historical prices and daily log returns for the three companies |
| **Analysis of returns** | Stationarity, distributional properties, autocorrelation, volatility clustering, and asymmetry |
| **Risk measures** | Historical-simulation VaR/Expected Shortfall and comparison of GARCH-family models |
| **Forecasting** | Conditional-volatility forecasts and model-based VaR/Expected Shortfall projections |

## Methods and functionality

### 1. Exploratory return analysis

The dashboard transforms observed prices into daily log returns and examines common stylised facts of financial time series. Its analyses include:

- **Stationarity:** Augmented Dickey–Fuller (ADF) test on log returns.
- **Distributional shape:** skewness, excess kurtosis, Q–Q plots, and Jarque–Bera normality tests.
- **Serial dependence:** autocorrelation of returns and squared returns.
- **Time-varying volatility:** visual inspection of volatility clustering using squared returns.
- **Asymmetric reactions:** exploration of the relationship between returns and subsequent squared returns.

These diagnostics help motivate the choice of conditional-volatility models and innovation distributions.

### 2. Conditional-volatility models

The application fits and compares the following specifications using the Python [`arch`](https://arch.readthedocs.io/) library:

| Model | Purpose |
| --- | --- |
| **GARCH(1,1)** | Captures persistent, time-varying conditional volatility |
| **GJR-GARCH(1,1)** | Allows positive and negative return shocks to affect volatility differently |
| **EGARCH(1,1)** | Models log conditional variance and can represent asymmetric shock responses |

Users can compare normal, Student's *t*, and skewed Student's *t* innovation distributions in the model-comparison interface. Model examination includes **log-likelihood, AIC, BIC, estimated parameters**, and residual diagnostics, including Ljung–Box tests and visual checks of standardised residuals.

These measures support model assessment; they do not, by themselves, establish superior out-of-sample forecast performance.

### 3. Market-risk estimation

The dashboard computes two downside-risk measures:

- **Value at Risk (VaR):** a loss threshold associated with a selected confidence level.
- **Expected Shortfall (ES):** the average loss conditional on exceeding the corresponding VaR threshold.

The **historical-simulation** section estimates both measures from rolling empirical return windows. A separate **GARCH-based** workflow generates rolling one-step-ahead risk estimates from fitted conditional-volatility models and plots them against realised returns.

### 4. Volatility and risk forecasting

The forecasting page lets users select a time series, model specification, innovation distribution, estimation split, forecast horizon, and VaR confidence level. It then displays projected conditional volatility, return means, VaR, and ES.

The interface provides visual comparisons and forecast tables for exploring how model assumptions affect risk estimates. Forecast outputs should be treated as exploratory: the repository does not provide a comprehensive benchmarked out-of-sample forecast evaluation or formal VaR backtesting study.

## Data

The application uses historical price data sourced from [Yahoo Finance](https://finance.yahoo.com/), covering **2007–2023**, and stored in the bundled [`data/prices.xlsx`](data/prices.xlsx) file for:

- Goldman Sachs
- Bank of America
- MetLife

Daily log returns are calculated as `log(P_t / P_(t-1))`. The application uses this historical dataset rather than retrieving live market prices.

## Run locally

**Requirements:** Python 3.11 and `pip`.

```bash
git clone https://github.com/Leo-s13/financial_econometrics.git
cd financial_econometrics

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run dashboard.py
```

Streamlit will display the local address (typically `http://localhost:8501`).

**Main technologies:** Python, Streamlit, pandas, NumPy, SciPy, statsmodels, `arch`, and Plotly.

## Project structure

```text
financial_econometrics/
├── dashboard.py       # Streamlit interface and analysis/model code
├── data/
│   └── prices.xlsx    # Historical price data
├── requirements.txt   # Python dependencies
└── README.md
```

## Scope

This is an **educational financial econometrics project** focused on the modelling and interpretation of historical equity-market risk. It is not a live trading system, portfolio optimiser, or source of investment advice. Model-based risk estimates depend on the selected sample, specification, distributional assumptions, and estimation window.
