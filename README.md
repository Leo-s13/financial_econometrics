# Financial Econometrics Dashboard

Econometric analysis of stock price volatility using GARCH models for Goldman Sachs, Bank of America, and MetLife.

## Features

- Stylized facts analysis (stationarity, heavy tails, volatility clustering, leverage effects)
- GARCH family models (GARCH, GJR-GARCH, EGARCH)
- Risk measures (VaR and Expected Shortfall)

## Quick Start

### Prerequisites
- Python 3.11
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/Leo-s13/financial_econometrics.git
cd financial_econometrics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run dashboard.py
```

The app will open at `http://localhost:8501`

## Dependencies

- streamlit
- pandas, numpy
- plotly, matplotlib, seaborn
- statsmodels, arch
- openpyxl

## Troubleshooting

**Module errors:**
```bash
pip install -r requirements.txt --upgrade
```