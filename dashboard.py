import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model
from typing import Dict, Tuple

# --- Configuration & Style ---
st.set_page_config(
    page_title="Institutional Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling for the interface
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #4e8cff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading & Processing Layer ---

@st.cache_data
def load_and_process_data(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Loads raw Excel data and performs robust preprocessing.
    Cached to optimize performance on interaction.
    """
    try:
        # Load raw data (skipping the header structure row 0 used in the notebook)
        # Note: We assume the file is in the same directory or 'data/' folder
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please check the path.")
        return {}

    stocks = {}
    
    # Define mapping based on the notebook structure
    # GS: cols 0-1, BAC: cols 2-3, MET: cols 4-5
    mappings = [
        ("Goldman Sachs (GS)", 0, 2),
        ("Bank of America (BAC)", 2, 4),
        ("Metlife (MET)", 4, 6)
    ]

    for name, start_col, end_col in mappings:
        df = data.iloc[1:, start_col:end_col].copy()
        df.columns = ['date', 'price']
        
        # Robust conversion of Excel serial dates
        df['date'] = pd.to_numeric(df['date'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], unit='D', origin='1899-12-30')
        
        # Numeric conversion and return calculations
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna().sort_values('date').reset_index(drop=True)
        
        # Financial Engineering Features
        df['return'] = df['price'].pct_change()
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df['volatility_30d'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)
        
        stocks[name] = df.dropna()

    return stocks

# --- Visualization Components ---

def plot_interactive_price(df: pd.DataFrame, ticker: str):
    """Generates a professional candlestick/line chart using Plotly."""
    fig = px.line(df, x='date', y='price', title=f'{ticker} Price History')
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def plot_returns_distribution(df: pd.DataFrame, ticker: str):
    """Plots histogram of returns with KDE-like overlay."""
    fig = px.histogram(df, x='log_return', nbins=100, 
                       title=f'{ticker} Log-Returns Distribution',
                       opacity=0.75, marginal="box")
    fig.update_layout(template="plotly_white", xaxis_title="Log Return")
    return fig

def fit_garch_model(returns: pd.Series):
    """
    Fits a GARCH(1,1) model to the returns.
    This fixes the 'NameError' seen in the original notebook traceback.
    """
    # Scale returns by 100 for better numerical stability in GARCH optimization
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    return results

# --- Main Application Logic ---

def main():
    st.title("🏦 Institutional Stock Analysis Dashboard")
    st.markdown("Interactive financial analysis environment for GS, BAC, and MET.")

    # 1. Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File uploader allows user to swap data easily
        uploaded_file = st.file_uploader("Upload Price Data (xlsx)", type=['xlsx'])
        
        if uploaded_file is not None:
            data_source = uploaded_file
        else:
            # Fallback to local path if configured, or show warning
            data_source = 'data/prices.xlsx' 
            # In a real deployment, we might disable this default if file is missing
        
        selected_stock = st.selectbox(
            "Select Asset", 
            ["Goldman Sachs (GS)", "Bank of America (BAC)", "Metlife (MET)"]
        )
        
        analysis_mode = st.radio(
            "Analysis View",
            ["Market Overview", "Risk & Returns", "GARCH Volatility"]
        )

    # 2. Load Data
    # Note: We use st.spinner to give visual feedback during data processing
    with st.spinner('Loading and processing financial data...'):
        stock_data = load_and_process_data(data_source)

    if not stock_data:
        st.warning("Please upload the 'prices.xlsx' file to proceed.")
        return

    # Get data for selected stock
    df = stock_data[selected_stock]

    # 3. Dynamic Layout based on selection
    
    if analysis_mode == "Market Overview":
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        latest_price = df['price'].iloc[-1]
        start_price = df['price'].iloc[0]
        total_return = ((latest_price - start_price) / start_price) * 100
        
        col1.metric("Current Price", f"${latest_price:.2f}")
        col2.metric("Total Return", f"{total_return:.2f}%")
        col3.metric("Observations", f"{len(df)}")
        col4.metric("Data Range", f"{df['date'].dt.year.min()} - {df['date'].dt.year.max()}")

        # Interactive Charts
        st.plotly_chart(plot_interactive_price(df, selected_stock), use_container_width=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(10).style.format({"price": "${:.2f}", "return": "{:.4f}"}))

    elif analysis_mode == "Risk & Returns":
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_returns_distribution(df, selected_stock), use_container_width=True)
        
        with col2:
            # Volatility Cone / Rolling Volatility
            fig_vol = px.line(df, x='date', y='volatility_30d', 
                              title=f'{selected_stock} 30-Day Rolling Volatility (Annualized)')
            fig_vol.update_traces(line_color='red')
            fig_vol.update_layout(template="plotly_white")
            st.plotly_chart(fig_vol, use_container_width=True)
            
        st.info("""
        **Risk Metrics:**
        - **Log Returns:** Used for statistical properties (additivity).
        - **Rolling Volatility:** Standard deviation of returns over a 30-day window, annualized.
        """)

    elif analysis_mode == "GARCH Volatility":
        st.subheader(f"GARCH(1,1) Volatility Modeling: {selected_stock}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Fit Model
            with st.spinner("Fitting GARCH model via Maximum Likelihood..."):
                garch_result = fit_garch_model(df['log_return'])
                
                # Forecast
                forecasts = garch_result.forecast(horizon=30)
                # Extract volatility forecast (sqrt of variance)
                vol_forecast = np.sqrt(forecasts.variance.dropna().values[-1, :])
                
                # Plotting Conditional Volatility
                # The conditional_volatility attr gives the estimated volatility for the historical data
                df['conditional_volatility'] = garch_result.conditional_volatility
                
                fig_garch = go.Figure()
                fig_garch.add_trace(go.Scatter(x=df['date'], y=df['abs_return'] if 'abs_return' in df else np.abs(df['log_return']*100),
                                             mode='lines', name='Absolute Returns', opacity=0.3, line=dict(color='gray')))
                fig_garch.add_trace(go.Scatter(x=df['date'], y=df['conditional_volatility'],
                                             mode='lines', name='Conditional Volatility (GARCH)', line=dict(color='blue')))
                
                fig_garch.update_layout(title="Estimated Conditional Volatility vs Absolute Returns", 
                                      template="plotly_white", yaxis_title="Volatility (%)")
                st.plotly_chart(fig_garch, use_container_width=True)

        with col2:
            st.markdown("### Model Summary")
            st.text(garch_result.summary().as_text())
            
            st.markdown("### Forecast (Next 30 Days)")
            forecast_df = pd.DataFrame({
                "Day": range(1, 31),
                "Forecast Volatility": vol_forecast
            })
            st.dataframe(forecast_df, height=300)

if __name__ == "__main__":
    main()