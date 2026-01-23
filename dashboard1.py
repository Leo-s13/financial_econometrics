import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from arch import arch_model
from statsmodels.tsa.stattools import acf

# --- Configuration & Style ---
st.set_page_config(
    page_title="Institutional Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stPlotlyChart { background-color: white; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .fact-box {
        padding: 15px;
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Preprocessing ---

@st.cache_data
def load_data(file):
    """Loads and preprocesses data exactly as per the original notebook."""
    try:
        # Load data depending on source (uploaded or local default)
        data = pd.read_excel(file) if file else pd.read_excel('data/prices.xlsx')
    except:
        return None

    stocks = {}
    
    # Notebook Logic: Manual slicing for GS, BAC, MET based on specific columns
    mappings = [
        ("Goldman Sachs (GS)", 0, 2),
        ("Bank of America (BAC)", 2, 4),
        ("Metlife (MET)", 4, 6)
    ]

    for name, start_col, end_col in mappings:
        df = data.iloc[1:, start_col:end_col].copy()
        df.columns = ['date', 'price']
        
        # Excel date to datetime conversion
        df['date'] = pd.to_numeric(df['date'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], unit='D', origin='1899-12-30')
        
        # Numeric conversion
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Returns Calculation (Log & Simple)
        df['return'] = df['price'].pct_change()
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Drop NA and reset index
        df = df.dropna().sort_values('date').reset_index(drop=True)
        stocks[name] = df

    return stocks

# --- 2. Helper Functions for Analysis ---

def calculate_autocorrelation(series, lags=50):
    """Calculates ACF for plotting."""
    lag_corr = acf(series, nlags=lags)
    return pd.DataFrame({'Lag': range(len(lag_corr)), 'Autocorrelation': lag_corr})

def stylized_facts_check(df, ticker):
    """Computes metrics for the 8 Stylized Facts."""
    returns = df['log_return']
    
    # Fact 1: Non-Gaussianity (Jarque-Bera)
    jb_test = stats.jarque_bera(returns)
    kurtosis = stats.kurtosis(returns)
    skewness = stats.skew(returns)
    
    # Fact 2: Volatility Clustering (ARCH test implied by ACF of squared returns)
    
    return {
        "Kurtosis": kurtosis,
        "Skewness": skewness,
        "JB_P_Value": jb_test.pvalue,
        "Mean": returns.mean(),
        "Std_Dev": returns.std()
    }

# --- 3. Main Application ---

def main():
    # --- Sidebar ---
    st.sidebar.title("Configuration")
    
    uploaded_file = st.sidebar.file_uploader("Upload 'prices.xlsx'", type=['xlsx'])
    # If no file uploaded, try to use a local path (for demo purposes) or warn user
    stock_data = load_data(uploaded_file)

    if not stock_data:
        st.info("👋 Please upload the `prices.xlsx` file to begin the analysis.")
        return

    selected_ticker = st.sidebar.selectbox("Select Asset", list(stock_data.keys()))
    df = stock_data[selected_ticker]

    # --- Navigation ---
    tab1, tab2, tab3 = st.tabs([
        "1. Analysis of Returns (Stylized Facts)", 
        "2. Risk Measures", 
        "3. Forecasting"
    ])

    # ==============================================================================
    # SECTION 1: ANALYSIS OF RETURNS (8 STYLIZED FACTS)
    # ==============================================================================
    with tab1:
        st.header(f"1. Analysis of Returns: {selected_ticker}")
        st.markdown("This section analyzes the *Stylized Facts* of financial returns as outlined in the original research.")

        # --- Visual Overview ---
        col1, col2 = st.columns(2)
        with col1:
            fig_price = px.line(df, x='date', y='price', title=f"{selected_ticker} Price Series")
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            fig_ret = px.line(df, x='date', y='log_return', title=f"{selected_ticker} Log Returns")
            st.plotly_chart(fig_ret, use_container_width=True)

        # --- The Stylized Facts ---
        
        # Fact 1 & 2: Absence of Autocorrelation in Returns / Heavy Tails
        st.subheader("Statistical Properties")
        stats_data = stylized_facts_check(df, selected_ticker)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Kurtosis (Heavy Tails)", f"{stats_data['Kurtosis']:.2f}", delta="Normal = 3.0", delta_color="inverse")
        c2.metric("Skewness", f"{stats_data['Skewness']:.2f}")
        c3.metric("Jarque-Bera p-val", f"{stats_data['JB_P_Value']:.4f}", help="< 0.05 implies non-normal")

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("##### Distribution vs Normal")
            fig_dist = go.Figure()
            # Empirical Distribution
            fig_dist.add_trace(go.Histogram(x=df['log_return'], histnorm='probability density', 
                                          name='Actual Returns', opacity=0.7))
            # Normal Distribution Overlay
            x_range = np.linspace(df['log_return'].min(), df['log_return'].max(), 100)
            pdf = stats.norm.pdf(x_range, stats_data['Mean'], stats_data['Std_Dev'])
            fig_dist.add_trace(go.Scatter(x=x_range, y=pdf, mode='lines', name='Normal Distribution', line=dict(color='red')))
            fig_dist.update_layout(title="Return Distribution (Heavy Tails Check)", hovermode="x unified")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            st.markdown("##### QQ Plot")
            # Create QQ plot data manually for Plotly
            qq_x, qq_y = stats.probplot(df['log_return'], dist="norm")
            fig_qq = px.scatter(x=qq_x[0], y=qq_x[1], labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
            fig_qq.add_shape(type="line", x0=min(qq_x[0]), y0=min(qq_x[0]), x1=max(qq_x[0]), y1=max(qq_x[0]), line=dict(color="red"))
            fig_qq.update_layout(title="Q-Q Plot (Normality Check)")
            st.plotly_chart(fig_qq, use_container_width=True)

        # Fact 3 & 4: Volatility Clustering & Autocorrelation decay
        st.subheader("Volatility Clustering & Autocorrelation")
        st.markdown("""
        **Stylized Fact:** Returns themselves show little autocorrelation, but *absolute* or *squared* returns (volatility) show significant, slowly decaying autocorrelation.
        """)
        
        acf_returns = calculate_autocorrelation(df['log_return'])
        acf_abs_returns = calculate_autocorrelation(np.abs(df['log_return']))
        
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=acf_returns['Lag'], y=acf_returns['Autocorrelation'], name='Log Returns (Raw)'))
        fig_acf.add_trace(go.Bar(x=acf_abs_returns['Lag'], y=acf_abs_returns['Autocorrelation'], name='Absolute Returns (Volatility)', opacity=0.7))
        fig_acf.update_layout(title="Autocorrelation Function (ACF)", xaxis_title="Lag", yaxis_title="Autocorrelation")
        st.plotly_chart(fig_acf, use_container_width=True)

    # ==============================================================================
    # SECTION 2: RISK MEASURES
    # ==============================================================================
    with tab2:
        st.header(f"2. Risk Measures: {selected_ticker}")
        
        # User Inputs for Risk
        col1, col2 = st.columns([1, 3])
        with col1:
            confidence_level = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
            window_size = st.number_input("Rolling Window (Days)", value=252)
            investment_amt = st.number_input("Investment Amount ($)", value=10000)

        # 1. Historical VaR
        var_hist = np.percentile(df['log_return'], (1 - confidence_level) * 100)
        cvar_hist = df[df['log_return'] <= var_hist]['log_return'].mean()
        
        # 2. Parametric VaR (Normal)
        mean_ret = df['log_return'].mean()
        std_ret = df['log_return'].std()
        var_param = stats.norm.ppf(1 - confidence_level, mean_ret, std_ret)
        
        # Display Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Historical VaR ({confidence_level:.0%})", f"{var_hist:.2%}", f"${var_hist * investment_amt:,.2f}")
        c2.metric(f"Parametric VaR ({confidence_level:.0%})", f"{var_param:.2%}", f"${var_param * investment_amt:,.2f}")
        c3.metric(f"CVaR / Expected Shortfall", f"{cvar_hist:.2%}", f"${cvar_hist * investment_amt:,.2f}")
        c4.metric("Max Drawdown", f"{(df['price'] / df['price'].cummax() - 1).min():.2%}")

        # Visualization: Rolling Volatility and Returns
        st.subheader("Rolling Risk Metrics")
        
        # Calculate Rolling Volatility
        df['rolling_vol'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)
        
        fig_vol = px.line(df, x='date', y='rolling_vol', title=f"30-Day Rolling Volatility (Annualized)")
        fig_vol.add_hline(y=df['rolling_vol'].mean(), line_dash="dash", line_color="green", annotation_text="Average Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

        # Visualization: Drawdown
        df['drawdown'] = (df['price'] / df['price'].cummax()) - 1
        fig_dd = px.area(df, x='date', y='drawdown', title="Historical Drawdown")
        fig_dd.update_traces(fillcolor='rgba(255,0,0,0.2)', line_color='red')
        st.plotly_chart(fig_dd, use_container_width=True)

    # ==============================================================================
    # SECTION 3: FORECASTING (GARCH)
    # ==============================================================================
    with tab3:
        st.header(f"3. Volatility Forecasting: {selected_ticker}")
        st.markdown("Implementation of GARCH(1,1) model to forecast future volatility.")

        col_input, col_graph = st.columns([1, 3])
        
        with col_input:
            st.info("The GARCH(1,1) model is fitted to the entire history of log returns to predict conditional volatility.")
            horizon = st.slider("Forecast Horizon (Days)", 1, 30, 7)
            
            # Button to trigger training (as it can be heavy)
            train_btn = st.button("Fit Model & Forecast")

        if train_btn:
            with st.spinner("Fitting GARCH model..."):
                # Scale returns for better optimization stability
                scaled_returns = df['log_return'] * 100 
                
                # Model Specification
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
                res = model.fit(disp='off')
                
                # Forecasting
                forecasts = res.forecast(horizon=horizon)
                var_forecast = forecasts.variance.iloc[-1]
                vol_forecast = np.sqrt(var_forecast)

                # Show Model Summary
                st.text("Model Summary:")
                st.code(res.summary().as_text())

        # Always show historical fit if model exists or using defaults
        # For the interactive look, we fit on load or use cached simple fit for display
        scaled_returns = df['log_return'] * 100 
        model_display = arch_model(scaled_returns, vol='Garch', p=1, q=1)
        res_display = model_display.fit(disp='off')
        
        # Extract conditional volatility
        df['conditional_volatility'] = res_display.conditional_volatility

        # Plot Fitted Volatility vs Returns
        fig_garch = go.Figure()
        
        # Absolute returns as proxy for actual volatility
        fig_garch.add_trace(go.Scatter(
            x=df['date'], 
            y=np.abs(scaled_returns), 
            mode='lines', 
            name='|Returns| (Proxy)', 
            line=dict(color='lightgrey', width=1),
            opacity=0.5
        ))
        
        # Fitted GARCH volatility
        fig_garch.add_trace(go.Scatter(
            x=df['date'], 
            y=df['conditional_volatility'], 
            mode='lines', 
            name='Conditional Volatility (GARCH)', 
            line=dict(color='blue', width=2)
        ))
        
        fig_garch.update_layout(title="GARCH(1,1) In-Sample Fit", yaxis_title="Volatility (%)")
        st.plotly_chart(fig_garch, use_container_width=True)

        # Show forecast if button clicked
        if train_btn:
            st.subheader(f"{horizon}-Day Volatility Forecast")
            forecast_dates = pd.date_range(start=df['date'].iloc[-1], periods=horizon+1)[1:]
            
            fig_fcast = go.Figure()
            fig_fcast.add_trace(go.Scatter(
                x=forecast_dates, 
                y=vol_forecast, 
                mode='lines+markers', 
                name='Forecast Volatility',
                line=dict(color='orange', dash='dot')
            ))
            fig_fcast.update_layout(title="Future Volatility Forecast", xaxis_title="Date", yaxis_title="Predicted Volatility (%)")
            st.plotly_chart(fig_fcast, use_container_width=True)

if __name__ == "__main__":
    main()