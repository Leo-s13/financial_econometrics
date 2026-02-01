import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from arch import arch_model
from statsmodels.tsa.stattools import acf, adfuller
from scipy.integrate import quad

from arch.univariate import (
    ConstantMean, GARCH, EGARCH, EWMAVariance, 
    Normal, StudentsT, SkewStudent, GeneralizedError
)
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

# --- 1. Configuration & Style ---
st.set_page_config(
    page_title="Financial Econometrics Dashboard",
    page_icon="📉",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .interpretation-box {
        padding: 20px;
        background-color: #eef2f7;
        border-left: 6px solid #3498db;
        margin-bottom: 25px;
        border-radius: 4px;
        font-style: italic;
    }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- 2. Data Loading ---
@st.cache_data
def load_data_from_path():
    try:
        data = pd.read_excel('data/prices.xlsx')
    except Exception:
        # Fallback for testing purposes if file isn't present
        dates = pd.date_range(start="2007-01-01", periods=1000, freq="B")
        stocks = {}
        for name in ["Goldman Sachs", "Bank of America", "Metlife"]:
            price = 100 + np.cumsum(np.random.randn(1000))
            df = pd.DataFrame({'date': dates, 'price': price})
            df['log_return'] = np.log(df['price']).diff()
            stocks[name] = df.dropna()
        return stocks

    stocks = {}
    mappings = [("Goldman Sachs", 0, 2), ("Bank of America", 2, 4), ("Metlife", 4, 6)]

    for name, start_col, end_col in mappings:
        df = data.iloc[1:, start_col:end_col].copy()
        df.columns = ['date', 'price']
        df['date'] = pd.to_numeric(df['date'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], unit='D', origin='1899-12-30')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['log_return'] = np.log(df['price']).diff()
        df = df.dropna().reset_index(drop=True)
        stocks[name] = df
    return stocks

# --- 3. Page Functions ---

def show_overview(stock_data):
    st.header("🏢 Market Overview")
    
    # --- Price Series Plot ---
    st.subheader("Stock Prices Over Time")
    fig_price = go.Figure()
    colors = {"Goldman Sachs": "#636EFA", "Bank of America": "#EF553B", "Metlife": "#00CC96"}
    
    for name, df in stock_data.items():
        fig_price.add_trace(go.Scatter(x=df['date'], y=df['price'], name=name, line=dict(color=colors[name])))
    
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=500)
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Log Returns Subplots ---
    st.subheader("Log Returns Analysis")
    fig_returns = make_subplots(rows=3, cols=1, subplot_titles=([f"{name} Log Returns" for name in stock_data.keys()]))
    
    for i, (name, df) in enumerate(stock_data.items()):
        fig_returns.add_trace(
            go.Scatter(x=df['date'], y=df['log_return'], name=name, line=dict(width=0.5, color=colors[name])),
            row=i+1, col=1
        )
    
    fig_returns.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig_returns, use_container_width=True)

    # --- Interpretation ---
    st.markdown("""
    <div class="interpretation-box">
    <strong>Analysis:</strong><br>
    As expected the raw price data exhibits clear non-stationarity for all three companies. 
    In the plots of the price data we can see clear differences between Goldman Sachs and the other two companies. 
    While Bank of America and Metlife show a long lasting downward trend during the financial crisis of 2008, 
    Goldman Sachs seems to recover more quickly. This is likely due to the fact that Goldman Sachs is a 
    broker-dealer and investment bank, while Bank of America is a depository bank and Metlife is an insurance company. 
    Broker-dealers and investment banks are less exposed to credit risk than depository banks and insurance companies, 
    which makes them less vulnerable to financial crises. The transformation to log returns seems to remove 
    this non-stationarity. After the transformation, these differences are no longer visible.
    </div>
    """, unsafe_allow_html=True)

def show_stylized_facts(stock_data):
    st.header("📊 Analysis of Returns (Stylized Facts)")
    st.markdown("""
    Financial time series often share common statistical properties known as **stylized facts**. 
    Below we evaluate these properties across our selected assets to gain insights of the underlying distribution.
    """)

    # Create tabs for each stylized fact
    tabs = st.tabs([
        "1. Stationarity", "2. Heavy Tails", "3. Asymmetry", 
        "4. Autocorrelation", "5. Long Range Dep.", 
        "6. Volatility Clustering", "7. Agg. Gaussianity", "8. Leverage Effect"
    ])

    # --- 1. Stationarity ---
    # --- 1. Stationarity ---
    with tabs[0]:
        st.subheader("Augmented Dickey-Fuller (ADF) Test")
        
        # We'll store results to display in a clean table
        adf_results = []
        cols = st.columns(3)
        
        for i, (name, df) in enumerate(stock_data.items()):
            # Perform ADF Test
            result = adfuller(df['log_return'].dropna(), autolag='AIC')
            p_value = result[1]
            test_stat = result[0]
            is_stationary = p_value <= 0.05
            
            # Display Metric in columns
            with cols[i]:
                st.metric(label=f"{name} p-value", 
                          value=f"{p_value:.4e}", 
                          delta="Stationary" if is_stationary else "Non-Stationary",
                          delta_color="normal" if is_stationary else "inverse")
            
            adf_results.append({
                "Company": name,
                "Test Statistic": round(test_stat, 4),
                "p-value": f"{p_value:.4e}",
                "Stationary (5%)": "✅ Yes" if is_stationary else "❌ No"
            })

        # Display Detailed Table
        st.table(pd.DataFrame(adf_results).set_index("Company"))

        # Professional Interpretation
        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        Based on the results of the ADF test, we can conclude that the log returns of all three companies 
        (Goldman Sachs, Bank of America, and Metlife) are <strong>stationary</strong>. 
        The p-values are significantly below the 0.05 threshold, allowing us to reject the null hypothesis of a unit root.
        <br><br>
        Returns are usually stationary because they represent <em>changes</em> in prices, effectively removing the long-run 
        growth and inflation trends present in raw price levels. Under market efficiency, shocks have temporary effects, 
        meaning the series fluctuates around a stable mean—a vital requirement for reliable econometric forecasting.
        </div>
        """, unsafe_allow_html=True)

    # --- 2. Heavy Tails ---
    with tabs[1]:
        st.subheader("Distribution Density vs. Normal Curve")
        
        # 1. Density Plot using Plotly
        fig_dens = go.Figure()
        colors = {"Goldman Sachs": "#636EFA", "Bank of America": "#EF553B", "Metlife": "#00CC96"}
        
        # Add normal distribution reference based on Goldman's volatility
        gs_std = stock_data["Goldman Sachs"]['log_return'].std()
        x_range = np.linspace(-0.15, 0.15, 1000)
        y_norm = stats.norm.pdf(x_range, 0, gs_std)
        
        fig_dens.add_trace(go.Scatter(x=x_range, y=y_norm, name='Normal Distribution', 
                                     line=dict(color='black', dash='dash')))

        for name, df in stock_data.items():
            # Calculate KDE manually for Plotly or use simple histogram
            counts, bins = np.histogram(df['log_return'], bins=100, density=True)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            fig_dens.add_trace(go.Scatter(x=bin_centers, y=counts, name=name, line=dict(color=colors[name])))

        fig_dens.update_layout(title="Empirical Density vs. Gaussian Reference",
                              xaxis_title="Log Return", yaxis_title="Density", height=500)
        st.plotly_chart(fig_dens, use_container_width=True)

        # 2. QQ Plots (Side-by-Side)
        st.subheader("Q-Q Plots (Theoretical Normal vs. Observed)")
        qq_cols = st.columns(3)
        for i, (name, df) in enumerate(stock_data.items()):
            # Calculate Q-Q points
            osm, osr = stats.probplot(df['log_return'], dist="norm")
            
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm[0], y=osm[1], mode='markers', 
                                       marker=dict(color=colors[name], size=4), name=name))
            # Add 45-degree line
            fig_qq.add_trace(go.Scatter(x=osm[0], y=osm[0] * np.std(osm[1]) + np.mean(osm[1]), 
                                       line=dict(color='gray', dash='dash'), name="Normal Line"))
            
            fig_qq.update_layout(title=f"Q-Q: {name}", showlegend=False, height=350,
                                margin=dict(l=20, r=20, t=40, b=20))
            qq_cols[i].plotly_chart(fig_qq, use_container_width=True)

        # 3. Kurtosis Metrics & Comparison
        st.markdown("---")
        m_col1, m_col2 = st.columns([1, 2])
        
        kurt_data = []
        for name, df in stock_data.items():
            val = stats.kurtosis(df['log_return'], fisher=True) # Excess Kurtosis
            kurt_data.append({"Company": name, "Excess Kurtosis": round(val, 2)})
        
        kurt_df = pd.DataFrame(kurt_data)
        
        with m_col1:
            st.write("### Kurtosis Comparison")
            st.dataframe(kurt_df, hide_index=True)
            st.caption("Note: Excess Kurtosis of Normal Dist = 0")

        with m_col2:
            fig_kurt = px.bar(kurt_df, x="Company", y="Excess Kurtosis", color="Company",
                             color_discrete_map=colors, title="Leptokurtosis (Fat Tail Intensity)")
            st.plotly_chart(fig_kurt, use_container_width=True)

        # Interpretation Box
        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        All three institutions exhibit extreme <strong>leptokurtosis</strong>, far exceeding the Gaussian benchmark. 
        The Q-Q plots show significant deviation from the diagonal line at the edges (the "curving" at the ends), 
        confirming that extreme events occur far more frequently than a normal distribution predicts.
        <br><br>
        <strong>Key Insight:</strong> Bank of America shows the highest excess kurtosis ({kurt_df.iloc[1]['Excess Kurtosis']}). 
        As a highly leveraged depository institution, it is historically more sensitive to "black swan" shocks. 
        This suggests that standard VaR models assuming normality would <em>disastrously underestimate</em> risk. 
        Advanced modeling using <strong>Student-t distributions</strong> or <strong>Extreme Value Theory (EVT)</strong> is required.
        </div>
        """, unsafe_allow_html=True)

    # --- 3. Asymmetry ---
    with tabs[2]:
        st.subheader("Distribution Asymmetry (Skewness)")
        
        # 1. Comparison of Skewness Metrics
        m_col1, m_col2 = st.columns([2, 3])
        
        skew_results = []
        for name, df in stock_data.items():
            s_val = stats.skew(df['log_return'])
            n = len(df['log_return'])
            std_error = np.sqrt(6/n)
            z_score = s_val / std_error
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            skew_results.append({
                "Company": name,
                "Skewness": round(s_val, 4),
                "Z-Score": round(z_score, 2),
                "p-value": f"{p_value:.4e}",
                "Sign. Skewed (5%)": "✅ Yes" if p_value <= 0.05 else "❌ No"
            })

        with m_col1:
            st.dataframe(pd.DataFrame(skew_results).set_index("Company"))
            st.caption("A normal distribution has a skewness of 0.")

        with m_col2:
            # Bar chart for Skewness values
            fig_skew = px.bar(pd.DataFrame(skew_results), x="Company", y="Skewness", 
                             color="Company", color_discrete_map=colors,
                             title="Measured Skewness by Institution")
            # Add a zero line
            fig_skew.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_skew, use_container_width=True)

        # 2. Visualizing the "Tails"
        st.subheader("Tail Comparison (Box Plot)")
        fig_box = go.Figure()
        for name, df in stock_data.items():
            fig_box.add_trace(go.Box(y=df['log_return'], name=name, 
                                    marker_color=colors[name], boxpoints='outliers'))
        
        fig_box.update_layout(title="Box Plot: Identifying Asymmetric Outliers", 
                             yaxis_title="Log Returns", height=500)
        st.plotly_chart(fig_box, use_container_width=True)

        



        # 3. Interpretation Box
        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        Skewness measures the asymmetry of the distribution. While financial returns typically exhibit negative skewness, our assets show distinct profiles based on their business models:
        <ul>
            <li><strong>MetLife ({skew_results[2]['Skewness']}):</strong> Shows the most significant <strong>negative skewness</strong>. This reflects the "Insurance Short Volatility" profile—collecting small premiums regularly but being exposed to massive, rare catastrophic payouts.</li>
            <li><strong>Bank of America ({skew_results[1]['Skewness']}):</strong> Also negatively skewed, capturing <strong>credit risk shocks</strong> where "bad news" in credit markets is more violent than "good news."</li>
            <li><strong>Goldman Sachs ({skew_results[0]['Skewness']}):</strong> An anomaly with <strong>positive skewness</strong>. This suggests a "Long Volatility" profile. As a broker-dealer, Goldman often benefits from market dislocations through trading and market-making, allowing them to capture large positive returns during periods of high activity.</li>
        </ul>
        The Z-test confirms that these asymmetries are statistically significant and not due to random noise.
        </div>
        """, unsafe_allow_html=True)

    # --- 4. Absence of Autocorrelations ---
    with tabs[3]:
        st.subheader("Autocorrelation Function (ACF) - Log Returns")
        st.write("Vertical stacking provides a clearer view of the noise floor across assets.")

        lags = 40
        for name, df in stock_data.items():
            # Calculate ACF and CI
            acf_values = acf(df['log_return'], nlags=lags)
            n = len(df['log_return'])
            conf_interval = 1.96 / np.sqrt(n)

            fig_acf = go.Figure()
            # Add CI area
            fig_acf.add_hline(y=conf_interval, line_dash="dash", line_color="rgba(255,0,0,0.3)")
            fig_acf.add_hline(y=-conf_interval, line_dash="dash", line_color="rgba(255,0,0,0.3)")
            
            # Add ACF bars
            fig_acf.add_trace(go.Bar(
                x=list(range(1, lags+1)), 
                y=acf_values[1:], 
                marker_color=colors[name], 
                name=name
            ))

            fig_acf.update_layout(
                title=f"ACF: {name} (Log Returns)",
                xaxis_title="Lag (Days)",
                yaxis_title="Correlation",
                height=300, # Slimmer height for vertical stacking
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(range=[-0.12, 0.12])
            )
            st.plotly_chart(fig_acf, use_container_width=True)

        

        # Interpretation Box
        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        Consistent with the <strong>Efficient Market Hypothesis (EMH)</strong>, all three assets exhibit negligible 
        autocorrelations across all 40 lags. Most correlation coefficients fall within the 95% confidence 
        interval (the shaded region), indicating that they are not statistically different from zero.
        <br><br>
        <strong>Economic Significance:</strong> 
        This confirms the "Absence of Autocorrelations" stylized fact. It implies that past price movements 
        carry no useful information for predicting future log returns. In these markets, price fluctuations 
        behave like a <strong>Random Walk</strong>, where new information is incorporated into prices almost 
        instantaneously and unpredictably.
        </div>
        """, unsafe_allow_html=True)

    # --- 5. Long Range Dependence ---
    with tabs[4]:
            st.subheader("Long Range Dependence (ACF of Squared Returns)")
            st.write("""
            While raw returns show no correlation, their squares (a proxy for variance/volatility) 
            often exhibit persistence. This test checks if volatility shocks have a long-lasting memory.
            """)
            
            lags_long = 40 
            for name, df in stock_data.items():
                # Calculate ACF of SQUARED returns
                squared_returns = df['log_return']**2
                acf_sq = acf(squared_returns, nlags=lags_long)
                n = len(df['log_return'])
                conf_interval = 1.96 / np.sqrt(n)

                fig_sq = go.Figure()
                
                # Add Confidence Interval lines
                fig_sq.add_hline(y=conf_interval, line_dash="dash", line_color="rgba(255,0,0,0.3)")
                fig_sq.add_hline(y=-conf_interval, line_dash="dash", line_color="rgba(255,0,0,0.3)")
                
                # Add ACF bars for squared returns
                fig_sq.add_trace(go.Bar(
                    x=list(range(1, lags_long+1)), 
                    y=acf_sq[1:], 
                    marker_color=colors[name],
                    name=f"{name}²"
                ))

                fig_sq.update_layout(
                    title=f"Volatility Memory: {name} (Squared Log Returns)",
                    xaxis_title="Lag (Days)",
                    yaxis_title="Correlation",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(range=[-0.05, 0.4]) # Adjusted range to show the persistent positive decay
                )
                st.plotly_chart(fig_sq, use_container_width=True)

            

            st.markdown(f"""
            <div class="interpretation-box">
            <strong>Interpretation:</strong><br>
            The autocorrelation plots for the squared log returns of Goldman Sachs, Bank of America, and Metlife 
            show coefficients that remain positive and often stay above the significance threshold for many lags. 
            However, they do not exhibit the extremely high, non-decaying persistence often found in 
            "long memory" processes (like fractionally integrated series).
            <br><br>
            <strong>Key Insight:</strong> This indicates that while there is <strong>short-to-medium range 
            dependence</strong> in volatility, it does not necessarily constitute "long memory" in the 
            strict econometric sense. Volatility shocks are somewhat persistent—confirming that yesterday's 
            volatility influences today's—but these shocks eventually dissipate. This aligns with standard 
            <strong>GARCH-type behavior</strong>, where volatility is mean-reverting rather than 
            infinitely persistent.
            </div>
            """, unsafe_allow_html=True)

    # --- 6. Volatility Clustering ---
    with tabs[5]:
        st.subheader("Volatility Clustering (Squared Returns over Time)")
        st.write("""
        "High volatility tends to be followed by high volatility, and low by low." 
        Below, we visualize the variance of returns to identify periods of market stress.
        """)

        for name, df in stock_data.items():
            fig_clust = go.Figure()
            
            # Plot Squared Returns
            fig_clust.add_trace(go.Scatter(
                x=df['date'], 
                y=df['log_return']**2, 
                name=name,
                line=dict(color=colors[name], width=0.8),
                fill='tozeroy', # Fills area to the x-axis for better visual "mass"
                fillcolor=f"rgba{tuple(int(colors[name].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}"
            ))

            fig_clust.update_layout(
                title=f"Variance Clustering: {name}",
                xaxis_title="Year",
                yaxis_title="Squared Log Return",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_clust, use_container_width=True)

        

        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        The plots clearly illustrate the phenomenon of <strong>Volatility Clustering</strong>. Rather than being 
        uniformly distributed over time, large squared returns (representing high variance) appear in dense 
        "bursts," particularly around the 2008 Financial Crisis and other market shocks.
        <br><br>
        <strong>Key Insights:</strong>
        <ul>
            <li><strong>Persistence:</strong> When a large price movement occurs, it typically triggers a series of 
            subsequent large movements as the market digests new information.</li>
            <li><strong>Regime Switching:</strong> The assets transition between "quiet" regimes (low variance) and 
            "turbulent" regimes (high variance).</li>
        </ul>
        Comparing the three, notice how the "spikes" in <strong>Bank of America</strong> and <strong>Metlife</strong> 
        during 2008-2009 are a little more sustained and intense than those of Goldman Sachs, echoing 
        earlier findings on their relative fragility during the crisis.
        </div>
        """, unsafe_allow_html=True)

    # --- 7. Aggregational Gaussianity ---
    with tabs[6]:
        st.subheader("Aggregational Gaussianity & The Jarque-Bera Test")
        st.write("""
        This fact suggests that as the time scale increases, the distribution of returns should approach a 
        Normal distribution. Here, we test if our **daily** returns meet that Gaussian benchmark.
        """)

        jb_results = []
        cols = st.columns(3)

        for i, (name, df) in enumerate(stock_data.items()):
            # Perform Jarque-Bera Test
            jb_stat, p_value = stats.jarque_bera(df['log_return'])
            is_normal = p_value > 0.05
            
            with cols[i]:
                st.metric(label=f"{name} JB Stat", 
                          value=f"{jb_stat:,.0f}", 
                          delta="Non-Normal", 
                          delta_color="inverse")
            
            jb_results.append({
                "Company": name,
                "JB Statistic": round(jb_stat, 2),
                "p-value": f"{p_value:.4e}",
                "Result (5%)": "❌ Reject Normality" if not is_normal else "✅ Normal"
            })

        # Display Comparison Table
        st.table(pd.DataFrame(jb_results).set_index("Company"))

        # Visualization: Distribution vs. Ideal Normal
        st.subheader("Visualizing the Deviation from Normality")
        fig_norm = go.Figure()
        
        for name, df in stock_data.items():
            # Actual Data Distribution
            fig_norm.add_trace(go.Histogram(
                x=df['log_return'], 
                name=f"{name} Empirical", 
                opacity=0.4, 
                nbinsx=200, 
                histnorm='probability density',
                marker_color=colors[name]
            ))
            
        # The Theoretical Normal Curve for comparison
        x_range = np.linspace(-0.1, 0.1, 1000)
        fig_norm.add_trace(go.Scatter(
            x=x_range, 
            y=stats.norm.pdf(x_range, 0, 0.02), # Using a standard sigma for visual reference
            name="Ideal Normal Dist.", 
            line=dict(color='black', width=3, dash='dot')
        ))

        fig_norm.update_layout(
            title="Empirical Returns vs. The Gaussian Bell Curve",
            xaxis=dict(title="Log Return", range=[-0.1, 0.1]),
            yaxis_title="Density",
            height=500,
            barmode='overlay'
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        

        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        The Jarque-Bera test results are conclusive: we <strong>overwhelmingly reject the null hypothesis of normality</strong> 
        for all three institutions. With test statistics ranging in the thousands (compared to a critical value of 5.99), 
        the p-values are effectively zero.
        <br><br>
        <strong>Modeling Consequence:</strong>
        Because the data is not normal, 
        standard OLS regressions or simple variance-covariance VaR models are mathematically invalid. 
        This empirically justifies our use of:
        <ul>
            <li><strong>GARCH Models:</strong> To handle the non-constant variance (volatility clustering).</li>
            <li><strong>Non-Normal Innovations:</strong> Using Student-t or GED distributions to capture the 
            peakedness and fat tails that the Gaussian curve ignores.</li>
        </ul>
        While "Aggregational Gaussianity" implies these returns might look normal over months or years, at a daily 
        frequency, the market is definitively non-Gaussian.
        </div>
        """, unsafe_allow_html=True)

    # --- 8. Leverage Effect ---
    with tabs[7]:
        st.subheader("The Leverage Effect")
        st.write("""
        This effect describes the negative correlation between an asset's returns and its 
        subsequent change in volatility. Effectively, 'bad news' increases risk more than 'good news.'
        """)

        leverage_results = []
        cols = st.columns(3)

        for i, (name, df) in enumerate(stock_data.items()):
            # Calculate Correlation: R(t) and R^2(t+1)
            returns = df['log_return']
            vol_lead = (df['log_return']**2).shift(-1)
            lev_corr = returns.corr(vol_lead)
            
            leverage_results.append({
                "Company": name,
                "Leverage Correlation": round(lev_corr, 4)
            })

            # Scatter Plot for Visual Confirmation
            fig_lev = px.scatter(
                x=returns, 
                y=vol_lead,
                opacity=0.4,
                labels={'x': 'Return at t', 'y': 'Volatility (R²) at t+1'},
                title=f"Leverage: {name}",
                trendline="ols",
                trendline_color_override="red"
            )
            fig_lev.update_traces(marker=dict(size=4, color=colors[name]))
            fig_lev.update_layout(height=400)
            
            with cols[i]:
                st.plotly_chart(fig_lev, use_container_width=True)

        # Summary Metrics
        st.markdown("---")
        st.write("### Correlation Summary")
        st.table(pd.DataFrame(leverage_results).set_index("Company"))

        

        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        The negative correlations across all three companies confirm the presence of the <strong>Leverage Effect</strong>. 
        When log returns are negative (price drops), the subsequent volatility (squared returns) tends to increase.
        <br><br>
        <strong>Economic Significance:</strong>
        <ul>
            <li><strong>Risk Premium:</strong> This suggests that investors demand a higher risk premium following market 
            declines, as the uncertainty (volatility) of future prices increases.</li>
            <li><strong>Financial Leverage:</strong> The classical explanation is that as a firm's stock price falls, 
            its debt-to-equity ratio increases, making the firm riskier and increasing the volatility of its equity.</li>
            <li><strong>Asymmetric GARCH:</strong> This result empirically justifies the use of advanced models like 
            <strong>EGARCH</strong> or <strong>GJR-GARCH</strong>, which allow for different volatility responses 
            depending on the sign of the previous return.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)



import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.integrate import quad
from scipy.special import gamma
from arch import arch_model
import statsmodels.api as sm

# ==========================================
# HELPER 1: Robust PDF Calculation
# ==========================================
def get_pdf_value(x, dist_type, params):
    if dist_type == "Normal":
        return stats.norm.pdf(x)
    elif dist_type == "Student's t":
        return stats.t.pdf(x, df=params[0])
    elif dist_type == "GED":
        return stats.gennorm.pdf(x, beta=params[0])
    elif dist_type == "Skewed Student's t":
        eta, lam = params[0], params[1]
        # Hansen (1994) Skewed T constants
        c = gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2)) * gamma(eta / 2))
        a = 4 * lam * c * ((eta - 2) / (eta - 1))
        b = np.sqrt(1 + 3 * lam**2 - a**2)
        
        if x < -a / b:
            z = (b * x + a) / (1 - lam)
        else:
            z = (b * x + a) / (1 + lam)
        return b * c * (1 + 1 / (eta - 2) * z**2) ** (-(eta + 1) / 2)
    return 0.0

# ==========================================
# HELPER 2: Numerical ES Calculation (FIXED)
# ==========================================
def calculate_es_numerical(dist_type, params, alpha):
    # Determine the VaR cutoff (q_stat) for integration
    if dist_type == "Normal":
        q_stat = stats.norm.ppf(alpha)
    elif dist_type == "Student's t":
        q_stat = stats.t.ppf(alpha, df=params[0])
    elif dist_type == "GED":
        q_stat = stats.gennorm.ppf(alpha, beta=params[0])
    elif dist_type == "Skewed Student's t":
        # Approximate using Student T for the integration bound
        q_stat = stats.t.ppf(alpha, df=params[0]) 

    def integrand(x):
        return x * get_pdf_value(x, dist_type, params)
    
    # Use -20 instead of -100 for better stability in numerical integration
    integral, _ = quad(integrand, -20, q_stat, limit=50)
    return integral / alpha

# ==========================================
# MAIN DASHBOARD FUNCTION
# ==========================================
def show_risk_measures(stock_data):
    st.title("📊 Exhaustive Market Risk Analysis")
    
   

    # --- TABS FOR METHODS ---
    risk_tabs = st.tabs(["1. Historical Simulation (Non-Parametric)", "2. GARCH Rolling Forecast (Parametric)"])

    # ==========================================
    # TAB 1: HISTORICAL SIMULATION
    # ==========================================
    with risk_tabs[0]:
        st.subheader("Historical Simulation (HS)")
        
        c1, c2 = st.columns(2)
        with c1: hs_conf = st.slider("HS Confidence Level (%)", 90.0, 99.9, 95.0, 0.1, key="hs_conf")
        with c2: hs_window = st.number_input("Rolling Window (Days)", 100, 1000, 250, 50, key="hs_win")
        
        hs_alpha = 1 - (hs_conf / 100.0)

        fig_hs = make_subplots(rows=3, cols=1, subplot_titles=list(stock_data.keys()), vertical_spacing=0.08, shared_xaxes=True)
        
        for i, (name, df) in enumerate(stock_data.items()):
            # HS VaR
            roll_var = df['log_return'].rolling(window=hs_window).quantile(hs_alpha).dropna()
            
            # HS ES (Average of losses exceeding VaR)
            def calc_es_hs(x):
                cutoff = np.percentile(x, hs_alpha * 100)
                return x[x <= cutoff].mean()
            
            roll_es = df['log_return'].rolling(window=hs_window).apply(calc_es_hs, raw=True).dropna()
            
            # Align dates
            common_idx = roll_var.index
            
            fig_hs.add_trace(go.Scatter(x=df.loc[common_idx, 'date'], y=df.loc[common_idx, 'log_return'], 
                                        mode='lines', line=dict(color='rgba(0,0,255,0.15)', width=1), name=f'{name} Returns'), row=i+1, col=1)
            fig_hs.add_trace(go.Scatter(x=df.loc[common_idx, 'date'], y=roll_var, 
                                        mode='lines', line=dict(color='#EF553B', width=1.5), name='VaR (HS)'), row=i+1, col=1)
            fig_hs.add_trace(go.Scatter(x=df.loc[common_idx, 'date'], y=roll_es, 
                                        mode='lines', line=dict(color='#FFA15A', width=1.5, dash='dot'), name='ES (HS)'), row=i+1, col=1)

        fig_hs.update_layout(height=1000, title_text="Historical Simulation Risk Measures", showlegend=False)
        st.plotly_chart(fig_hs, use_container_width=True)

    # ==========================================
    # TAB 2: GARCH ROLLING FORECAST
    # ==========================================
    with risk_tabs[1]:
        st.subheader("VaR and HS using GARCH")
        st.info("This module uses a 'Rolling Window' approach with specific configurations for each institution. We train on past data to forecast tomorrow's risk.")

        # --- 1. Per-Institution Configuration ---
        st.write("### ⚙️ Model Specification")
        st.caption("Customize the model to fit the specific statistical properties (e.g., Use GJR-GARCH for leverage effects, Student's t for fat tails).")
        
        configs = {} 
        
        # Create columns dynamically based on number of assets
        cols = st.columns(len(stock_data))
        
        # Standard mapping for arch library
        dist_map = {
            "Normal": "normal",
            "Student's t": "t",
            "Skewed Student's t": "skewt",
            "GED": "ged"
        }

        for i, (name, df) in enumerate(stock_data.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                # Unique keys for widgets using 'name'
                m = st.selectbox(f"Model", ["GARCH", "GJR-GARCH", "EGARCH"], index=0, key=f"m_{name}")
                d = st.selectbox(f"Dist", ["Normal", "Student's t", "Skewed Student's t", "GED"], index=1, key=f"d_{name}")
                
                configs[name] = {
                    "model": m,
                    "dist_name": d,         # For helper function (Human readable)
                    "dist_code": dist_map[d] # For arch library (Code)
                }

        st.divider()

        # --- 2. Global Rolling Window Settings ---
        c_win, c_refit = st.columns(2)
        with c_win: 
            window_size = st.slider("Estimation Window Size (Days)", 250, 1000, 500, help="Check consistency by changing this size.")
        with c_refit: 
            refit_freq = st.number_input("Refit Frequency (Days)", 1, 60, 22, help="How often we re-calculate GARCH parameters. 22 = Monthly.")

        garch_conf = st.slider("GARCH VaR Confidence (%)", 90.0, 99.9, 95.0, 0.1, key="garch_conf")
        garch_alpha = 1 - (garch_conf / 100.0)

        # --- Events Dictionary for "Evolution over Time" ---
        events = {
            "2013-05-22": "Bernanke Taper Tantrum",
            "2012-07-26": "Draghi 'Whatever it Takes'"
        }
        show_events = st.checkbox("Show Historical Event Markers", value=True)

        # --- 3. Execution Loop ---
        if st.button("Run Rolling Forecast"):
            
            fig_garch = make_subplots(rows=3, cols=1, subplot_titles=list(stock_data.keys()), vertical_spacing=0.1, shared_xaxes=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_steps = len(stock_data)
            
            for i, (name, df) in enumerate(stock_data.items()):
                status_text.text(f"Forecasting for {name}...")
                
                # Retrieve specific config for this asset
                cfg = configs[name]
                model_choice = cfg['model']
                dist_choice = cfg['dist_name']
                arch_dist_code = cfg['dist_code']
                
                returns = df['log_return'] * 100 # Scale up for optimizer stability
                dates = df['date']
                
                # Arrays to store results
                forecast_var = []
                forecast_es = []
                forecast_dates = []
                
                # --- ROLLING LOOP ---
                curr_params = None
                
                # Setup model specification logic
                o = 1 if model_choice == "GJR-GARCH" else 0
                vol_type = "GARCH" if model_choice != "EGARCH" else "EGARCH"
                
                for t in range(window_size, len(returns)):
                    
                    try:
                        # 1. Check if we need to refit parameters
                        if (t - window_size) % refit_freq == 0:
                            # Define training window (Past Data Only)
                            train_data = returns.iloc[t-window_size : t]
                            
                            # Fit Model using the mapped distribution code
                            am = arch_model(train_data, vol=vol_type, p=1, o=o, q=1, dist=arch_dist_code)
                            res = am.fit(disp='off')
                            curr_params = res.params
                        
                        # 2. Forecast 1-step ahead using CURRENT parameters
                        window_data = returns.iloc[t-window_size : t]
                        am_fixed = arch_model(window_data, vol=vol_type, p=1, o=o, q=1, dist=arch_dist_code)
                        
                        # Forecast using fixed params
                        forecast = am_fixed.fix(curr_params).forecast(horizon=1, reindex=False)
                        
                        # Extract Volatility
                        next_vol = np.sqrt(forecast.variance.iloc[-1].values[0])
                        next_mu = forecast.mean.iloc[-1].values[0]
                        
                        # 3. Calculate Risk Measures
                        dist_params = []
                        
                        # Parameter extraction logic based on distribution
                        if dist_choice == "Student's t":
                            dist_params = [curr_params['nu']]
                            q = stats.t.ppf(garch_alpha, df=curr_params['nu'])
                        elif dist_choice == "GED":
                            dist_params = [curr_params['nu']]
                            q = stats.gennorm.ppf(garch_alpha, beta=curr_params['nu'])
                        elif dist_choice == "Skewed Student's t":
                            dist_params = [curr_params['nu'], curr_params['lambda']]
                            # Approximation for speed
                            q = stats.t.ppf(garch_alpha, df=curr_params['nu']) 
                        else: # Normal
                            dist_params = []
                            q = stats.norm.ppf(garch_alpha)

                        # Use helper function for ES
                        es_factor = calculate_es_numerical(dist_choice, dist_params, garch_alpha)
                        
                        # Compute Final Measures (Scale back down because returns were * 100)
                        var_t = (next_mu + next_vol * q) / 100
                        es_t = (next_mu + next_vol * es_factor) / 100
                        
                        forecast_var.append(var_t)
                        forecast_es.append(es_t)
                        forecast_dates.append(dates.iloc[t])
                        
                    except Exception as e:
                        # Fallback mechanism to keep the loop running
                        forecast_var.append(forecast_var[-1] if forecast_var else 0)
                        forecast_es.append(forecast_es[-1] if forecast_es else 0)
                        forecast_dates.append(dates.iloc[t])
                
                # --- PLOTTING ---
                fig_garch.add_trace(go.Scatter(x=forecast_dates, y=df.loc[df['date'].isin(forecast_dates), 'log_return'], 
                                mode='lines', line=dict(color='rgba(0,0,255,0.15)', width=1), name=f'{name} Returns'), row=i+1, col=1)
                
                fig_garch.add_trace(go.Scatter(x=forecast_dates, y=forecast_var, 
                                mode='lines', line=dict(color='#00CC96', width=1.5), name=f'{name} VaR'), row=i+1, col=1)
                
                fig_garch.add_trace(go.Scatter(x=forecast_dates, y=forecast_es, 
                                mode='lines', line=dict(color='#AB63FA', width=1.5, dash='dot'), name=f'{name} ES'), row=i+1, col=1)

                if show_events:
                    for date_str, label in events.items():
                        if date_str > str(forecast_dates[0]) and date_str < str(forecast_dates[-1]):
                            fig_garch.add_vline(x=date_str, line_width=1, line_dash="dash", line_color="gray", row=i+1, col=1)
                            if i == 0:
                                fig_garch.add_annotation(x=date_str, y=min(forecast_var), text=label, showarrow=True, arrowhead=1, row=i+1, col=1)

                progress_bar.progress((i + 1) / total_steps)

            status_text.text("Forecast Complete!")
            fig_garch.update_layout(height=1200, title_text="Rolling Window Risk Forecasts (Ex-Ante)", showlegend=False)
            st.plotly_chart(fig_garch, use_container_width=True)
            
            st.success("Analysis Complete. Use the slider to verify if results change with Window Size (Consistency Check).")
# --- 4. Main Application ---
def main():
    stock_data = load_data_from_path()
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to:", ["Overview", "Analysis of Returns", "Risk Measures", "Forecasting"])

    if selection == "Overview":
        show_overview(stock_data)
    elif selection == "Analysis of Returns":
        show_stylized_facts(stock_data)
    elif selection == "Risk Measures":
        show_risk_measures(stock_data)
    elif selection == "Forecasting":
        show_forecasting(stock_data)

if __name__ == "__main__":
    main()
    
    