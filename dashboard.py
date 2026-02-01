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
import warnings
import time

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
    data = pd.read_excel('data/prices.xlsx')
        

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
    st.plotly_chart(fig_price, width='stretch')

    # --- Log Returns Subplots ---
    st.subheader("Log Returns Analysis")
    fig_returns = make_subplots(rows=3, cols=1, subplot_titles=([f"{name} Log Returns" for name in stock_data.keys()]))
    
    for i, (name, df) in enumerate(stock_data.items()):
        fig_returns.add_trace(
            go.Scatter(x=df['date'], y=df['log_return'], name=name, line=dict(width=0.5, color=colors[name])),
            row=i+1, col=1
        )
    
    fig_returns.update_layout(height=700, showlegend=False)
    st.plotly_chart(fig_returns, width='stretch')

    # --- Interpretation ---
    st.markdown("""
    <div class="interpretation-box">
    <strong>Analysis:</strong><br>
    As expected the raw price data visually exhibits clear non-stationarity for all three companies. 
    In the plots of the price data we can see clear differences between Goldman Sachs and the other two companies. 
    While Bank of America and Metlife show a long lasting downward trend following the financial crisis of 2008, 
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

        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        Based on the results of the ADF test, we can conclude that the log returns of all three companies 
        (Goldman Sachs, Bank of America, and Metlife) are <strong>stationary</strong>. 
        The p-values are significantly below the 0.05 threshold, allowing us to reject the null hypothesis of a unit root.
        <br><br>
        Prices are usually non‑stationary, but returns are stationary in second order (constant mean and variance, stable autocovariance function).
        Model implication:
        We will work with returns not prices.
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
        st.plotly_chart(fig_dens, width='stretch')

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
            qq_cols[i].plotly_chart(fig_qq, width='stretch')

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
            st.plotly_chart(fig_kurt, width='stretch')

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
            st.plotly_chart(fig_skew, width='stretch')

        # 2. Visualizing the "Tails"
        st.subheader("Tail Comparison (Box Plot)")
        fig_box = go.Figure()
        for name, df in stock_data.items():
            fig_box.add_trace(go.Box(y=df['log_return'], name=name, 
                                    marker_color=colors[name], boxpoints='outliers'))
        
        fig_box.update_layout(title="Box Plot: Identifying Asymmetric Outliers", 
                             yaxis_title="Log Returns", height=500)
        st.plotly_chart(fig_box, width='stretch')

        


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
        This is the leverage effect: negative returns are followed by larger increases in volatility than positive returns of the same size
        <br><br>
        <strong>Implications for our models:</strong> Plain GARCH(1,1) has a symmetric variance equation. Therefore wit MetLife and Bank of America we should fit
        an asymmetric GARCH model (GJR or EGARCH) with possibly skewed t innovations, not just symmetric GARCH(1,1) with Normal.
        <br>
        Goldman Sachs exhibits significantly positive skewness, unlike the other two institutions. 
        Since plain GARCH with symmetric innovations cannot reproduce this right skewed unconditional distribution, it is best to model GS 
        with an asymmetric GARCH variance equation to capture the leverage effect and a skewed innovation distribution (skewed t) 
        to match the empirical positive skewness.
        </div>
        """, unsafe_allow_html=True)

    # --- 4. Absence of Autocorrelations ---
    with tabs[3]:
        st.subheader("Autocorrelation Function (ACF) - Log Returns")
        st.write("Lags k ≥ 1 at daily frequency.")

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
            st.plotly_chart(fig_acf, width='stretch')

        

        # Interpretation Box
        st.markdown(f"""
        <div class="interpretation-box">
        <strong>Interpretation:</strong><br>
        Consistent with the <strong>Efficient Market Hypothesis</strong>, all three assets exhibit negligible 
        autocorrelations across all 40 lags. Most correlation coefficients fall within the 95% confidence 
        interval (the shaded region), indicating that they are not statistically different from zero.
        <br><br>
        For the models we will later introduce this means we can focus from mean prediction to variance prediction and specify a very simple conditional mean.
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
                    yaxis=dict(range=[-0.05, 0.4])
                )
                st.plotly_chart(fig_sq, width='stretch')

            

            st.markdown(f"""
            <div class="interpretation-box">
            <strong>Interpretation:</strong><br>
            The autocorrelation plots for the squared log returns of Goldman Sachs, Bank of America, and Metlife 
            show coefficients that remain positive and often stay above the significance threshold for many lags. 
            <br><br>
            The persistent positive ACF of squared returns is empirical evidence for the ARCH effect and motivates
            fitting ARCH/GARCH volatility models later on.
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
                fill='tozeroy', 
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
            st.plotly_chart(fig_clust, width='stretch')

        

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
            y=stats.norm.pdf(x_range, 0, 0.02), 
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
        st.plotly_chart(fig_norm, width='stretch')

        

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
        This again empirically justifies our use of GARCH Models to handle the non-constant variance (volatility clustering) 
        and non-Normal innovations to capture the fat tails that the Gaussian curve ignores.</li>
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
                st.plotly_chart(fig_lev, width='stretch')

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
            declines, as the uncertainty of future prices increases.</li>
            <li><strong>Financial Leverage:</strong> One explanation can be, that as a firm's stock price falls, 
            its debt-to-equity ratio increases, making the firm riskier and increasing the volatility of its equity.</li>
        <br>   
        This further leads us towards using asymmetric GARCH (GJR‑GARCH, EGARCH) or skewed innovations in mopdelling.
        This is so our models will be able to account, that negative shocks have a different impact than positive ones.
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
# HELPER 2: Numerical ES Calculation 
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
    
    integral, _ = quad(integrand, -20, q_stat, limit=50)
    return integral / alpha


def get_dist_and_params_from_fit(result, dist_type):
    """
    Map UI dist_type to arch distribution name and extract parameters
    for use with get_pdf_value / calculate_es_numerical.
    Returns (arch_dist_name, params_for_helpers).
    """
    p = result.params 

    if dist_type == "Normal":

        return "normal", [] 

    elif dist_type == "Student's t":

        df = float(p.get("nu", p[-1]))
        return "t", [df]

    elif dist_type == "GED":

        beta = float(p.get("nu", p[-1]))
        return "ged", [beta]

    elif dist_type == "Skewed Student's t":

        eta = float(p.get("nu", p[-2]))
        lam = float(p.get("lambda", p[-1]))
        return "skewt", [eta, lam]

    else:
        # fallback: normal
        return "normal", []


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats
from arch import arch_model
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def compute_diagnostics(model_result):
    """Compute all diagnostic measures for a GARCH model"""
    diag = {}

    # Standardized residuals
    std_resid = model_result.std_resid
    std_resid_sq = std_resid ** 2

    # ACF
    diag['acf_z'] = acf(std_resid, nlags=20, fft=False)
    diag['acf_z2'] = acf(std_resid_sq, nlags=20, fft=False)

    # Ljung-Box tests
    try:
        lb_z = acorr_ljungbox(std_resid, lags=20, return_df=True)
        diag['lb_z_stat'] = lb_z['lb_stat'].iloc[-1]
        diag['lb_z_pval'] = lb_z['lb_pvalue'].iloc[-1]
    except:
        diag['lb_z_stat'] = np.nan
        diag['lb_z_pval'] = np.nan

    try:
        lb_z2 = acorr_ljungbox(std_resid_sq, lags=20, return_df=True)
        diag['lb_z2_stat'] = lb_z2['lb_stat'].iloc[-1]
        diag['lb_z2_pval'] = lb_z2['lb_pvalue'].iloc[-1]
    except:
        diag['lb_z2_stat'] = np.nan
        diag['lb_z2_pval'] = np.nan

    # Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(std_resid)
    diag['jb_stat'] = jb_stat
    diag['jb_pval'] = jb_pval

    # Distribution moments
    diag['skewness'] = stats.skew(std_resid)
    diag['kurtosis'] = stats.kurtosis(std_resid, fisher=True)

    # Model info
    diag['aic'] = model_result.aic
    diag['bic'] = model_result.bic
    diag['loglik'] = model_result.loglikelihood

    # Store raw data for plots
    diag['std_resid'] = std_resid
    diag['std_resid_sq'] = std_resid_sq

    return diag

def plot_model_diagnostics(model_result, model_name, dist_name):
    """Create comprehensive 4-panel diagnostic plot"""

    std_resid = model_result.std_resid
    std_resid_sq = std_resid ** 2

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'ACF: Standardized Residuals (z_t)',
            f'ACF: Squared Std. Residuals (z²_t)',
            f'Q-Q Plot: Normality Check',
            f'Histogram vs. Normal Distribution'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Compute ACF
    acf_z = acf(std_resid, nlags=20, fft=False)
    acf_z2 = acf(std_resid_sq, nlags=20, fft=False)
    n = len(std_resid)
    ci = 1.96 / np.sqrt(n)

    # Plot 1: ACF of z_t
    fig.add_trace(
        go.Bar(x=list(range(1, 21)), y=acf_z[1:21], 
               marker_color='#636EFA', showlegend=False),
        row=1, col=1
    )
    fig.add_hline(y=ci, line_dash="dash", line_color="red", line_width=1, row=1, col=1)
    fig.add_hline(y=-ci, line_dash="dash", line_color="red", line_width=1, row=1, col=1)

    # Plot 2: ACF of z²_t
    fig.add_trace(
        go.Bar(x=list(range(1, 21)), y=acf_z2[1:21], 
               marker_color='#EF553B', showlegend=False),
        row=1, col=2
    )
    fig.add_hline(y=ci, line_dash="dash", line_color="red", line_width=1, row=1, col=2)
    fig.add_hline(y=-ci, line_dash="dash", line_color="red", line_width=1, row=1, col=2)

    # Plot 3: Q-Q plot
    osm, osr = stats.probplot(std_resid, dist="norm")
    fig.add_trace(
        go.Scatter(x=osm[0], y=osm[1], mode='markers', 
                   marker=dict(color='#636EFA', size=3, opacity=0.5),
                   showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=osm[0], y=osm[0], mode='lines',
                   line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )

    # Plot 4: Histogram
    fig.add_trace(
        go.Histogram(x=std_resid, nbinsx=50, histnorm='probability density',
                     marker_color='#636EFA', opacity=0.7, showlegend=False),
        row=2, col=2
    )
    x_range = np.linspace(std_resid.min(), std_resid.max(), 200)
    fig.add_trace(
        go.Scatter(x=x_range, y=stats.norm.pdf(x_range), mode='lines',
                   line=dict(color='red', dash='dash', width=2), showlegend=False),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_xaxes(title_text="Standardized Residual", row=2, col=2)

    fig.update_yaxes(title_text="ACF", row=1, col=1)
    fig.update_yaxes(title_text="ACF", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    fig.update_layout(
        height=700,
        title_text=f"<b>Diagnostics: {model_name} - {dist_name}</b>",
        template='plotly_white'
    )

    return fig

# ==========================================
# MAIN FUNCTION
# ==========================================

def show_risk_measures(stock_data):
    st.title("📊 Market Risk Analysis")

    risk_tabs = st.tabs([
        "1. Historical Simulation", 
        "2. GARCH Model Comparison"
    ])

    # ==========================================
    # TAB 1: HISTORICAL SIMULATION 
    # ==========================================
    with risk_tabs[0]:
        st.subheader("Historical Simulation (HS)")

        st.markdown("""
        **Historical Simulation** calculates risk measures directly from past returns.
        VaR and ES are displayed as positive numbers representing potential losses.
        """)

        c1, c2 = st.columns(2)
        with c1: 
            hs_conf = st.slider("HS Confidence Level (%)", 90.0, 99.9, 95.0, 0.1, key="hs_conf")
        with c2: 
            hs_window = st.number_input("Rolling Window (Days)", 100, 1000, 250, 50, key="hs_win")

        hs_alpha = 1 - (hs_conf / 100.0)

        if st.button("📊 Calculate HS Risk Measures", key="hs_estimate", type="primary"):
            progress_bar = st.progress(0)

            fig_hs = make_subplots(
                rows=3, cols=1, 
                subplot_titles=list(stock_data.keys()), 
                vertical_spacing=0.08, 
                shared_xaxes=True
            )

            n_stocks = len(stock_data)
            for i, (name, df) in enumerate(stock_data.items()):
                progress_bar.progress(int((i / n_stocks) * 100))

                roll_var = -df['log_return'].rolling(window=hs_window).quantile(hs_alpha).dropna()

                def calc_es_hs(x):
                    cutoff = np.percentile(x, hs_alpha * 100)
                    return -x[x <= cutoff].mean()

                roll_es = df['log_return'].rolling(window=hs_window).apply(calc_es_hs, raw=True).dropna()
                common_idx = roll_var.index

                fig_hs.add_trace(
                    go.Scatter(x=df.loc[common_idx, 'date'], y=-df.loc[common_idx, 'log_return'],
                               mode='lines', line=dict(color='rgba(0,0,255,0.15)', width=1), 
                               name=f'{name} Losses'),
                    row=i+1, col=1
                )
                fig_hs.add_trace(
                    go.Scatter(x=df.loc[common_idx, 'date'], y=roll_var, 
                               mode='lines', line=dict(color='#EF553B', width=1.5), name='VaR'),
                    row=i+1, col=1
                )
                fig_hs.add_trace(
                    go.Scatter(x=df.loc[common_idx, 'date'], y=roll_es, 
                               mode='lines', line=dict(color='#FFA15A', width=1.5, dash='dot'), name='ES'),
                    row=i+1, col=1
                )

            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()

            fig_hs.update_layout(height=1000, showlegend=False)
            st.plotly_chart(fig_hs, use_container_width=True)
            st.success("Historical Simulation completed!")

    # ==========================================
    # TAB 2: GARCH MODEL COMPARISON
    # ==========================================
    with risk_tabs[1]:
        st.header("GARCH Model Comparison")
        st.markdown("""
        Usage: Compare the selected GARCH models based on their ability to model the real data.
        """)

        # --- Configuration Section ---
        st.subheader("Configuration")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            series_name = st.selectbox("Select Time Series", list(stock_data.keys()), key="garch_series")
        with col2:
            conf_level = st.slider("VaR Confidence Level (%)", 90.0, 99.9, 95.0, 0.1, key="garch_conf")
        with col3:
            alpha = 1 - conf_level / 100.0
            df = stock_data[series_name]
            returns = df['log_return'].dropna().reset_index(drop=True)
            dates = df['date'].iloc[returns.index].reset_index(drop=True)
            
            # Window selection method
            window_method = st.radio(
                "Estimation Window Method",
                ["Number of Days", "Date Range"],
                horizontal=True,
                key="window_method"
            )
            
            if window_method == "Number of Days":
                estimation_window = st.number_input(
                    "Estimation Window (days)", 
                    min_value=500, 
                    max_value=len(returns)-100, 
                    value=min(1500, len(returns)-100),
                    step=100
                )
                # Use first N observations
                estimation_indices = range(estimation_window)
            else:  # Date Range
                min_date = dates.min()
                max_date = dates.max()
                
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="est_start"
                    )
                with col_date2:
                    end_date = st.date_input(
                        "End Date",
                        value=min_date + pd.Timedelta(days=1000),
                        min_value=min_date,
                        max_value=max_date,
                        key="est_end"
                    )
                
                # Convert to datetime for comparison
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Get indices for date range
                estimation_indices = dates[(dates >= start_date) & (dates <= end_date)].index
                
                if len(estimation_indices) < 100:
                    st.error(f"Selected date range has only {len(estimation_indices)} observations. Need at least 100.")


        st.divider()

        # --- Model & Distribution Selection ---
        col_select_1, col_select_2 = st.columns(2)
        
        with col_select_1:
            st.markdown("**Volatility Models**")
            col_m1, col_m2, col_m3 = st.columns(3)
            models_to_test = []
            with col_m1:
                if st.checkbox("GARCH(1,1)", value=True, key="m_garch"):
                    models_to_test.append(("GARCH(1,1)", {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1}))
            with col_m2:
                if st.checkbox("GJR-GARCH(1,1)", value=True, key="m_gjr"):
                    models_to_test.append(("GJR-GARCH(1,1)", {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1}))
            with col_m3:
                if st.checkbox("EGARCH(1,1)", value=False, key="m_egarch"):
                    models_to_test.append(("EGARCH(1,1)", {'vol': 'EGARCH', 'p': 1, 'o': 0, 'q': 1}))

        with col_select_2:
            st.markdown("**Error Distributions**")
            col_d1, col_d2, col_d3 = st.columns(3)
            dists_to_test = []
            with col_d1:
                if st.checkbox("Normal", value=True, key="d_normal"):
                    dists_to_test.append(("Normal", 'normal'))
            with col_d2:
                if st.checkbox("Student-t", value=True, key="d_t"):
                    dists_to_test.append(("Student-t", 't'))
            with col_d3:
                if st.checkbox("Skewed-t", value=False, key="d_skewt"):
                    dists_to_test.append(("Skewed-t", 'skewt'))

        if len(models_to_test) == 0 or len(dists_to_test) == 0:
            st.warning("Please select at least one model and one distribution.")
            st.stop()

        st.markdown("")
        
        # --- Estimation Action ---
        if st.button("Estimate Selected Models", type="primary", key="estimate_all"):
        
            returns_est = returns.iloc[estimation_indices]  
        
            # Data preprocessing and validation
            if len(returns_est) < 100:
                st.error("Estimation window too small. Need at least 100 observations.")
                st.stop()

            
            # Check for extreme values
            if returns_est.abs().max() > 0.5:
                st.warning("⚠️ Extreme returns detected (>50%). Results may be unreliable.")
            
            # Demean returns for better numerical stability
            returns_mean = returns_est.mean()
            returns_demeaned = returns_est - returns_mean
            
            all_results = {}
            total_models = len(models_to_test) * len(dists_to_test)
            progress_bar = st.progress(0)
            status_text = st.empty()

            idx = 0
            for model_name, model_config in models_to_test:
                for dist_name, dist_config in dists_to_test:
                    idx += 1
                    status_text.text(f"Estimating {model_name} with {dist_name} ({idx}/{total_models})...")
                    progress_bar.progress(int(idx / total_models * 100))

                    try:

                        am = arch_model(
                            returns_demeaned,
                            vol=model_config['vol'],
                            p=model_config['p'],
                            o=model_config['o'],
                            q=model_config['q'],
                            mean='Constant',
                            dist=dist_config,
                            rescale=True 
                        )

                        
                        result = am.fit(
                            disp='off',
                            show_warning=False,
                            options={'maxiter': 1000} 
                        )
                        
                        # Validate estimation results
                        if not np.isfinite(result.loglikelihood):
                            st.warning(f"⚠️ {model_name} with {dist_name}: Invalid log-likelihood")
                            continue
                        
                        # Check parameter bounds
                        params = result.params
                        if 'omega' in params.index and params['omega'] <= 0:
                            st.warning(f"⚠️ {model_name} with {dist_name}: omega ≤ 0, skipping")
                            continue
                        
                        # Compute diagnostics
                        diagnostics = compute_diagnostics(result)
                        
                        # Additional validation of diagnostics
                        if not all(np.isfinite(v) for v in diagnostics.values() if isinstance(v, (int, float))):
                            st.warning(f"⚠️ {model_name} with {dist_name}: Invalid diagnostics")
                            continue

                        key = f"{model_name}_{dist_name}"
                        all_results[key] = {
                            'model': result,
                            'model_name': model_name,
                            'dist_name': dist_name,
                            'config': model_config,
                            'dist_config': dist_config,
                            'diagnostics': diagnostics,
                            'returns_mean': returns_mean  # Store for adjustment
                        }

                    except Exception as e:
                        st.warning(f"Failed to fit {model_name} with {dist_name}: {str(e)}")
                        continue

            progress_bar.empty()
            status_text.empty()

            if len(all_results) == 0:
                st.error("No models successfully estimated. Check your data and settings.")
                st.stop()

            st.success(f"Successfully estimated {len(all_results)} models.")

            # Store in session state
            st.session_state['garch_results'] = all_results
            st.session_state['series_name'] = series_name
            st.session_state['returns'] = returns
            st.session_state['dates'] = dates
            st.session_state['alpha'] = alpha
            st.session_state['conf_level'] = conf_level

        # --- Results View ---
        if 'garch_results' in st.session_state:
            all_results = st.session_state['garch_results']

            st.divider()
            st.subheader("Model Comparison Overview")

            # Create comparison table
            comparison_data = []
            for key, res in all_results.items():
                diag = res['diagnostics']
                comparison_data.append({
                    'Model': res['model_name'],
                    'Distribution': res['dist_name'],
                    'Log-Likelihood': diag['loglik'],
                    'AIC': diag['aic'],
                    'BIC': diag['bic'],
                    'LB(z) p-value': diag['lb_z_pval'],
                    'LB(z²) p-value': diag['lb_z2_pval'],
                    'JB p-value': diag['jb_pval'],
                    'Skewness': diag['skewness'],
                    'Ex. Kurtosis': diag['kurtosis'],
                    'Key': key
                })

            df_comp = pd.DataFrame(comparison_data)
            

            # Style the table
            def color_pvalue(val):
                if isinstance(val, (int, float)):
                    if val > 0.05:
                        return 'background-color: #d4edda; color: #155724;'
                    elif val > 0.01:
                        return 'background-color: #fff3cd; color: #856404;'
                    else:
                        return 'background-color: #f8d7da; color: #721c24;'
                return ''

            styled_df = df_comp[['Model', 'Distribution', 'Log-Likelihood', 'AIC', 'BIC',
                                'LB(z) p-value', 'LB(z²) p-value', 'JB p-value',
                                'Skewness', 'Ex. Kurtosis']].style.applymap(
                color_pvalue,
                subset=['LB(z) p-value', 'LB(z²) p-value', 'JB p-value']
            ).format({
                'Log-Likelihood': '{:.2f}',
                'AIC': '{:.2f}',
                'BIC': '{:.2f}',
                'LB(z) p-value': '{:.4f}',
                'LB(z²) p-value': '{:.4f}',
                'JB p-value': '{:.4f}',
                'Skewness': '{:.4f}',
                'Ex. Kurtosis': '{:.2f}'
            })

            st.dataframe(styled_df, use_container_width=True)

            st.caption("""
            **P-Value Indicators:** Green (p > 0.05): Test Pass | Yellow (0.01 < p ≤ 0.05): Borderline | Red (p ≤ 0.01): Test Fail. 
            """)

            # --- Detailed Examination ---
            st.divider()
            st.subheader("Detailed Model Examination")

            model_keys = list(all_results.keys())
            model_labels = [f"{all_results[k]['model_name']} - {all_results[k]['dist_name']}" 
                        for k in model_keys]

            selected_label = st.selectbox(
                "Select model to examine",
                model_labels,
                key="selected_model"
            )

            selected_key = model_keys[model_labels.index(selected_label)]
            selected_result = all_results[selected_key]

            # Display detailed diagnostics
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown("**Estimated Parameters**")
                params_df = pd.DataFrame({
                    'Parameter': selected_result['model'].params.index,
                    'Coefficient': selected_result['model'].params.values,
                    'Std Error': selected_result['model'].std_err.values,
                    'p-value': selected_result['model'].pvalues.values
                })

                styled_params = params_df.style.format({
                    'Coefficient': '{:.6f}',
                    'Std Error': '{:.6f}',
                    'p-value': '{:.4f}'
                }).applymap(
                    lambda v: 'background-color: #d4edda; color: #155724' if v < 0.05 else 'background-color: #f8d7da; color: #721c24',
                    subset=['p-value']
                )

                st.dataframe(styled_params, use_container_width=True, hide_index=True)

            with col_d2:
                st.markdown("**Diagnostic Test Summary**")
                diag = selected_result['diagnostics']

                test_summary = pd.DataFrame({
                    'Test': [
                        'Ljung-Box (z_t)',
                        'Ljung-Box (z²_t)',
                        'Jarque-Bera (Normality)',
                        'Skewness',
                        'Excess Kurtosis'
                    ],
                    'Statistic': [
                        f"{diag['lb_z_stat']:.2f}",
                        f"{diag['lb_z2_stat']:.2f}",
                        f"{diag['jb_stat']:.2f}",
                        f"{diag['skewness']:.4f}",
                        f"{diag['kurtosis']:.2f}"
                    ],
                    'p-value': [
                        f"{diag['lb_z_pval']:.4f}",
                        f"{diag['lb_z2_pval']:.4f}",
                        f"{diag['jb_pval']:.4f}",
                        '-',
                        '-'
                    ],
                    'Status': [
                        'Pass' if diag['lb_z_pval'] > 0.05 else 'Fail',
                        'Pass' if diag['lb_z2_pval'] > 0.05 else 'Fail',
                        'Good fit' if diag['jb_pval'] > 0.05 else 'Poor fit',
                        'Symmetric' if abs(diag['skewness']) < 0.5 else 'Skewed',
                        'Normal' if abs(diag['kurtosis']) < 3 else 'Fat tails'
                    ]
                })

                st.dataframe(test_summary, use_container_width=True, hide_index=True)
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Log-Likelihood", f"{diag['loglik']:.2f}")
                m2.metric("AIC", f"{diag['aic']:.2f}")
                m3.metric("BIC", f"{diag['bic']:.2f}")

            # Diagnostic plots
            st.markdown("**Diagnostic Plots**")
            fig_diag = plot_model_diagnostics(
                selected_result['model'],
                selected_result['model_name'],
                selected_result['dist_name']
            )
            st.plotly_chart(fig_diag, use_container_width=True)

            with st.expander("Interpretation Guide for Plots"):
                st.markdown("""
                * **Top-left (ACF of z_t):** Should show no significant autocorrelation. Bars within red dashed bands indicate the model captures serial dependence.
                * **Top-right (ACF of z²_t):** Bars within bands indicate volatility clustering is fully captured (no ARCH effects).
                * **Bottom-left (Q-Q Plot):** Points following the diagonal line indicate a good distributional fit. Tails deviating suggest needing Student-t or Skewed-t distributions.
                * **Bottom-right (Histogram):** Comparison of empirical data (blue) vs assumed distribution (red dashed).
                """)

            # --- Forecasting Section ---
            st.divider()
            st.subheader("Rolling Window Forecast")
            st.markdown("""
                        We Calculate VaR and ES based on a rolling window. The model can be selected via the dropdown in the model examination above.
                        """)
            st.info(f"Selected for analysis: **{selected_result['model_name']} with {selected_result['dist_name']}**")
            
            col_f1, col_f2 = st.columns([1, 2])
            
            with col_f1:
                rolling_window = st.number_input(
                    "Rolling Window Size (days)",
                    min_value=200, max_value=1000, value=500, step=50,
                    key="rolling_window"
                )
            
            with col_f2:
                st.write("") 
                st.write("") 
                use_this_model = st.button(
                    f"Set Model & Generate Forecast",
                    type="primary",
                    key="use_model"
                )

            if use_this_model:
                st.session_state['forecast_model'] = selected_result

            if 'forecast_model' in st.session_state and use_this_model:
                forecast_model = st.session_state['forecast_model']
                returns = st.session_state['returns']
                dates = st.session_state['dates']
                alpha = st.session_state['alpha']
                conf_level = st.session_state['conf_level']

                with st.spinner("Generating rolling forecasts..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    n_total = len(returns)
                    total_iterations = n_total - rolling_window
                    forecast_results = []

              
                    scaling_factor = 100
                    returns_scaled = returns * scaling_factor

                    for i, t in enumerate(range(rolling_window, n_total)):
                        if i % 10 == 0:
                            percent_complete = int((i / total_iterations) * 100)
                            progress_bar.progress(percent_complete)
                            status_text.text(f"Processing day {i}/{total_iterations}...")

                        window_returns = returns_scaled.iloc[t-rolling_window:t]

                        try:
                            am = arch_model(
                                window_returns, 
                                vol=forecast_model['config']['vol'],
                                p=forecast_model['config']['p'], 
                                o=forecast_model['config']['o'], 
                                q=forecast_model['config']['q'],
                                mean='Constant', 
                                dist=forecast_model['dist_config'],
                                rescale=False
                            )
                            
                            res = am.fit(disp='off', show_warning=False)

                            fcast = res.forecast(horizon=1, reindex=False)
                            mu_t = fcast.mean.values[-1, 0]
                            sig_t = np.sqrt(fcast.variance.values[-1, 0])

                            if np.isnan(sig_t) or sig_t > (window_returns.std() * 10):
                                continue

                            params = res.params
                            
                            if forecast_model['dist_name'] == 'Student-t':
                                nu = params['nu']
                                q_z = stats.t.ppf(alpha, df=nu)
                                es_z = -(stats.t.pdf(q_z, df=nu) / alpha) * ((nu + q_z**2) / (nu - 1))
                            else: # Default/Normal
                                q_z = stats.norm.ppf(alpha)
                                es_z = -stats.norm.pdf(q_z) / alpha

                            var_t = -(mu_t + sig_t * q_z) / scaling_factor
                            es_t = -(mu_t + sig_t * es_z) / scaling_factor

                            forecast_results.append({
                                'date': dates.iloc[t],
                                'return': returns.iloc[t],
                                'var': var_t,
                                'es': es_t
                            })
                        except:
                            continue

                    progress_bar.progress(100)
                    status_text.empty()

                    if len(forecast_results) == 0:
                        st.error("No forecasts generated.")
                    else:
                        df_forecast = pd.DataFrame(forecast_results)

                        # Plot
                        fig_rolling = go.Figure()

                        fig_rolling.add_trace(go.Scatter(
                            x=df_forecast['date'], y=-df_forecast['return'],
                            mode='lines', line=dict(color='rgba(0,0,255,0.15)', width=1),
                            name='Actual Losses'
                        ))

                        fig_rolling.add_trace(go.Scatter(
                            x=df_forecast['date'], y=df_forecast['var'],
                            mode='lines', line=dict(color='#EF553B', width=2),
                            name=f'VaR ({conf_level:.1f}%)'
                        ))

                        fig_rolling.add_trace(go.Scatter(
                            x=df_forecast['date'], y=df_forecast['es'],
                            mode='lines', line=dict(color='#FFA15A', width=2, dash='dot'),
                            name=f'ES ({conf_level:.1f}%)'
                        ))

                        fig_rolling.update_layout(
                            title=f"Rolling {forecast_model['model_name']} ({forecast_model['dist_name']}) - {st.session_state['series_name']}",
                            xaxis_title="Date", yaxis_title="Loss Magnitude",
                            hovermode='x unified', height=600, template='plotly_white'
                        )

                        st.plotly_chart(fig_rolling, use_container_width=True)
                        st.success("Rolling forecast calculation completed.")










# -----------------------------------
# Helper: ES for standardized residual
# -----------------------------------
def es_standard(dist_type, alpha, params):
    """
    Return (z_var, z_es) for standardized innovation Z at level alpha.
    dist_type: "Normal", "Student's t", "Skewed Student's t", "GED"
    params: dict with needed parameters, e.g. {"nu": ..., "lam": ..., "beta": ...}
    """
    if dist_type == "Normal":
        q = stats.norm.ppf(alpha)
        es = stats.norm.pdf(q) / alpha
        return q, es

    elif dist_type == "Student's t":
        nu = params["nu"]
        q = stats.t.ppf(alpha, df=nu)
        es = (stats.t.pdf(q, df=nu) * (nu + q**2)) / ((nu - 1) * alpha)
        return q, es

    elif dist_type == "GED":
        beta = params["beta"]
        # Approximate ES numerically for GED
        from scipy.integrate import quad
        def pdf(x):
            return stats.gennorm.pdf(x, beta=beta)
        def integrand(x):
            return x * pdf(x)
        q = stats.gennorm.ppf(alpha, beta=beta)
        integral, _ = quad(integrand, -20, q, limit=100)
        es = integral / alpha
        return q, es

    elif dist_type == "Skewed Student's t":

        from scipy.integrate import quad
        nu = params["nu"]
        lam = params["lam"]


        from math import gamma, sqrt, pi
        c = gamma((nu + 1) / 2) / (sqrt(pi * (nu - 2)) * gamma(nu / 2))
        a = 4 * lam * c * ((nu - 2) / (nu - 1))
        b = np.sqrt(1 + 3 * lam**2 - a**2)

        def skewt_pdf(x):
            if x < -a / b:
                z = (b * x + a) / (1 - lam)
            else:
                z = (b * x + a) / (1 + lam)
            return b * c * (1 + z**2 / (nu - 2)) ** (-(nu + 1) / 2)

        # Find quantile q by root finding on CDF
        def cdf(x):
            val, _ = quad(skewt_pdf, -20, x, limit=100)
            return val

        # crude bisection for q
        low, high = -20.0, 10.0
        for _ in range(50):
            mid = 0.5 * (low + high)
            if cdf(mid) < alpha:
                low = mid
            else:
                high = mid
        q = 0.5 * (low + high)

        def integrand(x):
            return x * skewt_pdf(x)
        integral, _ = quad(integrand, -20, q, limit=100)
        es = integral / alpha
        return q, es

    else:
        # fallback: normal
        q = stats.norm.ppf(alpha)
        es = stats.norm.pdf(q) / alpha
        return q, es

def show_forecasting(stock_data):
    # Header with subtitle
    st.title("GARCH-Family Volatility Forecasting")
    st.caption("Forecast future volatility using parameters estimated from historical split.")
    
    # ========================================
    # SECTION 1: Model Configuration
    # ========================================
    with st.container():
        st.markdown("### Configuration")
        st.markdown("""Choose the GARCH Configuration for each stock that was found to be the best in the Risk Measures tab.\\
                 *My suggestions based on the stylized facts and the fitted GARCH models:*\\
                 **Goldman Sachs:** GJR-GARCH(1,1) with Skewed-t\\
                 **Bank of America:** GJR-GARCH(1,1) with Student-t\\
                 **MetLife:** EGARCH(1,1) with Student-t
                 """)
        
        # Series selection
        col_a, col_b = st.columns([2, 1])
        with col_a:
            series_name = st.selectbox(
                "Time series",
                list(stock_data.keys()),
                index=0
            )
        with col_b:
            df = stock_data[series_name][['date', 'log_return']].dropna().reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])
            st.metric("Observations", f"{len(df):,}")
        
        # Model Specs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_type = st.selectbox(
                "Model",
                ["GARCH(1,1)", "GJR-GARCH(1,1)", "EGARCH(1,1)", "IGARCH(1,1)", "GARCH-M(1,1)"],
                index=0
            )
        with col2:
            dist_choice = st.selectbox(
                "Distribution",
                ["Normal", "Student's t", "Skewed Student's t", "GED"],
                index=1
            )
        with col3:
            forecast_horizon = st.number_input(
                "Horizon (days)",
                min_value=1, max_value=60, value=10, step=1
            )
        with col4:
            conf = st.slider("VaR Confidence (%)", 90.0, 99.9, 95.0, 0.1)
        
        # Split controls
        st.markdown("#### Estimation Split")
        st.info("Parameters are estimated on **In-Sample** data only. Then, those parameters are applied to the **Full History** to forecast the true future.")
        
        col_split1, col_split2, col_split3 = st.columns([2, 1, 1])
        with col_split1:
            split_frac = st.slider("In-sample proportion", 0.5, 0.95, 0.8, 0.05)
        
        n = len(df)
        n_est = int(n * split_frac)
        est_data = df.iloc[:n_est].copy()
        
        with col_split2:
            st.metric("In-sample (Training)", f"{n_est:,}")
        with col_split3:
            st.metric("Out-of-sample (Test)", f"{n - n_est:,}")
    
    st.divider()
    
    # ========================================
    # SECTION 2: Estimation & Forecasting
    # ========================================
    alpha = 1 - conf / 100.0
    
    run_forecast = st.button("Estimate & Forecast Future", type="primary", use_container_width=True)
    
    if run_forecast:
        returns_est = est_data['log_return'] 
        returns_all = df['log_return']    
        
        # Map distribution
        dist_map = {
            "Normal": "normal", "Student's t": "t", 
            "Skewed Student's t": "skewt", "GED": "ged"
        }
        arch_dist = dist_map.get(dist_choice, "normal")
        
        # Map model type
        vol = 'GARCH'
        p, o, q = 1, 0, 1
        mean_spec = 'Constant'
        extra_kwargs = {}
        
        if model_type == "GARCH(1,1)":
            p, o, q = 1, 0, 1
        elif model_type == "GJR-GARCH(1,1)":
            p, o, q = 1, 1, 1
        elif model_type == "EGARCH(1,1)":
            vol = 'EGARCH'
        elif model_type == "IGARCH(1,1)":
            extra_kwargs["power"] = 2.0
        elif model_type == "GARCH-M(1,1)":
            mean_spec = 'GARCH-M'
        
        # 1. ESTIMATION PHASE (In-Sample Only)
        with st.spinner("1. Estimating parameters on in-sample split..."):
            am_est = arch_model(
                returns_est, vol=vol, p=p, o=o, q=q, 
                mean=mean_spec, dist=arch_dist, rescale=False, **extra_kwargs
            )
            try:
                res_est = am_est.fit(disp='off')
            except Exception as e:
                st.error(f"Estimation failed: {e}")
                return

        # 2. FILTERING PHASE (Apply params to Full Data)
        with st.spinner("2. Applying parameters to full history for future forecast..."):
            # Create a new model instance with ALL data
            am_all = arch_model(
                returns_all, vol=vol, p=p, o=o, q=q, 
                mean=mean_spec, dist=arch_dist, rescale=False, **extra_kwargs
            )
            # Fix parameters to those learned in step 1
            res_all = am_all.fix(res_est.params)
            
            # Forecast from the very end of the data
            fcast = res_all.forecast(horizon=forecast_horizon, reindex=False)
        
        st.success("Forecast generated successfully!")
        
        # ========================================
        # Parameter Display
        # ========================================
        st.markdown("### Estimated Parameters (from In-Sample)")
        params_df = res_est.params.to_frame("Coefficient")
        params_df['Std Error'] = res_est.std_err
        params_df['t-statistic'] = res_est.tvalues
        params_df['p-value'] = res_est.pvalues
        st.dataframe(params_df.style.format("{:.6f}"))

        # ========================================
        # Process Forecast Paths
        # ========================================
        mu_path = fcast.mean.values[-1, :]
        sig2_path = fcast.variance.values[-1, :]
        sig_path = np.sqrt(sig2_path)
        
        # Generate Future Dates
        last_date = df['date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_horizon, 
            freq='B' # Business days
        )
        
        # ========================================
        # VaR & ES Calculation
        # ========================================
        params = res_est.params
        dist_params = {}
        
        # Robust parameter extraction
        if dist_choice == "Student's t":
            dist_params["nu"] = params.get('nu', params.iloc[-1])
        elif dist_choice == "Skewed Student's t":
            dist_params["nu"] = params.get('nu', params.iloc[-2])
            dist_params["lam"] = params.get('lambda', params.iloc[-1])
        elif dist_choice == "GED":
            dist_params["beta"] = params.get('nu', params.iloc[-1])
            
        z_var, z_es = es_standard(dist_choice, alpha, dist_params)
        
        var_path = np.abs(mu_path + sig_path * z_var)
        es_path = np.abs(mu_path + sig_path * z_es)
        
        # ========================================
        # Visualization
        # ========================================
        st.markdown(f"### Forecast: Next {forecast_horizon} Days")
        
        tab1, tab2, tab3 = st.tabs(["Volatility & Return", "Risk Metrics", "Raw Data"])
        
        with tab1:
            fig = go.Figure()

            subset = df.iloc[-100:]
            fig.add_trace(go.Scatter(x=subset['date'], y=subset['log_return'], 
                                     line=dict(color='gray', width=1), name='Recent History'))
            
            # Forecast Mean
            fig.add_trace(go.Scatter(x=forecast_dates, y=mu_path, 
                                     line=dict(color='blue', width=2), name='Forecast Mean'))
            
            # Forecast Volatility (Confidence Interval style)
            # Upper/Lower bands for volatility visual (mean +/- 2 sigma)
            upper = mu_path + 2*sig_path
            lower = mu_path - 2*sig_path
            
            fig.add_trace(go.Scatter(x=forecast_dates, y=upper, mode='lines', 
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_dates, y=lower, mode='lines', 
                                     line=dict(width=0), fill='tonexty', 
                                     fillcolor='rgba(0,0,255,0.1)', name='2σ Volatility Band'))
            
            fig.update_layout(title="Return Forecast with Volatility Bands", 
                              xaxis_title="Date", yaxis_title="Log Return", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(x=forecast_dates, y=var_path, 
                                          mode='lines+markers', name=f'VaR {conf}%', line=dict(color='red')))
            fig_risk.add_trace(go.Scatter(x=forecast_dates, y=es_path, 
                                          mode='lines+markers', name=f'ES {conf}%', line=dict(color='orange', dash='dash')))
            fig_risk.update_layout(title=f"Risk Forecast (Loss Magnitude)", 
                                   xaxis_title="Date", yaxis_title="Loss", template="plotly_white")
            st.plotly_chart(fig_risk, use_container_width=True)

        with tab3:
            results_df = pd.DataFrame({
                "Date": forecast_dates,
                "Forecast Mean": mu_path,
                "Forecast Vol": sig_path,
                f"VaR ({conf}%)": var_path,
                f"ES ({conf}%)": es_path
            })
            st.dataframe(results_df)
                
    st.markdown("""
                ---
    ## Interpretation and Notes:
---
    ### Evolution of Market Risk Over Time

    All three time series exhibit **time-varying behavior** that reflects changing market conditions:

    - **Crisis Periods**: Elevated volatility during the 2008 financial crisis and 2020 pandemic indicate heightened sensitivity to market movements
    - **Calm Periods**: More stable and lower volatility estimates during normal market conditions
    - **Business Cycles**: The alternating pattern reflects the dynamic nature of financial markets

    ### Impact of Estimation Window Size

    The choice of estimation window affects the results as follows:

    #### Smaller Windows (e.g., 250 observations)
    - Produce more **volatile estimates**
    - **Quickly adapt** to regime changes and recent market conditions
    - Higher responsiveness but less stability

    #### Larger Windows (e.g., 500-1000 observations)
    - Generate **smoother trajectories**
    - **Delayed response** to structural breaks
    - Greater stability but slower adaptation

    This trade-off explains why parameters estimated during one market regime may poorly predict risk in subsequent periods, particularly when transitioning from calm to stress conditions. Therefore doing forecasts on the whole sample with fixed parameters (as above) may not capture evolving risk dynamics effectively.

    ### Comparison Across the three Institutions

    #### During Systemic Crises
    - Systematic risk increases across the board regardless of business models
    - when there is trouble all are affected

    #### During Normal Periods
    - Each institution type exhibits similar baseline volatility levels. A deeper analysis is in the Overview tab

    ### Consistency with Past Events

    The observed patterns align with major historical events:
    - Sharp increases during the 2008 financial crisis
    - Volatility spikes in March 2020 (COVID-19 market turmoil)
    - Gradual stabilization during recovery periods

    ---

    """)









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
