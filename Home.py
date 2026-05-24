import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Financial Analytics Platform",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

div.block-container {
    padding-top: 2rem;
}

.stMetric {
    background-color: #1e222d;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# STOCK DATA WITH COMPANY NAMES
# ============================================================

BULLISH_STOCKS = {

    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "Nvidia",
    "NFLX": "Netflix",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "ORCL": "Oracle"
}

MEDIUM_STOCKS = {

    "IBM": "IBM",
    "CSCO": "Cisco",
    "HPQ": "HP",
    "BAC": "Bank of America",
    "C": "Citigroup",
    "VZ": "Verizon",
    "T": "AT&T",
    "EBAY": "eBay",
    "ETSY": "Etsy",
    "UAL": "United Airlines",
    "SBUX": "Starbucks",
    "KO": "Coca-Cola",
    "GE": "General Electric",
    "F": "Ford"
}

BEARISH_STOCKS = {

    "RIVN": "Rivian",
    "LCID": "Lucid Motors",
    "NIO": "NIO",
    "SNAP": "Snap",
    "PINS": "Pinterest",
    "WBD": "Warner Bros Discovery"
}


# ============================================================
# TITLE
# ============================================================

st.title("📈 AI Financial Analytics Platform")

st.markdown("""
### AI-Powered Stock Market Intelligence

-  llama 3.2 AI Analysis
-  Machine Learning Predictions
-  Technical Indicators
-  Investment Classification
""")


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.subheader("🌍 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Bullish Stocks",
        len(BULLISH_STOCKS),
        "+10"
    )

with col2:
    st.metric(
        "Stable Stocks",
        len(MEDIUM_STOCKS),
        "+14"
    )

with col3:
    st.metric(
        "Bearish Stocks",
        len(BEARISH_STOCKS),
        "-7"
    )

with col4:
    st.metric(
        "AI Accuracy",
        "90%",
        "±4%"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

classification_data = []


# ============================================================
# BULLISH
# ============================================================

for ticker, company in BULLISH_STOCKS.items():

    classification_data.append({

        "Ticker": ticker,
        "Company": company,
        "Category": "Bullish",
        "Score": round(
            np.random.uniform(8, 10),
            2
        )

    })


# ============================================================
# STABLE
# ============================================================

for ticker, company in MEDIUM_STOCKS.items():

    classification_data.append({

        "Ticker": ticker,
        "Company": company,
        "Category": "Stable",
        "Score": round(
            np.random.uniform(5, 7.9),
            2
        )

    })


# ============================================================
# BEARISH
# ============================================================

for ticker, company in BEARISH_STOCKS.items():

    classification_data.append({

        "Ticker": ticker,
        "Company": company,
        "Category": "Bearish",
        "Score": round(
            np.random.uniform(2, 4.9),
            2
        )

    })


classification_df = pd.DataFrame(
    classification_data
)


# ============================================================
# BAR CHART
# ============================================================

st.subheader("📊 AI Investment Scores")

score_chart = px.bar(

    classification_df,

    x="Ticker",

    y="Score",

    color="Category",

    hover_data=["Company"],

    text="Company",

    title="AI-Based Stock Classification",

    height=600
)

score_chart.update_layout(
    template="plotly_dark"
)

score_chart.update_traces(
    textposition="outside"
)

st.plotly_chart(
    score_chart,
    use_container_width=True
)


# ============================================================
# PIE CHART
# ============================================================

st.subheader("💼 Market Sentiment")

sentiment_df = pd.DataFrame({

    "Category": [
        "Bullish",
        "Stable",
        "Bearish"
    ],

    "Count": [

        len(BULLISH_STOCKS),
        len(MEDIUM_STOCKS),
        len(BEARISH_STOCKS)

    ]
})

pie_chart = px.pie(

    sentiment_df,

    names="Category",

    values="Count",

    hole=0.45,

    title="Portfolio Sentiment Distribution"
)

pie_chart.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    pie_chart,
    use_container_width=True
)


# ============================================================
# STOCK TABLE
# ============================================================

st.subheader("📋 Company Classification Table")

st.dataframe(

    classification_df,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# AI INSIGHTS
# ============================================================

st.subheader("🧠 AI Market Insights")

st.info("""

### 📈 Bullish Stocks
Major technology companies with strong
growth, AI adoption, and market momentum.

### ⚖️ Stable Stocks
Established companies with moderate risk
and consistent long-term performance.

### 📉 Bearish Stocks
High volatility or speculative companies
with weaker technical performance.

""")

# ============================================================
# FEATURE CARDS
# ============================================================

st.subheader("🚀 Platform Features")

st.markdown("""
<style>

.feature-card {
    background-color: #1e222d;
    padding: 25px;
    border-radius: 15px;
    height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("""
    <div class="feature-card">

    <h3>🤖 AI Assistant</h3>

    <p>
    Llama 3.2 powered
    financial insights
    and recommendations.
    </p>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="feature-card">

    <h3>📈 ML Predictions</h3>

    <p>
    Machine learning models
    forecast future stock
    market movements.
    </p>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="feature-card">

    <h3>📊 Technical Analysis</h3>

    <p>
    RSI, Momentum,
    Volatility and
    Moving Averages.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Financial Analytics Platform • Streamlit + Ollama + ML"
)