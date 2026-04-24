# Portfolio Risk Stress Testing Dashboard

## Overview

The **Portfolio Risk Stress Testing Dashboard** is a professional-grade quantitative tool designed for retail and institutional traders to perform real-time portfolio analytics, risk management, and Monte Carlo scenario simulation. Migrated fully from an external AI sentiment model, it now strictly utilizes pure mathematical asset pricing properties to build dynamic, interactive risk insights. 

Powered by **Yahoo Finance** pricing data and wrapped in a sleek, dark-mode **Streamlit** user interface, the tool actively processes historical market benchmarks against a user-defined equity portfolio to uncover underlying volatilities and concentration risks.

## Core Features

- **Dynamic Data Ingestion & Caching**: Batched historical fetching mapped onto robust internal state caching (`@st.cache_data`) for extreme efficiency and API safety.
- **Deep Portfolio Metrics Profile**: Computes core returns, annualized volatility, Sharpe, Sortino, Treynor ratios, Information Ratios, skewness, and kurtosis. 
- **Risk Sub-Systems**:
  - **VaR (Value at Risk)**: Daily metric breakdowns using Historical, Parametric, and Conditional Expected Shortfall models.
  - **Scenario Stress Testing**: Market shock scenario simulations factoring in historical relationships with the broader market (`^GSPC`) to formulate specific asset-level Beta sensitivity. 
  - **Correlation & Concentration Tracking**: Interactive asset correlation matrices bound to Herfindahl-Hirschman Index (HHI) measures for concentration risk and rolling day correlation windows.
  - **Animated Monte Carlo Simulation**: Progressive simulated rendering of 10,000 geometric pathways charting one year forward, extracting probability limits (5th/50th/95th percentiles) spanning bear to bull trajectories.

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly (`plotly.express` & `plotly.graph_objects`)
- **Backend Analytics**: Python, Pandas, NumPy, SciPy
- **Data Integration**: `yfinance`

## Installation & Setup

### **Clone the Repository, Create Virtual Environment, and Install Dependencies**

```bash
git clone https://github.com/IshaPatro/Risk-Dashboard.git
cd Risk-Dashboard

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
