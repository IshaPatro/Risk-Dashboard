RiskRadar: AI-Powered Stock Insights Dashboard

Overview

QuantumRisk is a real-time stock market dashboard that integrates AI-powered risk analysis to provide insights into market volatility, stock performance, and financial risks. Using live stock data, AI models classify risk levels based on key financial indicators like volatility, beta, Value at Risk (VaR), and Sharpe Ratio. The dashboard provides a clear, color-coded risk assessment to help traders make informed decisions.

Features

Real-time stock market data visualization 📊

AI-driven risk classification (High, Medium, Low) 🔍

Customizable AG-Grid table with filtering & sorting 🔧

Pinned Risk Level column for better visibility 📌

Auto-updating dashboard for live monitoring 🔄

Technology Stack

Frontend: Streamlit, AG-Grid

Backend: Python, Pandas

AI Models: Hugging Face Transformer Models (FinBERT, Flan-T5, etc.)

Data Sources: Yahoo Finance, Alpha Vantage API

Deployment: Streamlit Cloud / AWS / Heroku

Logic & AI Risk Assessment

1️⃣ Data Collection

The dashboard fetches real-time stock data from APIs like Yahoo Finance. The key data points include:

Stock Price 📈

Daily % Change 🔄

Volatility (Historical & Implied) 🌪️

Beta (Market Sensitivity) 📊

Value at Risk (VaR 95%) 📉

Sharpe Ratio (Risk-adjusted Returns) 💰

Latest Financial News 📰

2️⃣ AI Risk Classification

The risk assessment model evaluates the stock based on the following conditions:

High Risk 🔴

Volatility > 20% OR

Beta > 1.5 OR

AI sentiment on news is negative

Medium Risk 🟠

10% < Volatility ≤ 20% OR

1.2 < Beta ≤ 1.5

Low Risk 🟢

Volatility ≤ 10% AND

Beta ≤ 1.2

3️⃣ AI-Powered Risk Classification (FinBERT/Flan-T5 Model)

The AI model takes the latest stock news and financial metrics to predict risk levels:

prompt = f'''
Latest Financial News: {latest_news}
Stock Risk Analysis:
- Volatility: {volatility}%
- Beta: {beta if beta else "Unknown"}
- VaR (95%): {var_95}%
- Sharpe Ratio: {sharpe_ratio}
Classify the risk as High, Medium, or Low.
'''
response = risk_model(prompt)  # AI Model Response

4️⃣ Dynamic AG-Grid Table

Risk Level Column pinned to the right 📌

Conditional formatting applied:

High Risk = Red 🔴

Medium Risk = Orange 🟠

Low Risk = Green 🟢

5️⃣ Live Updates & User Interaction

Auto-refresh every 10-30 seconds for latest data

Interactive AG-Grid with sorting, filtering, and pagination

Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/yourusername/QuantumRisk.git
cd QuantumRisk

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py
