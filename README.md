# RiskRadar: AI-Powered Stock Insights Dashboard

## Overview

QuantumRisk is a **real-time stock market dashboard** that integrates **AI-powered risk analysis** to provide insights into market volatility, stock performance, and financial risks. Using **live stock data**, AI models classify risk levels based on key financial indicators like **volatility, beta, Value at Risk (VaR), and Sharpe Ratio**. The dashboard provides a clear, color-coded risk assessment to help traders make informed decisions.

## Features

- **Real-time stock market data visualization** 📊  
- **AI-driven risk classification (High, Medium, Low)** 🔍  
- **Customizable AG-Grid table with filtering & sorting** 🔧  
- **Pinned Risk Level column for better visibility** 📌  
- **Auto-updating dashboard for live monitoring** 🔄  

## Technology Stack

- **Frontend**: Streamlit, AG-Grid  
- **Backend**: Python, Pandas, NumPy  
- **AI Model**: **ProsusAI/FinBERT** (for sentiment-based risk classification)  
- **Data Sources**: Yahoo Finance (for stock prices), Custom Calculations  
- **Deployment**: Streamlit Cloud / AWS / Heroku  

---

## Logic & AI Risk Assessment

### 1️⃣ **Data Collection & Risk Metric Calculation**

The dashboard fetches **real-time stock prices** from Yahoo Finance but calculates key risk metrics internally using **historical stock data**. Specifically, **volatility, Sharpe Ratio, and Value at Risk (VaR) are computed from historical stock returns** rather than being fetched from an external source.

### 2️⃣ **AI Risk Classification using ProsusAI/FinBERT**

The **ProsusAI/FinBERT** model is used to analyze **financial news sentiment**, which plays a crucial role in determining risk levels. If sentiment is negative, the stock is considered **high risk**, whereas neutral or positive sentiment influences the classification accordingly.

#### **Risk Classification Conditions**
- **High Risk** 🔴  
  - Volatility > 30% OR  
  - Beta > 1.5 OR  
  - AI sentiment on news is **negative**  
- **Medium Risk** 🟠  
  - 10% < Volatility ≤ 20% OR  
  - 1.2 < Beta ≤ 1.5  
- **Low Risk** 🟢  
  - Volatility ≤ 10% AND  
  - Beta ≤ 1.2  

### 3️⃣ **Dynamic AG-Grid Table**

- **Risk Level Column** pinned to the right 📌  
- **Conditional formatting** applied:  
  - High Risk = **Red** 🔴  
  - Medium Risk = **Orange** 🟠  
  - Low Risk = **Green** 🟢  

### 4️⃣ **Live Updates & User Interaction**

- Auto-refresh every 10-30 seconds for latest data  
- Interactive AG-Grid with sorting, filtering, and pagination  

---

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
