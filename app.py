import time
import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import news, stock_info
from st_aggrid import AgGrid, GridOptionsBuilder
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

def get_sentiment(text):
    if not GEMINI_API_KEY:
        return 0.0
    
    try:
        prompt = f"""
        Analyze the sentiment of this financial news headline and provide a numerical score:
        
        News: "{text}"
        
        Please respond with only a number between -1 and 1, where:
        -1 = very negative
        0 = neutral
        1 = very positive
        
        Score:
        """
        
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        
        try:
            score = float(score_text)
            return max(-1, min(1, score))
        except ValueError:
            return 0.0
            
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0.0

def calculate_risk_metrics(historical_data, risk_free_rate=0.0, confidence_level=0.95):
    if historical_data is None or len(historical_data) < 2:
        return {"Volatility": 0.0, "Sharpe Ratio": 0.0, "VaR (95%)": 0.0}
    
    historical_data['Daily Return'] = historical_data['Close'].pct_change()
    daily_returns = historical_data['Daily Return'].dropna()
    
    if len(daily_returns) < 2:
        return {"Volatility": 0.0, "Sharpe Ratio": 0.0, "VaR (95%)": 0.0}
    
    volatility = daily_returns.std() * np.sqrt(252)
    avg_return = daily_returns.mean()
    sharpe_ratio = (avg_return - risk_free_rate) / daily_returns.std() if daily_returns.std() > 0 else 0
    var = np.percentile(daily_returns, 100 * (1 - confidence_level))

    return {
        "Volatility": round(volatility * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "VaR (95%)": round(var * 100, 2)
    }

def fetch_stock_data(ticker, retries=3, delay=5):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                print(f"No data found for {ticker}")
                return None
            
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")
            previous_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
            
            if current_price and previous_close:
                one_day_change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)
            else:
                one_day_change_pct = None
                
            historical_data = stock.history(period="30d")
            risk_metrics = calculate_risk_metrics(historical_data)
    
            return {
                "Ticker": ticker,
                "Current Price ($)": current_price,
                "1-Day Change (%)": one_day_change_pct,
                "PE Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
                "PB Ratio": round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
                "Volatility": risk_metrics["Volatility"],
                "Beta": info.get("beta") if info.get("beta") else None,
                "VaR (95%)": risk_metrics["VaR (95%)"],
                "Sharpe Ratio": risk_metrics["Sharpe Ratio"]
            }
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                return None
    
    return None

def fetch_news(ticker):
    try:
        news_list = news.get_yf_rss(ticker)
        if news_list:
            return news_list[0]["title"]
        return "No recent news available"
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return "No recent news available"

def assess_risk(volatility, beta, var_95, sharpe_ratio, latest_news):
    sentiment_score = get_sentiment(latest_news)
    
    if sentiment_score < -0.50:
        risk_level = "High"
    elif sentiment_score < -0.20:
        risk_level = "Medium"
    elif sentiment_score > 0.20:
        risk_level = "Low"
    else:
        risk_level = "Medium"

    if volatility > 30 or (beta and beta > 1.5):
        risk_level = "High"
    elif volatility > 10 or (beta and beta > 1.2):
        if risk_level == "Low":
            risk_level = "Medium"
    
    return risk_level

try:
    all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
except Exception as e:
    print(f"Error fetching S&P 500 list: {e}")
    all_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B", "UNH", "JNJ",
        "V", "PG", "JPM", "HD", "MA", "NFLX", "DIS", "ADBE", "PYPL", "CMCSA",
        "VZ", "INTC", "T", "PFE", "WMT", "KO", "PEP", "ABT", "CRM", "CSCO",
        "XOM", "TMO", "COST", "AVGO", "ACN", "DHR", "LLY", "TXN", "NEE", "WFC",
        "QCOM", "BMY", "MDT", "UNP", "PM", "LOW", "IBM", "AMGN", "HON", "SPGI"
    ]

st.title("RiskRadar: AI-Powered Stock Insights Dashboard")
tickers = st.multiselect("Select Stock Tickers", all_tickers, default=["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"])

data = []
for i, ticker in enumerate(tickers):    
    stock_info = fetch_stock_data(ticker)
    if stock_info:
        news_title = fetch_news(ticker)
        stock_info["Latest News"] = news_title      
        risk_level = assess_risk(
            stock_info["Volatility"],
            stock_info["Beta"],
            stock_info["VaR (95%)"],
            stock_info["Sharpe Ratio"],
            stock_info["Latest News"]
        )
        stock_info["Risk"] = risk_level
        data.append(stock_info)

if data:
    df = pd.DataFrame(data)
    
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        minWidth=50
    )
    gb.configure_column("Latest News", minWidth=200)
    gb.configure_column("Ticker", pinned="left")
    gb.configure_column("Risk", pinned="right")
    gb.configure_pagination()
    gb.configure_side_bar()
    grid_options = gb.build()
    grid_options["domLayout"] = "autoHeight"
    
    AgGrid(df, gridOptions=grid_options)
else:
    st.error("No data could be fetched. Please check your internet connection and try again.")