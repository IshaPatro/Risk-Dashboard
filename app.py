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
import pickle
from datetime import datetime, timedelta

load_dotenv()

# if "GEMINI_API_KEY" in st.secrets:
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
# else:
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

@st.cache_data(ttl=1800)
def get_cached_stock_data(ticker):
    return fetch_stock_data_internal(ticker)

def fetch_stock_data_internal(ticker):
    max_retries = 2
    base_delay = 10
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            time.sleep(3)
            
            info = stock.info
            if not info or len(info) < 5:
                print(f"Limited data found for {ticker}")
                return None
            
            time.sleep(2)
            
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")
            previous_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
            
            if current_price and previous_close:
                one_day_change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)
            else:
                one_day_change_pct = None
            
            time.sleep(2)
            
            try:
                historical_data = stock.history(period="30d")
                risk_metrics = calculate_risk_metrics(historical_data)
            except Exception as e:
                print(f"Error fetching historical data for {ticker}: {e}")
                risk_metrics = {"Volatility": 0.0, "Sharpe Ratio": 0.0, "VaR (95%)": 0.0}
    
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
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited for {ticker}. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Rate limit exceeded for {ticker} after all retries")
                    return None
            else:
                print(f"Error fetching data for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
    
    return None

def fetch_stock_data(ticker):
    return get_cached_stock_data(ticker)

@st.cache_data(ttl=3600)
def fetch_news(ticker):
    try:
        time.sleep(1)
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

st.warning("⚠️ Yahoo Finance has strict rate limits. Please select only 2-3 stocks to avoid errors. Data is cached for 30 minutes.")

tickers = st.multiselect("Select Stock Tickers (Max 3 recommended)", all_tickers, default=["AAPL", "MSFT", "GOOGL"])

if len(tickers) > 5:
    st.error("⛔ Please select maximum 5 stocks to avoid rate limiting issues.")
    st.stop()

if not tickers:
    st.warning("Please select at least one stock ticker.")
    st.stop()

if len(tickers) > 3:
    st.warning(f"⚠️ You selected {len(tickers)} stocks. This may take longer and could hit rate limits.")

progress_bar = st.progress(0)
status_text = st.empty()

data = []
for i, ticker in enumerate(tickers):    
    status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(tickers)})")
    progress_bar.progress((i + 1) / len(tickers))
    
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
    else:
        st.warning(f"⚠️ Could not fetch data for {ticker} - possibly rate limited")
    
    if i < len(tickers) - 1:
        time.sleep(3)

progress_bar.empty()
status_text.text("✅ Data fetching completed!")

if data:
    st.success(f"Successfully loaded data for {len(data)} out of {len(tickers)} stocks")
    
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
    
    st.info("💡 Data is cached for 30 minutes. Refresh the page to get updated data.")
else:
    st.error("❌ No data could be fetched. Yahoo Finance may be rate limiting. Please wait a few minutes and try with fewer stocks.")
    st.info("💡 Try selecting only 2-3 stocks and wait a few minutes between requests.")