import time
import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import news, stock_info
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from yfinance.exceptions import YFRateLimitError
import requests_cache
import os
import requests
import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

load_dotenv()
if "HF_API_KEY" in st.secrets:
    HF_API_KEY = st.secrets["HF_API_KEY"] 
else:
    HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def get_sentiment(text):
    """Classifies news sentiment on a scale of -1 (negative) to 1 (positive)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    sentiment_score = scores[:, 1] - scores[:, 0]  
    return round(sentiment_score.item(), 3) 

def calculate_risk_metrics(historical_data, risk_free_rate=0.0, confidence_level=0.95):
    """Calculate volatility, Sharpe Ratio, and VaR from historical stock data"""
    historical_data['Daily Return'] = historical_data['Close'].pct_change()
    daily_returns = historical_data['Daily Return'].dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    avg_return = daily_returns.mean()
    sharpe_ratio = (avg_return - risk_free_rate) / daily_returns.std()
    var = np.percentile(daily_returns, 100 * (1 - confidence_level))

    return {
        "Volatility": round(volatility * 100, 2),  
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "VaR (95%)": round(var * 100, 2) 
    } 

session = requests_cache.CachedSession("yfinance_cache", expire_after=1800)
def fetch_stock_data(ticker, retries=3, delay=5):
    """Fetch stock price details from Yahoo Finance with error handling and caching."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker, session=session)
            info = stock.info
            if not info:
                print(f"No data found for {ticker}")
                return None
            
            current_price = info.get("regularMarketPrice")
            previous_close = info.get("regularMarketPreviousClose")
            if current_price and previous_close:
                one_day_change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)
            else:
                one_day_change_pct = None
                
            historical_data = stock.history(period="30d")
            risk_metrics = calculate_risk_metrics(historical_data)
    
            return {
                "Ticker": ticker,
                "Current Price": current_price,
                "1-Day Change (%)": one_day_change_pct,
                "PE Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
                "PB Ratio": round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
                "Volatility": risk_metrics["Volatility"],
                "Beta": info.get("beta") if info.get("beta") else None,
                "VaR (95%)": risk_metrics["VaR (95%)"],
                "Sharpe Ratio": risk_metrics["Sharpe Ratio"]
            }
        
        except YFRateLimitError:
            print(f"Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
            time.sleep(delay)
            delay *= 2 
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            time.sleep(delay)        
    return None 

def fetch_news(ticker):
    """Fetch latest Yahoo Finance news for a ticker and assign risk scores."""
    news_list = news.get_yf_rss(ticker)
    if news_list:
        title = news_list[0]["title"] 
        return title
    return "No recent news", 0.0

def assess_risk(volatility, beta, var_95, sharpe_ratio, latest_news):
    """Assess financial risk based on sentiment analysis and stock risk metrics."""

    prompt = f"Analyze the sentiment of this financial news: {latest_news}"
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()
        print("Raw API Response:", response_json)
        
        if not isinstance(response_json, list) or not response_json or not isinstance(response_json[0], list):
            return "API Error"

        scores = {item["label"]: item["score"] for item in response_json[0]}
        negative_score = scores.get("negative", 0)
        positive_score = scores.get("positive", 0)
        
        if negative_score > 0.50:
            risk_level = "High"
        elif negative_score > 0.20:
            risk_level = "Medium"
        elif positive_score > negative_score:
            risk_level = "Low"
        else:
            risk_level = "Medium"

        if volatility > 30 or (beta and beta > 1.5):
            risk_level = "High"
        elif volatility > 10 or (beta and beta > 1.2):
            risk_level = max(risk_level, "Medium")  
        else:
            risk_level = min(risk_level, "Medium")  

    except Exception as e:
        return "API Error"

    return risk_level

all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
st.title("RiskRadar: AI-Powered Stock Insights Dashboard")
tickers = st.multiselect("Select Stock Tickers", all_tickers, default=["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"])

data = []
for ticker in tickers:
    stock_info = fetch_stock_data(ticker)
    news_title = fetch_news(ticker)
    if stock_info:
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








