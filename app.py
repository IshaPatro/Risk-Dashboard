import time
import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import news, stock_info
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from yfinance.exceptions import YFRateLimitError
import requests_cache
import os
import requests
import streamlit as st

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def get_sentiment(text):
    """Classifies news sentiment on a scale of -1 (negative) to 1 (positive)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    sentiment_score = scores[:, 1] - scores[:, 0]  
    return round(sentiment_score.item(), 3)  

session = requests_cache.CachedSession("yfinance_cache", expire_after=1800)

def fetch_stock_data(ticker, retries=3, delay=5):
    """Fetch stock price details from Yahoo Finance with error handling and caching."""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker, session=session)
            info = stock.info

            if not info:
                os.write(f"No data found for {ticker}")
                return None

            current_price = info.get("regularMarketPrice")
            previous_close = info.get("regularMarketPreviousClose")

            if current_price and previous_close:
                one_day_change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)
            else:
                one_day_change_pct = None

            return {
                "Ticker": ticker,
                "Current Price": current_price,
                "1-Day Change (%)": one_day_change_pct,
                "PE Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
                "PB Ratio": round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
                "Volatility": round(volatility * 100, 2),  
                "Beta": info.get("beta") if info.get("beta") else None,
                "VaR (95%)": round(var_95 * 100, 2), 
                "Sharpe Ratio": round(sharpe_ratio, 2),
            }
        
        except YFRateLimitError:
            os.write(f"Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
            time.sleep(delay)
            delay *= 2 
        
        except Exception as e:
            os.write(f"Error fetching data for {ticker}: {e}")
            time.sleep(delay) 
            
    return None 
def fetch_and_analyze_news(ticker):
    """Fetch latest Yahoo Finance news for a ticker and assign risk scores."""
    news_list = news.get_yf_rss(ticker)
    if news_list:
        title = news_list[0]["title"] 
        sentiment = get_sentiment(title)
        return title, sentiment
    return "No recent news", 0.0

def assess_risk(volatility, beta, var_95, sharpe_ratio, latest_news):
    """Function to assess overall financial risk based on key metrics and latest news using an LLM model."""

    prompt = f"""
    Latest Financial News: {latest_news}

    Stock Risk Analysis:
    - Volatility: {round(volatility, 2)}%
    - Beta: {beta if beta else "Unknown"}
    - VaR (95%): {round(var_95, 2)}%
    - Sharpe Ratio: {round(sharpe_ratio, 2)}

    Based on market trends and stock risk metrics, determine the risk level as High, Medium, or Low.
    Also, provide a reasoning explanation for the risk level.
    """

    model_name = "ProsusAI/finbert"
    risk_model = pipeline("text-classification", model=model_name)
    response = risk_model(prompt, max_length=200, do_sample=True)[0]['generated_text']
    os.write(response)
    
    if "High" in response:
        risk_level = "High"
    elif "Medium" in response:
        risk_level = "Medium"
    elif "Low" in response:
        risk_level = "Low"
    elif volatility > 20 or (beta and beta > 1.5):
        risk_level = "High"
    elif volatility > 10 or (beta and beta > 1.2):
        risk_level = "Medium"
    else:
        risk_level = "Low"
    return risk_level, response

all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
st.title("📈 Real-Time Stock Dashboard with News Sentiment Analysis")
tickers = st.multiselect("Select Stock Tickers", all_tickers, default=["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"])

data = []
for ticker in tickers:
    stock_info = fetch_stock_data(ticker)
    news_title, sentiment = fetch_and_analyze_news(ticker)
    if stock_info:
        stock_info["News Sentiment"] = sentiment
        stock_info["Latest News"] = news_title
        risk_level, risk_reason = assess_risk(
            stock_info["Volatility"],
            stock_info["Beta"],
            stock_info["VaR (95%)"],
            stock_info["Sharpe Ratio"],
            stock_info["Latest News"]
        )
        stock_info["Risk Level"] = risk_level
        stock_info["Risk Reason"] = risk_reason
        data.append(stock_info)

df = pd.DataFrame(data)

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination()
gb.configure_side_bar()
grid_options = gb.build()
AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=True)

