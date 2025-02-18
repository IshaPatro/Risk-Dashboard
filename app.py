import time
import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_fin import news, stock_info
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

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

def fetch_stock_data(ticker):
    """Fetch stock price details from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    current_price = info.get("regularMarketPrice")
    previous_close = info.get("regularMarketPreviousClose")
    
    if current_price and previous_close:
        one_day_change_pct = ((current_price - previous_close) / previous_close) * 100
        one_day_change_pct = round(one_day_change_pct, 2) 
    else:
        one_day_change_pct = None
    
    return {
        "Ticker": ticker,
        "Current Price": current_price,
        "1-Day Change (%)": one_day_change_pct,
        "PE Ratio": round(info.get("trailingPE"), 2),
        "PB Ratio": round(info.get("priceToBook"), 2)
    }

def fetch_and_analyze_news(ticker):
    """Fetch latest Yahoo Finance news for a ticker and assign risk scores."""
    news_list = news.get_yf_rss(ticker)
    if news_list:
        title = news_list[0]["title"] 
        sentiment = get_sentiment(title)
        return title, sentiment
    return "No recent news", 0.0

cellstyle_jscode = """
function(params) {
    if (params.value < 0) {
        return {
            'color': 'red'
        }
    } else {
        return {
            'color': 'green'
        }
    }
}
"""

all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
st.title("📈 Real-Time Stock Dashboard with News Sentiment Analysis")
tickers = st.multiselect("Select Stock Tickers", all_tickers, default=["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"])

data = []
for ticker in tickers:
    stock_info = fetch_stock_data(ticker)
    news_title, sentiment = fetch_and_analyze_news(ticker)
    stock_info["News Sentiment"] = sentiment
    stock_info["Latest News"] = news_title
    data.append(stock_info)

df = pd.DataFrame(data)

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination()
gb.configure_side_bar()
# gb.configure_column("1-Day Change (%)", cellStyle=cellstyle_jscode)
grid_options = gb.build()
AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=True)

