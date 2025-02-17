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
    return {
        "Ticker": ticker,
        "Current Price": info.get("regularMarketPrice"),
        "High": info.get("dayHigh"),
        "Low": info.get("dayLow"),
        "PE Ratio": info.get("trailingPE"),
        "PB Ratio": info.get("priceToBook")
    }

def fetch_and_analyze_news(ticker):
    """Fetch latest Yahoo Finance news for a ticker and assign risk scores."""
    news_list = news.get_yf_rss(ticker)
    if news_list:
        title = news_list[0]["title"] 
        sentiment = get_sentiment(title)
        return title, sentiment
    return "No recent news", 0.0

def apply_color(val):
    """Apply color from red (-1) to green (1) based on sentiment."""
    color = f"rgba({255 - int(255 * (val + 1) / 2)}, {int(255 * (val + 1) / 2)}, 0, 1)"  # Red to Green scale
    return f'background-color: {color}; color: white'


all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
st.title("📈 Real-Time Stock Dashboard with News Sentiment Analysis")
tickers = st.multiselect("Select Stock Tickers", all_tickers, default=["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"])

cellstyle_jscode = JsCode("""
function(params){
    if (params.value == '0') {
        return {
            'color': 'black', 
            'backgroundColor': 'orange',
        }
    }
    if (params.value < '0') {
        return{
            'color': 'white',
            'backgroundColor': 'red',
        }
    }
    if (params.value > '0') {
        return{
            'color': 'white',
            'backgroundColor': 'green',
        }
    }
}
""")

data = []
for ticker in tickers:
    stock_info = fetch_stock_data(ticker)
    news_title, sentiment = fetch_and_analyze_news(ticker)
    stock_info["News Sentiment"] = sentiment
    stock_info["Latest News"] = news_title
    data.append(stock_info)

df = pd.DataFrame(data)
df_styled = df.style.applymap(lambda x: apply_color(x) if isinstance(x, (int, float)) else '', subset=["News Sentiment"])

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination()
gb.configure_side_bar()
grid_options = gb.build()
AgGrid(df_styled.data, gridOptions=grid_options, fit_columns_on_grid_load=True)

