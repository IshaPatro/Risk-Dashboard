import time
import streamlit as st
import pandas as pd
import requests
from st_aggrid import AgGrid, GridOptionsBuilder
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

if not ALPHA_VANTAGE_API_KEY:
    st.error("⚠️ Please set your ALPHA_VANTAGE_API_KEY in your environment variables or Streamlit secrets")
    st.info("Get your free API key from: https://www.alphavantage.co/support/#api-key")
    st.stop()

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

@st.cache_data(ttl=3600)
def get_alpha_vantage_quote(symbol):
    """Get real-time quote from Alpha Vantage GLOBAL_QUOTE endpoint"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" in data:
            return data["Global Quote"]
        elif "Error Message" in data:
            print(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return None
        elif "Note" in data:
            print(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
            return None
        else:
            print(f"Unexpected response format for {symbol}: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request error for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_alpha_vantage_overview(symbol):
    """Get company overview from Alpha Vantage"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Symbol" in data:
            return data
        else:
            print(f"No overview data available for {symbol}")
            return None
            
    except Exception as e:
        print(f"Error fetching overview for {symbol}: {e}")
        return None

def calculate_basic_metrics(current_price, prev_close):
    """Calculate basic metrics from price data"""
    if current_price and prev_close:
        try:
            change_pct = ((current_price - prev_close) / prev_close) * 100
            return round(change_pct, 2)
        except (ValueError, ZeroDivisionError):
            return None
    return None

def fetch_stock_data(ticker):
    """Fetch stock data from Alpha Vantage"""
    quote_data = get_alpha_vantage_quote(ticker)
    
    if not quote_data:
        return None
    
    try:
        current_price = float(quote_data.get("05. price", 0))
        prev_close = float(quote_data.get("08. previous close", 0))
        change_pct = calculate_basic_metrics(current_price, prev_close)
        
        overview_data = get_alpha_vantage_overview(ticker)
        
        pe_ratio = None
        pb_ratio = None
        beta = None
        
        if overview_data:
            try:
                pe_ratio = float(overview_data.get("PERatio", 0)) if overview_data.get("PERatio") != "None" else None
                pb_ratio = float(overview_data.get("PriceToBookRatio", 0)) if overview_data.get("PriceToBookRatio") != "None" else None
                beta = float(overview_data.get("Beta", 0)) if overview_data.get("Beta") != "None" else None
            except (ValueError, TypeError):
                pass
        
        volatility = calculate_volatility_estimate(current_price, prev_close)
        
        return {
            "Ticker": ticker,
            "Current Price ($)": current_price,
            "1-Day Change (%)": change_pct,
            "PE Ratio": round(pe_ratio, 2) if pe_ratio else None,
            "PB Ratio": round(pb_ratio, 2) if pb_ratio else None,
            "Volatility (Est.)": volatility,
            "Beta": beta,
            "Volume": int(quote_data.get("06. volume", 0)),
            "Market Cap": overview_data.get("MarketCapitalization", "N/A") if overview_data else "N/A"
        }
        
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error processing data for {ticker}: {e}")
        return None

def calculate_volatility_estimate(current_price, prev_close):
    """Estimate volatility from daily price change"""
    if current_price and prev_close:
        try:
            daily_return = abs((current_price - prev_close) / prev_close)
            volatility_est = daily_return * np.sqrt(252) * 100
            return round(volatility_est, 2)
        except (ValueError, ZeroDivisionError):
            return 0.0
    return 0.0

@st.cache_data(ttl=7200)
def fetch_news_simple(ticker):
    """Simple news fetching - placeholder for now"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 1,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "feed" in data and len(data["feed"]) > 0:
            return data["feed"][0]["title"]
        else:
            return f"Recent market activity for {ticker}"
            
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return f"Market data available for {ticker}"

def assess_risk_simple(volatility, beta, change_pct, latest_news):
    """Simplified risk assessment"""
    sentiment_score = get_sentiment(latest_news)
    
    risk_level = "Medium"
    
    if sentiment_score < -0.5:
        risk_level = "High"
    elif sentiment_score > 0.3:
        risk_level = "Low"
    
    if volatility and volatility > 25:
        risk_level = "High"
    elif volatility and volatility > 15:
        if risk_level == "Low":
            risk_level = "Medium"
    
    if beta and beta > 1.5:
        risk_level = "High"
    
    if change_pct and abs(change_pct) > 5:
        risk_level = "High"
    
    return risk_level

# Popular stock tickers for selection
popular_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", "CRM",
    "PYPL", "INTC", "AMD", "ORCL", "IBM", "CSCO", "V", "MA", "JPM", "BAC",
    "WMT", "TGT", "COST", "HD", "NKE", "SBUX", "MCD", "KO", "PEP", "JNJ",
    "PFE", "MRNA", "UNH", "CVS", "XOM", "CVX", "NEE", "DIS", "UBER", "LYFT"
]

st.title("RiskRadar: AI-Powered Stock Insights Dashboard")
tickers = st.multiselect(
    "Select Stock Tickers", 
    popular_tickers, 
    default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    help="Choose up to 5 stocks to avoid hitting API limits"
)

if len(tickers) > 5:
    st.error("⛔ Please select maximum 5 stocks to avoid API rate limits (25 requests/day)")
    st.stop()

if not tickers:
    st.warning("Please select at least one stock ticker.")
    st.stop()

progress_bar = st.progress(0)
status_text = st.empty()

data = []
api_calls_used = 0

for i, ticker in enumerate(tickers):
    status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(tickers)})")
    progress_bar.progress((i + 1) / len(tickers))
    
    stock_info = fetch_stock_data(ticker)
    api_calls_used += 2  # Quote + Overview calls
    
    if stock_info:
        news_title = fetch_news_simple(ticker)
        api_calls_used += 1  # News call
        
        stock_info["Latest News"] = news_title
        
        risk_level = assess_risk_simple(
            stock_info.get("Volatility (Est.)"),
            stock_info.get("Beta"),
            stock_info.get("1-Day Change (%)"),
            stock_info["Latest News"]
        )
        stock_info["Risk Level"] = risk_level
        data.append(stock_info)
    else:
        st.warning(f"⚠️ Could not fetch data for {ticker}")
    
    # Small delay between requests
    if i < len(tickers) - 1:
        time.sleep(1)

progress_bar.empty()

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
    gb.configure_column("Risk Level", pinned="right")
    gb.configure_pagination()
    gb.configure_side_bar()
    grid_options = gb.build()
    grid_options["domLayout"] = "autoHeight"
    
    AgGrid(df, gridOptions=grid_options)
    
else:
    st.error("❌ No data could be fetched from Alpha Vantage")
    st.info("💡 Check your API key and ensure you haven't exceeded the 25 requests/day limit")
    st.info("🔗 Get your free API key: https://www.alphavantage.co/support/#api-key")