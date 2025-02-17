# Risk Analysis Dashboard
This project is a Risk Analysis Dashboard built with Streamlit. It fetches real-time stock data for selected tickers, performs sentiment analysis on the latest news articles, and displays the results with color-coded risk scores. The app helps users track market sentiment and perform real-time risk assessments based on stock data and news sentiment.

# Features:
Real-Time Stock Data: Fetches real-time stock data including price, high, low, PE ratio, and PB ratio using the Yahoo Finance API.
Sentiment Analysis: Uses the finbert-tone BERT model for sentiment analysis on the latest news articles for selected tickers.
Interactive Table: Displays the stock data and sentiment analysis results in an interactive table with color-coded sentiment indicators (red for negative, green for positive).
Risk Score Analysis: The sentiment score from news headlines is translated into a risk score that helps assess the potential impact on the stock.
