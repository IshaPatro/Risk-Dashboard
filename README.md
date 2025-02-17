#Risk Analysis Dashboard
This project is a Risk Analysis Dashboard built with Streamlit. It fetches real-time stock data for selected tickers, performs sentiment analysis on the latest news articles, and displays the results with color-coded risk scores. The app helps users track market sentiment and perform real-time risk assessments based on stock data and news sentiment.

Features:
Real-Time Stock Data: Fetches real-time stock data including price, high, low, PE ratio, and PB ratio using the Yahoo Finance API.
Sentiment Analysis: Uses the finbert-tone BERT model for sentiment analysis on the latest news articles for selected tickers.
Interactive Table: Displays the stock data and sentiment analysis results in an interactive table with color-coded sentiment indicators (red for negative, green for positive).
Risk Score Analysis: The sentiment score from news headlines is translated into a risk score that helps assess the potential impact on the stock.
Files:
app.py: The main Streamlit app that powers the dashboard, including fetching stock data, performing sentiment analysis, and displaying results in a table.
requirements.txt: A file listing all dependencies required to run the app.
Installation
Follow the steps below to set up and run the app locally:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/risk-analysis-dashboard.git
cd risk-analysis-dashboard
Create and activate a virtual environment: (If you're using Python 3, you can create and activate a virtual environment as follows)

bash
Copy
Edit
python3 -m venv .env
source .env/bin/activate   # For macOS/Linux
.\.env\Scripts\activate    # For Windows
Install dependencies: Use pip to install all required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app: Start the app with the following command:

bash
Copy
Edit
streamlit run app.py
Access the app: After running the above command, open your browser and go to http://localhost:8501 to view the dashboard.

