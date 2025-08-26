import yfinance as yf
import pandas as pd

print("Testing financial data collection...")

# Simple test with one company
ticker = yf.Ticker("INTC")  # Intel
data = ticker.history(period="1mo")

if not data.empty:
    print(f"Success! Got {len(data)} days of data for Intel")
    print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
    print(f"30-day volatility: {data['Close'].pct_change().std() * 100:.2f}%")
else:
    print("No data received")