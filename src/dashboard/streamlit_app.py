import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os

st.title("🏭 Semiconductor Supply Chain Risk Dashboard")
st.markdown("Real-time risk analysis of semiconductor companies")

# Get the project root directory
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data" / "raw" / "financial_data"

# Check if data directory exists
if not data_dir.exists():
    st.error(f"Data directory not found: {data_dir}")
    st.stop()

# Load the most recent risk data
csv_files = list(data_dir.glob("financial_risk_indicators_*.csv"))

if csv_files:
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    st.success(f"Loaded data from: {latest_file.name}")
    
    # Risk Overview Metrics
    st.header("📊 Risk Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Companies Analyzed", len(df))
    with col2:
        avg_risk = df['Financial_Risk_Score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.2f}")
    with col3:
        high_risk_count = len(df[df['Financial_Risk_Score'] > 4])
        st.metric("High Risk Companies", high_risk_count)
    
    # Risk Score Chart
    st.header("🎯 Financial Risk Scores")
    fig = px.bar(df.sort_values('Financial_Risk_Score', ascending=False), 
                 x='Company', y='Financial_Risk_Score',
                 title='Financial Risk Scores by Company',
                 color='Financial_Risk_Score',
                 color_continuous_scale='RdYlBu_r')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility Analysis
    st.header("📈 Volatility Analysis")
    fig2 = px.scatter(df, x='Volatility_30d', y='Financial_Risk_Score',
                     size='Current_Price', hover_name='Company',
                     title='Risk Score vs Volatility')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Data Table
    st.header("📋 Detailed Risk Data")
    st.dataframe(df, use_container_width=True)
    
else:
    st.warning("No risk indicator data found. Please run the financial data collector first.")
    st.code("python src/data_collection/financial_data.py")