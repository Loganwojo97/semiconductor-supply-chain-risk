import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
import json

# Configure Plotly to work properly with Streamlit
import plotly.io as pio
pio.renderers.default = "browser"

st.set_page_config(page_title="Semiconductor Risk Dashboard", layout="wide")

st.title("🏭 Semiconductor Supply Chain Risk Dashboard")
st.markdown("Real-time risk analysis of semiconductor companies with ML-driven insights")

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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies Analyzed", len(df))
    with col2:
        avg_risk = df['Financial_Risk_Score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.2f}")
    with col3:
        high_risk_count = len(df[df['Financial_Risk_Score'] > 4])
        st.metric("High Risk Companies", high_risk_count)
    with col4:
        if 'Volatility_30d' in df.columns:
            avg_vol = df['Volatility_30d'].mean()
            st.metric("Average Volatility", f"{avg_vol:.1%}")
    
    # ML Model Insights
    st.header("🤖 ML Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        # Create feature importance based on your actual model results
        feature_data = {
            'Feature': ['Volatility_30d', 'Current_vs_52w_High', 'Price_Change_30d', 'Current_vs_52w_Low', 'price_level', 'Volume_Change_Ratio'],
            'Importance': [0.704, 0.758, 0.660, 0.273, 0.247, 0.157]
        }
        feature_df = pd.DataFrame(feature_data)
        fig_features = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                             title='Most Predictive Risk Factors',
                             color='Importance',
                             color_continuous_scale='viridis')
        st.plotly_chart(fig_features, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Features Used", "6")
            st.metric("Model Type", "Correlation Analysis")
        with col2b:
            st.metric("Risk Std Dev", f"{df['Financial_Risk_Score'].std():.2f}")
            st.metric("Data Quality", "✅ High")
        
        # Top risk companies
        st.subheader("Highest Risk Alert")
        if high_risk_count > 0:
            top_risk = df.nlargest(3, 'Financial_Risk_Score')[['Company', 'Financial_Risk_Score']]
            for _, row in top_risk.iterrows():
                st.warning(f"🚨 {row['Company']}: {row['Financial_Risk_Score']:.2f}")
        else:
            st.success("✅ No companies in critical risk category")
    
    # Risk Score Chart
st.header("🎯 Financial Risk Scores")
fig = px.bar(df.sort_values('Financial_Risk_Score', ascending=False), 
             x='Company', y='Financial_Risk_Score',
             title='Financial Risk Scores by Company',
             color='Financial_Risk_Score',
             color_continuous_scale='RdYlBu_r')
fig.update_layout(
    xaxis_tickangle=-45,
    height=500,
    showlegend=False
)
fig.update_traces(
    hovertemplate='<b>%{x}</b><br>Risk Score: %{y:.2f}<extra></extra>'
)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Feature Importance Chart
with col1:
    st.subheader("Feature Importance")
    feature_data = {
        'Feature': ['Current_vs_52w_High', 'Volatility_30d', 'Price_Change_30d', 'Current_vs_52w_Low', 'price_level', 'Volume_Change_Ratio'],
        'Importance': [0.758, 0.704, 0.660, 0.273, 0.247, 0.157]
    }
    feature_df = pd.DataFrame(feature_data)
    fig_features = px.bar(feature_df, 
                         y='Feature', x='Importance', 
                         orientation='h',
                         title='Most Predictive Risk Factors',
                         color='Importance',
                         color_continuous_scale='viridis')
    fig_features.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_features, use_container_width=True, config={'displayModeBar': False})

# Volatility Analysis
if 'Volatility_30d' in df.columns:
    st.header("📈 Volatility vs Risk Analysis")
    fig2 = px.scatter(df, x='Volatility_30d', y='Financial_Risk_Score',
                     size='Current_Price', hover_name='Company',
                     title='Risk Score vs 30-Day Volatility',
                     labels={'Volatility_30d': 'Volatility (30-day)', 
                            'Financial_Risk_Score': 'Risk Score'})
    fig2.update_layout(height=500)
    fig2.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Volatility: %{x:.1%}<br>Risk Score: %{y:.2f}<extra></extra>'
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    # News sentiment integration (if available)
    st.header("📰 News Sentiment Analysis")
    news_dir = project_root / "data" / "raw" / "news_sentiment"
    news_files = list(news_dir.glob("news_analysis_*.csv")) if news_dir.exists() else []
    
    if news_files:
        latest_news = max(news_files, key=lambda f: f.stat().st_mtime)
        try:
            news_df = pd.read_csv(latest_news)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_sentiment = news_df['sentiment_score'].mean()
                st.metric("Average News Sentiment", f"{avg_sentiment:.2f}")
            with col2:
                avg_risk_signal = news_df['risk_signal'].mean()
                st.metric("News Risk Signal", f"{avg_risk_signal:.2f}")
            with col3:
                st.metric("Articles Analyzed", len(news_df))
            
            # Top risk headlines
            st.subheader("📊 Highest Risk Headlines")
            if 'weighted_risk_signal' in news_df.columns:
                top_risk_news = news_df.nlargest(5, 'weighted_risk_signal')[['title', 'weighted_risk_signal', 'category']]
                st.dataframe(top_risk_news, hide_index=True)
        except Exception as e:
            st.info("News data available but couldn't be processed")
    else:
        st.info("Run news collector to see sentiment analysis: `python src/data_collection/news_collector.py`")
    
    # Data Table
    st.header("📋 Detailed Risk Data")
    st.dataframe(df, use_container_width=True)
    
    # Footer with metadata
    st.markdown("---")
    st.caption(f"Last updated: {latest_file.stat().st_mtime} | Data source: Real-time financial APIs")
    
else:
    st.warning("No risk indicator data found. Please run the financial data collector first.")
    st.code("python src/data_collection/financial_data.py")
    
    st.header("Getting Started")
    st.markdown("""
    1. Run data collection: `python src/data_collection/financial_data.py`
    2. Run news analysis: `python src/data_collection/news_collector.py`  
    3. Train ML model: `python src/models/risk_predictor.py`
    4. Refresh this dashboard
    """)