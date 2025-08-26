"""
Financial data collection for semiconductor companies.
Collects stock prices, volatility, and financial metrics.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.settings import SEMICONDUCTOR_COMPANIES, RANDOM_STATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """Collects financial data for semiconductor companies."""
    
    def __init__(self, companies: Dict[str, str] = None):
        """Initialize with company symbols."""
        self.companies = companies or SEMICONDUCTOR_COMPANIES
        self.data_dir = Path("data/raw/financial_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_stock_data(self, 
                          period: str = "2y", 
                          save_to_file: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect stock price data for semiconductor companies.
        
        Args:
            period: Time period to collect data for ('1y', '2y', '5y', etc.)
            save_to_file: Whether to save data to CSV files
            
        Returns:
            Dictionary mapping company names to stock data DataFrames
        """
        logger.info(f"Collecting stock data for {len(self.companies)} companies...")
        
        stock_data = {}
        failed_companies = []
        
        for company_name, symbol in self.companies.items():
            try:
                logger.info(f"Fetching data for {company_name} ({symbol})")
                
                # Download stock data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    logger.warning(f"No data found for {company_name} ({symbol})")
                    failed_companies.append(company_name)
                    continue
                
                # Add company info
                data['Company'] = company_name
                data['Symbol'] = symbol
                data['Date'] = data.index
                
                stock_data[company_name] = data
                
                # Save to file if requested
                if save_to_file:
                    filename = f"{company_name.replace(' ', '_').lower()}_stock_data.csv"
                    filepath = self.data_dir / filename
                    data.to_csv(filepath)
                    logger.info(f"Saved data for {company_name} to {filepath}")
                    
            except Exception as e:
                logger.error(f"Failed to collect data for {company_name}: {str(e)}")
                failed_companies.append(company_name)
        
        logger.info(f"Successfully collected data for {len(stock_data)} companies")
        if failed_companies:
            logger.warning(f"Failed to collect data for: {', '.join(failed_companies)}")
            
        return stock_data
    
    def calculate_risk_indicators(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate financial risk indicators for each company.
        
        Args:
            stock_data: Dictionary of stock data DataFrames
            
        Returns:
            DataFrame with risk indicators for each company
        """
        logger.info("Calculating financial risk indicators...")
        
        risk_indicators = []
        
        for company_name, data in stock_data.items():
            try:
                # Calculate returns
                data['Daily_Return'] = data['Close'].pct_change()
                
                # Calculate volatility (30-day rolling)
                data['Volatility_30d'] = data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
                
                # Calculate recent performance metrics
                recent_data = data.tail(30)  # Last 30 days
                
                indicators = {
                    'Company': company_name,
                    'Symbol': data['Symbol'].iloc[0],
                    'Date': data.index[-1],
                    'Current_Price': data['Close'].iloc[-1],
                    'Price_Change_30d': ((data['Close'].iloc[-1] / data['Close'].iloc[-30]) - 1) * 100,
                    'Volatility_30d': data['Volatility_30d'].iloc[-1],
                    'Avg_Volume_30d': recent_data['Volume'].mean(),
                    'Volume_Change_Ratio': recent_data['Volume'].iloc[-1] / recent_data['Volume'].mean(),
                    'High_52w': data['High'].max(),
                    'Low_52w': data['Low'].min(),
                    'Current_vs_52w_High': (data['Close'].iloc[-1] / data['High'].max()) * 100,
                    'Current_vs_52w_Low': (data['Close'].iloc[-1] / data['Low'].min()) * 100,
                }
                
                risk_indicators.append(indicators)
                
            except Exception as e:
                logger.error(f"Failed to calculate indicators for {company_name}: {str(e)}")
        
        risk_df = pd.DataFrame(risk_indicators)
        
        # Add risk scores (simple scoring for now)
        if not risk_df.empty:
            # Higher volatility = higher risk
            risk_df['Volatility_Score'] = pd.qcut(risk_df['Volatility_30d'], 
                                                q=5, labels=[1,2,3,4,5], duplicates='drop')
            
            # Recent negative performance = higher risk
            risk_df['Performance_Score'] = pd.qcut(-risk_df['Price_Change_30d'], 
                                                 q=5, labels=[1,2,3,4,5], duplicates='drop')
            
            # Composite risk score
            risk_df['Financial_Risk_Score'] = (
                pd.to_numeric(risk_df['Volatility_Score'], errors='coerce') * 0.6 + 
                pd.to_numeric(risk_df['Performance_Score'], errors='coerce') * 0.4
            )
        
        return risk_df
    
    def save_risk_indicators(self, risk_df: pd.DataFrame, filename: str = None) -> Path:
        """Save risk indicators to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_risk_indicators_{timestamp}.csv"
        
        filepath = self.data_dir / filename
        risk_df.to_csv(filepath, index=False)
        logger.info(f"Saved risk indicators to {filepath}")
        
        return filepath
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get detailed company information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'employee_count': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'business_summary': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {str(e)}")
            return {}

def main():
    """Main function to demonstrate the financial data collector."""
    logger.info("Starting financial data collection...")
    
    # Initialize collector
    collector = FinancialDataCollector()
    
    # Collect stock data
    stock_data = collector.collect_stock_data(period="1y")
    
    if stock_data:
        # Calculate risk indicators
        risk_indicators = collector.calculate_risk_indicators(stock_data)
        
        # Display summary
        print("\n" + "="*50)
        print("FINANCIAL RISK SUMMARY")
        print("="*50)
        print(f"Companies analyzed: {len(risk_indicators)}")
        print(f"Data collection date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not risk_indicators.empty:
            print(f"\nTop 5 Highest Risk Companies (by Financial Risk Score):")
            top_risk = risk_indicators.nlargest(5, 'Financial_Risk_Score')
            for _, company in top_risk.iterrows():
                print(f"  {company['Company']}: {company['Financial_Risk_Score']:.2f}")
            
            print(f"\nTop 5 Most Volatile Companies (30-day volatility):")
            top_volatile = risk_indicators.nlargest(5, 'Volatility_30d')
            for _, company in top_volatile.iterrows():
                print(f"  {company['Company']}: {company['Volatility_30d']:.1%}")
        
        # Save results
        filepath = collector.save_risk_indicators(risk_indicators)
        print(f"\nResults saved to: {filepath}")
        
    else:
        logger.error("No stock data collected!")

if __name__ == "__main__":
    main()