import schedule
import time
from src.data_collection.financial_data import FinancialDataCollector
from src.data_collection.news_collector import AdvancedNewsCollector

def daily_data_update():
    # Collect financial data
    collector = FinancialDataCollector()
    financial_data = collector.collect_stock_data()
    
    # Collect news data
    news_collector = AdvancedNewsCollector()
    news_results = news_collector.collect_and_analyze()
    
    print(f"Updated data for {len(financial_data)} companies")
    return True

# Schedule daily updates
schedule.every().day.at("09:00").do(daily_data_update)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check hourly