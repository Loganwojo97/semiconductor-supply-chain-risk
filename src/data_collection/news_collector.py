"""
Advanced news collection and sentiment analysis for supply chain risk assessment.
Complete working version with proper error handling.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import json
from textblob import TextBlob
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNewsCollector:
    """Advanced news collection with multi-source aggregation and sentiment analysis."""
    
    def __init__(self, news_api_key: str = None):
        """Initialize with API keys and configuration."""
        self.news_api_key = news_api_key
        self.data_dir = Path("data/raw/news_sentiment")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Semiconductor-specific keywords for enhanced relevance
        self.semiconductor_keywords = [
            'semiconductor', 'chip shortage', 'supply chain', 'foundry', 'wafer',
            'TSMC', 'Intel', 'NVIDIA', 'AMD', 'Qualcomm', 'Micron',
            'trade war', 'tariff', 'export restriction', 'sanctions',
            'Taiwan', 'China semiconductor', 'chip manufacturing',
            'silicon shortage', 'automotive chips', 'GPU shortage'
        ]
        
        # Geopolitical risk keywords
        self.geopolitical_keywords = [
            'US China trade', 'Taiwan tensions', 'export controls',
            'semiconductor sanctions', 'chip war', 'technology transfer',
            'national security', 'strategic technology', 'dual use technology'
        ]
        
        # Supply chain disruption keywords
        self.disruption_keywords = [
            'factory shutdown', 'port congestion', 'logistics crisis',
            'shipping delay', 'raw material shortage', 'power outage',
            'COVID lockdown', 'supply chain disruption', 'production halt'
        ]
        
        # Sentiment weight modifiers for different types of news
        self.sentiment_weights = {
            'geopolitical': 1.5,
            'supply_chain': 1.3,
            'company_specific': 1.2,
            'general_market': 1.0
        }
    
    def collect_news_from_api(self, 
                             query: str, 
                             days_back: int = 7,
                             sources: str = None,
                             language: str = 'en') -> List[Dict]:
        """
        Collect news from NewsAPI with advanced filtering.
        """
        if not self.news_api_key:
            logger.warning("No NewsAPI key provided. Using sample data.")
            return self._get_sample_news_data()
        
        base_url = "https://newsapi.org/v2/everything"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'q': query,
            'apiKey': self.news_api_key,
            'language': language,
            'sortBy': 'relevancy',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'pageSize': 100
        }
        
        if sources:
            params['sources'] = sources
        
        try:
            logger.info(f"Fetching news for query: {query}")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"Found {len(articles)} articles")
            return articles
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch news: {e}")
            return self._get_sample_news_data()
    
    def _get_sample_news_data(self) -> List[Dict]:
        """Generate realistic sample news data for demonstration."""
        sample_articles = [
            {
                'title': 'TSMC Reports Strong Q4 Earnings Despite Supply Chain Challenges',
                'description': 'Taiwan Semiconductor Manufacturing Company reports robust quarterly results while navigating ongoing supply chain complexities and geopolitical tensions.',
                'content': 'TSMC demonstrated resilience in Q4 2024, posting strong financial results despite facing continued supply chain pressures.',
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat(),
                'source': {'name': 'Reuters'},
                'url': 'https://example.com/tsmc-earnings'
            },
            {
                'title': 'US Expands Semiconductor Export Controls to China',
                'description': 'The Biden administration announces new restrictions on semiconductor equipment exports to Chinese companies.',
                'content': 'The United States has implemented additional export controls targeting Chinese semiconductor companies.',
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat(),
                'source': {'name': 'Financial Times'},
                'url': 'https://example.com/us-china-semiconductors'
            },
            {
                'title': 'Intel Announces $20B Arizona Fab Expansion',
                'description': 'Intel reveals plans to expand its Arizona semiconductor manufacturing facilities.',
                'content': 'Intel Corporation announced a significant expansion of its Arizona fabrication facilities.',
                'publishedAt': (datetime.now() - timedelta(days=3)).isoformat(),
                'source': {'name': 'TechCrunch'},
                'url': 'https://example.com/intel-arizona-expansion'
            },
            {
                'title': 'Global Chip Shortage Continues to Impact Auto Industry',
                'description': 'Automotive manufacturers report ongoing production delays due to persistent semiconductor shortages.',
                'content': 'The global automotive industry continues to grapple with semiconductor shortages.',
                'publishedAt': (datetime.now() - timedelta(days=4)).isoformat(),
                'source': {'name': 'Automotive News'},
                'url': 'https://example.com/auto-chip-shortage'
            },
            {
                'title': 'Taiwan Strengthens Semiconductor Security Measures',
                'description': 'Taiwan implements new security protocols for its critical semiconductor infrastructure.',
                'content': 'Taiwan has introduced enhanced security measures to protect its vital semiconductor manufacturing infrastructure.',
                'publishedAt': (datetime.now() - timedelta(days=5)).isoformat(),
                'source': {'name': 'Bloomberg'},
                'url': 'https://example.com/taiwan-semiconductor-security'
            }
        ]
        
        logger.info(f"Using {len(sample_articles)} sample news articles for demonstration")
        return sample_articles
    
    def analyze_sentiment_advanced(self, text: str) -> Dict[str, float]:
        """Advanced sentiment analysis with context-aware scoring."""
        if not text or text.strip() == '':
            return {
                'sentiment_score': 0.0,
                'magnitude': 0.0,
                'confidence': 0.0,
                'risk_signal': 0.0
            }
        
        # Clean text
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Basic sentiment analysis using TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Calculate metrics
        magnitude = abs(polarity)
        confidence = 1 - subjectivity
        risk_signal = max(0, (-polarity + 1) / 2)
        
        # Enhance risk signal for negative sentiment
        if polarity < -0.3:
            risk_signal *= 1.2
        elif polarity < 0:
            risk_signal *= 1.1
        
        return {
            'sentiment_score': float(polarity),
            'magnitude': float(magnitude),
            'confidence': float(confidence),
            'risk_signal': float(min(1.0, risk_signal))
        }
    
    def categorize_news_type(self, title: str, description: str, content: str) -> Tuple[str, float]:
        """Categorize news and assign relevance weight."""
        full_text = f"{title} {description} {content}".lower()
        
        # Check for different types of content
        geopolitical_matches = sum(1 for keyword in self.geopolitical_keywords 
                                 if keyword.lower() in full_text)
        supply_chain_matches = sum(1 for keyword in self.disruption_keywords 
                                 if keyword.lower() in full_text)
        semiconductor_matches = sum(1 for keyword in self.semiconductor_keywords 
                                  if keyword.lower() in full_text)
        
        # Determine category
        if geopolitical_matches >= 2:
            return 'geopolitical', self.sentiment_weights['geopolitical']
        elif supply_chain_matches >= 2:
            return 'supply_chain', self.sentiment_weights['supply_chain']
        elif semiconductor_matches >= 2:
            return 'company_specific', self.sentiment_weights['company_specific']
        else:
            return 'general_market', self.sentiment_weights['general_market']
    
    def process_news_batch(self, articles: List[Dict]) -> pd.DataFrame:
        """Process a batch of news articles with sentiment analysis."""
        processed_articles = []
        
        logger.info(f"Processing {len(articles)} articles...")
        
        for i, article in enumerate(articles):
            try:
                # Extract article data safely
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                published_at = article.get('publishedAt', '')
                source_name = article.get('source', {}).get('name', 'Unknown')
                url = article.get('url', '')
                
                # Skip articles with insufficient content
                if len(title) < 10 and len(description) < 20:
                    continue
                
                # Combine text for sentiment analysis
                full_text = f"{title}. {description}"
                if content and len(content) > len(description):
                    full_text = f"{title}. {content}"
                
                # Analyze sentiment
                sentiment_results = self.analyze_sentiment_advanced(full_text)
                
                # Categorize news
                category, weight = self.categorize_news_type(title, description, content)
                
                # Calculate weighted risk signal
                weighted_risk = sentiment_results['risk_signal'] * weight
                
                # Parse publication date
                try:
                    pub_date = pd.to_datetime(published_at)
                except:
                    pub_date = datetime.now()
                
                processed_articles.append({
                    'title': title,
                    'description': description,
                    'source': source_name,
                    'published_date': pub_date,
                    'url': url,
                    'category': category,
                    'relevance_weight': weight,
                    'sentiment_score': sentiment_results['sentiment_score'],
                    'sentiment_magnitude': sentiment_results['magnitude'],
                    'sentiment_confidence': sentiment_results['confidence'],
                    'risk_signal': sentiment_results['risk_signal'],
                    'weighted_risk_signal': weighted_risk,
                    'text_length': len(full_text)
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(articles)} articles")
                    
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                continue
        
        if not processed_articles:
            logger.warning("No articles were successfully processed")
            return pd.DataFrame()
        
        df = pd.DataFrame(processed_articles)
        logger.info(f"Successfully processed {len(df)} articles")
        
        return df
    
    def calculate_aggregate_sentiment(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics from processed news."""
        if df.empty:
            return {
                'overall_sentiment': 0.0,
                'risk_level': 0.0,
                'article_count': 0,
                'confidence': 0.0
            }
        
        # Weight sentiment by relevance and confidence
        weights = df['relevance_weight'] * df['sentiment_confidence']
        
        if weights.sum() == 0:
            return {
                'overall_sentiment': 0.0,
                'risk_level': 0.0,
                'article_count': len(df),
                'confidence': 0.0
            }
        
        # Calculate weighted averages
        overall_sentiment = np.average(df['sentiment_score'], weights=weights)
        risk_level = np.average(df['weighted_risk_signal'], weights=weights)
        avg_confidence = df['sentiment_confidence'].mean()
        
        return {
            'overall_sentiment': float(overall_sentiment),
            'risk_level': float(risk_level),
            'article_count': len(df),
            'confidence': float(avg_confidence)
        }
    
    def collect_and_analyze(self, queries: List[str] = None, days_back: int = 7) -> Dict:
        """Main method to collect and analyze news."""
        if queries is None:
            queries = ['semiconductor supply chain', 'chip shortage', 'Taiwan TSMC']
        
        all_articles = []
        
        for query in queries:
            articles = self.collect_news_from_api(query, days_back=days_back)
            all_articles.extend(articles)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        # Process articles
        processed_df = self.process_news_batch(unique_articles)
        
        if not processed_df.empty:
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_analysis_{timestamp}.csv"
            filepath = self.data_dir / filename
            processed_df.to_csv(filepath, index=False)
            logger.info(f"Saved analysis to: {filepath}")
            
            # Calculate aggregate metrics
            aggregate_metrics = self.calculate_aggregate_sentiment(processed_df)
            
            return {
                'processed_articles': processed_df,
                'aggregate_metrics': aggregate_metrics,
                'file_saved': str(filepath)
            }
        else:
            return {
                'processed_articles': pd.DataFrame(),
                'aggregate_metrics': {},
                'error': 'No articles processed successfully'
            }

def main():
    """Main function to demonstrate the news collector."""
    logger.info("Starting advanced news collection and analysis...")
    
    # Initialize collector
    collector = AdvancedNewsCollector()
    
    # Collect and analyze news
    results = collector.collect_and_analyze(
        queries=['semiconductor', 'chip shortage', 'supply chain'],
        days_back=7
    )
    
    # Display results
    print("\n" + "="*60)
    print("NEWS SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    df = results['processed_articles']
    metrics = results['aggregate_metrics']
    
    print(f"Articles processed: {len(df)}")
    print(f"Overall sentiment: {metrics.get('overall_sentiment', 0):.2f}")
    print(f"Risk level: {metrics.get('risk_level', 0):.2f}")
    print(f"Confidence: {metrics.get('confidence', 0):.2f}")
    
    if not df.empty:
        print(f"\nTop Risk Headlines:")
        top_risk = df.nlargest(5, 'weighted_risk_signal')
        for _, article in top_risk.iterrows():
            print(f"  • {article['title'][:80]}...")
            print(f"    Risk: {article['weighted_risk_signal']:.2f} | Sentiment: {article['sentiment_score']:.2f}")
        
        print(f"\nNews Categories:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} articles")
    
    print(f"\nData saved to: {results.get('file_saved', 'N/A')}")
    
    return results

if __name__ == "__main__":
    main()