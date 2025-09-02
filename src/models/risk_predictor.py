import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class SupplyChainRiskPredictor:
    def __init__(self):
        self.feature_importance = None
        
    def engineer_features(self, df):
        """Create features from available columns only"""
        available_cols = df.columns.tolist()
        print(f"Available columns: {available_cols}")
        
        feature_columns = []
        
        # Use only columns that exist
        possible_features = ['Volatility_30d', 'Price_Change_30d', 'Volume_Change_Ratio', 
                           'Current_vs_52w_High', 'Current_vs_52w_Low']
        
        for col in possible_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Create simple derived features
        if 'Current_Price' in df.columns:
            df['price_level'] = pd.qcut(df['Current_Price'], q=3, labels=[0, 1, 2], duplicates='drop')
            feature_columns.append('price_level')
        
        if len(feature_columns) == 0:
            raise ValueError("No suitable features found in the dataset")
        
        print(f"Using features: {feature_columns}")
        return df[feature_columns].fillna(0)
    
    def simple_train_validate(self, df):
        """Simple validation without sklearn"""
        X = self.engineer_features(df)
        y = df['Financial_Risk_Score']
        
        print(f"Training with {len(X)} samples and {len(X.columns)} features")
        
        # Simple correlation-based feature importance
        if len(X.columns) > 0:
            correlations = []
            for col in X.columns:
                corr = X[col].corr(y)
                correlations.append({'feature': col, 'importance': abs(corr)})
            
            self.feature_importance = pd.DataFrame(correlations).sort_values('importance', ascending=False)
        
        # Simple metrics
        mean_risk = y.mean()
        std_risk = y.std()
        
        results = {
            'mean_risk_score': mean_risk,
            'std_risk_score': std_risk,
            'feature_count': len(X.columns),
            'sample_count': len(df)
        }
        
        return results

def main():
    """Train the model with current data"""
    data_dir = Path("data/raw/financial_data")
    csv_files = list(data_dir.glob("financial_risk_indicators_*.csv"))
    
    if not csv_files:
        print("No data found. Run: python src/data_collection/financial_data.py")
        return
    
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    print(f"Training model with {len(df)} companies...")
    print("Columns in dataset:")
    print(df.columns.tolist())
    
    # Initialize and train
    predictor = SupplyChainRiskPredictor()
    results = predictor.simple_train_validate(df)
    
    # Print results
    print("\nModel Training Results:")
    print("=" * 50)
    print(f"Companies analyzed: {results['sample_count']}")
    print(f"Features used: {results['feature_count']}")
    print(f"Mean risk score: {results['mean_risk_score']:.3f}")
    print(f"Risk score std: {results['std_risk_score']:.3f}")
    
    # Feature importance
    if predictor.feature_importance is not None:
        print("\nFeature Importance (by correlation):")
        for _, row in predictor.feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return predictor

if __name__ == "__main__":
    main()