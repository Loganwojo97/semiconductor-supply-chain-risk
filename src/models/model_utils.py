import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime

def save_model_results(predictor, results):
    """Save model and results for dashboard use"""
    results_dir = Path("results/model_performance")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature importance for dashboard
    if predictor.feature_importance is not None:
        feature_path = results_dir / "latest_feature_importance.csv"
        predictor.feature_importance.to_csv(feature_path, index=False)
    
    # Save model metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'companies_analyzed': results['sample_count'],
        'features_used': results['feature_count'],
        'mean_risk_score': results['mean_risk_score']
    }
    
    with open(results_dir / "latest_metrics.json", 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    print(f"Model results saved to {results_dir}")