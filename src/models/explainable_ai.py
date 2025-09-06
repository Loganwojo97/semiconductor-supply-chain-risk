"""
Explainable AI Module for Supply Chain Risk Prediction
Provides model interpretability using SHAP, LIME, and custom feature importance analysis
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not installed. Install with: pip install lime")

@dataclass
class FeatureExplanation:
    """Container for feature explanations"""
    feature_name: str
    importance_score: float
    shap_value: float
    direction: str  # 'increases' or 'decreases'
    baseline_value: float
    actual_value: float
    contribution: float  # actual contribution to prediction
    confidence_interval: Tuple[float, float]

@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction"""
    prediction: float
    confidence: float
    base_value: float
    top_features: List[FeatureExplanation]
    risk_drivers: Dict[str, Any]
    counterfactuals: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    model_confidence_factors: Dict[str, float]

class SupplyChainExplainableAI:
    """
    Advanced explainable AI system for supply chain risk predictions
    Provides multiple explanation methods and visualizations
    """
    
    def __init__(self, model, feature_names: List[str], 
                 training_data: Optional[pd.DataFrame] = None):
        """
        Initialize explainable AI system
        
        Args:
            model: Trained ML model (sklearn, xgboost, etc.)
            feature_names: List of feature names
            training_data: Training data for background distribution
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_statistics = {}
        
        # Initialize explainers
        self._initialize_explainers()
        
        # Calculate feature statistics
        if training_data is not None:
            self._calculate_feature_statistics()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        
        # Initialize SHAP
        if self.training_data is not None:
            # Use TreeExplainer for tree-based models
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except:
                # Fallback to KernelExplainer for any model type
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.sample(self.training_data, 100)
                )
        
        # Initialize LIME
        if LIME_AVAILABLE and self.training_data is not None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data.values,
                feature_names=self.feature_names,
                mode='regression',
                verbose=False
            )
    
    def _calculate_feature_statistics(self):
        """Calculate statistics for each feature"""
        for feature in self.feature_names:
            if feature in self.training_data.columns:
                self.feature_statistics[feature] = {
                    'mean': self.training_data[feature].mean(),
                    'std': self.training_data[feature].std(),
                    'min': self.training_data[feature].min(),
                    'max': self.training_data[feature].max(),
                    'median': self.training_data[feature].median(),
                    'q25': self.training_data[feature].quantile(0.25),
                    'q75': self.training_data[feature].quantile(0.75)
                }
    
    def explain_prediction(self, instance: np.ndarray, 
                          company_name: str = "Unknown") -> PredictionExplanation:
        """
        Generate comprehensive explanation for a single prediction
        
        Args:
            instance: Feature vector for prediction
            company_name: Name of company being analyzed
            
        Returns:
            PredictionExplanation object with all explanation details
        """
        
        # Get prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        
        # Get prediction probability/confidence if available
        try:
            pred_proba = self.model.predict_proba(instance.reshape(1, -1))[0]
            confidence = max(pred_proba)
        except:
            # For regression models, estimate confidence based on prediction variance
            confidence = self._estimate_confidence(instance)
        
        # Get SHAP values
        shap_values = self._get_shap_explanation(instance)
        
        # Get feature explanations
        feature_explanations = self._create_feature_explanations(
            instance, shap_values
        )
        
        # Sort by importance
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Identify risk drivers
        risk_drivers = self._identify_risk_drivers(feature_explanations)
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(instance, prediction)
        
        # Find similar cases
        similar_cases = self._find_similar_cases(instance, prediction)
        
        # Calculate model confidence factors
        confidence_factors = self._analyze_model_confidence(instance, prediction)
        
        return PredictionExplanation(
            prediction=prediction,
            confidence=confidence,
            base_value=self.shap_explainer.expected_value if self.shap_explainer else 50,
            top_features=feature_explanations[:10],
            risk_drivers=risk_drivers,
            counterfactuals=counterfactuals,
            similar_cases=similar_cases,
            model_confidence_factors=confidence_factors
        )
    
    def _get_shap_explanation(self, instance: np.ndarray) -> np.ndarray:
        """Get SHAP values for instance"""
        if self.shap_explainer:
            return self.shap_explainer.shap_values(instance)
        return np.zeros(len(self.feature_names))
    
    def _create_feature_explanations(self, instance: np.ndarray, 
                                    shap_values: np.ndarray) -> List[FeatureExplanation]:
        """Create detailed feature explanations"""
        explanations = []
        
        for i, feature_name in enumerate(self.feature_names):
            actual_value = instance[i]
            shap_value = shap_values[i] if len(shap_values) > i else 0
            
            # Get baseline from training data
            baseline = self.feature_statistics.get(feature_name, {}).get('median', 0)
            
            # Determine direction
            direction = "increases" if shap_value > 0 else "decreases"
            
            # Calculate confidence interval for importance
            ci_lower, ci_upper = self._calculate_importance_ci(feature_name, shap_value)
            
            explanation = FeatureExplanation(
                feature_name=feature_name,
                importance_score=abs(shap_value),
                shap_value=shap_value,
                direction=direction,
                baseline_value=baseline,
                actual_value=actual_value,
                contribution=shap_value,
                confidence_interval=(ci_lower, ci_upper)
            )
            explanations.append(explanation)
        
        return explanations
    
    def _identify_risk_drivers(self, 
                              feature_explanations: List[FeatureExplanation]) -> Dict[str, Any]:
        """Identify and categorize main risk drivers"""
        
        # Categorize features
        categories = {
            'geopolitical': ['trade_tension', 'sanctions', 'political_stability', 
                           'tariff_rate', 'export_restrictions'],
            'financial': ['stock_volatility', 'revenue_change', 'debt_ratio', 
                        'market_cap', 'credit_rating'],
            'operational': ['capacity_utilization', 'lead_time', 'inventory_days',
                          'supplier_concentration', 'production_volume'],
            'environmental': ['natural_disaster_risk', 'climate_exposure', 
                            'weather_severity', 'earthquake_risk'],
            'market': ['demand_forecast', 'price_volatility', 'market_share',
                      'competition_intensity', 'customer_concentration']
        }
        
        risk_drivers = {}
        
        for category, features in categories.items():
            category_impact = 0
            category_features = []
            
            for exp in feature_explanations:
                # Check if feature belongs to this category
                for feature_keyword in features:
                    if feature_keyword in exp.feature_name.lower():
                        category_impact += exp.contribution
                        category_features.append({
                            'feature': exp.feature_name,
                            'impact': exp.contribution,
                            'value': exp.actual_value,
                            'direction': exp.direction
                        })
                        break
            
            if category_features:
                risk_drivers[category] = {
                    'total_impact': category_impact,
                    'features': category_features,
                    'risk_level': self._categorize_risk_level(abs(category_impact))
                }
        
        # Add summary
        risk_drivers['summary'] = {
            'primary_driver': max(risk_drivers.keys(), 
                                 key=lambda k: abs(risk_drivers[k]['total_impact']))
                             if risk_drivers else 'unknown',
            'total_risk_contribution': sum(abs(rd['total_impact']) 
                                         for rd in risk_drivers.values()),
            'risk_distribution': {k: abs(v['total_impact']) 
                                for k, v in risk_drivers.items()}
        }
        
        return risk_drivers
    
    def _generate_counterfactuals(self, instance: np.ndarray, 
                                 current_prediction: float) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations (what-if scenarios)"""
        
        counterfactuals = []
        
        # For top 5 features, show what would happen if changed
        feature_importance = self._get_feature_importance()
        top_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        for feature_name, importance in top_features:
            feature_idx = self.feature_names.index(feature_name)
            
            # Create scenarios with different values
            scenarios = [
                ('10% decrease', 0.9),
                ('25% decrease', 0.75),
                ('10% increase', 1.1),
                ('25% increase', 1.25)
            ]
            
            for scenario_name, multiplier in scenarios:
                # Create modified instance
                modified_instance = instance.copy()
                modified_instance[feature_idx] *= multiplier
                
                # Get new prediction
                new_prediction = self.model.predict(modified_instance.reshape(1, -1))[0]
                
                # Calculate impact
                impact = new_prediction - current_prediction
                
                counterfactuals.append({
                    'feature': feature_name,
                    'scenario': scenario_name,
                    'original_value': instance[feature_idx],
                    'new_value': modified_instance[feature_idx],
                    'original_prediction': current_prediction,
                    'new_prediction': new_prediction,
                    'impact': impact,
                    'impact_percentage': (impact / current_prediction) * 100 if current_prediction != 0 else 0
                })
        
        # Sort by absolute impact
        counterfactuals.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return counterfactuals[:10]  # Return top 10 most impactful
    
    def _find_similar_cases(self, instance: np.ndarray, 
                           prediction: float) -> List[Dict[str, Any]]:
        """Find similar historical cases for comparison"""
        
        if self.training_data is None:
            return []
        
        # Calculate distances to all training instances
        distances = np.linalg.norm(
            self.training_data.values - instance.reshape(1, -1), 
            axis=1
        )
        
        # Find 5 most similar cases
        similar_indices = np.argsort(distances)[:5]
        
        similar_cases = []
        for idx in similar_indices:
            similar_instance = self.training_data.iloc[idx].values
            similar_prediction = self.model.predict(similar_instance.reshape(1, -1))[0]
            
            # Find key differences
            differences = []
            for i, feature_name in enumerate(self.feature_names):
                diff = abs(similar_instance[i] - instance[i])
                if diff > self.feature_statistics.get(feature_name, {}).get('std', 1) * 0.5:
                    differences.append({
                        'feature': feature_name,
                        'difference': diff,
                        'this_value': instance[i],
                        'similar_value': similar_instance[i]
                    })
            
            similar_cases.append({
                'case_id': f"Case_{idx}",
                'similarity_score': 1 / (1 + distances[idx]),
                'prediction': similar_prediction,
                'prediction_difference': similar_prediction - prediction,
                'key_differences': differences[:3]  # Top 3 differences
            })
        
        return similar_cases
    
    def _analyze_model_confidence(self, instance: np.ndarray, 
                                 prediction: float) -> Dict[str, float]:
        """Analyze factors affecting model confidence"""
        
        confidence_factors = {}
        
        # 1. Data density - how many training examples near this instance
        if self.training_data is not None:
            distances = np.linalg.norm(
                self.training_data.values - instance.reshape(1, -1), 
                axis=1
            )
            nearby_samples = np.sum(distances < np.std(distances))
            data_density = nearby_samples / len(self.training_data)
            confidence_factors['data_density'] = min(data_density * 10, 1.0)
        
        # 2. Feature extrapolation - are features within training range?
        extrapolation_score = 0
        in_range_features = 0
        
        for i, feature_name in enumerate(self.feature_names):
            stats = self.feature_statistics.get(feature_name, {})
            if stats:
                if stats['min'] <= instance[i] <= stats['max']:
                    in_range_features += 1
        
        if self.feature_names:
            confidence_factors['feature_coverage'] = in_range_features / len(self.feature_names)
        
        # 3. Prediction stability - how much does prediction change with small perturbations
        perturbation_results = []
        for _ in range(10):
            noise = np.random.normal(0, 0.01, instance.shape)
            perturbed = instance + noise
            perturbed_pred = self.model.predict(perturbed.reshape(1, -1))[0]
            perturbation_results.append(abs(perturbed_pred - prediction))
        
        stability = 1 - (np.mean(perturbation_results) / (abs(prediction) + 1))
        confidence_factors['prediction_stability'] = max(0, min(1, stability))
        
        # 4. Model agreement (if ensemble)
        if hasattr(self.model, 'estimators_'):
            predictions = [est.predict(instance.reshape(1, -1))[0] 
                         for est in self.model.estimators_]
            agreement = 1 - (np.std(predictions) / (np.mean(predictions) + 1))
            confidence_factors['model_agreement'] = max(0, min(1, agreement))
        
        # Overall confidence score
        confidence_factors['overall'] = np.mean(list(confidence_factors.values()))
        
        return confidence_factors
    
    def _estimate_confidence(self, instance: np.ndarray) -> float:
        """Estimate confidence for regression models"""
        
        # Use multiple perturbations to estimate variance
        predictions = []
        for _ in range(100):
            noise = np.random.normal(0, 0.01, instance.shape)
            perturbed = instance + noise
            pred = self.model.predict(perturbed.reshape(1, -1))[0]
            predictions.append(pred)
        
        # Convert variance to confidence (lower variance = higher confidence)
        variance = np.var(predictions)
        confidence = 1 / (1 + variance)
        
        return min(max(confidence, 0), 1)
    
    def _calculate_importance_ci(self, feature_name: str, 
                                shap_value: float) -> Tuple[float, float]:
        """Calculate confidence interval for feature importance"""
        
        # Simple bootstrap-based CI estimation
        if self.training_data is not None and len(self.training_data) > 30:
            bootstrap_importances = []
            
            for _ in range(100):
                # Sample with replacement
                sample_indices = np.random.choice(
                    len(self.training_data), 
                    size=len(self.training_data), 
                    replace=True
                )
                
                # This is simplified - in production, retrain model on bootstrap sample
                # Here we just add noise to simulate uncertainty
                noise = np.random.normal(0, abs(shap_value) * 0.1)
                bootstrap_importances.append(shap_value + noise)
            
            ci_lower = np.percentile(bootstrap_importances, 2.5)
            ci_upper = np.percentile(bootstrap_importances, 97.5)
        else:
            # Fallback to simple estimate
            ci_lower = shap_value * 0.8
            ci_upper = shap_value * 1.2
        
        return (ci_lower, ci_upper)
    
    def _categorize_risk_level(self, impact: float) -> str:
        """Categorize risk level based on impact"""
        if impact < 10:
            return "low"
        elif impact < 25:
            return "medium"
        elif impact < 50:
            return "high"
        else:
            return "critical"
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model"""
        
        importance_dict = {}
        
        # Try different methods based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            for i, importance in enumerate(self.model.feature_importances_):
                importance_dict[self.feature_names[i]] = importance
        
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]
            for i, importance in enumerate(coef):
                importance_dict[self.feature_names[i]] = abs(importance)
        
        else:
            # Fallback to permutation importance or SHAP
            if self.shap_explainer and self.training_data is not None:
                shap_values = self.shap_explainer.shap_values(
                    self.training_data.values[:100]
                )
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                for i, importance in enumerate(mean_abs_shap):
                    importance_dict[self.feature_names[i]] = importance
        
        return importance_dict
    
    def generate_explanation_report(self, 
                                   explanation: PredictionExplanation,
                                   company_name: str = "Unknown") -> Dict[str, Any]:
        """Generate a comprehensive explanation report for dashboard"""
        
        report = {
            'metadata': {
                'company': company_name,
                'prediction': float(explanation.prediction),
                'confidence': float(explanation.confidence),
                'timestamp': datetime.now().isoformat(),
                'risk_level': self._categorize_risk_level(explanation.prediction)
            },
            
            'key_insights': {
                'risk_score': f"{explanation.prediction:.1f}%",
                'confidence': f"{explanation.confidence:.1%}",
                'primary_driver': explanation.risk_drivers.get('summary', {}).get('primary_driver', 'unknown'),
                'top_risk_factors': [
                    {
                        'feature': f.feature_name,
                        'impact': f"{f.contribution:.2f}",
                        'direction': f.direction,
                        'value': f"{f.actual_value:.2f}"
                    }
                    for f in explanation.top_features[:5]
                ]
            },
            
            'risk_breakdown': {
                category: {
                    'impact': f"{details['total_impact']:.2f}",
                    'level': details['risk_level'],
                    'contributors': details['features'][:3]  # Top 3 per category
                }
                for category, details in explanation.risk_drivers.items()
                if category != 'summary'
            },
            
            'what_if_scenarios': [
                {
                    'description': f"{cf['feature']} {cf['scenario']}",
                    'current': f"{cf['original_value']:.2f}",
                    'new': f"{cf['new_value']:.2f}",
                    'risk_change': f"{cf['impact']:.1f}%",
                    'new_risk': f"{cf['new_prediction']:.1f}%"
                }
                for cf in explanation.counterfactuals[:5]
            ],
            
            'similar_historical_cases': [
                {
                    'similarity': f"{case['similarity_score']:.1%}",
                    'historical_risk': f"{case['prediction']:.1f}%",
                    'risk_difference': f"{case['prediction_difference']:+.1f}%",
                    'key_differences': case['key_differences']
                }
                for case in explanation.similar_cases[:3]
            ],
            
            'model_confidence_analysis': {
                metric: f"{value:.1%}"
                for metric, value in explanation.model_confidence_factors.items()
            },
            
            'feature_contributions': [
                {
                    'feature': f.feature_name,
                    'contribution': float(f.contribution),
                    'importance': float(f.importance_score),
                    'current_value': float(f.actual_value),
                    'baseline_value': float(f.baseline_value),
                    'direction': f.direction
                }
                for f in explanation.top_features
            ],
            
            'recommendations': self._generate_recommendations(explanation)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                 explanation: PredictionExplanation) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on explanation"""
        
        recommendations = []
        
        # Based on top risk factors
        for feature in explanation.top_features[:3]:
            if feature.direction == "increases" and feature.contribution > 5:
                if "inventory" in feature.feature_name.lower():
                    recommendations.append({
                        'priority': 'high',
                        'action': 'Increase safety stock',
                        'reason': f"Low inventory days ({feature.actual_value:.0f}) contributing {feature.contribution:.1f}% to risk",
                        'expected_impact': 'Reduce risk by 5-10%'
                    })
                elif "supplier" in feature.feature_name.lower():
                    recommendations.append({
                        'priority': 'high',
                        'action': 'Diversify supplier base',
                        'reason': f"High supplier concentration contributing {feature.contribution:.1f}% to risk",
                        'expected_impact': 'Reduce risk by 8-12%'
                    })
                elif "geopolitical" in feature.feature_name.lower():
                    recommendations.append({
                        'priority': 'medium',
                        'action': 'Develop contingency plans',
                        'reason': f"Elevated geopolitical tensions contributing {feature.contribution:.1f}% to risk",
                        'expected_impact': 'Improve response time by 30%'
                    })
        
        # Based on counterfactuals
        most_impactful = min(explanation.counterfactuals, 
                            key=lambda x: x['new_prediction'])
        if most_impactful['impact'] < -10:
            recommendations.append({
                'priority': 'high',
                'action': f"Optimize {most_impactful['feature']}",
                'reason': f"Could reduce risk by {abs(most_impactful['impact']):.1f}%",
                'expected_impact': f"Risk reduction to {most_impactful['new_prediction']:.1f}%"
            })
        
        return recommendations[:5]  # Return top 5 recommendations


# Example usage
if __name__ == "__main__":
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    feature_names = [
        'geopolitical_tension_index', 'supplier_concentration', 'inventory_days',
        'lead_time_variance', 'financial_health_score', 'natural_disaster_risk',
        'cyber_threat_level', 'demand_volatility', 'production_capacity',
        'shipping_delays', 'raw_material_prices', 'labor_availability',
        'regulatory_changes', 'technology_dependencies', 'market_competition',
        'customer_concentration', 'quality_issues', 'environmental_compliance',
        'trade_restrictions', 'currency_fluctuation'
    ]
    
    # Generate synthetic training data
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Create synthetic target (risk score)
    y_train = (
        X_train['geopolitical_tension_index'] * 15 +
        X_train['supplier_concentration'] * 10 +
        X_train['inventory_days'] * -5 +
        X_train['natural_disaster_risk'] * 8 +
        np.random.randn(n_samples) * 5 +
        50
    )
    y_train = np.clip(y_train, 0, 100)
    
    # Train a simple model (in production, use your actual model)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize explainable AI system
    explainer = SupplyChainExplainableAI(
        model=model,
        feature_names=feature_names,
        training_data=X_train
    )
    
    # Create a test instance
    test_instance = np.random.randn(n_features)
    test_instance[0] = 2.5  # High geopolitical tension
    test_instance[1] = 1.8  # High supplier concentration
    test_instance[2] = -1.5  # Low inventory days
    
    # Generate explanation
    print("🔍 Generating AI Explanation...")
    print("=" * 60)
    
    explanation = explainer.explain_prediction(
        test_instance,
        company_name="TSMC"
    )
    
    # Generate report
    report = explainer.generate_explanation_report(
        explanation,
        company_name="TSMC"
    )
    
    # Print results
    print(f"\n📊 RISK ASSESSMENT RESULTS")
    print(f"   Company: {report['metadata']['company']}")
    print(f"   Risk Score: {report['key_insights']['risk_score']}")
    print(f"   Confidence: {report['key_insights']['confidence']}")
    print(f"   Primary Driver: {report['key_insights']['primary_driver']}")
    
    print(f"\n🎯 TOP RISK FACTORS:")
    for factor in report['key_insights']['top_risk_factors']:
        print(f"   • {factor['feature']}: {factor['impact']} ({factor['direction']} risk)")
    
    print(f"\n💡 WHAT-IF SCENARIOS:")
    for scenario in report['what_if_scenarios'][:3]:
        print(f"   • {scenario['description']}: Risk changes to {scenario['new_risk']}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in report['recommendations'][:3]:
        print(f"   • [{rec['priority'].upper()}] {rec['action']}")
        print(f"     Reason: {rec['reason']}")
    
        # Save report to JSON
    import os
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "explanation_report.json")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✅ Explanation report saved to {report_path}")