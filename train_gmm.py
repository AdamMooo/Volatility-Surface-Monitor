#!/usr/bin/env python3
"""
GMM Backtesting Script

This script performs comprehensive backtesting of GMM regime classification models
using extended historical data to find the best performing configuration.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os

from analytics.regime_classifier import GMMRegimeClassifier
from dashboard import load_synthetic_historical_data

def main():
    print("GMM Regime Classification Backtesting")
    print("=" * 50)

    # Load extended historical data
    print("Loading extended historical dataset...")
    historical_data = load_synthetic_historical_data()

    print(f"Dataset shape: {historical_data.shape}")
    print(f"Date range: {historical_data.index.min()} to {historical_data.index.max()}")
    print(f"Duration: {(historical_data.index.max() - historical_data.index.min()).days} days")

    # Test different feature combinations
    feature_sets = {
        'volatility_only': ['avg_atm_vol', 'roughness'],
        'volatility_surface': ['avg_atm_vol', 'avg_25d_skew', 'roughness', 'avg_atm_curvature'],
        'market_context': ['avg_atm_vol', 'unemployment_rate', 'yield_curve_spread', 'spy_return'],
        'comprehensive': ['avg_atm_vol', 'avg_25d_skew', 'roughness', 'unemployment_rate', 'yield_curve_spread', 'spy_return']
    }

    results = {}

    print("\nTesting different feature combinations...")
    for name, features in feature_sets.items():
        print(f"\nTesting feature set: {name}")
        print(f"Features: {features}")
        
        # Filter data to available features
        available_features = [f for f in features if f in historical_data.columns]
        if len(available_features) != len(features):
            print(f"Warning: Some features not available. Using: {available_features}")
        
        data_subset = historical_data[available_features].dropna()
        
        if len(data_subset) < 1000:
            print(f"Insufficient data for {name}: {len(data_subset)} samples")
            continue
        
        # Initialize and fit GMM
        gmm = GMMRegimeClassifier(n_components=5, random_state=42, feature_cols=available_features)
        gmm.fit(data_subset)
        
        # Calculate model metrics
        from sklearn.mixture import GaussianMixture
        scaler = gmm.scaler
        scaled_data = scaler.transform(data_subset.values)
        
        bic = gmm.gmm.bic(scaled_data)
        aic = gmm.gmm.aic(scaled_data)
        log_likelihood = gmm.gmm.score(scaled_data)
        
        results[name] = {
            'features': available_features,
            'n_samples': len(data_subset),
            'bic': bic,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'gmm': gmm
        }
        
        print(f"Samples: {len(data_subset)}")
        print(f"BIC: {bic:.2f}, AIC: {aic:.2f}")
        print(f"Average Log Likelihood: {log_likelihood:.4f}")

    # Compare model performance
    comparison_df = pd.DataFrame({
        name: {
            'BIC': results[name]['bic'],
            'AIC': results[name]['aic'],
            'Log Likelihood': results[name]['log_likelihood'],
            'Features': len(results[name]['features']),
            'Samples': results[name]['n_samples']
        }
        for name in results.keys()
    }).T

    print("\nModel Comparison:")
    print(comparison_df.round(2))

    # Select best performing model
    best_model_name = comparison_df['BIC'].idxmin()
    best_gmm = results[best_model_name]['gmm']
    best_features = results[best_model_name]['features']

    print(f"\nBest model: {best_model_name}")
    print(f"Features: {best_features}")

    # Time series cross-validation
    print("\nPerforming time series cross-validation...")
    cv_data = historical_data[best_features].dropna()
    cv_data = cv_data.sort_index()

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(cv_data)):
        train_data = cv_data.iloc[train_idx]
        test_data = cv_data.iloc[test_idx]
        
        # Fit on training data
        temp_gmm = GMMRegimeClassifier(n_components=5, random_state=42, feature_cols=best_features)
        temp_gmm.fit(train_data)
        
        # Score on test data
        test_scaled = temp_gmm.scaler.transform(test_data.values)
        test_score = temp_gmm.gmm.score(test_scaled)
        cv_scores.append(test_score)
        
        print(f"Fold {fold+1}: Train={len(train_data)}, Test={len(test_data)}, Score={test_score:.4f}")

    print(f"\nCV Average Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Generate regime predictions
    print("\nAnalyzing regime predictions...")
    predictions = []

    for i in range(len(cv_data)):
        current_point = cv_data.iloc[i:i+1]
        pred = best_gmm.predict(current_point.to_dict('records')[0])
        predictions.append(pred['regime'])

    cv_data = cv_data.copy()
    cv_data['predicted_regime'] = predictions

    print("Regime Distribution:")
    print(cv_data['predicted_regime'].value_counts().sort_index())

    # Test with current market data
    print("\nTesting with current market conditions...")
    current_market_data = {
        'avg_atm_vol': 0.18,  # Current ATM vol ~18%
        'avg_25d_skew': -0.08,  # Typical skew
        'roughness': 0.06,  # Surface roughness
        'unemployment_rate': 4.2,
        'yield_curve_spread': 0.8,
        'spy_return': 0.001
    }

    current_test = {k: v for k, v in current_market_data.items() if k in best_features}
    prediction = best_gmm.predict(current_test)

    print("Current Market Regime Prediction:")
    print(f"Predicted Regime: {prediction['regime']}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Distribution Fit: {prediction['distribution_fit']}")

    # Save the best model
    print("\nSaving the best model...")
    model_dir = project_root / "src" / "data"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "gmm_regime_classifier.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(best_gmm, f)

    print(f"Model saved to {model_path}")
    print(f"Features: {best_features}")
    print(f"Training samples: {len(cv_data)}")
    print(f"Cross-validation score: {np.mean(cv_scores):.4f}")

    print("\nGMM backtesting complete! The improved model is now ready for the dashboard.")
if __name__ == "__main__":
    main()