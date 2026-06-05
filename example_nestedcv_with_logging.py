"""
Example of using NestedCV functions with logging and sanity checks.

This example demonstrates:
- Data quality checks with sanity reports
- Logging of regression analysis
- Using Linear and NonLinear NestedCV with log parameter
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from Brainlat.regresion_model import Regression_Linear_NestedCV, Regression_NonLinear_NestedCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from skopt.space import Real, Integer

# Create sample data
np.random.seed(42)
n_samples = 100
n_features = 5

X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
y = X['feature_0'] * 2 + X['feature_1'] * 1.5 + np.random.randn(n_samples) * 0.5

print("=" * 70)
print("BrainLat NestedCV with Logging and Sanity Checks")
print("=" * 70)

# ============================================================================
# Example 1: Linear NestedCV with logging
# ============================================================================
print("\n[1] Linear NestedCV with Logging")
print("-" * 70)

linear_model = Ridge()
linear_param_space = {'alpha': Real(0.01, 10)}

coef_df_linear, pred_df_linear, params_linear = Regression_Linear_NestedCV(
    X, y,
    model=linear_model,
    param_space=linear_param_space,
    outer_splits=3,
    inner_splits=2,
    n_iter=5,
    log=True  
)

print("\nLinear NestedCV Results:")
print(coef_df_linear.head())
print(f"\nPredictions shape: {pred_df_linear.shape}")

# ============================================================================
# Example 2: NonLinear NestedCV with logging
# ============================================================================
print("\n" + "=" * 70)
print("[2] NonLinear NestedCV with Logging")
print("-" * 70)

nonlinear_model = GradientBoostingRegressor(random_state=42)
nonlinear_param_space = {
    'n_estimators': Integer(10, 50),
    'max_depth': Integer(2, 5),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform')
}

coef_df_nl, pred_df_nl, params_nl = Regression_NonLinear_NestedCV(
    X, y,
    model=nonlinear_model,
    param_space=nonlinear_param_space,
    outer_splits=3,
    inner_splits=2,
    n_iter=5,
    scoring='r2',
    log=True  # Enable logging
)

print("\nNonLinear NestedCV Results:")
print(coef_df_nl.head())
print(f"\nPredictions shape: {pred_df_nl.shape}")

# ============================================================================
# Example 3: Without logging (only console output)
# ============================================================================
print("\n" + "=" * 70)
print("[3] Linear NestedCV without logging (console only)")
print("-" * 70)

coef_df_no_log, pred_df_no_log, params_no_log = Regression_Linear_NestedCV(
    X, y,
    model=Ridge(),
    param_space={'alpha': Real(0.01, 10)},
    outer_splits=2,
    inner_splits=2,
    n_iter=3,
    log=False  # Disable logging
)

print("\n" + "=" * 70)
print("✓ All examples completed!")
print("=" * 70)
print("\nLogs have been saved to: brainlat_logs/")
print("Check the following files:")
print("  - linear_nestedcv_regression.log")
print("  - nonlinear_nestedcv_regression.log")
print("  - linear_nestedcv_sanity_check.txt")
print("  - nonlinear_nestedcv_sanity_check.txt")
