"""
Example script demonstrating BrainLat library usage.

This script shows how to use the BrainLat library for:
1. Data preparation and visualization
2. Multicollinearity assessment (VIF)
3. Feature direction analysis
4. Gap-corrected regression modeling
5. Comprehensive diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add BrainLat to path
sys.path.insert(0, '.')

# Import BrainLat modules
from Brainlat.age_gap_models import regression_gbr
from Brainlat.regresion_model import regression_ols, regression_ridge
from Brainlat.tools import compute_directions, calculate_vif, scale_features
from Brainlat.graphics import (
    plot_diagnostic_panel, 
    plot_predictions_vs_actual,
    plot_feature_importance,
    plot_gap_analysis
)


def generate_sample_data(n_samples=100, n_features=5, seed=42):
    """Generate synthetic biomarker data."""
    np.random.seed(seed)
    
    # Create features with some correlation
    X_base = np.random.randn(n_samples, n_features)
    X = pd.DataFrame(X_base, columns=[f'biomarker_{i+1}' for i in range(n_features)])
    
    # Create target with non-linear relationships
    y = (
        0.5 * X['biomarker_1'] +
        -0.3 * X['biomarker_2'] +
        0.2 * X['biomarker_1']**2 +
        0.15 * X['biomarker_3'] * X['biomarker_4'] +
        np.random.randn(n_samples) * 0.5
    )
    
    return X, y


def main():
    """Main analysis workflow."""
    
    print("=" * 70)
    print("BrainLat Example Workflow: Neurodegeneration Biomarker Analysis")
    print("=" * 70)
    print()
    
    # ========== 1. Generate Sample Data ==========
    print("1. Generating sample biomarker data...")
    X, y = generate_sample_data(n_samples=100, n_features=5)
    print(f"   - Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   - Target range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # ========== 2. Multicollinearity Assessment ==========
    print("2. Assessing multicollinearity (VIF)...")
    vif_df = calculate_vif(X)
    print(vif_df.to_string(index=False))
    print("   Note: VIF > 5-10 indicates potential multicollinearity")
    print()
    
    # ========== 3. Feature Direction Analysis ==========
    print("3. Computing feature directions...")
    directions = compute_directions(X, y, bonferroni_correction=True)
    print(directions.to_string(index=False))
    print()
    
    # ========== 4. Gap-Corrected Gradient Boosting ==========
    print("4. Fitting Gradient Boosting with gap correction...")
    coef_df_gbr, r2_list_gbr, pred_df_gbr = regression_gbr(
        X, y,
        min_iter=1,
        max_iter=2,
        n_splits=5,
        params_b=None,
        shaps_comp=False
    )
    print("   Model Summary:")
    print(coef_df_gbr.loc['_intercept', :].to_string())
    print()
    print("   Feature Importances:")
    print(coef_df_gbr.iloc[1:, 0:2].to_string())
    print()
    
    # ========== 5. Predictions and Gap Analysis ==========
    print("5. Gap correction analysis:")
    print(f"   - Mean original gap: {pred_df_gbr['GAP'].mean():.4f}")
    print(f"   - Std original gap: {pred_df_gbr['GAP'].std():.4f}")
    print(f"   - Mean corrected gap: {pred_df_gbr['GAP_corrected'].mean():.4f}")
    print(f"   - Std corrected gap: {pred_df_gbr['GAP_corrected'].std():.4f}")
    print()
    
    # ========== 6. OLS Regression for Comparison ==========
    print("6. Fitting OLS for comparison...")
    results_ols = regression_ols(X, y, n_splits=5, scale=True)
    coef_df_ols = results_ols['coef_df']
    print("   Model Summary:")
    print(coef_df_ols.loc['_intercept', :].to_string())
    print()
    
    # ========== 7. Model Comparison ==========
    print("7. Model Comparison:")
    print(f"   {'Model':<20} {'R²':<10} {'RMSE':<10}")
    print("   " + "-" * 40)
    print(f"   {'GBR (gap-corrected)':<20} "
          f"{coef_df_gbr.loc['_intercept', 'R2']:<10.4f} "
          f"{coef_df_gbr.loc['_intercept', 'rmse']:<10.4f}")
    print(f"   {'OLS':<20} "
          f"{coef_df_ols.loc['_intercept', 'R2']:<10.4f} "
          f"{coef_df_ols.loc['_intercept', 'rmse']:<10.4f}")
    print()
    
    # ========== 8. Visualization ==========
    print("8. Creating diagnostic visualizations...")
    
    # Diagnostic panel for GBR
    y_pred_gbr = pred_df_gbr['y_pred'].values
    fig, axes = plot_diagnostic_panel(y.values, y_pred_gbr, coef_df_gbr)
    plt.savefig('diagnostic_panel_gbr.png', dpi=100, bbox_inches='tight')
    print("   - Saved: diagnostic_panel_gbr.png")
    
    # Gap analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_gap_analysis(pred_df_gbr, ax=axes)
    plt.savefig('gap_analysis.png', dpi=100, bbox_inches='tight')
    print("   - Saved: gap_analysis.png")
    
    plt.close('all')
    print()
    
    # ========== 9. Summary Statistics ==========
    print("9. Summary Statistics:")
    print(f"   - Number of iterations/folds: {len(r2_list_gbr)}")
    print(f"   - Mean R² across iterations: {np.mean(r2_list_gbr):.4f}")
    print(f"   - Std R² across iterations: {np.std(r2_list_gbr):.4f}")
    print()
    
    print("=" * 70)
    print("Analysis complete! Check diagnostic plots and results above.")
    print("=" * 70)
    
    return {
        'X': X,
        'y': y,
        'vif': vif_df,
        'directions': directions,
        'coef_gbr': coef_df_gbr,
        'pred_gbr': pred_df_gbr,
        'coef_ols': coef_df_ols,
        'pred_ols': results_ols['predictions']
    }


if __name__ == "__main__":
    results = main()
