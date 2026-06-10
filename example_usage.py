"""
Example script demonstrating BrainLat library usage with newly integrated features.

This script showcases:
1. Multicollinearity (VIF) and feature direction assessment.
2. Nested GBR Cross-Validation with Bayesian Hyperparameter Search and execution logging.
3. Leave-One-Site-Out (LOGO) GBR regression modeling.
4. Bootstrap Odds Ratios (OR) and Relative Risks (RR) analyses.
5. Generalized Additive Models (GAM) and Meta-GAM ensembling with LightGBM.
6. Stats & diagnostics: Delta AIC comparison, diagnostic reports, and forest plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add BrainLat to path
sys.path.insert(0, '.')

# Import BrainLat modules
from Brainlat.age_gap_models import (
    regression_gbr,
    Regression_GBR_nested,
    Regression_GBR_leave_one_site_out,
    Regression_GBR_nested_heldout,
    Regression_GBR_nested_heldout_mean_hyp
)
from Brainlat.regresion_model import Regression_Linear_NestedCV
from Brainlat.tools import (
    compute_directions,
    calculate_vif,
    compute_odds_ratios_bootstrap,
    compute_relative_risks_bootstrap
)
from Brainlat.gam_model import run_regressions_and_ensemble_cv
from Brainlat.stats import calculate_delta_aic, bootstrap_delta_aic
from Brainlat.graphics import (
    plot_diagnostic_panel,
    plot_gap_analysis,
    plot_or_rr_forest
)


def generate_rich_sample_data(n_samples=80, n_features=3, seed=42):
    """Generate synthetic rich biomarker/exposome and longitudinal data."""
    np.random.seed(seed)
    
    # Generate continuous predictor features (e.g. biomarkers/exposomes)
    X_base = np.random.randn(n_samples, n_features)
    X = pd.DataFrame(X_base, columns=[f'biomarker_{i+1}' for i in range(n_features)])
    
    # Target variable (e.g., Age)
    y = 50 + 8 * X['biomarker_1'] - 4 * X['biomarker_2'] + np.random.randn(n_samples) * 2
    
    # Binary classification outcome (e.g., Clinical status: 0 = Control, 1 = Disease)
    y_bin = np.where(y > 50 + np.random.randn(n_samples) * 3, 1, 0)
    
    # Grouping variable representing study sites/countries
    groups = pd.Series(np.random.choice(['Site_A', 'Site_B', 'Site_C'], size=n_samples))
    
    # Time variable representing longitudinal follow-up (in years)
    delta_time = np.random.uniform(1.0, 5.0, size=n_samples)
    
    return X, y, y_bin, groups, delta_time


def main():
    """Main demonstration workflow."""
    
    # Create logs/plots directory if they don't exist
    os.makedirs('brainlat_logs', exist_ok=True)
    os.makedirs('example_plots', exist_ok=True)
    
    # ========== 1. Generate Rich Sample Data ==========
    print("1. Generating synthetic biomarker, diagnostic, and longitudinal data...")
    X, y, y_bin, groups, delta_time = generate_rich_sample_data(n_samples=80, n_features=3)
    print(f"   - Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   - Age Target Range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   - Site Distribution: {dict(groups.value_counts())}")
    print()
    
    # ========== 2. Pre-Modeling Diagnostics (VIF & Directions) ==========
    print("2. Pre-modeling diagnostics...")
    vif_df = calculate_vif(X)
    print("   Variance Inflation Factors (VIF):")
    print(vif_df.to_string(index=False))
    
    directions_df = compute_directions(X, y)
    print("\n   Biomarker univariate directions:")
    print(directions_df.to_string(index=False))
    print()
    
    # ========== 3. Nested GBR Cross-Validation (with Logging) ==========
    print("3. Running Nested GBR CV with Bayesian Hyperparameter Search & Logging...")
    print("   [Log will be saved to brainlat_logs/gbr_nested_regression.log]")
    
    # We run 3 outer folds and 2 inner folds for demonstration speed
    coef_df, r2_mean, pred_df, outer_cv = Regression_GBR_nested(
        X, y, outer_splits=3, inner_splits=2, log=True
    )
    
    print(f"\n   Nested GBR CV Results:")
    print(f"   - Mean R² across folds: {r2_mean:.4f}")
    print(f"   - MSE: {coef_df.loc['_intercept', 'mse']:.4f}")
    print(f"   - MAE (Mean Absolute Error): {coef_df.loc['_intercept', 'MAE']:.4f}")
    print()
    
    # ========== 4. Leave-One-Site-Out (LOGO) GBR CV ==========
    print("4. Running Leave-One-Site-Out (LOGO) GBR CV...")
    print("   [Log will be saved to brainlat_logs/gbr_logo_regression.log]")
    
    logo_coef, logo_r2, logo_preds, logo_cv = Regression_GBR_leave_one_site_out(
        X, y, groups=groups, inner_splits=2, log=True
    )
    print(f"\n   LOGO GBR Results:")
    print(f"   - Mean R² across sites: {logo_r2:.4f}")
    print(f"   - MAE: {logo_coef.loc['_intercept', 'MAE']:.4f}")
    print()
    
    # ========== 5. Odds Ratios & Relative Risks Bootstrap ==========
    print("5. Running Odds Ratios (OR) and Relative Risks (RR) bootstrap analyses...")
    print("   Computing Odds Ratios (Logistic Regression Bootstrap)...")
    df_or = compute_odds_ratios_bootstrap(X, y_bin, n_iterations=20, test_size=0.2, random_state=42)
    print(df_or[['Feature', 'OR', '2.5%', '97.5%', 'P>|z|']].to_string(index=False))
    
    # Relative Risks from corrected gap (BAG = y_pred - y)
    y_gap_corrected = pred_df['GAP_corrected'].values
    print("\n   Computing Relative Risks (Poisson/Log-Binomial GLM Bootstrap)...")
    df_rr = compute_relative_risks_bootstrap(X, y_gap_corrected, delta_time, n_iterations=20, test_size=0.2, random_state=42)
    print(df_rr[['Feature', 'RR', '2.5%', '97.5%', 'P>|z|']].to_string(index=False))
    print()
    
    # ========== 6. GAM & Meta-GAM Ensemble CV ==========
    print("6. Fitting GAM models and building a Meta-GAM Ensemble using LightGBM...")
    # Prepare a dataframe containing all predictors and the BAG target variable
    df_all = X.copy()
    df_all['BAG'] = y_gap_corrected
    
    gam_results, all_gam_list = run_regressions_and_ensemble_cv(
        df_all=df_all,
        target_features=X.columns,
        category_title="Exposomes",
        topn=2,
        outcome='BAG',
        normalize=True,
        n_cv_splits=3,
        file_save='example_plots',
        dosave=True,
        log=True
    )
    print("\n   GAM & Meta-GAM CV comparisons (sorted by Delta AIC):")
    print(gam_results)
    print()
    
    # ========== 7. OLS Regression for Comparison (Linear NestedCV) ==========
    print("7. Fitting standard Linear OLS (NestedCV)...")
    print("   [Log will be saved to brainlat_logs/linear_nestedcv_regression.log]")
    from sklearn.linear_model import LinearRegression
    coef_df_ols, pred_df_ols, _ = Regression_Linear_NestedCV(
        X, y, LinearRegression(), param_space=None, outer_splits=3, log=True
    )
    print("   OLS Model Summary:")
    print(coef_df_ols.loc['_intercept', :].to_string())
    print()
    
    # ========== 8. Model Comparison & Delta AIC ==========
    print("8. Comparing GBR vs OLS predictions with Bootstrap Delta AIC...")
    y_pred_ols = pred_df_ols['y_pred'].values
    y_pred_gbr = pred_df['y_pred'].values
    
    # Delta AIC compared to null
    delta_aic_gbr = calculate_delta_aic(y, y_pred_gbr, k=X.shape[1])
    delta_aic_ols = calculate_delta_aic(y, y_pred_ols, k=X.shape[1])
    print(f"   - GBR Delta AIC (vs Null): {delta_aic_gbr:.2f}")
    print(f"   - OLS Delta AIC (vs Null): {delta_aic_ols:.2f}")
    
    mean_diff, ci = bootstrap_delta_aic(y, y_pred_gbr, k=X.shape[1], n_bootstrap=100, ci=95)
    print(f"   - Bootstrap GBR Delta AIC (Mean [95% CI]): {mean_diff:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    print()
    
    # ========== 9. Save Diagnostic Plots ==========
    print("9. Generating visual plots...")
    
    # 9.1 Diagnostic panel for GBR
    plot_diagnostic_panel(y.values, y_pred_gbr, coef_df)
    plt.savefig('example_plots/gbr_diagnostics.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("   - Saved: example_plots/gbr_diagnostics.png")
    
    # 9.2 Gap Analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_gap_analysis(pred_df, ax=axes)
    plt.savefig('example_plots/gbr_gap_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("   - Saved: example_plots/gbr_gap_analysis.png")
    
    # 9.3 Forest Plot of Odds Ratios
    plot_or_rr_forest(df_or, effect_type='OR', filename="example_plots/or_forest_plot")
    print("   - Saved: example_plots/or_forest_plot.png / .pdf")
    
    # 9.4 Forest Plot of Relative Risks
    plot_or_rr_forest(df_rr, effect_type='RR', filename="example_plots/rr_forest_plot")
    print("   - Saved: example_plots/rr_forest_plot.png / .pdf")
    print()
    
    print("=" * 80)
    print("BrainLat Demonstration Complete! See generated logs and plots in:")
    print("  - Logs: brainlat_logs/")
    print("  - Plots: example_plots/")
    print("=" * 80)


if __name__ == "__main__":
    main()
