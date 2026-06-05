# BrainLat: Neurodegenerative Disease Analysis Library

**BrainLat** is a comprehensive Python library for advanced statistical analysis and machine learning modeling in neurodegenerative disease research, focusing on aging biomarkers and disease progression analysis.

##  Modules

| Module | Purpose |
|--------|---------|
| **age_gap_models** | Gradient Boosting with gap correction (non-linear) |
| **regresion_model** | OLS, Ridge, Lasso, ElasticNet (linear & regularized) |
| **stats** | Statistical functions (t-tests, F-tests, metrics) |
| **tools** | VIF, feature directions, scaling utilities |
| **graphics** | Diagnostic visualizations |
| **clasification_model** | Classification framework (future v0.2+) |

##  Key Features

### Gap-Corrected Regression (`brainlat.age_gap_models`)
- **Gradient Boosting with automatic gap correction**
- Gap correction uses GLM(gap, target) with Gaussian family
- K-fold cross-validation (configurable splits)
- Feature importances extraction
- Diagnostic metrics: R², adjusted R², RMSE, MDE, MAE
-  **Note**: Non-linear model → feature importances more reliable than t-values

### Standard Regression Models (`brainlat.regresion_model`)
- **OLS**: Full statistical inference (t-values, p-values, F-tests)
- **Ridge**: L2 regularization (multicollinearity handling)
- **Lasso**: L1 regularization (automatic feature selection)
- **ElasticNet**: Balanced L1/L2 regularization
- All with cross-validation and diagnostic metrics

### Statistical Tools (`brainlat.stats`)
- Coefficient standard errors, t-values, p-values
- Performance metrics: R², adjusted R², F-statistic
- Diagnostic measures: MDA, MAE, RMSE, MSE

### Feature Analysis (`brainlat.tools`)
- **Compute Directions**: Univariate feature analysis with Bonferroni correction
- **VIF**: Multicollinearity detection (VIF > 5-10 = problems)
- **Scaling**: MinMax and Z-score standardization

### Visualization (`brainlat.graphics`)
- Feature importance barplots
- Predictions vs actual scatter plots
- Residual diagnostics
- Gap analysis distributions
- Comprehensive 4-panel diagnostic panel

##  Quick Start

### Installation
```bash
cd /path/to/RedLat
pip install -r requirements.txt
```

### Example 1: Gap-Corrected Regression
```python
import pandas as pd
import numpy as np
from brainlat.age_gap_models import regression_gbr

X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
y = np.random.randn(100)

coef_df, r2_list, predictions = regression_gbr(X, y, n_splits=5)
print(coef_df.loc['_intercept', 'R2'])  # Model R²
```

### Example 2: OLS Regression
```python
from brainlat.regresion_model import regression_ols

results = regression_ols(X, y, n_splits=5)
print(results['coef_df'])  # Includes t-values and p-values
```

### Example 3: Feature Analysis
```python
from brainlat.tools import compute_directions, calculate_vif

vif = calculate_vif(X)
print(vif[vif['VIF'] > 5])  # High multicollinearity

directions = compute_directions(X, y, bonferroni_correction=True)
print(directions)
```

### Example 4: Diagnostics
```python
from brainlat.graphics import plot_diagnostic_panel

fig, axes = plot_diagnostic_panel(y, predictions['y_pred'].values, coef_df)
fig.savefig('diagnostics.png')
```

##  Complete API Reference

### brainlat.age_gap_models

**`regression_gbr(X, y, min_iter=1, max_iter=2, n_splits=10, params_b=None, shaps_comp=False)`**
- Gradient Boosting with gap correction
- Returns: `(coef_df, r2_list, predictions_df)` or with SHAP explainer
- Gap correction is automatic, using GLM(gap, target)
-  Non-linear: use feature importances, not coefficients

### brainlat.regresion_model

**`regression_ols(X, y, n_splits=10, scale=True)`**
- OLS with full statistical inference
- Returns: `dict` with 'coef_df', 'predictions', 'cv_scores'

**`regression_ridge(X, y, alpha=1.0, n_splits=10, scale=True)`**
- Ridge regression (L2 penalty)

**`regression_lasso(X, y, alpha=0.1, n_splits=10, scale=True)`**
- Lasso regression (L1 penalty)

**`regression_elasticnet(X, y, alpha=0.1, l1_ratio=0.5, n_splits=10, scale=True)`**
- ElasticNet (balanced L1/L2)

### brainlat.stats

| Function | Purpose |
|----------|---------|
| `coef_se(X, y_true, y_pred)` | Standard errors of coefficients |
| `coef_tval(coef, X, y_true, y_pred)` | T-values for coefficients |
| `coef_pval(coef, X, y_true, y_pred)` | P-values for coefficients |
| `calculate_r_squared(y_true, y_pred)` | R² (0-1) |
| `calculate_adjusted_r_squared(y_true, y_pred, n_feat)` | Adjusted R² |
| `calculate_f_statistic(y_true, y_pred, n_feat)` | F-test + p-value |
| `mean_directional_accuracy(y_true, y_pred)` | MDA metric |
| `mean_absolute_error(y_true, y_pred)` | MAE |
| `root_mean_squared_error(y_true, y_pred)` | RMSE |

### brainlat.tools

**`compute_directions(X, y, bonferroni_correction=True)`**
- Univariate feature directions with OLS
- Returns: DataFrame with Feature, coef, p_value

**`calculate_vif(X)`**
- Variance Inflation Factor for each feature
- Returns: DataFrame with Feature, VIF

**`scale_features(X, scale_range=(0.05, 0.95))`**
- MinMax scaling
- Returns: (scaled_X, scaler)

**`standardize_features(X)`**
- Z-score standardization
- Returns: (standardized_X, mean, std)

### brainlat.graphics

| Function | Returns |
|----------|---------|
| `plot_feature_importance(coef_df, results_df=None)` | Barplot axes |
| `plot_predictions_vs_actual(y_true, y_pred)` | Scatter plot axes |
| `plot_residuals(y_true, y_pred)` | 2 diagnostic axes |
| `plot_coef_with_error(coef_df)` | Barplot with error bars |
| `plot_gap_analysis(results_df)` | 2 distribution plots |
| `plot_diagnostic_panel(y_true, y_pred, coef_df)` | Figure + 4 axes |

##  Important Notes

### Understanding Gap Correction
- **Gap**: Difference between predicted and actual values
- **Corrected Gap**: Gap adjusted by GLM model to reduce systematic bias
- Most useful when predictions show systematic over/under-estimation

### Non-Linear Models vs Linear Models
| Aspect | GBR (Non-Linear) | OLS (Linear) |
|--------|------------------|--------------|
| **Interpretation** | Use feature importances | Use coefficients & t-values |
| **T-values** | Approximate, unreliable | Precise, reliable |
| **Flexibility** | High (captures non-linearity) | Low (linear relationships) |
| **Gap Correction** | Built-in | N/A |
| **Inference** | Limited | Full statistical |

### Feature Scaling
- All models automatically scale features to (0.05, 0.95)
- Avoids extreme values and improves numerical stability
- Pre-processing done internally

### Cross-Validation
- K-fold (default: 5-10 splits)
- Provides model stability estimates (std of R², RMSE)
- Helps detect overfitting

##  Typical Workflow

```python
# 1. Check multicollinearity
vif = calculate_vif(X)
print(vif[vif['VIF'] > 5])  # If any > 5-10, investigate

# 2. Analyze feature directions
directions = compute_directions(X, y)
print(directions)  # See which features drive the outcome

# 3. Fit models
coef_gbr, _, pred_gbr = regression_gbr(X, y, n_splits=5)
results_ols = regression_ols(X, y, n_splits=5)

# 4. Compare models
print(f"GBR R²: {coef_gbr.loc['_intercept', 'R2']:.4f}")
print(f"OLS R²: {results_ols['coef_df'].loc['_intercept', 'R2']:.4f}")

# 5. Visualize
fig, axes = plot_diagnostic_panel(y, pred_gbr['y_pred'], coef_gbr)
fig.savefig('diagnostics.png')
```

##  Dependencies

```
numpy>=1.21.0           # Numerical computing
pandas>=1.3.0           # Data manipulation  
scikit-learn>=1.0.0     # ML algorithms
scipy>=1.7.0            # Scientific functions
statsmodels>=0.13.0     # Statistical modeling
matplotlib>=3.4.0       # Plotting
seaborn>=0.11.0         # Statistical graphics
xgboost>=1.5.0          # Gradient boosting
shap>=0.41.0            # Model explainability (optional)
lxml>=4.6.0             # HTML parsing
```

Install all with:
```bash
pip install -r requirements.txt
```

##  Testing

Run the example workflow:
```bash
python example_usage.py
```

This demonstrates:
- Data generation
- VIF calculation
- Feature direction analysis
- GBR model fitting
- OLS comparison
- Diagnostic visualizations

##  Important Considerations

1. **No Missing Data**: Pre-process your data before passing to BrainLat
2. **Sample Size**: Minimum 20-50 samples recommended depending on features
3. **Feature Importances**: For GBR (non-linear), focus on these not t-values
4. **Gap Correction**: Most useful with systematic bias in predictions
5. **Random State**: Set explicitly if reproducibility is needed