"""
Gap-corrected regression models for aging and neurodegenerative disease analysis.

This module implements non-linear regression models that compute and correct
for systematic prediction gaps (bias) in the predictions.
"""

import numpy as np
import pandas as pd
import warnings
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.api import GLM
from statsmodels.api import families
import scipy

from .stats import (
    mean_directional_accuracy, mean_absolute_error
)
from .diagnostics import DataDiagnostics, Logger, generate_sanity_report

warnings.filterwarnings("ignore")


def regression_gbr(X, y, min_iter=1, max_iter=2, n_splits=10, 
                   params_b=None, shaps_comp=False, log=False):
    """
    Gradient Boosting Regressor with gap correction.
    
    Implements a non-linear gradient boosting model with systematic gap
    correction using GLM. This is useful for capturing complex patterns
    in aging and neurodegenerative disease biomarkers.
    
    NOTE: As a non-linear model, t-values and p-values are NOT computed.
    Use feature importances for interpretation of variable contributions.
    
    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix with feature names as columns
    y : array-like, shape (n_samples,)
        Target variable
    min_iter : int, default=1
        Minimum iteration number for random state variation
    max_iter : int, default=2
        Maximum iteration number (exclusive)
    n_splits : int, default=10
        Number of KFold splits for cross-validation
    params_b : dict or None, default=None
        Additional parameters for GradientBoostingRegressor
    shaps_comp : bool, default=False
        Whether to compute SHAP explainer (requires shap library)
    log : bool, default=False
        Whether to save logs and diagnostic reports to files
    
    Returns
    -------
    results : list
        If shaps_comp=False:
            [coef_df, r_squared_list, results_labels_df]
        If shaps_comp=True:
            [coef_df, r_squared_list, results_labels_df, explainer]
        
        - coef_df : pd.DataFrame
            Coefficients table with feature importances and model statistics
        - r_squared_list : list
            R² values for each iteration
        - results_labels_df : pd.DataFrame
            Predictions, gaps, and corrected gaps for each fold
        - explainer : shap.Explainer (optional)
            SHAP explainer for model interpretation
    
    Examples
    --------
    >>> from brainlat.age_gap_models import regression_gbr
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    >>> y = np.random.randn(100)
    >>> coef_df, r2_list, pred_df = regression_gbr(X, y, n_splits=5, log=True)
    >>> print(coef_df)
    """
    
    # Initialize logger if needed
    logger = Logger() if log else None
    
    if logger:
        logger.add_message(f"Starting GBR regression analysis...")
    
    # Perform sanity checks
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="GBR Regression Input")
    
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/gbr_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check completed: {report['n_samples']} samples, {report['n_features']} features")
    else:
        # Still print to console even if not logging
        report_str = generate_sanity_report(report)
        print("\n" + report_str)
    
    if params_b is None:
        params_b = -1
    
    # Scale features to (0.05, 0.95) to avoid extreme values
    scaler = MinMaxScaler((0.05, 0.95))
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    if logger:
        logger.add_message("Features scaled to (0.05, 0.95) range")
    
    lista_vars = list(X.columns)
    results_df_all = pd.DataFrame()
    r_squared_list = []
    
    for iteration in range(min_iter, max_iter):
        y_labels = []
        y_predicts = []
        
        y_pred_cv = []
        y_test_cv = []
        r_squared_cv = []
        mse_cv = []
        rmse_cv = []
        
        results_labels_df = pd.DataFrame(
            columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'ID']
        )
        
        # Initialize coefficient storage
        coef_array = np.zeros([len(lista_vars) + 1, n_splits])
        
        iter_split = 0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=iteration)
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train Gradient Boosting model
            if params_b == -1:
                model = GradientBoostingRegressor(random_state=42)
            else:
                model = GradientBoostingRegressor(random_state=42, **params_b)
            
            model.fit(X_train, y_train)
            
            # Store feature importances (intercept = NaN)
            coef_array[0, iter_split] = np.nan
            coef_array[1::, iter_split] = model.feature_importances_
            
            # Make predictions
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            # Calculate gaps
            gap_test = y_pred_test - y_test
            gap_train = y_pred_train - y_train
            
            # Correct gaps using GLM
            model_gap = GLM(gap_train, y_train, family=families.Gaussian())
            results_gap = model_gap.fit()
            corrected_gap = gap_test - results_gap.predict(gap_test)
            
            # Store results
            y_labels.extend(list(y_test))
            y_predicts.extend(list(y_pred_test))
            y_pred_cv.extend(list(y_pred_test))
            y_test_cv.extend(list(y_test))
            
            # Calculate metrics
            r2_fold = r2_score(y_test, y_pred_test)
            mse_fold = np.round(mean_squared_error(y_test, y_pred_test), 6)
            rmse_fold = np.round(math.sqrt(mean_squared_error(y_test, y_pred_test)), 6)
            
            r_squared_cv.append(r2_fold)
            mse_cv.append(mse_fold)
            rmse_cv.append(rmse_fold)
            
            # Create results dataframe
            result = np.column_stack((y_test, y_pred_test, gap_test, corrected_gap))
            temp_df = pd.DataFrame(
                result,
                columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected']
            )
            temp_df['ID'] = X_test.index
            
            results_labels_df = pd.concat(
                [results_labels_df, temp_df],
                ignore_index=True
            )
            
            iter_split += 1
        
        # Aggregate cross-validation results
        n = len(y_predicts)
        p = X.shape[1]
        r_squared = r2_score(y_labels, y_predicts)
        r_squared_list.append(r_squared)
        
        # Calculate diagnostic metrics
        mda = mean_directional_accuracy(y_labels, y_predicts)
        mae = mean_absolute_error(y_labels, y_predicts)
        
        # Adjusted R²
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        # MSE and RMSE
        mse = np.round(mean_squared_error(y_labels, y_predicts), 6)
        rmse = np.round(math.sqrt(mean_squared_error(y_labels, y_predicts)), 6)
        
        # F-statistic (approximate for non-linear model)
        F = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if (1 - r_squared) > 0 else np.inf
        p_value = np.round(scipy.stats.f.sf(F, p, n - p - 1), 15)
        F2 = r_squared / (1 - r_squared) if (1 - r_squared) > 0 else np.inf
        
        # Calculate mean coefficients and standard deviations
        coef_array_mean = np.zeros([len(lista_vars) + 1, 1])
        coef_array_std = np.zeros([len(lista_vars) + 1, 1])
        
        for j in range(len(lista_vars) + 1):
            coef_array_mean[j] = coef_array[j, :].mean()
            coef_array_std[j] = coef_array[j, :].std()
        
        # Build coefficient dataframe (using 'Model' instead of '_intercept' for non-linear)
        coef_df = pd.DataFrame(
            index=['Model'] + lista_vars,
            columns=['Feature_Importance_mean', 'Feature_Importance_std']
        )
        
        coef_df['Feature_Importance_mean'] = coef_array_mean
        coef_df['Feature_Importance_std'] = coef_array_std
        
        # Add model statistics to Model row (non-linear, so no intercept)
        coef_df.loc['Model', 'R2'] = r_squared
        coef_df.loc['Model', 'R2 adj'] = r_squared_adj
        coef_df.loc['Model', 'R2 [+-]'] = 1 * np.std(r_squared_cv)
        coef_df.loc['Model', 'F2'] = F2
        coef_df.loc['Model', 'mse'] = mse
        coef_df.loc['Model', 'mse [+-]'] = 1 * np.std(mse_cv)
        coef_df.loc['Model', 'rmse'] = rmse
        coef_df.loc['Model', 'rmse [+-]'] = 1 * np.std(rmse_cv)
        coef_df.loc['Model', 'outcome var'] = np.var(y)
        coef_df.loc['Model', 'F'] = F
        coef_df.loc['Model', 'F-p_value'] = p_value
        coef_df.loc['Model', 'MDE'] = mda
        coef_df.loc['Model', 'MAE'] = mae
    
    if logger:
        logger.add_message(f"GBR model trained successfully with R² = {r_squared:.4f}")
    
    results_labels_df['y_pred_corrected'] = (
        results_labels_df['y_labels'] + results_labels_df['GAP_corrected']
    )
    results_labels_df = results_labels_df[[
        'ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected'
    ]]
    
    if shaps_comp:
        try:
            import shap
            if params_b == -1:
                model_final = GradientBoostingRegressor(random_state=42)
            else:
                model_final = GradientBoostingRegressor(random_state=42, **params_b)
            
            model_final.fit(X, y)
            explainer = shap.Explainer(model_final, X)
            
            if logger:
                logger.add_message("SHAP explainer computed")
                logger.save('gbr_regression')
            
            return [coef_df, r_squared_list, results_labels_df, explainer]
        except ImportError:
            if logger:
                logger.add_warning("SHAP library not installed")
            print("Warning: SHAP library not installed. Returning without explainer.")
            if logger:
                logger.save('gbr_regression')
            return [coef_df, r_squared_list, results_labels_df]
    else:
        if logger:
            logger.save('gbr_regression')
        return [coef_df, r_squared_list, results_labels_df]
