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


def Regression_GBR_nested(X, y, outer_splits=10, inner_splits=5, shaps_comp=False, log=False):
    """
    Nested Cross Validation with Bayesian Search for Gradient Boosting Regressor.
    
    Tunes hyperparameters inside an inner CV loop and evaluates model and computes
    gap correction using linear regression of training gaps on training age in the outer loop.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series or array-like
        Target age/variable
    outer_splits : int, default=10
        Number of outer KFold splits
    inner_splits : int, default=5
        Number of inner KFold splits for Bayesian hyperparameter tuning
    shaps_comp : bool, default=False
        Whether to compute SHAP values
    log : bool, default=False
        Whether to generate logs and quality reports
        
    Returns
    -------
    [coef_df, r_squared_mean, results_labels_df, outer_cv_split] (or with explainer if shaps_comp=True)
    """
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from scipy.stats import linregress
    from sklearn.metrics import mean_absolute_error as sklearn_mae
    
    # Initialize logger
    logger = Logger() if log else None
    if logger:
        logger.add_message("Starting Nested GBR CV analysis...")
        
    # Data diagnostics
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="GBR Nested Input")
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/gbr_nested_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check: {report['n_samples']} samples, {report['n_features']} features")
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)
        
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    scaler = MinMaxScaler((0.05, 0.95))
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'max_depth': Integer(1, 5),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    }
    
    y_labels = []
    y_predicts = []
    r_squared_l = []
    mse_l = []
    rmse_l = []
    
    results_labels_df = pd.DataFrame(columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'ID'])
    coef_array = np.zeros([X.shape[1] + 1, outer_splits])
    lista_vars = list(X.columns)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        msg = f"Fold: {fold}"
        if logger:
            logger.add_message(msg)
        else:
            print(msg)
            
        scaling_data = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(scaling_data, columns=X_train.columns, index=X_train.index)
        
        scaling_data = scaler.transform(X_test)
        X_test = pd.DataFrame(scaling_data, columns=X_test.columns, index=X_test.index)
        
        model = GradientBoostingRegressor(random_state=42)
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_grid,
            n_iter=3,  # small value for stability/performance in CV
            scoring='r2',
            cv=inner_cv,
            n_jobs=-1,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        gap_test = y_pred - y_test
        gap_train = best_model.predict(X_train) - y_train
        
        slope, intercept, _, _, _ = linregress(y_train, gap_train)
        corrected_gap = gap_test - (slope * y_test + intercept)
        
        result = np.column_stack((y_test, y_pred, gap_test, corrected_gap))
        temp_df = pd.DataFrame(result, columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected'])
        temp_df['ID'] = X_test.index
        results_labels_df = pd.concat([results_labels_df, temp_df], ignore_index=True)
        
        y_labels.extend(y_test)
        y_predicts.extend(y_pred)
        
        r_squared_l.append(r2_score(y_test, y_pred))
        mse_l.append(mean_squared_error(y_test, y_pred))
        rmse_l.append(math.sqrt(mse_l[-1]))
        
        coef_array[0, fold] = np.nan
        coef_array[1:, fold] = best_model.feature_importances_
        
    y_labels = np.array(y_labels)
    y_predicts = np.array(y_predicts)
    
    r_squared = r2_score(y_labels, y_predicts)
    r_squared_mean = np.mean(r_squared_l)
    n = len(y_labels)
    p = X.shape[1]
    k = p - 1
    
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mse = np.mean(mse_l)
    rmse = np.mean(rmse_l)
    mae = sklearn_mae(y_labels, y_predicts)
    F = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if (1 - r_squared) > 0 else np.inf
    p_value = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15) if F != np.inf else 0.0
    F2 = r_squared / (1 - r_squared) if (1 - r_squared) > 0 else np.inf
    
    coef_df = pd.DataFrame(index=['_intercept'] + lista_vars,
                           columns=['Estimate mean', 'Estimate std', 't value', 'p value'])
    
    coef_df['Estimate mean'] = coef_array.mean(axis=1)
    coef_df['Estimate std'] = coef_array.std(axis=1)
    
    coef_df.loc['_intercept', 'R2'] = r_squared
    coef_df.loc['_intercept', 'R2 adj'] = r_squared_adj
    coef_df.loc['_intercept', 'R2 [+-]'] = np.std(r_squared_l)
    coef_df.loc['_intercept', 'F2'] = F2
    coef_df.loc['_intercept', 'mse'] = mse
    coef_df.loc['_intercept', 'mse [+-]'] = np.std(mse_l)
    coef_df.loc['_intercept', 'rmse'] = rmse
    coef_df.loc['_intercept', 'rmse [+-]'] = np.std(rmse_l)
    coef_df.loc['_intercept', 'outcome var'] = np.var(y)
    coef_df.loc['_intercept', 'F'] = F
    coef_df.loc['_intercept', 'F-p_value'] = p_value
    coef_df.loc['_intercept', 'MDE'] = mean_directional_accuracy(y_labels, y_predicts)
    coef_df.loc['_intercept', 'MAE'] = mae
    
    results_labels_df['y_pred_corrected'] = results_labels_df['y_labels'] + results_labels_df['GAP_corrected']
    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected']]
    
    if logger:
        logger.add_message(f"Nested GBR CV completed with R² = {r_squared:.4f}")
        logger.save('gbr_nested_regression')
        
    if shaps_comp:
        import shap
        explainer = shap.Explainer(best_model, X)
        return [coef_df, r_squared_mean, results_labels_df, explainer, outer_cv.split(X)]
    else:
        return [coef_df, r_squared_mean, results_labels_df, outer_cv.split(X)]


def Regression_GBR_leave_one_site_out(X, y, groups, inner_splits=5, shaps_comp=False, log=False):
    """
    Leave-One-Site-Out (LOGO) cross validation for GBR.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series or array-like
        Target variable
    groups : pd.Series
        Site/Group indicator for each sample
    inner_splits : int, default=5
        Inner KFold splits for hyperparameter tuning
    shaps_comp : bool, default=False
        Whether to compute SHAP values
    log : bool, default=False
        Whether to generate logs
        
    Returns
    -------
    [coef_df, r_squared_mean, results_labels_df, outer_cv_split] (or with explainer if shaps_comp=True)
    """
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from scipy.stats import linregress
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import mean_absolute_error as sklearn_mae
    
    # Initialize logger
    logger = Logger() if log else None
    if logger:
        logger.add_message("Starting LOGO GBR CV analysis...")
        
    # Diagnostics
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="GBR LOGO Input")
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/gbr_logo_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check: {report['n_samples']} samples, {report['n_features']} features")
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)
        
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    groups = pd.Series(groups) if not isinstance(groups, pd.Series) else groups
    
    scaler = MinMaxScaler((0.05, 0.95))
    outer_cv = LeaveOneGroupOut()
    
    param_grid = {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'max_depth': Integer(1, 5),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    }
    
    y_labels = []
    y_predicts = []
    r_squared_l = []
    mse_l = []
    rmse_l = []
    
    results_labels_df = pd.DataFrame(columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'ID'])
    coef_array = np.zeros([X.shape[1] + 1, len(np.unique(groups))])
    lista_vars = list(X.columns)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        vc = groups.iloc[test_idx].value_counts()
        country_name = vc.index[0]
        n_obs = vc.values[0]
        msg = f"Fold: {fold} {country_name} - {n_obs} samples"
        if logger:
            logger.add_message(msg)
        else:
            print(msg)
            
        scaling_data = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(scaling_data, columns=X_train.columns, index=X_train.index)
        
        scaling_data = scaler.transform(X_test)
        X_test = pd.DataFrame(scaling_data, columns=X_test.columns, index=X_test.index)
        
        model = GradientBoostingRegressor(random_state=42)
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_grid,
            n_iter=3,
            scoring='r2',
            cv=inner_cv,
            n_jobs=-1,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        gap_test = y_pred - y_test
        gap_train = best_model.predict(X_train) - y_train
        
        slope, intercept, _, _, _ = linregress(y_train, gap_train)
        corrected_gap = gap_test - (slope * y_test + intercept)
        
        result = np.column_stack((y_test, y_pred, gap_test, corrected_gap))
        temp_df = pd.DataFrame(result, columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected'])
        temp_df['ID'] = X_test.index
        results_labels_df = pd.concat([results_labels_df, temp_df], ignore_index=True)
        
        y_labels.extend(y_test)
        y_predicts.extend(y_pred)
        
        r_squared_l.append(r2_score(y_test, y_pred))
        mse_l.append(mean_squared_error(y_test, y_pred))
        rmse_l.append(math.sqrt(mse_l[-1]))
        
        coef_array[0, fold] = np.nan
        coef_array[1:, fold] = best_model.feature_importances_
        
    y_labels = np.array(y_labels)
    y_predicts = np.array(y_predicts)
    
    r_squared = r2_score(y_labels, y_predicts)
    r_squared_mean = np.mean(r_squared_l)
    n = len(y_labels)
    p = X.shape[1]
    k = p - 1
    
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mse = np.mean(mse_l)
    rmse = np.mean(rmse_l)
    mae = sklearn_mae(y_labels, y_predicts)
    F = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if (1 - r_squared) > 0 else np.inf
    p_value = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15) if F != np.inf else 0.0
    F2 = r_squared / (1 - r_squared) if (1 - r_squared) > 0 else np.inf
    
    coef_df = pd.DataFrame(index=['_intercept'] + lista_vars,
                           columns=['Estimate mean', 'Estimate std', 't value', 'p value'])
    
    coef_df['Estimate mean'] = coef_array.mean(axis=1)
    coef_df['Estimate std'] = coef_array.std(axis=1)
    
    coef_df.loc['_intercept', 'R2'] = r_squared
    coef_df.loc['_intercept', 'R2 adj'] = r_squared_adj
    coef_df.loc['_intercept', 'R2 [+-]'] = np.std(r_squared_l)
    coef_df.loc['_intercept', 'F2'] = F2
    coef_df.loc['_intercept', 'mse'] = mse
    coef_df.loc['_intercept', 'mse [+-]'] = np.std(mse_l)
    coef_df.loc['_intercept', 'rmse'] = rmse
    coef_df.loc['_intercept', 'rmse [+-]'] = np.std(rmse_l)
    coef_df.loc['_intercept', 'outcome var'] = np.var(y)
    coef_df.loc['_intercept', 'F'] = F
    coef_df.loc['_intercept', 'F-p_value'] = p_value
    coef_df.loc['_intercept', 'MAE'] = mae
    
    results_labels_df['y_pred_corrected'] = results_labels_df['y_labels'] + results_labels_df['GAP_corrected']
    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected']]
    
    if logger:
        logger.add_message(f"LOGO GBR completed with R² = {r_squared:.4f}")
        logger.save('gbr_logo_regression')
        
    if shaps_comp:
        import shap
        explainer = shap.Explainer(best_model, X)
        return [coef_df, r_squared_mean, results_labels_df, explainer, outer_cv.split(X)]
    else:
        return [coef_df, r_squared_mean, results_labels_df, outer_cv.split(X, y, groups=groups)]


def Regression_GBR_nested_heldout(X, y, X_heldout=None, y_heldout=None, outer_splits=10, inner_splits=5, shaps_comp=False, log=False):
    """
    Nested CV with evaluation on an independent heldout dataset.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series or array-like
        Target variable
    X_heldout : pd.DataFrame, optional
        Held-out features dataframe
    y_heldout : pd.Series or array-like, optional
        Held-out target variable
    outer_splits : int, default=10
        Outer splits
    inner_splits : int, default=5
        Inner splits
    shaps_comp : bool, default=False
        Compute SHAPs
    log : bool, default=False
        Generate logs
        
    Returns
    -------
    [coef_df, r_squared_mean, results_labels_df, heldout_all_df, outer_cv_split] (or with explainer)
    """
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from scipy.stats import linregress
    from sklearn.metrics import mean_absolute_error as sklearn_mae
    
    # Initialize logger
    logger = Logger() if log else None
    if logger:
        logger.add_message("Starting GBR CV with Held-out analysis...")
        
    # Diagnostics
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="GBR Nested-Heldout Input")
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/gbr_nested_heldout_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check: {report['n_samples']} samples, {report['n_features']} features")
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)
        
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    scaler = MinMaxScaler((0.05, 0.95))
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'max_depth': Integer(1, 5),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    }
    
    y_labels = []
    y_predicts = []
    r_squared_l = []
    mse_l = []
    rmse_l = []
    
    results_labels_df = pd.DataFrame(columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'ID'])
    heldout_all_df = pd.DataFrame(columns=['ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected'])
    coef_array = np.zeros([X.shape[1] + 1, outer_splits])
    lista_vars = list(X.columns)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        msg = f"Fold: {fold}"
        if logger:
            logger.add_message(msg)
        else:
            print(msg)
            
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        model = GradientBoostingRegressor(random_state=42)
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_grid,
            n_iter=3,
            scoring='r2',
            cv=inner_cv,
            n_jobs=-1,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        gap_test = y_pred - y_test
        gap_train = best_model.predict(X_train) - y_train
        
        slope, intercept, _, _, _ = linregress(y_train, gap_train)
        corrected_gap = gap_test - (slope * y_test + intercept)
        
        temp_df = pd.DataFrame({
            'y_labels': y_test,
            'y_pred': y_pred,
            'GAP': gap_test,
            'GAP_corrected': corrected_gap,
            'ID': X_test.index
        })
        results_labels_df = pd.concat([results_labels_df, temp_df], ignore_index=True)
        
        y_labels.extend(y_test)
        y_predicts.extend(y_pred)
        
        r_squared_l.append(r2_score(y_test, y_pred))
        mse_l.append(mean_squared_error(y_test, y_pred))
        rmse_l.append(math.sqrt(mse_l[-1]))
        
        coef_array[0, fold] = np.nan
        coef_array[1:, fold] = best_model.feature_importances_
        
        # Predict on heldout
        if X_heldout is not None and y_heldout is not None:
            # fit scaler to heldout separately or use training scaler? 
            # Original code fit on X_heldout: scaler.fit(X_heldout)
            scaler_ho = MinMaxScaler((0.05, 0.95))
            X_heldout_scaled = scaler_ho.fit_transform(X_heldout)
            y_pred_heldout = best_model.predict(X_heldout_scaled)
            gap_heldout = y_pred_heldout - y_heldout
            gap_heldout_corrected = gap_heldout - (slope * y_heldout + intercept)
            y_pred_corrected_heldout = y_heldout + gap_heldout_corrected
            
            temp_heldout_df = pd.DataFrame({
                'ID': X_heldout.index,
                'y_labels': y_heldout,
                'y_pred': y_pred_heldout,
                'GAP': gap_heldout,
                'GAP_corrected': gap_heldout_corrected,
                'y_pred_corrected': y_pred_corrected_heldout
            })
            heldout_all_df = pd.concat([heldout_all_df, temp_heldout_df], ignore_index=True)
            
    y_labels = np.array(y_labels)
    y_predicts = np.array(y_predicts)
    
    r_squared = r2_score(y_labels, y_predicts)
    r_squared_mean = np.mean(r_squared_l)
    n = len(y_labels)
    p = X.shape[1]
    k = p - 1
    
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mse = np.mean(mse_l)
    rmse = np.mean(rmse_l)
    mae = sklearn_mae(y_labels, y_predicts)
    F = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if (1 - r_squared) > 0 else np.inf
    p_value = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15) if F != np.inf else 0.0
    F2 = r_squared / (1 - r_squared) if (1 - r_squared) > 0 else np.inf
    
    # Calculate MDE on heldout if available
    def mean_directional_accuracy_heldout(y_true, y_pred):
        return np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        
    mde_val = mean_directional_accuracy_heldout(y_labels, y_predicts)
    
    coef_df = pd.DataFrame(index=['_intercept'] + lista_vars,
                           columns=['Estimate mean', 'Estimate std', 't value', 'p value'])
    
    coef_df['Estimate mean'] = coef_array.mean(axis=1)
    coef_df['Estimate std'] = coef_array.std(axis=1)
    
    coef_df.loc['_intercept', 'R2'] = r_squared
    coef_df.loc['_intercept', 'R2 adj'] = r_squared_adj
    coef_df.loc['_intercept', 'R2 [+-]'] = np.std(r_squared_l)
    coef_df.loc['_intercept', 'F2'] = F2
    coef_df.loc['_intercept', 'mse'] = mse
    coef_df.loc['_intercept', 'mse [+-]'] = np.std(mse_l)
    coef_df.loc['_intercept', 'rmse'] = rmse
    coef_df.loc['_intercept', 'rmse [+-]'] = np.std(rmse_l)
    coef_df.loc['_intercept', 'outcome var'] = np.var(y)
    coef_df.loc['_intercept', 'F'] = F
    coef_df.loc['_intercept', 'F-p_value'] = p_value
    coef_df.loc['_intercept', 'MDE'] = mde_val
    coef_df.loc['_intercept', 'MAE'] = mae
    
    results_labels_df['y_pred_corrected'] = results_labels_df['y_labels'] + results_labels_df['GAP_corrected']
    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected']]
    
    if logger:
        logger.add_message(f"Nested Heldout GBR CV completed with R² = {r_squared:.4f}")
        logger.save('gbr_nested_heldout_regression')
        
    if shaps_comp:
        import shap
        explainer = shap.Explainer(best_model, X)
        return [coef_df, r_squared_mean, results_labels_df, heldout_all_df, explainer, outer_cv.split(X)]
    else:
        return [coef_df, r_squared_mean, results_labels_df, heldout_all_df, outer_cv.split(X)]


def Regression_GBR_nested_heldout_mean_hyp(X, y, X_heldout=None, y_heldout=None, outer_splits=10, inner_splits=5, shaps_comp=False, log=False):
    """
    Nested CV that averages the best hyperparameters and fits a final model to make predictions.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series or array-like
        Target variable
    X_heldout : pd.DataFrame, optional
        Held-out features dataframe
    y_heldout : pd.Series or array-like, optional
        Held-out target variable
    outer_splits : int, default=10
        Outer splits
    inner_splits : int, default=5
        Inner splits
    shaps_comp : bool, default=False
        Compute SHAPs
    log : bool, default=False
        Generate logs
        
    Returns
    -------
    [coef_df, r_squared_mean, results_labels_df, heldout_all_df, outer_cv_split] (or with explainer)
    """
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from scipy.stats import linregress
    from sklearn.metrics import mean_absolute_error as sklearn_mae
    
    # Initialize logger
    logger = Logger() if log else None
    if logger:
        logger.add_message("Starting GBR CV with Held-out and Averaged Hyperparameters analysis...")
        
    # Diagnostics
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="GBR Nested-Heldout-Mean-Hyp Input")
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/gbr_nested_heldout_mean_hyp_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check: {report['n_samples']} samples, {report['n_features']} features")
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)
        
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    scaler = MinMaxScaler((0.05, 0.95))
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    
    param_grid = {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'max_depth': Integer(1, 5),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    }
    
    y_labels = []
    y_predicts = []
    r_squared_l = []
    mse_l = []
    rmse_l = []
    
    results_labels_df = pd.DataFrame(columns=['y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'ID'])
    coef_array = np.zeros([X.shape[1] + 1, outer_splits])
    lista_vars = list(X.columns)
    
    best_params_list = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        msg = f"Fold: {fold}"
        if logger:
            logger.add_message(msg)
        else:
            print(msg)
            
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        model = GradientBoostingRegressor(random_state=42)
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_grid,
            n_iter=3,
            scoring='r2',
            cv=inner_cv,
            n_jobs=-1,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        best_model = bayes_search.best_estimator_
        
        best_params_list.append(bayes_search.best_params_)
        
        y_pred = best_model.predict(X_test)
        gap_test = y_pred - y_test
        gap_train = best_model.predict(X_train) - y_train
        
        slope, intercept, _, _, _ = linregress(y_train, gap_train)
        corrected_gap = gap_test - (slope * y_test + intercept)
        
        temp_df = pd.DataFrame({
            'y_labels': y_test,
            'y_pred': y_pred,
            'GAP': gap_test,
            'GAP_corrected': corrected_gap,
            'ID': X_test.index
        })
        results_labels_df = pd.concat([results_labels_df, temp_df], ignore_index=True)
        
        y_labels.extend(y_test)
        y_predicts.extend(y_pred)
        
        r_squared_l.append(r2_score(y_test, y_pred))
        mse_l.append(mean_squared_error(y_test, y_pred))
        rmse_l.append(math.sqrt(mse_l[-1]))
        
        coef_array[0, fold] = np.nan
        coef_array[1:, fold] = best_model.feature_importances_
        
    # Average hyperparameters
    def average_params(params_list):
        df_p = pd.DataFrame(params_list)
        averaged = {}
        for col in df_p.columns:
            if df_p[col].dtype == 'float' or df_p[col].dtype == 'int':
                averaged[col] = df_p[col].mean()
                if df_p[col].dtype == 'int':
                    averaged[col] = int(round(averaged[col]))
            else:
                averaged[col] = df_p[col].mode()[0]
        return averaged
        
    avg_params = average_params(best_params_list)
    
    heldout_all_df = pd.DataFrame(columns=['ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected'])
    
    # Train final model on entire X and predict heldout
    if X_heldout is not None and y_heldout is not None:
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        scaler_ho = MinMaxScaler((0.05, 0.95))
        X_heldout_scaled = scaler_ho.fit_transform(X_heldout)
        
        final_model = GradientBoostingRegressor(random_state=42, **avg_params)
        final_model.fit(X_scaled, y)
        
        y_pred_heldout = final_model.predict(X_heldout_scaled)
        gap_heldout = y_pred_heldout - y_heldout
        
        gap_train_full = final_model.predict(X_scaled) - y
        slope, intercept, _, _, _ = linregress(y, gap_train_full)
        gap_heldout_corrected = gap_heldout - (slope * y_heldout + intercept)
        y_pred_corrected_heldout = y_heldout + gap_heldout_corrected
        
        heldout_all_df = pd.DataFrame({
            'ID': X_heldout.index,
            'y_labels': y_heldout,
            'y_pred': y_pred_heldout,
            'GAP': gap_heldout,
            'GAP_corrected': gap_heldout_corrected,
            'y_pred_corrected': y_pred_corrected_heldout
        })
        
    y_labels = np.array(y_labels)
    y_predicts = np.array(y_predicts)
    
    r_squared = r2_score(y_labels, y_predicts)
    r_squared_mean = np.mean(r_squared_l)
    n = len(y_labels)
    p = X.shape[1]
    k = p - 1
    
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mse = np.mean(mse_l)
    rmse = np.mean(rmse_l)
    mae = sklearn_mae(y_labels, y_predicts)
    F = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if (1 - r_squared) > 0 else np.inf
    p_value = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15) if F != np.inf else 0.0
    F2 = r_squared / (1 - r_squared) if (1 - r_squared) > 0 else np.inf
    
    # MDE calculation
    def mean_directional_accuracy_heldout(y_true, y_pred):
        return np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        
    mde_val = mean_directional_accuracy_heldout(y_labels, y_predicts)
    
    coef_df = pd.DataFrame(index=['_intercept'] + lista_vars,
                           columns=['Estimate mean', 'Estimate std', 't value', 'p value'])
    
    coef_df['Estimate mean'] = coef_array.mean(axis=1)
    coef_df['Estimate std'] = coef_array.std(axis=1)
    
    coef_df.loc['_intercept', 'R2'] = r_squared
    coef_df.loc['_intercept', 'R2 adj'] = r_squared_adj
    coef_df.loc['_intercept', 'R2 [+-]'] = np.std(r_squared_l)
    coef_df.loc['_intercept', 'F2'] = F2
    coef_df.loc['_intercept', 'mse'] = mse
    coef_df.loc['_intercept', 'mse [+-]'] = np.std(mse_l)
    coef_df.loc['_intercept', 'rmse'] = rmse
    coef_df.loc['_intercept', 'rmse [+-]'] = np.std(rmse_l)
    coef_df.loc['_intercept', 'outcome var'] = np.var(y)
    coef_df.loc['_intercept', 'F'] = F
    coef_df.loc['_intercept', 'F-p_value'] = p_value
    coef_df.loc['_intercept', 'MDE'] = mde_val
    coef_df.loc['_intercept', 'MAE'] = mae
    
    results_labels_df['y_pred_corrected'] = results_labels_df['y_labels'] + results_labels_df['GAP_corrected']
    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred', 'GAP', 'GAP_corrected', 'y_pred_corrected']]
    
    if logger:
        logger.add_message(f"Averaged Hyperparameters GBR completed with R² = {r_squared:.4f}")
        logger.save('gbr_nested_heldout_mean_hyp_regression')
        
    if shaps_comp:
        import shap
        explainer = shap.Explainer(final_model if X_heldout is not None else best_model, X)
        return [coef_df, r_squared_mean, results_labels_df, heldout_all_df, explainer, outer_cv.split(X)]
    else:
        return [coef_df, r_squared_mean, results_labels_df, heldout_all_df, outer_cv.split(X)]

