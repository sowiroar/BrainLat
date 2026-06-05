"""
Statistical utility functions for regression analysis.

This module provides statistical calculations for regression models including:
- Coefficient calculations (standard errors, t-values, p-values)
- Model performance metrics (R², F-tests)
- Additional diagnostic measures (MDE, MAE)
- Hosmer-Lemeshow test for classification models
"""

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, combine_pvalues
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings("ignore")


def coef_se(X, y_true, y_pred):
    """
    Calculate standard errors of coefficients for OLS regression.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (without intercept)
    y_true : array-like, shape (n_samples,)
        True target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns
    -------
    se : array, shape (n_features + 1,)
        Standard errors for intercept and all coefficients
    """
    n = X.shape[0]
    
    # Add intercept column
    X_with_intercept = np.hstack((np.ones((n, 1)), np.matrix(X)))
    
    # Calculate standard error of regression
    mse = mean_squared_error(y_true, y_pred)
    
    # Standard errors from covariance matrix diagonal
    se_matrix = scipy.linalg.sqrtm(
        mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    )
    
    return np.diagonal(se_matrix)


def coef_tval(coef_array_mean, X, y_true, y_pred):
    """
    Calculate t-values for regression coefficients.
    
    Parameters
    ----------
    coef_array_mean : array-like, shape (n_features + 1, 1)
        Mean coefficient estimates (intercept and features)
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y_true : array-like, shape (n_samples,)
        True target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns
    -------
    t_values : array, shape (n_features + 1,)
        T-values for all coefficients
    """
    se = coef_se(X, y_true, y_pred)
    
    # Intercept t-value
    t_intercept = np.array(coef_array_mean[0][0] / se[0])
    
    # Feature t-values
    t_features = np.array(coef_array_mean[1::].flatten() / se[1:])
    
    return np.append(t_intercept, t_features)


def coef_pval(coef_array_mean, X, y_true, y_pred):
    """
    Calculate p-values for regression coefficients using t-test.
    
    Parameters
    ----------
    coef_array_mean : array-like, shape (n_features + 1, 1)
        Mean coefficient estimates
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y_true : array-like, shape (n_samples,)
        True target values
    y_pred : array-like, shape (n_samples,)
        Predicted target values
    
    Returns
    -------
    p_values : array, shape (n_features + 1,)
        Two-tailed p-values for all coefficients
    """
    n = X.shape[0]
    t_vals = coef_tval(coef_array_mean, X, y_true, y_pred)
    p_vals = 2 * (1 - scipy.stats.t.cdf(np.abs(t_vals), n - 1))
    
    return p_vals


def calculate_r_squared(y_true, y_pred):
    """
    Calculate R² (coefficient of determination).
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns
    -------
    r2 : float
        R² value between 0 and 1
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2


def calculate_adjusted_r_squared(y_true, y_pred, n_features):
    """
    Calculate adjusted R² accounting for number of features.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_features : int
        Number of features in the model
    
    Returns
    -------
    r2_adj : float
        Adjusted R² value
    """
    n = len(y_true)
    r2 = calculate_r_squared(y_true, y_pred)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    return r2_adj


def calculate_f_statistic(y_true, y_pred, n_features):
    """
    Calculate F-statistic for overall model significance.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_features : int
        Number of features
    
    Returns
    -------
    f_stat : float
        F-statistic value
    p_value : float
        P-value for the F-statistic
    """
    n = len(y_true)
    r2 = calculate_r_squared(y_true, y_pred)
    
    if r2 >= 1.0 or (1 - r2) <= 0:
        return np.inf, 0.0
    
    f_stat = (r2 / n_features) / ((1 - r2) / (n - n_features - 1))
    p_value = scipy.stats.f.sf(f_stat, n_features, n - n_features - 1)
    
    return f_stat, p_value


def mean_directional_accuracy(y_true, y_pred):
    """
    Calculate Mean Directional Accuracy (MDA).
    
    Measures the proportion of correctly predicted directional changes
    in the target variable.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns
    -------
    mda : float
        Mean directional accuracy (between -1 and 1)
    """
    differences = np.array(y_pred) - np.array(y_true)
    signs = np.sign(differences)
    mda = np.mean(signs)
    
    return mda


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns
    -------
    mae : float
        Mean absolute error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    absolute_errors = np.abs(y_pred - y_true)
    mae = np.mean(absolute_errors)
    
    return mae


def mean_squared_error_custom(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns
    -------
    mse : float
        Mean squared error
    """
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns
    -------
    rmse : float
        Root mean squared error
    """
    return np.sqrt(mean_squared_error_custom(y_true, y_pred))


def hosmer_lemeshow_test(observed, predicted_probs, n_groups=10):
    """
    Single Hosmer-Lemeshow test on a set of predictions.
    
    Parameters
    ----------
    observed : array-like
        Binary observed outcomes (0 or 1)
    predicted_probs : array-like
        Predicted probabilities
    n_groups : int, default=10
        Number of bins to partition predictions
    
    Returns
    -------
    hl_statistic : float
        Hosmer-Lemeshow chi-square statistic
    p_value : float
        P-value for the test
    """
    df_hl = pd.DataFrame({
        'observed': observed,
        'predicted_probability': predicted_probs
    })
    df_hl['group'] = pd.qcut(df_hl['predicted_probability'], n_groups, duplicates='drop')
    
    hl_table = df_hl.groupby('group').apply(
        lambda x: pd.Series({
            'observed': x['observed'].sum(),
            'expected': x['predicted_probability'].sum(),
            'total': len(x)
        })
    )

    hl_table['observed_neg'] = hl_table['total'] - hl_table['observed']
    hl_table['expected_neg'] = hl_table['total'] - hl_table['expected']

    hl_statistic = (((hl_table['observed'] - hl_table['expected']) ** 2) / hl_table['expected'] +
                    ((hl_table['observed_neg'] - hl_table['expected_neg']) ** 2) / hl_table['expected_neg']).sum()

    df = n_groups - 2
    p_value = 1 - chi2.cdf(hl_statistic, df)
    
    return hl_statistic, p_value


def hosmer_lemeshow(observed, predicted_probs, n_repeats=100, n_groups=10, test_size=0.2, random_state=None):
    """
    Hosmer-Lemeshow test with repeated stratified subsampling.
    
    Performs the Hosmer-Lemeshow goodness-of-fit test with multiple resampling
    iterations using stratified shuffle split. Combines p-values using Fisher's method.
    
    Parameters
    ----------
    observed : array-like
        Binary observed outcomes (0 or 1)
    predicted_probs : array-like
        Predicted probabilities from a classification model
    n_repeats : int, default=100
        Number of StratifiedShuffleSplit iterations
    n_groups : int, default=10
        Number of bins for Hosmer-Lemeshow test
    test_size : float, default=0.2
        Proportion of samples for each test split
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'hl_stat_mean': Mean HL statistic across iterations
        - 'fisher_p_value': Combined p-value using Fisher's method
        - 'p_from_mean_stat': P-value from mean statistic
        - 'all_hl_stats': List of HL statistics from each iteration
        - 'all_p_values': List of p-values from each iteration
    
    Notes
    -----
    Requires: pip install scikit-optimize (for cross-validation utilities)
    
    References
    ----------
    Hosmer, D. W., & Lemeshow, S. (2000). Applied logistic regression. Wiley.
    """
    hl_stats = []
    p_values = []

    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state)

    observed = np.array(observed)
    predicted_probs = np.array(predicted_probs)

    for train_index, test_index in sss.split(predicted_probs, observed):
        y_test = observed[test_index]
        prob_test = predicted_probs[test_index]
        try:
            hl_stat, p_val = hosmer_lemeshow_test(y_test, prob_test, n_groups)
            hl_stats.append(hl_stat)
            p_values.append(p_val)
        except Exception:
            continue  

    hl_stat_mean = np.mean(hl_stats)
    fisher_stat, fisher_p_value = combine_pvalues(p_values, method='fisher')
    
    df = n_groups - 2
    p_from_mean_stat = 1 - chi2.cdf(hl_stat_mean, df)

    return {
        'hl_stat_mean': hl_stat_mean,
        'fisher_p_value': fisher_p_value,
        'p_from_mean_stat': p_from_mean_stat,
        'all_hl_stats': hl_stats,
        'all_p_values': p_values
    }
