"""
Utility tools for feature analysis and data processing.

This module provides utilities for:
- Feature direction computation
- Variance Inflation Factor (VIF) calculation
- Data scaling and standardization
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_directions(X, y, bonferroni_correction=True):
    """
    Compute feature directions using univariate OLS regression.
    
    For each feature, fit a simple linear regression with only that feature
    and extract the direction and significance of the relationship.
    
    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    bonferroni_correction : bool, default=True
        Whether to apply Bonferroni correction to p-values
    
    Returns
    -------
    directions_df : pd.DataFrame
        DataFrame with columns ['Feature', 'coef', 'p_value']
        Sorted by absolute coefficient value
    
    Examples
    --------
    >>> import pandas as pd
    >>> from brainlat.tools import compute_directions
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    >>> y = np.random.randn(100)
    >>> dirs = compute_directions(X, y)
    """
    df_directions = pd.DataFrame(columns=['Feature', 'coef', 'p_value'])
    
    n_features = len(X.columns)
    correction_factor = n_features if bonferroni_correction else 1
    
    for feature_name in X.columns:
        X_single = X[[feature_name]]
        
        # Scale the feature
        scaler = MinMaxScaler((0.05, 0.95))
        X_scaled = scaler.fit_transform(X_single)
        X_scaled = pd.DataFrame(X_scaled, columns=X_single.columns, index=X_single.index)
        
        # Add intercept for OLS
        X_scaled['intercept'] = 1
        
        # Fit OLS model
        model = sm.OLS(y, X_scaled).fit()
        
        # Extract coefficients and p-values directly from model
        coef_value = model.params[feature_name]
        p_value = model.pvalues[feature_name]
        
        df_directions = pd.concat([
            df_directions,
            pd.DataFrame({
                'Feature': [feature_name],
                'coef': [coef_value],
                'p_value': [p_value * correction_factor]
            })
        ], ignore_index=True)
    
    # Sort by absolute coefficient
    df_directions['abs_coef'] = df_directions['coef'].abs()
    df_directions = df_directions.sort_values('abs_coef', ascending=False).drop('abs_coef', axis=1)
    df_directions = df_directions.reset_index(drop=True)
    
    return df_directions

# Añadir intercepto o no intercepto y que plotee un bar plot
def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for all features.
    
    VIF measures how much the variance of a regression coefficient increases
    due to multicollinearity. VIF > 5-10 indicates potential multicollinearity.
    
    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix
    
    Returns
    -------
    vif_df : pd.DataFrame
        DataFrame with columns ['Feature', 'VIF']
    
    Examples
    --------
    >>> import pandas as pd
    >>> from brainlat.tools import calculate_vif
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    >>> vif = calculate_vif(X)
    """
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    
    return vif_data


def _get_r_squared_for_feature(X, feature_idx):
    """
    Get R² from regressing one feature on all others (helper for VIF).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    feature_idx : int
        Index of feature to regress
    
    Returns
    -------
    r_squared : float
        R² value
    """
    y = X.iloc[:, feature_idx]
    X_others = X.drop(X.columns[feature_idx], axis=1)
    X_others = sm.add_constant(X_others)
    
    model = sm.OLS(y, X_others).fit()
    
    return model.rsquared

# Añadir que tipo de escalar hacer, min max, cualquier tipo, log, robust
def scale_features(X, scale_range=(0.05, 0.95)):
    """
    Scale features to a specified range using MinMaxScaler.
    
    Parameters
    ----------
    X : pd.DataFrame or array-like, shape (n_samples, n_features)
        Feature matrix
    scale_range : tuple, default=(0.05, 0.95)
        Min and max values for scaling
    
    Returns
    -------
    X_scaled : pd.DataFrame
        Scaled features with same index and columns as input
    scaler : MinMaxScaler
        Fitted scaler object for transforming new data
    
    Examples
    --------
    >>> from brainlat.tools import scale_features
    >>> X_scaled, scaler = scale_features(X)
    """
    scaler = MinMaxScaler(scale_range)
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    X_scaled_array = scaler.fit_transform(X_array)
    
    if isinstance(X, pd.DataFrame):
        X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    else:
        X_scaled = X_scaled_array
    
    return X_scaled, scaler

