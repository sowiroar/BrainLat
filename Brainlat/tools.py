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


def compute_odds_ratios_bootstrap(X, y, covars=None, n_iterations=1000, test_size=0.2, random_state=42):
    """
    Compute Odds Ratios (OR) for features using bootstrap logistic regression.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with predictor features.
    y : array-like
        Binary target variable (0 or 1), e.g. GAP_bin.
    covars : pd.DataFrame or pd.Series or list, optional
        Covariates to include in the models.
    n_iterations : int, default=1000
        Number of bootstrap train-test split iterations.
    test_size : float, default=0.2
        Test split proportion for each iteration.
    random_state : int, default=42
        Seed for reproducibility.
        
    Returns
    -------
    df_results : pd.DataFrame
        DataFrame with OR, 2.5% CI, 97.5% CI, beta, SE, z, and p-value.
    """
    from scipy.stats import norm
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)
    df_results = pd.DataFrame()
    
    # Process covariates
    if covars is not None:
        if isinstance(covars, (pd.DataFrame, pd.Series)):
            df_cov = pd.DataFrame(covars)
        else:
            df_cov = pd.DataFrame(covars)
    else:
        df_cov = None
        
    for feature in X.columns:
        or_values = []
        ci_low_values = []
        ci_high_values = []
        beta_values = []
        se_values = []
        z_values = []
        
        # Prepare inputs
        X_feat = X[[feature]].copy()
        if df_cov is not None:
            X_model = pd.concat([X_feat, df_cov], axis=1).dropna()
            y_model = y.loc[X_model.index]
        else:
            X_model = X_feat.dropna()
            y_model = y.loc[X_model.index]
            
        for iteration in range(n_iterations):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_model, y_model, test_size=test_size, random_state=random_state + iteration
                )
                
                scaler = MinMaxScaler((0.05, 0.95))
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                
                X_train_scaled["intercept"] = 1
                
                # Fit logistic regression
                model = sm.Logit(y_train, X_train_scaled).fit(disp=0)
                
                params = model.params
                conf = np.exp(model.conf_int())
                conf["OR"] = np.exp(params)
                conf["z"] = model.tvalues
                conf["P>|z|"] = model.pvalues
                conf.columns = ["2.5%", "97.5%", "OR", "z", "P>|z|"]
                
                or_values.append(conf.loc[feature, "OR"])
                ci_low_values.append(conf.loc[feature, "2.5%"])
                ci_high_values.append(conf.loc[feature, "97.5%"])
                beta_values.append(model.params[feature])
                se_values.append(model.bse[feature])
                z_values.append(model.tvalues[feature])
                
            except Exception:
                continue
                
        if len(or_values) > 1:
            or_values = np.array(or_values)
            ci_low_values = np.array(ci_low_values)
            ci_high_values = np.array(ci_high_values)
            beta_values = np.array(beta_values)
            se_values = np.array(se_values)
            
            or_mean = np.mean(or_values)
            ci_low_mean = np.mean(ci_low_values)
            ci_high_mean = np.mean(ci_high_values)
            beta_mean = np.mean(beta_values)
            se_mean = np.mean(se_values)
            
            z_combined = beta_mean / se_mean
            p_combined = 2 * (1 - norm.cdf(abs(z_combined)))
            z_mean_original = np.mean(z_values)
        else:
            ci_low_mean = np.nan
            ci_high_mean = np.nan
            or_mean = np.nan
            beta_mean = np.nan
            se_mean = np.nan
            z_combined = np.nan
            p_combined = np.nan
            z_mean_original = np.nan
            
        df_temp = pd.DataFrame({
            "Feature": [feature],
            "2.5%": [ci_low_mean],
            "97.5%": [ci_high_mean],
            "OR": [or_mean],
            "beta_mean": [beta_mean],
            "SE_mean": [se_mean],
            "z": [z_combined],
            "P>|z|": [f"{p_combined:.2e}" if not np.isnan(p_combined) else np.nan],
            "z_mean_original": [z_mean_original],
            "n_iterations_ok": [len(or_values)]
        })
        df_results = pd.concat([df_results, df_temp], ignore_index=True)
        
    return df_results


def compute_relative_risks_bootstrap(X, y_gap_corrected, delta_time, covars=None, n_iterations=1000, test_size=0.2, random_state=42):
    """
    Compute Relative Risks (RR) using bootstrap GLM (log link binomial family)
    after residualizing y_gap_corrected against delta_time.
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with predictor features.
    y_gap_corrected : array-like
        Gap corrected values to be residualized.
    delta_time : array-like
        Time variable for residualization.
    covars : pd.DataFrame or pd.Series or list, optional
        Covariates to include in the models.
    n_iterations : int, default=1000
        Number of bootstrap iterations.
    test_size : float, default=0.2
        Test split proportion.
    random_state : int, default=42
        Seed for reproducibility.
        
    Returns
    -------
    df_results : pd.DataFrame
        DataFrame with RR, 2.5% CI, 97.5% CI, z, p-val, and statistics.
    """
    from scipy.stats import norm
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    y_gap_corrected = np.asarray(y_gap_corrected)
    delta_time = np.asarray(delta_time)
    
    # Residualize y_gap_corrected against delta_time
    X_resid = sm.add_constant(delta_time)
    resid_model = sm.OLS(y_gap_corrected, X_resid).fit()
    resid = resid_model.resid
    
    # Define binary GAP outcome as Series aligned with X
    y_bin = pd.Series(np.where(resid > 0, 1, 0), index=X.index)
    
    df_results = pd.DataFrame()
    
    # Process covariates
    if covars is not None:
        if isinstance(covars, (pd.DataFrame, pd.Series)):
            df_cov = pd.DataFrame(covars)
        else:
            df_cov = pd.DataFrame(covars)
    else:
        df_cov = None
        
    for feature in X.columns:
        rr_values = []
        ci_low_values = []
        ci_high_values = []
        z_values = []
        
        # Prepare inputs
        X_feat = X[[feature]].copy()
        if df_cov is not None:
            X_model = pd.concat([X_feat, df_cov], axis=1).dropna()
            y_model = y_bin.loc[X_model.index]
        else:
            X_model = X_feat.dropna()
            y_model = y_bin.loc[X_model.index]
            
        for iteration in range(n_iterations):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_model, y_model, test_size=test_size, random_state=random_state + iteration
                )
                
                scaler = MinMaxScaler((0.05, 0.95))
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                
                X_train_scaled["intercept"] = 1
                
                # Fit GLM with binomial family and log link for Relative Risk
                model = sm.GLM(
                    y_train,
                    X_train_scaled,
                    family=sm.families.Binomial(link=sm.families.links.log())
                ).fit(disp=0)
                
                params = model.params
                conf = np.exp(model.conf_int())
                conf["RR"] = np.exp(params)
                conf["z"] = model.tvalues
                conf["P>|z|"] = model.pvalues
                conf.columns = ["2.5%", "97.5%", "RR", "z", "P>|z|"]
                
                rr_values.append(conf.loc[feature, "RR"])
                ci_low_values.append(conf.loc[feature, "2.5%"])
                ci_high_values.append(conf.loc[feature, "97.5%"])
                z_values.append(conf.loc[feature, "z"])
                
            except Exception:
                continue
                
        if len(rr_values) > 1:
            rr_values = np.array(rr_values)
            ci_low_values = np.array(ci_low_values)
            ci_high_values = np.array(ci_high_values)
            z_values = np.array(z_values)
            
            rr_mean = np.mean(rr_values)
            ci_low_mean = np.mean(ci_low_values)
            ci_high_mean = np.mean(ci_high_values)
            z_mean_original = np.mean(z_values)
            
            # P-value calculation from combined effect and CI
            if rr_mean > 0 and ci_low_mean > 0 and ci_high_mean > 0:
                se_log_effect = (np.log(ci_high_mean) - np.log(ci_low_mean)) / (2 * 1.96)
                if se_log_effect > 0:
                    z_combined = np.log(rr_mean) / se_log_effect
                    p_combined = 2 * (1 - norm.cdf(abs(z_combined)))
                else:
                    z_combined = np.nan
                    p_combined = np.nan
            else:
                z_combined = np.nan
                p_combined = np.nan
        else:
            ci_low_mean = np.nan
            ci_high_mean = np.nan
            rr_mean = np.nan
            z_combined = np.nan
            p_combined = np.nan
            z_mean_original = np.nan
            
        df_temp = pd.DataFrame({
            "Feature": [feature],
            "2.5%": [ci_low_mean],
            "97.5%": [ci_high_mean],
            "RR": [rr_mean],
            "z": [z_combined],
            "P>|z|": [f"{p_combined:.2e}" if not np.isnan(p_combined) else np.nan],
            "z_mean_original": [z_mean_original],
            "n_iterations_ok": [len(rr_values)]
        })
        df_results = pd.concat([df_results, df_temp], ignore_index=True)
        
    return df_results


