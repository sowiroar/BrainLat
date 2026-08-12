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


def calculate_delta_aic(y_true, y_pred, k):
    """
    Calculate delta AIC between the fitted model and a null model.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    k : int
        Number of parameters (features) in the model
        
    Returns
    -------
    delta_aic : float
        Difference: null_aic - model_aic. Larger positive values indicate better fit.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    rss = np.sum((y_true - y_pred) ** 2)
    n = len(y_true)
    if rss <= 0 or n <= 0:
        return -np.inf
    
    aic = n * np.log(rss / n) + 2 * k
    
    null_rss = np.sum((y_true - y_true.mean()) ** 2)
    if null_rss <= 0:
        null_aic = 2
    else:
        null_aic = n * np.log(null_rss / n) + 2
        
    return null_aic - aic


def bootstrap_delta_aic(y_true, predictions, k, n_bootstrap=1000, ci=80, random_state=42):
    """
    Calculate confidence interval for delta AIC using bootstrapping.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    predictions : array-like
        Predicted values
    k : int
        Number of parameters in the model
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    ci : int, default=80
        Confidence interval percentile (e.g. 80 for 80% CI, 95 for 95% CI)
    random_state : int, default=42
        Seed for reproducibility
        
    Returns
    -------
    mean_delta : float
        Mean delta AIC across bootstrap samples
    ci_bounds : tuple
        (lower_bound, upper_bound) at the specified confidence level
    """
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    
    rng = np.random.default_rng(random_state)
    bootstrapped_deltas = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_sample = y_true[indices]
        pred_sample = predictions[indices]
        bootstrapped_deltas.append(calculate_delta_aic(y_sample, pred_sample, k))
        
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    
    lower_bound = np.percentile(bootstrapped_deltas, lower_pct)
    upper_bound = np.percentile(bootstrapped_deltas, upper_pct)
    
    return float(np.mean(bootstrapped_deltas)), (float(lower_bound), float(upper_bound))


def exploratory_factor_analysis(X, n_factors=10, rotation='varimax', scale=True):
    """
    Perform Exploratory Factor Analysis (EFA).

    Fits an EFA model using the FactorAnalyzer library, computes eigenvalues,
    determines the optimal number of factors (eigenvalue > 1), and returns
    the factor loadings matrix.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
        Feature matrix. If DataFrame, column names are used as labels.
    n_factors : int, default=10
        Number of factors to extract. Should be larger than expected
        optimal number for initial exploration.
    rotation : str, default='varimax'
        Rotation method: 'varimax', 'promax', 'oblimin', 'oblimax',
        'quartimin', 'quartimax', 'equamax', or None.
    scale : bool, default=True
        Whether to apply MinMaxScaler (0.05, 0.95) before fitting.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'eigenvalues' : np.ndarray
            Eigenvalues for all components.
        - 'n_optimal' : int
            Number of factors with eigenvalue > 1 (Kaiser criterion).
        - 'loadings' : pd.DataFrame
            Factor loadings matrix with features as rows and factors as columns.
        - 'variance_explained' : np.ndarray
            Proportion of variance explained by each factor.

    Examples
    --------
    >>> from brainlat.stats import exploratory_factor_analysis
    >>> import pandas as pd, numpy as np
    >>> X = pd.DataFrame(np.random.randn(200, 20), columns=[f'v{i}' for i in range(20)])
    >>> results = exploratory_factor_analysis(X, n_factors=10)
    >>> print(f"Optimal factors: {results['n_optimal']}")
    >>> print(results['loadings'].head())

    Notes
    -----
    Requires: pip install factor-analyzer
    """
    from factor_analyzer import FactorAnalyzer
    from sklearn.preprocessing import MinMaxScaler

    # Convert to array if needed, preserve column names
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_values = X.values
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        X_values = np.asarray(X)

    # Scale if requested
    if scale:
        scaler = MinMaxScaler((0.05, 0.95))
        X_scaled = scaler.fit_transform(X_values)
    else:
        X_scaled = X_values.copy()

    # Fit factor analyzer
    fa = FactorAnalyzer()
    fa.set_params(n_factors=n_factors, rotation=rotation)
    fa.fit(X_scaled)

    # Get eigenvalues
    eigenvalues, _ = fa.get_eigenvalues()

    # Optimal number of factors (Kaiser criterion: eigenvalue > 1)
    n_optimal_arr = np.where(eigenvalues <= 1)[0]
    n_optimal = n_optimal_arr[0] if len(n_optimal_arr) > 0 else n_factors

    # Get loadings as DataFrame
    loadings = np.abs(fa.loadings_)
    factor_names = [f'Factor {i+1}' for i in range(n_factors)]
    loadings_df = pd.DataFrame(
        loadings, index=feature_names, columns=factor_names
    )

    # Variance explained
    variance = fa.get_factor_variance()
    # variance returns (SS Loadings, Proportion Var, Cumulative Var)
    variance_explained = variance[1] if len(variance) > 1 else np.array([])

    return {
        'eigenvalues': eigenvalues,
        'n_optimal': int(n_optimal),
        'loadings': loadings_df,
        'variance_explained': variance_explained
    }


def get_efa_clusters(loadings_df, threshold=0.75, n_factors=None):
    """
    Group variables into factor clusters based on loading threshold.

    Given a loadings DataFrame from EFA, assigns each variable to the
    factor where its loading exceeds the threshold.

    Parameters
    ----------
    loadings_df : pd.DataFrame
        Factor loadings matrix with features as rows, factors as columns.
        Typically from ``exploratory_factor_analysis()['loadings']``.
    threshold : float, default=0.75
        Minimum loading value to assign a variable to a factor.
    n_factors : int or None, default=None
        Number of factors to consider. If None, uses all columns.

    Returns
    -------
    clusters : list of list of str
        List where clusters[i] contains the variable names assigned to
        factor i. Empty list if no variables exceed the threshold.
    all_features : list of str
        Flat list of all selected features across all factors.
    cluster_dict : dict
        Dictionary mapping factor index (int) to list of variable names.
        Only includes factors with at least one variable.

    Examples
    --------
    >>> from brainlat.stats import exploratory_factor_analysis, get_efa_clusters
    >>> results = exploratory_factor_analysis(X, n_factors=10)
    >>> clusters, features, cdict = get_efa_clusters(results['loadings'], threshold=0.6)
    >>> for i, c in enumerate(clusters):
    ...     if c:
    ...         print(f"Factor {i+1}: {c}")
    """
    if loadings_df is None or loadings_df.empty:
        return [], [], {}

    if n_factors is None:
        n_factors = loadings_df.shape[1]
    else:
        n_factors = min(n_factors, loadings_df.shape[1])

    clusters = []
    cluster_dict = {}

    for i in range(n_factors):
        col = loadings_df.iloc[:, i]
        selected = list(loadings_df[col > threshold].index)

        clusters.append(selected)
        if len(selected) > 0:
            cluster_dict[i] = selected

    # Flat list of all selected features
    all_features = []
    for cluster in clusters:
        all_features.extend(cluster)

    return clusters, all_features, cluster_dict


def fit_cox_model(df, duration_col, event_col, formula=None, covariates=None, alpha=0.05):
    """
    Fit a Cox Proportional Hazards regression model using lifelines.
    Automatically performs a test of proportional hazards assumption using Schoenfeld residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables.
    duration_col : str
        Name of the column containing the duration/time-to-event values.
    event_col : str
        Name of the column containing the binary event-observed indicators (1 if event, 0 if censored).
    formula : str, optional
        A patsy formula to specify covariates (e.g. "Age + Sex + GAP_corrected").
    covariates : list of str, optional
        List of column names to include as covariates in the model. Ignored if formula is provided.
    alpha : float, default=0.05
        Significance level for the proportional hazards test.

    Returns
    -------
    cph : CoxPHFitter
        The fitted lifelines CoxPHFitter model.
    ph_test : StatisticalResult
        StatisticalResult object containing the proportional hazards test results 
        (chi-squared statistics, p-values, degrees of freedom).
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    import pandas as pd

    cph = CoxPHFitter()

    if formula is not None:
        cph.fit(df, duration_col=duration_col, event_col=event_col, formula=formula)
    else:
        columns_to_include = [duration_col, event_col]
        if covariates is not None:
            columns_to_include.extend(covariates)
        df_model = df[columns_to_include].dropna()
        cph.fit(df_model, duration_col=duration_col, event_col=event_col)

    # Check proportional hazards assumption using Schoenfeld residuals
    ph_test = proportional_hazard_test(cph, df, time_transform='rank')

    return cph, ph_test



