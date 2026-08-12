"""
Standard regression models without gap correction.

This module provides linear and non-linear regression models including:
- OLS (Ordinary Least Squares)
- Ridge regression
- Lasso regression
- ElasticNet regression
- Linear Nested Cross Validation with Bayesian hyperparameter optimization

These models return traditional statistical measures (t-values, p-values)
without gap correction.
"""

import numpy as np
import pandas as pd
import warnings
import math
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
import scipy
from skopt import BayesSearchCV

from .stats import (
    mean_directional_accuracy, mean_absolute_error as mean_absolute_error_custom,
    coef_tval, coef_pval
)
from .diagnostics import DataDiagnostics, Logger, generate_sanity_report

warnings.filterwarnings("ignore")

def Regression_Linear_NestedCV(
    X,
    y,
    model,
    param_space,
    outer_splits=5,
    inner_splits=3,
    n_iter=30,
    scoring='neg_mean_squared_error',
    random_state=42,
    log=False
):
    """
    Nested Cross Validation con Bayesian Search para regresores lineales sklearn.

    Soporta cualquier regresor con interfaz sklearn, incluyendo:
        LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
        BayesianRidge, Lars, LassoLars, TweedieRegressor,
        QuantileRegressor, SGDRegressor, etc.

    Parameters
    ----------
    X            : pd.DataFrame con features
    y            : array-like con target
    model        : instancia del regresor (sin fitear)
    param_space  : dict con espacios skopt. Si es {} o None, entrena sin tuning
                   (equivalente a un CV simple).
    outer_splits : folds externos (evaluación)
    inner_splits : folds internos (optimización bayesiana)
    n_iter       : evaluaciones bayesianas por fold externo
    scoring      : métrica para BayesSearchCV
    random_state : semilla
    log          : bool, default=False
                   Whether to save logs and diagnostic reports

    Returns
    -------
    [coef_df, results_labels_df, best_params_list]

    Notes
    -----
    - t-value y p-value solo son válidos para modelos con coef_ e intercept_
      definidos (LinearRegression, Ridge, Lasso, ElasticNet...).
      Para modelos sin estos atributos (e.g. algunos SGD configs) se dejan NaN.
    - Instalación requerida: pip install scikit-optimize
    """
    
    # Initialize logger if needed
    logger = Logger() if log else None
    
    if logger:
        logger.add_message(f"Starting Linear NestedCV analysis...")
    
    # Perform sanity checks
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="Linear NestedCV Input")
    
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/linear_nestedcv_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check: {report['n_samples']} samples, {report['n_features']} features")
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)

    y          = np.array(y)
    lista_vars = list(X.columns)
    n_vars     = len(lista_vars)
    has_tuning = bool(param_space)

    # Acumuladores
    y_labels, y_predicts = [], []
    r_squared_l, mse_l, rmse_l = [], [], []
    best_params_list = []
    results_labels_df = pd.DataFrame(columns=['ID', 'y_labels', 'y_pred'])

    coef_array = np.full([n_vars + 1, outer_splits], np.nan)

    outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    for iter_, (train_index, test_index) in enumerate(outer_kf.split(X)):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            X_train_raw = X.iloc[train_index]
            X_test_raw  = X.iloc[test_index]
            y_train     = y[train_index]
            y_test      = y[test_index]

            # --- Escalado dentro del fold: fit solo en train ---
            scaler  = MinMaxScaler((0.05, 0.95))
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=lista_vars, index=X_train_raw.index
            )
            X_test  = pd.DataFrame(
                scaler.transform(X_test_raw),
                columns=lista_vars, index=X_test_raw.index
            )

            # --- Búsqueda de hiperparámetros (loop interno) ---
            if has_tuning:
                inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
                searcher = BayesSearchCV(
                    estimator=model,
                    search_spaces=param_space,
                    n_iter=n_iter,
                    cv=inner_kf,
                    scoring=scoring,
                    refit=True,
                    random_state=random_state,
                    n_jobs=-1,
                    optimizer_kwargs={'base_estimator': 'GP'}
                )
                searcher.fit(X_train, y_train)
                best_model  = searcher.best_estimator_
                best_params = searcher.best_params_
            else:
                # Sin espacio de parámetros: entrena directamente
                model.fit(X_train, y_train)
                best_model  = model
                best_params = {}

            best_params_list.append(best_params)

            # --- Predicción en fold externo ---
            predicted_values = best_model.predict(X_test)

            y_labels.extend(list(y_test))
            y_predicts.extend(list(predicted_values))

            r_squared_l.append(r2_score(y_test, predicted_values))
            mse_l.append(np.round(mean_squared_error(y_test, predicted_values), 6))
            rmse_l.append(np.round(math.sqrt(mean_squared_error(y_test, predicted_values)), 6))

            # --- Coeficientes e intercept (si el modelo los expone) ---
            if hasattr(best_model, 'intercept_') and hasattr(best_model, 'coef_'):
                intercept = np.atleast_1d(best_model.intercept_)[0]
                coefs     = np.atleast_1d(best_model.coef_).flatten()
                coef_array[0, iter_]  = intercept
                coef_array[1:, iter_] = coefs[:n_vars]

            # --- Guardar predicciones ---
            temp_df = pd.DataFrame(
                np.column_stack((y_test, predicted_values)),
                columns=['y_labels', 'y_pred']
            )
            temp_df['ID'] = X_test.index
            results_labels_df = pd.concat([results_labels_df, temp_df], ignore_index=True)


    n = len(y_predicts)
    p = X.shape[1]

    r_squared     = r2_score(y_labels, y_predicts)
    k             = p - 1
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mde           = mean_directional_accuracy(y_labels, y_predicts)
    mae           = mean_absolute_error_custom(y_labels, y_predicts)
    mse           = np.round(mean_squared_error(y_labels, y_predicts), 6)
    rmse          = np.round(math.sqrt(mse), 6)
    F2            = r_squared / (1 - r_squared) if r_squared < 1 else np.inf
    F             = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if r_squared < 1 else np.inf
    p_value       = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15)


    coef_array_mean = coef_array.mean(axis=1).reshape(-1, 1)
    coef_array_std  = coef_array.std(axis=1).reshape(-1, 1)

    coef_df = pd.DataFrame(
        index=['_intercept'] + lista_vars,
        columns=['Estimate mean', 'Estimate std', 't value', 'p value']
    )
    coef_df['Estimate mean'] = coef_array_mean.flatten()
    coef_df['Estimate std']  = coef_array_std.flatten()

    # t y p value solo si todos los folds devolvieron coeficientes
    if not np.all(np.isnan(coef_array)):
        try:
            coef_df['t value'] = coef_tval(coef_array_mean, X.values, y_labels, y_predicts)
            coef_df['p value'] = coef_pval(coef_array_mean, X.values, y_labels, y_predicts)
        except Exception:
            pass

    coef_df.loc['_intercept', 'R2']          = r_squared
    coef_df.loc['_intercept', 'R2 adj']      = r_squared_adj
    coef_df.loc['_intercept', 'R2 [+-]']     = 1.96 * np.std(r_squared_l)
    coef_df.loc['_intercept', 'F2']          = F2
    coef_df.loc['_intercept', 'mse']         = mse
    coef_df.loc['_intercept', 'mse [+-]']    = 1.96 * np.std(mse_l)
    coef_df.loc['_intercept', 'rmse']        = rmse
    coef_df.loc['_intercept', 'rmse [+-]']   = 1.96 * np.std(rmse_l)
    coef_df.loc['_intercept', 'outcome var'] = np.var(y)
    coef_df.loc['_intercept', 'F']           = F
    coef_df.loc['_intercept', 'F-p_value']   = p_value
    coef_df.loc['_intercept', 'MDE']         = mde
    coef_df.loc['_intercept', 'MAE']         = mae

    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred']]

    if logger:
        logger.add_message(f"Linear NestedCV completed with R² = {r_squared:.4f}")
        logger.save('linear_nestedcv_regression')

    return [coef_df.fillna(''), results_labels_df, best_params_list]


def Regression_NonLinear_NestedCV(
    X,
    y,
    model,
    param_space,
    outer_splits=5,
    inner_splits=3,
    n_iter=30,
    scoring='r2',
    random_state=42,
    log=False
):
    """
    Nested Cross Validation con Bayesian Search para regresores no-lineales.

    Soporta cualquier regresor sklearn no-lineal, incluyendo:
        GradientBoostingRegressor, RandomForestRegressor, SVR, XGBRegressor,
        ExtraTreesRegressor, AdaBoostRegressor, etc.

    Parameters
    ----------
    X            : pd.DataFrame con features
    y            : array-like con target
    model        : instancia del regresor (sin fitear)
    param_space  : dict con espacios skopt (Real, Integer, Categorical)
    outer_splits : folds externos (evaluación)
    inner_splits : folds internos (optimización bayesiana)
    n_iter       : evaluaciones bayesianas por fold externo
    scoring      : métrica para BayesSearchCV (típicamente 'r2')
    random_state : semilla
    log          : bool, default=False
                   Whether to save logs and diagnostic reports

    Returns
    -------
    [coef_df, results_labels_df, best_params_list]

    Notes
    -----
    - Para modelos no-lineales, se usan feature importances en lugar de coeficientes
    - t-values y p-values no son significativas para estos modelos
    - Instalación requerida: pip install scikit-optimize
    """
    
    # Initialize logger if needed
    logger = Logger() if log else None
    
    if logger:
        logger.add_message(f"Starting NonLinear NestedCV analysis...")
    
    # Perform sanity checks
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="NonLinear NestedCV Input")
    
    if log:
        report_str = generate_sanity_report(report, filename="brainlat_logs/nonlinear_nestedcv_sanity_check.txt")
        print("\n" + report_str)
        logger.add_message(f"Data quality check: {report['n_samples']} samples, {report['n_features']} features")
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)

    y          = np.array(y)
    lista_vars = list(X.columns)
    n_vars     = len(lista_vars)

    # Acumuladores
    y_labels, y_predicts = [], []
    r_squared_l, mse_l, rmse_l = [], [], []
    best_params_list = []
    results_labels_df = pd.DataFrame(columns=['ID', 'y_labels', 'y_pred'])

    # Array para importances o coefs
    importances_array = np.full([n_vars, outer_splits], np.nan)

    outer_kf = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    for iter_, (train_index, test_index) in enumerate(outer_kf.split(X)):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            X_train_raw = X.iloc[train_index]
            X_test_raw  = X.iloc[test_index]
            y_train     = y[train_index]
            y_test      = y[test_index]

            # --- Escalado dentro del fold: fit solo en train ---
            scaler  = MinMaxScaler((0.05, 0.95))
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=lista_vars, index=X_train_raw.index
            )
            X_test  = pd.DataFrame(
                scaler.transform(X_test_raw),
                columns=lista_vars, index=X_test_raw.index
            )

            # --- Búsqueda de hiperparámetros (loop interno) ---
            inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
            searcher = BayesSearchCV(
                estimator=model,
                search_spaces=param_space,
                n_iter=n_iter,
                cv=inner_kf,
                scoring=scoring,
                refit=True,
                random_state=random_state,
                n_jobs=-1,
                optimizer_kwargs={'base_estimator': 'GP'}
            )
            searcher.fit(X_train, y_train)
            best_model  = searcher.best_estimator_
            best_params = searcher.best_params_

            best_params_list.append(best_params)

            # --- Predicción en fold externo ---
            predicted_values = best_model.predict(X_test)

            y_labels.extend(list(y_test))
            y_predicts.extend(list(predicted_values))

            r_squared_l.append(r2_score(y_test, predicted_values))
            mse_l.append(np.round(mean_squared_error(y_test, predicted_values), 6))
            rmse_l.append(np.round(math.sqrt(mean_squared_error(y_test, predicted_values)), 6))

            # --- Feature importances ---
            if hasattr(best_model, 'feature_importances_'):
                importances_array[:, iter_] = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                coefs = best_model.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
                importances_array[:, iter_] = coefs

            # --- Guardar predicciones ---
            temp_df = pd.DataFrame(
                np.column_stack((y_test, predicted_values)),
                columns=['y_labels', 'y_pred']
            )
            temp_df['ID'] = X_test.index
            results_labels_df = pd.concat([results_labels_df, temp_df], ignore_index=True)


    n = len(y_predicts)
    p = X.shape[1]

    r_squared     = r2_score(y_labels, y_predicts)
    k             = p - 1
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mde           = mean_directional_accuracy(y_labels, y_predicts)
    mae           = mean_absolute_error_custom(y_labels, y_predicts)
    mse           = np.round(mean_squared_error(y_labels, y_predicts), 6)
    rmse          = np.round(math.sqrt(mse), 6)
    F2            = r_squared / (1 - r_squared) if r_squared < 1 else np.inf
    F             = (r_squared / p) / ((1 - r_squared) / (n - p - 1)) if r_squared < 1 else np.inf
    p_value       = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15)

    # --- Tabla de importances/coeficientes ---
    importances_mean = np.nanmean(importances_array, axis=1).reshape(-1, 1)
    importances_std  = np.nanstd(importances_array, axis=1).reshape(-1, 1)

    coef_df = pd.DataFrame(
        index=['_intercept'] + lista_vars,
        columns=['Estimate mean', 'Estimate std', 't value', 'p value']
    )
    
    # Primera fila es intercept (NaN para modelos no-lineales)
    coef_df.loc['_intercept', 'Estimate mean'] = np.nan
    coef_df.loc['_intercept', 'Estimate std'] = np.nan
    
    # Importances para features
    coef_df.iloc[1:, 0] = importances_mean.flatten()
    coef_df.iloc[1:, 1] = importances_std.flatten()

    coef_df.loc['_intercept', 'R2']          = r_squared
    coef_df.loc['_intercept', 'R2 adj']      = r_squared_adj
    coef_df.loc['_intercept', 'R2 [+-]']     = 1.96 * np.std(r_squared_l)
    coef_df.loc['_intercept', 'F2']          = F2
    coef_df.loc['_intercept', 'mse']         = mse
    coef_df.loc['_intercept', 'mse [+-]']    = 1.96 * np.std(mse_l)
    coef_df.loc['_intercept', 'rmse']        = rmse
    coef_df.loc['_intercept', 'rmse [+-]']   = 1.96 * np.std(rmse_l)
    coef_df.loc['_intercept', 'outcome var'] = np.var(y)
    coef_df.loc['_intercept', 'F']           = F
    coef_df.loc['_intercept', 'F-p_value']   = p_value
    coef_df.loc['_intercept', 'MDE']         = mde
    coef_df.loc['_intercept', 'MAE']         = mae

    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred']]

    if logger:
        logger.add_message(f"NonLinear NestedCV completed with R² = {r_squared:.4f}")
        logger.save('nonlinear_nestedcv_regression')

    return [coef_df.fillna(''), results_labels_df, best_params_list]


def Regression_XGB(X, y, min_iter=1, max_iter=2, n_splits=10, log=False):
    """
    XGBoost Regressor with K-Fold cross-validation (no gap correction).

    Implements a non-linear XGBoost regression model with K-Fold CV.
    Uses feature importances instead of coefficients. This is equivalent
    to Regression_NonLinear_NestedCV but without Bayesian hyperparameter
    tuning — uses default XGBRegressor parameters for simplicity.

    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix with feature names as columns.
    y : array-like, shape (n_samples,)
        Target variable.
    min_iter : int, default=1
        Minimum iteration number for random state variation.
    max_iter : int, default=2
        Maximum iteration number (exclusive).
    n_splits : int, default=10
        Number of KFold splits for cross-validation.
    log : bool, default=False
        Whether to save logs and diagnostic reports to files.

    Returns
    -------
    results : list
        [coef_df, r_squared, results_labels_df]

        - coef_df : pd.DataFrame
            Feature importances and model statistics.
            Index row 'Model' has aggregate metrics (R², F, etc.).
        - r_squared : float
            Overall R² across all folds.
        - results_labels_df : pd.DataFrame
            Predictions for each sample with columns
            ['ID', 'y_labels', 'y_pred'].

    Examples
    --------
    >>> from brainlat.regresion_model import Regression_XGB
    >>> import pandas as pd, numpy as np
    >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = np.random.randn(100)
    >>> coef_df, r2, pred_df = Regression_XGB(X, y, n_splits=5, log=True)
    """
    from xgboost import XGBRegressor

    # Initialize logger if needed
    logger = Logger() if log else None

    if logger:
        logger.add_message("Starting XGB regression analysis...")

    # Perform sanity checks
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="XGB Regression Input")

    if log:
        report_str = generate_sanity_report(
            report, filename="brainlat_logs/xgb_regression_sanity_check.txt"
        )
        print("\n" + report_str)
        logger.add_message(
            f"Data quality check completed: {report['n_samples']} samples, "
            f"{report['n_features']} features"
        )
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)

    y = np.array(y)
    lista_vars = list(X.columns)
    n_vars = len(lista_vars)

    results_df_all = pd.DataFrame()
    r_squared_final = 0

    for iteration in range(min_iter, max_iter):
        y_labels = []
        y_predicts = []

        y_pred_cv = []
        y_test_cv = []
        r_squared_l = []
        mse_l = []
        rmse_l = []

        results_labels_df = pd.DataFrame(columns=['y_labels', 'y_pred', 'ID'])

        coef_array = np.zeros([n_vars + 1, n_splits])

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=iteration)

        iter_split = 0
        for train_index, test_index in kf.split(X):
            X_train_raw, X_test_raw = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            # Scale within the fold to avoid data leakage
            scaler = MinMaxScaler((0.05, 0.95))
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=X_train_raw.columns, index=X_train_raw.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test_raw),
                columns=X_test_raw.columns, index=X_test_raw.index
            )

            model = XGBRegressor(
                objective='reg:squarederror', random_state=42
            )
            model.fit(X_train, y_train)

            # Store feature importances (intercept = NaN for non-linear)
            coef_array[0, iter_split] = np.nan
            coef_array[1::, iter_split] = np.array(model.feature_importances_)

            predicted_values = model.predict(X_test)

            y_labels.extend(list(y_test))
            y_predicts.extend(list(predicted_values))

            r_squared_l.append(r2_score(y_test, predicted_values))
            mse_l.append(
                np.round(mean_squared_error(y_test, predicted_values), 6)
            )
            rmse_l.append(
                np.round(
                    math.sqrt(mean_squared_error(y_test, predicted_values)), 6
                )
            )

            result = np.column_stack((y_test, predicted_values))
            temp_df = pd.DataFrame(result, columns=['y_labels', 'y_pred'])
            temp_df['ID'] = X_test.index

            results_labels_df = pd.concat(
                [results_labels_df, temp_df], ignore_index=True
            )

            iter_split += 1

        # Aggregate cross-validation results
        n = len(y_predicts)
        p = X.shape[1]
        r_squared = r2_score(y_labels, y_predicts)
        r_squared_final = r_squared

        k = p - 1
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        mse = np.round(mean_squared_error(y_labels, y_predicts), 6)
        rmse = np.round(math.sqrt(mean_squared_error(y_labels, y_predicts)), 6)

        # F-statistic (corrected: use r_squared and p as df1)
        if (1 - r_squared) > 0:
            F = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
            p_value = np.round(scipy.stats.f.sf(F, p, (n - p - 1)), 15)
            F2 = r_squared / (1 - r_squared)
        else:
            F = np.inf
            p_value = 0.0
            F2 = np.inf

        # Diagnostic metrics
        mda = mean_directional_accuracy(y_labels, y_predicts)
        mae = mean_absolute_error_custom(y_labels, y_predicts)

        # Calculate mean importances
        coef_array_mean = np.zeros([n_vars + 1, 1])
        coef_array_std = np.zeros([n_vars + 1, 1])

        for j in range(n_vars + 1):
            coef_array_mean[j] = coef_array[j, :].mean()
            coef_array_std[j] = coef_array[j, :].std()

        coef_df = pd.DataFrame(
            index=['Model'] + lista_vars,
            columns=['Feature_Importance_mean', 'Feature_Importance_std']
        )

        coef_df['Feature_Importance_mean'] = coef_array_mean
        coef_df['Feature_Importance_std'] = coef_array_std

        coef_df.loc['Model', 'R2'] = r_squared
        coef_df.loc['Model', 'R2 adj'] = r_squared_adj
        coef_df.loc['Model', 'R2 [+-]'] = 1 * np.std(r_squared_l)
        coef_df.loc['Model', 'F2'] = F2
        coef_df.loc['Model', 'mse'] = mse
        coef_df.loc['Model', 'mse [+-]'] = 1 * np.std(mse_l)
        coef_df.loc['Model', 'rmse'] = rmse
        coef_df.loc['Model', 'rmse [+-]'] = 1 * np.std(rmse_l)
        coef_df.loc['Model', 'outcome var'] = np.var(y)
        coef_df.loc['Model', 'F'] = F
        coef_df.loc['Model', 'F-p_value'] = p_value
        coef_df.loc['Model', 'MDE'] = mda
        coef_df.loc['Model', 'MAE'] = mae

    if logger:
        logger.add_message(
            f"XGB Regression completed with R² = {r_squared_final:.4f}"
        )
        logger.save('xgb_regression')

    results_labels_df = results_labels_df[['ID', 'y_labels', 'y_pred']]

    return [coef_df, r_squared_final, results_labels_df]