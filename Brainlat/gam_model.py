"""
Generalized Additive Models (GAM) and Meta-GAM ensembling for BrainLat.
"""

import os
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from .stats import calculate_delta_aic
from .diagnostics import DataDiagnostics, Logger

def evaluate_gam_with_cv(X_train, y_train, param_grid):
    """
    Perform K-fold CV on training data to select best n_splines and lam for GAM.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target array.
    param_grid : dict
        Dictionary containing grid of n_splines and lam.
        
    Returns
    -------
    best_params : tuple
        (best_n_splines, best_lam)
    """
    from pygam import LinearGAM, s
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_score = -np.inf
    best_params = None

    for n_splines in param_grid['n_splines']:
        for lam in param_grid['lam']:
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                try:
                    gam = LinearGAM(s(0, n_splines=n_splines), lam=lam).fit(X_tr, y_tr)
                    val_pred = gam.predict(X_val)
                    delta_aic = calculate_delta_aic(y_val, val_pred, k=1)
                    cv_scores.append(delta_aic)
                except Exception:
                    cv_scores.append(-np.inf)

            mean_cv_score = np.mean(cv_scores)
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = (n_splines, lam)

    return best_params

def run_regressions_and_ensemble_cv(
    df_all,
    target_features,
    category_title,
    topn,
    outcome='BAG',
    normalize=True,
    n_cv_splits=5,
    file_save='',
    dosave=True,
    log=False
):
    """
    Run regression analysis using GAMs for multiple predictors and build a Meta-GAM ensemble.
    
    Parameters
    ----------
    df_all : pd.DataFrame
        Input data containing target features and outcome.
    target_features : list of str
        Predictors to evaluate.
    category_title : str
        Category title for plots and logging.
    topn : int
        Number of top individual models to select for ensembling.
    outcome : str, default='BAG'
        Target variable.
    normalize : bool, default=True
        Whether to normalize predictors X (Y is not normalized in this CV version).
    n_cv_splits : int, default=5
        Outer CV splits.
    file_save : str, default=''
        Directory prefix to save SVG, PNG, PDF charts.
    dosave : bool, default=True
        Whether to generate and save charts.
    log : bool, default=False
        Whether to generate log files.
        
    Returns
    -------
    results_df : pd.DataFrame
        Summary table comparing all models (individual and Meta-GAM).
    all_results : list of dict
        Detailed dictionary results for all models.
    """
    import lightgbm as lgb
    from pygam import LinearGAM, s
    from .graphics import plot_delta_aic_comparison, plot_gam_combined_curves
    
    logger = Logger() if log else None
    if logger:
        logger.add_message(f"Starting GAM regressions and ensemble CV for {category_title}...")
        
    predictor_cols = list(target_features)
    model_results = []
    param_grid = {
        'n_splines': [5],
        'lam': [0.05, 0.1, 0.5, 1, 5, 10]
    }
    all_predictor_data = {}
    
    if dosave and file_save:
        folder_path = os.path.join(file_save, outcome)
        os.makedirs(folder_path, exist_ok=True)
        if logger:
            logger.add_message(f"Created/verified plot output folder: {folder_path}")
            
    for predictor_col in predictor_cols:
        if predictor_col not in df_all.columns or outcome not in df_all.columns:
            continue
            
        subset_df = df_all[[predictor_col, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        if subset_df.empty:
            continue
            
        X = subset_df[[predictor_col]].values
        y = subset_df[outcome].values
        
        if np.unique(X).size < 2 or len(y) < 15:
            continue
            
        X_orig = X.copy()
        y_orig = y.copy()
        
        scaler_X_global = MinMaxScaler().fit(X)
        X_grid_common = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        
        fold_curves = []
        fold_delta_aics = []
        fold_p_values = []
        fold_mdes = []
        fold_test_predictions = {}
        
        kf_outer = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
        
        for fold_i, (train_idx, test_idx) in enumerate(kf_outer.split(X)):
            X_train_, X_test_ = X[train_idx], X[test_idx]
            y_train_, y_test_ = y[train_idx], y[test_idx]
            
            if len(y_train_) < 10 or len(y_test_) < 5:
                continue
                
            if normalize:
                scaler_X_fold = MinMaxScaler().fit(X_train_)
                X_train = scaler_X_fold.transform(X_train_)
                X_test = scaler_X_fold.transform(X_test_)
                X_grid_fold = scaler_X_fold.transform(X_grid_common)
            else:
                X_train = X_train_
                X_test = X_test_
                X_grid_fold = X_grid_common
                
            best_params = evaluate_gam_with_cv(X_train, y_train_, param_grid)
            if best_params is None:
                continue
            best_n_splines, best_lam = best_params
            
            try:
                gam = LinearGAM(s(0, n_splines=best_n_splines), lam=best_lam).fit(X_train, y_train_)
                y_grid = gam.predict(X_grid_fold)
                fold_curves.append(y_grid)
                
                predictions = gam.predict(X_test)
                fold_delta_aics.append(calculate_delta_aic(y_test_, predictions, k=1))
                
                X_pred_sm = sm.add_constant(predictions)
                ols_model = sm.OLS(y_test_, X_pred_sm).fit()
                if len(ols_model.pvalues) >= 2:
                    fold_p_values.append(ols_model.pvalues[1])
                else:
                    fold_p_values.append(1.0)
                    
                fold_mdes.append(np.mean(predictions - y_test_))
                fold_test_predictions[fold_i] = {
                    'test_idx': test_idx,
                    'predictions': predictions,
                    'y_test': y_test_
                }
            except Exception:
                continue
                
        if len(fold_curves) == 0:
            continue
            
        curves_array = np.array(fold_curves)
        mean_curve = curves_array.mean(axis=0)
        std_curve = curves_array.std(axis=0)
        cv_ci_lower = mean_curve - 2.56 * std_curve
        cv_ci_upper = mean_curve + 2.56 * std_curve
        
        delta_aics_arr = np.array(fold_delta_aics)
        mean_delta_aic = delta_aics_arr.mean()
        std_delta_aic = delta_aics_arr.std()
        ci_delta_aic = (
            mean_delta_aic - 2.56 * std_delta_aic,
            mean_delta_aic + 2.56 * std_delta_aic
        )
        
        mean_p_value = np.mean(fold_p_values) if fold_p_values else 1.0
        mean_mde = np.mean(fold_mdes)
        
        all_predictor_data[predictor_col] = {
            'fold_test_predictions': fold_test_predictions,
            'y_orig': y_orig,
            'X_orig': X_orig,
            'n': len(y_orig)
        }
        
        model_results.append({
            'predictor': f"GAM: {predictor_col}",
            'delta_aic': mean_delta_aic,
            'ci_delta_aic': ci_delta_aic,
            'p_value': mean_p_value,
            'mde': mean_mde,
            'predictions': mean_curve,
            'actual': y_orig,
            'X_grid': X_grid_common[:, 0],
            'mean_curve': mean_curve,
            'cv_ci_lower': cv_ci_lower,
            'cv_ci_upper': cv_ci_upper,
            'X_orig': X_orig,
            'y_orig': y_orig,
            'scaler_X_global': scaler_X_global,
            'original_order': predictor_cols.index(predictor_col)
        })
        
    model_results_sorted = sorted(model_results, key=lambda x: x['delta_aic'], reverse=True)[:topn]
    
    if len(model_results_sorted) == 0:
        if logger:
            logger.add_warning("No model comparisons succeeded.")
        return pd.DataFrame(), None
        
    # Ensembling via Meta-GAM (LightGBM)
    top_predictor_cols = [
        res['predictor'].replace("GAM: ", "") for res in model_results_sorted
    ]
    
    meta_fold_delta_aics = []
    meta_fold_p_values = []
    meta_fold_mdes = []
    
    kf_meta = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
    ref_col = top_predictor_cols[0]
    ref_data = all_predictor_data.get(ref_col)
    
    meta_results = []
    if ref_data is not None:
        ref_subset = df_all[[ref_col, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
        X_ref = ref_subset[[ref_col]].values
        y_ref = ref_subset[outcome].values
        
        for fold_i, (train_idx, test_idx) in enumerate(kf_meta.split(X_ref)):
            y_test_meta = y_ref[test_idx]
            
            fold_preds = []
            valid = True
            for pred_col in top_predictor_cols:
                pdata = all_predictor_data.get(pred_col)
                if pdata is None or fold_i not in pdata['fold_test_predictions']:
                    valid = False
                    break
                fold_preds.append(pdata['fold_test_predictions'][fold_i]['predictions'])
                
            if not valid:
                continue
                
            min_len = min(len(p) for p in fold_preds)
            X_meta_test = np.column_stack([
                np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(p)), p)
                for p in fold_preds
            ])
            y_meta_test = np.interp(
                np.linspace(0, 1, min_len), np.linspace(0, 1, len(y_test_meta)), y_test_meta
            )
            
            train_preds = []
            valid_train = True
            for pred_col in top_predictor_cols:
                pdata = all_predictor_data.get(pred_col)
                if pdata is None:
                    valid_train = False
                    break
                other_preds = [
                    finfo['predictions']
                    for fi, finfo in pdata['fold_test_predictions'].items()
                    if fi != fold_i
                ]
                if len(other_preds) == 0:
                    valid_train = False
                    break
                train_preds.append(np.concatenate(other_preds))
                
            if not valid_train:
                continue
                
            min_len_train = min(len(p) for p in train_preds)
            X_meta_train = np.column_stack([
                np.interp(np.linspace(0, 1, min_len_train), np.linspace(0, 1, len(p)), p)
                for p in train_preds
            ])
            
            other_y = [
                finfo['y_test']
                for fi, finfo in all_predictor_data[ref_col]['fold_test_predictions'].items()
                if fi != fold_i
            ]
            if len(other_y) == 0:
                continue
                
            y_meta_train_raw = np.concatenate(other_y)
            y_meta_train = np.interp(
                np.linspace(0, 1, min_len_train), np.linspace(0, 1, len(y_meta_train_raw)), y_meta_train_raw
            )
            
            # Fit meta-model
            meta_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
            meta_model.fit(X_meta_train, y_meta_train)
            ensemble_predictions = meta_model.predict(X_meta_test)
            
            if np.var(ensemble_predictions) < 1e-10:
                continue
                
            meta_fold_delta_aics.append(
                calculate_delta_aic(y_meta_test, ensemble_predictions, k=len(top_predictor_cols))
            )
            
            X_pred_meta_sm = sm.add_constant(ensemble_predictions, has_constant='add')
            ols_meta = sm.OLS(y_meta_test, X_pred_meta_sm).fit()
            if len(ols_meta.pvalues) >= 2:
                meta_fold_p_values.append(ols_meta.pvalues[1])
            else:
                meta_fold_p_values.append(1.0)
                
            meta_fold_mdes.append(np.mean(ensemble_predictions - y_meta_test))
            
        if len(meta_fold_delta_aics) > 0:
            meta_delta_arr = np.array(meta_fold_delta_aics)
            mean_delta_meta = meta_delta_arr.mean()
            std_delta_meta = meta_delta_arr.std()
            ci_delta_meta = (
                mean_delta_meta - 2.56 * std_delta_meta,
                mean_delta_meta + 2.56 * std_delta_meta
            )
            mean_p_meta = np.mean(meta_fold_p_values) if meta_fold_p_values else 1.0
            mean_mde_meta = np.mean(meta_fold_mdes)
            
            meta_results.append({
                'predictor': 'Meta-GAM: LightGBM',
                'delta_aic': mean_delta_meta,
                'ci_delta_aic': ci_delta_meta,
                'p_value': mean_p_meta,
                'mde': mean_mde_meta,
                'predictions': np.array([mean_delta_meta]),
                'actual': np.array([mean_delta_meta])
            })
            
    all_results = model_results_sorted + meta_results
    all_results = sorted(all_results, key=lambda x: x['delta_aic'], reverse=True)
    
    # Build results table
    results_df = pd.DataFrame({
        'Delta AIC [99% CI]': [
            f"{result['delta_aic']:.2f} [{result['ci_delta_aic'][0]:.2f}, {result['ci_delta_aic'][1]:.2f}]"
            for result in all_results
        ],
        'p-value': [
            '<0.001' if result['p_value'] < 0.001 else f"{result['p_value']:.4f}"
            for result in all_results
        ],
        'MDE': [
            f"{result['mde']:.4f}" for result in all_results
        ]
    }, index=[result['predictor'] for result in all_results])
    
    # Generate plots if requested
    if dosave and file_save:
        predictors = [r['predictor'] for r in all_results]
        delta_aics = [r['delta_aic'] for r in all_results]
        ci_errors = np.array([
            [r['delta_aic'] - r['ci_delta_aic'][0], r['ci_delta_aic'][1] - r['delta_aic']]
            for r in all_results
        ]).T
        pvals = [r['p_value'] for r in all_results]
        mdes = [r['mde'] for r in all_results]
        
        # Save Delta AIC bar chart
        plot_delta_aic_comparison(
            predictors, delta_aics, ci_errors, pvals, mdes,
            category_title, filename=os.path.join(folder_path, f"{category_title}_Metamodel")
        )
        
        # Save combined GAM curves
        best_gams = [res for res in model_results_sorted if res['predictor'].startswith("GAM:")]
        if best_gams:
            gams_info = []
            for res in best_gams:
                gams_info.append({
                    'predictor_name': res['predictor'].replace("GAM: ", ""),
                    'X_plot': res['X_orig'][:, 0] if res['X_orig'].ndim > 1 else res['X_orig'],
                    'y_plot': res['y_orig'],
                    'XX_plot': res['X_grid'],
                    'y_pred_plot': res['mean_curve'],
                    'y_lower_plot': res['cv_ci_lower'],
                    'y_upper_plot': res['cv_ci_upper'],
                    'delta_aic': res['delta_aic'],
                    'p_value': res['p_value']
                })
            plot_gam_combined_curves(
                gams_info, outcome, show_ci=True, normalize=normalize,
                filename=os.path.join(folder_path, f"{category_title}_GAM_combined")
            )
            
    if logger:
        logger.add_message("GAM CV regressions and Meta-GAM ensemble built successfully.")
        logger.save('gam_cv_ensemble')
        
    return results_df, all_results
