"""
Classification models for BrainLat.

This module provides classification models including:
- XGBClassifier with ROC curve analysis and repeated K-Fold cross-validation
- Metrics: AUC, F1, Accuracy, Recall, Precision, Confusion Matrix

All functions follow the BrainLat logging/diagnostics methodology.
"""

import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, RepeatedKFold
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    accuracy_score, recall_score, precision_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler

from .diagnostics import DataDiagnostics, Logger, generate_sanity_report

warnings.filterwarnings("ignore")


def classifier_xgb_roc(
    X, y,
    params=None,
    n_splits=10,
    n_repeats=2,
    test_size=0.2,
    random_state=10000,
    scale_range=(0.05, 0.95),
    num_boost_round=1000,
    early_stopping_rounds=10,
    log=False
):
    """
    XGBClassifier with Repeated K-Fold CV and ROC analysis.

    Trains an XGBoost classifier using Repeated K-Fold cross-validation,
    computing AUC, F1, accuracy, recall, precision, and confusion matrices
    for train, validation, and held-out test sets across all folds.

    Parameters
    ----------
    X : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix with feature names as columns.
    y : array-like, shape (n_samples,)
        Binary target variable (0 or 1).
    params : dict or None, default=None
        XGBoost parameters. If None, defaults to
        {'objective': 'binary:logistic', 'eval_metric': 'logloss'}.
    n_splits : int, default=10
        Number of K-Fold splits per repeat.
    n_repeats : int, default=2
        Number of K-Fold repeats.
    test_size : float, default=0.2
        Proportion of data held out as the final test set.
    random_state : int, default=10000
        Random seed for train/test split and CV.
    scale_range : tuple, default=(0.05, 0.95)
        MinMaxScaler range for feature scaling.
    num_boost_round : int, default=1000
        Maximum boosting rounds for XGBoost.
    early_stopping_rounds : int, default=10
        Early stopping patience.
    log : bool, default=False
        Whether to save logs and diagnostic reports to files.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'metrics' : dict
            Per-split metrics for 'train', 'val', 'test' sets.
            Each contains lists of: 'auc', 'f1_score', 'accuracy_score',
            'recall_score', 'precision_score', 'confusion_matrix'.
        - 'roc' : dict
            ROC curve data for the test set:
            'tpr_mean', 'tpr_std', 'tpr_upper', 'tpr_lower',
            'fpr_mean', 'auc_mean'.
        - 'summary' : dict
            Aggregated mean ± std for key metrics on the test set:
            'auc_mean', 'auc_std', 'f1_mean', 'f1_std',
            'accuracy_mean', 'accuracy_std', 'recall_mean', 'recall_std',
            'precision_mean', 'precision_std'.

    Examples
    --------
    >>> from brainlat.clasification_model import classifier_xgb_roc
    >>> import pandas as pd, numpy as np
    >>> X = pd.DataFrame(np.random.randn(200, 5), columns=[f'f{i}' for i in range(5)])
    >>> y = np.random.randint(0, 2, 200)
    >>> results = classifier_xgb_roc(X, y, log=True)
    >>> print(f"AUC: {results['summary']['auc_mean']:.3f}")
    """

    # Initialize logger if needed
    logger = Logger() if log else None

    if logger:
        logger.add_message("Starting XGB classification with ROC analysis...")

    # Perform sanity checks
    diagnostics = DataDiagnostics()
    report = diagnostics.check_data_quality(X, y, name="XGB Classifier Input")

    if log:
        report_str = generate_sanity_report(
            report, filename="brainlat_logs/xgb_classifier_sanity_check.txt"
        )
        print("\n" + report_str)
        logger.add_message(
            f"Data quality check completed: {report['n_samples']} samples, "
            f"{report['n_features']} features"
        )
    else:
        report_str = generate_sanity_report(report)
        print("\n" + report_str)

    # Validate binary target
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        msg = (
            f"Expected binary target (2 classes), got {len(unique_classes)} classes: "
            f"{unique_classes}"
        )
        if logger:
            logger.add_error(msg)
        raise ValueError(msg)

    # Default XGBoost parameters
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

    # Train/test split (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if logger:
        logger.add_message(
            f"Train/test split: {len(X_train_full)} train, {len(X_test)} test"
        )

    # Scale features
    scaler = MinMaxScaler(scale_range)
    X_train_full_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_full),
        columns=X_train_full.columns,
        index=X_train_full.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Repeated K-Fold CV
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    folds = list(cv.split(X_train_full_scaled, y_train_full))

    metric_names = [
        'auc', 'f1_score', 'accuracy_score',
        'recall_score', 'precision_score', 'confusion_matrix'
    ]
    results_metrics = {
        'train': {m: [] for m in metric_names},
        'val':   {m: [] for m in metric_names},
        'test':  {m: [] for m in metric_names}
    }

    # Convert to DataFrames for iloc indexing
    X_train_df = pd.DataFrame(X_train_full_scaled)
    y_train_df = pd.DataFrame(y_train_full)

    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        dtrain = xgb.DMatrix(
            X_train_df.iloc[train_idx, :],
            label=y_train_df.iloc[train_idx].values.ravel()
        )
        dval = xgb.DMatrix(
            X_train_df.iloc[val_idx, :],
            label=y_train_df.iloc[val_idx].values.ravel()
        )

        model = xgb.train(
            dtrain=dtrain,
            params=params,
            evals=[(dtrain, 'train'), (dval, 'val')],
            num_boost_round=num_boost_round,
            verbose_eval=False,
            early_stopping_rounds=early_stopping_rounds,
        )

        sets = [dtrain, dval, dtest]
        set_names = ['train', 'val', 'test']

        for i, ds_name in enumerate(set_names):
            y_preds = model.predict(sets[i])
            labels = sets[i].get_label()

            results_metrics[ds_name]['auc'].append(
                roc_auc_score(labels, y_preds)
            )
            results_metrics[ds_name]['f1_score'].append(
                f1_score(labels, np.round(y_preds))
            )
            results_metrics[ds_name]['accuracy_score'].append(
                accuracy_score(labels, np.round(y_preds))
            )
            results_metrics[ds_name]['recall_score'].append(
                recall_score(labels, np.round(y_preds))
            )
            results_metrics[ds_name]['precision_score'].append(
                precision_score(labels, np.round(y_preds), zero_division=0)
            )
            results_metrics[ds_name]['confusion_matrix'].append(
                confusion_matrix(labels, np.round(y_preds))
            )

    # Compute ROC curve statistics for test set
    total_folds = n_splits * n_repeats
    fpr_mean = np.linspace(0, 1, 100)
    interp_tprs = []

    for fold_i in range(total_folds):
        # Recompute ROC for each fold's test predictions
        # We stored AUC but not fpr/tpr directly, so recompute from stored data
        pass

    # Alternative: recompute ROC from the dtest predictions across folds
    # We need to store fpr/tpr per fold
    fpr_list = []
    tpr_list = []
    interp_tprs = []

    # Re-run to collect fpr/tpr (lightweight since we only predict)
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        dtrain = xgb.DMatrix(
            X_train_df.iloc[train_idx, :],
            label=y_train_df.iloc[train_idx].values.ravel()
        )
        dval = xgb.DMatrix(
            X_train_df.iloc[val_idx, :],
            label=y_train_df.iloc[val_idx].values.ravel()
        )

        model = xgb.train(
            dtrain=dtrain,
            params=params,
            evals=[(dtrain, 'train'), (dval, 'val')],
            num_boost_round=num_boost_round,
            verbose_eval=False,
            early_stopping_rounds=early_stopping_rounds,
        )

        y_preds_test = model.predict(dtest)
        fpr, tpr, _ = roc_curve(y_test, y_preds_test)

        interp_tpr = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std = 2 * np.std(interp_tprs, axis=0)
    tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
    tpr_lower = tpr_mean - tpr_std
    auc_mean = np.mean(results_metrics['test']['auc'])

    roc_data = {
        'fpr_mean': fpr_mean,
        'tpr_mean': tpr_mean,
        'tpr_std': tpr_std,
        'tpr_upper': tpr_upper,
        'tpr_lower': tpr_lower,
        'auc_mean': auc_mean
    }

    # Summary statistics
    summary = {}
    for metric in ['auc', 'f1_score', 'accuracy_score', 'recall_score', 'precision_score']:
        vals = results_metrics['test'][metric]
        key = metric.replace('_score', '')
        summary[f'{key}_mean'] = np.mean(vals)
        summary[f'{key}_std'] = np.std(vals)

    if logger:
        logger.add_message(
            f"XGB Classifier completed — "
            f"AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}, "
            f"F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}, "
            f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}"
        )
        logger.save('xgb_classifier')

    return {
        'metrics': results_metrics,
        'roc': roc_data,
        'summary': summary
    }