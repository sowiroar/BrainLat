"""
Diagnostic graphics and visualization functions.

This module provides plotting utilities for regression diagnosis including:
- Coefficient/feature importance plots
- Residual diagnostics
- Predicted vs actual plots
- Distribution plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")


def plot_feature_importance(coef_df, results_df=None, figsize=(10, 6), ax=None):
    """
    Plot feature importances or coefficients as a horizontal bar chart.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Coefficients dataframe with 'Estimate mean' column and '_intercept' index
    results_df : pd.DataFrame, optional
        Results dataframe with 'y_labels' and 'y_pred' columns for computing R²
    figsize : tuple, default=(10, 6)
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    
    Examples
    --------
    >>> from brainlat.graphics import plot_feature_importance
    >>> plot_feature_importance(coef_df, results_df)
    >>> plt.show()
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Extract feature importances (exclude intercept)
    df_plot = coef_df.iloc[1:, 0:2].sort_values(
        by='Estimate mean',
        ascending=True
    )
    
    # Create horizontal bar plot
    ax.barh(range(len(df_plot)), df_plot['Estimate mean'])
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot.index)
    ax.set_xlabel('Feature Importance / Coefficient')
    ax.set_title('Feature Importances')
    
    # Add statistics if available
    if results_df is not None:
        r2 = np.round(coef_df.loc['_intercept', 'R2'], 3)
        rmse = np.round(coef_df.loc['_intercept', 'rmse'], 3)
        ax.set_title(f'Feature Importances (R²: {r2}, RMSE: {rmse})')
    
    return ax


def plot_predictions_vs_actual(y_true, y_pred, figsize=(8, 8), ax=None):
    """
    Plot predicted vs actual values with correlation.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    figsize : tuple, default=(8, 8)
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> from brainlat.graphics import plot_predictions_vs_actual
    >>> plot_predictions_vs_actual(y_true, y_pred)
    >>> plt.show()
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure numeric types
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Calculate correlation
    corr, p_val = pearsonr(y_true, y_pred)
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predictions vs Actual (r={corr:.3f}, p={p_val:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_residuals(y_true, y_pred, figsize=(12, 5), ax=None):
    """
    Plot residuals for regression diagnostics.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    figsize : tuple, default=(12, 5)
        Figure size
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axes object(s) to plot on
    
    Returns
    -------
    axes : matplotlib.axes.Axes or array of Axes
        The axes object(s)
    
    Examples
    --------
    >>> from brainlat.graphics import plot_residuals
    >>> plot_residuals(y_true, y_pred)
    >>> plt.show()
    """
    
    residuals = np.array(y_true) - np.array(y_pred)
    
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
    
    # Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    return axes


def plot_coef_with_error(coef_df, figsize=(10, 6), ax=None):
    """
    Plot coefficients with error bars.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Coefficients dataframe with 'Estimate mean' and 'Estimate std' columns
    figsize : tuple, default=(10, 6)
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> from brainlat.graphics import plot_coef_with_error
    >>> plot_coef_with_error(coef_df)
    >>> plt.show()
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Extract features (exclude intercept)
    df_plot = coef_df.iloc[1:, 0:2].sort_values(
        by='Estimate mean',
        ascending=True
    )
    
    # Get standard errors
    std_errors = df_plot['Estimate std'].values
    
    # Create bar plot with error bars
    x_pos = np.arange(len(df_plot))
    ax.barh(x_pos, df_plot['Estimate mean'], xerr=std_errors, capsize=5)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df_plot.index)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Coefficients with Standard Error')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
    
    return ax


def plot_gap_analysis(results_df, figsize=(12, 5), ax=None):
    """
    Plot analysis of prediction gaps and gap corrections.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with 'GAP' and 'GAP_corrected' columns
    figsize : tuple, default=(12, 5)
        Figure size
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axes object(s) to plot on
    
    Returns
    -------
    axes : matplotlib.axes.Axes or array of Axes
        The axes object(s)
    
    Examples
    --------
    >>> from brainlat.graphics import plot_gap_analysis
    >>> plot_gap_analysis(results_df)
    >>> plt.show()
    """
    
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
    
    # Original gaps
    axes[0].hist(results_df['GAP'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Original Gap')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Original Gaps')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Corrected gaps
    axes[1].hist(results_df['GAP_corrected'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Corrected Gap')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Corrected Gaps')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    return axes


def plot_diagnostic_panel(y_true, y_pred, coef_df=None, figsize=(15, 10)):
    """
    Create a comprehensive diagnostic panel with multiple plots.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    coef_df : pd.DataFrame, optional
        Coefficients dataframe for plotting importances
    figsize : tuple, default=(15, 10)
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : array of Axes
        Array of axes objects
    
    Examples
    --------
    >>> from brainlat.graphics import plot_diagnostic_panel
    >>> fig, axes = plot_diagnostic_panel(y_true, y_pred, coef_df)
    >>> plt.show()
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Predictions vs actual
    plot_predictions_vs_actual(y_true, y_pred, ax=axes[0, 0])
    
    # Residual plots
    residual_axes = [axes[0, 1], axes[1, 0]]
    plot_residuals(y_true, y_pred, ax=residual_axes)
    
    # Feature importances if available
    if coef_df is not None:
        plot_feature_importance(coef_df, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'No coefficient data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importances')
    
    plt.tight_layout()
    
    return fig, axes
