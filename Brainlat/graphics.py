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


def plot_delta_aic_comparison(predictors, delta_aics, ci_errors, pvals, mdes, category_title, filename=None):
    """
    Plot horizontal bar chart for Delta AIC comparisons with error bars.
    
    Parameters
    ----------
    predictors : list of str
        Predictor names.
    delta_aics : list of float
        Delta AIC values.
    ci_errors : array-like, shape (2, n_predictors)
        Error bars: [delta_aic - lower_ci, upper_ci - delta_aic].
    pvals : list of float
        P-values.
    mdes : list of float
        MDE values.
    category_title : str
        Title category (e.g. 'Social Exposomes').
    filename : str, optional
        Filename to save plot (formats derived: .svg, .png, .pdf).
    """
    import matplotlib.pyplot as plt
    import os
    
    plt.figure(figsize=(12, 8))
    
    # Plot horizontal bars
    bars = plt.barh(predictors[::-1], delta_aics[::-1], color='skyblue', xerr=ci_errors[:, ::-1], capsize=5)
    plt.xlabel('Delta AIC')
    plt.title(f'{category_title} - Model Comparison by Delta AIC')
    plt.subplots_adjust(left=0.4)
    
    # Legend with metrics
    legend_entries = []
    for pred, d_aic, ci_err_col, p_val, mde in zip(predictors, delta_aics, ci_errors.T, pvals, mdes):
        p_str = '<0.001' if p_val < 0.001 else f"{p_val:.4f}"
        low_ci = d_aic - ci_err_col[0]
        high_ci = d_aic + ci_err_col[1]
        stats_text = f"{pred}\nMDE: {mde:.4f}\nDelta AIC: {d_aic:.2f} [{low_ci:.2f}, {high_ci:.2f}]\np-value: {p_str}"
        legend_entries.append(plt.Rectangle((0, 0), 1, 1, fc='skyblue', alpha=0.5, label=stats_text))
        
    plt.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if filename:
        base, _ = os.path.splitext(filename)
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(base + ext, format=ext.replace('.', ''), bbox_inches='tight')
    plt.close()


def plot_gam_combined_curves(gams_info, outcome, show_ci=True, normalize=True, filename=None):
    """
    Plot combined GAM curves for multiple predictors.
    
    Parameters
    ----------
    gams_info : list of dict
        Each dict contains keys: 'predictor_name', 'X_plot', 'y_plot', 'XX_plot', 'y_pred_plot',
        'y_lower_plot', 'y_upper_plot', 'delta_aic', 'p_value'
    outcome : str
        Outcome variable name (e.g. 'BAG').
    show_ci : bool, default=True
        Whether to plot confidence interval bands.
    normalize : bool, default=True
        Whether data is normalized.
    filename : str, optional
        Filename to save plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']
    colors = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, res in enumerate(gams_info):
        pred_name = res['predictor_name']
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        X_plot = np.asarray(res['X_plot'])
        y_plot = np.asarray(res['y_plot'])
        XX_plot = np.asarray(res['XX_plot'])
        y_pred_plot = np.asarray(res['y_pred_plot'])
        y_lower_plot = np.asarray(res['y_lower_plot'])
        y_upper_plot = np.asarray(res['y_upper_plot'])
        
        p_val = res['p_value']
        p_value_str = '<0.001' if p_val < 0.001 else f"{p_val:.4f}"
        label = f"{pred_name} | ΔAIC: {res['delta_aic']:.2f} | p: {p_value_str}"
        
        # Scatter plot with horizontal jitter
        jitter = np.random.uniform(-0.02, 0.02, size=len(X_plot))
        ax.scatter(X_plot + jitter, y_plot, alpha=0.2, color=color, marker=marker, s=15)
        ax.plot(XX_plot, y_pred_plot, color=color, linestyle='-', linewidth=2, label=label)
        
        if show_ci:
            ax.fill_between(XX_plot, y_lower_plot, y_upper_plot, color=color, alpha=0.15)
            
    xlabel = 'Exposomes (normalized 0-1)' if normalize else 'Exposomes'
    ylabel = f'{outcome} (normalized 0-1)' if normalize else outcome
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(False)
    plt.tight_layout()
    
    if filename:
        base, _ = os.path.splitext(filename)
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(base + ext, format=ext.replace('.', ''), bbox_inches='tight')
    plt.close()


def plot_or_rr_forest(df_plot, effect_type='OR', filename=None):
    """
    Generate forest plot for Odds Ratios or Relative Risks.
    
    Parameters
    ----------
    df_plot : pd.DataFrame
        DataFrame containing columns: 'Feature', effect_type ('OR' or 'RR'), '2.5%', '97.5%'
    effect_type : str, default='OR'
        Type of effect: 'OR' or 'RR'
    filename : str, optional
        Filename to save plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    plt.figure(figsize=(8, 6))
    
    # Reverse order to plot top feature at the top
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)
    
    features = df_plot['Feature']
    effects = np.asarray(df_plot[effect_type], dtype=float)
    lower = np.asarray(df_plot['2.5%'], dtype=float)
    upper = np.asarray(df_plot['97.5%'], dtype=float)
    
    # Error bar values: (effect - lower, upper - effect)
    xerr = np.array([effects - lower, upper - effects])
    
    plt.errorbar(effects, range(len(features)), xerr=xerr, fmt="none", c="k", capsize=5)
    plt.scatter(effects, range(len(features)), color="red", zorder=10)
    
    plt.yticks(range(len(features)), features)
    plt.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5)
    
    plt.xlabel(f"{effect_type} Value")
    plt.ylabel("Features")
    plt.title(f"{effect_type} Forest Plot [95% CI]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if filename:
        base, _ = os.path.splitext(filename)
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(base + ext, format=ext.replace('.', ''), bbox_inches='tight')
    plt.close()


def plot_density_scatter(
    data,
    x_col,
    y_col,
    label=None,
    color='red',
    levels=10,
    fill=True,
    alpha=1.0,
    figsize=(5, 4.5),
    ax=None,
    filename=None
):
    """
    2D KDE density contour plot using seaborn.

    Creates a kernel density estimate contour plot showing the
    bivariate distribution of two variables.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the columns to plot.
    x_col : str
        Column name for the x-axis variable.
    y_col : str
        Column name for the y-axis variable.
    label : str or None, default=None
        Legend label for the contours.
    color : str, default='red'
        Color for the KDE contours.
    levels : int, default=10
        Number of contour levels.
    fill : bool, default=True
        Whether to fill contours.
    alpha : float, default=1.0
        Transparency of contours.
    figsize : tuple, default=(5, 4.5)
        Figure size.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to plot on. If None, creates a new figure.
    filename : str or None, default=None
        If provided, saves the plot to this path (.png, .svg, .pdf).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from brainlat.graphics import plot_density_scatter
    >>> import pandas as pd, numpy as np
    >>> data = pd.DataFrame({'age': np.random.normal(50, 10, 500),
    ...                      'score': np.random.normal(55, 12, 500)})
    >>> ax = plot_density_scatter(data, 'age', 'score', label='HC')
    >>> plt.show()
    """
    import os

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(
        data=data,
        x=x_col,
        y=y_col,
        color=color,
        label=label,
        levels=levels,
        alpha=alpha,
        fill=fill,
        ax=ax
    )

    if label is not None:
        ax.plot([], [], color=color, label=label)
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc='upper left',
            borderaxespad=0., frameon=False
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if filename:
        base, _ = os.path.splitext(filename)
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(base + ext, format=ext.replace('.', ''), bbox_inches='tight')

    return ax


def plot_radial_bar(
    df,
    labels_col,
    values_col,
    color_col=None,
    secondary_col=None,
    inner_radius=2.0,
    outer_radius=7.0,
    cmap='magma_r',
    show_secondary=True,
    title='',
    figsize=(6, 8),
    filename=None
):
    """
    Radial (polar) bar chart with optional secondary indicator.

    Creates a circular bar chart where each bar represents a category,
    with bar length proportional to a primary value and optional dots
    on dashed lines for a secondary value.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the columns to plot.
    labels_col : str
        Column with category labels (plotted around the circle).
    values_col : str
        Column with primary values (bar length).
    color_col : str or None, default=None
        Column used for bar coloring. If None, uses values_col.
    secondary_col : str or None, default=None
        Column for secondary indicator (dashed lines + dots).
    inner_radius : float, default=2.0
        Size of the central hole.
    outer_radius : float, default=7.0
        Maximum radius of the chart.
    cmap : str, default='magma_r'
        Matplotlib colormap name.
    show_secondary : bool, default=True
        Whether to show the secondary indicator.
    title : str, default=''
        Chart title.
    figsize : tuple, default=(6, 8)
        Figure size.
    filename : str or None, default=None
        If provided, saves the plot (.png, .svg, .pdf).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The polar axes object.

    Examples
    --------
    >>> from brainlat.graphics import plot_radial_bar
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({
    ...     'region': ['A', 'B', 'C', 'D', 'E'],
    ...     'score': np.random.randint(100, 500, 5),
    ...     'count': np.random.randint(10, 100, 5)
    ... })
    >>> fig, ax = plot_radial_bar(df, 'region', 'score', color_col='count')
    >>> plt.show()
    """
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    import os

    N = len(df)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N * 0.9

    # Normalize values for bar height
    norm_val = Normalize(df[values_col].min(), df[values_col].max())
    bar_height = norm_val(df[values_col]) * (outer_radius - inner_radius)

    # Color mapping
    color_data = df[color_col] if color_col is not None else df[values_col]
    norm_color = Normalize(color_data.min(), color_data.max())
    colormap = get_cmap(cmap)
    colors = colormap(norm_color(color_data))

    # Create polar figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")
    ax.set_ylim(0, outer_radius + 0.5)

    # Main bars
    ax.bar(
        angles, bar_height, width=width, bottom=inner_radius,
        color=colors, edgecolor="none", alpha=0.9, zorder=2
    )

    # Secondary indicator (dashed lines + dots)
    if show_secondary and secondary_col is not None and secondary_col in df.columns:
        norm_sec = Normalize(df[secondary_col].min(), df[secondary_col].max())
        sec_height = norm_sec(df[secondary_col]) * (outer_radius - inner_radius)

        for ang, h in zip(angles, sec_height):
            r_end = inner_radius + h
            ax.plot(
                [ang, ang], [inner_radius, r_end],
                linestyle="--", linewidth=0.7, color="black",
                alpha=0.5, zorder=1
            )
            ax.scatter(ang, r_end, s=20, color="black", zorder=4)

    # Labels around the circle
    for ang, label in zip(angles, df[labels_col]):
        r_text = outer_radius + 0.3
        rotation = np.degrees(ang)
        if np.pi / 2 < ang < 3 * np.pi / 2:
            rotation += 180
            align = "right"
        else:
            align = "left"

        ax.text(
            ang, r_text, str(label),
            ha=align, va="center",
            rotation=rotation, rotation_mode="anchor",
            fontsize=9
        )

    # Style
    ax.set_yticks(np.linspace(inner_radius, outer_radius, 4))
    ax.set_yticklabels([])
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
    ax.set_xticks([])

    if title:
        plt.title(title, fontsize=14, pad=30, fontweight="bold")

    # Colorbar
    cax = plt.axes([0.25, 0.08, 0.5, 0.02])
    cb = ColorbarBase(
        cax, cmap=colormap, norm=norm_color, orientation="horizontal"
    )
    color_label = color_col if color_col is not None else values_col
    cb.set_label(color_label)

    plt.tight_layout(rect=[0, 0.1, 1, 0.90])

    if filename:
        base, _ = os.path.splitext(filename)
        for ext_fmt in ['.png', '.svg', '.pdf']:
            plt.savefig(base + ext_fmt, format=ext_fmt.replace('.', ''), bbox_inches='tight')

    return fig, ax


def plot_roc_curve(
    roc_data,
    figsize=(8, 8),
    use_plotly=False,
    title='ROC Curve',
    filename=None
):
    """
    Plot ROC curve with confidence band from classifier results.

    Visualizes the mean ROC curve with ±2σ confidence band from
    repeated cross-validation results, as returned by
    ``classifier_xgb_roc()``.

    Parameters
    ----------
    roc_data : dict
        ROC data dictionary with keys: 'fpr_mean', 'tpr_mean',
        'tpr_upper', 'tpr_lower', 'auc_mean'.
        Typically from ``classifier_xgb_roc()['roc']``.
    figsize : tuple, default=(8, 8)
        Figure size.
    use_plotly : bool, default=False
        If True, uses Plotly for an interactive chart.
        If False, uses matplotlib.
    title : str, default='ROC Curve'
        Plot title.
    filename : str or None, default=None
        If provided, saves the plot (.png, .svg, .pdf).

    Returns
    -------
    fig : figure object
        Matplotlib Figure or Plotly Figure.

    Examples
    --------
    >>> from brainlat.clasification_model import classifier_xgb_roc
    >>> from brainlat.graphics import plot_roc_curve
    >>> results = classifier_xgb_roc(X, y)
    >>> fig = plot_roc_curve(results['roc'])
    >>> plt.show()
    """
    import os

    fpr_mean = roc_data['fpr_mean']
    tpr_mean = roc_data['tpr_mean']
    tpr_upper = roc_data['tpr_upper']
    tpr_lower = roc_data['tpr_lower']
    auc_val = roc_data['auc_mean']

    if use_plotly:
        try:
            import plotly.graph_objects as go

            fig = go.Figure([
                go.Scatter(
                    x=fpr_mean, y=tpr_upper,
                    line=dict(color='rgba(52, 152, 219, 0.5)', width=1),
                    hoverinfo="skip", showlegend=False, name='upper'
                ),
                go.Scatter(
                    x=fpr_mean, y=tpr_lower,
                    fill='tonexty',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(52, 152, 219, 0.5)', width=1),
                    hoverinfo="skip", showlegend=False, name='lower'
                ),
                go.Scatter(
                    x=fpr_mean, y=tpr_mean,
                    line=dict(color='rgba(41, 128, 185, 1.0)', width=2),
                    hoverinfo="skip", showlegend=True,
                    name=f'AUC: {auc_val:.3f}'
                )
            ])

            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig.update_layout(
                title=title,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=figsize[0] * 100, height=figsize[1] * 100
            )

            return fig

        except ImportError:
            warnings.warn("Plotly not installed, falling back to matplotlib.")

    # Matplotlib version
    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        fpr_mean, tpr_lower, tpr_upper,
        color='royalblue', alpha=0.2, label='±2σ CI'
    )
    ax.plot(
        fpr_mean, tpr_mean,
        color='royalblue', linewidth=2,
        label=f'Mean ROC (AUC = {auc_val:.3f})'
    )
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    if filename:
        base, _ = os.path.splitext(filename)
        for ext in ['.png', '.svg', '.pdf']:
            plt.savefig(base + ext, format=ext.replace('.', ''), bbox_inches='tight')

    return fig


def plot_sfs_frequency(df_freq, threshold=0.5, figsize=(10, 6), ax=None):
    """
    Plot SFS selection frequencies as a horizontal bar chart.

    Parameters
    ----------
    df_freq : pd.DataFrame
        DataFrame with columns 'Feature' and 'Frequency', sorted in descending order.
    threshold : float, default=0.5
        Robustness threshold for stability selection (0.0 to 1.0).
    figsize : tuple, default=(10, 6)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Sort in ascending order of frequency for horizontal plot
    df_plot = df_freq.sort_values(by='Frequency', ascending=True)

    # Assign colors based on threshold
    colors = ['#3498db' if f >= threshold else '#bdc3c7' for f in df_plot['Frequency']]

    bars = ax.barh(range(len(df_plot)), df_plot['Frequency'], color=colors, edgecolor='none')
    
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['Feature'], fontsize=10)
    
    # Add vertical line at threshold
    ax.axvline(x=threshold, color='#e74c3c', linestyle='--', linewidth=2, label=f'Threshold ({threshold*100}%)')
    
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('Selection Frequency', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('SFS Feature Selection Frequency (Stability Selection)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)

    # Add frequency values at the end of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{width*100:.1f}%',
            va='center',
            ha='left',
            fontsize=9,
            color='#2c3e50'
        )

    return ax


def plot_survival_curves(df, group_col, duration_col, event_col, figsize=(10, 6), ax=None):
    """
    Plot Kaplan-Meier survival curves for groups defined by group_col.
    Also performs a Log-Rank test to compare survival functions if there are 2 or more groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the variables.
    group_col : str
        Column defining the groups (e.g. "GAP_corrected_bin" or "Group").
    duration_col : str
        Column with time-to-event values.
    event_col : str
        Column with binary indicators for events (1 if event observed, 0 if censored).
    figsize : tuple, default=(10, 6)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    groups = df[group_col].dropna().unique()
    kmf = KaplanMeierFitter()

    for group in sorted(groups):
        mask = df[group_col] == group
        df_group = df[mask].dropna(subset=[duration_col, event_col])
        if len(df_group) == 0:
            continue
        kmf.fit(
            df_group[duration_col],
            event_observed=df_group[event_col],
            label=f"{group_col}: {group}"
        )
        kmf.plot_survival_function(ax=ax, ci_show=True)

    # Perform Log-Rank test if there are exactly 2 groups
    if len(groups) == 2:
        g1_mask = df[group_col] == groups[0]
        g2_mask = df[group_col] == groups[1]
        df_g1 = df[g1_mask].dropna(subset=[duration_col, event_col])
        df_g2 = df[g2_mask].dropna(subset=[duration_col, event_col])
        
        if len(df_g1) > 0 and len(df_g2) > 0:
            lr_res = logrank_test(
                df_g1[duration_col],
                df_g2[duration_col],
                event_observed_A=df_g1[event_col],
                event_observed_B=df_g2[event_col]
            )
            ax.set_title(f"Kaplan-Meier Survival Curves (Log-Rank p-value: {lr_res.p_value:.4e})", fontsize=14)
        else:
            ax.set_title("Kaplan-Meier Survival Curves", fontsize=14)
    else:
        ax.set_title("Kaplan-Meier Survival Curves", fontsize=14)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.grid(True, alpha=0.3)

    return ax



