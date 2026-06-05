"""
Diagnostics module for data quality checks and logging.

Provides sanity checks, outlier detection, and logging utilities
for regression models.
"""

import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
from scipy import stats


class DataDiagnostics:
    """Data quality diagnostics and sanity checks."""
    
    @staticmethod
    def check_data_quality(X, y, name="Dataset"):
        """
        Perform comprehensive data quality checks.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Target variable
        name : str, default="Dataset"
            Name for the report
            
        Returns
        -------
        report : dict
            Dictionary with diagnostic information
        """
        report = {
            'name': name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'features': list(X.columns),
            'dtype_check': {},
            'missing_values': {},
            'outliers': {},
            'warnings': [],
            'errors': []
        }
        
        # Check data types
        for col in X.columns:
            dtype = X[col].dtype
            report['dtype_check'][col] = str(dtype)
            
            # Warn if categorical/object type
            if dtype == 'object':
                warning_msg = f"Column '{col}' is categorical (object type). Consider encoding."
                report['warnings'].append(warning_msg)
                warnings.warn(warning_msg, UserWarning)
        
        # Check for missing values
        for col in X.columns:
            n_missing = X[col].isna().sum()
            pct_missing = (n_missing / len(X)) * 100
            report['missing_values'][col] = {
                'count': int(n_missing),
                'percentage': float(np.round(pct_missing, 2))
            }
            
            if n_missing > 0:
                warning_msg = f"Column '{col}' has {n_missing} missing values ({pct_missing:.2f}%)"
                report['warnings'].append(warning_msg)
                warnings.warn(warning_msg, UserWarning)
        
        # Target missing values
        y_array = np.asarray(y)
        y_missing = np.isnan(y_array).sum() if y_array.dtype.kind == 'f' else 0
        if y_missing > 0:
            warning_msg = f"Target has {y_missing} missing values"
            report['warnings'].append(warning_msg)
            warnings.warn(warning_msg, UserWarning)
        
        # Outlier detection (IQR method)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                n_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                pct_outliers = (n_outliers / len(X)) * 100
                
                report['outliers'][col] = {
                    'count': int(n_outliers),
                    'percentage': float(np.round(pct_outliers, 2)),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                
                if pct_outliers > 10:
                    severity = "SEVERE" if pct_outliers > 20 else "MODERATE"
                    warning_msg = f"[{severity}] Column '{col}' has {pct_outliers:.2f}% outliers"
                    report['warnings'].append(warning_msg)
                    warnings.warn(warning_msg, UserWarning)
        
        # Check for zero variance
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if X[col].std() == 0:
                    error_msg = f"Column '{col}' has zero variance (constant values)"
                    report['errors'].append(error_msg)
                    warnings.warn(error_msg, UserWarning)
        
        # Target statistics
        y_array = np.asarray(y, dtype=float)
        report['target_stats'] = {
            'mean': float(np.mean(y_array)),
            'std': float(np.std(y_array)),
            'min': float(np.min(y_array)),
            'max': float(np.max(y_array))
        }
        
        return report
    
    @staticmethod
    def detect_outliers(X, method='iqr', threshold=1.5):
        """
        Detect outliers in features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        method : str, default='iqr'
            Method: 'iqr' for Interquartile Range or 'zscore'
        threshold : float, default=1.5
            Threshold for outlier detection
            
        Returns
        -------
        outlier_mask : np.ndarray
            Boolean array indicating outlier rows
        outlier_info : dict
            Detailed outlier information per feature
        """
        outlier_mask = np.zeros(len(X), dtype=bool)
        outlier_info = {}
        
        if method == 'iqr':
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                    
                    col_outliers = (X[col] < lower) | (X[col] > upper)
                    outlier_mask |= col_outliers
                    outlier_info[col] = {
                        'n_outliers': int(col_outliers.sum()),
                        'lower_bound': float(lower),
                        'upper_bound': float(upper)
                    }
        
        elif method == 'zscore':
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    z_scores = np.abs(stats.zscore(X[col].dropna()))
                    col_outliers = np.abs(stats.zscore(X[col])) > threshold
                    outlier_mask |= col_outliers
                    outlier_info[col] = {
                        'n_outliers': int(col_outliers.sum()),
                        'threshold': threshold
                    }
        
        return outlier_mask, outlier_info


class Logger:
    """Simple logging system for regression models."""
    
    def __init__(self, log_dir='brainlat_logs'):
        """
        Initialize logger.
        
        Parameters
        ----------
        log_dir : str, default='brainlat_logs'
            Directory to store log files
        """
        self.log_dir = log_dir
        self.messages = []
        self.warnings = []
        self.errors = []
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def add_message(self, message):
        """Add a message to log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.messages.append(full_msg)
        print(full_msg)
    
    def add_warning(self, warning):
        """Add a warning to log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] WARNING: {warning}"
        self.warnings.append(full_msg)
        warnings.warn(warning, UserWarning)
    
    def add_error(self, error):
        """Add an error to log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] ERROR: {error}"
        self.errors.append(full_msg)
        warnings.warn(error, UserWarning)
    
    def save(self, filename):
        """
        Save logs to file.
        
        Parameters
        ----------
        filename : str
            Filename (without extension, will add .log)
        """
        filepath = os.path.join(self.log_dir, f"{filename}.log")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"BrainLat Log File: {filename}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            if self.messages:
                f.write("MESSAGES:\n")
                f.write("-" * 70 + "\n")
                for msg in self.messages:
                    f.write(msg + "\n")
                f.write("\n")
            
            if self.warnings:
                f.write("WARNINGS:\n")
                f.write("-" * 70 + "\n")
                for warn in self.warnings:
                    f.write(warn + "\n")
                f.write("\n")
            
            if self.errors:
                f.write("ERRORS:\n")
                f.write("-" * 70 + "\n")
                for err in self.errors:
                    f.write(err + "\n")
                f.write("\n")
        
        print(f"\n✓ Log saved to: {filepath}")


def generate_sanity_report(diagnostics_report, filename=None):
    """
    Generate a formatted sanity check report.
    
    Parameters
    ----------
    diagnostics_report : dict
        Report from DataDiagnostics.check_data_quality()
    filename : str or None
        If provided, save report to file
        
    Returns
    -------
    report_str : str
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"SANITY CHECK REPORT: {diagnostics_report['name']}")
    lines.append(f"Timestamp: {diagnostics_report['timestamp']}")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Samples: {diagnostics_report['n_samples']}")
    lines.append(f"Features: {diagnostics_report['n_features']}")
    lines.append(f"  - {', '.join(diagnostics_report['features'][:5])}" + 
                 (f" ... +{len(diagnostics_report['features'])-5} more" 
                  if len(diagnostics_report['features']) > 5 else ""))
    lines.append("")
    
    # Data types
    lines.append("DATA TYPES")
    lines.append("-" * 70)
    for col, dtype in diagnostics_report['dtype_check'].items():
        lines.append(f"  {col}: {dtype}")
    lines.append("")
    
    # Missing values
    lines.append("MISSING VALUES")
    lines.append("-" * 70)
    total_missing = 0
    for col, info in diagnostics_report['missing_values'].items():
        if info['count'] > 0:
            lines.append(f"  {col}: {info['count']} ({info['percentage']:.2f}%)")
            total_missing += info['count']
    if total_missing == 0:
        lines.append("  No missing values detected ✓")
    lines.append("")
    
    # Outliers
    lines.append("OUTLIERS (IQR Method)")
    lines.append("-" * 70)
    total_outliers = 0
    for col, info in diagnostics_report['outliers'].items():
        if info['count'] > 0:
            severity = "SEVERE" if info['percentage'] > 20 else "MODERATE" if info['percentage'] > 10 else "MILD"
            lines.append(f"  {col}: {info['count']} ({info['percentage']:.2f}%) [{severity}]")
            total_outliers += info['count']
    if total_outliers == 0:
        lines.append("  No significant outliers detected ✓")
    lines.append("")
    
    # Warnings and Errors
    if diagnostics_report['warnings']:
        lines.append("WARNINGS")
        lines.append("-" * 70)
        for warning in diagnostics_report['warnings']:
            lines.append(f"  ⚠ {warning}")
        lines.append("")
    
    if diagnostics_report['errors']:
        lines.append("ERRORS")
        lines.append("-" * 70)
        for error in diagnostics_report['errors']:
            lines.append(f"  ✗ {error}")
        lines.append("")
    
    # Target statistics
    lines.append("TARGET STATISTICS")
    lines.append("-" * 70)
    ts = diagnostics_report['target_stats']
    lines.append(f"  Mean:  {ts['mean']:.4f}")
    lines.append(f"  Std:   {ts['std']:.4f}")
    lines.append(f"  Min:   {ts['min']:.4f}")
    lines.append(f"  Max:   {ts['max']:.4f}")
    lines.append("")
    
    lines.append("=" * 70)
    
    report_str = "\n".join(lines)
    
    # Save if filename provided
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"✓ Report saved to: {filename}")
    
    return report_str
