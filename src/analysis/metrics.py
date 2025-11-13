"""Metrics for evaluating PINN performance: RMSE, MSE, energy drift, coverage, ECE."""

import numpy as np
from typing import Dict, Optional, Tuple


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        predictions: predicted values
        targets: target values

    Returns:
        RMSE
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Squared Error.

    Args:
        predictions: predicted values
        targets: target values

    Returns:
        MSE
    """
    return np.mean((predictions - targets) ** 2)


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        predictions: predicted values
        targets: target values

    Returns:
        MAE
    """
    return np.mean(np.abs(predictions - targets))


def relative_error(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Relative error.

    Args:
        predictions: predicted values
        targets: target values
        epsilon: small constant to avoid division by zero

    Returns:
        relative error
    """
    return np.mean(np.abs(predictions - targets) / (np.abs(targets) + epsilon))


def max_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Maximum absolute error.

    Args:
        predictions: predicted values
        targets: target values

    Returns:
        max error
    """
    return np.max(np.abs(predictions - targets))


def compute_energy(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    g: float = 9.81,
    L: float = 1.0,
    m: float = 1.0,
) -> np.ndarray:
    """
    Compute total energy.

    Args:
        theta: angle
        theta_dot: angular velocity
        g: gravitational acceleration
        L: pendulum length
        m: pendulum mass

    Returns:
        total energy array
    """
    kinetic = 0.5 * m * L**2 * theta_dot**2
    potential = m * g * L * (1 - np.cos(theta))
    return kinetic + potential


def energy_drift(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    g: float = 9.81,
    L: float = 1.0,
    m: float = 1.0,
) -> Dict[str, float]:
    """
    Compute energy drift metrics.

    Args:
        theta: angle trajectory
        theta_dot: angular velocity trajectory
        g: gravitational acceleration
        L: pendulum length
        m: pendulum mass

    Returns:
        dictionary with drift metrics
    """
    energy = compute_energy(theta, theta_dot, g, L, m)
    E0 = energy[0]

    # Absolute drift
    abs_drift = energy - E0

    # Relative drift
    rel_drift = abs_drift / (np.abs(E0) + 1e-10)

    return {
        "mean_abs_drift": np.mean(np.abs(abs_drift)),
        "max_abs_drift": np.max(np.abs(abs_drift)),
        "mean_rel_drift": np.mean(np.abs(rel_drift)),
        "max_rel_drift": np.max(np.abs(rel_drift)),
        "final_rel_drift": np.abs(rel_drift[-1]),
    }


def parameter_error(
    predicted_params: Dict[str, float],
    true_params: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute parameter estimation errors.

    Args:
        predicted_params: predicted parameters
        true_params: true parameters

    Returns:
        dictionary of errors
    """
    errors = {}

    for key in true_params.keys():
        if key in predicted_params:
            pred = predicted_params[key]
            true = true_params[key]
            abs_error = abs(pred - true)
            rel_error = abs_error / (abs(true) + 1e-10)

            errors[f"{key}_abs_error"] = abs_error
            errors[f"{key}_rel_error"] = rel_error

    return errors


def compute_coverage(
    predictions: np.ndarray,
    targets: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        predictions: mean predictions (not used, but kept for API consistency)
        targets: true values
        lower_bound: lower confidence bound
        upper_bound: upper confidence bound

    Returns:
        coverage (fraction of targets within bounds)
    """
    within_bounds = (targets >= lower_bound) & (targets <= upper_bound)
    return np.mean(within_bounds)


def compute_prediction_interval_width(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> Dict[str, float]:
    """
    Compute statistics of prediction interval widths.

    Args:
        lower_bound: lower confidence bound
        upper_bound: upper confidence bound

    Returns:
        dictionary with width statistics
    """
    width = upper_bound - lower_bound

    return {
        "mean_width": np.mean(width),
        "median_width": np.median(width),
        "max_width": np.max(width),
        "min_width": np.min(width),
    }


def calibration_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for uncertainty quantification.

    Args:
        predictions: mean predictions
        targets: true values
        uncertainties: predicted standard deviations
        n_bins: number of bins

    Returns:
        expected_frequencies: expected frequencies for each bin
        observed_frequencies: observed frequencies for each bin
    """
    # Compute normalized errors
    errors = np.abs(predictions - targets)
    normalized_errors = errors / (uncertainties + 1e-10)

    # Create bins
    bins = np.linspace(0, 3, n_bins + 1)  # Up to 3 sigma
    expected_frequencies = np.zeros(n_bins)
    observed_frequencies = np.zeros(n_bins)

    for i in range(n_bins):
        # Expected frequency (from Gaussian assumption)
        # P(|Z| < z) where Z ~ N(0, 1)
        from scipy.stats import norm

        z_lower = bins[i]
        z_upper = bins[i + 1]
        expected_frequencies[i] = norm.cdf(z_upper) - norm.cdf(z_lower)

        # Observed frequency
        in_bin = (normalized_errors >= z_lower) & (normalized_errors < z_upper)
        observed_frequencies[i] = np.mean(in_bin)

    return expected_frequencies, observed_frequencies


def expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        predictions: mean predictions
        targets: true values
        uncertainties: predicted standard deviations
        n_bins: number of bins

    Returns:
        ECE
    """
    expected_freq, observed_freq = calibration_curve(
        predictions, targets, uncertainties, n_bins
    )

    ece = np.mean(np.abs(expected_freq - observed_freq))

    return ece


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    theta_dot_pred: Optional[np.ndarray] = None,
    theta_dot_true: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    predicted_params: Optional[Dict[str, float]] = None,
    true_params: Optional[Dict[str, float]] = None,
    g: float = 9.81,
    L: float = 1.0,
    m: float = 1.0,
) -> Dict[str, float]:
    """
    Compute all metrics.

    Args:
        predictions: predicted angles
        targets: true angles
        theta_dot_pred: predicted angular velocities
        theta_dot_true: true angular velocities
        uncertainties: predicted uncertainties
        lower_bound: lower confidence bound
        upper_bound: upper confidence bound
        predicted_params: predicted parameters
        true_params: true parameters
        g: gravitational acceleration
        L: pendulum length
        m: pendulum mass

    Returns:
        dictionary of all metrics
    """
    metrics = {}

    # Basic errors
    metrics["rmse_theta"] = rmse(predictions, targets)
    metrics["mse_theta"] = mse(predictions, targets)
    metrics["mae_theta"] = mae(predictions, targets)
    metrics["max_error_theta"] = max_error(predictions, targets)
    metrics["rel_error_theta"] = relative_error(predictions, targets)

    # Velocity errors (if available)
    if theta_dot_pred is not None and theta_dot_true is not None:
        metrics["rmse_theta_dot"] = rmse(theta_dot_pred, theta_dot_true)
        metrics["mae_theta_dot"] = mae(theta_dot_pred, theta_dot_true)

    # Energy drift (if velocities available)
    if theta_dot_pred is not None:
        drift_metrics = energy_drift(predictions, theta_dot_pred, g, L, m)
        metrics.update(drift_metrics)

    # Coverage (if bounds available)
    if lower_bound is not None and upper_bound is not None:
        metrics["coverage"] = compute_coverage(predictions, targets, lower_bound, upper_bound)

        width_metrics = compute_prediction_interval_width(lower_bound, upper_bound)
        metrics.update(width_metrics)

    # Calibration (if uncertainties available)
    if uncertainties is not None:
        metrics["ece"] = expected_calibration_error(predictions, targets, uncertainties)

    # Parameter errors (if available)
    if predicted_params is not None and true_params is not None:
        param_errors = parameter_error(predicted_params, true_params)
        metrics.update(param_errors)

    return metrics


def compute_parameter_ci_coverage(
    param_values: np.ndarray,
    true_value: float,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute CI coverage for a parameter from ensemble.
    
    Args:
        param_values: array of parameter estimates from ensemble [n_models]
        true_value: true parameter value
        confidence: confidence level
        
    Returns:
        dictionary with coverage and CI bounds
    """
    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bound = np.percentile(param_values, lower_percentile)
    upper_bound = np.percentile(param_values, upper_percentile)
    
    coverage = 1.0 if (lower_bound <= true_value <= upper_bound) else 0.0
    
    return {
        'coverage': coverage,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'width': upper_bound - lower_bound,
        'mean': np.mean(param_values),
        'std': np.std(param_values),
    }


def compute_ensemble_parameter_metrics(
    param_stats: Dict[str, Dict[str, float]],
    true_params: Dict[str, float],
    confidence_levels: list = None,
) -> Dict[str, Dict]:
    """
    Compute CI coverage and statistics for all parameters.
    
    Args:
        param_stats: dictionary with parameter statistics from ensemble
                     {'g': {'values': [...], 'mean': ..., 'std': ...}, ...}
        true_params: dictionary with true parameter values
        confidence_levels: list of confidence levels to compute
        
    Returns:
        dictionary with metrics for each parameter
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95]
    
    results = {}
    
    for param_name in ['g', 'L', 'damping']:
        if param_name not in param_stats or param_name not in true_params:
            continue
            
        param_values = np.array(param_stats[param_name]['values'])
        true_value = true_params[param_name]
        
        param_results = {
            'mean': param_stats[param_name]['mean'],
            'std': param_stats[param_name]['std'],
            'true_value': true_value,
            'abs_error': abs(param_stats[param_name]['mean'] - true_value),
            'rel_error': abs(param_stats[param_name]['mean'] - true_value) / abs(true_value) if true_value != 0 else 0,
        }
        
        # Compute coverage for each confidence level
        for conf in confidence_levels:
            ci_metrics = compute_parameter_ci_coverage(param_values, true_value, conf)
            param_results[f'ci_{int(conf*100)}'] = ci_metrics
        
        results[param_name] = param_results
    
    return results


def compute_trajectory_ci_coverage(
    predictions: np.ndarray,
    true_values: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute CI coverage for trajectory predictions.
    
    Args:
        predictions: array of predictions [n_models, n_points]
        true_values: true trajectory values [n_points]
        confidence: confidence level
        
    Returns:
        dictionary with coverage statistics
    """
    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    
    # Point-wise coverage
    within_bounds = (true_values >= lower_bound) & (true_values <= upper_bound)
    coverage = np.mean(within_bounds)
    
    # Average width
    width = upper_bound - lower_bound
    avg_width = np.mean(width)
    
    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'min_width': np.min(width),
        'max_width': np.max(width),
    }


def expected_calibration_error_params(
    param_values: np.ndarray,
    true_value: float,
    n_bins: int = 10,
) -> float:
    """
    Compute ECE for parameter estimates.
    
    This measures how well the empirical distribution of parameter estimates
    matches the expected confidence levels.
    
    Args:
        param_values: array of parameter estimates [n_models]
        true_value: true parameter value
        n_bins: number of bins for calibration
        
    Returns:
        ECE value
    """
    # Compute empirical CDF
    sorted_values = np.sort(param_values)
    n = len(sorted_values)
    
    # Find where true value falls in the distribution
    rank = np.searchsorted(sorted_values, true_value)
    empirical_quantile = rank / n
    
    # For a well-calibrated ensemble, the true value should be uniformly
    # distributed across quantiles. We measure deviation from uniform.
    # This is a simplified ECE for parameter estimation.
    
    # Compute expected quantile (should be 0.5 for symmetric distribution)
    expected_quantile = 0.5
    
    # Return absolute deviation
    return abs(empirical_quantile - expected_quantile)

