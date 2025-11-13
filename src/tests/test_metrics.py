"""Tests for metrics and analysis functions."""

import pytest
import numpy as np

from src.analysis.metrics import (
    rmse,
    mse,
    mae,
    relative_error,
    max_error,
    compute_energy,
    energy_drift,
    parameter_error,
    compute_all_metrics,
)


class TestBasicMetrics:
    """Tests for basic error metrics."""

    def test_rmse_zero(self):
        """Test RMSE with zero error."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])

        result = rmse(pred, target)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_rmse_nonzero(self):
        """Test RMSE with nonzero error."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 3.1])

        result = rmse(pred, target)
        assert result == pytest.approx(0.1, abs=1e-6)

    def test_mse(self):
        """Test MSE."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 3.1])

        result = mse(pred, target)
        assert result == pytest.approx(0.01, abs=1e-6)

    def test_mae(self):
        """Test MAE."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 3.1])

        result = mae(pred, target)
        assert result == pytest.approx(0.1, abs=1e-6)

    def test_max_error(self):
        """Test max error."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.2, 3.05])

        result = max_error(pred, target)
        assert result == pytest.approx(0.2, abs=1e-6)

    def test_relative_error(self):
        """Test relative error."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.2, 3.3])

        result = relative_error(pred, target)
        assert result > 0.0


class TestEnergyMetrics:
    """Tests for energy-related metrics."""

    def test_compute_energy(self):
        """Test energy computation."""
        theta = np.array([0.0, 0.1, 0.2])
        theta_dot = np.array([0.0, 0.1, 0.2])

        energy = compute_energy(theta, theta_dot, g=9.81, L=1.0, m=1.0)

        assert energy.shape == (3,)
        assert np.all(energy >= 0.0)

    def test_energy_drift(self):
        """Test energy drift computation."""
        theta = np.array([0.1, 0.09, 0.08, 0.07])
        theta_dot = np.array([0.0, 0.01, 0.02, 0.03])

        drift = energy_drift(theta, theta_dot, g=9.81, L=1.0, m=1.0)

        assert "mean_abs_drift" in drift
        assert "max_abs_drift" in drift
        assert "mean_rel_drift" in drift
        assert "max_rel_drift" in drift
        assert "final_rel_drift" in drift


class TestParameterError:
    """Tests for parameter error computation."""

    def test_parameter_error_zero(self):
        """Test parameter error with perfect match."""
        pred = {"g": 9.81, "L": 1.0}
        true = {"g": 9.81, "L": 1.0}

        errors = parameter_error(pred, true)

        assert errors["g_abs_error"] == pytest.approx(0.0, abs=1e-10)
        assert errors["L_abs_error"] == pytest.approx(0.0, abs=1e-10)

    def test_parameter_error_nonzero(self):
        """Test parameter error with mismatch."""
        pred = {"g": 10.0, "L": 1.1}
        true = {"g": 9.81, "L": 1.0}

        errors = parameter_error(pred, true)

        assert errors["g_abs_error"] == pytest.approx(0.19, abs=1e-6)
        assert errors["L_abs_error"] == pytest.approx(0.1, abs=1e-6)
        assert "g_rel_error" in errors
        assert "L_rel_error" in errors


class TestComputeAllMetrics:
    """Tests for comprehensive metrics computation."""

    def test_compute_all_metrics_basic(self):
        """Test basic metrics computation."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 3.1])

        metrics = compute_all_metrics(pred, target)

        assert "rmse_theta" in metrics
        assert "mse_theta" in metrics
        assert "mae_theta" in metrics
        assert "max_error_theta" in metrics

    def test_compute_all_metrics_with_velocity(self):
        """Test metrics with velocity."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 3.1])
        theta_dot_pred = np.array([0.1, 0.2, 0.3])
        theta_dot_true = np.array([0.1, 0.2, 0.3])

        metrics = compute_all_metrics(
            pred, target, theta_dot_pred=theta_dot_pred, theta_dot_true=theta_dot_true
        )

        assert "rmse_theta_dot" in metrics
        assert "mae_theta_dot" in metrics

    def test_compute_all_metrics_with_params(self):
        """Test metrics with parameters."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 2.1, 3.1])
        pred_params = {"g": 10.0}
        true_params = {"g": 9.81}

        metrics = compute_all_metrics(
            pred, target, predicted_params=pred_params, true_params=true_params
        )

        assert "g_abs_error" in metrics
        assert "g_rel_error" in metrics


class TestCoverage:
    """Tests for coverage metrics."""

    def test_coverage_perfect(self):
        """Test perfect coverage."""
        from src.analysis.metrics import compute_coverage

        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.9, 1.9, 2.9])
        upper = np.array([1.1, 2.1, 3.1])

        coverage = compute_coverage(pred, target, lower, upper)
        assert coverage == pytest.approx(1.0, abs=1e-6)

    def test_coverage_partial(self):
        """Test partial coverage."""
        from src.analysis.metrics import compute_coverage

        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.5, 3.0])  # 2.5 is outside bounds
        lower = np.array([0.9, 1.9, 2.9])
        upper = np.array([1.1, 2.1, 3.1])

        coverage = compute_coverage(pred, target, lower, upper)
        assert coverage == pytest.approx(2.0 / 3.0, abs=1e-6)

