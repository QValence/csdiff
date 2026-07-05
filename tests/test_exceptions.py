"""
Tests for exception and warning behaviour.
"""
import numpy as np
import pytest
import warnings as _warnings

from csdiff import (
    derivative,
    gradient,
    jacobian,
    directional_derivative,
    ComplexStepError,
    NonanalyticWarning,
    StepSizeWarning,
)


# ------------------------------------------------------------------ #
#   ComplexStepError — function that cannot accept complex input
# ------------------------------------------------------------------ #

def _float_only(x):
    """
    Mimics a C extension that cannot handle complex input.

    Explicitly rejects complex arrays with TypeError, matching the behaviour
    of many C/Fortran extensions that cast inputs to double internally.
    """
    if np.iscomplexobj(x):
        raise TypeError(
            "_float_only does not accept complex-valued input (simulated C extension)."
        )
    return np.sum(x**2)


def test_derivative_complex_step_error():
    """derivative() raises ComplexStepError for a float-only function."""
    def f(x):
        return float(x) ** 2  # float() raises TypeError on complex

    with pytest.raises(ComplexStepError):
        derivative(f, 1.0)


def test_gradient_complex_step_error():
    """gradient() raises ComplexStepError for a float-only function."""
    with pytest.raises(ComplexStepError):
        gradient(_float_only, np.array([1.0, 2.0]))


def test_jacobian_complex_step_error():
    """jacobian() raises ComplexStepError for a float-only function."""
    def _float_only_vec(x):
        if np.iscomplexobj(x):
            raise TypeError("no complex")
        return np.array([np.sum(x**2), np.sum(x)])

    with pytest.raises(ComplexStepError):
        jacobian(_float_only_vec, np.array([1.0, 2.0]))


def test_error_message_contains_suggestion():
    """ComplexStepError message must mention a fallback suggestion."""
    with pytest.raises(ComplexStepError, match="finite differences|fallback"):
        gradient(_float_only, np.array([1.0]))


# ------------------------------------------------------------------ #
#   NonanalyticWarning — function that silently discards imaginary part
# ------------------------------------------------------------------ #

def _abs_based(x):
    """Uses np.abs on a complex intermediate — Im is silently lost."""
    return np.abs(x[0]) + x[1]  # np.abs returns a real value for complex input


def test_gradient_nonanaly_warning():
    """gradient() issues NonanalyticWarning when Im(f(x+ih)) ≈ 0."""
    x0 = np.array([1.0, 2.0])
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        try:
            gradient(_abs_based, x0)
        except Exception:
            pass  # error may follow the warning
        nonanaly = [x for x in w if issubclass(x.category, NonanalyticWarning)]
        assert len(nonanaly) >= 1, "NonanalyticWarning was not issued"


# ------------------------------------------------------------------ #
#   StepSizeWarning
# ------------------------------------------------------------------ #

def test_step_too_large_warns():
    """h > sqrt(eps) must issue StepSizeWarning."""
    with pytest.warns(StepSizeWarning, match="large"):
        derivative(lambda x: x**2, 1.0, h=1e-4)


def test_step_too_small_warns():
    """h < eps² must issue StepSizeWarning."""
    h_tiny = np.finfo(float).eps ** 2 / 10
    with pytest.warns(StepSizeWarning, match="small"):
        derivative(lambda x: x**2, 1.0, h=h_tiny)


def test_step_negative_raises():
    """h <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        derivative(lambda x: x**2, 1.0, h=-1e-10)


def test_step_zero_raises():
    with pytest.raises(ValueError, match="positive"):
        derivative(lambda x: x**2, 1.0, h=0.0)


# ------------------------------------------------------------------ #
#   Dimension validation — wrong function for (n, m)
# ------------------------------------------------------------------ #

def test_derivative_raises_for_vector_output():
    """derivative() raises ValueError when f returns a vector (use jacobian)."""
    def f_vec(x):
        return np.array([x**2, x**3], dtype=complex)
    with pytest.raises(ValueError, match="jacobian"):
        derivative(f_vec, 2.0)


def test_gradient_raises_for_n1():
    """gradient() raises ValueError when wrt arg has 1 element (use derivative or jacobian)."""
    with pytest.raises(ValueError, match="derivative"):
        gradient(lambda x: x[0]**2, np.array([3.0]))


def test_jacobian_raises_for_scalar_output():
    """jacobian() raises ValueError when f is scalar-valued (use gradient or derivative)."""
    with pytest.raises(ValueError, match="gradient"):
        jacobian(lambda x: np.sum(x**2), np.array([1.0, 2.0]))


def test_jacobian_raises_for_n1_m1():
    """jacobian() raises ValueError when n=1 and m=1 (use derivative)."""
    with pytest.raises(ValueError, match="derivative"):
        jacobian(lambda x: np.array([x[0]**2]), np.array([2.0]))


def test_directional_derivative_raises_for_scalar_function():
    """directional_derivative() raises ValueError when f: ℝ→ℝ (use derivative)."""
    with pytest.raises(ValueError, match="derivative"):
        directional_derivative(
            lambda x: x[0]**2, np.array([2.0]), v=np.array([1.0])
        )


# ------------------------------------------------------------------ #
#   TypeError for bad wrt
# ------------------------------------------------------------------ #

def test_wrt_none_multi_arg_raises_typeerror():
    """Multiple args with wrt=None must raise TypeError."""
    def f(t, x):
        return np.sum(x)

    with pytest.raises(TypeError, match="wrt"):
        gradient(f, 1.0, np.array([1.0, 2.0]))


def test_wrt_list_raises_typeerror():
    """wrt as a list (not tuple) must raise TypeError."""
    with pytest.raises(TypeError, match="tuple"):
        jacobian(lambda x: x, np.array([1.0, 2.0]), wrt=["wrong"])


# ------------------------------------------------------------------ #
#   directional_derivative ValueError for mismatched v
# ------------------------------------------------------------------ #

def test_directional_v_size_mismatch():
    """v with wrong size must raise ValueError."""
    x0 = np.array([1.0, 2.0, 3.0])
    v_bad = np.array([1.0])
    with pytest.raises(ValueError, match="v has"):
        directional_derivative(lambda x: np.sum(x), x0, v=v_bad)
