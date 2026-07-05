"""
Tests for return_diagnostics=True across all public functions.
"""
import numpy as np
import pytest

from csdiff import derivative, gradient, jacobian, directional_derivative, DiffInfo


# ------------------------------------------------------------------ #
#   derivative
# ------------------------------------------------------------------ #

def test_derivative_no_diag_returns_float():
    result = derivative(lambda x: x**2, 2.0)
    assert isinstance(result, float)


def test_derivative_diag_returns_tuple():
    result, info = derivative(lambda x: x**2, 2.0, return_diagnostics=True)
    assert isinstance(result, float)
    assert isinstance(info, DiffInfo)


def test_derivative_diag_n_calls():
    _, info = derivative(lambda x: x**2, 2.0, return_diagnostics=True)
    assert info.n_calls == 1


def test_derivative_diag_elapsed():
    _, info = derivative(lambda x: x**2, 2.0, return_diagnostics=True)
    assert info.elapsed >= 0.0


def test_derivative_diag_h_default():
    eps = np.finfo(float).eps
    _, info = derivative(lambda x: x**2, 2.0, return_diagnostics=True)
    assert abs(info.h - eps**1.5) < 1e-30


def test_derivative_diag_h_custom():
    _, info = derivative(lambda x: x**2, 2.0, h=1e-10, return_diagnostics=True)
    assert info.h == 1e-10


# ------------------------------------------------------------------ #
#   gradient (serial)
# ------------------------------------------------------------------ #

def test_gradient_no_diag_returns_array():
    result = gradient(lambda x: np.sum(x**2), np.ones(3))
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, tuple)


def test_gradient_diag_returns_tuple():
    result, info = gradient(
        lambda x: np.sum(x**2), np.ones(3), return_diagnostics=True
    )
    assert isinstance(result, np.ndarray)
    assert isinstance(info, DiffInfo)


def test_gradient_diag_n_calls_serial():
    n = 5
    x0 = np.ones(n)
    _, info = gradient(lambda x: np.sum(x**2), x0, return_diagnostics=True)
    assert info.n_calls == n


# ------------------------------------------------------------------ #
#   jacobian (serial)
# ------------------------------------------------------------------ #

def test_jacobian_diag_n_calls_serial():
    n = 4
    x0 = np.ones(n)
    _, info = jacobian(lambda x: x**2, x0, return_diagnostics=True)
    assert info.n_calls == n


# ------------------------------------------------------------------ #
#   directional_derivative
# ------------------------------------------------------------------ #

def test_dd_diag_n_calls_always_one():
    x0 = np.ones(10)
    v = np.ones(10)
    _, info = directional_derivative(
        lambda x: np.sum(x**2), x0, v=v, return_diagnostics=True
    )
    assert info.n_calls == 1


def test_dd_diag_returns_tuple():
    x0 = np.array([1.0, 2.0])
    v = np.array([0.0, 1.0])
    result, info = directional_derivative(
        lambda x: np.sum(x**2), x0, v=v, return_diagnostics=True
    )
    assert isinstance(info, DiffInfo)
    assert isinstance(result, (float, np.ndarray))


# ------------------------------------------------------------------ #
#   DiffInfo immutability
# ------------------------------------------------------------------ #

def test_diffinfo_frozen():
    _, info = derivative(lambda x: x, 1.0, return_diagnostics=True)
    with pytest.raises((AttributeError, TypeError)):
        info.n_calls = 999  # frozen dataclass must reject assignment


def test_diffinfo_repr():
    _, info = derivative(lambda x: x, 1.0, return_diagnostics=True)
    r = repr(info)
    assert "DiffInfo" in r
    assert "n_calls" in r
    assert "elapsed" in r
    assert "h=" in r
