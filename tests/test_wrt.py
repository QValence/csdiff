"""
Tests for wrt= dispatch — all forms: str, int, tuple of str/int, mixed.

Reference function: f(t, x, u, p) where
  t : float scalar
  x : array (nx,)
  u : array (nu,)
  p : float scalar (unused — tests that non-wrt args pass through unchanged)

Analytical Jacobians:
  J wrt x : shape (m, nx)
  J wrt u : shape (m, nu)
  J wrt t : shape (m, 1)
  J wrt (x, u) : shape (m, nx+nu)
"""
import numpy as np
import pytest

from csdiff import jacobian, gradient


# ------------------------------------------------------------------ #
#   Reference function and analytical Jacobians
# ------------------------------------------------------------------ #

NX, NU, M = 2, 3, 4  # input / output dimensions

def f_ref(t, x, u, p):
    """
    f: ℝ × ℝ² × ℝ³ × ℝ → ℝ⁴
    f_i(t, x, u) = t * x[i % nx] + u[i % nu]   for i = 0..3
    Unused argument p is present to test that it passes through unchanged.
    """
    out = np.array([
        t * x[0] + u[0],
        t * x[1] + u[1],
        t * x[0] + u[2],
        t * x[1] + u[0],
    ])
    return out


def analytical_J_x(t, x, u):
    """∂f/∂x: shape (4, 2)."""
    return np.array([
        [t, 0.0],
        [0.0, t],
        [t, 0.0],
        [0.0, t],
    ])


def analytical_J_u(t, x, u):
    """∂f/∂u: shape (4, 3)."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])


def analytical_J_t(t, x, u):
    """∂f/∂t: shape (4, 1)."""
    return np.array([[x[0]], [x[1]], [x[0]], [x[1]]])


# Evaluation point
T0 = 2.0
X0 = np.array([3.0, 5.0])
U0 = np.array([1.0, 4.0, 2.0])
P0 = 99.0  # arbitrary, unused in f


# ------------------------------------------------------------------ #
#   wrt by name (str)
# ------------------------------------------------------------------ #

def test_jacobian_wrt_str_x():
    J = jacobian(f_ref, T0, X0, U0, P0, wrt="x")
    np.testing.assert_allclose(J, analytical_J_x(T0, X0, U0), rtol=1e-13)


def test_jacobian_wrt_str_u():
    J = jacobian(f_ref, T0, X0, U0, P0, wrt="u")
    np.testing.assert_allclose(J, analytical_J_u(T0, X0, U0), rtol=1e-13)


def test_jacobian_wrt_str_t():
    """wrt a scalar argument: Jacobian has shape (m, 1)."""
    J = jacobian(f_ref, T0, X0, U0, P0, wrt="t")
    assert J.shape == (M, 1)
    np.testing.assert_allclose(J, analytical_J_t(T0, X0, U0), rtol=1e-13)


# ------------------------------------------------------------------ #
#   wrt by position (int)
# ------------------------------------------------------------------ #

def test_jacobian_wrt_int_x():
    """wrt=1 selects x (position 1 in args)."""
    J = jacobian(f_ref, T0, X0, U0, P0, wrt=1)
    np.testing.assert_allclose(J, analytical_J_x(T0, X0, U0), rtol=1e-13)


def test_jacobian_wrt_int_u():
    """wrt=2 selects u (position 2)."""
    J = jacobian(f_ref, T0, X0, U0, P0, wrt=2)
    np.testing.assert_allclose(J, analytical_J_u(T0, X0, U0), rtol=1e-13)


def test_jacobian_wrt_int_t():
    """wrt=0 selects t (scalar, position 0)."""
    J = jacobian(f_ref, T0, X0, U0, P0, wrt=0)
    assert J.shape == (M, 1)
    np.testing.assert_allclose(J, analytical_J_t(T0, X0, U0), rtol=1e-13)


# ------------------------------------------------------------------ #
#   str == int consistency
# ------------------------------------------------------------------ #

def test_str_int_agree_x():
    """wrt='x' and wrt=1 must produce identical results."""
    J_str = jacobian(f_ref, T0, X0, U0, P0, wrt="x")
    J_int = jacobian(f_ref, T0, X0, U0, P0, wrt=1)
    np.testing.assert_array_equal(J_str, J_int)


def test_str_int_agree_u():
    J_str = jacobian(f_ref, T0, X0, U0, P0, wrt="u")
    J_int = jacobian(f_ref, T0, X0, U0, P0, wrt=2)
    np.testing.assert_array_equal(J_str, J_int)


# ------------------------------------------------------------------ #
#   Combined wrt — tuple[str | int]
# ------------------------------------------------------------------ #

def test_jacobian_wrt_tuple_str():
    """wrt=('x','u') → J shape (m, nx+nu) = (4, 5)."""
    J = jacobian(f_ref, T0, X0, U0, P0, wrt=("x", "u"))
    expected = np.hstack([
        analytical_J_x(T0, X0, U0),
        analytical_J_u(T0, X0, U0),
    ])
    assert J.shape == (M, NX + NU)
    np.testing.assert_allclose(J, expected, rtol=1e-13)


def test_jacobian_wrt_tuple_int():
    """wrt=(1, 2) is equivalent to wrt=('x','u')."""
    J_str = jacobian(f_ref, T0, X0, U0, P0, wrt=("x", "u"))
    J_int = jacobian(f_ref, T0, X0, U0, P0, wrt=(1, 2))
    np.testing.assert_array_equal(J_str, J_int)


def test_jacobian_wrt_tuple_mixed():
    """wrt=(1, 'u') (mixed str/int) must equal wrt=(1, 2)."""
    J_mixed = jacobian(f_ref, T0, X0, U0, P0, wrt=(1, "u"))
    J_int   = jacobian(f_ref, T0, X0, U0, P0, wrt=(1, 2))
    np.testing.assert_array_equal(J_mixed, J_int)


def test_combined_equals_hstack():
    """Combined wrt Jacobian equals np.hstack of individual Jacobians."""
    J_combined = jacobian(f_ref, T0, X0, U0, P0, wrt=("x", "u"))
    J_x = jacobian(f_ref, T0, X0, U0, P0, wrt="x")
    J_u = jacobian(f_ref, T0, X0, U0, P0, wrt="u")
    np.testing.assert_allclose(J_combined, np.hstack([J_x, J_u]), rtol=1e-14)


# ------------------------------------------------------------------ #
#   wrt omitted — single-arg auto-detection
# ------------------------------------------------------------------ #

def test_single_arg_no_wrt_needed():
    """When f has a single argument, wrt=None is inferred automatically."""
    result = jacobian(lambda x: x**2, np.array([1.0, 2.0, 3.0]))
    expected = np.diag([2.0, 4.0, 6.0])
    np.testing.assert_allclose(result, expected, atol=1e-14)


# ------------------------------------------------------------------ #
#   Error cases
# ------------------------------------------------------------------ #

def test_missing_wrt_multi_arg_raises():
    """Omitting wrt with multiple args must raise TypeError."""
    with pytest.raises(TypeError, match="wrt"):
        jacobian(f_ref, T0, X0, U0, P0)


def test_wrt_list_raises():
    """wrt=[...] (a list) must raise TypeError (use tuple instead)."""
    with pytest.raises(TypeError, match="tuple"):
        jacobian(f_ref, T0, X0, U0, P0, wrt=["x", "u"])


def test_wrt_str_unknown_name_raises():
    """wrt='z' where z is not a parameter name must raise ValueError."""
    with pytest.raises(ValueError, match="'z'"):
        jacobian(f_ref, T0, X0, U0, P0, wrt="z")


def test_wrt_int_out_of_range_raises():
    """wrt=10 (beyond number of args) must raise IndexError."""
    with pytest.raises(IndexError, match="out of range"):
        jacobian(f_ref, T0, X0, U0, P0, wrt=10)


# ------------------------------------------------------------------ #
#   gradient with wrt
# ------------------------------------------------------------------ #

def test_gradient_wrt_str():
    """gradient() with wrt on a multi-arg scalar function."""
    def g(t, x):
        return t * np.sum(x**2)

    x0 = np.array([2.0, 3.0])
    result = gradient(g, 5.0, x0, wrt="x")
    expected = 5.0 * 2 * x0  # ∂(t*||x||²)/∂x = 2tx
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_gradient_wrt_int():
    """gradient() with wrt=1 on a multi-arg function."""
    def g(t, x):
        return t * np.sum(x**2)

    x0 = np.array([2.0, 3.0])
    result_str = gradient(g, 5.0, x0, wrt="x")
    result_int = gradient(g, 5.0, x0, wrt=1)
    np.testing.assert_array_equal(result_str, result_int)
