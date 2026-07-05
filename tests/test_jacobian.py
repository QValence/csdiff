"""
Tests for jacobian() — f: ℝⁿ → ℝᵐ with m > 1.
"""
import numpy as np
import pytest

from csdiff import jacobian, DiffInfo


# ------------------------------------------------------------------ #
#   Accuracy vs analytical
# ------------------------------------------------------------------ #

def test_jacobian_diagonal():
    """f(x) = x² element-wise → J = diag(2x)."""
    x0 = np.array([1.0, 2.0, 3.0])
    result = jacobian(lambda x: x**2, x0)
    expected = np.diag(2 * x0)
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_jacobian_linear():
    """f(x) = A @ x → J = A."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x0 = np.array([1.0, 1.0])
    result = jacobian(lambda x: A @ x, x0)
    np.testing.assert_allclose(result, A, rtol=1e-14)


def test_jacobian_nonlinear():
    """f: ℝ² → ℝ² with known Jacobian."""
    def f(x):
        return np.array([x[0]**2 * x[1], np.sin(x[0]) + x[1]**3])

    x0 = np.array([1.0, 2.0])
    result = jacobian(f, x0)
    J_analytical = np.array([
        [2 * x0[0] * x0[1], x0[0]**2],
        [np.cos(x0[0]),      3 * x0[1]**2],
    ])
    np.testing.assert_allclose(result, J_analytical, rtol=1e-13)


def test_jacobian_convention():
    """J[i, j] = ∂fᵢ/∂xⱼ — rows are output dims, cols are input dims."""
    def f(x):
        return np.array([x[0] + 2 * x[1], 3 * x[0]])

    x0 = np.array([1.0, 1.0])
    J = jacobian(f, x0)
    expected = np.array([[1.0, 2.0], [3.0, 0.0]])
    np.testing.assert_allclose(J, expected, atol=1e-14)


# ------------------------------------------------------------------ #
#   Output shape and type
# ------------------------------------------------------------------ #

def test_output_shape_mxn():
    """jacobian() returns (m, n) ndarray."""
    m, n = 4, 3
    A = np.random.default_rng(0).standard_normal((m, n))
    x0 = np.zeros(n)
    result = jacobian(lambda x: A @ x, x0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (m, n)


def test_scalar_input_vector_output():
    """jacobian() handles f: ℝ→ℝᵐ (n=1, m>1), returns shape (m, 1)."""
    result = jacobian(
        lambda x: np.array([x[0], x[0]**2, x[0]**3]),
        np.array([2.0])
    )
    assert result.shape == (3, 1)
    np.testing.assert_allclose(result.ravel(), [1.0, 4.0, 12.0], rtol=1e-14)


# ------------------------------------------------------------------ #
#   Dimension validation errors
# ------------------------------------------------------------------ #

def test_raises_for_scalar_output_array_input():
    """jacobian() with scalar f output and array input → use gradient()."""
    x0 = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="gradient"):
        jacobian(lambda x: np.sum(x**2), x0)


def test_raises_for_scalar_input_scalar_output():
    """jacobian() with scalar input and scalar output → use derivative()."""
    with pytest.raises(ValueError, match="derivative"):
        jacobian(lambda x: np.array([x[0]**2]), np.array([2.0]))
    # Note: f returns a 1-element array → m=1


# ------------------------------------------------------------------ #
#   Diagnostics
# ------------------------------------------------------------------ #

def test_diagnostics_n_calls_serial():
    """Serial Jacobian of ℝⁿ function uses exactly n calls."""
    n = 4
    x0 = np.ones(n)
    _, info = jacobian(lambda x: x**2, x0, return_diagnostics=True)
    assert info.n_calls == n


# ------------------------------------------------------------------ #
#   Batched evaluation
# ------------------------------------------------------------------ #

def test_jacobian_batched_vectorized():
    """Vectorized linear f: J is constant A for every batch point."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def f_batched(x):
        # x: (batch, 2) → (batch, 3)  or  x: (2,) → (3,)
        if x.ndim == 2:
            return (A @ x.T).T
        return A @ x

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = jacobian(f_batched, X, batched=True)
    assert result.shape == (3, 3, 2)
    for Ji in result:
        np.testing.assert_allclose(Ji, A, rtol=1e-14)


def test_jacobian_batched_serial_fallback():
    """Non-vectorized f (rejects 2-D input) falls back to serial loop."""
    def f(x):
        if np.ndim(x) != 1:
            raise TypeError("only 1-D input supported")
        return np.array([x[0] ** 2, x[1] ** 3])

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = jacobian(f, X, batched=True)
    assert result.shape == (2, 2, 2)
    np.testing.assert_allclose(result[0], np.diag([2.0, 12.0]), rtol=1e-13)
    np.testing.assert_allclose(result[1], np.diag([6.0, 48.0]), rtol=1e-13)


def test_jacobian_batched_diagnostics():
    """return_diagnostics works with batched=True."""
    def f(x):
        return np.array([x[0] ** 2, x[1] ** 2])

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    result, info = jacobian(f, X, batched=True, return_diagnostics=True)
    assert result.shape == (2, 2, 2)
    assert isinstance(info, DiffInfo)
    assert info.n_calls >= 1


def test_jacobian_batched_m_equals_probe_batch_non_vectorized():
    """f: ℝ²→ℝ³ with m==_PROBE_BATCH==3 must not be falsely detected as vectorized.

    A single-probe strategy with batch size 3 produces shape (3, 2) output for
    this f, which is indistinguishable from a vectorized (batch=3, m=2) response.
    The two-probe fix ensures the serial fallback is used and gives the correct
    shape (n_batch, m, n) = (4, 3, 2).
    """
    def f(x):
        # Single-point function using x[0], x[1] — not vectorized
        return np.array([x[0] ** 2 + x[1], x[0] * x[1], x[0] - x[1]])

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    result = jacobian(f, X, batched=True)
    assert result.shape == (4, 3, 2), f"expected (4, 3, 2), got {result.shape}"

    # Verify values at the first point (1, 0): analytical J = [[2,1],[0,1],[1,-1]]
    J_expected = np.array([[2.0, 1.0], [0.0, 1.0], [1.0, -1.0]])
    np.testing.assert_allclose(result[0], J_expected, rtol=1e-13)


def test_jacobian_batched_m_equals_probe_batch_vectorized():
    """Vectorized f: ℝ²→ℝ³ with m==_PROBE_BATCH==3 uses the fast vectorized path."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

    def f(x):
        # x: (batch, 2) → (batch, 3)  or  (2,) → (3,)
        if x.ndim == 2:
            return (A @ x.T).T
        return A @ x

    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    result = jacobian(f, X, batched=True)
    assert result.shape == (3, 3, 2), f"expected (3, 3, 2), got {result.shape}"
    for Ji in result:
        np.testing.assert_allclose(Ji, A, rtol=1e-14)
