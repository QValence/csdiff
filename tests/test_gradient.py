"""
Tests for gradient() — f: ℝⁿ → ℝ with n > 1.
"""
import numpy as np
import pytest

from csdiff import gradient, DiffInfo, NonanalyticWarning


# ------------------------------------------------------------------ #
#   Accuracy vs analytical
# ------------------------------------------------------------------ #

def test_gradient_quadratic():
    """f(x) = ||x||² → ∇f = 2x."""
    x0 = np.array([1.0, 2.0, 3.0])
    result = gradient(lambda x: np.dot(x, x), x0)
    expected = 2 * x0
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_gradient_mixed():
    """f(x, y) = x^2 * y → ∂f/∂x = 2xy, ∂f/∂y = x^2."""
    x0 = np.array([2.0, 3.0])
    f = lambda v: v[0]**2 * v[1]
    result = gradient(f, x0)
    expected = np.array([2 * 2.0 * 3.0, 2.0**2])
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_gradient_exp_sum():
    """f(x) = sum(exp(x)) → ∇f = exp(x)."""
    x0 = np.array([0.5, 1.0, 1.5])
    result = gradient(lambda x: np.sum(np.exp(x)), x0)
    expected = np.exp(x0)
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_gradient_log_sum():
    """f(x) = sum(log(x)) → ∇f = 1/x."""
    x0 = np.array([1.0, 2.0, 4.0])
    result = gradient(lambda x: np.sum(np.log(x)), x0)
    expected = 1.0 / x0
    np.testing.assert_allclose(result, expected, rtol=1e-14)


# ------------------------------------------------------------------ #
#   Output shape and type
# ------------------------------------------------------------------ #

def test_output_shape():
    """gradient() must return a 1-D ndarray of length n."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    result = gradient(lambda x: np.sum(x), x0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert result.ndim == 1


# ------------------------------------------------------------------ #
#   Dimension validation errors
# ------------------------------------------------------------------ #

def test_raises_for_n1_scalar_output():
    """gradient() with 1-element input and scalar output → use derivative()."""
    x0 = np.array([2.0])
    with pytest.raises(ValueError, match="derivative"):
        gradient(lambda x: x[0]**3, x0)


def test_raises_for_n1_vector_output():
    """gradient() with 1-element input and vector output → use jacobian()."""
    x0 = np.array([2.0])
    with pytest.raises(ValueError, match="jacobian"):
        gradient(lambda x: np.array([x[0], x[0]**2]), x0)


def test_raises_for_vector_output():
    """gradient() with vector f output → use jacobian()."""
    x0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="jacobian"):
        gradient(lambda x: x, x0)  # f: ℝ² → ℝ²


# ------------------------------------------------------------------ #
#   Diagnostics
# ------------------------------------------------------------------ #

def test_diagnostics_n_calls_serial():
    """Serial gradient of ℝⁿ function uses exactly n calls."""
    n = 5
    x0 = np.ones(n)
    _, info = gradient(lambda x: np.sum(x**2), x0, return_diagnostics=True)
    assert info.n_calls == n


def test_diagnostics_elapsed_positive():
    x0 = np.array([1.0, 2.0])
    _, info = gradient(lambda x: np.sum(x), x0, return_diagnostics=True)
    assert info.elapsed >= 0.0


# ------------------------------------------------------------------ #
#   Batched evaluation
# ------------------------------------------------------------------ #

def test_gradient_batched_vectorized():
    """Vectorized f: np.sum(x**2, axis=-1) supports batched input."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = gradient(lambda x: np.sum(x ** 2, axis=-1), X, batched=True)
    np.testing.assert_allclose(result, 2 * X, rtol=1e-14)
    assert result.shape == (2, 3)


def test_gradient_batched_serial_fallback():
    """Non-vectorized f (rejects 2-D input) falls back to serial loop.

    NonanalyticWarning is expected for batch points where the true gradient
    component is zero (Im(f(x+ih·eₖ)) ≈ 0 is indistinguishable from a silent
    float cast at that point).
    """
    def f(x):
        if np.ndim(x) != 1:
            raise TypeError("only 1-D input supported")
        return np.dot(x, x)

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with pytest.warns(NonanalyticWarning):
        result = gradient(f, X, batched=True)
    np.testing.assert_allclose(result, 2 * X, rtol=1e-14)
    assert result.shape == (3, 2)


def test_gradient_batched_multiarg_wrt():
    """batched=True with wrt= in a multi-argument function."""
    def f(t, x):
        return t * np.sum(x ** 2, axis=-1)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = gradient(f, 2.0, X, wrt="x", batched=True)
    np.testing.assert_allclose(result, 2 * 2.0 * X, rtol=1e-14)
    assert result.shape == (2, 2)


def test_gradient_batched_output_shape_n4():
    """Batch of 5 points in ℝ⁴ → output (5, 4)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((5, 4))
    result = gradient(lambda x: np.sum(x ** 2, axis=-1), X, batched=True)
    np.testing.assert_allclose(result, 2 * X, rtol=1e-14)
    assert result.shape == (5, 4)


def test_gradient_batched_diagnostics():
    """return_diagnostics aggregates n_calls over the batch."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    _, info = gradient(lambda x: np.sum(x ** 2, axis=-1), X, batched=True,
                       return_diagnostics=True)
    assert isinstance(info, DiffInfo)
    assert info.n_calls >= 1


def test_gradient_batched_tuple_wrt_raises():
    """batched=True with tuple wrt raises ValueError explaining ambiguity."""
    def f(x, u):
        return np.sum(x ** 2, axis=-1) + np.sum(u ** 2, axis=-1)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="ambiguous"):
        gradient(f, X, np.ones((2, 2)), wrt=("x", "u"), batched=True)


def test_gradient_batched_1d_wrt_raises():
    """batched=True with 1-D wrt arg raises ValueError."""
    with pytest.raises(ValueError, match="2-D"):
        gradient(lambda x: np.sum(x ** 2), np.array([1.0, 2.0]), batched=True)
