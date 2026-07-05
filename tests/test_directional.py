"""
Tests for directional_derivative() — J(x)@v in one evaluation.
"""
import numpy as np
import pytest

from csdiff import directional_derivative, jacobian, gradient, NonanalyticWarning


# ------------------------------------------------------------------ #
#   Consistency with jacobian @ v  (m > 1 cases)
# ------------------------------------------------------------------ #

def _jacobian_cases():
    """Generate (A, x0, v) triples where m >= 2 (Jacobian-applicable)."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = rng.integers(2, 6)   # n ∈ {2, 3, 4, 5}
        m = rng.integers(2, 5)   # m ∈ {2, 3, 4} — always uses jacobian()
        A = rng.standard_normal((m, n))
        x0 = rng.standard_normal(n)
        v = rng.standard_normal(n)
        yield A, x0, v


@pytest.mark.parametrize("A,x0,v", list(_jacobian_cases()))
def test_dd_matches_jacobian_times_v(A, x0, v):
    """directional_derivative(f, x, v=v) ≈ jacobian(f, x) @ v for m > 1."""
    f = lambda x: A @ x
    dd = directional_derivative(f, x0, v=v)
    J = jacobian(f, x0)
    np.testing.assert_allclose(dd, J @ v, rtol=1e-13)


# ------------------------------------------------------------------ #
#   Consistency with gradient @ v  (m == 1 cases)
# ------------------------------------------------------------------ #

def test_dd_matches_gradient_times_v():
    """directional_derivative with m=1 → float equal to gradient(f) @ v."""
    rng = np.random.default_rng(0)
    n = 4
    a = rng.standard_normal(n)
    x0 = rng.standard_normal(n)
    v = rng.standard_normal(n)
    f = lambda x: a @ x  # linear scalar function
    dd = directional_derivative(f, x0, v=v)
    expected = float(gradient(f, x0) @ v)
    assert isinstance(dd, float)
    assert abs(dd - expected) < 1e-13


# ------------------------------------------------------------------ #
#   Return types
# ------------------------------------------------------------------ #

def test_dd_scalar_output_returns_float():
    """For f: ℝⁿ → ℝ (m=1), directional_derivative returns a Python float."""
    x0 = np.array([1.0, 2.0])
    v = np.array([1.0, 0.0])
    result = directional_derivative(lambda x: np.sum(x**2), x0, v=v)
    assert isinstance(result, float)
    assert abs(result - 2.0) < 1e-14  # ∂(x₀²+x₁²)/∂x₀ = 2x₀ = 2


def test_dd_vector_output_returns_array():
    """For f: ℝⁿ → ℝᵐ (m > 1), directional_derivative returns an ndarray."""
    A = np.eye(3)
    x0 = np.ones(3)
    v = np.array([1.0, 2.0, 3.0])
    result = directional_derivative(lambda x: A @ x, x0, v=v)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, v, rtol=1e-13)


# ------------------------------------------------------------------ #
#   Accuracy
# ------------------------------------------------------------------ #

def test_dd_nonlinear():
    """Non-linear f: ∇(x@x) = 2x, so ∇(x@x)·v = 2*x@v."""
    x0 = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, -1.0, 2.0])
    result = directional_derivative(lambda x: x @ x, x0, v=v)
    expected = 2 * x0 @ v
    assert abs(result - expected) < 1e-13


# ------------------------------------------------------------------ #
#   Diagnostics
# ------------------------------------------------------------------ #

def test_dd_n_calls_always_one():
    """directional_derivative must always use exactly 1 function call."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v = np.ones(5)
    _, info = directional_derivative(
        lambda x: np.sum(x**2), x0, v=v, return_diagnostics=True
    )
    assert info.n_calls == 1


# ------------------------------------------------------------------ #
#   Error handling
# ------------------------------------------------------------------ #

def test_v_wrong_size_raises():
    """v with wrong number of elements must raise ValueError."""
    x0 = np.array([1.0, 2.0, 3.0])
    v_wrong = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="v has"):
        directional_derivative(lambda x: np.sum(x), x0, v=v_wrong)


def test_raises_for_scalar_function():
    """directional_derivative on f: ℝ→ℝ (n=1, m=1) → use derivative()."""
    with pytest.raises(ValueError, match="derivative"):
        directional_derivative(lambda x: x[0]**2, np.array([2.0]), v=np.array([1.0]))


# ------------------------------------------------------------------ #
#   Batched evaluation
# ------------------------------------------------------------------ #

def test_directional_batched_fixed_v_vectorized():
    """Vectorized scalar f with fixed direction v."""
    f = lambda x: np.sum(x ** 2, axis=-1)  # (batch, n) → (batch,)
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = np.array([1.0, 0.0])
    result = directional_derivative(f, X, v=v, batched=True)
    # ∇(x²+y²)·[1,0] = 2x
    np.testing.assert_allclose(result, [2.0, 0.0, 2.0], rtol=1e-14)
    assert result.shape == (3,)


def test_directional_batched_varying_v():
    """Per-point direction vectors v of shape (n_batch, n)."""
    f = lambda x: np.sum(x ** 2, axis=-1)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])  # diagonal directions
    result = directional_derivative(f, X, v=V, batched=True)
    np.testing.assert_allclose(result, [2.0, 2.0], rtol=1e-14)
    assert result.shape == (2,)


def test_directional_batched_serial_fallback():
    """Non-vectorized scalar f (rejects 2-D input) falls back to serial.

    NonanalyticWarning is expected for the batch point where the true
    directional derivative is zero (Im(f(x+ih·v)) ≈ 0 at x=[0,1], v=[1,0]).
    """
    def f(x):
        if np.ndim(x) != 1:
            raise TypeError("only 1-D input supported")
        return np.dot(x, x)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([1.0, 0.0])
    with pytest.warns(NonanalyticWarning):
        result = directional_derivative(f, X, v=v, batched=True)
    np.testing.assert_allclose(result, [2.0, 0.0], rtol=1e-14)
    assert result.shape == (2,)


def test_directional_batched_vector_f():
    """Batched JVP for vector-valued f: shape (n_batch, m)."""
    A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # (3, 2)

    def f_vec(x):
        # Accepts (batch, 2) → (batch, 3)  or  (2,) → (3,)
        if x.ndim == 2:
            return (A @ x.T).T
        return A @ x

    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    v = np.array([1.0, 0.0])
    result = directional_derivative(f_vec, X, v=v, batched=True)
    # J@v = A @ [1,0] = first column of A = [1, 0, 1]
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], A @ v, rtol=1e-13)
    np.testing.assert_allclose(result[1], A @ v, rtol=1e-13)


def test_directional_batched_v_size_mismatch_raises():
    """Mismatched batch size between X and v raises ValueError."""
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    V_wrong = np.array([[1.0, 0.0]])  # batch size 1, not 2
    with pytest.raises(ValueError, match="batch size"):
        directional_derivative(lambda x: np.sum(x ** 2, axis=-1),
                               X, v=V_wrong, batched=True)


def test_directional_batched_vector_f_m_equals_probe_batch():
    """Non-vectorized f: ℝ²→ℝ³ with m==_PROBE_BATCH==3 must not be falsely detected as vectorized.

    Same false-positive risk as the jacobian case: two-probe detection is needed to
    correctly fall back to serial and return the right shape (n_batch, m) = (4, 3).
    """
    def f(x):
        # Single-point function — not vectorized
        return np.array([x[0] ** 2 + x[1], x[0] * x[1], x[0] - x[1]])

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    v = np.array([1.0, 0.0])
    result = directional_derivative(f, X, v=v, batched=True)
    assert result.shape == (4, 3), f"expected (4, 3), got {result.shape}"

    # At point (1, 0): J = [[2,1],[0,1],[1,-1]], J@[1,0] = [2, 0, 1]
    np.testing.assert_allclose(result[0], [2.0, 0.0, 1.0], rtol=1e-13)
