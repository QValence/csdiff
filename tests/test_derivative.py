"""
Tests for derivative() — f: ℝ → ℝ, scalar case.
"""
import numpy as np
import pytest

from csdiff import derivative, DiffInfo


# ------------------------------------------------------------------ #
#   Accuracy vs analytical
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("x0, f, df", [
    (2.0,  lambda x: x**3,       lambda x: 3*x**2),
    (0.5,  lambda x: x**5 - x,   lambda x: 5*x**4 - 1),
    (1.0,  lambda x: np.exp(x),  lambda x: np.exp(x)),
    (1.5,  lambda x: np.log(x),  lambda x: 1.0/x),
    (0.3,  lambda x: np.sin(x),  lambda x: np.cos(x)),
    (0.7,  lambda x: np.cos(x),  lambda x: -np.sin(x)),
    (1.2,  lambda x: np.sqrt(x), lambda x: 0.5/np.sqrt(x)),
    (2.0,  lambda x: 1.0/x,      lambda x: -1.0/x**2),
])
def test_accuracy(x0, f, df):
    """Relative error vs analytical derivative must be below 1e-14."""
    result = derivative(f, x0)
    analytical = df(x0)
    rel_err = abs(result - analytical) / max(abs(analytical), 1e-300)
    assert rel_err < 1e-14, (
        f"f={f.__name__ if hasattr(f,'__name__') else 'lambda'}, "
        f"x0={x0}: rel_err={rel_err:.2e}"
    )


def test_polynomial_exact():
    """f(x) = x^2 at x=3 → f'=6. Expect near-exact machine precision."""
    result = derivative(lambda x: x**2, 3.0)
    assert abs(result - 6.0) < 1e-14


def test_returns_float():
    """derivative() must return a plain Python float, not an array."""
    result = derivative(lambda x: x**2, 1.0)
    assert isinstance(result, float)


# ------------------------------------------------------------------ #
#   Diagnostics
# ------------------------------------------------------------------ #

def test_return_diagnostics_false_by_default():
    """Without return_diagnostics, only the float is returned."""
    result = derivative(lambda x: x, 1.0)
    assert isinstance(result, float)


def test_return_diagnostics_true():
    """With return_diagnostics=True, returns (float, DiffInfo)."""
    result, info = derivative(lambda x: x**2, 2.0, return_diagnostics=True)
    assert isinstance(result, float)
    assert isinstance(info, DiffInfo)
    assert info.n_calls == 1
    assert info.elapsed >= 0.0
    assert info.h > 0.0
    assert abs(result - 4.0) < 1e-14


# ------------------------------------------------------------------ #
#   Custom step size
# ------------------------------------------------------------------ #

def test_custom_h():
    """User-supplied h is used and appears in DiffInfo."""
    h = 1e-15
    result, info = derivative(lambda x: x**2, 2.0, h=h, return_diagnostics=True)
    assert info.h == h
    # Accuracy should still be reasonable (h is within valid range)
    assert abs(result - 4.0) < 1e-10


# ------------------------------------------------------------------ #
#   Batched evaluation
# ------------------------------------------------------------------ #

def test_derivative_batched_vectorized():
    """numpy sin is vectorized — vectorized path returns (n_batch,)."""
    x0 = np.linspace(0.1, np.pi - 0.1, 10)
    result = derivative(np.sin, x0, batched=True)
    np.testing.assert_allclose(result, np.cos(x0), rtol=1e-14)
    assert result.shape == (10,)


def test_derivative_batched_serial_fallback():
    """Non-vectorized f (rejects array input) falls back to serial loop."""
    def scalar_only(x):
        # complex(x) works on complex scalars; raises TypeError on arrays
        return complex(x) ** 2

    x0 = np.array([1.0, 2.0, 3.0])
    result = derivative(scalar_only, x0, batched=True)
    np.testing.assert_allclose(result, [2.0, 4.0, 6.0], rtol=1e-14)
    assert result.shape == (3,)


def test_derivative_batched_diagnostics():
    """return_diagnostics works with batched=True."""
    x0 = np.array([1.0, 2.0])
    result, info = derivative(lambda x: x ** 2, x0, batched=True,
                              return_diagnostics=True)
    np.testing.assert_allclose(result, [2.0, 4.0], rtol=1e-14)
    assert isinstance(info, DiffInfo)
    assert info.n_calls >= 1  # 1 (vectorized) or 2 (serial)


def test_derivative_batched_wrong_shape_raises():
    """batched=True with 2-D x0 raises ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        derivative(np.sin, np.ones((3, 2)), batched=True)
