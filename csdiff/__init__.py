"""
csdiff — complex step differentiation for black-box callables
=============================================================

Computes first-order derivatives of arbitrary callables f: ℝⁿ → ℝᵐ using
the **complex step method**, which achieves machine-precision accuracy
(~1e-16 relative error for float64) without step-size tuning.

Formula: f'(x) ≈ Im(f(x + ih)) / h
  - Truncation error O(h²), no cancellation error.
  - Default h = eps^(3/2) ≈ 1.05e-23 gives full machine precision.

Public API — dimension dispatch
---------------------------------
  n = number of scalar elements in the wrt argument (input size)
  m = number of scalar elements returned by f (output size)

  derivative(f, x0, ...)                   n=1, m=1  →  float
  gradient(f, *args, wrt=None, ...)        n>1, m=1  →  ndarray (n,)
  jacobian(f, *args, wrt=None, ...)        n≥1, m>1  →  ndarray (m, n)
  directional_derivative(f, *args, v, ...) n≥1, m≥1  →  float or ndarray (m,)

Each function probes f once to determine (n, m) and raises a descriptive
ValueError if the wrong function was called for that case.

Function signatures
-------------------
derivative(f, x0, *, h=None, return_diagnostics=False)
    First derivative of f: ℝ → ℝ at scalar x0.

gradient(f, *args, wrt=None, h=None, return_diagnostics=False)
    Gradient of f: ℝⁿ → ℝ.  Returns ndarray shape (n,).

jacobian(f, *args, wrt=None, h=None, return_diagnostics=False)
    Jacobian of f: ℝⁿ → ℝᵐ.  Returns ndarray shape (m, n).
    Convention: J[i, j] = ∂fᵢ/∂xⱼ.

directional_derivative(f, *args, v, wrt=None, h=None, return_diagnostics=False)
    Jacobian-vector product J(x)@v in one function evaluation.

DiffInfo
    Frozen dataclass with n_calls, elapsed, h (returned when
    return_diagnostics=True).

ComplexStepError, NonanalyticWarning, StepSizeWarning
    Exception and warning types raised/issued by this package.

Multi-argument functions
------------------------
When f is declared as f(t, x, u, p) rather than f(X), pass all arguments
and specify which to differentiate via ``wrt``:

    gradient(f, t0, x0, u0, p0, wrt="x")      # by parameter name
    gradient(f, t0, x0, u0, p0, wrt=1)         # by position (0-indexed)
    jacobian(f, t0, x0, u0, p0, wrt=("x","u")) # combined wrt multiple args

Internal modules (importable but not in __all__)
------------------------------------------------
csdiff.core      : differentiation kernels (gradient_serial, jacobian_serial, …)
                   and probe().
csdiff.template  : CallTemplate — argument binding and flat ↔ structured conversion.
csdiff.step      : default_step(), validate_step().
csdiff.info      : DiffInfo dataclass.
csdiff.exceptions: ComplexStepError, NonanalyticWarning, StepSizeWarning.
"""

from csdiff.functions import (
    derivative,
    gradient,
    jacobian,
    directional_derivative,
)
from csdiff.info import DiffInfo
from csdiff.exceptions import ComplexStepError, NonanalyticWarning, StepSizeWarning

__all__ = [
    # Differentiation functions
    "derivative",
    "gradient",
    "jacobian",
    "directional_derivative",
    # Diagnostics
    "DiffInfo",
    # Exceptions / warnings
    "ComplexStepError",
    "NonanalyticWarning",
    "StepSizeWarning",
]

__version__ = "0.1.0"
