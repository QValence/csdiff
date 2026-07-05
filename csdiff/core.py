"""
Differentiation kernels and complex-support probe.

All kernels share the same interface:

    g       : callable, (n_flat,) complex → scalar or ndarray
    x_flat  : np.ndarray, shape (n_flat,), dtype float — evaluation point
    h       : float — complex step size

``g`` is typically a ``CallTemplate`` instance or a thin lambda wrapping f.
The kernels know nothing about argument structure; that is handled upstream.

probe()
-------
Calls g at x_flat with component 0 perturbed by ih.  This is identical to
the k=0 step in the gradient/Jacobian loops, so its result (k0_imag) is
passed back to the calling function and reused as the k=0 column — saving
one redundant f-evaluation.  Returns (m, k0_imag).
"""
from __future__ import annotations

import warnings

import numpy as np

from csdiff.exceptions import ComplexStepError, NonanalyticWarning


# --------------------------------------------------------------------------- #
#   Probe
# --------------------------------------------------------------------------- #

def probe(g, x_flat: np.ndarray, h: float) -> tuple[int, np.ndarray]:
    """
    Verify complex support, determine output size m, and return the k=0 result.

    Evaluates g at x_flat with component 0 perturbed by ih — the same step
    the serial gradient/Jacobian loops take at k=0.  The imaginary part is
    returned so the caller can reuse it as the first column of the derivative
    instead of repeating the evaluation.

    Parameters
    ----------
    g : callable
        (n_flat,) complex → scalar or ndarray.  Typically a CallTemplate.
    x_flat : np.ndarray, shape (n_flat,), dtype float
        Real evaluation point.
    h : float
        Complex step size used for the perturbation.

    Returns
    -------
    m : int
        Number of scalar outputs (1 for scalar f, m for array-valued f).
    k0_imag : np.ndarray, shape (m,)
        Im(g(x + ih·e₀)).  Pass as ``k0_imag`` to gradient_serial or
        jacobian_serial to skip the k=0 evaluation in their loops.

    Raises
    ------
    ComplexStepError
        If g raises TypeError, ValueError, or OverflowError with complex input.

    Warns
    -----
    NonanalyticWarning
        If Im(g(x+ih)) ≈ 0, suggesting g silently discards the imaginary part.
    """
    xc = x_flat.astype(complex)
    xc[0] += 1j * h  # perturb first component — same as loop iteration k=0

    try:
        result = g(xc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ComplexStepError(
            f"f raised {type(exc).__name__} when called with complex-valued input:\n"
            f"  {exc}\n\n"
            "Complex step differentiation requires the function to accept and propagate "
            "complex values throughout all of its operations.\n\n"
            "Common causes:\n"
            "  - np.abs() on a complex intermediate: use np.sqrt(x.real**2 + x.imag**2)\n"
            "    or restructure so abs() is never applied to a perturbed variable.\n"
            "  - np.sign(), np.floor(), np.ceil(), np.round() on a complex value.\n"
            "  - A conditional branch (if x > 0) that implicitly casts complex → float.\n"
            "  - A C/Fortran extension that internally converts its input to float.\n\n"
            "Suggestion: use central finite differences (h ≈ 1e-6) as a fallback."
        ) from exc

    result_arr = np.asarray(result).ravel()
    m = result_arr.size

    # Heuristic: for h ≈ 1e-23 and |f'| ≈ 1, Im ≈ 1e-23.
    # Threshold 1e-200 catches silent float casts while allowing extremely
    # small (but non-zero) true derivatives to pass through.
    k0_imag = np.imag(result_arr)
    imag_norm = float(np.max(np.abs(k0_imag)))
    if imag_norm < 1e-200:
        warnings.warn(
            "Im(f(x+ih)) ≈ 0 at the probe point. f may be silently discarding "
            "the imaginary part of its inputs.\n"
            "Common causes: np.abs, np.sign, a conditional branch, or a C/Fortran "
            "extension that casts inputs to float.\n"
            "Derivative estimates will be numerically zero (incorrect).",
            NonanalyticWarning,
            stacklevel=3,  # surfaces at the user-facing call (gradient, jacobian, …)
        )

    # Return k0_imag so callers can use it as the k=0 column result and skip
    # that evaluation in the serial loop (saves 1 of n+1 total f-calls).
    return m, k0_imag


# --------------------------------------------------------------------------- #
#   Kernels
# --------------------------------------------------------------------------- #

def gradient_serial(
    g, x_flat: np.ndarray, h: float, k0_imag: np.ndarray | None = None
) -> np.ndarray:
    """
    Gradient of a scalar-output function via serial complex step.

    Each of the n input dimensions is perturbed independently:
        grad[k] = Im(g(x + ih·eₖ)) / h ≈ ∂f/∂xₖ

    Cost: n function evaluations, or n-1 when k0_imag is supplied (the k=0
    result was already computed by probe() and is reused here).

    Parameters
    ----------
    g : callable
        (n_flat,) complex → scalar.  Must support complex input.
    x_flat : np.ndarray, shape (n,), dtype float
        Evaluation point.
    h : float
        Complex step size.
    k0_imag : np.ndarray of shape (1,) or None
        Pre-computed Im(g(x + ih·e₀)) from probe().  When provided the loop
        starts at k=1 and grad[0] is filled from this value.

    Returns
    -------
    np.ndarray, shape (n,)
        Gradient vector.
    """
    n = x_flat.size
    grad = np.empty(n)
    xc = x_flat.astype(complex)

    start = 0
    if k0_imag is not None:
        grad[0] = float(k0_imag[0]) / h
        start = 1

    for k in range(start, n):
        xc[k] += 1j * h
        grad[k] = np.imag(g(xc)) / h
        xc[k] = x_flat[k]

    return grad


def jacobian_serial(
    g, x_flat: np.ndarray, h: float, m: int, k0_imag: np.ndarray | None = None
) -> np.ndarray:
    """
    Jacobian of a vector-output function via serial complex step.

    Each column of J is computed from one function evaluation:
        J[:, k] = Im(g(x + ih·eₖ)) / h ≈ ∂f/∂xₖ

    Cost: n function evaluations, or n-1 when k0_imag is supplied (the k=0
    result was already computed by probe() and is reused here).

    Parameters
    ----------
    g : callable
        (n_flat,) complex → (m,) array.  Must support complex input.
    x_flat : np.ndarray, shape (n,), dtype float
        Evaluation point.
    h : float
        Complex step size.
    m : int
        Output dimension (number of rows in J), pre-determined by probe().
    k0_imag : np.ndarray of shape (m,) or None
        Pre-computed Im(g(x + ih·e₀)) from probe().  When provided the loop
        starts at k=1 and J[:, 0] is filled from this value.

    Returns
    -------
    np.ndarray, shape (m, n)
        Jacobian matrix.  J[i, j] = ∂fᵢ/∂xⱼ.
    """
    n = x_flat.size
    J = np.empty((m, n))
    xc = x_flat.astype(complex)

    start = 0
    if k0_imag is not None:
        J[:, 0] = k0_imag / h
        start = 1

    for k in range(start, n):
        xc[k] += 1j * h
        J[:, k] = np.imag(np.asarray(g(xc)).ravel()) / h
        xc[k] = x_flat[k]

    return J


def directional_derivative(
    g, x_flat: np.ndarray, v_flat: np.ndarray, h: float
):
    """
    Jacobian-vector product J(x) @ v in one function evaluation.

    Uses the identity:  Im(g(x + ih·v)) / h = J(x) @ v

    This follows from the Taylor expansion:
        g(x + ih·v) = g(x) + ih·J(x)·v  +  O(h²)
    Taking the imaginary part and dividing by h gives J(x)·v exactly up to
    O(h²) truncation error (which is ≈ 10⁻⁴⁶ for h = default_step()).

    Cost: 1 function evaluation.  ComplexStepError and NonanalyticWarning
    are detected from this single evaluation — no separate probe is needed.

    Parameters
    ----------
    g : callable
        (n_flat,) complex → scalar or ndarray.  Must support complex input.
    x_flat : np.ndarray, shape (n,), dtype float
        Evaluation point.
    v_flat : np.ndarray, shape (n,), dtype float
        Direction vector.  Need not be unit-normalised; result is J@v,
        not the unit-direction derivative J@(v/‖v‖).
    h : float
        Complex step size.

    Returns
    -------
    float
        If g produces scalar output (m=1): the scalar J(x)@v.
    np.ndarray, shape (m,)
        If g produces vector output: the Jacobian-vector product.
    """
    xc = x_flat.astype(complex) + 1j * h * v_flat

    try:
        result = g(xc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ComplexStepError(
            f"f raised {type(exc).__name__} when called with complex-valued input:\n"
            f"  {exc}\n\n"
            "Complex step differentiation requires the function to accept and propagate "
            "complex values throughout all of its operations.\n\n"
            "Common causes:\n"
            "  - np.abs() on a complex intermediate: use np.sqrt(x.real**2 + x.imag**2)\n"
            "    or restructure so abs() is never applied to a perturbed variable.\n"
            "  - np.sign(), np.floor(), np.ceil(), np.round() on a complex value.\n"
            "  - A conditional branch (if x > 0) that implicitly casts complex → float.\n"
            "  - A C/Fortran extension that internally converts its input to float.\n\n"
            "Suggestion: use central finite differences (h ≈ 1e-6) as a fallback."
        ) from exc

    result_arr = np.asarray(result).ravel()

    imag_norm = float(np.max(np.abs(np.imag(result_arr))))
    if imag_norm < 1e-200:
        warnings.warn(
            "Im(f(x+ih·v)) ≈ 0 at the evaluation point. f may be silently discarding "
            "the imaginary part of its inputs.\n"
            "Common causes: np.abs, np.sign, a conditional branch, or a C/Fortran "
            "extension that casts inputs to float.\n"
            "Derivative estimates will be numerically zero (incorrect).",
            NonanalyticWarning,
            stacklevel=3,
        )

    jvp = np.imag(result_arr) / h

    if jvp.ndim == 0 or jvp.size == 1:
        return float(jvp.ravel()[0])
    return jvp
