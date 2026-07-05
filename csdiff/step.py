"""
Default step size selection and validation for complex step differentiation.

Derivation of the default step
-------------------------------
The complex step formula gives:

    f'(x) = Im(f(x + ih)) / h  +  O(h²)

Unlike finite differences, there is *no* cancellation error: the imaginary
part is extracted, not subtracted from a nearby value.  The only error is
the O(h²) truncation term.

Setting h = eps^(3/2) (where eps = machine epsilon for the dtype) gives:

    truncation error ≈ h² = eps³  ≈ 1e-48  for float64  (eps ≈ 2.22e-16)
                                  ≈ 1e-21  for float32  (eps ≈ 1.19e-7)

Both are far below machine epsilon, so the result is accurate to full
machine precision.

Lower bound on h: Im(f(x+ih)) ≈ h * f'(x).  For |f'| ≈ 1 and h < eps^2
this product approaches float64 underflow (~5e-324), causing Im → 0 and a
zero derivative estimate.  The default h = eps^(3/2) is far above this.

Upper bound on h: h > sqrt(eps) makes the truncation error approach eps
itself, losing accuracy.  The default is well below this limit.
"""
import warnings

import numpy as np

from csdiff.exceptions import StepSizeWarning


def default_step(dtype=float) -> float:
    """
    Return the default complex step size for a given float dtype.

    The returned value h = eps^(3/2) gives truncation error h² = eps³ which
    is negligible compared to machine epsilon eps, while keeping
    Im(f(x+ih)) = h * f'(x) well above float underflow for all practical
    derivative magnitudes.

    Parameters
    ----------
    dtype : dtype-like, optional
        A NumPy float dtype or anything accepted by ``np.finfo``.
        Default: ``float`` (which resolves to float64).

    Returns
    -------
    float
        Optimal complex step size for the given precision.
        ≈ 1.05e-23 for float64, ≈ 4.1e-11 for float32.

    Examples
    --------
    >>> default_step()
    1.0517578125e-23
    >>> default_step(np.float32)
    4.0986...e-11
    """
    eps = np.finfo(dtype).eps
    return float(eps ** 1.5)


def validate_step(h: float, dtype=float) -> None:
    """
    Warn if h is outside the recommended range for the given float dtype.

    Parameters
    ----------
    h : float
        The step size to validate.
    dtype : dtype-like, optional
        NumPy float dtype used to derive recommended bounds. Default: float64.

    Raises
    ------
    ValueError
        If h is not strictly positive.

    Warns
    -----
    StepSizeWarning
        If h > sqrt(eps): truncation error O(h²) approaches machine epsilon,
        risking accuracy loss.
    StepSizeWarning
        If h < eps²: Im(f(x+ih)) ≈ h*f'(x) may underflow for |f'| ≈ 1,
        producing a zero derivative estimate.

    Examples
    --------
    >>> validate_step(1e-20)        # silent — within safe range
    >>> validate_step(1e-4)         # issues StepSizeWarning (too large)
    """
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h!r}.")

    eps = np.finfo(dtype).eps

    if h > eps ** 0.5:
        # Truncation error h² ≈ eps — accuracy degrades to ~sqrt(eps) digits
        warnings.warn(
            f"h={h:.2e} is large: truncation error O(h²) ≈ {h**2:.1e} "
            f"approaches machine epsilon ({eps:.1e}). "
            f"For full precision use h < {eps**0.5:.1e} "
            f"(default: {eps**1.5:.1e}).",
            StepSizeWarning,
            stacklevel=3,
        )
    elif h < eps ** 2:
        # h*|f'(x)| ≈ eps^2 * 1 ≈ 5e-32 for float64 — near underflow territory
        warnings.warn(
            f"h={h:.2e} is very small: Im(f(x+ih)) ≈ h*f'(x) may underflow "
            f"to zero for |f'(x)| ≈ 1 (float64 underflow ≈ 5e-324). "
            f"Recommended: h > {eps**2:.1e} "
            f"(default: {eps**1.5:.1e}).",
            StepSizeWarning,
            stacklevel=3,
        )
