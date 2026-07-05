"""
Custom exceptions and warnings for csdiff.
"""


class ComplexStepError(RuntimeError):
    """
    Raised when complex step differentiation cannot be applied to f.

    This occurs when f raises an exception with complex-valued input (e.g. a
    C extension that casts inputs to float) or when Im(f(x+ih)) is
    numerically zero for all outputs, indicating that f silently discards the
    imaginary part.

    The exception message explains the likely cause and suggests a fallback.
    """


class NonanalyticWarning(UserWarning):
    """
    Issued when Im(f(x+ih)) ≈ 0 at the probe point.

    This suggests that f may not be analytic in the complex sense — for example
    because it uses np.abs(), np.sign(), conditional branches on real values,
    or a C/Fortran extension that internally converts its inputs to float.
    In these cases the imaginary part is silently lost and the derivative
    estimate will be numerically zero (incorrect).
    """


class StepSizeWarning(UserWarning):
    """
    Issued when the requested step size h is outside the recommended range.

    Too large (h > sqrt(eps)): truncation error O(h²) approaches machine
    epsilon, degrading accuracy.

    Too small (h < eps²): Im(f(x+ih)) ≈ h * f'(x) may underflow to zero for
    derivatives of typical magnitude, producing a zero result.

    The recommended default ``default_step()`` avoids both extremes.
    """
