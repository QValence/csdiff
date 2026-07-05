"""
DiffInfo: lightweight diagnostics container for complex step computations.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffInfo:
    """
    Diagnostics returned when ``return_diagnostics=True``.

    This is a frozen dataclass (immutable after construction).

    Attributes
    ----------
    n_calls : int
        Number of times f was evaluated to compute the derivative.
        Does not include the one-time probe call used to verify complex support.

        Typical values:
        - ``derivative``: always 1 (scalar input needs one evaluation).
        - ``gradient`` / ``jacobian`` (serial): n (one per input dimension).
        - ``gradient`` / ``jacobian`` (batched=True, vectorized): n.
        - ``gradient`` / ``jacobian`` (batched=True, serial fallback): n_batch × n.
        - ``directional_derivative``: always 1 in both serial and vectorized mode.
        - ``directional_derivative`` (batched=True, serial fallback): n_batch.

    elapsed : float
        Total wall-clock time spent inside f across all counted calls, in
        seconds.  Measured with ``time.perf_counter()``.  Does not include
        overhead from this package (argument binding, array allocation, etc.).

    h : float
        The complex step size actually used.  Equals the user-supplied h, or
        the dtype-derived default when h=None was passed.
    """
    n_calls: int
    elapsed: float
    h: float

    def __repr__(self) -> str:
        return (
            f"DiffInfo(n_calls={self.n_calls}, "
            f"elapsed={self.elapsed:.3e}s, "
            f"h={self.h:.3e})"
        )
