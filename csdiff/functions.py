"""
Public API: derivative, gradient, jacobian, directional_derivative.

Calling convention (all functions except derivative)
------------------------------------------------------
    f(*args, *, wrt=None, h=None, return_diagnostics=False, batched=False)

  f       : callable whose derivative is sought
  *args   : all positional arguments to f at the evaluation point
  wrt     : which argument to differentiate wrt (required when len(args) > 1)
  h       : complex step size (default: default_step(x_wrt.dtype))
  return_diagnostics : if True, return (result, DiffInfo) instead of result
  batched : if True, the wrt argument's first dimension is the batch dimension;
            output shape is (n_batch, *existing_output_shape)

wrt resolution
--------------
- len(args) == 1 and wrt is None : differentiate wrt the only argument
- len(args) >  1 and wrt is None : TypeError (wrt required)
- wrt = str                      : argument selected by parameter name
- wrt = int                      : argument selected by 0-based position
- wrt = tuple[str | int]         : multiple arguments combined (jacobian only)

Dimension dispatch
------------------
Each function validates (n, m) via a probe evaluation and raises ValueError
with a clear redirect if the wrong function was called:

  n=1, m=1  →  derivative() only  (scalar in, scalar out)
  n>1, m=1  →  gradient() only    (array in, scalar out)
  n≥1, m>1  →  jacobian()         (any in, vector out — includes f: ℝ→ℝᵐ)

directional_derivative() applies to any (n, m) except n=1, m=1.
"""
import time
import warnings

import numpy as np

from csdiff.template import CallTemplate
from csdiff.core import (
    probe,
    gradient_serial,
    jacobian_serial,
    directional_derivative as _dd_kernel,
)
from csdiff.step import default_step, validate_step
from csdiff.info import DiffInfo
from csdiff.exceptions import ComplexStepError, NonanalyticWarning


# --------------------------------------------------------------------------- #
#   Internal helpers
# --------------------------------------------------------------------------- #

def _check_single_arg_wrt(args, wrt, fn_name):
    """Raise TypeError if multiple args were given without specifying wrt."""
    if len(args) > 1 and wrt is None:
        raise TypeError(
            f"{fn_name}() received {len(args)} positional arguments but wrt is None. "
            "When f has more than one argument you must specify which one to "
            "differentiate: e.g. wrt='x', wrt=1, or wrt=('x', 'u')."
        )


def _build(f, args, wrt, h):
    """
    Build a CallTemplate, resolve h, validate h, and return (tmpl, x_flat, h).

    This is the common setup shared by gradient, jacobian, and
    directional_derivative.
    """
    tmpl = CallTemplate(f, args, wrt)
    x_flat = tmpl.flat_point()
    h_actual = h if h is not None else default_step(x_flat.dtype)
    validate_step(h_actual, x_flat.dtype)
    return tmpl, x_flat, h_actual


def _counting_wrapper(g):
    """
    Wrap g so each call is timed and counted.

    Returns (g_wrapped, get_stats) where get_stats() → (n_calls, elapsed).
    The wrapper adds two perf_counter() calls per evaluation (negligible
    compared to even the cheapest numpy function).
    """
    counter = {"n": 0, "t": 0.0}

    def g_wrapped(xc):
        t0 = time.perf_counter()
        result = g(xc)
        counter["t"] += time.perf_counter() - t0
        counter["n"] += 1
        return result

    def get_stats():
        return counter["n"], counter["t"]

    return g_wrapped, get_stats


def _wrt_to_index(f, args, wrt):
    """Resolve wrt to a single positional index for batch dispatch.

    Raises ValueError for tuple wrt (ambiguous: which sub-argument is batched?)
    """
    if wrt is None:
        return 0
    if isinstance(wrt, tuple):
        raise ValueError(
            "batched=True does not support combined wrt (tuple of targets): "
            "it is ambiguous which argument carries the batch dimension. "
            f"Got wrt={wrt!r}. "
            "Differentiate wrt a single argument at a time, e.g. wrt='x'."
        )
    # Use a dummy CallTemplate to reuse existing wrt-by-name/position logic.
    # Scalar 0 placeholders are fine — we only need wrt_indices, not shapes.
    dummy = tuple(0 for _ in args)
    return CallTemplate(f, dummy, wrt).wrt_indices[0]


# Two distinct probe batch sizes are required for reliable vectorized detection.
#
# A single probe with batch size B fails when the function output dimension m
# equals B: a non-vectorized f(x) that indexes x[0], x[1], … and stacks m
# values returns shape (m, n_flat) when called on a (B, n_flat) batch input,
# and (m, n_flat).shape[0] == B passes the shape check as a false positive.
#
# Two probes with different sizes (3 and 5) rule this out: a truly vectorized f
# returns (3, …) for batch-3 and (5, …) for batch-5; a non-vectorized f returns
# the same fixed (m, n_flat) shape for both, so its shape[0] fails to match the
# second batch size.  3 and 5 are chosen because m=1, 2 are uncommon via jacobian
# (gradient handles those), and primes minimise accidental collisions.
_PROBE_BATCH = 3
_PROBE_BATCH_B = 5


def _probe_vectorized(g_batch, x_flat, h, expected_ndim=None):
    """Return (vectorized, probe_out) by testing g_batch with two batch sizes.

    A non-vectorized f fed a (B, n) input may coincidentally return a (B, n)
    output (false positive) when its output dimension m == B.  Two probes with
    sizes _PROBE_BATCH=3 and _PROBE_BATCH_B=5 catch this: a truly vectorized f
    scales its first output dimension with the batch size; a non-vectorized one
    does not, so it fails the second probe.

    If expected_ndim is given, also require out.ndim == expected_ndim.
    Pass expected_ndim=None to accept any ndim (used by _apply_batched_dd).

    Suppresses warnings during the probe — they surface in the serial fallback
    via the normal probe() function.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            probe_X = np.tile(x_flat, (_PROBE_BATCH, 1)).astype(complex)
            probe_X[:, 0] += 1j * h
            out = np.asarray(g_batch(probe_X))
            if out.shape[0] != _PROBE_BATCH:
                return False, None
            if expected_ndim is not None and out.ndim != expected_ndim:
                return False, None

            # Second probe with a different batch size to rule out non-vectorized
            # functions that happen to return shape[0] == _PROBE_BATCH by accident.
            probe_X_b = np.tile(x_flat, (_PROBE_BATCH_B, 1)).astype(complex)
            probe_X_b[:, 0] += 1j * h
            out_b = np.asarray(g_batch(probe_X_b))
            if out_b.shape[0] != _PROBE_BATCH_B or out_b.shape[1:] != out.shape[1:]:
                return False, None

            return True, out
        except Exception:
            pass
    return False, None


def _batched_setup(fn_name, f, args, wrt, h):
    """
    Validate and build the common batch context shared by all three batched functions.

    Resolves wrt to an index, extracts and validates X_batch, builds a single-point
    CallTemplate from the first batch element, and returns the batch-callable form.

    Returns
    -------
    (wrt_idx, X_batch, n_batch, x_flat, h_actual, n, g_batch, X_flat_batch)
    """
    wrt_idx = _wrt_to_index(f, args, wrt)
    X_batch = np.asarray(args[wrt_idx], dtype=float)
    if X_batch.ndim < 2:
        raise ValueError(
            f"{fn_name}() with batched=True requires the wrt argument to be "
            f"at least 2-D (n_batch, n), got shape {X_batch.shape}. "
            "The first dimension is the batch size."
        )
    n_batch = X_batch.shape[0]

    args_s = list(args)
    args_s[wrt_idx] = X_batch[0]
    tmpl, x_flat, h_actual = _build(f, tuple(args_s), wrt, h)
    n = x_flat.size

    g_batch = tmpl.batch_callable()
    X_flat_batch = X_batch.reshape(n_batch, n)
    return wrt_idx, X_batch, n_batch, x_flat, h_actual, n, g_batch, X_flat_batch


def _batched_serial_loop(call_one, args, wrt_idx, X_batch, h_actual, return_diagnostics):
    """
    Serial fallback loop for batched gradient/jacobian.

    Calls ``call_one(args_i, return_diagnostics)`` for each batch element i and
    stacks the results.  ``call_one`` must return either a plain array (when
    return_diagnostics=False) or (array, DiffInfo) (when True).
    """
    results = []
    total_calls, total_elapsed = 0, 0.0
    for i in range(len(X_batch)):
        args_i = list(args)
        args_i[wrt_idx] = X_batch[i]
        if return_diagnostics:
            r, info = call_one(args_i, True)
            results.append(r)
            total_calls += info.n_calls
            total_elapsed += info.elapsed
        else:
            results.append(call_one(args_i, False))
    stacked = np.stack(results)
    if return_diagnostics:
        return stacked, DiffInfo(n_calls=total_calls, elapsed=total_elapsed, h=h_actual)
    return stacked


def _apply_batched_gradient(f, args, wrt, h, return_diagnostics):
    wrt_idx, X_batch, n_batch, x_flat, h_actual, n, g_batch, X_flat_batch = \
        _batched_setup("gradient", f, args, wrt, h)

    vectorized, _ = _probe_vectorized(g_batch, x_flat, h_actual, expected_ndim=1)
    if vectorized:
        G = np.empty((n_batch, n))
        X_c = X_flat_batch.astype(complex)
        for k in range(n):
            X_c[:, k] += 1j * h_actual
            G[:, k] = np.imag(np.asarray(g_batch(X_c))) / h_actual
            X_c[:, k] = X_flat_batch[:, k]
        if return_diagnostics:
            return G, DiffInfo(n_calls=n, elapsed=0.0, h=h_actual)
        return G

    def call_one(args_i, rd):
        return gradient(f, *args_i, wrt=wrt, h=h, return_diagnostics=rd)
    return _batched_serial_loop(call_one, args, wrt_idx, X_batch, h_actual, return_diagnostics)


def _apply_batched_jacobian(f, args, wrt, h, return_diagnostics):
    wrt_idx, X_batch, n_batch, x_flat, h_actual, n, g_batch, X_flat_batch = \
        _batched_setup("jacobian", f, args, wrt, h)

    vectorized, probe_out = _probe_vectorized(g_batch, x_flat, h_actual, expected_ndim=2)
    if vectorized:
        m = probe_out.shape[1]  # probe_out.shape = (_PROBE_BATCH, m)
        J = np.empty((n_batch, m, n))
        X_c = X_flat_batch.astype(complex)
        for k in range(n):
            X_c[:, k] += 1j * h_actual
            J[:, :, k] = np.imag(np.asarray(g_batch(X_c))) / h_actual
            X_c[:, k] = X_flat_batch[:, k]
        if return_diagnostics:
            return J, DiffInfo(n_calls=n, elapsed=0.0, h=h_actual)
        return J

    def call_one(args_i, rd):
        return jacobian(f, *args_i, wrt=wrt, h=h, return_diagnostics=rd)
    return _batched_serial_loop(call_one, args, wrt_idx, X_batch, h_actual, return_diagnostics)


def _apply_batched_dd(f, args, v, wrt, h, return_diagnostics):
    wrt_idx, X_batch, n_batch, x_flat, h_actual, n, g_batch, X_flat_batch = \
        _batched_setup("directional_derivative", f, args, wrt, h)

    v_arr = np.asarray(v, dtype=float)
    v_is_batched = v_arr.ndim >= 2
    if v_is_batched and v_arr.shape[0] != n_batch:
        raise ValueError(
            f"v has batch size {v_arr.shape[0]} but the wrt argument has batch "
            f"size {n_batch}. When v is 2-D, its first dimension must match."
        )
    V_flat = v_arr.reshape(n_batch, n) if v_is_batched else np.tile(v_arr.ravel(), (n_batch, 1))

    vectorized, _ = _probe_vectorized(g_batch, x_flat, h_actual)
    if vectorized:
        X_c = (X_flat_batch + 1j * h_actual * V_flat).astype(complex)
        out = np.imag(np.asarray(g_batch(X_c))) / h_actual
        if return_diagnostics:
            return out, DiffInfo(n_calls=1, elapsed=0.0, h=h_actual)
        return out

    results = []
    total_calls, total_elapsed = 0, 0.0
    for i in range(n_batch):
        args_i = list(args)
        args_i[wrt_idx] = X_batch[i]
        v_i = v_arr[i] if v_is_batched else v_arr
        if return_diagnostics:
            r, info = directional_derivative(
                f, *args_i, v=v_i, wrt=wrt, h=h, return_diagnostics=True
            )
            results.append(np.atleast_1d(r))
            total_calls += info.n_calls
            total_elapsed += info.elapsed
        else:
            results.append(np.atleast_1d(
                directional_derivative(f, *args_i, v=v_i, wrt=wrt, h=h)
            ))
    stacked = np.stack(results)
    if stacked.shape[-1] == 1:  # scalar f: (n_batch, 1) → (n_batch,)
        stacked = stacked.squeeze(-1)
    if return_diagnostics:
        return stacked, DiffInfo(n_calls=total_calls, elapsed=total_elapsed, h=h_actual)
    return stacked


# --------------------------------------------------------------------------- #
#   Public functions
# --------------------------------------------------------------------------- #

def derivative(f, x0, *, h=None, return_diagnostics=False, batched=False):
    """
    First derivative of a strictly scalar-in, scalar-out function f: ℝ → ℝ.

    This is the specialised n=1, m=1 case.  x0 must be a Python scalar (float,
    int, or complex-castable).  For vector input or vector output use
    ``gradient()`` or ``jacobian()`` respectively.

    Parameters
    ----------
    f : callable
        f: ℝ → ℝ.  Must accept a complex scalar and return a complex scalar.
        Called as ``f(x0 + 1j*h)``.
    x0 : float or array-like of float
        Scalar evaluation point, or 1-D array of shape (n_batch,) when
        ``batched=True``.
    h : float or None, optional
        Complex step size.  Default: ``default_step(float)`` ≈ 1.05e-23.
    return_diagnostics : bool, optional
        If True, return ``(result, DiffInfo)`` instead of just ``result``.
        ``DiffInfo.n_calls`` is always 1.
    batched : bool, optional
        If True, x0 must be a 1-D array of shape (n_batch,).  f is called
        with the entire array first (vectorized attempt); if that fails, f is
        called once per element (serial fallback).  Output shape: (n_batch,).

    Returns
    -------
    float
        f'(x0).
    np.ndarray, shape (n_batch,)
        When ``batched=True``.
    tuple[result, DiffInfo]
        If ``return_diagnostics=True``.

    Raises
    ------
    ComplexStepError
        If f raises an exception with complex input.
    ValueError
        If f returns more than one value (use jacobian() instead).

    Warns
    -----
    NonanalyticWarning
        If Im(f(x0+ih)) ≈ 0 (f may silently discard the imaginary part).
    StepSizeWarning
        If h is outside the recommended range.

    Examples
    --------
    >>> from csdiff import derivative
    >>> import numpy as np
    >>> derivative(lambda x: x**3, 2.0)
    12.0
    >>> dx, info = derivative(np.exp, 1.0, return_diagnostics=True)
    >>> abs(dx - np.e) < 1e-14
    True
    >>> info.n_calls
    1
    """
    if batched:
        h_actual = h if h is not None else default_step(float)
        validate_step(h_actual)
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.ndim != 1:
            raise ValueError(
                f"derivative() with batched=True requires x0 to be 1-D "
                f"(n_batch,), got shape {x0_arr.shape}."
            )
        # Vectorized attempt: works for numpy ufuncs and vectorized functions
        try:
            result_c = f(x0_arr.astype(complex) + 1j * h_actual)
            arr = np.asarray(result_c)
            if arr.shape == x0_arr.shape:
                deriv = np.imag(arr) / h_actual
                if return_diagnostics:
                    return deriv, DiffInfo(n_calls=1, elapsed=0.0, h=h_actual)
                return deriv
        except Exception:
            pass
        # Serial fallback
        if return_diagnostics:
            pairs = [derivative(f, float(xi), h=h, return_diagnostics=True)
                     for xi in x0_arr]
            rs, infos = zip(*pairs)
            return np.array(rs), DiffInfo(
                n_calls=sum(i.n_calls for i in infos),
                elapsed=sum(i.elapsed for i in infos),
                h=infos[0].h,
            )
        return np.array([derivative(f, float(xi), h=h) for xi in x0_arr])

    h_actual = h if h is not None else default_step(float)
    validate_step(h_actual)

    x0_c = complex(x0) + 1j * h_actual

    t0 = time.perf_counter() if return_diagnostics else None

    try:
        result = f(x0_c)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ComplexStepError(
            f"f raised {type(exc).__name__} when called with complex input:\n"
            f"  {exc}\n\n"
            "derivative() requires f to accept complex-valued scalars.\n"
            "Common causes: an np.abs / np.sign on the input, or a C extension\n"
            "that converts its argument to float.\n"
            "Suggestion: use central finite differences (h ≈ 1e-6) as a fallback."
        ) from exc

    elapsed = (time.perf_counter() - t0) if return_diagnostics else 0.0

    result_arr = np.asarray(result)
    if result_arr.size > 1:
        raise ValueError(
            f"derivative() requires f: ℝ→ℝ (scalar output), but f returned "
            f"{result_arr.size} values. "
            "Use jacobian() for f: ℝ→ℝᵐ with m > 1."
        )

    im = float(np.imag(result))
    if abs(im) < 1e-200:
        warnings.warn(
            "Im(f(x0+ih)) ≈ 0. f may be silently discarding the imaginary part "
            "(e.g. via np.abs, a conditional, or a C extension). "
            "The derivative estimate will be zero (incorrect).",
            NonanalyticWarning,
            stacklevel=2,
        )

    deriv = im / h_actual

    if return_diagnostics:
        return deriv, DiffInfo(n_calls=1, elapsed=elapsed, h=h_actual)
    return deriv


def gradient(f, *args, wrt=None, h=None, return_diagnostics=False, batched=False):
    """
    Gradient of f: ℝⁿ → ℝ with n > 1 (scalar output, array input required).

    Parameters
    ----------
    f : callable
        Must return a scalar and accept array input with n > 1 elements.
    *args : any
        All positional arguments to f at the evaluation point.  The argument
        identified by ``wrt`` must be array-like; it is flattened to a 1-D
        vector of length n internally.
    wrt : str | int | None, optional
        Which argument to differentiate.
        - None (default): valid only when ``len(args) == 1``.
        - str: argument selected by parameter name (requires inspectable sig).
        - int: argument selected by 0-based position.
        Raises ``TypeError`` when ``len(args) > 1`` and ``wrt`` is None.
    h : float or None, optional
        Complex step size.  Default: ``default_step(x_wrt.dtype)``.
    return_diagnostics : bool, optional
        If True, return ``(result, DiffInfo)`` instead of just ``result``.
    batched : bool, optional
        If True, the wrt argument must be 2-D with shape (n_batch, n).
        f is first called with the full batch (vectorized attempt); on failure,
        falls back to a serial loop.  Output shape: (n_batch, n).

    Returns
    -------
    np.ndarray, shape (n,)
        Gradient vector ∇f at the wrt evaluation point.
    np.ndarray, shape (n_batch, n)
        When ``batched=True``.
    tuple[np.ndarray, DiffInfo]
        If ``return_diagnostics=True``.

    Raises
    ------
    TypeError
        If ``len(args) > 1`` and ``wrt`` is None.
    ValueError
        If n == 1 (use derivative() or jacobian()) or m > 1 (use jacobian()).
    ComplexStepError
        If f does not support complex arithmetic.

    Warns
    -----
    NonanalyticWarning, StepSizeWarning

    Examples
    --------
    >>> import numpy as np
    >>> from csdiff import gradient
    >>> gradient(lambda x: np.sum(x**2), np.array([1., 2., 3.]))
    array([2., 4., 6.])

    >>> def f(t, x, u):
    ...     return t * np.dot(x, u)
    >>> gradient(f, 2.0, np.array([1., 0.]), np.array([0., 3.]), wrt="x")
    array([0., 6.])
    """
    _check_single_arg_wrt(args, wrt, "gradient")
    if batched:
        return _apply_batched_gradient(f, args, wrt, h, return_diagnostics)

    tmpl, x_flat, h_actual = _build(f, args, wrt, h)

    n = x_flat.size
    m, k0_imag = probe(tmpl, x_flat, h_actual)

    if n == 1 and m == 1:
        raise ValueError(
            "gradient() requires n > 1, but the wrt argument has n=1 element "
            "and f is scalar-valued (m=1). "
            "Use derivative() for f: ℝ→ℝ."
        )
    if n == 1:
        raise ValueError(
            f"gradient() requires n > 1, but the wrt argument has n=1 element "
            f"and f returns m={m} values. "
            "Use jacobian() for f: ℝ→ℝᵐ."
        )
    if m > 1:
        raise ValueError(
            f"gradient() requires scalar output (m=1), but f returns m={m} values "
            f"with n={n} inputs. "
            "Use jacobian() for f: ℝⁿ→ℝᵐ."
        )

    g = tmpl
    if return_diagnostics:
        # Wrap g so all n calls go through the counter; k0_imag is not passed so
        # the kernel re-runs k=0 through the wrapper and n_calls stays exactly n.
        g, get_stats = _counting_wrapper(g)
        result = gradient_serial(g, x_flat, h_actual)
        n_calls, elapsed = get_stats()
        return result, DiffInfo(n_calls=n_calls, elapsed=elapsed, h=h_actual)

    # Non-diagnostics hot path: reuse k0_imag from probe to skip one f-call.
    return gradient_serial(g, x_flat, h_actual, k0_imag=k0_imag)


def jacobian(f, *args, wrt=None, h=None, return_diagnostics=False, batched=False):
    """
    Jacobian of f: ℝⁿ → ℝᵐ with m > 1.

    Also handles f: ℝ → ℝᵐ (n=1, m>1), returning shape (m, 1).

    Parameters
    ----------
    f : callable
        Output must be array-like of shape (m,) with m > 1.
    *args : any
        All positional arguments to f.  The argument(s) identified by ``wrt``
        must be array-like.
    wrt : str | int | tuple[str | int] | None, optional
        Which argument(s) to differentiate.
        - None: valid only when ``len(args) == 1``.
        - str: argument by parameter name.
        - int: argument by 0-based position.
        - tuple of str/int: **combined** — multiple arguments are concatenated
          into one flat input of length n = Σ nᵢ.  The Jacobian covers all
          combined inputs: shape (m, Σ nᵢ).
        Raises ``TypeError`` when ``len(args) > 1`` and ``wrt`` is None.
    h : float or None, optional
        Complex step size.  Default: ``default_step(x_wrt.dtype)``.
    return_diagnostics : bool, optional
        If True, return ``(result, DiffInfo)`` instead of just ``result``.
    batched : bool, optional
        If True, the wrt argument must be 2-D with shape (n_batch, n).
        f is first called with the full batch (vectorized attempt); on failure,
        falls back to a serial loop.  Output shape: (n_batch, m, n).
        Not supported for combined wrt (tuple).

    Returns
    -------
    np.ndarray, shape (m, n)
        Jacobian matrix.  **Convention**: J[i, j] = ∂fᵢ/∂xⱼ
        (rows = output components, columns = input components).
    np.ndarray, shape (n_batch, m, n)
        When ``batched=True``.
    tuple[np.ndarray, DiffInfo]
        If ``return_diagnostics=True``.

    Raises
    ------
    TypeError
        If ``len(args) > 1`` and ``wrt`` is None.
    ValueError
        If m == 1 (use derivative() or gradient() instead).
    ComplexStepError
        If f does not support complex arithmetic.

    Warns
    -----
    NonanalyticWarning, StepSizeWarning

    Examples
    --------
    >>> import numpy as np
    >>> from csdiff import jacobian
    >>> jacobian(lambda x: x**2, np.array([1., 2., 3.]))
    array([[2., 0., 0.],
           [0., 4., 0.],
           [0., 0., 6.]])

    >>> def f(t, x, u):
    ...     return np.array([t * x[0] + u[0], x[1] * u[1]])
    >>> jacobian(f, 1.0, np.array([2., 3.]), np.array([4., 5.]), wrt="x")
    array([[1., 0.],
           [0., 5.]])

    >>> # Combined wrt: Jacobian wrt both x and u simultaneously
    >>> jacobian(f, 1.0, np.array([2., 3.]), np.array([4., 5.]), wrt=("x", "u"))
    array([[1., 0., 1., 0.],
           [0., 5., 0., 3.]])
    """
    _check_single_arg_wrt(args, wrt, "jacobian")
    if batched:
        return _apply_batched_jacobian(f, args, wrt, h, return_diagnostics)

    tmpl, x_flat, h_actual = _build(f, args, wrt, h)

    n = x_flat.size
    m, k0_imag = probe(tmpl, x_flat, h_actual)

    if m == 1 and n == 1:
        raise ValueError(
            "jacobian() requires m > 1, but f is scalar-valued (m=1) with "
            "scalar input (n=1). "
            "Use derivative() for f: ℝ→ℝ."
        )
    if m == 1:
        raise ValueError(
            f"jacobian() requires m > 1, but f is scalar-valued (m=1) with "
            f"n={n} inputs. "
            "Use gradient() for f: ℝⁿ→ℝ."
        )

    g = tmpl
    if return_diagnostics:
        g, get_stats = _counting_wrapper(g)
        result = jacobian_serial(g, x_flat, h_actual, m)
        n_calls, elapsed = get_stats()
        return result, DiffInfo(n_calls=n_calls, elapsed=elapsed, h=h_actual)

    return jacobian_serial(g, x_flat, h_actual, m, k0_imag=k0_imag)


def directional_derivative(
    f, *args, v, wrt=None, h=None, return_diagnostics=False, batched=False
):
    """
    Jacobian-vector product J(x) @ v in **one** function evaluation.

    Formula: ``Im(f(x + ih·v)) / h = J(x) @ v``  (exact up to O(h²) truncation).

    For f: ℝⁿ → ℝ (scalar output), this equals the scalar directional
    derivative ∇f(x) · v.

    Parameters
    ----------
    f : callable
        f: ℝⁿ → ℝᵐ.  Must accept complex input.  n=1, m=1 is not supported
        (use derivative() instead).
    *args : any
        All positional arguments to f.
    v : array-like, shape (n,) or (n_batch, n)
        Direction vector.  **Keyword-only**.  Need not be unit-normalised;
        the result is J@v (scaled by ‖v‖), not the unit-direction derivative.
        When ``batched=True``, v may be (n,) (same direction for all batch
        points) or (n_batch, n) (per-point directions).
    wrt : str | int | None, optional
        Which argument to differentiate.  Required when ``len(args) > 1``.
    h : float or None, optional
        Complex step size.  Default: ``default_step(x_wrt.dtype)``.
    return_diagnostics : bool, optional
        If True, return ``(result, DiffInfo)`` instead of just ``result``.
        ``DiffInfo.n_calls`` is always 1.
    batched : bool, optional
        If True, the wrt argument must be 2-D with shape (n_batch, n).
        f is first called with the full batch (vectorized attempt); on failure,
        falls back to a serial loop.
        Output shape: (n_batch,) for scalar f, (n_batch, m) for vector f.

    Returns
    -------
    float
        If f: ℝⁿ → ℝ (scalar output, m=1): the scalar J(x)@v = ∇f·v.
    np.ndarray, shape (m,)
        If f: ℝⁿ → ℝᵐ (vector output, m>1): the Jacobian-vector product.
    np.ndarray, shape (n_batch,) or (n_batch, m)
        When ``batched=True``.
    tuple[result, DiffInfo]
        If ``return_diagnostics=True``.

    Raises
    ------
    TypeError
        If ``len(args) > 1`` and ``wrt`` is None.
    ValueError
        If v does not have the same number of elements as the wrt argument,
        or if f: ℝ→ℝ (use derivative() instead).
    ComplexStepError
        If f does not support complex arithmetic.

    Warns
    -----
    NonanalyticWarning, StepSizeWarning

    Examples
    --------
    >>> import numpy as np
    >>> from csdiff import directional_derivative, jacobian
    >>> x0 = np.array([1., 2.])
    >>> v  = np.array([1., 1.])
    >>> # Directional derivative equals jacobian @ v
    >>> dd = directional_derivative(lambda x: x @ x, x0, v=v)
    >>> J  = jacobian(lambda x: x @ x, x0)
    >>> abs(dd - (J @ v).item()) < 1e-12
    True
    """
    _check_single_arg_wrt(args, wrt, "directional_derivative")
    if batched:
        return _apply_batched_dd(f, args, v, wrt, h, return_diagnostics)

    tmpl, x_flat, h_actual = _build(f, args, wrt, h)

    v_flat = np.asarray(v, dtype=float).ravel()
    if v_flat.size != x_flat.size:
        raise ValueError(
            f"v has {v_flat.size} element(s) but the wrt argument has "
            f"{x_flat.size} element(s). v must have the same shape as the "
            "wrt argument (after flattening)."
        )

    n = x_flat.size

    g = tmpl
    if return_diagnostics:
        g, get_stats = _counting_wrapper(g)

    # _dd_kernel handles ComplexStepError and NonanalyticWarning internally —
    # no separate probe() call is needed (saves 1 redundant f-evaluation).
    result = _dd_kernel(g, x_flat, v_flat, h_actual)

    # n==1, m==1 check: derive m from the result shape rather than the probe.
    m = 1 if isinstance(result, float) else np.asarray(result).size
    if n == 1 and m == 1:
        raise ValueError(
            "directional_derivative() does not apply to f: ℝ→ℝ (n=1, m=1). "
            "Use derivative() for scalar functions."
        )

    if return_diagnostics:
        n_calls, elapsed = get_stats()
        return result, DiffInfo(n_calls=n_calls, elapsed=elapsed, h=h_actual)
    return result
