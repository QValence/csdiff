"""
CallTemplate: argument binding and flat ↔ structured conversion.

The differentiation kernels in core.py always work with the simple interface:

    g(x_flat_complex: ndarray shape (n_flat,)) → scalar or ndarray

CallTemplate wraps an arbitrary callable f and handles:
  1. Resolving ``wrt`` (by parameter name or position) to indices in ``args``.
  2. Flattening the identified wrt argument(s) into a contiguous 1-D vector.
  3. Rebuilding the full structured call from a flat complex vector at each
     derivative evaluation, keeping all non-wrt arguments fixed.

Design note: the template stores the original ``args`` list and replaces only
the wrt elements by index.  This avoids any dependency on dict-based binding
and works for any function signature, including those with *args parameters
or uninspectable C extensions (as long as wrt is specified by position).
"""
import inspect

import numpy as np


class CallTemplate:
    """
    Binds f's call context and exposes a flat complex-input interface.

    Parameters
    ----------
    f : callable
        The function to differentiate.
    args : tuple
        All positional arguments passed to f at the evaluation point.
    wrt : str | int | tuple[str | int] | None
        Which argument(s) to differentiate with respect to.
        - None: valid only when len(args) == 1 (single-argument shortcut).
        - str: argument selected by parameter name (requires inspectable sig).
        - int: argument selected by 0-based position in args.
        - tuple: multiple arguments combined into one flat input.

    Attributes
    ----------
    f : callable
    args : list
        Original positional arguments (all of them, including non-wrt ones).
    param_names : list[str] or None
        Ordered positional parameter names from f's signature, or None
        when the signature is not inspectable (e.g. built-ins, C extensions).
    wrt_indices : list[int]
        Positions (in args) of the wrt arguments.
    wrt_shapes : list[tuple]
        Original shape of each wrt array, used to reshape on rebuild.
    wrt_sizes : list[int]
        Number of scalar elements in each wrt array.
    n_flat : int
        Total length of the flat differentiation input (sum of wrt_sizes).
    """

    def __init__(self, f, args, wrt):
        self.f = f
        self.args = list(args)

        # Attempt to extract positional parameter names for wrt-by-name support.
        # Only POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD parameters are relevant;
        # VAR_POSITIONAL (*args) and keyword-only parameters are excluded.
        try:
            sig = inspect.signature(f)
            self.param_names = [
                p.name
                for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        except (ValueError, TypeError):
            # inspect.signature raises for some built-ins and C extensions
            self.param_names = None

        # Resolve wrt → list of integer indices into args
        if wrt is None:
            # Single-argument shortcut: differentiate wrt the only argument.
            # The check len(args) == 1 is enforced by the calling function,
            # not here, to keep error messages at the user-facing call site.
            wrt_indices = [0]
        else:
            # Normalise scalar wrt to a length-1 tuple for uniform handling
            if isinstance(wrt, (str, int)):
                wrt = (wrt,)
            elif not isinstance(wrt, tuple):
                raise TypeError(
                    f"wrt must be str, int, or a tuple of str/int, "
                    f"got {type(wrt).__name__}. "
                    "Use a tuple (not a list) to differentiate wrt multiple arguments."
                )

            wrt_indices = []
            for w in wrt:
                if isinstance(w, int):
                    if not (0 <= w < len(args)):
                        raise IndexError(
                            f"wrt={w} is out of range: f was called with "
                            f"{len(args)} positional argument(s) "
                            f"(valid positions: 0 to {len(args) - 1})."
                        )
                    wrt_indices.append(w)

                elif isinstance(w, str):
                    if self.param_names is None:
                        raise TypeError(
                            f"wrt='{w}' (by name) requires an inspectable function "
                            "signature, but f's signature could not be determined. "
                            "This happens with some C extensions and built-ins. "
                            "Use wrt=int (position index) instead."
                        )
                    if w not in self.param_names:
                        raise ValueError(
                            f"Parameter '{w}' not found in f's signature. "
                            f"Available positional parameters: {self.param_names}."
                        )
                    wrt_indices.append(self.param_names.index(w))

                else:
                    raise TypeError(
                        f"Each element of wrt must be str or int, "
                        f"got {type(w).__name__}."
                    )

        self.wrt_indices = wrt_indices

        # Characterise each wrt argument so we can flatten and rebuild it
        wrt_arrays = [np.asarray(args[i]) for i in wrt_indices]
        self.wrt_shapes = [a.shape for a in wrt_arrays]
        self.wrt_sizes = [a.size for a in wrt_arrays]
        self.n_flat = sum(self.wrt_sizes)

    # ------------------------------------------------------------------ #
    #   Properties
    # ------------------------------------------------------------------ #

    @property
    def wrt_names(self):
        """
        Parameter names of the wrt targets, or None if unavailable.

        Returns None when param_names could not be determined (e.g. for
        C extensions).  In that case wrt was resolved by position.

        Returns
        -------
        list[str] or None
        """
        if self.param_names is None:
            return None
        return [self.param_names[i] for i in self.wrt_indices]

    # ------------------------------------------------------------------ #
    #   Core interface used by the differentiation kernels
    # ------------------------------------------------------------------ #

    def flat_point(self) -> np.ndarray:
        """
        Return the current wrt values as a real flat array of shape (n_flat,).

        For a single wrt argument of shape (p, q), this is a.ravel() of
        length p*q.  For combined wrt, the arrays are concatenated in the
        order they appear in the wrt tuple.

        Returns
        -------
        np.ndarray, shape (n_flat,), dtype float64
        """
        parts = [
            np.asarray(self.args[i], dtype=float).ravel()
            for i in self.wrt_indices
        ]
        # np.concatenate on a single-element list is fine and returns a copy
        return np.concatenate(parts)

    def __call__(self, x_flat_complex: np.ndarray):
        """
        Substitute x_flat_complex into the wrt positions and call f.

        The flat complex vector is split and reshaped to match the original
        wrt argument shapes before substitution.  All non-wrt arguments are
        passed through unchanged (as references, not copies).

        Parameters
        ----------
        x_flat_complex : np.ndarray, shape (n_flat,), dtype complex
            Flat complex perturbation of the wrt arguments.

        Returns
        -------
        Output of f (scalar or array).
        """
        # Shallow copy: non-wrt entries are shared references (safe — we never
        # modify them, only replace the wrt slots with new complex arrays).
        new_args = list(self.args)

        offset = 0
        for idx, shape, size in zip(
            self.wrt_indices, self.wrt_shapes, self.wrt_sizes
        ):
            # Slice out this wrt argument's portion and reshape to original shape
            new_args[idx] = x_flat_complex[offset: offset + size].reshape(shape)
            offset += size

        return self.f(*new_args)

    def batch_callable(self):
        """
        Return a callable for batched Jacobian computation (``batched=True``).

        The returned callable accepts a 2-D complex array X of shape
        (n_flat, n_flat) — one perturbed copy of the flat input per row —
        and passes the batched wrt argument to f directly.  f must therefore
        accept the wrt argument with an extra leading batch dimension:

            original shape (p, q)  →  batch shape (n_flat, p, q)

        Only supported when a single wrt target is specified.  Raises
        NotImplementedError for combined wrt (tuple with len > 1).

        Returns
        -------
        callable
            g_batch(X: complex ndarray (n, n_flat)) → f's output for the batch
        """
        if len(self.wrt_indices) > 1:
            raise NotImplementedError(
                "batched=True is not supported for combined wrt (tuple with "
                "multiple targets) in v1. Use batched=False, or restructure "
                "f to accept a batched input for a single wrt argument."
            )

        idx = self.wrt_indices[0]
        shape = self.wrt_shapes[0]
        base_args = list(self.args)  # frozen reference copy

        def g_batch(X):
            # X: (batch, n_flat) complex.
            # Reshape the wrt argument to (batch, *original_shape) and call f.
            batch = X.shape[0]
            new_args = list(base_args)
            new_args[idx] = X.reshape(batch, *shape)
            return self.f(*new_args)

        return g_batch
