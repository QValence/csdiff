# csdiff

Complex step differentiation for black-box callables.

Computes first-order derivatives of arbitrary functions f: ℝⁿ → ℝᵐ with **machine-precision accuracy** (~10⁻¹⁶ relative error for float64) and **no step-size tuning**. The function only needs to accept complex-valued inputs — no source code changes required.

```
f'(x)  ≈  Im(f(x + ih)) / h
```

- Truncation error O(h²), zero cancellation error (unlike finite differences).
- Default step h = ε^(3/2) ≈ 1.05 × 10⁻²³ is optimal for float64 automatically.
- Works with any analytic function: polynomials, exponentials, ODE solvers written in pure Python/numpy, etc.

---

## Installation

```bash
pip install -e .
```

**Requirements**: Python ≥ 3.9, numpy ≥ 1.22.  
**Examples** additionally need scipy ≥ 1.9.

---

## Dimension conventions

Let **n** be the number of scalar inputs (the size of the `wrt` argument after flattening) and **m** the number of scalar outputs returned by f.

| n | m | Correct function | Returns |
|---|---|---|---|
| 1 | 1 | `derivative(f, x0)` | `float` |
| > 1 | 1 | `gradient(f, x0)` | `ndarray (n,)` |
| ≥ 1 | > 1 | `jacobian(f, x0)` | `ndarray (m, n)` |
| ≥ 1 | ≥ 1* | `directional_derivative(f, x0, v=v)` | `float` or `ndarray (m,)` |

\* except n = m = 1 (use `derivative` instead).

**Jacobian convention**: `J[i, j] = ∂fᵢ/∂xⱼ` — rows index output components, columns index input components.

Each function probes f once to determine (n, m) and raises a descriptive `ValueError` if the wrong function is used for that case.

---

## API

### `derivative(f, x0, *, h=None, return_diagnostics=False, batched=False)`

First derivative of f: ℝ → ℝ. `x0` must be a scalar float.

```python
from csdiff import derivative
import numpy as np

derivative(lambda x: x**3, 2.0)          # → 12.0
dx, info = derivative(np.exp, 1.0, return_diagnostics=True)
# dx ≈ e,  info.n_calls = 1
```

### `gradient(f, *args, wrt=None, h=None, return_diagnostics=False, batched=False)`

Gradient of f: ℝⁿ → ℝ with n > 1.

```python
from csdiff import gradient

gradient(lambda x: np.sum(x**2), np.array([1., 2., 3.]))
# → array([2., 4., 6.])

# Multi-argument function — specify which argument with wrt=
def cost(t, theta, u):
    return np.dot(theta, u) * t

gradient(cost, 2.0, np.array([1., 0.5]), np.array([3., 4.]), wrt="theta")
# → array([6., 8.])
```

### `jacobian(f, *args, wrt=None, h=None, return_diagnostics=False, batched=False)`

Jacobian of f: ℝⁿ → ℝᵐ with m > 1. Also handles f: ℝ → ℝᵐ (returns shape `(m, 1)`).

```python
from csdiff import jacobian

# f: ℝⁿ → ℝᵐ — returns (m, n)
jacobian(lambda x: x**2, np.array([1., 2., 3.]))
# → array([[2., 0., 0.],
#          [0., 4., 0.],
#          [0., 0., 6.]])

# Combined wrt — Jacobian covers both arguments simultaneously
def f(x, u):
    return np.array([x[0]*u[0], x[1]*u[1]])

jacobian(f, np.array([1., 2.]), np.array([3., 4.]), wrt=("x", "u"))
# shape (2, 4): J wrt [x0, x1, u0, u1]
```

### `directional_derivative(f, *args, v, wrt=None, h=None, return_diagnostics=False, batched=False)`

Jacobian-vector product J(x)·v in **one** evaluation, independent of n.

```python
from csdiff import directional_derivative

x0 = np.array([1., 2., 3.])
v  = np.array([1., 0., -1.])
directional_derivative(lambda x: np.sum(x**2), x0, v=v)
# → float,  equals gradient(f, x0) @ v = 2*1 + 0 + 2*3*(-1) = -4.0
```

---

## Batched evaluation

Pass `batched=True` to evaluate derivatives at multiple points simultaneously. The `wrt` argument's first dimension is treated as the batch dimension; all other arguments stay fixed.

```python
import numpy as np
from csdiff import gradient, jacobian

# 900-point meshgrid in ℝ²
X, Y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
XY = np.column_stack([X.ravel(), Y.ravel()])   # (900, 2)

G = gradient(lambda xy: np.sum(xy**2, axis=-1), XY, batched=True)   # (900, 2)

# f(t, x, u, p) — differentiate wrt x at a batch of states (p fixed)
J_all = jacobian(f, t0, X_batch, u0, p0, wrt="x", batched=True)   # (n_batch, m, n)
```

### Output shapes with `batched=True`

| Function | Single-point output | Batched output |
|---|---|---|
| `derivative(f, x0_batch, batched=True)` | `float` | `(n_batch,)` |
| `gradient(f, X_batch, batched=True)` | `(n,)` | `(n_batch, n)` |
| `jacobian(f, X_batch, batched=True)` | `(m, n)` | `(n_batch, m, n)` |
| `directional_derivative(..., batched=True)` | `float` or `(m,)` | `(n_batch,)` or `(n_batch, m)` |

### Vectorized-first strategy

If f supports batched input natively (e.g. written with `np.sum(..., axis=-1)`), csdiff detects this automatically and uses a single call across all n columns, regardless of `n_batch`. If not, it falls back to a serial loop per batch point — the results are identical either way.

| Function | Vectorized cost | Serial cost |
|---|---|---|
| `derivative` | 1 eval | n_batch evals |
| `gradient`, `jacobian` | n evals (n = wrt size) | n_batch × n evals |
| `directional_derivative` | 1 eval | n_batch evals |

### `wrt=` rules with `batched=True`

- `wrt` must identify a **single** argument — tuple `wrt` is rejected with `ValueError` (ambiguous which sub-argument carries the batch rows).
- The `wrt` argument must be at least 2-D with shape `(n_batch, n)`.
- `wrt=` is still required when `len(args) > 1`.
- For `directional_derivative`, `v` may be `(n,)` (same direction for all points) or `(n_batch, n)` (per-point direction).

To compute derivatives wrt two arguments at paired points `(X[i], P[i])`, use a serial loop with combined `wrt`:

```python
G_xp = np.stack([
    gradient(f, t0, X[i], u0, P[i], wrt=("x", "p"))
    for i in range(len(X))
])
```

---

## Multi-argument calling convention

When f is declared as `f(t, x, u, p)` rather than `f(X)`, pass all arguments positionally and use `wrt=` to specify which one to differentiate:

```python
gradient(f, t0, x0, u0, p0, wrt="x")       # by parameter name
gradient(f, t0, x0, u0, p0, wrt=1)          # by 0-based position
jacobian(f, t0, x0, u0, p0, wrt=("x", "u")) # combined — shape (m, nx+nu)
```

`wrt` is required only when `len(args) > 1`.

---

## Diagnostics

All functions accept `return_diagnostics=True` to return `(result, DiffInfo)`:

```python
from csdiff import gradient, DiffInfo

grad, info = gradient(f, x0, return_diagnostics=True)
info.n_calls  # number of f evaluations (not counting the probe)
info.elapsed  # total wall-clock time inside f (seconds)
info.h        # complex step size used
```

---

## Exceptions and warnings

| Type | Meaning |
|---|---|
| `ComplexStepError` | f raised an exception when called with complex input |
| `NonanalyticWarning` | Im(f(x+ih)) ≈ 0 — f may be discarding the imaginary part |
| `StepSizeWarning` | h outside recommended range [ε², √ε] |
| `ValueError` | wrong function called for the (n, m) case |

---

## Examples

See [examples/](examples/) for four worked examples with simple and engineering cases:

| File | Function | Simple | Engineering |
|---|---|---|---|
| [ex_derivative.py](examples/ex_derivative.py) | `derivative` | Critical point of a polynomial | Newton's method for van der Waals equation of state |
| [ex_gradient.py](examples/ex_gradient.py) | `gradient` | Gradient of sin(x₁)cos(x₂)+exp(x₃) | PID controller tuning via gradient descent |
| [ex_jacobian.py](examples/ex_jacobian.py) | `jacobian` | 2-link robot arm Jacobian | Shooting method for a ballistic trajectory BVP |
| [ex_directional_derivative.py](examples/ex_directional_derivative.py) | `directional_derivative` | Directional derivative of a scalar field | Tangent linear model for the Lorenz system |
| [ex_batched.py](examples/ex_batched.py) | all four + `wrt=` | — | Sigmoid grid, potential field, van der Pol linearisation, pendulum JVP |

---

## Internal modules

Importable but not in `__all__`:

| Module | Contents |
|---|---|
| `csdiff.core` | `probe()`, `gradient_serial()`, `jacobian_serial()`, `directional_derivative()` |
| `csdiff.template` | `CallTemplate` — argument binding and flat ↔ structured conversion |
| `csdiff.step` | `default_step()`, `validate_step()` |
| `csdiff.info` | `DiffInfo` frozen dataclass |
| `csdiff.exceptions` | `ComplexStepError`, `NonanalyticWarning`, `StepSizeWarning` |
