"""
Example — batched=True
======================
All four differentiation functions accept a ``batched=True`` keyword that
evaluates at a whole batch of points without a Python loop.

When the underlying function f natively supports a leading batch dimension
(e.g. f(X) where X has shape (n_batch, n) returns (n_batch, ...)), csdiff
reuses the same internal perturbation loop as for a single point — so the
total number of f-evaluations is identical to the single-point case regardless
of n_batch.  If f does not support batched input, the library falls back
transparently to a serial loop.

Output shapes
─────────────
  derivative(f, x_batch, batched=True)                 (n_batch,)
  gradient(f, X_batch, batched=True)                   (n_batch, n)
  jacobian(f, X_batch, batched=True)                   (n_batch, m, n)
  directional_derivative(f, X_batch, v=V, batched=True)
      scalar f                                         (n_batch,)
      vector f                                         (n_batch, m)
"""
import numpy as np
from csdiff import derivative, gradient, jacobian, directional_derivative


# ─────────────────────────────────────────────────────────────────────────────
# 1.  derivative — sweep of evaluation points
# ─────────────────────────────────────────────────────────────────────────────
# Evaluating the derivative of a function over a dense grid is the most
# natural use of batched=True for derivative().
#
# Application: sensitivity of a sigmoid activation function σ(x) = 1/(1+e^{-x})
# over the input range [-6, 6].  The derivative σ'(x) = σ(x)(1 - σ(x)) has a
# maximum at x=0 and decays symmetrically.
#
# numpy's exp and the arithmetic below are naturally vectorised, so the
# vectorised path fires: 1 f-evaluation for all n_batch points.

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

x_grid = np.linspace(-6.0, 6.0, 200)

sig_prime, info = derivative(sigmoid, x_grid, batched=True, return_diagnostics=True)
analytical      = sigmoid(x_grid) * (1.0 - sigmoid(x_grid))  # σ(1 - σ)

print("=== derivative batched — sigmoid sensitivity sweep ===")
print(f"  x_grid: {len(x_grid)} points in [-6, 6]")
print(f"  output shape : {sig_prime.shape}")
print(f"  n_calls      : {info.n_calls}  "
      f"({'vectorised' if info.n_calls == 1 else 'serial'})")
print(f"  max |error|  : {np.max(np.abs(sig_prime - analytical)):.2e}")
print(f"  peak σ'(x)   : {sig_prime.max():.6f} at x = "
      f"{x_grid[sig_prime.argmax()]:.3f}  (expected 0.25 at x=0)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  gradient — scalar field on a 2-D meshgrid
# ─────────────────────────────────────────────────────────────────────────────
# The canonical batched gradient use case: evaluating ∇V at every node of a
# grid produced by np.meshgrid.
#
# Application: gravitational potential of a point mass at the origin,
# regularised to avoid the singularity:
#
#   V(x, y) = -G·M / sqrt(x² + y²  + ε²)
#
# Analytical gradient: ∂V/∂x = G·M·x / (x²+y²+ε²)^{3/2},  similarly ∂V/∂y.
#
# np.sum and np.sqrt support batched (n_batch, 2) input, so the vectorised
# path fires: 2 f-evaluations (= n = 2 input dimensions) for all n_batch points.

GM  = 4.0 * np.pi**2   # G·M in units where Earth's orbit is 1 AU, 1 yr
eps = 0.1              # softening to keep the gradient finite at origin

def potential(xy):
    x, y = xy[..., 0], xy[..., 1]
    return -GM / np.sqrt(x**2 + y**2 + eps**2)

# 30 × 30 grid over [-3, 3] AU
x_vals = np.linspace(-3.0, 3.0, 30)
y_vals = np.linspace(-3.0, 3.0, 30)
X, Y = np.meshgrid(x_vals, y_vals)
XY_batch = np.column_stack([X.ravel(), Y.ravel()])   # (900, 2)

G_batch, info = gradient(potential, XY_batch, batched=True,
                         return_diagnostics=True)

# Reshape for visualisation (or further computation)
Gx = G_batch[:, 0].reshape(X.shape)   # ∂V/∂x at each grid node
Gy = G_batch[:, 1].reshape(X.shape)   # ∂V/∂y at each grid node

# Analytical gradient for comparison
r3 = (X**2 + Y**2 + eps**2) ** 1.5
Gx_ref = GM * X / r3
Gy_ref = GM * Y / r3

print("=== gradient batched — gravitational potential on 2-D grid ===")
print(f"  Grid: {X.shape[0]}×{X.shape[1]} = {XY_batch.shape[0]} points")
print(f"  output shape : {G_batch.shape}")
print(f"  n_calls      : {info.n_calls}  "
      f"({'vectorised — n=2 total regardless of n_batch' if info.n_calls == 2 else 'serial'})")
print(f"  max |∂V/∂x error| : {np.max(np.abs(Gx - Gx_ref)):.2e}")
print(f"  max |∂V/∂y error| : {np.max(np.abs(Gy - Gy_ref)):.2e}")
idx_10 = np.argmin(np.sum((XY_batch - [1.0, 0.0])**2, axis=1))
xy_10  = XY_batch[idx_10]
print(f"  Nearest grid point to (1, 0): ({xy_10[0]:.3f}, {xy_10[1]:.3f})")
print(f"  Gradient there: [{G_batch[idx_10, 0]:.4f}, {G_batch[idx_10, 1]:.4f}]  "
      f"(same sign as position — force points away from origin ✓)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  jacobian — linearisation at multiple operating points
# ─────────────────────────────────────────────────────────────────────────────
# In gain-scheduling control, a nonlinear plant is linearised at a family of
# operating points to produce a family of LTI models.  Each Jacobian is the
# state-space A-matrix at one operating point.
#
# Application: van der Pol oscillator (μ=1)
#
#   ẋ₁ = x₂
#   ẋ₂ = μ(1 - x₁²)x₂ - x₁
#
# Analytical Jacobian:
#
#   A(x) = [[0,          1        ],
#            [-2μx₁x₂-1, μ(1-x₁²)]]
#
# We linearise at 6 points along the limit cycle (x₁ ≈ 2cos(t), x₂ ≈ -2sin(t)).
# The vectorised path fires: 2 f-evaluations total for all 6 operating points.

mu = 1.0

def van_der_pol(x):
    """ẋ = f(x) for the van der Pol oscillator (2-state, vectorised)."""
    x1, x2 = x[..., 0], x[..., 1]
    return np.stack([x2,
                     mu * (1.0 - x1**2) * x2 - x1], axis=-1)

# Sample 6 states near the limit cycle
t_ops = np.linspace(0, 2 * np.pi, 6, endpoint=False)
X_ops = np.column_stack([2.0 * np.cos(t_ops),     # x₁ ≈ 2 cos(t)
                         -2.0 * np.sin(t_ops)])    # x₂ ≈ -2 sin(t)

J_all, info = jacobian(van_der_pol, X_ops, batched=True,
                       return_diagnostics=True)

# Analytical A-matrices
def A_analytical(x):
    x1, x2 = x[0], x[1]
    return np.array([[0,             1                ],
                     [-2*mu*x1*x2-1, mu*(1 - x1**2)  ]])

A_ref = np.stack([A_analytical(X_ops[i]) for i in range(len(X_ops))])

print("=== jacobian batched — van der Pol gain scheduling (6 operating points) ===")
print(f"  Operating points shape : {X_ops.shape}")
print(f"  Output shape           : {J_all.shape}  (n_batch, m, n)")
print(f"  n_calls                : {info.n_calls}  "
      f"({'vectorised — n=2 total for all 6 points' if info.n_calls == 2 else 'serial'})")
print(f"  max |J error| (all)    : {np.max(np.abs(J_all - A_ref)):.2e}")
print()
print(f"  {'t (rad)':>8}  {'A[0,0]':>7}  {'A[0,1]':>7}  "
      f"{'A[1,0]':>10}  {'A[1,1]':>10}")
print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*10}")
for i, t in enumerate(t_ops):
    A = J_all[i]
    print(f"  {t:>8.4f}  {A[0,0]:>7.3f}  {A[0,1]:>7.3f}  "
          f"{A[1,0]:>10.4f}  {A[1,1]:>10.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  directional_derivative — rate of change along a trajectory
# ─────────────────────────────────────────────────────────────────────────────
# The directional derivative ∇V(x) · v gives the instantaneous rate of change
# of a scalar field V in direction v.  When v = ẋ = f(x) (the system dynamics),
# this is the Lie derivative V̇ = dV/dt, which is used in stability analysis.
#
# Application: total mechanical energy of a damped pendulum monitored along a
# simulated trajectory.  At each sampled state, we compute V̇ in one f-call.
#
#   V(x) = ½x₂² + (g/L)(1 - cos x₁)
#   ẋ₁   = x₂
#   ẋ₂   = -c·x₂ - (g/L)·sin x₁
#
# Analytical: V̇ = -c·x₂²  (energy dissipation rate)
#
# The vectorised path fires: 1 f-evaluation for all n_batch trajectory points.

g_over_L, c = 9.81, 0.5   # pendulum parameters

def pendulum_energy(x):
    """Total energy V = ½ẋ² + (g/L)(1-cosθ), vectorised."""
    return 0.5 * x[..., 1]**2 + g_over_L * (1.0 - np.cos(x[..., 0]))

def pendulum_dynamics(x):
    """Damped pendulum ẋ = f(x), vectorised."""
    return np.stack([x[..., 1],
                     -c * x[..., 1] - g_over_L * np.sin(x[..., 0])], axis=-1)

# Sample 8 states along a decaying trajectory (large initial angle)
t_traj = np.linspace(0, 6.0, 8)
theta_traj = 1.8 * np.exp(-c / 2 * t_traj) * np.cos(np.sqrt(g_over_L) * t_traj)
omega_traj = np.gradient(theta_traj, t_traj)   # approximate ẋ₁
X_traj = np.column_stack([theta_traj, omega_traj])   # (8, 2)

# Velocity at each state (per-point direction vectors, shape (8, 2))
V_traj = pendulum_dynamics(X_traj)

Vdot_batch, info = directional_derivative(
    pendulum_energy, X_traj, v=V_traj, batched=True, return_diagnostics=True
)
Vdot_exact = -c * X_traj[:, 1]**2   # analytical: V̇ = -c·ẋ²

print("=== directional_derivative batched — energy rate along pendulum trajectory ===")
print(f"  Trajectory points shape : {X_traj.shape}")
print(f"  Direction v shape       : {V_traj.shape}  (per-point velocities)")
print(f"  Output shape            : {Vdot_batch.shape}")
print(f"  n_calls                 : {info.n_calls}  "
      f"({'vectorised — 1 evaluation for all 8 points' if info.n_calls == 1 else 'serial'})")
print(f"  max |V̇ error|           : {np.max(np.abs(Vdot_batch - Vdot_exact)):.2e}")
print()
print(f"  {'t':>5}  {'θ':>7}  {'ẋ':>8}  {'V̇ (csdiff)':>13}  {'V̇ (exact)':>12}")
print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*13}  {'─'*12}")
for i in range(len(t_traj)):
    print(f"  {t_traj[i]:>5.2f}  {X_traj[i,0]:>7.4f}  {X_traj[i,1]:>8.4f}  "
          f"{Vdot_batch[i]:>13.8f}  {Vdot_exact[i]:>12.6f}")
print()
print("  V̇ = -c·ẋ² ≤ 0 everywhere — energy is dissipated along the trajectory.")
print("  Cost: 1 f-evaluation for all 8 states (vs 2×8=16 for batched gradient).")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  wrt= with batched=True — gradient and jacobian
# ─────────────────────────────────────────────────────────────────────────────
# batched=True always batches the wrt argument.  Three rules apply on top of
# the normal wrt= contract:
#
#   (a) wrt must be a single argument (str or int), not a tuple — a tuple wrt
#       is ambiguous: the library cannot tell which sub-argument holds the rows.
#   (b) the wrt argument must be at least 2-D (n_batch, n); a 1-D array has no
#       batch dimension.
#   (c) wrt= is required whenever len(args) > 1, same as for batched=False.
#
# Both functions below are declared as f(t, x, u, p) and support a leading
# batch dimension on x and p via numpy's ... indexing.
#
# Scalar output (gradient):  J(t, x, u, p) = p₀·x₀² + p₁·x₁² + u·x₀
#   ∂J/∂x = [2p₀x₀ + u,  2p₁x₁]
#   ∂J/∂p = [x₀²,  x₁²]
#
# Vector output (jacobian):  ẋ(t, x, u, p) = [p₀·x₀ + u₀,  p₁·x₁² + u₁]
#   ∂ẋ/∂x = diag([p₀,  2p₁x₁])   (depends on x)
#   ∂ẋ/∂p = diag([x₀,  x₁²  ])   (depends on x, not p)

def quad_cost(t, x, u, p):
    """J = p₀·x₀² + p₁·x₁² + u·x₀   (scalar output, vectorised)."""
    return p[..., 0] * x[..., 0]**2 + p[..., 1] * x[..., 1]**2 + u * x[..., 0]

def quad_dynamics(t, x, u, p):
    """ẋ = [p₀·x₀ + u₀,  p₁·x₁² + u₁]   (vector output, vectorised)."""
    return np.stack([p[..., 0] * x[..., 0] + u[0],
                     p[..., 1] * x[..., 1]**2 + u[1]], axis=-1)

t0_w  = 1.0
u0_w  = np.array([0.5, -0.3])   # fixed control (1-D)
x0_w  = np.array([1.5, -0.8])   # fixed reference state (1-D)
p0_w  = np.array([2.0, 3.0])    # fixed parameter vector (1-D)

rng_w   = np.random.default_rng(0)
X_wrt   = rng_w.standard_normal((5, 2))    # (5, 2) — batch in x
P_wrt   = rng_w.uniform(0.5, 4.0, (5, 2)) # (5, 2) — batch in p

print("=== wrt= with batched=True ===")
print("  Scalar f: J(t, x, u, p) = p₀·x₀² + p₁·x₁² + u·x₀")
print("  Vector f: ẋ(t, x, u, p) = [p₀·x₀ + u₀,  p₁·x₁² + u₁]")
print()

# ── gradient — correct cases ──────────────────────────────────────────────────
print("  gradient (scalar output f: ℝⁿ→ℝ)")
print()

# [OK] wrt="x": X_wrt is (5,2); p, u are fixed 1-D → output (5, 2)
G_x = gradient(quad_cost, t0_w, X_wrt, u0_w[0], p0_w, wrt="x", batched=True)
G_x_ref = np.column_stack([2*p0_w[0]*X_wrt[:,0] + u0_w[0],
                            2*p0_w[1]*X_wrt[:,1]])
print(f"  [OK]  wrt='x',  X_wrt {X_wrt.shape} → gradient {G_x.shape},  "
      f"max err {np.max(np.abs(G_x - G_x_ref)):.2e}")

# [OK] wrt="p": P_wrt is (5,2); x, u are fixed 1-D → output (5, 2)
G_p = gradient(quad_cost, t0_w, x0_w, u0_w[0], P_wrt, wrt="p", batched=True)
G_p_ref = np.tile([x0_w[0]**2, x0_w[1]**2], (len(P_wrt), 1))  # ∂J/∂p = [x₀², x₁²]
print(f"  [OK]  wrt='p',  P_wrt {P_wrt.shape} → gradient {G_p.shape},  "
      f"max err {np.max(np.abs(G_p - G_p_ref)):.2e}")

print()

# [Err] tuple wrt — cannot determine which sub-argument carries the batch rows
try:
    gradient(quad_cost, t0_w, X_wrt, u0_w[0], P_wrt, wrt=("x", "p"), batched=True)
except ValueError as e:
    print(f"  [Err] wrt=('x','p')   → ValueError: {str(e).splitlines()[0]}")

# [Err] wrt arg is 1-D — batched=True requires the wrt arg to be at least 2-D
try:
    gradient(quad_cost, t0_w, x0_w, u0_w[0], p0_w, wrt="x", batched=True)
    #                         ^^^^— shape (2,), needs (n_batch, 2)
except ValueError as e:
    print(f"  [Err] x is 1-D        → ValueError: {str(e).splitlines()[0]}")

# [Err] wrt omitted — required whenever f has more than one argument
try:
    gradient(quad_cost, t0_w, X_wrt, u0_w[0], p0_w, batched=True)
except TypeError as e:
    print(f"  [Err] wrt omitted     → TypeError:  {str(e).splitlines()[0]}")

print()

# ── jacobian — correct cases ──────────────────────────────────────────────────
print("  jacobian (vector output f: ℝⁿ→ℝᵐ)")
print()

# [OK] wrt="x": X_wrt is (5,2); p, u are fixed → output (5, m, n_x) = (5, 2, 2)
# The Jacobian at each batch point differs because ∂ẋ₁/∂x₁ = 2p₁x₁ depends on x.
J_x = jacobian(quad_dynamics, t0_w, X_wrt, u0_w, p0_w, wrt="x", batched=True)
J_x_ref = np.array([np.diag([p0_w[0], 2*p0_w[1]*X_wrt[i, 1]])
                    for i in range(len(X_wrt))])
print(f"  [OK]  wrt='x',  X_wrt {X_wrt.shape} → jacobian {J_x.shape},  "
      f"max err {np.max(np.abs(J_x - J_x_ref)):.2e}")

# [OK] wrt="p": P_wrt is (5,2); x is fixed → output (5, m, n_p) = (5, 2, 2)
# ∂ẋ/∂p = diag([x₀, x₁²]) — constant across P_wrt since x is fixed.
J_p = jacobian(quad_dynamics, t0_w, x0_w, u0_w, P_wrt, wrt="p", batched=True)
J_p_ref = np.tile(np.diag([x0_w[0], x0_w[1]**2]), (len(P_wrt), 1, 1))
print(f"  [OK]  wrt='p',  P_wrt {P_wrt.shape} → jacobian {J_p.shape},  "
      f"max err {np.max(np.abs(J_p - J_p_ref)):.2e}")

print()

# [Err] tuple wrt — same restriction as for gradient
try:
    jacobian(quad_dynamics, t0_w, X_wrt, u0_w, P_wrt, wrt=("x", "p"), batched=True)
except ValueError as e:
    print(f"  [Err] wrt=('x','p')   → ValueError: {str(e).splitlines()[0]}")

# [Err] wrt arg is 1-D
try:
    jacobian(quad_dynamics, t0_w, x0_w, u0_w, p0_w, wrt="x", batched=True)
except ValueError as e:
    print(f"  [Err] x is 1-D        → ValueError: {str(e).splitlines()[0]}")

# [Err] wrt omitted
try:
    jacobian(quad_dynamics, t0_w, X_wrt, u0_w, p0_w, batched=True)
except TypeError as e:
    print(f"  [Err] wrt omitted     → TypeError:  {str(e).splitlines()[0]}")

print()

# ── Workaround: ∂J/∂(x,p) at paired (x,p) points — combined wrt + batched ───
# batched=True sweeps one argument while the other stays fixed.  To compute the
# full gradient ∂J/∂(x,p) at n paired (X_wrt[i], P_wrt[i]) points, use a serial
# loop with combined wrt (one call per point, n_x + n_p evaluations each):
G_xp = np.stack([
    gradient(quad_cost, t0_w, X_wrt[i], u0_w[0], P_wrt[i], wrt=("x", "p"))
    for i in range(len(X_wrt))
])
print("  [Workaround] ∂J/∂(x,p) at paired points — serial loop, wrt=('x','p'):")
print("    gradient(..., X[i], P[i], wrt=('x','p')) for i in range(n_batch)")
print(f"    output shape : {G_xp.shape}   (n_batch, n_x + n_p)")
