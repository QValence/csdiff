"""
Example — directional_derivative()
====================================
Use case: compute J(x)·v in a **single** function evaluation.

Given direction v ∈ ℝⁿ:

  directional_derivative(f, x0, v=v)  ≈  J(x0) @ v

where J = ∂f/∂x is the Jacobian (or ∇f for scalar f).  Cost: 1 evaluation,
independent of n.  Compare with jacobian(), which costs n evaluations to
build the full J and then multiplies — wasteful when only J·v is needed.

The formula is:  Im(f(x + ih·v)) / h = J(x)·v  (exact up to O(h²))

When f: ℝⁿ → ℝ  (scalar),  directional_derivative returns a float  (∇f·v).
When f: ℝⁿ → ℝᵐ (vector),  it returns an ndarray shape (m,)  (J·v).
"""
import numpy as np
from scipy.integrate import solve_ivp
from csdiff import directional_derivative, gradient, jacobian


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Simple case — directional derivative of a scalar field
# ─────────────────────────────────────────────────────────────────────────────
# f(x) = x₁²·x₂ + sin(x₃)   at   x₀ = (1, 2, π/4)
# Analytical ∇f = [2x₁x₂,  x₁²,  cos(x₃)]

f_simple = lambda x: x[0]**2 * x[1] + np.sin(x[2])
x0 = np.array([1.0, 2.0, np.pi / 4])
v  = np.array([1.0, -1.0, 2.0])          # arbitrary direction

dd = directional_derivative(f_simple, x0, v=v)
grad_f = gradient(f_simple, x0)
expected = float(grad_f @ v)

print("=== Simple: directional derivative of x₁²x₂ + sin(x₃) ===")
print(f"  x₀ = {x0}")
print(f"  v  = {v}")
print(f"  directional_derivative = {dd:.10f}   (1 evaluation)")
print(f"  gradient(f) @ v        = {expected:.10f}   ({len(x0)} evaluations)")
print(f"  |difference|           = {abs(dd - expected):.2e}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Engineering — linearised state propagation (tangent linear model)
# ─────────────────────────────────────────────────────────────────────────────
# Problem: sensitivity analysis for a nonlinear dynamic system.
#
# Consider the Lorenz system:
#
#   ẋ = σ(y - x)
#   ẏ = x(ρ - z) - y
#   ż = xy - βz
#
# f(x0) maps the initial state x0 ∈ ℝ³ to the final state x(T).
# The full Jacobian ∂f/∂x0 (the state transition matrix Φ) costs 3
# evaluations with jacobian().
#
# When we only need the effect of one specific perturbation direction v
# (e.g., "how does the trajectory change if we perturb the initial x-component
# with amplitude ε?"), directional_derivative gives the answer in 1 evaluation:
#
#   δx(T) ≈ J·v · ε   where   J·v = directional_derivative(f, x0, v=v)
#
# This is the Jacobian-vector product (JVP), fundamental to tangent linear
# models and forward-mode automatic differentiation.

sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0  # classic Lorenz parameters
T_lorenz = 1.0  # short time horizon (Lorenz is chaotic for longer intervals)

def lorenz(x0):
    """Integrate the Lorenz system from t=0 to T and return the final state."""
    def rhs(t, state):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
    sol = solve_ivp(rhs, [0, T_lorenz], x0, max_step=0.001,
                    rtol=1e-10, atol=1e-12)
    return sol.y[:, -1]  # final state

x0_lorenz = np.array([1.0, 0.0, 20.0])  # initial condition near the attractor

# Direction v: unit perturbation in the x-component of the initial condition
v_x = np.array([1.0, 0.0, 0.0])   # "what if x0 increases by ε?"
v_y = np.array([0.0, 1.0, 0.0])   # "what if y0 increases by ε?"

print("=== Engineering: Lorenz system — linearised sensitivity via JVP ===")
print(f"  Parameters: σ={sigma}, ρ={rho}, β={beta:.3f}, T={T_lorenz} s")
print(f"  Initial state: {x0_lorenz}")

# Compute JVPs in 1 evaluation each
jvp_x, info_x = directional_derivative(lorenz, x0_lorenz, v=v_x,
                                        return_diagnostics=True)
jvp_y, info_y = directional_derivative(lorenz, x0_lorenz, v=v_y,
                                        return_diagnostics=True)

print(f"\n  J·v_x (sensitivity to δx₀) = {jvp_x}   ({info_x.n_calls} eval)")
print(f"  J·v_y (sensitivity to δy₀) = {jvp_y}   ({info_y.n_calls} eval)")

# Verify against the first two columns of the full Jacobian (costs 3 evals)
J_full, info_J = jacobian(lorenz, x0_lorenz, return_diagnostics=True)
print(f"\n  Full Jacobian (first 2 columns, {info_J.n_calls} evals):")
print(f"  J[:, 0] = {J_full[:, 0]}")
print(f"  J[:, 1] = {J_full[:, 1]}")
print(f"\n  |JVP error, x-direction| = {np.max(np.abs(jvp_x - J_full[:, 0])):.2e}")
print(f"  |JVP error, y-direction| = {np.max(np.abs(jvp_y - J_full[:, 1])):.2e}")
print()
print("  Takeaway: 1 eval per direction vs 3 evals for the full Jacobian.")
print("  In n=1000 dimensions, JVP saves 999 evaluations per direction.")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-argument functions with wrt= — Lie derivative for stability
# ─────────────────────────────────────────────────────────────────────────────
# A dynamical system f(t, x, u, p) and a scalar energy function V(t, x, u, p).
# The time derivative of V along trajectories of ẋ=f is the Lie derivative:
#
#   V̇  =  ∂V/∂x · ẋ  =  ∇V · f(x)
#      =  directional_derivative(V, t, x, u, p, v=f(t,x,u,p), wrt="x")
#
# This costs 1 evaluation regardless of dimension n.  If V̇ < 0 for all x ≠ 0,
# V is a Lyapunov function and the equilibrium is stable.
#
# Application: stability analysis of a damped nonlinear pendulum.
#
#   ẋ₁ = x₂                           (θ̇)
#   ẋ₂ = −c·x₂ − (g/L)·sin(x₁)       (θ̈ = damping + restoring torque)
#
# Lyapunov candidate — total mechanical energy:
#
#   V(t, x, u, p) = ½·x₂² + (g/L)·(1 − cos(x₁))
#
# Analytical result: V̇ = −c·x₂²  ≤ 0  →  origin is Lyapunov-stable (LaSalle)

def pendulum_energy(t, x, u, p):
    """Total energy: ½θ̇² + (g/L)(1 - cosθ). Scalar."""
    c, gL = p[0], p[1]
    return 0.5 * x[1]**2 + gL * (1 - np.cos(x[0]))

def pendulum_dynamics(t, x, u, p):
    """Damped nonlinear pendulum: ẋ₁=x₂, ẋ₂=-c·x₂-(g/L)·sin(x₁)+u."""
    c, gL = p[0], p[1]
    return np.array([x[1], -c * x[1] - gL * np.sin(x[0]) + u[0]])

c_damp, gL_val = 0.5, 9.81          # damping coefficient, g/L (unit length)
p_pend  = np.array([c_damp, gL_val])
u_pend  = np.array([0.0])            # uncontrolled

print("=== Multi-arg (wrt='x'): Lie derivative for pendulum stability ===")
print(f"  Dynamics: θ̈ = -{c_damp}·θ̇ - {gL_val}·sin(θ)")
print(f"  Lyapunov: V = ½θ̇² + {gL_val}·(1-cosθ),  analytical V̇ = -{c_damp}·θ̇²")
print(f"  {'θ':>6}  {'θ̇':>6}  {'V':>7}  {'V̇ (csdiff)':>13}  {'V̇ (exact)':>12}")
print(f"  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*13}  {'─'*12}")

for x_test in [np.array([0.5,  1.0]),
               np.array([1.5,  0.5]),
               np.array([2.5,  0.0]),
               np.array([0.3,  3.0])]:
    f_val = pendulum_dynamics(0.0, x_test, u_pend, p_pend)
    Vdot  = directional_derivative(pendulum_energy, 0.0, x_test, u_pend, p_pend,
                                   v=f_val, wrt="x")
    V     = pendulum_energy(0.0, x_test, u_pend, p_pend)
    exact = -c_damp * x_test[1]**2
    print(f"  {x_test[0]:>6.2f}  {x_test[1]:>6.2f}  "
          f"{float(V):>7.3f}  {float(Vdot):>13.8f}  {exact:>12.6f}")

print("  V̇ = -c·θ̇² ≤ 0 everywhere — energy can only decrease along trajectories.")
print("  Cost: 1 evaluation per state regardless of the state dimension.")
