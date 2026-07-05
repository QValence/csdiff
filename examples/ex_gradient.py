"""
Example — gradient()
====================
Use case: f: ℝⁿ → ℝ with n > 1  (vector input, scalar output).

gradient(f, x0) computes ∇f(x0) using n complex evaluations.  It is the
right tool when you need the sensitivity of a scalar performance metric to
every component of a parameter vector — for example, in gradient-based
optimisation.

The key requirement is that f must propagate complex-valued inputs.
Standard numpy arithmetic and scipy functions built on numpy (like odeint/RK4
implemented in pure Python) satisfy this requirement automatically.
"""
import numpy as np
from csdiff import gradient


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Simple case — gradient of a smooth scalar field
# ─────────────────────────────────────────────────────────────────────────────
# f(x) = sin(x₁)·cos(x₂) + exp(x₃)  at  x₀ = (π/3, π/4, 0.5)
# Analytical: ∇f = [cos(x₁)cos(x₂),  -sin(x₁)sin(x₂),  exp(x₃)]

f_simple = lambda x: np.sin(x[0]) * np.cos(x[1]) + np.exp(x[2])

x0 = np.array([np.pi / 3, np.pi / 4, 0.5])
grad = gradient(f_simple, x0)
analytical = np.array([
    np.cos(x0[0]) * np.cos(x0[1]),
    -np.sin(x0[0]) * np.sin(x0[1]),
    np.exp(x0[2]),
])

print("=== Simple: gradient of sin(x₁)cos(x₂) + exp(x₃) ===")
print(f"  gradient   = {grad}")
print(f"  analytical = {analytical}")
print(f"  max error  = {np.max(np.abs(grad - analytical)):.2e}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Engineering — PID controller tuning via gradient descent
# ─────────────────────────────────────────────────────────────────────────────
# We want to tune a PID controller for a second-order plant:
#
#   ÿ + 2ζω ẏ + ω²y = ω²u,   ζ=0.5, ω=1 rad/s
#
# The control law is:
#
#   u(t) = Kp·e(t) + Ki·∫e dt + Kd·ė(t),   e(t) = r - y(t)   (r=1, step)
#
# Objective: minimise the integral squared error (ISE):
#
#   J(θ) = ∫₀ᵀ e(t)² dt,   θ = [Kp, Ki, Kd]
#
# J(θ) is black-box: it hides a time integration.  gradient(J, θ) gives the
# steepest-descent direction for gradient-based tuning.
#
# The integration is implemented with a fixed-step RK4 written in pure numpy
# so that complex-valued gains propagate through the arithmetic correctly —
# this is what makes the complex step work.

zeta, omega = 0.5, 1.0
T_sim = 20.0
dt    = 0.02          # RK4 step size


def pid_cost(theta):
    """
    Integrated squared tracking error for PID gains theta = [Kp, Ki, Kd].

    Uses a pure-numpy RK4 integrator so that complex-valued theta propagates
    through the arithmetic without dtype casting.
    State vector: [y, ẏ, ∫e dt]
    """
    Kp, Ki, Kd = theta[0], theta[1], theta[2]

    def rhs(s):
        y, ydot, ie = s[0], s[1], s[2]
        e = 1.0 - y
        u = Kp * e + Ki * ie - Kd * ydot
        return np.array([ydot, omega**2 * (u - y) - 2*zeta*omega*ydot, e])

    N = int(T_sim / dt)
    state = np.zeros(3, dtype=theta.dtype)  # matches real or complex dtype
    ise = 0 * Kp                            # zero with same dtype as theta

    for _ in range(N):
        e = 1.0 - state[0]
        ise += e**2 * dt
        k1 = rhs(state)
        k2 = rhs(state + dt/2 * k1)
        k3 = rhs(state + dt/2 * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return ise


theta0 = np.array([1.0, 0.1, 0.2])

print("=== Engineering: PID tuning for a second-order plant ===")
print(f"  Plant: ÿ + {2*zeta*omega}ẏ + {omega**2}y = {omega**2}u")
print(f"  Initial gains: Kp={theta0[0]}, Ki={theta0[1]}, Kd={theta0[2]}")
J0 = pid_cost(theta0)
print(f"  Initial ISE: {J0:.4f}")

grad, info = gradient(pid_cost, theta0, return_diagnostics=True)
print(f"  gradient(J, θ₀) = {np.real(grad)}   ({info.n_calls} evaluations)")
print(f"  Steepest descent direction: -∇J/‖∇J‖ = {-grad / np.linalg.norm(grad)}")

# A few gradient-descent steps to show improvement
print("\n  Gradient descent (α=0.05):")
theta = theta0.copy()
for step in range(4):
    J_val  = float(np.real(pid_cost(theta)))
    g      = np.real(gradient(pid_cost, theta))
    theta -= 0.05 * g
    print(f"  step {step+1}: ISE = {J_val:.4f}  "
          f"Kp={theta[0]:.3f}, Ki={theta[1]:.3f}, Kd={theta[2]:.3f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-argument functions with wrt= — spring–mass–damper identification
# ─────────────────────────────────────────────────────────────────────────────
# A damped spring declared as f(t, x, u, p) with p = [k, b]:
#
#   ẋ₁ = x₂
#   ẋ₂ = −k·x₁ − b·x₂ + u    (unit mass, no drive: u = 0)
#
# We observe the state at time T and want to recover k and b.  The fitting cost
#
#   J(x₀, x_obs, p) = ‖x_sim(x₀, p, T) − x_obs‖²
#
# is a black-box scalar.  gradient(J, x₀, x_obs, p, wrt="p") gives ∂J/∂p in
# 2 evaluations and can be supplied directly as jac= to scipy.optimize for
# machine-precision gradient-based optimisation with no finite-difference
# overhead.

from scipy.optimize import minimize as _minimize

def spring_dynamics(t, x, u, p):
    """Damped spring: ẋ₁=x₂, ẋ₂=−k·x₁−b·x₂+u (unit mass)."""
    k, b = p[0], p[1]
    return np.array([x[1], -k*x[0] - b*x[1] + u[0]])

def simulate_spring(x0, p, T=3.0, dt=0.02):
    """RK4 integration. Accepts complex p for complex step."""
    u_zero = np.array([0.0])
    state  = np.array(x0, dtype=p.dtype if np.iscomplexobj(p) else float)
    for _ in range(round(T / dt)):
        k1 = spring_dynamics(0, state,          u_zero, p)
        k2 = spring_dynamics(0, state+dt/2*k1,  u_zero, p)
        k3 = spring_dynamics(0, state+dt/2*k2,  u_zero, p)
        k4 = spring_dynamics(0, state+dt*k3,    u_zero, p)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return state

p_true  = np.array([4.0, 0.5])   # k=4 N/m (ω₀=2), b=0.5 Ns/m (ζ≈0.13)
x0_spr  = np.array([2.0, 0.0])   # released from 2 m, at rest
x_obs   = simulate_spring(x0_spr, p_true)   # "measured" endpoint

def spring_cost(x_init, x_observed, p):
    """Squared mismatch between simulated and observed endpoint."""
    diff = simulate_spring(x_init, p) - x_observed
    return diff @ diff   # stays complex when p is complex — keeps imaginary part

p_guess = np.array([2.5, 1.0])
J0      = float(np.real(spring_cost(x0_spr, x_obs, p_guess)))
grad_p, info = gradient(spring_cost, x0_spr, x_obs, p_guess, wrt="p",
                        return_diagnostics=True)

print("=== Multi-arg (wrt='p'): spring–mass–damper parameter identification ===")
print(f"  System: ẍ = −k·x − b·ẋ   (true: k={p_true[0]}, b={p_true[1]})")
print(f"  Guess:  k={p_guess[0]}, b={p_guess[1]}   →   J = {J0:.4f}")
print(f"  ∂J/∂p = {np.real(grad_p).round(4)}   ({info.n_calls} evaluations)")

# Supply gradient() as the Jacobian to L-BFGS-B:
res = _minimize(
    lambda p: float(np.real(spring_cost(x0_spr, x_obs, p))),
    p_guess,
    jac=lambda p: np.real(gradient(spring_cost, x0_spr, x_obs, p, wrt="p")),
    method="L-BFGS-B",
    bounds=[(0.01, 20.0)] * 2,
)
print(f"  L-BFGS-B: k={res.x[0]:.6f}, b={res.x[1]:.6f}   (J = {res.fun:.2e})")
print(f"  True:     k={p_true[0]:.6f}, b={p_true[1]:.6f}")
