"""
Example — jacobian()
====================
Use case: f: ℝⁿ → ℝᵐ with m > 1  (vector input, vector output).

jacobian(f, x0) computes the full Jacobian matrix J ∈ ℝ^{m×n} using n
complex evaluations.  Convention: J[i, j] = ∂fᵢ/∂xⱼ.

It is the right tool when you need the linear relationship between small
changes in a parameter vector and small changes in a vector of outputs —
for example, the robot Jacobian, the sensitivity matrix of an ODE solver,
or the residual Jacobian of a nonlinear system of equations.
"""
import numpy as np
from scipy.integrate import solve_ivp
from csdiff import jacobian


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Simple case — 2-link planar robot arm (forward kinematics Jacobian)
# ─────────────────────────────────────────────────────────────────────────────
# End-effector position of a 2-link arm:
#
#   x_ee = L1·cos(θ1) + L2·cos(θ1 + θ2)
#   y_ee = L1·sin(θ1) + L2·sin(θ1 + θ2)
#
# The Jacobian J = ∂(x_ee, y_ee)/∂(θ1, θ2) maps joint velocities to
# end-effector velocity:  [ẋ_ee, ẏ_ee]^T = J · [θ̇1, θ̇2]^T

L1, L2 = 0.5, 0.3   # link lengths (m)

def forward_kinematics(theta):
    """End-effector (x, y) for joint angles theta = [θ1, θ2]."""
    t1, t2 = theta[0], theta[1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.array([x, y])

theta0 = np.array([np.pi / 4, np.pi / 6])  # 45° and 30°
J_cs = jacobian(forward_kinematics, theta0)

# Analytical Jacobian for comparison
t1, t2 = theta0
J_analytical = np.array([
    [-L1*np.sin(t1) - L2*np.sin(t1+t2),  -L2*np.sin(t1+t2)],
    [ L1*np.cos(t1) + L2*np.cos(t1+t2),   L2*np.cos(t1+t2)],
])

print("=== Simple: 2-link robot arm Jacobian ===")
print(f"  Configuration: θ1={np.degrees(t1):.1f}°, θ2={np.degrees(t2):.1f}°")
print(f"  J (complex step):\n{J_cs}")
print(f"  J (analytical):\n{J_analytical}")
print(f"  max |error|: {np.max(np.abs(J_cs - J_analytical)):.2e}")
print()

# Differential velocity: joint velocity → end-effector velocity
theta_dot = np.array([0.1, -0.2])   # rad/s
ee_dot = J_cs @ theta_dot
print(f"  Joint velocity: {theta_dot} rad/s")
print(f"  End-effector velocity: {ee_dot} m/s  (via J·θ̇)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Engineering — sensitivity matrix of an ODE endpoint (shooting method)
# ─────────────────────────────────────────────────────────────────────────────
# Problem: solve a two-point boundary value problem (BVP) for a projectile
# with quadratic drag:
#
#   ẍ = 0                ÿ = -g - k·ṡ·ẏ   (k = drag coefficient, s = speed)
#
# with fixed launch position (0, 0) and fixed target (X_T, Y_T = 0).
# We seek the initial velocity v0 = [vx0, vy0] such that the projectile
# lands at the target at time T.
#
# The shooting method treats this as a root-finding problem:
#
#   r(v0) = [x(T) - X_T,  y(T) - Y_T]  →  0
#
# Each Newton step needs J = ∂r/∂v0 — the sensitivity of the terminal
# position to the initial velocity.  jacobian() computes this via two
# complex ODE integrations (one per column of J).

g = 9.81   # m/s²
k = 0.01   # drag coefficient (1/m)
T = 3.5    # flight time (s)
X_T = 60.0 # target downrange distance (m)

def simulate(v0):
    """
    Integrate the projectile ODE from t=0 to t=T and return terminal position.
    v0 = [vx0, vy0] — initial velocity components.
    """
    def odes(t, state):
        x, y, vx, vy = state
        speed = np.sqrt(vx**2 + vy**2)
        ax = -k * speed * vx
        ay = -g - k * speed * vy
        return [vx, vy, ax, ay]

    # Initial condition: starts at origin with velocity v0
    y0_state = [0.0, 0.0, v0[0], v0[1]]
    sol = solve_ivp(odes, [0, T], y0_state, max_step=0.01,
                    rtol=1e-8, atol=1e-10)
    return np.array([sol.y[0, -1], sol.y[1, -1]])  # (x_T, y_T)

def residual(v0):
    """BVP residual: terminal position minus target."""
    pos = simulate(v0)
    return pos - np.array([X_T, 0.0])

# Initial guess for launch velocity
v0_guess = np.array([X_T / T, 0.3 * g * T / 2])  # rough estimate

print("=== Engineering: projectile shooting method with drag ===")
print(f"  Target: X={X_T} m, Y=0 at T={T} s  (drag k={k})")
print(f"  Initial guess: vx0={v0_guess[0]:.2f} m/s, vy0={v0_guess[1]:.2f} m/s")

v0 = v0_guess.copy()
for k_iter in range(6):
    r = residual(v0)
    J = jacobian(residual, v0)    # ← 2 complex ODE integrations
    dv = np.linalg.solve(J, -r)  # Newton update
    v0 = v0 + dv
    print(f"  iter {k_iter+1}: |r| = {np.linalg.norm(r):.4e}  "
          f"vx0={v0[0]:.4f}  vy0={v0[1]:.4f}")
    if np.linalg.norm(r) < 1e-9:
        break

pos_final = simulate(v0)
print(f"  Converged: terminal position = ({pos_final[0]:.6f}, {pos_final[1]:.6f}) m")
print(f"  Launch: vx0={v0[0]:.4f} m/s,  vy0={v0[1]:.4f} m/s")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-argument functions with wrt= — linearising nonlinear dynamics
# ─────────────────────────────────────────────────────────────────────────────
# A dynamical system declared as f(t, x, u, p).  Planar drone (4 states):
#
#   x = [θ, ω, vy, vz],   u = [F₁, F₂],   p = [m, I, L, g]
#
#   θ̇ = ω,    ω̇ = (L/I) · (F₁ - F₂)
#   v̇y = -(F₁+F₂)/m · sin(θ),    v̇z = (F₁+F₂)/m · cos(θ) - g
#
# At hover (θ=0, ω=vy=vz=0): F₁=F₂=mg/2.  Linearising around this point:
#
#   A = jacobian(f, t, x_hover, u_hover, p, wrt="x")   →  ∂ẋ/∂x  (4×4)
#   B = jacobian(f, t, x_hover, u_hover, p, wrt="u")   →  ∂ẋ/∂u  (4×2)
#
# A and B are the state-space matrices required for LQR or pole placement.
# No hand-differentiation of trigonometric expressions needed.

def drone_dynamics(t, x, u, p):
    """Planar drone (4-state): ẋ = f(t, x, u, p)."""
    theta, omega, vy, vz = x[0], x[1], x[2], x[3]
    F1, F2 = u[0], u[1]
    m, I, L, grav = p[0], p[1], p[2], p[3]
    F_total = F1 + F2
    return np.array([
        omega,
        (L / I) * (F1 - F2),
        -(F_total / m) * np.sin(theta),
         (F_total / m) * np.cos(theta) - grav,
    ])

m_d, I_d, L_d = 1.5, 0.02, 0.25   # mass (kg), inertia (kg·m²), arm (m)
p_drone = np.array([m_d, I_d, L_d, 9.81])
x_hover = np.zeros(4)
u_hover = np.array([m_d * 9.81 / 2, m_d * 9.81 / 2])   # balanced rotors at hover

print("=== Multi-arg (wrt='x', wrt='u'): drone hover linearisation for LQR ===")
print(f"  Hover: F₁=F₂={u_hover[0]:.2f} N  (m={m_d} kg, I={I_d} kg·m², L={L_d} m)")

A = jacobian(drone_dynamics, 0.0, x_hover, u_hover, p_drone, wrt="x")
B = jacobian(drone_dynamics, 0.0, x_hover, u_hover, p_drone, wrt="u")

print(f"  A = ∂f/∂x  (shape {A.shape}):")
print(np.array2string(A, precision=4, suppress_small=True))
print(f"  B = ∂f/∂u  (shape {B.shape}):")
print(np.array2string(B, precision=4, suppress_small=True))

# Analytical check at hover (θ=0): A[0,1]=1, A[2,0]=-g; B[1,:]=[L/I,-L/I], B[3,:]=[1/m,1/m]
A_ref = np.zeros((4, 4))
A_ref[0, 1] = 1.0                       # θ̇ = ω
A_ref[2, 0] = -9.81                     # v̇y ≈ -g·θ  at hover
B_ref = np.zeros((4, 2))
B_ref[1, :] = [L_d / I_d, -L_d / I_d]  # ω̇ from differential thrust
B_ref[3, :] = 1.0 / m_d                 # v̇z from total thrust

print(f"  Max |A error| vs analytical: {np.max(np.abs(A - A_ref)):.2e}")
print(f"  Max |B error| vs analytical: {np.max(np.abs(B - B_ref)):.2e}")
print("  → Feed A, B into scipy.linalg.solve_continuous_are() for LQR gains K.")
