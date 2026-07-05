"""
Example — derivative()
======================
Use case: f: ℝ → ℝ  (scalar input, scalar output).

derivative(f, x0) computes f'(x0) with machine precision using a single
complex function evaluation.  It is the right tool when f maps one real
number to one real number and you need the rate of change.
"""
import numpy as np
from csdiff import derivative


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Simple case — critical point of a polynomial
# ─────────────────────────────────────────────────────────────────────────────
# f(x) = x^4 - 4x^2  has local extrema at x = 0, ±√2.
# f'(x) = 4x^3 - 8x = 4x(x^2 - 2),  so f'(√2) = 0.

f = lambda x: x**4 - 4 * x**2

x_star = np.sqrt(2.0)
df = derivative(f, x_star)
analytical = 4 * x_star**3 - 8 * x_star  # 0 at x = √2

print("=== Simple: critical point of x⁴ - 4x² ===")
print(f"  x*         = {x_star:.6f}")
print(f"  f'(x*)     = {df:.3e}   (should be ≈ 0)")
print(f"  analytical = {analytical:.3e}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Engineering — Newton step for a nonlinear equation
# ─────────────────────────────────────────────────────────────────────────────
# Problem: find the compressibility factor Z of a van der Waals gas.
#
# The van der Waals equation of state is:
#
#   (P + a/V²)(V - b) = RT
#
# Rearranged as a cubic in Z = PV/(RT), the residual is:
#
#   r(Z) = Z³ - (1 + B)Z² + AZ - AB   where  A = aP/(RT)², B = bP/(RT)
#
# Newton's method: Z_{k+1} = Z_k - r(Z_k) / r'(Z_k)
# derivative() gives r'(Z_k) without hand-differentiating the cubic.

# CO₂ parameters at T=300 K, P=50 bar
R = 83.14   # cm³·bar/(mol·K)
a = 3.640e6  # bar·cm⁶/mol²
b = 42.7     # cm³/mol
T = 300.0
P = 50.0

A = a * P / (R * T)**2
B = b * P / (R * T)

def residual(Z):
    return Z**3 - (1 + B)*Z**2 + A*Z - A*B

# Newton iterations starting from ideal-gas guess Z=1
Z = 1.0
print("=== Engineering: van der Waals compressibility factor (CO₂, 300 K, 50 bar) ===")
print(f"  A = {A:.6f},  B = {B:.6f}")
for k in range(6):
    r  = residual(Z)
    dr = derivative(residual, Z)   # ← complex step, no hand differentiation
    Z_new = Z - r / dr
    print(f"  iter {k+1}: Z = {Z_new:.8f}   |r| = {abs(r):.2e}")
    if abs(Z_new - Z) < 1e-12:
        break
    Z = Z_new

print(f"  Converged compressibility factor Z = {Z:.8f}")
print(f"  (ideal gas: Z = 1.000;  van der Waals correction visible)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Multi-argument functions — partial derivatives via lambda
# ─────────────────────────────────────────────────────────────────────────────
# derivative() requires a single-argument function f(x) → scalar.  When your
# model has the form f(t, N0, r, K), fix the other arguments with a lambda.
# This is the explicit version of what gradient/jacobian/directional_derivative
# do automatically via wrt=.
#
# Application: logistic population growth  (Verhulst model)
#
#   N(t; N0, r, K) = K / (1 + (K/N0 - 1)·exp(-r·t))
#
# Three scalar sensitivities of interest to an ecologist or resource manager:
#   ∂N/∂t  — how fast is the population growing right now?
#   ∂N/∂K  — if the habitat supports one more individual (ΔK=1), how many
#             extra individuals appear at time t?
#   ∂N/∂r  — sensitivity to the intrinsic growth rate (e.g. after a disease)

def logistic(t, N0, r, K):
    """Logistic growth: N(t) = K / (1 + (K/N0 - 1)*exp(-r*t))."""
    return K / (1 + (K / N0 - 1) * np.exp(-r * t))

N0_log, r_log, K_log = 50.0, 0.4, 500.0

print("=== Multi-arg: logistic growth — partial derivatives via lambda ===")
print(f"  dN/dt = r·N·(1-N/K),  N(0)={N0_log},  r={r_log},  K={K_log}")
print(f"  {'t':>4}  {'N(t)':>8}  {'∂N/∂t':>8}  {'∂N/∂K':>9}  {'∂N/∂r':>9}")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}")

for t_eval in [2.0, 5.0, 10.0, 20.0]:
    N_t   = logistic(t_eval, N0_log, r_log, K_log)
    dN_dt = derivative(lambda t: logistic(t, N0_log, r_log, K_log), t_eval)
    dN_dK = derivative(lambda K: logistic(t_eval, N0_log, r_log, K), K_log)
    dN_dr = derivative(lambda r: logistic(t_eval, N0_log, r, K_log), r_log)
    print(f"  {t_eval:>4.0f}  {N_t:>8.1f}  {dN_dt:>8.3f}  {dN_dK:>9.5f}  {dN_dr:>9.3f}")

print("  ∂N/∂t → 0 as t → ∞ confirms the population saturates at K.")
print("  The lambda pattern lets derivative() differentiate any scalar argument.")
