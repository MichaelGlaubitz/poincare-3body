import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import io

MU = 0.01215

def cr3bp_derivatives(t, state):
    x, y, vx, vy = state
    r1 = np.sqrt((x + MU)**2 + y**2)
    r2 = np.sqrt((x - 1 + MU)**2 + y**2)
    Omega_x = x - (1 - MU)*(x + MU)/r1**3 - MU*(x - 1 + MU)/r2**3
    Omega_y = y - (1 - MU)*y/r1**3 - MU*y/r2**3
    ax = Omega_x + 2*vy
    ay = Omega_y - 2*vx
    return [vx, vy, ax, ay]

def simulate_and_poincare(y0, t_max=500, n_points=5000):
    t_span = [0, t_max]
    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(cr3bp_derivatives, t_span, y0, method='DOP853', t_eval=t_eval, rtol=1e-9, atol=1e-11)
    y = sol.y
    z = y[1]
    sign_changes = np.diff(np.sign(z))
    idx = np.where(sign_changes != 0)[0]
    points = []
    for i in idx:
        if i + 1 >= len(z): continue
        s = z[i] / (z[i] - z[i+1])
        points.append([y[0][i] + s * (y[0][i+1] - y[0][i]), y[2][i] + s * (y[2][i+1] - y[2][i])])
    return np.array(points) if points else np.empty((0, 2))

st.set_page_config(page_title="Poincaré – 3-Körper-Problem: Grid-Scan", layout="wide")

st.title("🌍 Moon-Like Poincaré-Schnitte: Parameter-Scan")
st.markdown("Interaktiver **3 × 3 SCAN** des **Circular Restricted Three-Body Problems **(CR3BP).")

# --- Parameters for grid ---
col1, col2, col3 = st.columns(3)
with col1:
    base_E = st.number_input("Base Energie E", -4.5, -2.0, -3.4, 0.1, key="base_E")
with col2:
    base_x0 = st.slider("Base x₀", -1.2, 1.2, 0.95, 0.01, key="base_x0")
with col3:
    base_vx0 = st.slider("Base vx₀", -0.3, 0.3, 0.05, 0.001, key="base_vx0")

scan_E = st.slider("Δ Energie Scan", 0.0, 0.5, 0.1, 0.01, key="scan_E")
scan_x0 = st.slider("Δ x₀ Scan", 0.0, 0.2, 0.05, 0.01, key="scan_x0")

# Scan steps: 3 × 3 = 9
scan_steps = 3
E_list = [base_E + (i - 1) * scan_E for i in range(scan_steps)]
x0_list = [base_x0 + (j - 1) * scan_x0 for j in range(scan_steps)]

# --- Progress bar ---
progress_text = st.empty()
progress_bar = st.progress(0)
total_steps = scan_steps * scan_steps
steps_done = 0

# Compute all 9 plots
results = []
for i, E in enumerate(E_list):
    row_points = []
    for j, x0 in enumerate(x0_list):
        # vy0 from energy
        Omega = (x0**2)/2 + (1-MU)/abs(x0 + MU) + MU/abs(x0 - 1 + MU)
        vy0_sq = 2*(E + Omega)
        if vy0_sq < 0:
            vy0 = 0  # fallback
        else:
            vy0 = np.sqrt(vy0_sq)
        y0 = [x0, 0.0, base_vx0, vy0]
        t_max = 800
        points = simulate_and_poincare(y0, t_max, n_points=3000)
        row_points.append(points)
        steps_done += 1
        progress_bar.progress(steps_done / total_steps)
    results.append(row_points)

progress_bar.empty()

# --- Grid View: 3 × 3 plots ---
st.subheader("Poincaré-Schnitte: 3 × 3 Grid")
cols = st.columns(3)
for i, E in enumerate(E_list):
    for j, x0 in enumerate(x0_list):
        with cols[j]:
            st.markdown(f"**E = {E:.2f}, x₀ = {x0:.2f}**")
            fig, ax = plt.subplots(figsize=(3, 3))
            points = results[i][j]
            if len(points) > 0:
                ax.scatter(points[:,0], points[:,1], s=0.8, c='navy', alpha=0.7)
            ax.set_xlabel("x"); ax.set_ylabel("vₓ"); ax.set_title(f"E={E:.2f}, x₀={x0:.2f}", fontsize=8)
            ax.grid(True, which='both', alpha=0.3); plt.tight_layout()
            st.pyplot(fig)

# --- Export button ---
st.markdown("---")
st.markdown("### Export aller 9 Plots als PDF")
if st.button("PDF exportieren"):
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages("poincare_grid.pdf")
    for i, E in enumerate(E_list):
        for j, x0 in enumerate(x0_list):
            fig, ax = plt.subplots(figsize=(5, 4))
            points = results[i][j]
            if len(points) > 0:
                ax.scatter(points[:,0], points[:,1], s=1, c='navy', alpha=0.6)
            ax.set_xlabel("x"); ax.set_ylabel("vₓ")
            ax.set_title(f"E={E:.2f}, x₀={x0:.2f}")
            ax.grid(True); plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close()
    with open("poincare_grid.pdf", "rb") as f:
        st.download_button("💾 PDF aller 9 Plots", f.read(), "poincare_grid.pdf", "application/pdf")
