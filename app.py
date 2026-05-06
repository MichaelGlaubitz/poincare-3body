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

st.set_page_config(page_title="Poincaré – 3-Körper-Problem")
st.title("🌍 Moon-Like Poincaré-Schnitte vom CR3BP")
st.markdown("Interaktive Visualisierung des **Circular Restricted Three-Body Problems**.")
col1, col2 = st.columns([2, 1])
with col1:
    x0 = st.slider("Startposition x₀", -1.2, 1.2, 0.95, 0.01)
    vx0 = st.slider("Startgeschwindigkeit vx₀", -0.3, 0.3, 0.05, 0.001)
with col2:
    E = st.number_input("Energie E", -4.5, -2.0, -3.4, 0.1)
Omega = (x0**2)/2 + (1-MU)/abs(x0 + MU) + MU/abs(x0 - 1 + MU)
vy0_sq = 2*(E + Omega)
if vy0_sq < 0:
    st.error("⚠️ Für diese Energie und Position ist `vy0² < 0` → unphysikalisch.")
    vy0 = 0
else:
    vy0 = st.slider("Startgeschwindigkeit vy₀", -0.5, 0.5, np.sqrt(vy0_sq), 0.01)
t_max = st.slider("Integrationsdauer t_max", 100, 2000, 800, step=100)

if st.button("Berechnen", key="btn_calc"):
    y0 = [x0, 0.0, vx0, vy0]
    @st.cache_data(show_spinner=False)
    def cached_simulate_and_poincare(y0, t_max):
        return simulate_and_poincare(y0, t_max, n_points=10000)
    with st.spinner("Simuliere 3-Körper-Dynamik ..."):
        points = cached_simulate_and_poincare(y0, t_max)
    st.success(f"{len(points)} Punkte im Poincaré-Schnitt gefunden!")
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(points) > 0:
        ax.scatter(points[:,0], points[:,1], s=1.5, c='navy', alpha=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("vₓ"); ax.set_title(f"Poincaré-Schnitt bei y=0, vy>0 (E={E})"); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=150); st.pyplot(fig)
    st.download_button("💾 Plot als PNG", buf.getvalue(), "poincare.png", "image/png")
    st.download_button("📊 Daten als CSV", "\n".join([",".join(map(str, p)) for p in points]), "poincare_data.csv", "text/csv")
else:
    st.info("Klicke auf **Berechnen**.")
