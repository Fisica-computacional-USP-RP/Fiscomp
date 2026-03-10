import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# PENDULO SIMPLES
# ============================================================
# EDO do problema:
#   theta'' + (g/L)*sin(theta) = 0
#
# Neste arquivo comparamos dois integradores:
# 1) Euler-Cromer
# 2) Runge-Kutta de 4a ordem (RK4)
#
# A visualizacao final mostra apenas dois paineis:
# - aparato experimental com os dois metodos
# - grafico normalizado theta(t)/theta0 com as duas curvas


# --------------------------
# BLOCO 1: PARAMETROS
# --------------------------
g = 9.81
L = 1.0

theta0 = np.deg2rad(20.0)
omega0 = 0.0

dt = 0.02
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


# --------------------------
# BLOCO 2: EULER-CROMER
# --------------------------
def simular_euler():
    theta = np.zeros(N)
    omega = np.zeros(N)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(N - 1):
        alpha = -(g / L) * np.sin(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt

    return theta, omega


# --------------------------
# BLOCO 3: RK4
# --------------------------
def derivadas(theta, omega):
    return omega, -(g / L) * np.sin(theta)


def simular_rk4():
    theta = np.zeros(N)
    omega = np.zeros(N)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(N - 1):
        th = theta[i]
        om = omega[i]

        k1_th, k1_om = derivadas(th, om)
        k2_th, k2_om = derivadas(th + 0.5 * dt * k1_th, om + 0.5 * dt * k1_om)
        k3_th, k3_om = derivadas(th + 0.5 * dt * k2_th, om + 0.5 * dt * k2_om)
        k4_th, k4_om = derivadas(th + dt * k3_th, om + dt * k3_om)

        theta[i + 1] = th + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega


# --------------------------
# BLOCO 4: SAIDA CRUA
# --------------------------
def mostrar_saida_numerica(nome_metodo, theta):
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)    theta(rad)    theta/theta0")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {theta[i]:10.6f}   {theta[i] / theta0:11.6f}")


# --------------------------
# BLOCO 5: ANIMACAO + GRAFICO
# --------------------------
def animar_comparacao(theta_euler, theta_rk4):
    x_euler = L * np.sin(theta_euler)
    y_euler = -L * np.cos(theta_euler)
    x_rk4 = L * np.sin(theta_rk4)
    y_rk4 = -L * np.cos(theta_rk4)

    theta_euler_n = theta_euler / theta0
    theta_rk4_n = theta_rk4 / theta0

    max_frames = 180
    passo_frame = max(1, N // max_frames)
    frames_anim = np.arange(0, N, passo_frame)
    intervalo_ms = max(15, int(1000 * dt * passo_frame))

    fig, (ax_pendulo, ax_theta) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Painel 1: aparato com os dois metodos
    ax_pendulo.set_title("Pendulo: Euler-Cromer x RK4")
    ax_pendulo.set_xlabel("x (m)")
    ax_pendulo.set_ylabel("y (m)")
    ax_pendulo.set_xlim(-1.2 * L, 1.2 * L)
    ax_pendulo.set_ylim(-1.2 * L, 0.2 * L)
    ax_pendulo.axis("equal")
    ax_pendulo.grid(alpha=0.3)
    ax_pendulo.scatter([0.0], [0.0], color="black", s=30, label="pivo")

    haste_euler, = ax_pendulo.plot([], [], color="tab:orange", lw=2, label="Euler-Cromer")
    massa_euler, = ax_pendulo.plot([], [], "o", color="tab:orange", ms=10)
    haste_rk4, = ax_pendulo.plot([], [], color="tab:green", lw=2, label="RK4")
    massa_rk4, = ax_pendulo.plot([], [], "o", color="tab:green", ms=10)
    ax_pendulo.legend(loc="lower left")

    # Painel 2: grafico normalizado com os dois metodos
    amp = 1.1 * max(np.max(np.abs(theta_euler_n)), np.max(np.abs(theta_rk4_n)), 1.0)
    ax_theta.set_title("Angulo normalizado")
    ax_theta.set_xlabel("tempo (s)")
    ax_theta.set_ylabel("theta(t)/theta0")
    ax_theta.set_xlim(0.0, t_final)
    ax_theta.set_ylim(-amp, amp)
    ax_theta.grid(alpha=0.3)

    linha_euler, = ax_theta.plot([], [], color="tab:orange", lw=2, label="Euler-Cromer")
    linha_rk4, = ax_theta.plot([], [], color="tab:green", lw=2, label="RK4")
    ax_theta.legend(loc="upper right")

    def init():
        haste_euler.set_data([], [])
        massa_euler.set_data([], [])
        haste_rk4.set_data([], [])
        massa_rk4.set_data([], [])
        linha_euler.set_data([], [])
        linha_rk4.set_data([], [])
        return haste_euler, massa_euler, haste_rk4, massa_rk4, linha_euler, linha_rk4

    def update(frame):
        haste_euler.set_data([0.0, x_euler[frame]], [0.0, y_euler[frame]])
        massa_euler.set_data([x_euler[frame]], [y_euler[frame]])
        haste_rk4.set_data([0.0, x_rk4[frame]], [0.0, y_rk4[frame]])
        massa_rk4.set_data([x_rk4[frame]], [y_rk4[frame]])

        linha_euler.set_data(t[: frame + 1], theta_euler_n[: frame + 1])
        linha_rk4.set_data(t[: frame + 1], theta_rk4_n[: frame + 1])
        return haste_euler, massa_euler, haste_rk4, massa_rk4, linha_euler, linha_rk4

    ani = FuncAnimation(fig, update, frames=frames_anim, init_func=init, interval=intervalo_ms, blit=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()
    return ani


# --------------------------
# BLOCO 6: EXECUCAO
# --------------------------
theta_euler, omega_euler = simular_euler()
theta_rk4, omega_rk4 = simular_rk4()

mostrar_saida_numerica("Euler-Cromer", theta_euler)
mostrar_saida_numerica("Runge-Kutta 4", theta_rk4)
animar_comparacao(theta_euler, theta_rk4)
