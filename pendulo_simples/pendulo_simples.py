import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# PENDULO SIMPLES (didatico)
# Metodos: definicao de derivada, Euler e RK4
# EDO: theta'' + (g/L)*sin(theta) = 0
# ============================================================

# BLOCO 1 - Parametros
G = 9.81
L = 1.0
THETA0 = np.deg2rad(20.0)
OMEGA0 = 0.0
DT = 0.01
T_FINAL = 20.0
T = np.arange(0.0, T_FINAL + DT, DT)


def mostrar_amostra(nome, theta, n=10):
    """Tabela curta somente com tempo e posicao."""
    x = L * np.sin(theta)
    print(f"\nSaida numerica - {nome}")
    print(" i   t(s)    theta(rad)      x(m)")
    for i in range(min(n, len(T))):
        print(f"{i:2d}  {T[i]:6.3f}   {theta[i]:10.6f}   {x[i]:8.5f}")


# BLOCO 2 - Definicao de derivada (diferenca finita de 2a ordem)
def simular_definicao_derivada():
    n = len(T)
    theta = np.zeros(n)

    theta[0] = THETA0
    alpha0 = -(G / L) * np.sin(THETA0)
    theta[1] = THETA0 + OMEGA0 * DT + 0.5 * alpha0 * DT**2

    for i in range(1, n - 1):
        alpha_i = -(G / L) * np.sin(theta[i])
        theta[i + 1] = 2 * theta[i] - theta[i - 1] + alpha_i * DT**2

    return theta


# BLOCO 3 - Euler
def simular_euler():
    n = len(T)
    theta = np.zeros(n)
    omega = np.zeros(n)

    theta[0] = THETA0
    omega[0] = OMEGA0

    for i in range(n - 1):
        alpha = -(G / L) * np.sin(theta[i])
        omega[i + 1] = omega[i] + alpha * DT
        theta[i + 1] = theta[i] + omega[i] * DT

    return theta


# BLOCO 4 - RK4
def derivadas(theta, omega):
    return omega, -(G / L) * np.sin(theta)


def simular_rk4():
    n = len(T)
    theta = np.zeros(n)
    omega = np.zeros(n)

    theta[0] = THETA0
    omega[0] = OMEGA0

    for i in range(n - 1):
        th, om = theta[i], omega[i]

        k1_th, k1_om = derivadas(th, om)
        k2_th, k2_om = derivadas(th + 0.5 * DT * k1_th, om + 0.5 * DT * k1_om)
        k3_th, k3_om = derivadas(th + 0.5 * DT * k2_th, om + 0.5 * DT * k2_om)
        k4_th, k4_om = derivadas(th + DT * k3_th, om + DT * k3_om)

        theta[i + 1] = th + (DT / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om + (DT / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta


def animar_aparato_e_curvas(theta_aparato, theta_def, theta_euler, theta_rk4, titulo="Pendulo"):
    """Esquerda: aparato do pendulo. Direita: 3 curvas de posicao sendo formadas."""
    max_frames = 420
    passo = max(1, len(T) // max_frames)

    t_anim = T[::passo]
    th_app = theta_aparato[::passo]
    th_def = theta_def[::passo]
    th_eul = theta_euler[::passo]
    th_rk = theta_rk4[::passo]

    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(titulo)

    # Painel esquerdo: aparato
    ax_p.set_aspect("equal")
    ax_p.set_xlim(-1.2 * L, 1.2 * L)
    ax_p.set_ylim(-1.2 * L, 0.2 * L)
    ax_p.set_title("Aparato experimental: pendulo")
    ax_p.plot(0, 0, "ko", ms=6)
    haste, = ax_p.plot([], [], lw=2, color="tab:blue")
    bob, = ax_p.plot([], [], "o", ms=10, color="tab:red")

    # Painel direito: 3 curvas em tempo real
    amp = max(np.max(np.abs(np.concatenate([th_def, th_eul, th_rk]))) * 1.2, 0.2)
    ax_g.set_xlim(0, T[-1])
    ax_g.set_ylim(-amp, amp)
    ax_g.set_title("theta(t): Def. derivada, Euler e RK4")
    ax_g.set_xlabel("tempo (s)")
    ax_g.set_ylabel("theta (rad)")
    ax_g.grid(alpha=0.3)

    linha_def, = ax_g.plot([], [], color="black", linestyle="--", linewidth=2.0, label="Def. derivada", zorder=4)
    linha_eul, = ax_g.plot([], [], color="tab:orange", linewidth=1.8, label="Euler", zorder=3)
    linha_rk, = ax_g.plot([], [], color="tab:green", linewidth=1.8, label="RK4", zorder=2)

    pt_def, = ax_g.plot([], [], "o", color="black", ms=4)
    pt_eul, = ax_g.plot([], [], "o", color="tab:orange", ms=4)
    pt_rk, = ax_g.plot([], [], "o", color="tab:green", ms=4)
    ax_g.legend(loc="upper right")

    def init():
        haste.set_data([], [])
        bob.set_data([], [])
        linha_def.set_data([], [])
        linha_eul.set_data([], [])
        linha_rk.set_data([], [])
        pt_def.set_data([], [])
        pt_eul.set_data([], [])
        pt_rk.set_data([], [])
        return haste, bob, linha_def, linha_eul, linha_rk, pt_def, pt_eul, pt_rk

    def update(i):
        th = th_app[i]
        x = L * np.sin(th)
        y = -L * np.cos(th)
        haste.set_data([0, x], [0, y])
        bob.set_data([x], [y])

        tt = t_anim[: i + 1]
        yd = th_def[: i + 1]
        ye = th_eul[: i + 1]
        yr = th_rk[: i + 1]

        linha_def.set_data(tt, yd)
        linha_eul.set_data(tt, ye)
        linha_rk.set_data(tt, yr)

        pt_def.set_data([tt[-1]], [yd[-1]])
        pt_eul.set_data([tt[-1]], [ye[-1]])
        pt_rk.set_data([tt[-1]], [yr[-1]])
        return haste, bob, linha_def, linha_eul, linha_rk, pt_def, pt_eul, pt_rk

    ani = FuncAnimation(fig, update, frames=len(t_anim), init_func=init, interval=30, blit=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    theta_def = simular_definicao_derivada()
    theta_euler = simular_euler()
    theta_rk4 = simular_rk4()

    mostrar_amostra("Definicao de derivada", theta_def)
    mostrar_amostra("Euler", theta_euler)
    mostrar_amostra("RK4", theta_rk4)

    # Aparato animado com RK4; grafico do lado mostra as 3 curvas se formando.
    animar_aparato_e_curvas(theta_rk4, theta_def, theta_euler, theta_rk4, titulo="Pendulo: aparato + curvas")
