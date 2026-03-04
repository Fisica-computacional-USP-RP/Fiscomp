import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# PENDULO SIMPLES (didatico)
# Metodos: definicao de derivada, Euler e RK4
# EDO: theta'' + (g/L)*sin(theta) = 0
# ============================================================

# BLOCO 1 - Parametros
g = 9.81
L = 1.0
theta0 = np.deg2rad(20.0)
omega0 = 0.0
dt = 0.01
t_final = 20.0
t = np.arange(0.0, t_final + dt, dt)


def mostrar_amostra(nome, theta, omega, n=10):
    x = L * np.sin(theta)
    print(f"\nSaida numerica - {nome}")
    print(" i   t(s)    theta(rad)    omega(rad/s)    x(m)")
    for i in range(min(n, len(t))):
        print(f"{i:2d}  {t[i]:6.3f}   {theta[i]:10.6f}    {omega[i]:11.6f}   {x[i]:8.5f}")


# BLOCO 2 - Metodo por definicao de derivada (diferenca finita 2a ordem)
def simular_definicao_derivada():
    n = len(t)
    theta = np.zeros(n)
    omega = np.zeros(n)

    theta[0] = theta0
    alpha0 = -(g / L) * np.sin(theta0)
    theta[1] = theta0 + omega0 * dt + 0.5 * alpha0 * dt**2

    for i in range(1, n - 1):
        alpha_i = -(g / L) * np.sin(theta[i])
        theta[i + 1] = 2 * theta[i] - theta[i - 1] + alpha_i * dt**2

    omega[0] = omega0
    omega[1:-1] = (theta[2:] - theta[:-2]) / (2 * dt)
    omega[-1] = (theta[-1] - theta[-2]) / dt
    return theta, omega


# BLOCO 3 - Metodo de Euler
def simular_euler():
    n = len(t)
    theta = np.zeros(n)
    omega = np.zeros(n)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(n - 1):
        alpha = -(g / L) * np.sin(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i] * dt

    return theta, omega


# BLOCO 4 - Metodo RK4
def derivadas(theta, omega):
    return omega, -(g / L) * np.sin(theta)


def simular_rk4():
    n = len(t)
    theta = np.zeros(n)
    omega = np.zeros(n)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(n - 1):
        th, om = theta[i], omega[i]

        k1_th, k1_om = derivadas(th, om)
        k2_th, k2_om = derivadas(th + 0.5 * dt * k1_th, om + 0.5 * dt * k1_om)
        k3_th, k3_om = derivadas(th + 0.5 * dt * k2_th, om + 0.5 * dt * k2_om)
        k4_th, k4_om = derivadas(th + dt * k3_th, om + dt * k3_om)

        theta[i + 1] = th + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega


def plotar_comparativo(theta_def, omega_def, theta_euler, omega_euler, theta_rk4, omega_rk4):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    axs[0].plot(t, theta_def, label="Def. derivada", linewidth=1.2)
    axs[0].plot(t, theta_euler, label="Euler", linewidth=1.2)
    axs[0].plot(t, theta_rk4, label="RK4", linewidth=2)
    axs[0].set_title("Comparacao de theta(t)")
    axs[0].set_xlabel("tempo (s)")
    axs[0].set_ylabel("theta (rad)")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    axs[1].plot(t, omega_def, label="Def. derivada", linewidth=1.2)
    axs[1].plot(t, omega_euler, label="Euler", linewidth=1.2)
    axs[1].plot(t, omega_rk4, label="RK4", linewidth=2)
    axs[1].set_title("Comparacao de omega(t)")
    axs[1].set_xlabel("tempo (s)")
    axs[1].set_ylabel("omega (rad/s)")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    axs[2].plot(t, np.abs(theta_rk4 - theta_def), label="|RK4 - Def. derivada|")
    axs[2].plot(t, np.abs(theta_rk4 - theta_euler), label="|RK4 - Euler|")
    axs[2].set_title("Erro absoluto contra RK4")
    axs[2].set_xlabel("tempo (s)")
    axs[2].set_ylabel("erro")
    axs[2].grid(alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def animar_pendulo(theta, titulo="Pendulo", max_frames=400):
    passo = max(1, len(t) // max_frames)
    t_anim = t[::passo]
    theta_anim = theta[::passo]

    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(titulo)

    ax_p.set_aspect("equal")
    ax_p.set_xlim(-1.2 * L, 1.2 * L)
    ax_p.set_ylim(-1.2 * L, 0.2 * L)
    ax_p.set_title("Movimento do pendulo")
    ax_p.plot(0, 0, "ko", ms=6)
    haste, = ax_p.plot([], [], lw=2, color="tab:blue")
    bob, = ax_p.plot([], [], "o", ms=10, color="tab:red")

    amp = max(np.max(np.abs(theta_anim)) * 1.2, 0.2)
    ax_g.set_xlim(0, t[-1])
    ax_g.set_ylim(-amp, amp)
    ax_g.set_title("Angulo ao longo do tempo")
    ax_g.set_xlabel("tempo (s)")
    ax_g.set_ylabel("theta (rad)")
    linha_theta, = ax_g.plot([], [], color="tab:orange")

    def update(i):
        th = theta_anim[i]
        x = L * np.sin(th)
        y = -L * np.cos(th)
        haste.set_data([0, x], [0, y])
        bob.set_data([x], [y])
        linha_theta.set_data(t_anim[: i + 1], theta_anim[: i + 1])
        return haste, bob, linha_theta

    ani = FuncAnimation(fig, update, frames=len(theta_anim), interval=30, blit=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    theta_def, omega_def = simular_definicao_derivada()
    theta_euler, omega_euler = simular_euler()
    theta_rk4, omega_rk4 = simular_rk4()

    mostrar_amostra("Definicao de derivada", theta_def, omega_def)
    mostrar_amostra("Euler", theta_euler, omega_euler)
    mostrar_amostra("RK4", theta_rk4, omega_rk4)

    plotar_comparativo(theta_def, omega_def, theta_euler, omega_euler, theta_rk4, omega_rk4)

    # Animacao principal (troque o metodo aqui se quiser)
    animar_pendulo(theta_rk4, titulo="Pendulo - RK4")
