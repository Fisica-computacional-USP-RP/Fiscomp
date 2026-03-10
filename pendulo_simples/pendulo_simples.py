import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


g = 9.81
L = 1.0

theta0 = np.deg2rad(20.0)
omega0 = 0.0

dt = 0.01
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


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


def mostrar_saida_numerica(nome_metodo, theta):
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)    theta(rad)")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {theta[i]:10.6f}")


def indices_animacao(total_pontos, max_frames=160):
    return np.unique(np.linspace(0, total_pontos - 1, min(max_frames, total_pontos), dtype=int))


def animar_pendulo(theta, nome_metodo):
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    frames_anim = indices_animacao(N)
    salto = max(1, frames_anim[1] - frames_anim[0]) if len(frames_anim) > 1 else 1
    intervalo_ms = max(20, int(1000 * dt * salto))
    amp = 1.1 * max(np.max(np.abs(theta)), abs(theta0))
    margem = 0.22

    fig, (ax_pendulo, ax_theta) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_pendulo.set_title(f"Pendulo simples - {nome_metodo}")
    ax_pendulo.set_xlabel("x (m)")
    ax_pendulo.set_ylabel("y (m)")
    ax_pendulo.set_xlim(-(L + margem), L + margem)
    ax_pendulo.set_ylim(-(L + margem), 0.22)
    ax_pendulo.set_aspect("equal", adjustable="box")
    ax_pendulo.grid(alpha=0.3)
    ax_pendulo.scatter([0.0], [0.0], color="black", s=28)

    haste, = ax_pendulo.plot([], [], color="tab:green", lw=2.5)
    massa, = ax_pendulo.plot([], [], "o", color="tab:green", ms=12)

    ax_theta.set_title(f"{nome_metodo} - theta(t)")
    ax_theta.set_xlabel("tempo (s)")
    ax_theta.set_ylabel("theta (rad)")
    ax_theta.set_xlim(0.0, t_final)
    ax_theta.set_ylim(-amp, amp)
    ax_theta.grid(alpha=0.3)
    linha_theta, = ax_theta.plot([], [], color="tab:green", lw=2)

    def init():
        haste.set_data([], [])
        massa.set_data([], [])
        linha_theta.set_data([], [])
        return haste, massa, linha_theta

    def update(frame):
        haste.set_data([0.0, x[frame]], [0.0, y[frame]])
        massa.set_data([x[frame]], [y[frame]])
        linha_theta.set_data(t[: frame + 1], theta[: frame + 1])
        return haste, massa, linha_theta

    ani = FuncAnimation(
        fig,
        update,
        frames=frames_anim,
        init_func=init,
        interval=intervalo_ms,
        blit=False,
        cache_frame_data=False,
    )
    fig._ani = ani
    plt.tight_layout()
    plt.show()
    return ani


theta_euler, omega_euler = simular_euler()
theta_rk4, omega_rk4 = simular_rk4()

mostrar_saida_numerica("Euler-Cromer", theta_euler)
mostrar_saida_numerica("Runge-Kutta 4", theta_rk4)
anim_bonus = animar_pendulo(theta_rk4, "RK4")
