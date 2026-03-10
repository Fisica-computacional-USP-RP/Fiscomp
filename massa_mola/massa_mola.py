import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


m = 1.0
k = 1.0

x0 = 1.0
v0 = 0.0

dt = 0.02
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


def simular_euler():
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

    for i in range(N - 1):
        a = -(k / m) * x[i]
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt

    return x, v


def derivadas(x, v):
    return v, -(k / m) * x


def simular_rk4():
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

    for i in range(N - 1):
        xi = x[i]
        vi = v[i]

        k1_x, k1_v = derivadas(xi, vi)
        k2_x, k2_v = derivadas(xi + 0.5 * dt * k1_x, vi + 0.5 * dt * k1_v)
        k3_x, k3_v = derivadas(xi + 0.5 * dt * k2_x, vi + 0.5 * dt * k2_v)
        k4_x, k4_v = derivadas(xi + dt * k3_x, vi + dt * k3_v)

        x[i + 1] = xi + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v[i + 1] = vi + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return x, v


def mostrar_saida_numerica(nome_metodo, x):
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)      x(m)")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {x[i]:10.6f}")


def indices_animacao(total_pontos, max_frames=160):
    return np.unique(np.linspace(0, total_pontos - 1, min(max_frames, total_pontos), dtype=int))


def desenhar_mola(x_massa, y_nivel, x_parede=-1.2, espiras=12, amplitude=0.05, pontos=120):
    x_final = x_massa - 0.08
    if x_final <= x_parede + 0.05:
        x_final = x_parede + 0.05

    xs = np.linspace(x_parede, x_final, pontos)
    fase = np.linspace(0.0, 2.0 * np.pi * espiras, pontos)
    ys = y_nivel + amplitude * np.sin(fase)
    ys[0] = y_nivel
    ys[-1] = y_nivel
    return xs, ys


def animar_massa_mola(x, nome_metodo):
    frames_anim = indices_animacao(N)
    salto = max(1, frames_anim[1] - frames_anim[0]) if len(frames_anim) > 1 else 1
    intervalo_ms = max(20, int(1000 * dt * salto))
    amp = 1.1 * max(np.max(np.abs(x)), abs(x0))
    xmin = min(-1.45, np.min(x) - 0.45)
    xmax = max(1.45, np.max(x) + 0.45)

    fig, (ax_mola, ax_xt) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_mola.set_title(f"Sistema massa-mola - {nome_metodo}")
    ax_mola.set_xlabel("x (m)")
    ax_mola.set_ylabel("y")
    ax_mola.set_xlim(xmin, xmax)
    ax_mola.set_ylim(-0.35, 0.35)
    ax_mola.grid(alpha=0.3)
    ax_mola.axvline(-1.2, color="gray", lw=3)

    linha_mola, = ax_mola.plot([], [], color="tab:green", lw=2.5)
    massa, = ax_mola.plot([], [], "o", color="tab:green", ms=12)

    ax_xt.set_title(f"{nome_metodo} - x(t)")
    ax_xt.set_xlabel("tempo (s)")
    ax_xt.set_ylabel("x (m)")
    ax_xt.set_xlim(0.0, t_final)
    ax_xt.set_ylim(-amp, amp)
    ax_xt.grid(alpha=0.3)
    linha_xt, = ax_xt.plot([], [], color="tab:green", lw=2)

    def init():
        linha_mola.set_data([], [])
        massa.set_data([], [])
        linha_xt.set_data([], [])
        return linha_mola, massa, linha_xt

    def update(frame):
        xs, ys = desenhar_mola(x[frame], y_nivel=0.0)
        linha_mola.set_data(xs, ys)
        massa.set_data([x[frame]], [0.0])
        linha_xt.set_data(t[: frame + 1], x[: frame + 1])
        return linha_mola, massa, linha_xt

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


x_euler, v_euler = simular_euler()
x_rk4, v_rk4 = simular_rk4()

mostrar_saida_numerica("Euler-Cromer", x_euler)
mostrar_saida_numerica("Runge-Kutta 4", x_rk4)
anim_bonus = animar_massa_mola(x_rk4, "RK4")
