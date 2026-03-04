import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# MASSA-MOLA (didatico)
# Metodos: definicao de derivada, Euler e RK4
# EDO: x'' + (k/m)*x = 0
# ============================================================

# BLOCO 1 - Parametros
m = 1.0
k = 1.0
x0 = 1.0
v0 = 0.0
dt = 0.01
t_final = 20.0
t = np.arange(0.0, t_final + dt, dt)


def mostrar_amostra(nome, x, v, n=10):
    print(f"\nSaida numerica - {nome}")
    print(" i   t(s)      x(m)        v(m/s)")
    for i in range(min(n, len(t))):
        print(f"{i:2d}  {t[i]:6.3f}   {x[i]:10.6f}   {v[i]:10.6f}")


# BLOCO 2 - Metodo por definicao de derivada
def simular_definicao_derivada():
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)

    x[0] = x0
    a0 = -(k / m) * x0
    x[1] = x0 + v0 * dt + 0.5 * a0 * dt**2

    for i in range(1, n - 1):
        a_i = -(k / m) * x[i]
        x[i + 1] = 2 * x[i] - x[i - 1] + a_i * dt**2

    v[0] = v0
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[-1] = (x[-1] - x[-2]) / dt
    return x, v


# BLOCO 3 - Metodo de Euler
def simular_euler():
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)

    x[0] = x0
    v[0] = v0

    for i in range(n - 1):
        a = -(k / m) * x[i]
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i] * dt

    return x, v


# BLOCO 4 - Metodo RK4
def derivadas(x, v):
    return v, -(k / m) * x


def simular_rk4():
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)

    x[0] = x0
    v[0] = v0

    for i in range(n - 1):
        xi, vi = x[i], v[i]

        k1_x, k1_v = derivadas(xi, vi)
        k2_x, k2_v = derivadas(xi + 0.5 * dt * k1_x, vi + 0.5 * dt * k1_v)
        k3_x, k3_v = derivadas(xi + 0.5 * dt * k2_x, vi + 0.5 * dt * k2_v)
        k4_x, k4_v = derivadas(xi + dt * k3_x, vi + dt * k3_v)

        x[i + 1] = xi + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v[i + 1] = vi + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return x, v


def plotar_comparativo(x_def, x_euler, x_rk4):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(t, x_def, label="Def. derivada", linewidth=1.2)
    ax.plot(t, x_euler, label="Euler", linewidth=1.2)
    ax.plot(t, x_rk4, label="RK4", linewidth=2)
    ax.set_title("Comparacao de posicao x(t)")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("x (m)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def desenhar_mola(x_massa, x_parede=-1.2, espiras=12, amplitude=0.08, pontos=220):
    x_final = x_massa - 0.08
    if x_final <= x_parede + 0.05:
        x_final = x_parede + 0.05

    xs = np.linspace(x_parede, x_final, pontos)
    fase = np.linspace(0.0, 2.0 * np.pi * espiras, pontos)
    ys = amplitude * np.sin(fase)
    ys[0] = 0.0
    ys[-1] = 0.0
    return xs, ys


def animar_massa_mola(x, titulo="Massa-mola", max_frames=400):
    passo = max(1, len(t) // max_frames)
    t_anim = t[::passo]
    x_anim = x[::passo]

    fig, (ax_mola, ax_xt) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    ax_mola.set_title(f"{titulo} - Mola oscilando")
    ax_mola.set_xlabel("x (m)")
    ax_mola.set_ylabel("y")
    xmin = min(-1.3, np.min(x_anim) - 0.3)
    xmax = max(1.3, np.max(x_anim) + 0.3)
    ax_mola.set_xlim(xmin, xmax)
    ax_mola.set_ylim(-0.4, 0.4)
    ax_mola.grid(alpha=0.3)
    ax_mola.axvline(-1.2, color="gray", lw=3)

    linha_mola, = ax_mola.plot([], [], color="tab:blue", lw=2)
    massa, = ax_mola.plot([], [], "o", color="tab:red", ms=14)

    ax_xt.set_title(f"{titulo} - Posicao x(t)")
    ax_xt.set_xlabel("tempo (s)")
    ax_xt.set_ylabel("x (m)")
    ax_xt.set_xlim(0.0, t[-1])
    margem = 0.1 * max(abs(np.min(x_anim)), abs(np.max(x_anim)), 1e-6)
    ax_xt.set_ylim(np.min(x_anim) - margem, np.max(x_anim) + margem)
    ax_xt.grid(alpha=0.3)
    linha_xt, = ax_xt.plot([], [], color="tab:orange", lw=2)

    def update(i):
        xs, ys = desenhar_mola(x_anim[i])
        linha_mola.set_data(xs, ys)
        massa.set_data([x_anim[i]], [0.0])
        linha_xt.set_data(t_anim[: i + 1], x_anim[: i + 1])
        return linha_mola, massa, linha_xt

    ani = FuncAnimation(fig, update, frames=len(x_anim), interval=30, blit=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x_def, v_def = simular_definicao_derivada()
    x_euler, v_euler = simular_euler()
    x_rk4, v_rk4 = simular_rk4()

    mostrar_amostra("Definicao de derivada", x_def, v_def)
    mostrar_amostra("Euler", x_euler, v_euler)
    mostrar_amostra("RK4", x_rk4, v_rk4)

    plotar_comparativo(x_def, x_euler, x_rk4)

    # Animacao principal (troque o metodo aqui se quiser)
    animar_massa_mola(x_rk4, titulo="Massa-mola - RK4")
