import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# MASSA-MOLA (didatico)
# Metodos: definicao de derivada, Euler e RK4
# EDO: x'' + (k/m)*x = 0
# ============================================================

# BLOCO 1 - Parametros
M = 1.0
K = 1.0
X0 = 1.0
V0 = 0.0
DT = 0.01
T_FINAL = 20.0
T = np.arange(0.0, T_FINAL + DT, DT)


def mostrar_amostra(nome, x, n=10):
    """Tabela curta somente com tempo e posicao."""
    print(f"\nSaida numerica - {nome}")
    print(" i   t(s)      x(m)")
    for i in range(min(n, len(T))):
        print(f"{i:2d}  {T[i]:6.3f}   {x[i]:10.6f}")


# BLOCO 2 - Definicao de derivada (diferenca finita de 2a ordem)
def simular_definicao_derivada():
    n = len(T)
    x = np.zeros(n)

    x[0] = X0
    a0 = -(K / M) * X0
    x[1] = X0 + V0 * DT + 0.5 * a0 * DT**2

    for i in range(1, n - 1):
        a_i = -(K / M) * x[i]
        x[i + 1] = 2 * x[i] - x[i - 1] + a_i * DT**2

    return x


# BLOCO 3 - Euler
def simular_euler():
    n = len(T)
    x = np.zeros(n)
    v = np.zeros(n)

    x[0] = X0
    v[0] = V0

    for i in range(n - 1):
        a = -(K / M) * x[i]
        v[i + 1] = v[i] + a * DT
        x[i + 1] = x[i] + v[i] * DT

    return x


# BLOCO 4 - RK4
def derivadas(x, v):
    return v, -(K / M) * x


def simular_rk4():
    n = len(T)
    x = np.zeros(n)
    v = np.zeros(n)

    x[0] = X0
    v[0] = V0

    for i in range(n - 1):
        xi, vi = x[i], v[i]

        k1_x, k1_v = derivadas(xi, vi)
        k2_x, k2_v = derivadas(xi + 0.5 * DT * k1_x, vi + 0.5 * DT * k1_v)
        k3_x, k3_v = derivadas(xi + 0.5 * DT * k2_x, vi + 0.5 * DT * k2_v)
        k4_x, k4_v = derivadas(xi + DT * k3_x, vi + DT * k3_v)

        x[i + 1] = xi + (DT / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v[i + 1] = vi + (DT / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return x


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


def animar_aparato_e_curvas(x_aparato, x_def, x_euler, x_rk4, titulo="Massa-mola"):
    """Esquerda: aparato massa-mola. Direita: 3 curvas de posicao sendo formadas."""
    max_frames = 420
    passo = max(1, len(T) // max_frames)

    t_anim = T[::passo]
    xa = x_aparato[::passo]
    xd = x_def[::passo]
    xe = x_euler[::passo]
    xr = x_rk4[::passo]

    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(titulo)

    # Painel esquerdo: aparato
    ax_s.set_title("Aparato experimental: massa-mola")
    ax_s.set_xlabel("x (m)")
    ax_s.set_ylabel("y")
    xmin = min(-1.3, np.min(xa) - 0.3)
    xmax = max(1.3, np.max(xa) + 0.3)
    ax_s.set_xlim(xmin, xmax)
    ax_s.set_ylim(-0.4, 0.4)
    ax_s.grid(alpha=0.3)
    ax_s.axvline(-1.2, color="gray", lw=3)

    linha_mola, = ax_s.plot([], [], color="tab:blue", lw=2)
    massa, = ax_s.plot([], [], "o", color="tab:red", ms=14)

    # Painel direito: 3 curvas em tempo real
    amp = max(np.max(np.abs(np.concatenate([xd, xe, xr]))) * 1.25, 0.2)
    ax_g.set_xlim(0, T[-1])
    ax_g.set_ylim(-amp, amp)
    ax_g.set_title("x(t): Def. derivada, Euler e RK4")
    ax_g.set_xlabel("tempo (s)")
    ax_g.set_ylabel("x (m)")
    ax_g.grid(alpha=0.3)

    linha_def, = ax_g.plot([], [], color="black", linestyle="--", linewidth=2.0, label="Def. derivada", zorder=4)
    linha_eul, = ax_g.plot([], [], color="tab:orange", linewidth=1.8, label="Euler", zorder=3)
    linha_rk, = ax_g.plot([], [], color="tab:green", linewidth=1.8, label="RK4", zorder=2)

    pt_def, = ax_g.plot([], [], "o", color="black", ms=4)
    pt_eul, = ax_g.plot([], [], "o", color="tab:orange", ms=4)
    pt_rk, = ax_g.plot([], [], "o", color="tab:green", ms=4)
    ax_g.legend(loc="upper right")

    def init():
        linha_mola.set_data([], [])
        massa.set_data([], [])
        linha_def.set_data([], [])
        linha_eul.set_data([], [])
        linha_rk.set_data([], [])
        pt_def.set_data([], [])
        pt_eul.set_data([], [])
        pt_rk.set_data([], [])
        return linha_mola, massa, linha_def, linha_eul, linha_rk, pt_def, pt_eul, pt_rk

    def update(i):
        xs, ys = desenhar_mola(xa[i])
        linha_mola.set_data(xs, ys)
        massa.set_data([xa[i]], [0.0])

        tt = t_anim[: i + 1]
        yd = xd[: i + 1]
        ye = xe[: i + 1]
        yr = xr[: i + 1]

        linha_def.set_data(tt, yd)
        linha_eul.set_data(tt, ye)
        linha_rk.set_data(tt, yr)

        pt_def.set_data([tt[-1]], [yd[-1]])
        pt_eul.set_data([tt[-1]], [ye[-1]])
        pt_rk.set_data([tt[-1]], [yr[-1]])
        return linha_mola, massa, linha_def, linha_eul, linha_rk, pt_def, pt_eul, pt_rk

    ani = FuncAnimation(fig, update, frames=len(t_anim), init_func=init, interval=30, blit=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x_def = simular_definicao_derivada()
    x_euler = simular_euler()
    x_rk4 = simular_rk4()

    mostrar_amostra("Definicao de derivada", x_def)
    mostrar_amostra("Euler", x_euler)
    mostrar_amostra("RK4", x_rk4)

    # Aparato animado com RK4; grafico do lado mostra as 3 curvas se formando.
    animar_aparato_e_curvas(x_rk4, x_def, x_euler, x_rk4, titulo="Massa-mola: aparato + curvas")
