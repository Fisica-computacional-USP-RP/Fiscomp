import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# MASSA-MOLA (SEM AMORTECIMENTO)
# ============================================================
# EDO do problema:
#   x'' + (k/m)*x = 0
#
# Neste arquivo comparamos dois integradores:
# 1) Euler-Cromer
# 2) Runge-Kutta de 4a ordem (RK4)
#
# A visualizacao final mostra apenas dois paineis:
# - aparato experimental com os dois metodos
# - grafico normalizado x(t)/x0 com as duas curvas


# --------------------------
# BLOCO 1: PARAMETROS
# --------------------------
m = 1.0
k = 1.0

x0 = 1.0
v0 = 0.0

dt = 0.02
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


# --------------------------
# BLOCO 2: EULER-CROMER
# --------------------------
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


# --------------------------
# BLOCO 3: RK4
# --------------------------
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


# --------------------------
# BLOCO 4: SAIDA CRUA
# --------------------------
def mostrar_saida_numerica(nome_metodo, x):
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)      x(m)        x/x0")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {x[i]:10.6f}   {x[i] / x0:11.6f}")


# --------------------------
# BLOCO 5: DESENHO DA MOLA
# --------------------------
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


# --------------------------
# BLOCO 6: ANIMACAO + GRAFICO
# --------------------------
def animar_comparacao(x_euler, x_rk4):
    x_euler_n = x_euler / x0
    x_rk4_n = x_rk4 / x0

    max_frames = 180
    passo_frame = max(1, N // max_frames)
    frames_anim = np.arange(0, N, passo_frame)
    intervalo_ms = max(15, int(1000 * dt * passo_frame))

    fig, (ax_mola, ax_xt) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Painel 1: aparato com os dois metodos
    ax_mola.set_title("Massa-mola: Euler-Cromer x RK4")
    ax_mola.set_xlabel("x (m)")
    ax_mola.set_ylabel("y")
    xmin = min(-1.3, np.min(np.concatenate([x_euler, x_rk4])) - 0.3)
    xmax = max(1.3, np.max(np.concatenate([x_euler, x_rk4])) + 0.3)
    ax_mola.set_xlim(xmin, xmax)
    ax_mola.set_ylim(-0.35, 0.35)
    ax_mola.grid(alpha=0.3)
    ax_mola.axvline(-1.2, color="gray", lw=3)

    linha_mola_euler, = ax_mola.plot([], [], color="tab:orange", lw=2, label="Euler-Cromer")
    massa_euler, = ax_mola.plot([], [], "o", color="tab:orange", ms=12)
    linha_mola_rk4, = ax_mola.plot([], [], color="tab:green", lw=2, label="RK4")
    massa_rk4, = ax_mola.plot([], [], "o", color="tab:green", ms=12)
    ax_mola.legend(loc="upper right")

    # Painel 2: grafico normalizado com os dois metodos
    amp = 1.1 * max(np.max(np.abs(x_euler_n)), np.max(np.abs(x_rk4_n)), 1.0)
    ax_xt.set_title("Posicao normalizada")
    ax_xt.set_xlabel("tempo (s)")
    ax_xt.set_ylabel("x(t)/x0")
    ax_xt.set_xlim(0.0, t_final)
    ax_xt.set_ylim(-amp, amp)
    ax_xt.grid(alpha=0.3)

    linha_euler, = ax_xt.plot([], [], color="tab:orange", lw=2, label="Euler-Cromer")
    linha_rk4, = ax_xt.plot([], [], color="tab:green", lw=2, label="RK4")
    ax_xt.legend(loc="upper right")

    def init():
        linha_mola_euler.set_data([], [])
        massa_euler.set_data([], [])
        linha_mola_rk4.set_data([], [])
        massa_rk4.set_data([], [])
        linha_euler.set_data([], [])
        linha_rk4.set_data([], [])
        return linha_mola_euler, massa_euler, linha_mola_rk4, massa_rk4, linha_euler, linha_rk4

    def update(frame):
        xs_e, ys_e = desenhar_mola(x_euler[frame], y_nivel=0.12)
        xs_r, ys_r = desenhar_mola(x_rk4[frame], y_nivel=-0.12)

        linha_mola_euler.set_data(xs_e, ys_e)
        massa_euler.set_data([x_euler[frame]], [0.12])
        linha_mola_rk4.set_data(xs_r, ys_r)
        massa_rk4.set_data([x_rk4[frame]], [-0.12])

        linha_euler.set_data(t[: frame + 1], x_euler_n[: frame + 1])
        linha_rk4.set_data(t[: frame + 1], x_rk4_n[: frame + 1])
        return linha_mola_euler, massa_euler, linha_mola_rk4, massa_rk4, linha_euler, linha_rk4

    ani = FuncAnimation(fig, update, frames=frames_anim, init_func=init, interval=intervalo_ms, blit=False)
    fig._ani = ani
    plt.tight_layout()
    plt.show()
    return ani


# --------------------------
# BLOCO 7: EXECUCAO
# --------------------------
x_euler, v_euler = simular_euler()
x_rk4, v_rk4 = simular_rk4()

mostrar_saida_numerica("Euler-Cromer", x_euler)
mostrar_saida_numerica("Runge-Kutta 4", x_rk4)
animar_comparacao(x_euler, x_rk4)
