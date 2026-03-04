import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PENDULO SIMPLES
# ============================================================
# EDO do problema:
#   theta'' + (g/L)*sin(theta) = 0
#
# Neste arquivo vamos resolver a mesma EDO de duas formas:
# 1) Metodo de Euler
# 2) Metodo de Runge-Kutta de 4a ordem (RK4)
#
# Para cada metodo:
# - mostramos uma saida numerica curta
# - mostramos os graficos do pendulo


# --------------------------
# BLOCO 1: PARAMETROS
# --------------------------
# Parametros fisicos
g = 9.81             # gravidade (m/s^2)
L = 1.0              # comprimento do fio (m)

# Condicoes iniciais
theta0 = np.deg2rad(20.0)   # angulo inicial em radianos
omega0 = 0.0                # velocidade angular inicial (rad/s)

# Parametros numericos
dt = 0.01                   # passo de tempo (s)
t_final = 20.0              # tempo total (s)
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


# --------------------------
# BLOCO 2: METODO DE EULER
# --------------------------
def simular_euler():
    # Vetores para guardar angulo e velocidade no tempo
    theta = np.zeros(N)
    omega = np.zeros(N)

    # Coloca as condicoes iniciais no primeiro ponto
    theta[0] = theta0
    omega[0] = omega0

    # Avanca passo a passo no tempo
    for i in range(N - 1):
        # Aceleracao angular vinda da EDO
        alpha = -(g / L) * np.sin(theta[i])

        # Euler para velocidade
        omega[i + 1] = omega[i] + alpha * dt

        # Euler para posicao angular
        theta[i + 1] = theta[i] + omega[i] * dt

    return theta, omega


# --------------------------
# BLOCO 3: METODO RK4
# --------------------------
def derivadas(theta, omega):
    # Reescreve a EDO de 2a ordem como sistema de 1a ordem
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return dtheta_dt, domega_dt


def simular_rk4():
    # Vetores para guardar angulo e velocidade no tempo
    theta = np.zeros(N)
    omega = np.zeros(N)

    # Coloca as condicoes iniciais no primeiro ponto
    theta[0] = theta0
    omega[0] = omega0

    # Avanca passo a passo no tempo
    for i in range(N - 1):
        th = theta[i]
        om = omega[i]

        # Quatro avaliacoes do campo de derivadas (RK4)
        k1_th, k1_om = derivadas(th, om)
        k2_th, k2_om = derivadas(th + 0.5 * dt * k1_th, om + 0.5 * dt * k1_om)
        k3_th, k3_om = derivadas(th + 0.5 * dt * k2_th, om + 0.5 * dt * k2_om)
        k4_th, k4_om = derivadas(th + dt * k3_th, om + dt * k3_om)

        # Atualizacao final combinando os quatro ks
        theta[i + 1] = th + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega


# --------------------------
# BLOCO 4: SAIDA CRUA
# --------------------------
def mostrar_saida_numerica(nome_metodo, theta, omega):
    # Converte angulo para posicao horizontal
    x = L * np.sin(theta)

    # Mostra so os primeiros pontos para leitura didatica
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)    theta(rad)    omega(rad/s)    x(m)")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {theta[i]:10.6f}    {omega[i]:11.6f}   {x[i]:8.5f}")


# --------------------------
# BLOCO 5: GRAFICOS
# --------------------------
def plotar_pendulo(nome_metodo, theta):
    # Converte angulo em coordenadas cartesianas da massa
    x = L * np.sin(theta)
    y = -L * np.cos(theta)

    # Cria dois paineis: evolucao temporal e trajetoria espacial
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Painel 1: theta(t)
    axs[0].plot(t, theta, color="tab:blue")
    axs[0].set_title(f"{nome_metodo} - Angulo theta(t)")
    axs[0].set_xlabel("tempo (s)")
    axs[0].set_ylabel("theta (rad)")
    axs[0].grid(alpha=0.3)

    # Painel 2: trajetoria x-y do pendulo
    axs[1].plot(x, y, color="tab:orange")
    axs[1].scatter([0.0], [0.0], color="black", s=30, label="pivo")
    axs[1].set_title(f"{nome_metodo} - Trajetoria no plano x-y")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    axs[1].axis("equal")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# --------------------------
# BLOCO 6: EXECUCAO
# --------------------------
# 1) Primeiro Euler
# 2) Depois RK4

# Euler
theta_euler, omega_euler = simular_euler()
mostrar_saida_numerica("Euler", theta_euler, omega_euler)
plotar_pendulo("Euler", theta_euler)

# RK4
theta_rk4, omega_rk4 = simular_rk4()
mostrar_saida_numerica("Runge-Kutta 4", theta_rk4, omega_rk4)
plotar_pendulo("Runge-Kutta 4", theta_rk4)
