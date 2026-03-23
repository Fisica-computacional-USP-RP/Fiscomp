from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figuras"
FIG_DIR.mkdir(exist_ok=True)


def figura_investimento():
    taxas = [0.0925, 0.1375, 0.1175, 0.1225, 0.1500]
    capital_inicial = 1000.0
    aporte = 500.0
    anos = 5
    meses = anos * 12

    tempo = np.arange(meses + 1) / 12
    saldo = np.zeros(meses + 1)
    investido = np.zeros(meses + 1)

    saldo[0] = capital_inicial
    investido[0] = capital_inicial

    for mes in range(meses):
        taxa_anual = taxas[min(mes // 12, len(taxas) - 1)]
        taxa_mensal = (1 + taxa_anual) ** (1 / 12) - 1
        saldo[mes + 1] = saldo[mes] * (1 + taxa_mensal) + aporte
        investido[mes + 1] = investido[mes] + aporte

    plt.figure(figsize=(6.8, 3.8))
    plt.plot(tempo, investido, lw=2.3, label="Total investido")
    plt.plot(tempo, saldo, lw=2.3, label="Saldo estimado")
    plt.fill_between(tempo, investido, saldo, where=saldo >= investido, alpha=0.18)
    plt.xlabel("Tempo (anos)")
    plt.ylabel("Valor (R$)")
    plt.title("Investimento com aporte mensal")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "investimento_simples.png", dpi=220)
    plt.close()


def modelo_alpha(P0, r, alpha, t):
    if np.isclose(alpha, 1.0):
        return P0 * np.exp(r * t)

    base = P0 ** (1 - alpha) + (1 - alpha) * r * t
    y = np.full_like(t, np.nan, dtype=float)
    ok = base > 0
    y[ok] = base[ok] ** (1 / (1 - alpha))
    return y


def figura_alpha():
    P0 = 1000.0
    r = 0.12
    t = np.linspace(0, 4, 400)

    plt.figure(figsize=(6.8, 3.8))
    for alpha in [0.5, 1.0, 1.2]:
        plt.plot(t, modelo_alpha(P0, r, alpha, t), lw=2.3, label=fr"$\alpha={alpha}$")

    plt.xlabel("Tempo (anos)")
    plt.ylabel("Capital")
    plt.title("Regimes de crescimento no modelo com alpha")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regimes_alpha.png", dpi=220)
    plt.close()


def main():
    figura_investimento()
    figura_alpha()
    print(FIG_DIR / "investimento_simples.png")
    print(FIG_DIR / "regimes_alpha.png")


if __name__ == "__main__":
    main()
