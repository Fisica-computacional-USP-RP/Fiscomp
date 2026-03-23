import numpy as np
import matplotlib.pyplot as plt


taxas = [0.0925, 0.1375, 0.1175, 0.1225, 0.1500, 0.1500]  # taxas anuais usadas na simulacao
anos_taxas = ["2021", "2022", "2023", "2024", "2025", "2026"]  # rotulos para mostrar ao aluno


def ler_numero(texto, padrao):
    entrada = input(f"{texto} [{padrao}]: ").strip()  # le o valor digitado
    if entrada == "":
        return float(padrao)  # usa o padrao se o aluno apertar Enter
    return float(entrada.replace(",", "."))  # aceita virgula ou ponto


def reais(x):
    texto = f"{x:,.2f}"  # formata com 2 casas decimais
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")  # troca para padrao brasileiro


print("Simulador simples de investimento\n")

capital_inicial = ler_numero("Capital inicial em reais", 1000)  # dinheiro que ja existe no inicio
aporte_mensal = ler_numero("Aporte mensal em reais", 500)  # valor adicionado a cada mes
tempo_total = ler_numero("Tempo total em anos", 5)  # duracao da simulacao

meses = max(1, int(round(tempo_total * 12)))  # transforma anos em meses
tempo = np.arange(meses + 1) / 12  # eixo do tempo em anos
saldo = np.zeros(meses + 1)  # guarda o saldo com rendimento
investido = np.zeros(meses + 1)  # guarda quanto foi colocado do proprio bolso

saldo[0] = capital_inicial  # saldo inicial
investido[0] = capital_inicial  # investimento inicial

for mes in range(meses):
    indice_ano = min(mes // 12, len(taxas) - 1)  # escolhe a taxa do ano atual
    taxa_anual = taxas[indice_ano]  # pega a taxa anual correspondente
    taxa_mensal = (1 + taxa_anual) ** (1 / 12) - 1  # converte taxa anual para mensal

    saldo[mes + 1] = saldo[mes] * (1 + taxa_mensal) + aporte_mensal  # rende e depois recebe o aporte
    investido[mes + 1] = investido[mes] + aporte_mensal  # soma apenas o dinheiro aportado

valor_final = saldo[-1]  # ultimo valor da simulacao
total_investido = investido[-1]  # total colocado pelo usuario
rendimento = valor_final - total_investido  # lucro bruto do modelo

print("Resumo\n")
print(f"Capital inicial: R$ {reais(capital_inicial)}")
print(f"Aporte mensal:   R$ {reais(aporte_mensal)}")
print(f"Tempo total:     {tempo_total:.2f} anos")
print(f"Total investido: R$ {reais(total_investido)}")
print(f"Valor final:     R$ {reais(valor_final)}")
print(f"Rendimento:      R$ {reais(rendimento)}")

print("\nTaxas anuais usadas")
for i in range(max(1, int(np.ceil(tempo_total)))):
    indice_ano = min(i, len(taxas) - 1)  # repete a ultima taxa se o tempo for maior
    print(f"{anos_taxas[indice_ano]}: {100 * taxas[indice_ano]:.2f}% ao ano")

print("\nResumo por ano")
print("Ano   Investido (R$)   Saldo (R$)")
for i in range(0, meses + 1, 12):
    print(f"{tempo[i]:3.0f}   {reais(investido[i]):>14}   {reais(saldo[i]):>10}")

if meses % 12 != 0:
    print(f"{tempo[-1]:3.0f}   {reais(investido[-1]):>14}   {reais(saldo[-1]):>10}")  # mostra o ultimo ponto se nao fechar ano

plt.figure(figsize=(9, 5))
plt.plot(tempo, investido, lw=2.4, label="Total investido")  # linha do dinheiro colocado
plt.plot(tempo, saldo, lw=2.4, label="Saldo estimado")  # linha do saldo com rendimento
plt.fill_between(tempo, investido, saldo, where=saldo >= investido, alpha=0.2, label="Rendimento")  # area do ganho
plt.title("Simulacao simples de investimento")
plt.xlabel("Tempo (anos)")
plt.ylabel("Valor (R$)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
if "agg" in plt.get_backend().lower():
    plt.close()  # fecha a figura em ambiente sem janela grafica
else:
    plt.show()  # mostra o grafico no uso normal
