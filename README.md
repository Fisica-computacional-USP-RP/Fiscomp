# Fiscomp

Computational physics projects with didactic simulations for monitoria.

## Estrutura atual (didatica e simples)
- `pendulo_simples/`
  - `pendulo_simples.py`
  - `pendulo_simples.ipynb`
- `massa_mola/`
  - `massa_mola.py`
  - `massa_mola.ipynb`

## Cada modelo mostra
1. simulacao com Euler
2. simulacao com Runge-Kutta 4 (RK4)
3. saida crua (tabela inicial)
4. graficos comparativos

## Abrir direto no Colab
[![Pendulo no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pemodest0/Fiscomp/blob/main/pendulo_simples/pendulo_simples.ipynb)
[![Massa-Mola no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pemodest0/Fiscomp/blob/main/massa_mola/massa_mola.ipynb)

Passo a passo rapido:
1. Clique em um dos botoes acima.
2. No Colab, clique em `Runtime` -> `Run all`.
3. Para alterar parametros (`dt`, `t_final`, condicoes iniciais), edite a celula de parametros e rode novamente.
