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
1. simulacao por definicao de derivada (diferenca finita 2a ordem)
2. simulacao com Euler
3. simulacao com Runge-Kutta 4 (RK4)
4. saida crua (tabela inicial)
5. graficos comparativos de posicao
6. animacao no proprio notebook

## Abrir direto no Colab
[![Pendulo no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pemodest0/Fiscomp/blob/main/pendulo_simples/pendulo_simples.ipynb)
[![Massa-Mola no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pemodest0/Fiscomp/blob/main/massa_mola/massa_mola.ipynb)

Passo a passo rapido:
1. Clique em um dos botoes acima.
2. No Colab, clique em `Runtime` -> `Run all`.
3. Espere gerar os graficos comparativos antes da animacao (a animacao vem na ultima celula).
4. Para alterar parametros (`dt`, `t_final`, condicoes iniciais), edite a celula de parametros e rode novamente.
