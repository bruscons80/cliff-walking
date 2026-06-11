# Cliff Walking - Q-Learning Tabular vs Aproximacao de Funcao

Implementacao do problema Cliff Walking comparando um agente **Q-Learning tabular** com um agente que usa **aproximacao de funcao linear** e atualizacao por **semi-gradiente TD**.

---

## 1. Ambiente Cliff Walking

Grid de **4 linhas x 12 colunas** (48 estados). O agente comeca em `(3,0)` e precisa chegar em `(3,11)`.

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]   <- linha 0
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]   <- linha 1
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]   <- linha 2
[S][X][X][X][X][X][X][X][X][X][X][G]   <- linha 3
```

- **S** = inicio, **G** = objetivo, **X** = precipicio
- 4 acoes: cima, direita, baixo, esquerda
- Recompensa padrao: **-1** por passo
- Precipicio: recompensa **-100** e volta ao inicio (o episodio nao termina, apenas reinicia a posicao)
- O agente nao pode sair do grid (fica parado se tentar)

O caminho otimo tem **13 passos** (sobe, percorre a linha 2 ate o final, desce) com recompensa total de **-13**.

---

## 2. Q-Learning Tabular

### Estrutura

Tabela `Q(s, a)` com uma entrada para cada par estado-acao (48 estados x 4 acoes = 192 valores). Inicializada com zeros.

### Regra de atualizacao

Q-Learning e um metodo **off-policy**: a atualizacao usa o maximo sobre as acoes do proximo estado, independente da acao efetivamente tomada.

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

Onde:
- `alpha = 0.5` — taxa de aprendizado (valor alto funciona bem aqui porque o ambiente e deterministico)
- `gamma = 1.0` — sem desconto, ja que e episodico e queremos minimizar a soma total
- `r` — recompensa recebida
- `s'` — proximo estado

### Politica epsilon-greedy

Com probabilidade `epsilon` escolhe acao aleatoria, senao escolhe `argmax Q(s, a)`.

O epsilon decai multiplicativamente a cada episodio:

```
epsilon <- max(epsilon_min, epsilon * decay)
```

Com `epsilon_start = 1.0`, `epsilon_min = 0.01` e `decay = 0.995`, o agente comeca explorando 100% e estabiliza em ~1% apos ~900 episodios.

---

## 3. Aproximacao de Funcao com Semi-Gradiente TD

### Representacao

Em vez de tabela, usamos uma funcao linear:

```
Q(s, a, w) = w^T * phi(s, a)
```

Onde `phi(s, a)` e o vetor de features e `w` sao os pesos aprendidos.

### Vetor de features phi(s, a)

Para cada estado `(r, c)` definimos 6 features:

| Feature | Formula | Motivacao |
|---|---|---|
| Linha normalizada | `r / 3` | Posicao vertical |
| Coluna normalizada | `c / 11` | Progresso horizontal em direcao ao objetivo |
| Distancia Manhattan normalizada | `(\|r - 3\| + \|c - 11\|) / 14` | Proximidade do objetivo |
| Perto do precipicio | `1 se vizinho e cliff, 0 caso contrario` | Perigo imediato |
| Linha do fundo | `1 se r = 3, 0 caso contrario` | Regiao do precipicio |
| Bias | `1` | Termo independente |

O vetor `phi(s, a)` completo tem **24 dimensoes** (6 features x 4 acoes). Para uma acao especifica, so o bloco correspondente tem valores; o resto e zero. Isso permite que cada acao tenha pesos independentes enquanto compartilha a mesma representacao do estado.

### Regra de atualizacao (semi-gradiente TD)

```
w <- w + alpha * delta * grad_w Q(s, a, w)
```

Onde o erro TD e:

```
delta = r + gamma * max_a' Q(s', a', w) - Q(s, a, w)
```

Como `Q(s, a, w) = w^T * phi(s, a)`, o gradiente e simplesmente:

```
grad_w Q(s, a, w) = phi(s, a)
```

Entao a atualizacao fica:

```
w <- w + alpha * [ r + gamma * max_a' Q(s', a', w) - Q(s, a, w) ] * phi(s, a)
```

E chamado **semi-gradiente** porque o target `r + gamma * max Q(s', a', w)` tambem depende de `w`, mas ignoramos esse gradiente. Diferenciamos apenas o termo `Q(s, a, w)` que esta sendo subtraido.

### Hiperparametros

- `alpha = 0.01` — muito menor que no tabular porque uma unica atualizacao afeta todos os estados (via pesos compartilhados). Alpha alto causa oscilacao e divergencia.
- `gamma = 1.0` e epsilon-greedy identicos ao tabular para comparacao justa.

---

## 4. Analise Comparativa

### Resultados (media de 10 execucoes, 1000 episodios)

| Metrica | Q-Learning Tabular | Aprox. Funcao |
|---|---|---|
| Parametros | 192 | 24 |
| Recompensa final (media ultimos 100 eps) | ~ -15.8 | ~ -16.6 |
| Convergencia (todas as runs) | 10/10 | 10/10 |
| Encontra caminho otimo | Sim | Sim |

### Generalizacao

A aproximacao de funcao generaliza: quando aprende que "perto do precipicio + ir pra baixo e ruim", isso vale para **todos** os estados proximos ao precipicio. No tabular, cada estado aprende independentemente.

Isso e vantajoso em ambientes grandes (milhares de estados) onde visitar todos os estados seria impraticavel. Porem, no Cliff Walking com apenas 48 estados, a tabela Q resolve sem dificuldade.

### Estabilidade

O Q-Learning tabular tem **convergencia garantida** (sob condicoes padrao de decaimento do alpha). Com aproximacao de funcao, a combinacao de bootstrapping + funcao parametrizada + treinamento off-policy (o **deadly triad** do Sutton & Barto) pode causar divergencia. A funcao linear e mais estavel que redes neurais, mas a garantia formal nao existe.

### Caso prejudicial da generalizacao

O precipicio cria uma **descontinuidade**: estados vizinhos como `(2,5)` (seguro) e `(3,5)` (precipicio) tem features muito parecidas mas exigem valores Q completamente diferentes. A funcao linear tenta suavizar essa transicao e acaba com valores imprecisos para ambos os lados. No tabular, `Q(2,5)` e `Q(3,5)` sao entradas separadas que nao interferem uma na outra.

---

## Como Executar

```bash
pip install numpy matplotlib jupyter
jupyter notebook cliff_walking.ipynb
```

Executar todas as celulas sequencialmente. O treinamento com multiplas seeds (secao 4) leva alguns segundos.
