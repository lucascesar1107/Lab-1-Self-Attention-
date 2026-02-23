# Lab P1-01 – Implementação do Scaled Dot-Product Attention

Este projeto implementa o mecanismo de Scaled Dot-Product Attention conforme descrito no artigo científico *Attention Is All You Need*.

A implementação segue a fórmula:

Attention(Q, K, V) = softmax((QK^T) / sqrt(dk)) V

A biblioteca utilizada para as operações matriciais foi o NumPy.

---

## Requisitos

- Python 3.x  
- NumPy  

Instalação do NumPy:

pip install numpy

---

## Execução

Para executar o exemplo de teste, utilize:

python test_attention.py

O script imprime as matrizes de entrada (Q, K, V) e o resultado da aplicação do mecanismo de atenção.

---

## Normalização pelo fator √dk

Após o cálculo do produto escalar QK^T, os valores são divididos por √dk, onde dk representa a dimensão das chaves (número de colunas de K).

Essa normalização reduz a magnitude dos valores antes da aplicação do softmax, evitando que a distribuição resultante fique excessivamente concentrada.

---

## Exemplo Numérico

Entrada utilizada no teste:

```
Q = [[1, 0],
     [0, 1]]

K = [[1, 0],
     [0, 1]]

V = [[10, 0],
     [0, 20]]

Saída obtida:

[[ 6.69  6.60]
 [ 3.30 13.39]]
```
Os valores acima correspondem ao resultado da aplicação completa da fórmula do Scaled Dot-Product Attention.
