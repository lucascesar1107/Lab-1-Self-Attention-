# Lab P1-01 - Implementação de Self-Attention

Este projeto implementa o mecanismo de Scaled Dot-Product Attention usando apenas NumPy.

A fórmula utilizada foi:

Attention(Q, K, V) = softmax((QK^T) / sqrt(dk)) V

## Requisitos

- Python 3.x
- Numpy

## Como executar

1. Instalar NumPy:
pip install numpy

2. Executar o teste:
python test_attention.py

## Sobre o fator sqrt(dk)

A divisão por sqrt(dk) é utilizada para evitar que os valores do produto escalar fiquem muito grandes, o que poderia afetar o resultado do softmax.

## Observação

O arquivo test_attention.py contém um exemplo numérico simples para validar o funcionamento da implementação.
