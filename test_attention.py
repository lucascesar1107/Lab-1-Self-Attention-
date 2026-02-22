import numpy as np
from attention import scaled_dot_product_attention

def main():
    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[10.0, 0.0],
                  [0.0, 20.0]])

    out = scaled_dot_product_attention(Q, K, V)

    print('Q:\n', Q)
    print('K:\n', K)
    print('V:\n', V)
    print('out:\n', out)

    assert out.shape == (2, 2)

if __name__ == '__main__':
    main()