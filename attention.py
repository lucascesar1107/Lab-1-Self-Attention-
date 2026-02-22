import numpy as np

def scaled_dot_product_attention(Q, K, V):
    dk = K.shape[1]

    scores = Q@K.T
    scores = scores/np.sqrt(dk)

    exp_scores = np.exp(scores)
    softmax_scores = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    output = softmax_scores @ V

    return output