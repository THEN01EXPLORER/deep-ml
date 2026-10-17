# import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Args:
        X: list of lists or 2D array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: 2D arrays/lists of shape (d_out, d_in)
    
    Returns:
        Nested Python list of shape (seq_len, d_out)
    """
    X = np.array(X, dtype=np.float64)
    
    # 1. Transpose linear weights from (d_out, d_in) to parameter-style (d_in, d_out)
    W_q = np.array(W_q_lin, dtype=np.float64).T
    W_k = np.array(W_k_lin, dtype=np.float64).T
    W_v = np.array(W_v_lin, dtype=np.float64).T
    
    seq_len = X.shape[0]
    d_out = W_q.shape[1]
    
    # 2. Compute Q, K, V
    Q = X @ W_q  # (seq_len, d_out)
    K = X @ W_k  # (seq_len, d_out)
    V = X @ W_v  # (seq_len, d_out)
    
    # 3. Compute scaled scores: S = (Q @ K.T) / sqrt(d_out)
    S = (Q @ K.T) / np.sqrt(d_out)  # (seq_len, seq_len)
    
    # 4. Apply causal mask: mask positions where key_idx > query_idx
    # np.triu with k=1 selects all strictly upper-triangular elements
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    S[mask] = -1e9
    
    # 5. Numerically stable softmax along keys (last axis)
    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    attn_weights = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    
    # 6. Compute output: attention @ V
    output = attn_weights @ V
    
    return output.tolist()