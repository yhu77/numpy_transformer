import numpy as np

# ======== Utility ========
def softmax(x):
    # Numerically stable softmax along last axis
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def create_causal_mask(seq_len, batch_size=1, n_heads=1):
    """
    Create a causal mask of shape (1, 1, seq_len, seq_len) that can be
    broadcast over (batch_size, n_heads, seq_len, seq_len).

    True  = keep
    False = mask out
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    # (1, 1, seq_len, seq_len) – will broadcast to (batch, heads, seq_len, seq_len)
    return mask[np.newaxis, np.newaxis, :, :]


# ======== Multihead Attention ========
class MultiheadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        # Projection matrices
        self.Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # Bias terms for projections (not strictly necessary, but closer to standard impls)
        self.bq = np.zeros((d_model,))
        self.bk = np.zeros((d_model,))
        self.bv = np.zeros((d_model,))
        self.bo = np.zeros((d_model,))

    def split_heads(self, x):
        """
        x: (batch_size, seq_len, d_model)
        return: (batch_size, n_heads, seq_len, depth)
        """
        x = np.reshape(x, (x.shape[0], -1, self.n_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch_size, seq_len, d_model)

        mask:
            - Can be shape (batch_size, n_heads, seq_len_q, seq_len_k)
              or broadcastable to that shape.
            - True / 1  => keep position
            - False / 0 => mask out (set score to -inf before softmax)
        """
        # Linear projections + bias
        Q = np.dot(query, self.Wq) + self.bq
        K = np.dot(key, self.Wk) + self.bk
        V = np.dot(value, self.Wv) + self.bv

        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Q: (batch, heads, seq_q, depth)
        # K: (batch, heads, seq_k, depth)
        dk = K.shape[-1]
        score = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) / np.sqrt(dk)
        # score: (batch, heads, seq_q, seq_k)

        if mask is not None:
            # Ensure boolean mask: True = keep, False = mask out
            if mask.dtype != bool:
                mask = mask.astype(bool)

            # Broadcast mask to score shape if needed
            if mask.shape != score.shape:
                mask = np.broadcast_to(mask, score.shape)

            # Where mask is False, set score to very negative
            score = np.where(mask, score, -1e9)

        attention_weights = softmax(score)
        context = np.matmul(attention_weights, V)  # (batch, heads, seq_q, depth)

        # Combine heads
        context = np.transpose(context, (0, 2, 1, 3))  # (batch, seq_q, heads, depth)
        context = np.reshape(context, (context.shape[0], -1, self.d_model))  # (batch, seq_q, d_model)

        out = np.dot(context, self.Wo) + self.bo  # (batch, seq_q, d_model)

        return out, attention_weights


# ======== Positional Encoding ========
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pos_encoding = self.positional_encoding()

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def positional_encoding(self):
        angle_rads = self.get_angles(
            np.arange(self.max_len)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :]
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return pos_encoding

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pos_encoding[:, :x.shape[1], :]
        return x


# ======== Feed Forward ========
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        # Using x @ W1, where x: (..., d_model), W1: (d_model, d_ff)
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)


    def forward(self, x):
        h1 = self.linear1.forward(x)
        h2 = self.relu.forward(h1)
        out = self.linear2.forward(h2)

        return out
    
    def backward(self, d_out):
        d_h2 = self.linear2.backward(d_out)
        d_h1 = self.relu.backward(d_h2)
        d_x  = self.linear1.backward(d_h1)

        return d_x
    
    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()

    def update(self, lr):
        self.linear1.update(lr)
        self.linear2.update(lr)


# ======== Encoder ========
class Encoder:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: same semantics as in MultiheadAttention
        """
        # Self-attention + residual + LN
        x_hat, _ = self.self_attention.forward(x, x, x, mask)
        x = layer_norm(x + x_hat)

        # Feed-forward + residual + LN
        x_hat = self.ff.forward(x)
        x = layer_norm(x + x_hat)

        return x


# ======== Decoder ========
class Decoder:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.encoder_decoder_attention = MultiheadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        x: (batch_size, tgt_seq_len, d_model)
        encoder_output: (batch_size, src_seq_len, d_model)

        src_mask:
            - mask over encoder positions (for encoder-decoder attention)

        tgt_mask:
            - mask over decoder positions (for causal/self-attention)
            - If None, a causal mask is created so that position i cannot
              attend to positions > i.
        """
        batch_size, tgt_seq_len, _ = x.shape

        # If no target mask is provided, create a causal one
        if tgt_mask is None:
            tgt_mask = create_causal_mask(tgt_seq_len, batch_size, self.self_attention.n_heads)

        # 1) Decoder self-attention (with causal mask)
        x_hat, _ = self.self_attention.forward(x, x, x, tgt_mask)
        x = layer_norm(x + x_hat)

        # 2) Encoder-decoder cross attention
        x_hat, attention_weights = self.encoder_decoder_attention.forward(
            x, encoder_output, encoder_output, src_mask
        )
        x = layer_norm(x + x_hat)

        # 3) Feed-forward + residual + LN
        x_hat = self.ff.forward(x)
        x = layer_norm(x + x_hat)

        return x, attention_weights
    
# ======== Linear ========
class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = np.random.randn(in_dim, out_dim)/ np.sqrt(in_dim)
        self.b = np.zeros((out_dim,))

        self.dW = np.zeros((in_dim, out_dim))
        self.db = np.zeros((out_dim,))

        self.cache = {}
        
    def forward(self, x):
        self.cache["x"] = x
        y = x @ self.W + self.b

        return y
    
    def backward(self, dy):
        x = self.cache["x"]

        self.dW = np.tensordot(x, dy, axes=((0,1),(0,1)))
        self.db = np.sum(dy, axis=(0,1))
        dx = dy @ self.W.T

        return dx
    
# ======== ReLU ========
    
class ReLU:
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        # store mask for backward
        mask = (x > 0)
        self.cache["mask"] = mask
        y = np.maximum(0, x)
        return y

    def backward(self, dy):
        mask = self.cache["mask"]
        dx = dy * mask
        return dx

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        self.cache = {}

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * inv_std
        y = self.gamma * x_hat + self.beta

        self.cache["x_hat"] = x_hat
        self.cache["mean"] = mean
        self.cache["var"] = var
        self.cache["inv_std"] = inv_std
        return y 

    def backward(self, dy):
        x_hat = self.cache["x_hat"]
        inv_std = self.cache["inv_std"]

        # 1. dgamma, dbeta
        self.dgamma = np.sum(dy * x_hat, axis=(0,1))
        self.dbeta  = np.sum(dy, axis=(0,1))

        # 2. dx_hat
        dx_hat = dy * self.gamma

        # 3. dx using the LN backward formula
        D = x_hat.shape[-1]
        sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
        sum_dx_hat_xhat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)

        dx = (1.0 / D) * inv_std * (
            D * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_xhat
        )

        return dx


    def zero_grad(self):
        self.dgamma[...] = 0
        self.dbeta[...] = 0

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta
