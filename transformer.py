import numpy as np

# ======== Utility ========
def softmax(x):
    # Numerically stable softmax along last axis
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

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

def cross_entropy_loss(logits, target_ids):
    B, T, V = logits.shape

    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits_stable = logits - logits_max
    exp_logits = np.exp(logits_stable)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp

    log_probs = logits_stable - np.log(sum_exp)

    batch_idx = np.arange(B)[:, None]
    time_idx = np.arange(T)[None, :]

    log_probs_target = log_probs[batch_idx, time_idx, target_ids]

    loss = -np.mean(log_probs_target)

    d_logits = probs.copy()
    d_logits[batch_idx, time_idx, target_ids] -= 1.0
    d_logits /= (B * T)

    return loss, d_logits



# ======== Multihead Attention ========
class MultiheadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        self.bq = np.zeros((d_model,))
        self.bk = np.zeros((d_model,))
        self.bv = np.zeros((d_model,))
        self.bo = np.zeros((d_model,))

        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)

        self.dbq = np.zeros_like(self.bq)
        self.dbk = np.zeros_like(self.bk)
        self.dbv = np.zeros_like(self.bv)
        self.dbo = np.zeros_like(self.bo)


    def split_heads(self, x):
        """
        x: (batch_size, seq_len, d_model)
        return: (batch_size, n_heads, seq_len, depth)
        """
        x = np.reshape(x, (x.shape[0], -1, self.n_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, query, key, value, mask=None):

        # Linear projections + bias
        Q = np.dot(query, self.Wq) + self.bq
        K = np.dot(key, self.Wk) + self.bk
        V = np.dot(value, self.Wv) + self.bv

        Q_raw = Q.copy()
        K_raw = K.copy()
        V_raw = V.copy()

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
            if mask.dtype != bool:
                mask = mask.astype(bool)

            if mask.shape != score.shape:
                mask = np.broadcast_to(mask, score.shape)
            mask_bc = mask  
        else:
            mask_bc = None

        if mask_bc is not None:
            score = np.where(mask_bc, score, -1e9)

        attention_weights = softmax(score)
        context_split = np.matmul(attention_weights, V)  # (batch, heads, seq_q, depth)
        context = np.transpose(context_split, (0, 2, 1, 3))  # (batch, seq_q, heads, depth)
        context = np.reshape(context, (context.shape[0], -1, self.d_model))  # (batch, seq_q, d_model)

        out = np.dot(context, self.Wo) + self.bo  # (batch, seq_q, d_model)

        batch_size = query.shape[0]
        seq_len = query.shape[1]

        self.cache = {
            "query": query,
            "key": key,
            "value": value,
            "Q_raw": Q_raw,
            "K_raw": K_raw,
            "V_raw": V_raw,
            "Q": Q,
            "K": K,
            "V": V,
            "score": score,
            "mask": mask_bc,
            "attention": attention_weights,
            "context_split": context_split,
            "context": context,
            "shape": (batch_size, seq_len)
        }
        return out, attention_weights
    
    def backward(self, d_out):
        cache = self.cache
        query  = cache["query"]
        key    = cache["key"]
        value  = cache["value"]

        Q_raw  = cache["Q_raw"]
        K_raw  = cache["K_raw"]
        V_raw  = cache["V_raw"]

        Q      = cache["Q"]
        K      = cache["K"]
        V      = cache["V"]

        score  = cache["score"]
        mask   = cache["mask"]
        attn   = cache["attention"]

        context_split = cache["context_split"]  # (B, H, T, depth)
        context       = cache["context"]        # (B, T, d_model)

        batch_size, seq_len = cache["shape"]
        depth = self.depth
        H = self.n_heads

        self.d_context = d_out @ self.Wo.T
        self.dWo = np.tensordot(context, d_out, axes=([0,1], [0,1]))
        self.dbo = np.sum(d_out, axis=(0,1))

        d_context_reshaped = self.d_context.reshape(batch_size, seq_len, H, depth)
        d_context_split = np.transpose(d_context_reshaped, (0, 2, 1, 3))
        self.d_context_split = d_context_split

        self.d_attn = np.matmul(d_context_split, np.transpose(V, (0, 1, 3, 2)))
        self.d_V_split = np.matmul(np.transpose(attn, (0, 1, 3, 2)), d_context_split)

        s = np.sum(self.d_attn * attn, axis=-1, keepdims=True) 
        d_score = attn * (self.d_attn - s)   
        if mask is not None:
            d_score = np.where(mask, d_score, 0.0)
        self.d_score = d_score

        scale = 1.0 / np.sqrt(depth)
        d_Q_split = np.matmul(d_score, K) * scale
        d_K_split = np.matmul(np.transpose(d_score, (0,1,3,2)), Q) * scale
        self.d_Q_split = d_Q_split
        self.d_K_split = d_K_split

        d_Q_transposed = np.transpose(d_Q_split, (0, 2, 1, 3)) 
        d_Q_raw = d_Q_transposed.reshape(batch_size, seq_len, self.d_model)
        d_K_transposed = np.transpose(d_K_split, (0, 2, 1, 3)) 
        d_K_raw = d_K_transposed.reshape(batch_size, seq_len, self.d_model)
        d_V_transposed = np.transpose(self.d_V_split, (0, 2, 1, 3))  
        d_V_raw = d_V_transposed.reshape(batch_size, seq_len, self.d_model)

        self.d_Q_raw = d_Q_raw
        self.d_K_raw = d_K_raw
        self.d_V_raw = d_V_raw

        self.dWq = np.tensordot(query, d_Q_raw, axes=([0,1],[0,1]))
        self.dbq = np.sum(d_Q_raw, axis=(0,1))
        d_query_from_q = d_Q_raw @ self.Wq.T

        self.dWk = np.tensordot(key, d_K_raw, axes=([0,1],[0,1]))
        self.dbk = np.sum(d_K_raw, axis=(0,1))
        d_key_from_k = d_K_raw @ self.Wk.T

        self.dWv = np.tensordot(value, d_V_raw, axes=([0,1],[0,1]))
        self.dbv = np.sum(d_V_raw, axis=(0,1))
        d_value_from_v = d_V_raw @ self.Wv.T

        return d_query_from_q, d_key_from_k, d_value_from_v
    
    def zero_grad(self):
        self.dWq[...] = 0
        self.dbq[...] = 0
        self.dWk[...] = 0
        self.dbk[...] = 0
        self.dWv[...] = 0
        self.dbv[...] = 0
        self.dWo[...] = 0
        self.dbo[...] = 0

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.bq -= lr * self.dbq
        self.Wk -= lr * self.dWk
        self.bk -= lr * self.dbk
        self.Wv -= lr * self.dWv
        self.bv -= lr * self.dbv
        self.Wo -= lr * self.dWo
        self.bo -= lr * self.dbo


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
    
    def backward(self, d_out):
        return d_out


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

        # 1) Self-attention block
        self.self_attention = MultiheadAttention(d_model, n_heads)

        # 2) Feed-forward block
        self.ff = FeedForward(d_model, d_ff)

        # 3) Two LayerNorms: one after self-attention, one after FFN
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        # (later we’ll add: maybe caches for residuals if you want)

    def forward(self, x, mask=None):
        """
        x:   (batch_size, seq_len, d_model)
        mask: same semantics as in MultiheadAttention
        """
        sa_out, _ = self.self_attention.forward(x, x, x, mask)
        x_sa = x + sa_out 
        y1 = self.ln1.forward(x_sa)

        ff_out = self.ff.forward(y1)
        y1_ff = y1 + ff_out
        y2 = self.ln2.forward(y1_ff)

        self.cache = {
            "x": x,
            "x_sa": x_sa,
            "y1": y1,
            "y1_ff": y1_ff,
        }

        return y2

    def backward(self, d_out):
        cache = self.cache

        dy2 = d_out
        dy1_ff = self.ln2.backward(dy2)

        d_y1_from_residual2 = dy1_ff
        d_ff_out = dy1_ff

        dy1_from_ff = self.ff.backward(d_ff_out)
        dy1 = d_y1_from_residual2 + dy1_from_ff

        dx_sa = self.ln1.backward(dy1)

        d_x_from_residual = dx_sa
        d_sa_out = dx_sa

        dQ, dK, dV = self.self_attention.backward(d_sa_out)
        dx = d_x_from_residual + dQ + dK + dV
        return dx
    
    def zero_grad(self):
        self.self_attention.zero_grad()
        self.ff.zero_grad()
        self.ln1.zero_grad()
        self.ln2.zero_grad()

    def update(self, lr):
        self.self_attention.update(lr)
        self.ff.update(lr)
        self.ln1.update(lr)
        self.ln2.update(lr)




# ======== Decoder ========
class Decoder:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)

        self.cross_attention = MultiheadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.ln3 = LayerNorm(d_model)


    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # ----- Block 1: masked self-attention -----
        sa_out, _ = self.self_attention.forward(x, x, x, tgt_mask)
        x_sa = x + sa_out
        y1 = self.ln1.forward(x_sa)

        # ----- Block 2: cross-attention (query = decoder, key/value = encoder) -----
        ca_out, attn = self.cross_attention.forward(y1, enc_out, enc_out, src_mask)
        y1_ca = y1 + ca_out
        y2 = self.ln2.forward(y1_ca)

        # ----- Block 3: FFN -----
        ff_out = self.ff.forward(y2)
        y2_ff = y2 + ff_out
        y3 = self.ln3.forward(y2_ff)

        # Cache for backward
        self.cache = {
            "x": x,
            "x_sa": x_sa,
            "y1": y1,
            "y1_ca": y1_ca,
            "y2": y2,
            "y2_ff": y2_ff,
            "enc_out": enc_out,
            "attn": attn,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
        }

        return y3, attn
    
    def backward(self, d_out):
        dy2_ff = self.ln3.backward(d_out)

        d_y2_from_residual3 = dy2_ff
        d_ff_out = dy2_ff

        dy2_from_ff = self.ff.backward(d_ff_out)

        dy2 = d_y2_from_residual3 + dy2_from_ff

        dy1_ca = self.ln2.backward(dy2)

        d_y1_from_residual2 = dy1_ca
        d_ca_out = dy1_ca

        dQ, dK, dV = self.cross_attention.backward(d_ca_out)

        dy1 = d_y1_from_residual2 + dQ

        d_enc_from_cross = dK + dV

        dx_sa = self.ln1.backward(dy1)

        d_x_from_residual1 = dx_sa
        d_sa_out = dx_sa

        dQ, dK, dV = self.self_attention.backward(d_sa_out)
        dx = d_x_from_residual1 + dQ + dK + dV

        return dx, d_enc_from_cross
    
    def zero_grad(self):
        self.self_attention.zero_grad()
        self.cross_attention.zero_grad()
        self.ff.zero_grad()
        self.ln1.zero_grad()
        self.ln2.zero_grad()
        self.ln3.zero_grad()

    def update(self, lr):
        self.self_attention.update(lr)
        self.cross_attention.update(lr)
        self.ff.update(lr)
        self.ln1.update(lr)
        self.ln2.update(lr)
        self.ln3.update(lr)




    
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

        x2 = x.reshape(-1, self.in_dim)
        dy2 = dy.reshape(-1, self.out_dim)

        self.dW = x2.T @ dy2
        self.db = dy2.sum(axis=0)

        dx = dy @ self.W.T

        return dx

    
    def zero_grad(self):
        self.dW[...] = 0
        self.db[...] = 0

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
    
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

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.dW = np.zeros_like(self.W)

    def forward(self, token_ids):
        self.cache = {"token_ids": token_ids}
        return self.W[token_ids]
    
    def backward(self, d_out):
        token_ids = self.cache["token_ids"]
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, token_ids, d_out)
        return None
    
    def zero_grad(self):
        self.dW[...] = 0

    def update(self, lr):
        self.W -= lr * self.dW

class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, max_len=128):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len

        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.encoder = Encoder(d_model, n_heads, d_ff)
        self.decoder = Decoder(d_model, n_heads, d_ff)

        self.output_projection = Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        src_emb = self.src_embedding.forward(src_ids)
        src_emb = self.pos_encoding.forward(src_emb)
        enc_out = self.encoder.forward(src_emb, src_mask)

        tgt_emb = self.tgt_embedding.forward(tgt_ids)
        tgt_emb = self.pos_encoding.forward(tgt_emb)
        dec_out, attn = self.decoder.forward(tgt_emb, enc_out, src_mask, tgt_mask)

        logits = self.output_projection.forward(dec_out)

        return logits
    
    def backward(self, d_logits):
        d_dec_out = self.output_projection.backward(d_logits)
        d_tgt_emb, d_enc_from_dec = self.decoder.backward(d_dec_out)
        d_src_emb = self.encoder.backward(d_enc_from_dec)

        d_tgt_emb = self.pos_encoding.backward(d_tgt_emb)
        d_src_emb = self.pos_encoding.backward(d_src_emb)

        self.tgt_embedding.backward(d_tgt_emb)
        self.src_embedding.backward(d_src_emb)

    def zero_grad(self):
        self.src_embedding.zero_grad()
        self.tgt_embedding.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.output_projection.zero_grad()

    def update(self, lr):
        self.src_embedding.update(lr)
        self.tgt_embedding.update(lr)
        self.encoder.update(lr)
        self.decoder.update(lr)
        self.output_projection.update(lr)


