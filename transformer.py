import numpy as np

# ============================================================
# Utility
# ============================================================

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):
    """
    Lower-triangular causal mask of shape (1, 1, T, T),
    where True = allowed, False = masked.
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask[np.newaxis, np.newaxis, :, :]


def cross_entropy_loss(logits, target_ids, ignore_index=None):
    """
    logits: (B, T, V)
    target_ids: (B, T)
    ignore_index: token id (e.g., PAD) to ignore in loss and grad
    """
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

    if ignore_index is not None:
        mask = (target_ids != ignore_index)  # True for valid tokens
        valid_count = np.sum(mask)
        if valid_count == 0:
            # No valid tokens, avoid divide-by-zero
            loss = 0.0
            d_logits = np.zeros_like(logits)
            return loss, d_logits

        loss = -np.sum(log_probs_target[mask]) / valid_count
    else:
        mask = None
        loss = -np.mean(log_probs_target)

    d_logits = probs.copy()
    d_logits[batch_idx, time_idx, target_ids] -= 1.0

    if mask is not None:
        d_logits[~mask] = 0.0  # zero gradient on ignored positions
        d_logits /= np.sum(mask)
    else:
        d_logits /= (B * T)

    return loss, d_logits


# ============================================================
# Multihead Attention
# ============================================================

class MultiheadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        # Parameters
        self.Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        self.bq = np.zeros((d_model,))
        self.bk = np.zeros((d_model,))
        self.bv = np.zeros((d_model,))
        self.bo = np.zeros((d_model,))

        # Gradients
        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)

        self.dbq = np.zeros_like(self.bq)
        self.dbk = np.zeros_like(self.bk)
        self.dbv = np.zeros_like(self.bv)
        self.dbo = np.zeros_like(self.bo)

        self.cache = {}

    def split_heads(self, x):
        # x: (B, T, d_model) -> (B, H, T, depth)
        B, T, _ = x.shape
        x = np.reshape(x, (B, T, self.n_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, query, key, value, mask=None):
        # Linear projections
        Q = np.dot(query, self.Wq) + self.bq
        K = np.dot(key, self.Wk) + self.bk
        V = np.dot(value, self.Wv) + self.bv

        Q_raw = Q.copy()
        K_raw = K.copy()
        V_raw = V.copy()

        # Split heads
        Q = self.split_heads(Q)  # (B, H, T_q, depth)
        K = self.split_heads(K)  # (B, H, T_k, depth)
        V = self.split_heads(V)  # (B, H, T_k, depth)

        dk = K.shape[-1]
        score = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) / np.sqrt(dk)  # (B, H, T_q, T_k)

        # Mask: True = keep, False = mask out
        if mask is not None:
            if mask.dtype != bool:
                mask = mask.astype(bool)
            if mask.shape != score.shape:
                mask = np.broadcast_to(mask, score.shape)
            mask_bc = mask
            score = np.where(mask_bc, score, -1e9)
        else:
            mask_bc = None

        attention_weights = softmax(score)               # (B, H, T_q, T_k)
        context_split = np.matmul(attention_weights, V)  # (B, H, T_q, depth)

        # Merge heads
        context = np.transpose(context_split, (0, 2, 1, 3))  # (B, T_q, H, depth)
        B, T_q, H, D = context.shape
        context = np.reshape(context, (B, T_q, self.d_model))  # (B, T_q, d_model)

        out = np.dot(context, self.Wo) + self.bo

        batch_size = query.shape[0]
        T_q = query.shape[1]
        T_k = key.shape[1]

        self.cache = {
            "query": query, "key": key, "value": value,
            "Q_raw": Q_raw, "K_raw": K_raw, "V_raw": V_raw,
            "Q": Q, "K": K, "V": V,
            "score": score, "mask": mask_bc,
            "attention": attention_weights,
            "context_split": context_split,
            "context": context,
            "shape": {
                "batch_size": batch_size,
                "T_q": T_q,
                "T_k": T_k
            },
        }
        return out, attention_weights

    def backward(self, d_out):
        cache = self.cache
        query, key, value = cache["query"], cache["key"], cache["value"]
        Q, K, V = cache["Q"], cache["K"], cache["V"]
        score, mask, attn = cache["score"], cache["mask"], cache["attention"]
        context_split, context = cache["context_split"], cache["context"]
        shape = cache["shape"]
        batch_size = shape["batch_size"]
        T_q = shape["T_q"]
        T_k = shape["T_k"]
        depth = self.depth
        H = self.n_heads

        # dWo, dbo
        self.dWo = np.tensordot(context, d_out, axes=([0, 1], [0, 1]))
        self.dbo = np.sum(d_out, axis=(0, 1))
        d_context = d_out @ self.Wo.T  # (B, T_q, d_model)

        # Reshape back to (B, H, T_q, depth)
        d_context_reshaped = d_context.reshape(batch_size, T_q, H, depth)
        d_context_split = np.transpose(d_context_reshaped, (0, 2, 1, 3))

        # Gradients w.r.t. attention and V
        d_attn = np.matmul(d_context_split, np.transpose(V, (0, 1, 3, 2)))
        d_V_split = np.matmul(np.transpose(attn, (0, 1, 3, 2)), d_context_split)

        # Softmax backward
        s = np.sum(d_attn * attn, axis=-1, keepdims=True)
        d_score = attn * (d_attn - s)
        if mask is not None:
            d_score = np.where(mask, d_score, 0.0)

        scale = 1.0 / np.sqrt(depth)
        d_Q_split = np.matmul(d_score, K) * scale
        d_K_split = np.matmul(np.transpose(d_score, (0, 1, 3, 2)), Q) * scale

        # Merge heads back to (B, T_q, d_model) for Q
        d_Q_raw = (
            np.transpose(d_Q_split, (0, 2, 1, 3))
            .reshape(batch_size, T_q, self.d_model)
        )

        # Merge heads back to (B, T_k, d_model) for K and V
        d_K_raw = (
            np.transpose(d_K_split, (0, 2, 1, 3))
            .reshape(batch_size, T_k, self.d_model)
        )

        d_V_raw = (
            np.transpose(d_V_split, (0, 2, 1, 3))
            .reshape(batch_size, T_k, self.d_model)
        )

        # Gradients for Q branch
        self.dWq = np.tensordot(query, d_Q_raw, axes=([0, 1], [0, 1]))
        self.dbq = np.sum(d_Q_raw, axis=(0, 1))
        d_query_from_q = d_Q_raw @ self.Wq.T

        # Gradients for K branch
        self.dWk = np.tensordot(key, d_K_raw, axes=([0, 1], [0, 1]))
        self.dbk = np.sum(d_K_raw, axis=(0, 1))
        d_key_from_k = d_K_raw @ self.Wk.T

        # Gradients for V branch
        self.dWv = np.tensordot(value, d_V_raw, axes=([0, 1], [0, 1]))
        self.dbv = np.sum(d_V_raw, axis=(0, 1))
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


# ============================================================
# Positional Encoding
# ============================================================

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
        x = x + self.pos_encoding[:, :x.shape[1], :]
        return x

    def backward(self, d_out):
        # No trainable parameters, gradient passes through
        return d_out


# ============================================================
# Linear
# ============================================================

class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
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


# ============================================================
# ReLU
# ============================================================

class ReLU:
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        mask = (x > 0)
        self.cache["mask"] = mask
        y = np.maximum(0, x)
        return y

    def backward(self, dy):
        mask = self.cache["mask"]
        dx = dy * mask
        return dx


# ============================================================
# LayerNorm
# ============================================================

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

        self.dgamma = np.sum(dy * x_hat, axis=(0, 1))
        self.dbeta = np.sum(dy, axis=(0, 1))

        dx_hat = dy * self.gamma

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
        self.beta -= lr * self.dbeta


# ============================================================
# Feed Forward
# ============================================================

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
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
        d_x = self.linear1.backward(d_h1)
        return d_x

    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()

    def update(self, lr):
        self.linear1.update(lr)
        self.linear2.update(lr)


# ============================================================
# Encoder
# ============================================================

class Encoder:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.cache = {}

    def forward(self, x, src_mask=None):
        sa_out, _ = self.self_attention.forward(x, x, x, src_mask)
        x_sa = x + sa_out
        y1 = self.ln1.forward(x_sa)

        ff_out = self.ff.forward(y1)
        y1_ff = y1 + ff_out
        y2 = self.ln2.forward(y1_ff)

        self.cache = {
            "x": x, "x_sa": x_sa,
            "y1": y1, "y1_ff": y1_ff,
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


# ============================================================
# Decoder
# ============================================================

class Decoder:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.cross_attention = MultiheadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln3 = LayerNorm(d_model)
        self.cache = {}

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        sa_out, _ = self.self_attention.forward(x, x, x, tgt_mask)
        x_sa = x + sa_out
        y1 = self.ln1.forward(x_sa)

        ca_out, attn = self.cross_attention.forward(y1, enc_out, enc_out, src_mask)
        y1_ca = y1 + ca_out
        y2 = self.ln2.forward(y1_ca)

        ff_out = self.ff.forward(y2)
        y2_ff = y2 + ff_out
        y3 = self.ln3.forward(y2_ff)

        self.cache = {
            "x": x, "x_sa": x_sa, "y1": y1,
            "y1_ca": y1_ca, "y2": y2, "y2_ff": y2_ff,
            "enc_out": enc_out, "attn": attn,
            "src_mask": src_mask, "tgt_mask": tgt_mask,
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


# ============================================================
# Embedding
# ============================================================

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.dW = np.zeros_like(self.W)
        self.cache = {}

    def forward(self, token_ids):
        self.cache["token_ids"] = token_ids
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


# ============================================================
# Transformer
# ============================================================

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
        dec_out, _ = self.decoder.forward(tgt_emb, enc_out, src_mask, tgt_mask)

        logits = self.output_projection.forward(dec_out)
        return logits

    def backward(self, d_logits):
        d_dec_out = self.output_projection.backward(d_logits)
        d_tgt_emb, d_enc_back = self.decoder.backward(d_dec_out)
        d_src_emb = self.encoder.backward(d_enc_back)

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


# ============================================================
# Arithmetic Vocabulary
# ============================================================

class ArithmeticVocab:
    def __init__(self):
        self.tokens = ['0','1','2','3','4','5','6','7','8','9',
                       '+','-','*','=','<PAD>','<SOS>','<EOS>']
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for i, t in enumerate(self.tokens)}
        self.PAD = self.token2id['<PAD>']
        self.SOS = self.token2id['<SOS>']
        self.EOS = self.token2id['<EOS>']

    def encode(self, text):
        return [self.token2id[c] for c in text]

    def decode(self, ids):
        return ''.join(self.id2token[i] for i in ids)


# ============================================================
# Batch Preparation
# ============================================================

def prepare_batch(data_batch, vocab, max_src_len=10, max_tgt_len=6):
    """
    data_batch: list of (src_str, tgt_str)
    max_src_len: max tokens in source (e.g., "99+99" <- 5)
    max_tgt_len: max tokens in target including SOS & EOS
    """
    batch_size = len(data_batch)

    src_batch = np.full((batch_size, max_src_len), vocab.PAD, dtype=np.int32)
    tgt_batch = np.full((batch_size, max_tgt_len), vocab.PAD, dtype=np.int32)

    for i, (src, tgt) in enumerate(data_batch):
        src_ids = vocab.encode(src)
        tgt_ids = [vocab.SOS] + vocab.encode(tgt) + [vocab.EOS]

        src_ids = src_ids[:max_src_len]
        tgt_ids = tgt_ids[:max_tgt_len]

        src_batch[i, :len(src_ids)] = src_ids
        tgt_batch[i, :len(tgt_ids)] = tgt_ids

    return src_batch, tgt_batch


# ============================================================
# Generation
# ============================================================

def generate(model, src_ids, vocab, max_len=6):
    """
    Greedy decoding.
    src_ids: (B, T_src)
    Returns: decoded token ids (including SOS/EOS)
    """
    batch_size, src_len = src_ids.shape

    # Source mask: True where not PAD
    src_key_mask = (src_ids != vocab.PAD)[:, None, None, :]  # (B, 1, 1, T_src)

    src_emb = model.src_embedding.forward(src_ids)
    src_emb = model.pos_encoding.forward(src_emb)
    enc_out = model.encoder.forward(src_emb, src_mask=src_key_mask)

    # Start with SOS
    tgt_ids = np.full((batch_size, 1), vocab.SOS, dtype=np.int32)

    for _ in range(max_len - 1):
        T = tgt_ids.shape[1]
        causal_mask = create_causal_mask(T)              # (1, 1, T, T)
        tgt_key_mask = (tgt_ids != vocab.PAD)[:, None, None, :]  # (B, 1, 1, T)
        tgt_mask = causal_mask & tgt_key_mask            # (B, 1, T, T)

        tgt_emb = model.tgt_embedding.forward(tgt_ids)
        tgt_emb = model.pos_encoding.forward(tgt_emb)
        dec_out, _ = model.decoder.forward(tgt_emb, enc_out,
                                           src_mask=src_key_mask,
                                           tgt_mask=tgt_mask)
        logits = model.output_projection.forward(dec_out)

        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tgt_ids = np.concatenate([tgt_ids, next_token], axis=1)

        # Stop early if all ended
        if np.all(next_token == vocab.EOS):
            break

    return tgt_ids


# ============================================================
# Dataset Generation
# ============================================================

def generate_addition_dataset(n_samples, max_value=99, seed=0):
    """
    Generate random "a+b" -> "a+b result" pairs,
    where a, b in [0, max_value].
    """
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_samples):
        a = int(rng.integers(0, max_value + 1))
        b = int(rng.integers(0, max_value + 1))
        src = f"{a}+{b}"
        tgt = f"{a+b}"
        data.append((src, tgt))
    return data


# ============================================================
# Training Loop
# ============================================================

def train_arithmetic():
    np.random.seed(0)

    print("=" * 60)
    print("Training Transformer on Random Addition Dataset (0–99)")
    print("=" * 60)

    # Hyperparameters
    d_model = 32
    n_heads = 2
    d_ff = 64
    batch_size = 64
    n_epochs = 3000
    lr = 0.01
    max_src_len = 10
    max_tgt_len = 6

    # Dataset
    vocab = ArithmeticVocab()
    full_data = generate_addition_dataset(n_samples=5000, max_value=99, seed=42)
    np.random.shuffle(full_data)

    split = int(0.8 * len(full_data))
    train_data = full_data[:split]
    test_data = full_data[split:]

    print(f"Total problems:   {len(full_data)}")
    print(f"Training size:    {len(train_data)}")
    print(f"Testing size:     {len(test_data)}")
    print(f"Vocabulary size:  {len(vocab.tokens)}")

    # Model
    vocab_size = len(vocab.tokens)
    model = Transformer(vocab_size, vocab_size, d_model, n_heads, d_ff, max_len=20)

    num_batches = len(train_data) // batch_size

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(train_data)
        epoch_loss = 0.0

        for b in range(num_batches):
            batch = train_data[b * batch_size:(b + 1) * batch_size]
            src_batch, tgt_batch = prepare_batch(batch, vocab, max_src_len, max_tgt_len)

            # Masks
            src_key_mask = (src_batch != vocab.PAD)[:, None, None, :]  # (B, 1, 1, S)
            T = tgt_batch.shape[1]
            causal_mask = create_causal_mask(T)                        # (1, 1, T, T)
            tgt_key_mask = (tgt_batch != vocab.PAD)[:, None, None, :]  # (B, 1, 1, T)
            tgt_mask = causal_mask & tgt_key_mask                      # (B, 1, T, T)

            # Forward
            logits = model.forward(src_batch, tgt_batch,
                                   src_mask=src_key_mask,
                                   tgt_mask=tgt_mask)

            # Loss on shifted targets, ignore PAD
            loss, d_logits = cross_entropy_loss(
                logits[:, :-1, :],
                tgt_batch[:, 1:],
                ignore_index=vocab.PAD
            )

            # Backward
            model.zero_grad()
            full_dlogits = np.zeros_like(logits)
            full_dlogits[:, :-1, :] = d_logits
            model.backward(full_dlogits)

            # Update
            model.update(lr)

            epoch_loss += loss

        avg_loss = epoch_loss / max(1, num_batches)

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss = {avg_loss:.4f}")

        # Full evaluation every 100 epochs
        if epoch % 100 == 0:
            print("\n" + "=" * 60)
            print(f"Full Evaluation @ Epoch {epoch}")
            print("=" * 60)
            correct = 0
            for src, tgt in test_data[:20]:  # print first 20 examples
                src_batch, _ = prepare_batch([(src, tgt)], vocab, max_src_len, max_tgt_len)
                gen = generate(model, src_batch, vocab, max_len=max_tgt_len)

                gen_ids = gen[0, 1:]  # drop SOS
                eos = np.where(gen_ids == vocab.EOS)[0]
                if len(eos) > 0:
                    gen_ids = gen_ids[:eos[0]]

                pred = vocab.decode(gen_ids)
                print(f"{src} = {tgt} | Predicted: {pred}")

                if pred == tgt:
                    correct += 1

            # Accuracy on whole test set
            total_correct = 0
            for src, tgt in test_data:
                src_batch, _ = prepare_batch([(src, tgt)], vocab, max_src_len, max_tgt_len)
                gen = generate(model, src_batch, vocab, max_len=max_tgt_len)

                gen_ids = gen[0, 1:]
                eos = np.where(gen_ids == vocab.EOS)[0]
                if len(eos) > 0:
                    gen_ids = gen_ids[:eos[0]]
                pred = vocab.decode(gen_ids)
                if pred == tgt:
                    total_correct += 1

            acc = total_correct / len(test_data) * 100
            print(f"\nTest Accuracy (full test set): {total_correct}/{len(test_data)} = {acc:.2f}%")
            print("=" * 60 + "\n")

    # Final evaluation at the very end
    print("\n" + "=" * 60)
    print("Final Evaluation (first 20 test examples)")
    print("=" * 60)
    correct_display = 0
    for src, tgt in test_data[:20]:
        src_batch, _ = prepare_batch([(src, tgt)], vocab, max_src_len, max_tgt_len)
        gen = generate(model, src_batch, vocab, max_len=max_tgt_len)

        gen_ids = gen[0, 1:]
        eos = np.where(gen_ids == vocab.EOS)[0]
        if len(eos) > 0:
            gen_ids = gen_ids[:eos[0]]
        pred = vocab.decode(gen_ids)
        print(f"{src} = {tgt} | Predicted: {pred}")
        if pred == tgt:
            correct_display += 1

    total_correct = 0
    for src, tgt in test_data:
        src_batch, _ = prepare_batch([(src, tgt)], vocab, max_src_len, max_tgt_len)
        gen = generate(model, src_batch, vocab, max_len=max_tgt_len)

        gen_ids = gen[0, 1:]
        eos = np.where(gen_ids == vocab.EOS)[0]
        if len(eos) > 0:
            gen_ids = gen_ids[:eos[0]]
        pred = vocab.decode(gen_ids)
        if pred == tgt:
            total_correct += 1

    acc = total_correct / len(test_data) * 100
    print(f"\nFinal Test Accuracy: {total_correct}/{len(test_data)} = {acc:.2f}%")

    return model, vocab


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    train_arithmetic()
