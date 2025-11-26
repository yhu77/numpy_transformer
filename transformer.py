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


# ======== Tests ========

def test_softmax():
    """Test softmax computation"""
    print("Testing Softmax...")
    
    # Test 2D case
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = softmax(x)
    
    # Check if sums to 1
    assert np.allclose(np.sum(result, axis=-1), 1.0), "Softmax should sum to 1"
    
    # Check if all values are positive
    assert np.all(result > 0), "Softmax values should be positive"
    
    print("✓ Softmax tests passed")


def test_multihead_attention():
    """Test multi-head attention mechanism"""
    print("\nTesting Multi-head Attention...")
    
    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2
    
    # Create instance
    mha = MultiheadAttention(d_model, n_heads)
    
    # Create dummy input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = mha.forward(x, x, x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check attention weights shape
    assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len), \
        f"Expected attention shape {(batch_size, n_heads, seq_len, seq_len)}, got {attention_weights.shape}"
    
    # Check attention weights sum to 1
    attention_sum = np.sum(attention_weights, axis=-1)
    assert np.allclose(attention_sum, 1.0), "Attention weights should sum to 1"
    
    print("✓ Multi-head attention tests passed")


def test_positional_encoding():
    """Test positional encoding"""
    print("\nTesting Positional Encoding...")
    
    d_model = 512
    max_len = 100
    
    pos_enc = PositionalEncoding(d_model, max_len)
    
    # Check encoding shape
    assert pos_enc.pos_encoding.shape == (1, max_len, d_model), \
        f"Expected shape {(1, max_len, d_model)}, got {pos_enc.pos_encoding.shape}"
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    output = pos_enc.forward(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    
    # Check that encoding is deterministic
    output2 = pos_enc.forward(x)
    assert np.allclose(output, output2), "Positional encoding should be deterministic"
    
    print("✓ Positional encoding tests passed")


def test_feedforward():
    """Test feed-forward network"""
    print("\nTesting Feed-Forward Network...")
    
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 32
    
    ff = FeedForward(d_model, d_ff)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    output = ff.forward(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check ReLU activation (no negative values in intermediate layer)
    x_intermediate = np.dot(x, ff.W1) + ff.b1
    x_activated = np.maximum(0, x_intermediate)
    assert np.all(x_activated >= 0), "ReLU should produce non-negative values"
    
    print("✓ Feed-forward tests passed")


def test_encoder():
    """Test encoder layer"""
    print("\nTesting Encoder...")
    
    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2
    d_ff = 32
    
    encoder = Encoder(d_model, n_heads, d_ff)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    mask = None  # or create a proper mask
    
    output = encoder.forward(x, mask)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    
    # Check that output is different from input (transformation occurred)
    assert not np.allclose(output, x), "Encoder should transform the input"
    
    print("✓ Encoder tests passed")


def test_decoder():
    """Test decoder layer"""
    print("\nTesting Decoder...")
    
    batch_size = 2
    src_seq_len = 4
    tgt_seq_len = 3
    d_model = 8
    n_heads = 2
    d_ff = 32
    
    decoder = Decoder(d_model, n_heads, d_ff)
    
    x = np.random.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = np.random.randn(batch_size, src_seq_len, d_model)
    
    src_mask = None
    # Explicitly use causal mask (though Decoder will also create one if None)
    tgt_mask = create_causal_mask(tgt_seq_len, batch_size, n_heads)
    
    output, attention_weights = decoder.forward(x, encoder_output, src_mask, tgt_mask)
    
    # Check output shape
    assert output.shape == (batch_size, tgt_seq_len, d_model), \
        f"Expected shape {(batch_size, tgt_seq_len, d_model)}, got {output.shape}"
    
    # Check attention weights shape (cross-attention: tgt x src)
    assert attention_weights.shape == (batch_size, n_heads, tgt_seq_len, src_seq_len), \
        f"Expected attention shape {(batch_size, n_heads, tgt_seq_len, src_seq_len)}"
    
    print("✓ Decoder tests passed")


def test_masking():
    """Test masking functionality"""
    print("\nTesting Masking...")
    
    batch_size = 1
    seq_len = 4
    d_model = 8
    n_heads = 2
    
    mha = MultiheadAttention(d_model, n_heads)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Create a causal mask (for autoregressive generation)
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask = mask[np.newaxis, np.newaxis, :, :]  # Add batch and head dimensions
    mask = mask.astype(bool)
    
    output, attention_weights = mha.forward(x, x, x, mask)
    
    # Check that masked positions (upper triangle) have ~zero attention
    for h in range(n_heads):
        upper_triangle_mask = 1 - np.tril(np.ones((seq_len, seq_len)))
        upper_triangle = attention_weights[0, h] * upper_triangle_mask
        assert np.allclose(upper_triangle, 0, atol=1e-6), "Masked positions should have zero attention"
    
    print("✓ Masking tests passed")


def test_gradient_flow():
    """Test that gradients can flow (simple numerical gradient check)"""
    print("\nTesting Gradient Flow...")
    
    batch_size = 1
    seq_len = 2
    d_model = 4
    n_heads = 2
    
    mha = MultiheadAttention(d_model, n_heads)
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Compute output
    output, _ = mha.forward(x, x, x)
    
    # Simple loss (sum of outputs)
    loss = np.sum(output)
    
    # Numerical gradient check for a single input element
    epsilon = 1e-5
    x_perturbed = x.copy()
    x_perturbed[0, 0, 0] += epsilon
    output_perturbed, _ = mha.forward(x_perturbed, x_perturbed, x_perturbed)
    loss_perturbed = np.sum(output_perturbed)
    
    numerical_gradient = (loss_perturbed - loss) / epsilon
    
    # We only check that the loss is sensitive to this parameter (non-zero gradient)
    assert abs(numerical_gradient) > 1e-10, "Gradient should be non-zero (gradient flow exists)"
    
    print("✓ Gradient flow test passed")

def test_full_pipeline():
    """Test a simple end-to-end pipeline"""
    print("\nTesting Full Pipeline...")
    
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4
    d_model = 16
    n_heads = 2
    d_ff = 64
    
    # Initialize components
    pos_enc = PositionalEncoding(d_model, max_len=100)
    encoder = Encoder(d_model, n_heads, d_ff)
    decoder = Decoder(d_model, n_heads, d_ff)
    
    # Create dummy data (e.g., embedded tokens)
    src = np.random.randn(batch_size, src_seq_len, d_model)
    tgt = np.random.randn(batch_size, tgt_seq_len, d_model)
    
    # Add positional encoding
    src = pos_enc.forward(src)
    tgt = pos_enc.forward(tgt)
    
    # Encode
    encoder_output = encoder.forward(src, mask=None)
    
    # Decode with default causal mask (tgt_mask=None)
    decoder_output, cross_attention = decoder.forward(tgt, encoder_output, None, None)
    
    assert decoder_output.shape == (batch_size, tgt_seq_len, d_model), \
        "Final output shape mismatch"
    
    print("✓ Full pipeline test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Transformer Tests")
    print("=" * 50)
    
    try:
        test_softmax()
        test_multihead_attention()
        test_positional_encoding()
        test_feedforward()
        test_encoder()
        test_decoder()
        test_masking()
        test_gradient_flow()
        test_full_pipeline()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
