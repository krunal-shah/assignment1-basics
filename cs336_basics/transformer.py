from einops import einsum, rearrange
import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, dtype=None, device=None):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        variance = 2 / (in_features + out_features)
        stddev = variance**0.5
        torch.nn.init.trunc_normal_(self.weights, std=stddev, a=-3 * stddev, b=3 * stddev)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.weights, "... din, dout din -> ... dout")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=None, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
        torch.nn.init.trunc_normal_(self.weights, std=1.0, a=-3, b=3)

    def forward(self, x: torch.Tensor):
        return self.weights[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones((d_model), dtype=dtype, device=device))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = ((x**2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        return (x * self.weights / rms).to(dtype=x_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, dtype=None, device=None):
        super().__init__()
        d_ff = (8 * d_model) // 3 if not d_ff else d_ff
        self.w1 = Linear(d_model, d_ff, dtype, device)
        self.w2 = Linear(d_ff, d_model, dtype, device)
        self.w3 = Linear(d_model, d_ff, dtype, device)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        positions = torch.arange(max_seq_len).unsqueeze(-1)
        theta_exponent = torch.arange(d_k // 2).repeat_interleave(2)
        thetas = torch.tensor(theta, dtype=torch.float32) ** (2 * theta_exponent / d_k)
        thetas = positions / thetas
        cosine = torch.cos(thetas).to(device=device)
        sine = torch.sin(thetas).to(device=device)
        self.register_buffer("cosine", cosine, persistent=False)
        self.register_buffer("sine", sine, persistent=False)

    def forward(self, x, token_positions):
        # [x, y] [cos, sin]
        # [xcos - ysin, ycos + xsin]
        # [x, y][cos, cos] + [-y, x][sin, sin]

        x_dtype = x.dtype
        x_reversed = torch.empty_like(x)
        x_reversed[..., ::2] = -1 * x[..., 1::2]
        x_reversed[..., 1::2] = x[..., ::2]

        cosine = self.cosine[token_positions]
        sine = self.sine[token_positions]

        return (x * cosine + x_reversed * sine).to(dtype=x_dtype)


def softmax(x, dim):
    x_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    normalized_x = x - x.max(dim=dim, keepdim=True)[0]
    exp_x = normalized_x.exp()
    exp_x_sum = exp_x.sum(dim=dim, keepdim=True)
    return (exp_x / exp_x_sum).to(dtype=x_dtype)


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = key.shape[-1]
    logits = einsum(query, key, "... lq d, ... lk d -> ... lq lk") / (d_k**0.5)
    if mask is not None:
        logits = torch.where(mask, logits, float("-inf"))
    probs = softmax(logits, dim=-1)
    return einsum(probs, value, "... lq lk, ... lk d -> ... lq d")


class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_h = num_heads
        self.d_h = d_model // num_heads
        self.qkv = Linear(d_model, 3 * d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)
        self.rope = rope

    def forward(self, x, token_positions=None):
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        seq_len = x.shape[-2]
        q = rearrange(q, "... l (n_h d_h) -> ... n_h l d_h", n_h=self.n_h, d_h=self.d_h)
        k = rearrange(k, "... l (n_h d_h) -> ... n_h l d_h", n_h=self.n_h, d_h=self.d_h)

        if self.rope is not None:
            token_positions = token_positions if token_positions is not None else torch.arange(seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        v = rearrange(v, "... l (n_h d_h) -> ... n_h l d_h", n_h=self.n_h, d_h=self.d_h)
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).tril()
        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "... n_h l d_h -> ... l (n_h d_h)")
        return self.output_proj(output)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, num_heads, rope=rope, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
        self.attn_norm = RMSNorm(d_model, device=device)
        self.ffn_norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        y = x + self.attn(self.attn_norm(x))
        z = y + self.ffn(self.ffn_norm(y))
        return z


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, device=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.num_layers = num_layers
        self.rope = RotaryPositionEmbedding(
            theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device
        )
        self.layers = torch.nn.ModuleDict(
            {
                f"{i}": TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device)
                for i in range(num_layers)
            }
        )
        self.norm = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(self, x):
        x = x.to(torch.int32)
        x = self.embedding(x)
        for i in self.layers.keys():
            x = self.layers[i](x)
        x = self.lm_head(self.norm(x))
        return x


def cross_entropy(logits, labels):
    if len(logits.shape) != len(labels.shape):
        labels = labels.unsqueeze(-1)
    maximum_value = torch.max(logits, dim=-1, keepdim=True)[0]
    logits = logits - maximum_value
    logits = logits.to(torch.float32)

    numerator = torch.gather(logits, dim=-1, index=labels)
    log_sum_exp_values = torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    log_loss = -1 * numerator + log_sum_exp_values

    return torch.mean(log_loss)


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["iteration"] = iteration
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
