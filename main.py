import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

print("torch version:", torch.__version__)
print("torch device:", torch.cuda.get_device_name(device) if cuda_available else "CPU")


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""

    def __init__(self, dim, bias=False, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """Causal Multi-Query Self Attention"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, n_embd + (2 * n_embd // n_head), bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x).split((C, C // self.n_head, C // self.n_head), dim=2)
        q = qkv[0].view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = qkv[1].view(B, T, 1, C // self.n_head).transpose(1, 2)
        v = qkv[2].view(B, T, 1, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """ Swish Gated Linear Unit """
    def __init__(self, n_embd, dff, dropout):
        super(SwiGLU, self).__init__()
        self.l1 = nn.Linear(n_embd, dff * 2, bias=False)
        self.l2 = nn.Linear(dff, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.l1(x).chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.l2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ Causal Encoder Block """
    def __init__(self, n_embd, n_head, dff, dropout):
        super(Block, self).__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ffwd = SwiGLU(n_embd, dff, dropout)
        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + self.ffwd(self.ln2(x))
        return x


class PaLM(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, dff, dropout, vocab_size, block_size):
        super(PaLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dff, dropout) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, x, y=None):
        _, T = x.shape
        x = self.token_embedding(x) + self.pos_embedding(torch.arange(T, device=device))
        for block in self.blocks:
            x = block(x)
        x = self.head(self.ln(x))

        if y is None:
            logits = x
            loss = None
        else:
            B, T, C = x.shape
            logits = x.view(B * T, C)
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, ctx, num_tokens):
        for tok in (loop := trange(1, num_tokens + 1, leave=False)):
            loop.set_description("tokens [{}/{}]".format(tok, num_tokens))
            ctx_cond = ctx[:, -self.block_size :]
            logits, _ = self(ctx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ctx_next = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, ctx_next), dim=1)
        return ctx


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iterations)
    for k in trange(eval_iterations, leave=False):
        X, Y = get_batch()
        _, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


# Hyperparameters
n_embd = 384
n_head = 6
n_layer = 6
dff = n_embd * 4
dropout = 0.2
block_size = 256
batch_size = 32
learning_rate = 3e-4
iterations = 10000
eval_intervals = 1000
eval_iterations = 100

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

model = PaLM(
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dff=dff,
    dropout=dropout,
    vocab_size=vocab_size,
    block_size=block_size,
).to(device)

print("model parameters:", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in (loop := trange(1, iterations + 1, leave=False)):
    loop.set_description("training")
    if (iter - 1) % eval_intervals == 0:
        loop.set_description("evaluating")
        loop.write("step: {} - loss: {:.4f}".format(iter - 1, estimate_loss().numpy()))
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("step: {} - loss: {:.4f}".format(iterations, estimate_loss().numpy()))

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
with open("output.txt", "w") as f:
    f.write(decode(model.generate(context, 10000)[0].tolist()))
