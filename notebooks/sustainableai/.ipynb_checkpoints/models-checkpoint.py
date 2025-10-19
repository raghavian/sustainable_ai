import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sustainableai.datasets import encode,decode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a Simple GPT model
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_seq_len):
        super(SimpleGPT, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)
        
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(embed_size, num_heads, hidden_dim) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        out = self.fc_out(x)
        return out

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, hidden):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*28*28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,3*28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

### Basic Vision Transformer

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(28, 28), patch_size=7, in_chans=3, embed_dim=128):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"
        self.n_patches = (H // patch_size) * (W // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/ps, W/ps] -> [B, N, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # pre-norm
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

@torch.inference_mode()
def generate_text(model, stoi, itos, prompt, max_seq_len, max_length=200, temperature=1.0):
    model.eval()
    ids = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)

    steps = max(0, max_length - ids.size(1))
    for _ in range(steps):
        inp = ids[:, -max_seq_len:]  # strict window; model never sees > max_seq_len tokens
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(inp)[:, -1, :]  # [1, V]
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # stays on device
        ids = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist(), itos)

### ViT Model

class ViT(nn.Module):
    def __init__(
        self,
        img_size=(28, 28),
        patch_size=7,
        in_chans=3,
        num_classes=4,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, attn_dropout, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.patch_embed(x)                 # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # [B, 1+N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                        # [B, 1+N, D]
        cls_out = x[:, 0]                       # [B, D]
        return self.head(cls_out)               # [B, num_classes]

### Basic DenseNet 

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, drop_rate=0.0, bn_size=4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_ch, num_layers, growth_rate, drop_rate=0.0, bn_size=4):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layer = DenseLayer(ch, growth_rate, drop_rate, bn_size)
            layers.append(layer)
            ch += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_ch = ch

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_ch, compression=0.5):
        super().__init__()
        out_ch = int(in_ch * compression)
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.out_ch = out_ch

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x

class DenseNetSmall(nn.Module):
    def __init__(self, in_chans=3, num_classes=4, growth_rate=12, block_layers=(4, 4, 4),
                 compression=0.5, drop_rate=0.0):
        super().__init__()
        self.stem = nn.Conv2d(in_chans, 2 * growth_rate, kernel_size=3, padding=1, bias=False)
        ch = 2 * growth_rate

        blocks = []
        for i, nl in enumerate(block_layers):
            block = DenseBlock(ch, nl, growth_rate, drop_rate)
            ch = block.out_ch
            blocks.append(block)
            if i != len(block_layers) - 1:
                trans = Transition(ch, compression)
                ch = trans.out_ch
                blocks.append(trans)

        self.features = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(ch)
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


### Simple MLP

class MLP(nn.Module):
    def __init__(self, in_shape=(3, 28, 28), hidden=256, num_classes=4):
        super().__init__()
        c, h, w = in_shape
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c * h * w, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.flatten(x)         # [B, 2352]
        x = F.relu(self.fc1(x))     # [B, hidden]
        return self.fc2(x)          # [B, 4] logits

