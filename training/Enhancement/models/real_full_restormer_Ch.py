# real_full_restormer.py
# Portions of this code were generated with the assistance of ChatGPT (OpenAI, 2025) and subsequently modified by the author.
# Portions of the logic are adapted from the public Restormer dataset‑split utilities [2].

# [1]OpenAI. (2025). ChatGPT (May 2025 version) [Large language model]. https://chat.openai.com 
# [2] S.W. Zamir, "swz30/Restormer," GitHub repository, https://github.com/swz30/Restormer .

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MDTA(nn.Module):
   # def __init__(self, dim, num_heads=4):

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)


    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = self.dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        #self.attn = MDTA(dim, num_heads, window_size=8)  
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, inp_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(inp_channels, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8]):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)

        self.encoder2 = nn.Sequential(*[TransformerBlock(dim*2) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim*2)

        self.encoder3 = nn.Sequential(*[TransformerBlock(dim*4) for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim*4)

        self.latent = nn.Sequential(*[TransformerBlock(dim*8) for _ in range(num_blocks[3])])

        self.up3 = Upsample(dim*8)
        self.decoder3 = nn.Sequential(*[TransformerBlock(dim*4) for _ in range(num_blocks[2])])

        self.up2 = Upsample(dim*4)
        self.decoder2 = nn.Sequential(*[TransformerBlock(dim*2) for _ in range(num_blocks[1])])

        self.up1 = Upsample(dim*2)
        self.decoder1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Restormer] Total Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x_input):
        x1 = self.patch_embed(x_input)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.latent(self.down3(x3))

        x = self.up3(x4) + x3
        x = self.decoder3(x)
        x = self.up2(x) + x2
        x = self.decoder2(x)
        x = self.up1(x) + x1
        x = self.decoder1(x)

        return x_input + self.output(x)

