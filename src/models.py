import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_feature_count, hidden_feature_count):
        super(MLP, self).__init__()
        self.sequential = torch.nn.Sequential(
            nn.Linear(input_feature_count, hidden_feature_count),
            nn.GELU(),
            nn.Linear(hidden_feature_count, input_feature_count)
        )
    
    def forward(self, x):
        return self.sequential(x)


class TransformLayer(nn.Module):
    def __init__(self,d_model, n_heads, attn_mask: torch.Tensor = None):
        super(TransformLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm_1 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 3)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x += self.attention(self.norm_1(x))
        x += self.mlp(self.norm_2(x))
        return x



class TextTransformer(nn.Module):
    def __init__(self, width, n_heads, layers, attn_mask: torch.Tensor = None):
        super(TextTransformer, self).__init__()
        self.width = width # embedding size of text
        self.layers = layers 
        self.blocks = nn.Sequential(*[TransformLayer(width, n_heads, attn_mask) for _ in range(layers)])
    
    def forward(self, x):
        return self.blocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_res_size, width, patch_size, layers, n_heads, output_dim):
        super(VisionTransformer, self).__init__()
        self.patch_conv = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_res_size // patch_size) ** 2 + 1, width))


        self.norm_pre = nn.LayerNorm(width)
        self.transformer = TextTransformer(width, layers, n_heads)

        self.norm_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.patch_conv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x += self.positional_embedding.to(x.dtype)

        x = self.norm_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.norm_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        
        return x

class CLIP(torch.nn.Module):
    def __init__(self):
        pass
    pass