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
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class TransformLayer(nn.Module):
    def __init__(self,d_model, n_heads, attn_mask: torch.Tensor = None):
        super(TransformLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 3)
        self.norm_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x += self.attention(self.norm_1(x))
        x += self.mlp(self.norm_2(x))
        return x



class Transformer(nn.Module):
    def __init__(self, width, n_heads, layers, attn_mask: torch.Tensor = None):
        super(Transformer, self).__init__()
        self.width = width # embedding size of text
        self.layers = layers 
        self.blocks = nn.Sequential(*[TransformLayer(width, n_heads, attn_mask) for _ in range(layers)])
    
    def forward(self, x):
        return self.blocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_res_size, width, patch_size, layers_count, n_heads, output_dim):
        super(VisionTransformer, self).__init__()
        self.patch_conv = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_res_size // patch_size) ** 2 + 1, width))


        self.norm_pre = LayerNorm(width)
        self.transformer = Transformer(width, n_heads, layers_count)

        self.norm_post = LayerNorm(width)
        self.img_projection = nn.Parameter(scale * torch.randn(width, output_dim))
    
    @property
    def dtype(self):
        return self.patch_conv.weight.dtype

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

        if self.img_projection is not None:
            x = x @ self.img_projection
        
        return x
    

class TextTransformer(nn.Module):

    def __init__(self, vocab_size, width, context_length, output_dim, layers_count, n_heads):

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.norm = LayerNorm(width)

        self.transformer = Transformer(width, n_heads, layers_count, attn_mask=self.build_attn_mask())

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_attn_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def forward(self, x):
        x = self.token_embedding(x).to(x.dtype)  # [batch_size, n_ctx, d_model]

        x += self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm(x).to(x.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.text_projection is not None:
            x = x[torch.arange(x.shape[0]), x.argmax(dim=-1)] @ self.text_projection

        return x
    


class CLIP(nn.Module):

    def __init__(self, output_dim,
                 img_res, patch_size,image_width, vision_layers_count,
                 vocab_size, context_length,text_width, text_n_heads, text_layers_count):
        
        self.vision_transformer = VisionTransformer(input_res_size=img_res,
                                                    width=image_width,
                                                    patch_size=patch_size,
                                                    layers_count=vision_layers_count,
                                                    n_heads= image_width // 64,
                                                    output_dim=output_dim)

        self.text_transformer = TextTransformer(vocab_size=vocab_size,
                                                width=text_width,
                                                context_length=context_length,
                                                layers_count=text_layers_count,
                                                n_heads=text_n_heads,
                                                output_dim=output_dim)
        
        self.build_weights()

    
    def build_weights(self):
        pass
    
    def encode_image(self, img):
        pass

    def encode_text(self, text):
        pass

    def forward(self, img, text):
        pass


def convert_weights_to_fp16(model):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "img_projection"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)