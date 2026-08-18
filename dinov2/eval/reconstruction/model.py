import math
import torch
from torch import nn, Tensor

from dinov2.layers.block import Block


class Decoder(nn.Module):
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      num_layers: int,
      patch_size: int,
  ):
    super().__init__()
    self.patch_size = patch_size
    self.layers = nn.ModuleList([
        Block(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            drop=0.1,
            attn_drop=0.1,
            init_values=1e-4,
        )
        for _ in range(num_layers)
    ])
    self.pred_head = nn.Linear(embed_dim, patch_size ** 2 * 3)
    
  def forward(self, x: Tensor) -> Tensor:
    for layer in self.layers:
      x = layer(x)
    return self.pred_head(x)


class ReconstructionModel(nn.Module):
  def __init__(
      self,
      backbone: nn.Module,
      num_layers: int = 4,
  ):
    super().__init__()
    self.backbone = backbone
    self.patch_size = backbone.patch_size
    self.decoder1 = Decoder(
        embed_dim=backbone.embed_dim // 2,
        num_heads=backbone.num_heads,
        num_layers=num_layers,
        patch_size=self.patch_size,
    )
    self.decoder2 = Decoder(
        embed_dim=backbone.embed_dim // 2,
        num_heads=backbone.num_heads,
        num_layers=num_layers,
        patch_size=self.patch_size,
    )

  def unpatchify(self, patch_tokens):
    B, N, C = patch_tokens.shape
    h = w = int(math.sqrt(N))
    p = self.patch_size
    c = C // (p * p)
    x = patch_tokens.reshape(B, h, w, p, p, c)
    x = torch.einsum('nhwpqc->nchpwq', x)
    x = x.reshape(B, c, h * p, w * p)
    x = torch.sigmoid(x)  # Ensure output is in [0, 1] range
    return x

  def forward(self, x: Tensor) -> Tensor:
    with torch.no_grad():
      features = self.backbone.get_intermediate_layers(x)
      patch_token = features[0][0]
    px, cx = torch.tensor_split(patch_token, 2, dim=-1)
    p = self.unpatchify(self.decoder1(px))
    c = self.unpatchify(self.decoder2(cx))
    return p, c

