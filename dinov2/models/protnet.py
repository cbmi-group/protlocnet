# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused
from dinov2.layers import NestedTensorBlock as AttentionBlock
from dinov2.layers import ProtBlockChunk
from dinov2.layers import ProtAttentionBlock
from dinov2.layers import ProtCrossBlock


logger = logging.getLogger('prot')


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
  if not depth_first and include_root:
    fn(module=module, name=name)
  for child_name, child_module in module.named_children():
    child_name = ".".join((name, child_name)) if name else child_name
    named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
  if depth_first and include_root:
    fn(module=module, name=name)
  return module


class ProtLocNet(nn.Module):
  def __init__(
      self,
      img_size=224,
      patch_size=16,
      in_chans=3,
      embed_dim=768,
      depth=12,
      num_heads=12,
      injector_depth=2,
      decoder_embed_dim=512,
      decoder_depth=4,
      decoder_num_heads=8,
      mlp_ratio=4.0,
      qkv_bias=True,
      ffn_bias=True,
      proj_bias=True,
      drop_path_rate=0.0,
      drop_path_uniform=False,
      init_values=None,  # for layerscale: None or 0 => no layerscale
      embed_layer=PatchEmbed,
      act_layer=nn.GELU,
      ffn_layer="mlp",
      block_chunks=1,
      num_register_tokens=0,
      interpolate_antialias=False,
      interpolate_offset=0.1,
      branch_type: str = 'full',
      reconstruction_mode: bool = False,
      generation_mode: bool = False,
  ):
    """
    Args:
        img_size (int, tuple): input image size
        patch_size (int, tuple): patch size
        in_chans (int): number of input channels
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        proj_bias (bool): enable bias for proj in attn if True
        ffn_bias (bool): enable bias for ffn if True
        drop_path_rate (float): stochastic depth rate
        drop_path_uniform (bool): apply uniform drop rate across blocks
        weight_init (str): weight init scheme
        init_values (float): layer-scale init values
        embed_layer (nn.Module): patch embedding layer
        act_layer (nn.Module): MLP activation layer
        block_fn (nn.Module): transformer block class
        ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
        block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        num_register_tokens: (int) number of extra cls tokens (so-called "registers")
        interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
        interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        reconstruction_mode: (bool) flag to enable reconstruction mode
    """
    super().__init__()
    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    self.embed_dim = embed_dim  # because of concatenation of patch and context tokens
    self.num_features = embed_dim  # num_features for consistency with other models
    self.num_tokens = 1
    self.n_blocks = depth
    self.num_heads = num_heads
    self.patch_size = patch_size
    self.num_register_tokens = num_register_tokens
    self.interpolate_antialias = interpolate_antialias
    self.interpolate_offset = interpolate_offset
    self.reconstruction_mode = reconstruction_mode
    self.generation_mode = generation_mode

    self.p_patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim // 2)
    self.c_patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim // 2)
    self.num_patches = self.p_patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim // 2))
    # self.modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
    assert num_register_tokens >= 0
    self.register_tokens = (
        nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
    )

    if drop_path_uniform is True:
      dpr = [drop_path_rate] * depth
    else:
      dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

    if ffn_layer == "mlp":
      logger.info("using MLP layer as FFN")
      ffn_layer = Mlp
    elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
      logger.info("using SwiGLU layer as FFN")
      ffn_layer = SwiGLUFFNFused
    elif ffn_layer == "identity":
      logger.info("using Identity layer as FFN")

      def f(*args, **kwargs):
        return nn.Identity()

      ffn_layer = f
    else:
      raise NotImplementedError

    blocks_list = [
        ProtCrossBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path=dpr[i],
            norm_layer=norm_layer,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            branch_type=branch_type,
        )
        for i in range(depth)
    ]
    if block_chunks > 0:
      self.chunked_blocks = True
      self.blocks = ProtBlockChunk.create(block_chunks, depth, blocks_list)
    else:
      self.chunked_blocks = False
      self.blocks = nn.ModuleList(blocks_list)

    self.norm = norm_layer(embed_dim)
    self.head = nn.Identity()

    self.mask_token = nn.Parameter(torch.zeros(1, embed_dim // 2))

    if generation_mode:
      self.injector_embed = nn.Linear(embed_dim, embed_dim)
      self.injector_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches + self.num_register_tokens, embed_dim))
      injector_blocks_list = [
          AttentionBlock(
              dim=embed_dim,
              num_heads=num_heads,
              mlp_ratio=mlp_ratio,
              qkv_bias=qkv_bias,
              proj_bias=proj_bias,
              ffn_bias=ffn_bias,
              drop_path=dpr[i],
              norm_layer=norm_layer,
              act_layer=act_layer,
              ffn_layer=ffn_layer,
              init_values=init_values,
          ) for i in range(decoder_depth)
      ]
      self.injector_blocks = nn.ModuleList(injector_blocks_list)
      self.injector_pred = nn.Linear(embed_dim, embed_dim // 2)
      self.injector_norm = norm_layer(embed_dim // 2)

    if reconstruction_mode or generation_mode:
      # decoder specific
      self.decoder_embed = nn.Linear(embed_dim // 2, decoder_embed_dim)
      self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_register_tokens, decoder_embed_dim))
      decoder_blocks_list = [
          AttentionBlock(
              dim=decoder_embed_dim,
              num_heads=decoder_num_heads,
              mlp_ratio=mlp_ratio,
              qkv_bias=qkv_bias,
              proj_bias=proj_bias,
              ffn_bias=ffn_bias,
              drop_path=dpr[i],
              norm_layer=norm_layer,
              act_layer=act_layer,
              ffn_layer=ffn_layer,
              init_values=init_values,
          )
          for i in range(decoder_depth)
      ]
      self.decoder_blocks = nn.ModuleList(decoder_blocks_list)
      self.decoder_norm = norm_layer(decoder_embed_dim)
      expected_dim = patch_size ** 2
      self.protein_pred = nn.Linear(decoder_embed_dim, expected_dim)
      self.contour_pred = nn.Linear(decoder_embed_dim, expected_dim * 2)

    self.init_weights()

  def init_weights(self):
    trunc_normal_(self.pos_embed, std=0.02)
    nn.init.normal_(self.cls_token, std=1e-6)
    if self.register_tokens is not None:
      nn.init.normal_(self.register_tokens, std=1e-6)
    named_apply(init_weights_vit_timm, self)

  def interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    pos_embed = self.pos_embed
    pos_embed = torch.concat((pos_embed, pos_embed), dim=-1)
    # pos_embed = pos_embed + self.modality_embed
    if npatch == N and w == h:
      return pos_embed
    pos_embed = pos_embed.float() # in case of fp16 or bf16, upcast to float for the interpolation to avoid precision loss
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M
    kwargs = {}
    if self.interpolate_offset:
      # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
      # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
      sx = float(w0 + self.interpolate_offset) / M
      sy = float(h0 + self.interpolate_offset) / M
      kwargs["scale_factor"] = (sx, sy)
    else:
      # Simply specify an output size instead of a scale factor
      kwargs["size"] = (w0, h0)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=self.interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

  def prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    px = self.p_patch_embed(x[:, :1])
    cx = self.c_patch_embed(x[:, 1:])

    x = torch.cat((px, cx), dim=-1)
    if masks is not None:
      mask_token = torch.cat((self.mask_token, self.mask_token), -1)
      x = torch.where(masks.unsqueeze(-1), mask_token.to(x.dtype).unsqueeze(0), x)

    cls_token = self.cls_token
    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)

    if self.register_tokens is not None:
      x = torch.cat(
          (
              x[:, :1],
              self.register_tokens.expand(x.shape[0], -1, -1),
              x[:, 1:],
          ),
          dim=1,
      )
    return x

  def get_last_attn(self, x: torch.Tensor):
    x = self.prepare_tokens_with_masks(x)
    for i, blk in enumerate(self.blocks):
      if i < len(self.blocks) - 1:
        x = blk(x)
      else:
        attn = blk(x, return_attn=True)
    return attn

  def unpatchify(self, patch_tokens):
    B, N, C = patch_tokens.shape
    h = w = int(math.sqrt(N))
    p = self.patch_size
    c = C // (p * p)
    x = patch_tokens.reshape(B, h, w, p, p, c)
    x = torch.einsum('nhwpqc->nchpwq', x)
    x = x.reshape(B, c, h * p, w * p)
    return x

  def forward_injector(self, px, cx):
    x = torch.cat((px, cx), dim=-1)
    x = self.injector_embed(x)
    x = x + self.injector_pos_embed.to(dtype=x.dtype, device=x.device)
    for blk in self.injector_blocks:
      x = blk(x)
    x = self.injector_pred(x)
    x = self.injector_norm(x)
    return x

  def forward_decoder(self, x):
    x = x[:, 1:] # remove cls token
    px, cx = torch.chunk(x, 2, dim=-1)
    x = torch.cat((px, cx), dim=0)
    x = self.decoder_embed(x)
    x = x + self.decoder_pos_embed.to(dtype=x.dtype, device=x.device)

    for blk in self.decoder_blocks:
      x = blk(x)
    x = self.decoder_norm(x)
    x = x[:, self.num_register_tokens:]  # remove cls and register tokens
    px, cx = torch.chunk(x, 2, dim=0)
    px = self.unpatchify(self.protein_pred(px))
    cx = self.unpatchify(self.contour_pred(cx))
    return torch.cat((px, cx), dim=1)

  def forward_protein(self, x):
    x = x[:, 1:] # remove cls token
    x = self.decoder_embed(x)
    x = x + self.decoder_pos_embed.to(dtype=x.dtype, device=x.device)
    for blk in self.decoder_blocks:
      x = blk(x)
    x = self.decoder_norm(x)
    x = x[:, self.num_register_tokens:]  # remove cls and register tokens
    x = self.unpatchify(self.protein_pred(x))
    return x

  def forward_features_list(self, x_list, masks_list):
    x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
    px, cx = zip(*[torch.chunk(xi, 2, dim=-1) for xi in x])
    px = [pxi.contiguous() for pxi in px]
    cx = [cxi.contiguous() for cxi in cx]
    for blk in self.blocks:
      px, cx = blk(px, cx)

    output = []
    for pxi, cxi, masks in zip(px, cx, masks_list):
      x = torch.cat((pxi, cxi), dim=-1)
      x_norm = self.norm(x)
      output.append(
          {
              "x_norm_clstoken": x_norm[:, 0],
              "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
              "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
              "x_prenorm": x,
              "x_norm": x_norm,
              "masks": masks,
          }
      )
    return output

  def forward_features(self, x, masks=None, **_):
    if isinstance(x, list):
      return self.forward_features_list(x, masks)

    x = self.prepare_tokens_with_masks(x, masks)
    px, cx = torch.chunk(x, 2, dim=-1)

    for blk in self.blocks:
      px, cx = blk(px, cx)

    x = torch.cat((px, cx), dim=-1)
    x_norm = self.norm(x)

    return {
        "x_norm_clstoken": x_norm[:, 0],
        "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
        "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        "x_prenorm": x,
        "x_norm": x_norm,
        "masks": masks,
    }

  def _get_intermediate_layers_not_chunked(self, x, n=1):
    x = self.prepare_tokens_with_masks(x)
    px, cx = torch.chunk(x, 2, dim=-1)
    # If n is an int, take the n last blocks. If it's a list, take them
    output, total_block_len = [], len(self.blocks)
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    for i, blk in enumerate(self.blocks):
      px, cx = blk(px, cx)
      if i in blocks_to_take:
        output.append(torch.cat((px, cx), dim=-1))
    assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
    return output

  def _get_intermediate_layers_chunked(self, x, n=1):
    x = self.prepare_tokens_with_masks(x)
    px, cx = torch.chunk(x, 2, dim=-1)
    output, i, total_block_len = [], 0, len(self.blocks[-1])
    # If n is an int, take the n last blocks. If it's a list, take them
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    for block_chunk in self.blocks:
      for blk in block_chunk[i:]:  # Passing the nn.Identity()
        px, cx = blk(px, cx)
        if i in blocks_to_take:
          output.append(torch.cat((px, cx), dim=-1))
        i += 1
    assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
    return output

  def get_intermediate_layers(
      self,
      x: torch.Tensor,
      n: Union[int, Sequence] = 1,  # Layers or n last layers to take
      reshape: bool = False,
      norm=True,
      token_type: Optional[str] = None,
      **kwargs,
  ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
    if self.chunked_blocks:
      outputs = self._get_intermediate_layers_chunked(x, n)
    else:
      outputs = self._get_intermediate_layers_not_chunked(x, n)
    if norm:
      outputs = [self.norm(out) for out in outputs]
    if token_type == 'px':
      outputs = [out.chunk(2, dim=-1)[0] for out in outputs]
    elif token_type == 'cx':
      outputs = [out.chunk(2, dim=-1)[1] for out in outputs]
    class_tokens = [out[:, 0] for out in outputs]
    outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
    if reshape:
      B, _, w, h = x.shape
      outputs = [
          out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
          for out in outputs
      ]
    return tuple(zip(outputs, class_tokens))

  def generate(self, ref_image, contour):
    with torch.no_grad():
      rx = self.prepare_tokens_with_masks(ref_image)
      tx = self.prepare_tokens_with_masks(torch.cat((torch.zeros_like(contour[:, :1]), contour), dim=1))  # add dummy channel for contour
      tx = tx.chunk(2, dim=-1)[1]  # take context tokens only

      rx, cx = rx.chunk(2, dim=-1)  # take patch tokens only for reference branch
      for blk in self.blocks:
        rx, cx = blk(rx, cx)
        tx = blk.forward_contour(tx)

      x = torch.cat((rx, tx), dim=-1)
      x = self.norm(x)

    px, cx = torch.chunk(x, 2, dim=-1)
    generation = self.forward_injector(px, cx)
    generation = self.forward_protein(generation)
    return generation

  def decode_features(self, features):
    results = {}
    if self.reconstruction_mode:
      x_norm = features["x_norm"]
      reconstruction = self.forward_decoder(x_norm)
      results.update({"reconstruction": reconstruction})
    if self.generation_mode:
      x_norm = features["x_norm"]
      px, cx = torch.chunk(x_norm, 2, dim=-1)
      px0, px1 = torch.chunk(px, 2, dim=0)
      px = torch.cat((px1, px0), dim=0)
      generation = self.forward_injector(px, cx)
      generation = self.forward_protein(generation)
      results.update({"generation": generation})
    return results

  def forward(self, *args, **kwargs):
    features = self.forward_features(*args, **kwargs)
    is_training = kwargs.get("is_training", False)
    if is_training:
      return features
    token_type = kwargs.get("token_type", None)
    if token_type == 'px':
      return features["x_norm_clstoken"].chunk(2, dim=-1)[0]
    elif token_type == 'cx':
      return features["x_norm_clstoken"].chunk(2, dim=-1)[1]
    return features["x_norm_clstoken"]


def init_weights_vit_timm(module: nn.Module, name: str = ""):
  """ViT weight initialization, original timm impl (for reproducibility)"""
  if isinstance(module, nn.Linear):
    trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
      nn.init.zeros_(module.bias)
  if isinstance(module, nn.Conv2d):
    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')


def prot_small(patch_size=16, num_register_tokens=0, **kwargs):
  model = ProtLocNet(
      patch_size=patch_size,
      embed_dim=384,
      depth=8,
      num_heads=6,
      mlp_ratio=4,
      num_register_tokens=num_register_tokens,
      **kwargs,
  )
  return model


def prot_base(patch_size=16, num_register_tokens=0, **kwargs):
  model = ProtLocNet(
      patch_size=patch_size,
      embed_dim=768,
      depth=12,
      num_heads=12,
      mlp_ratio=4,
      num_register_tokens=num_register_tokens,
      **kwargs,
  )
  return model


def prot_large(patch_size=16, num_register_tokens=0, **kwargs):
  model = ProtLocNet(
      patch_size=patch_size,
      embed_dim=1024,
      depth=24,
      num_heads=16,
      mlp_ratio=4,
      num_register_tokens=num_register_tokens,
      **kwargs,
  )
  return model
