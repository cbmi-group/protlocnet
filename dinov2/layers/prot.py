from enum import Enum
from typing import Optional, Tuple, Dict, Any
import torch
from torch import nn, Tensor

from xformers.ops import memory_efficient_attention, unbind

from dinov2.layers.attention import MemEffAttention
from dinov2.layers.drop_path import DropPath
from dinov2.layers.layer_scale import LayerScale
from dinov2.layers.block import (
    NestedTensorBlock,
    add_residual,
    get_branges_scales,
    get_attn_bias_and_cat,
    drop_add_residual_stochastic_depth,
    drop_add_residual_stochastic_depth_list
)


import logging
logger = logging.getLogger('prot')


class BlockChunk(nn.ModuleList):
  def forward(self, x):
    for b in self:
      x = b(x)
    return x

  @staticmethod
  def create(block_chunks, depth, blocks, pre=0):
    if block_chunks > 0:
      chunked_blocks = []
      chunksize = depth // block_chunks
      for i in range(0, depth, chunksize):
        n = (i + pre)
        # this is to keep the block index consistent if we chunk the block list
        chunked_blocks.append([nn.Identity()] * n + blocks[i : i + chunksize])
      return nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
    return nn.ModuleList(blocks)


class ProtIdentity(nn.Module):
  def forward(self, px, cx, **kwargs):
    return px, cx


class ProtBlockChunk(BlockChunk):
  def forward(self, px, cx, **kwargs):
    for b in self:
      px, cx = b(px, cx, **kwargs)
    return px, cx

  @staticmethod
  def create(block_chunks, depth, blocks, pre=0):
    if block_chunks > 0:
      chunked_blocks = []
      chunksize = depth // block_chunks
      for i in range(0, depth, chunksize):
        n = i + pre
        # this is to keep the block index consistent if we chunk the block list
        chunked_blocks.append([ProtIdentity()] * n + blocks[i : i + chunksize])
      return nn.ModuleList([ProtBlockChunk(p) for p in chunked_blocks])
    return nn.ModuleList(blocks)


class AttentionBlockWithReparameterization(NestedTensorBlock):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    dim = kwargs.get('dim')
    self.mu = nn.Linear(dim, dim)
    self.log_var = nn.Linear(dim, dim)

  def reparameter(self, x: Tensor) -> Tensor:
    mu = self.mu(x)
    if self.training:
      log_var = self.log_var(x)
      std = torch.exp(log_var * 0.5)
      eps = torch.rand_like(mu)
      z = mu + std * eps
    else:
      z = mu
    return z

  def forward(self, x_or_x_list):
    if isinstance(x_or_x_list, (list, tuple)):
      x_or_x_list = [self.reparameter(xi) for xi in x_or_x_list]
    else:
      x_or_x_list = self.reparameter(x_or_x_list)
    return super().forward(x_or_x_list)


class ProtAttentionBlock(NestedTensorBlock):
  def __init__(
    self,
    dim: int,
    **kwargs,
  ):
    kwargs.pop('num_register_tokens')
    super().__init__(dim, **kwargs)

  def forward(self, px, cx):
    if isinstance(px, (list, tuple)):
      x = [torch.cat((pxi, cxi), dim=2) for pxi, cxi in zip(px, cx)]
      out = super().forward(x)
      px, cx = zip(*[torch.chunk(xi, 2, dim=2) for xi in out])
      return px, cx
    else:
      x = torch.cat((px, cx), dim=2)
      x = super().forward(x)
      px, cx = torch.chunk(x, 2, dim=2)
      return px, cx


class ProtBranchType(Enum):
  FULL         = 'full'
  LOCAL_ONLY   = 'local_only'
  GLOBAL_ONLY  = 'global_only'
  WITHOUT_GATE = 'without_gate'

  @classmethod
  def get(cls, name: str):
    if isinstance(name, cls):
      return name

    name = name.lower()
    if name in cls._value2member_map_:
      return cls._value2member_map_[name]
    else:
      raise ValueError(f'Unsupported branch type: {name}')


class CrossAttention(nn.Module):
  def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      attn_drop: float = 0.0,
      proj_drop: float = 0.0,
      norm_layer = None,
      init_values: Optional[float] = None,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim**-0.5

    self.normq = norm_layer(dim) if norm_layer else nn.Identity()
    self.normkv = norm_layer(dim) if norm_layer else nn.Identity()

    self.q = nn.Linear(dim, dim, bias=qkv_bias)
    self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
    self.attn_drop = attn_drop
    self.proj = nn.Linear(dim, dim, bias=proj_bias)
    self.proj_drop = nn.Dropout(proj_drop)
    self.ls = LayerScale(dim, init_values) if init_values else nn.Identity()

  def forward(self, q: Tensor, kv: Tensor, attn_bias=None) -> Tensor:
    B, N, C = q.shape
    q = self.q(self.normq(q)).reshape(B, N, self.num_heads, C // self.num_heads)
    kv = self.kv(self.normkv(kv)).reshape(B, N, 2, self.num_heads, C // self.num_heads)
    k, v = unbind(kv, 2)

    x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
    x = x.reshape([B, N, C])

    x = self.proj(x)
    x = self.proj_drop(x)
    x = self.ls(x)
    return x


class ProtCrossAttention(nn.Module):
  def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = False,
      proj_bias: bool = True,
      attn_drop: float = 0.0,
      proj_drop: float = 0.0,
      act_layer = nn.GELU,
      norm_layer = nn.LayerNorm,
      branch_type: str = 'full',
  ) -> None:
    super().__init__()
    self.num_heads = num_heads
    self.branch_type = ProtBranchType.get(branch_type)
    self.has_local = self.branch_type != ProtBranchType.GLOBAL_ONLY
    self.has_global = self.branch_type != ProtBranchType.LOCAL_ONLY
    logger.info(f'ProtCrossAttention initialized with branch type: {self.branch_type}')

    if self.has_local:
      self.rep_proj = nn.Sequential(
          nn.Linear(dim, dim),
          act_layer(),
          nn.Linear(dim, dim),
      )

    if self.branch_type != ProtBranchType.GLOBAL_ONLY:
      self.scale_proj = nn.Linear(dim, dim)
      self.shift_proj = nn.Linear(dim, dim)

    if self.branch_type != ProtBranchType.LOCAL_ONLY:
      self.q = nn.Linear(dim, dim, bias=qkv_bias)
      self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.cross_proj = nn.Sequential(
          nn.Linear(dim, dim * 2),
          act_layer(),
          nn.Linear(dim * 2, dim),
          norm_layer(dim),
      )

    if self.branch_type != ProtBranchType.WITHOUT_GATE:
      self.local_proj = nn.Sequential(
          nn.Linear(dim, dim * 2),
          act_layer(),
          nn.Linear(dim * 2, dim),
          norm_layer(dim),
      )
      self.fuse_gate = nn.Sequential(
          nn.Linear(dim * 2, dim // 2),
          act_layer(),
          nn.Linear(dim // 2, dim * 2),
          nn.SiLU(),
      )
    self.proj = nn.Linear(dim, dim, bias=proj_bias)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, px: Tensor, cx: Tensor, attn_bias = None) -> Tensor:
    B, nx, C = cx.shape

    if self.has_local:
      rep = self.rep_proj(cx)
      # local feature branch
      scale = torch.tanh(self.scale_proj(rep))
      shift = self.shift_proj(rep)
      local_feature = (1 + scale) * px + shift
    else:
      rep = cx

    if self.has_global:
      # global feature branch
      nq = px.shape[1]
      q = self.q(px).reshape(B, nq, self.num_heads, C // self.num_heads)
      kv = self.kv(rep).reshape(B, nx, 2, self.num_heads, C // self.num_heads)
      k, v = unbind(kv, 2)
      upd_cross = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
      cross_feature = upd_cross.reshape([B, nq, C])
      upd_cross = self.cross_proj(cross_feature)

    if self.branch_type == ProtBranchType.FULL:
      upd_local = self.local_proj(local_feature)
      # local and global feature fusion
      gate = self.fuse_gate(torch.cat([local_feature, cross_feature], dim=-1))
      gatel, gatec = gate.chunk(2, dim=-1)
      x = gatel * upd_local + gatec * upd_cross
    elif self.branch_type == ProtBranchType.LOCAL_ONLY:
      x = local_feature
    elif self.branch_type == ProtBranchType.GLOBAL_ONLY:
      x = upd_cross
    elif self.branch_type == ProtBranchType.WITHOUT_GATE:
      upd_local = self.local_proj(local_feature)
      x = upd_local + upd_cross
    else:
      raise ValueError(f'Unsupported branch type: {self.branch_type}')

    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class ProtCrossBlock(nn.Module):
  def __init__(
      self,
      dim,
      act_layer,
      ffn_layer,
      norm_layer,
      num_heads: int = 12,
      mlp_ratio: float = 4.,
      drop_path: float = 0.,
      init_values: Optional[float] = None,
      qkv_bias: bool = False,
      ffn_bias: bool = True,
      proj_bias: bool = True,
      branch_type: ProtBranchType = ProtBranchType.FULL,
      **_,
  ):
    super().__init__()
    dim = dim // 2 # the dimension for each branch
    num_heads = num_heads // 2 # the number of heads for each branch, so that the total number of heads in cross attention is the same as the original num_heads
    self.sample_drop_ratio = drop_path
    # Protein branch
    self.p_norm1 = norm_layer(dim)
    self.p_attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias)
    self.p_ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
    # Contour branch
    self.c_norm1 = norm_layer(dim)
    self.c_attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias)
    self.c_ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
    # Cross-Attention (direction selectable)
    self.p_cross_norm = norm_layer(dim)
    self.c_cross_norm = norm_layer(dim)
    self.cross_attn = ProtCrossAttention(
        dim, num_heads, qkv_bias, proj_bias, act_layer=act_layer,
        norm_layer=norm_layer, branch_type=branch_type)
    self.cross_ls = LayerScale(dim, init_values) if init_values else nn.Identity()
    # MLP
    self.p_norm2 = norm_layer(dim)
    self.p_mlp = ffn_layer(dim, int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias,)
    self.p_ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
    self.c_norm2 = norm_layer(dim)
    self.c_mlp = ffn_layer(dim, int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias,)
    self.c_ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
    # drop path
    self.dp = DropPath(drop_path) if drop_path > 0. else nn.Identity()

  def forward_residual(self, x: torch.Tensor, residual_func, drop_fn) -> torch.Tensor:
    if self.training and self.sample_drop_ratio > 0.1:
      return drop_add_residual_stochastic_depth(
          x, residual_func=residual_func,
          sample_drop_ratio=self.sample_drop_ratio,
      )
    elif self.training and self.sample_drop_ratio > 0.0:
      return x + drop_fn(residual_func(x))
    else:
      return x + residual_func(x)

  def forward_cross_nested(self, px, cx):
    def residual_func(px_cat, cx_cat, attn_bias):
      return self.cross_attn(self.p_cross_norm(px_cat), self.c_cross_norm(cx_cat), attn_bias=attn_bias)

    scaling_vector = self.cross_ls.gamma if isinstance(self.cross_ls, LayerScale) else None
    branges_scales = [get_branges_scales(x, sample_drop_ratio=self.sample_drop_ratio) for x in px]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    attn_bias, px_cat = get_attn_bias_and_cat(px, branges)
    _, cx_cat = get_attn_bias_and_cat(cx, branges)
    residual_list = attn_bias.split(residual_func(px_cat, cx_cat, attn_bias=attn_bias))

    outputs = []
    for pxi, brange, residual, residual_scale_factor in zip(px, branges, residual_list, residual_scale_factors):
      outputs.append(add_residual(pxi, brange, residual, residual_scale_factor, scaling_vector).view_as(pxi))
    return outputs

  def forward_nested(self, px: torch.Tensor, cx: torch.Tensor):
    if self.training and self.sample_drop_ratio > 0.0:
      cx = drop_add_residual_stochastic_depth_list(
          cx,
          residual_func=lambda x, attn_bias: self.c_attn(self.c_norm1(x), attn_bias=attn_bias),
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.c_ls1.gamma if isinstance(self.c_ls1, LayerScale) else None,
      )
      cx = drop_add_residual_stochastic_depth_list(
          cx,
          residual_func=lambda x, attn_bias: self.c_mlp(self.c_norm2(x)),
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.c_ls2.gamma if isinstance(self.c_ls2, LayerScale) else None,
      )
      px = drop_add_residual_stochastic_depth_list(
          px,
          residual_func=lambda x, attn_bias: self.p_attn(self.p_norm1(x), attn_bias=attn_bias),
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.p_ls1.gamma if isinstance(self.p_ls1, LayerScale) else None,
      )
      px = self.forward_cross_nested(px, cx)
      px = drop_add_residual_stochastic_depth_list(
          px,
          residual_func=lambda x, attn_bias: self.p_mlp(self.p_norm2(x)),
          sample_drop_ratio=self.sample_drop_ratio,
          scaling_vector=self.p_ls2.gamma if isinstance(self.p_ls2, LayerScale) else None,
      )
      return px, cx
    else:
      attn_bias, px = get_attn_bias_and_cat(px)
      _, cx = get_attn_bias_and_cat(cx)

      cx = cx + self.c_ls1(self.c_attn(self.c_norm1(cx), attn_bias=attn_bias))
      cx = cx + self.c_ls2(self.c_mlp(self.c_norm2(cx)))
      px = px + self.p_ls1(self.p_attn(self.p_norm1(px), attn_bias=attn_bias))
      px = px + self.cross_ls(
          self.cross_attn(self.p_cross_norm(px), self.c_cross_norm(cx), attn_bias=attn_bias))
      px = px + self.p_ls2(self.p_mlp(self.p_norm2(px)))
      return attn_bias.split(px), attn_bias.split(cx)

  def forward_cross(self, px: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
    if self.training and self.sample_drop_ratio > 0.1:
      # 1) extract subset using permutation
      b, n, d = px.shape
      sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
      brange = (torch.randperm(b, device=px.device))[:sample_subset_size]
      px_subset = self.p_cross_norm(px[brange])
      cx_subset = self.c_cross_norm(cx[brange])
      # 2) apply residual_func to get residual
      residual = self.cross_ls(self.cross_attn(px_subset, cx_subset))
      x_flat = px.flatten(1)
      residual = residual.flatten(1)
      residual_scale_factor = b / sample_subset_size
      # 3) add the residual
      result = torch.index_add(
          x_flat, 0, brange, residual.to(dtype=px.dtype),
          alpha=residual_scale_factor
      )
      px = result.view_as(px)
      return px
    elif self.training and self.sample_drop_ratio > 0.0:
      px = px + self.dp(self.cross_ls(self.cross_attn(
          self.p_cross_norm(px), self.c_cross_norm(cx))))
      return px
    else:
      px = px + self.cross_ls(self.cross_attn(
          self.p_cross_norm(px), self.c_cross_norm(cx)))
      return px

  def forward(self, px: torch.Tensor, cx: torch.Tensor):
    if isinstance(px, (list, tuple)):
      return self.forward_nested(px, cx)

    cx = self.forward_residual(cx, lambda x: self.c_ls1(self.c_attn(self.c_norm1(x))), self.dp)
    cx = self.forward_residual(cx, lambda x: self.c_ls2(self.c_mlp(self.c_norm2(x))), self.dp)

    px = self.forward_residual(px, lambda x: self.p_ls1(self.p_attn(self.p_norm1(x))), self.dp)
    px = self.forward_cross(px, cx)
    px = self.forward_residual(px, lambda x: self.p_ls2(self.p_mlp(self.p_norm2(x))), self.dp)
    return px, cx

  def forward_contour(self, cx: torch.Tensor):
    cx = self.forward_residual(cx, lambda x: self.c_ls1(self.c_attn(self.c_norm1(x))), self.dp)
    cx = self.forward_residual(cx, lambda x: self.c_ls2(self.c_mlp(self.c_norm2(x))), self.dp)
    return cx
