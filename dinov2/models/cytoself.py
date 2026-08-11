from typing import Collection, Optional, Sequence, Tuple, Union
from collections import OrderedDict
import copy
import math
import inspect
import logging
import numpy as np

import torch
from torch import nn
from torchvision.transforms import functional as TF

from dinov2.layers.vq import VectorQuantizer
from dinov2.layers.efficientnet import efficientenc_b0


default_block_args = [
    # block arguments for the first encoder
    {
        'blocks_args': [
            {
                'expand_ratio': 1,
                'kernel': 3,
                'stride': 1,
                'input_channels': 32,
                'out_channels': 16,
                'num_layers': 1,
            },
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 2,
                'input_channels': 16,
                'out_channels': 24,
                'num_layers': 2,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 1,
                'input_channels': 24,
                'out_channels': 40,
                'num_layers': 2,
            },
        ]
    },
    # block arguments for the second encoder
    {
        'blocks_args': [
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 2,
                'input_channels': 40,
                'out_channels': 80,
                'num_layers': 3,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 2,  # 1 in the original
                'input_channels': 80,
                'out_channels': 112,
                'num_layers': 3,
            },
            {
                'expand_ratio': 6,
                'kernel': 5,
                'stride': 2,
                'input_channels': 112,
                'out_channels': 192,
                'num_layers': 4,
            },
            {
                'expand_ratio': 6,
                'kernel': 3,
                'stride': 1,
                'input_channels': 192,
                'out_channels': 320,
                'num_layers': 1,
            },
        ]
    },
]


_ACT_DICT = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
    'hswish': nn.Hardswish,
    'mish': nn.Mish,
    'sigmoid': nn.Sigmoid,
    'logsigmoid': nn.LogSigmoid,
    'softmax': nn.Softmax,
    'logsoftmax': nn.LogSoftmax,
}


def duplicate_kwargs(arg1, arg2):
  if isinstance(arg1, dict):
    arg1 = [arg1] * len(arg2)
  else:
    if len(arg1) != len(arg2):
      raise ValueError(f'Length of arg1 ({len(arg1)}) and arg2 ({len(arg2)}) must match.')
  return arg1


def calc_emb_dim(vq_args, emb_shapes):
  vq_args_out = copy.deepcopy(vq_args)
  emb_shapes_out = []
  for i, varg in enumerate(vq_args_out):
    if 'embedding_dim' not in varg:
      raise ValueError(f'embedding_dim must be specified in vq_args for embedding {i}.')
    if 'channel_split' not in varg:
      varg['channel_split'] = inspect.signature(VectorQuantizer).parameters['channel_split'].default
    emb_shapes_out.append((varg['embedding_dim'] * varg['channel_split'],) + tuple(emb_shapes[i]))
  return vq_args_out, tuple(emb_shapes_out)


def calc_groups(in_channels, out_channels, verbose: bool = True):
  if int(in_channels / out_channels) * out_channels == in_channels:
    return out_channels
  else:
    if verbose:
      logging.warning(
          f'in_channels {in_channels} is indivisible by output channel {out_channels}.\n conv_gp is set to 1.',
          UserWarning,
      )
    return 1
  
  
class Conv2dBN(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      act: Optional[str] = 'swish',
      pad: Union[int, str, tuple] = 'same',
      conv_gp: Union[int, str] = 1,
      dilation: int = 1,
      use_bias: bool = False,
      bn_affine: bool = False,
      name: str = 'conv2dbn',
  ):
    super().__init__()
    if conv_gp == 'depthwise':
      conv_gp = calc_groups(in_channels, in_channels, verbose=True)

    self.name = name
    self.conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad,
        dilation=dilation,
        groups=conv_gp,
        bias=use_bias,
    )
    self.bn = None if act is None else nn.BatchNorm2d(out_channels, affine=bn_affine)
    self.act = act if act is None else _ACT_DICT.get(act.lower(), nn.ReLU)()

  def forward(self, x):
    x = self.conv(x)
    if self.act is not None:
      x = self.bn(x)
      x = self.act(x)
    return x
  

class ResidualBlockUnit2d(nn.Module):
  def __init__(
      self,
      num_channels: int,
      act: str = "swish",
      use_depthwise: bool = False,
      bn_affine: bool = False,
      name: str = 'resblock',
      **kwargs,
  ):
    super().__init__()
    act = act.lower()
    self.name = name
    self.conv1 = Conv2dBN(
        num_channels,
        num_channels,
        act=act,
        conv_gp='depthwise' if use_depthwise else 1,
        bn_affine=bn_affine,
        name=f'{name}_cvbn1',
        **kwargs,
    )
    self.conv2 = nn.Conv2d(
        num_channels,
        num_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=num_channels if use_depthwise else 1,
        bias=False,
        padding='same',
        **kwargs,
    )
    self.bn2 = nn.BatchNorm2d(num_channels, affine=bn_affine)
    self.act2 = _ACT_DICT.get(act.lower(), nn.ReLU)()
    
  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += identity
    x = self.act2(x)
    return x
 

class ResidualBlockRepeat(nn.Module):
  def __init__(
      self,
      num_channels: int,
      num_resblocks: int,
      act: str = "swish",
      use_depthwise: bool = False,
      bn_affine: bool = False,
      name: str = 'res_rpeat',
      **kwargs,
  ):
    super().__init__()
    act = act.lower()
    self.name = name
    layer_dict = OrderedDict()
    for i in range(num_resblocks):
      layer_dict[f'res{i+1}'] = ResidualBlockUnit2d(
          num_channels,
          act=act,
          use_depthwise=use_depthwise,
          bn_affine=bn_affine,
          name=f'{name}_res{i+1}',
          **kwargs,
      )
    self.res_repeat = nn.Sequential(layer_dict)
    
  def forward(self, x):
    return self.res_repeat(x)


class ResNetDecoder(nn.Module):
  def __init__(
      self,
      input_shape: tuple,
      output_shape: tuple,
      num_residual_layers: int = 2,
      num_hiddens: Optional[int] = None,
      num_hidden_decrease: bool = True,
      min_hiddens: int = 1,
      act: str = "swish",
      sampling_mode: str = 'bilinear',
      num_blocks: Optional[int] = None,
      use_upsampling: bool = True,
      use_depthwise: bool = False,
      name: str = 'decoder',
      linear_output: bool = True,
      **kwargs,
  ):
    super().__init__()
    input_shape = np.array(input_shape)
    output_shape = np.array(output_shape)
    act = act.lower()
    self.name = name
    
    if num_blocks is None:
      num_blocks = max(np.ceil(np.log2(output_shape[1:] / input_shape[1:])).astype(int))
      
    self.decoder = nn.ModuleDict()
    if num_hiddens is None:
      num_hiddens = input_shape[0]
    else:
      self.decoder['dec_first'] = Conv2dBN(
          input_shape[0],
          num_hiddens,
          act=act,
          conv_gp=1,
          name='dec_first',
          **kwargs,
      )
    _num_hiddens = num_hiddens
    for i in range(num_blocks):
      if use_upsampling:
        target_shape = tuple(np.ceil(output_shape[1:] / (2 ** (num_blocks - (i + 1)))).astype(int).tolist())
        self.decoder[f'up{i + 1}'] = nn.Upsample(size=target_shape, mode=sampling_mode, align_corners=False)
      
      self.decoder[f'resrep{i+1}'] = ResidualBlockRepeat(
          num_hiddens,
          num_residual_layers,
          act=act,
          use_depthwise=use_depthwise,
          name=f'res{i+1}',
          **kwargs,
      )
      
      if num_hidden_decrease:
        _num_hiddens = max(int(num_hiddens / 2), min_hiddens)

      self.decoder[f'resrep{i+1}last'] = Conv2dBN(
          num_hiddens,
          output_shape[0] if (i == num_blocks - 1 and not linear_output) else _num_hiddens,
          act=act,
          conv_gp=1,
          name=f'resrep{i + 1}last',
          **kwargs,
      )
      num_hiddens = _num_hiddens
      
      if i == num_blocks - 1 and linear_output:
        self.decoder['output_conv'] = nn.Conv2d(
            _num_hiddens,
            output_shape[0],
            kernel_size=3,
            stride=1,
            padding='same',
            dilation=1,
            groups=1,
            bias=False,
        )
      
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
        

  def forward(self, x):
    for _, lyr in self.decoder.items():
      x = lyr(x)
    return x


class FCBlock(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_features: int,
      num_layers: int,
      dropout_rate: float = 0.5,
      act: str = 'relu',
      last_activation: str = 'softmax',
  ):
    super().__init__()
    self.last_activation = last_activation
    self.fc_list = nn.ModuleList()
    for i in range(num_layers):
      self.fc_list.append(nn.Dropout(dropout_rate, inplace=False))
      self.fc_list.append(nn.Linear(in_channels if i == 0 else num_features, num_features if i < num_layers - 1 else out_channels))
      if i < num_layers - 1:
        self.fc_list.append(_ACT_DICT.get(act.lower(), nn.ReLU)())
        
    for m in self.modules():
      if isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
  def forward(self, x):
    if not torch.is_floating_point(x):
      x = x.type(torch.float32)
    for lyr in self.fc_list:
      x = lyr(x)
    return x


class Cytoself(nn.Module):
  def __init__(
      self,
      emb_shapes: Collection[Tuple[int, int]],
      vq_args: Union[dict, Collection[dict]],
      num_class: int,
      input_shape: Optional[Tuple[int, int, int]] = None,
      output_shape: Optional[Tuple[int, int, int]] = None,
      fc_input_type: str = 'vqvec',
      encoder_args: Optional[Collection[dict]] = None,
      decoder_args: Optional[Collection[dict]] = None,
      fc_args: Optional[Union[dict, Collection[dict]]] = None,
      data_ch = ['pro', 'nuc'],
  ):
    super().__init__()
    self.data_ch = data_ch
    self.embed_dim = None
    # Check vq_args and emb_shapes and compute emb_ch_splits
    vq_args = duplicate_kwargs(vq_args, emb_shapes)
    vq_args, emb_shapes = calc_emb_dim(vq_args, emb_shapes)
    self.emb_shapes = emb_shapes
    self.input_shape = input_shape
    self.encoders = self._const_encoders(input_shape, encoder_args)
    output_shape = output_shape or input_shape
    self.decoders = self._const_decoders(output_shape, decoder_args)
    self.vq_layers = nn.ModuleList()
    for i, varg in enumerate(vq_args):
      self.vq_layers.append(VectorQuantizer(**varg))

    fc_args_default = {'num_layers': 1, 'num_features': 1000}
    if fc_args is None:
      fc_args = {}
    fc_args.update({k: v for k, v in fc_args_default.items() if k not in fc_args})
    fc_args = duplicate_kwargs(fc_args, emb_shapes)
    
    self.fc_layers = nn.ModuleList()
    for i, shp in enumerate(emb_shapes):
      arg = fc_args[i]
      if fc_input_type == 'vqind':
        arg['in_channels'] = np.prod(shp[1:])
      elif fc_input_type == 'vqindhist':
        arg['in_channels'] = vq_args[i]['num_embeddings']
      else:
        arg['in_channels'] = np.prod(shp)
      arg['out_channels'] = num_class
      self.fc_layers.append(FCBlock(**arg))
      
  def _const_encoders(self, input_shape, encoder_args):
    encoder_args = encoder_args or default_block_args
    if len(self.emb_shapes) != len(encoder_args):
      raise ValueError(f'Length of emb_shapes ({len(self.emb_shapes)}) and encoder_args ({len(encoder_args)}) must match.')
 
    encoders = nn.ModuleList()
    for i, shp in enumerate(self.emb_shapes):
      encoder_args[i].update(
        {
            'in_channels': input_shape[0] if i == 0 else self.emb_shapes[i - 1][0],
            'out_channels': shp[0],
            'first_layer_stride': 2 if i == 0 else 1,
        }
      )
      encoders.append(efficientenc_b0(**encoder_args[i]))
    return encoders

  def _const_decoders(self, output_shape, decoder_args):
    decoder_args = decoder_args or [{}] * len(self.emb_shapes)
    decoders = nn.ModuleList()
    for i, shp in enumerate(self.emb_shapes):
      if i == 0:
        shp = (sum(i[0] for i in self.emb_shapes), ) + shp[1:]
      decoder_args[i].update(
        {
            'input_shape': shp,
            'output_shape': output_shape if i == 0 else self.emb_shapes[i - 1],
            'linear_output': i == 0,
        }
      )
      decoders.append(ResNetDecoder(**decoder_args[i]))
    return decoders
  
  def _connect_decoders(self, encoded_list):
    decoded_list = []
    for i, (encd, dec) in enumerate(zip(encoded_list[::-1], self.decoders[::-1])):
      if i < len(self.decoders) - 1:
        decoded_list.append(TF.resize(encd, self.emb_shapes[0][1:], interpolation='nearest'))
      else:
        encoded_list.append(encd)
        decoded_final = dec(torch.cat(encoded_list, 1))
    return decoded_final

  def forward(self, x, **kwargs):
    pass

  def get_intermediate_layers(
      self,
      x: torch.Tensor,
      n: Union[int, Sequence] = 1,
      token_type: str = 'vqvec2',
      **_,
  ):
    if token_type is None:
      raise ValueError('token_type must be specified.')
    
    pro, nuc, seg = x.chunk(3, dim=1)
    tensors = []
    if 'pro' in self.data_ch:
      tensors.append(pro)
    if 'nuc' in self.data_ch:
      tensors.append(nuc)
    if 'seg' in self.data_ch:
      tensors.append(seg)
    if len(tensors) == 0:
      raise ValueError(f'No valid data channels found in {self.data_ch}. Must be one of pro, nuc, seg.')

    r = None
    x = torch.cat(tensors, dim=1)    
    out_layer_name, out_layer_idx = token_type[:-1], int(token_type[-1]) - 1
    
    for i, enc in enumerate(self.encoders):
      encoded = enc(x)
      if out_layer_name == 'encoder' and i == out_layer_idx:
        r = encoded
        break

      _, quantized, _, _, _encoding_indices, _index_histogram, _ = self.vq_layers[i](encoded)
      
      if i == out_layer_idx:
        if out_layer_name == 'vqvec':
          r = quantized
          break
        elif out_layer_name == 'vqind':
          r = _encoding_indices
          break
        elif out_layer_name == 'vqindhist':
          r = _index_histogram
          break
      x = encoded

    if r is None:
      raise ValueError(f'Invalid token_type {token_type}. Must be one of encoder0, encoder1, ..., vqvec0, vqvec1, ..., vqind0, vqind1, ..., vqindhist0, vqindhist1, ...')
    r = r.flatten(1)
    return ((r.unsqueeze(1), r), )
