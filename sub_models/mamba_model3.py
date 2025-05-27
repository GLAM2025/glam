
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Optional
from functools import partial

from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim, d_state=32)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        
        # hidden_states =  self.norm(hidden_states)
        # feat = hidden_states # 43 就只是把它放在后面
        
        feat = hidden_states 
        hidden_states =  self.norm(hidden_states)
        

        return_last_state=False
        if return_last_state:   # 需要修改mamba_ssm的forward函数
            hidden_states, last_state = self.mixer(hidden_states, inference_params=inference_params, return_last_state=True)   
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
            
        return hidden_states + feat

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-6,  # mamba-transformer用的是1e-6.一般transformer用的是1e-5
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class StochasticMamba(nn.Module):
    def __init__(
        self,
        stoch_dim: int,
        action_dim: int,
        d_model: int,
        n_layer: int,
        max_length: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-6,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        add_mlp=True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.action_dim = action_dim
        self.d_model = d_model

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=False),
        )

        if add_mlp:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.LayerNorm(d_model),
                nn.SiLU(inplace=True),
            )
        else: self.output_proj = None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, samples, action, inference_params=None):
        if False:
            action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        hidden_states = feats
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params, 
            ) 

        if self.output_proj is not None:
            feats = hidden_states
            hidden_states = self.output_proj(hidden_states)
            hidden_states = hidden_states + feats

        return hidden_states
    
    def read_state(self, last_state):
        return torch.mean(last_state, dim=1)