from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from modeling.advanced_transformer.componet_module import LayerNorm, QuickGELU
from modeling.backbones.vit_pytorch import trunc_normal_


class ResidualAttentionBlockWithAdapter(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, index=None, shared=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        dropout = 0.0
        self.trainable_dpa_down_proj = nn.Linear(d_model, int(d_model // 8))
        self.trainable_dpa_down_proj_gate = nn.Linear(d_model, int(d_model // 8))
        self.trainable_dpa_up_proj = nn.Linear(int(d_model // 8), d_model)

        self.trainable_cls_tuning_mhsa = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trainable_cls_tuning_ffn = nn.Parameter(torch.zeros(1, 1, d_model))

        self._init_weights(self.trainable_dpa_up_proj)
        self._init_weights(self.trainable_dpa_down_proj_gate)
        self._init_weights(self.trainable_dpa_down_proj)

        self.trainable_shared_tca = shared

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.001)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def adapter_ffn(self, x: torch.Tensor):
        if True:
            x_cls = x[0].unsqueeze(0)
            x_patch = x[1:]
            adapter_cls = self.trainable_shared_tca(x_cls)
            # adapter_cls = torch.zeros_like(x_cls)
            
            adapter_x = self.trainable_dpa_down_proj(x_patch)
            if False:
                adapter_x_gate = self.trainable_dpa_down_proj_gate(x_patch)
                adapter_x = self.trainable_dpa_up_proj(F.silu(adapter_x_gate) * adapter_x)
            else:
                adapter_x = self.trainable_dpa_up_proj(F.relu(adapter_x))
            adapter_x = torch.cat((adapter_cls, adapter_x), dim=0)
        else:
            adapter_x = self.trainable_dpa_down_proj(x)
            adapter_x = self.trainable_dpa_up_proj(F.relu(adapter_x))
        return adapter_x


    def forward(self, x: torch.Tensor, modality=None, index=0):
        
        batch_size = x.shape[1]
        cls_tuning_mhsa = self.trainable_cls_tuning_mhsa.repeat(1, batch_size, 1)

        x = x + self.attention(self.ln_1(x) + torch.cat((cls_tuning_mhsa, torch.zeros_like(x[1:])), dim=0))

        adapter_x = self.adapter_ffn(x)
        cls_tuning_ffn = self.trainable_cls_tuning_ffn.repeat(1, batch_size, 1)
        
        x = x + self.mlp(self.ln_2(x) + torch.cat((cls_tuning_ffn, torch.zeros_like(x[1:])), dim=0)) + adapter_x
        
        return x
