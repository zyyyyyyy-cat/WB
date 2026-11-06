import torch
from torch import nn
from torch.nn import functional as F

from .block_adapter import ResidualAttentionBlockWithAdapter
from modeling.advanced_transformer.componet_module import LayerNorm
from modeling.backbones.vit_pytorch import trunc_normal_


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, cfg=None):
        super().__init__()

        self.width = width
        self.layers = layers

        self.trainable_tca = nn.Sequential(
            nn.Linear(self.width, int(self.width * 2)),
            nn.GELU(),
            nn.Linear(int(self.width * 2), self.width),
        )

        self.resblocks = nn.Sequential(
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=0, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=1, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=2, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=3, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=4, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=5, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=6, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=7, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=8, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=9, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=10, shared=self.trainable_tca),
            ResidualAttentionBlockWithAdapter(width, heads, attn_mask, index=11, shared=self.trainable_tca),
        )

        self.trainable_early_fusion = nn.Sequential(
            Nothing(self.width),
            Nothing(self.width),
            IFA(self.width),
            IFA(self.width),
            Nothing(self.width),
            Nothing(self.width),
            Nothing(self.width),
            Nothing(self.width),
            Nothing(self.width),
            Nothing(self.width),
            Nothing(self.width),
            Nothing(self.width),
        )

    def forward(self, RGB: torch.Tensor, NI: torch.Tensor, TI: torch.Tensor):
        # [129, *, 768]

        RGB, NI, TI = self.trainable_early_fusion[0](RGB, NI, TI)

        # 0
        RGB = self.resblocks[0](RGB, modality='rgb', index=0)
        NI = self.resblocks[0](NI, modality='nir', index=0)
        TI = self.resblocks[0](TI, modality='tir', index=0)

        RGB, NI, TI = self.trainable_early_fusion[1](RGB, NI, TI)

        # 1
        RGB = self.resblocks[1](RGB, modality='rgb', index=1)
        NI = self.resblocks[1](NI, modality='nir', index=1)
        TI = self.resblocks[1](TI, modality='tir', index=1)

        RGB, NI, TI = self.trainable_early_fusion[2](RGB, NI, TI)

        # 2
        RGB = self.resblocks[2](RGB, modality='rgb', index=2)
        NI = self.resblocks[2](NI, modality='nir', index=2)
        TI = self.resblocks[2](TI, modality='tir', index=2)

        RGB, NI, TI = self.trainable_early_fusion[3](RGB, NI, TI)

        # 3
        RGB = self.resblocks[3](RGB, modality='rgb', index=3)
        NI = self.resblocks[3](NI, modality='nir', index=3)
        TI = self.resblocks[3](TI, modality='tir', index=3)

        RGB, NI, TI = self.trainable_early_fusion[4](RGB, NI, TI)

        # 4
        RGB = self.resblocks[4](RGB, modality='rgb', index=4)
        NI = self.resblocks[4](NI, modality='nir', index=4)
        TI = self.resblocks[4](TI, modality='tir', index=4)

        RGB, NI, TI = self.trainable_early_fusion[5](RGB, NI, TI)

        # 5
        RGB = self.resblocks[5](RGB, modality='rgb', index=5)
        NI = self.resblocks[5](NI, modality='nir', index=5)
        TI = self.resblocks[5](TI, modality='tir', index=5)

        RGB, NI, TI = self.trainable_early_fusion[6](RGB, NI, TI)

        # 6
        RGB = self.resblocks[6](RGB, modality='rgb', index=6)
        NI = self.resblocks[6](NI, modality='nir', index=6)
        TI = self.resblocks[6](TI, modality='tir', index=6)

        RGB, NI, TI = self.trainable_early_fusion[7](RGB, NI, TI)

        # 7
        RGB = self.resblocks[7](RGB, modality='rgb', index=7)
        NI = self.resblocks[7](NI, modality='nir', index=7)
        TI = self.resblocks[7](TI, modality='tir', index=7)

        RGB, NI, TI = self.trainable_early_fusion[8](RGB, NI, TI)

        # 8
        RGB = self.resblocks[8](RGB, modality='rgb', index=8)
        NI = self.resblocks[8](NI, modality='nir', index=8)
        TI = self.resblocks[8](TI, modality='tir', index=8)

        RGB, NI, TI = self.trainable_early_fusion[9](RGB, NI, TI)

        # 9
        RGB = self.resblocks[9](RGB, modality='rgb', index=9)
        NI = self.resblocks[9](NI, modality='nir', index=9)
        TI = self.resblocks[9](TI, modality='tir', index=9)

        RGB, NI, TI = self.trainable_early_fusion[10](RGB, NI, TI)

        # 10
        RGB = self.resblocks[10](RGB, modality='rgb', index=10)
        NI = self.resblocks[10](NI, modality='nir', index=10)
        TI = self.resblocks[10](TI, modality='tir', index=10)
        
        RGB, NI, TI = self.trainable_early_fusion[11](RGB, NI, TI)

        # 11
        #RGB = self.resblocks[11](RGB, modality='rgb', index=11)
        #NI = self.resblocks[11](NI, modality='nir', index=11)
        #TI = self.resblocks[11](TI, modality='tir', index=11)
        

        return RGB, NI, TI


class IFA(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.width = d_model
        self.head = 1
        self.alpha = 1
        self.m = 1
        self.n = 1

        self.ln_rgb = LayerNorm(d_model)
        self.ln_nir = LayerNorm(d_model)
        self.ln_tir = LayerNorm(d_model)
        
        self.down_q_rgb = nn.Linear(d_model, d_model // 8)
        self.down_q_nir = nn.Linear(d_model, d_model // 8)
        self.down_q_tir = nn.Linear(d_model, d_model // 8)
        self.gate_q_rgb = nn.Linear(d_model, d_model // 8)
        self.gate_q_nir = nn.Linear(d_model, d_model // 8)
        self.gate_q_tir = nn.Linear(d_model, d_model // 8)

        self.down_kv_rgb = nn.Linear(d_model, d_model // 8)
        self.down_kv_nir = nn.Linear(d_model, d_model // 8)
        self.down_kv_tir = nn.Linear(d_model, d_model // 8)
        self.gate_kv_rgb = nn.Linear(d_model, d_model // 8)
        self.gate_kv_nir = nn.Linear(d_model, d_model // 8)
        self.gate_kv_tir = nn.Linear(d_model, d_model // 8)


        self.up_rgb = nn.Linear(d_model // 8, d_model)
        self.up_nir = nn.Linear(d_model // 8, d_model)
        self.up_tir = nn.Linear(d_model // 8, d_model)

        self.attn_rgb = nn.MultiheadAttention(self.width // 8, self.head)
        self.attn_nir = nn.MultiheadAttention(self.width // 8, self.head)
        self.attn_tir = nn.MultiheadAttention(self.width // 8, self.head)

        self.apply(self._init_weights)

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

    def forward(self, RGB, NI, TI):
        RGB_patch = self.ln_rgb(RGB[1:])
        NI_patch = self.ln_nir(NI[1:])
        TI_patch = self.ln_tir(TI[1:])

        RGB_q = self.down_q_rgb(RGB_patch) * F.silu(self.gate_q_rgb(RGB_patch))
        NI_q = self.down_q_nir(NI_patch) * F.silu(self.gate_q_nir(NI_patch))
        TI_q = self.down_q_tir(TI_patch) * F.silu(self.gate_q_tir(TI_patch))

        RGB_kv = self.down_kv_rgb(RGB_patch) * F.silu(self.gate_kv_rgb(RGB_patch))
        NI_kv = self.down_kv_nir(NI_patch) * F.silu(self.gate_kv_nir(NI_patch))
        TI_kv = self.down_kv_tir(TI_patch) * F.silu(self.gate_kv_tir(TI_patch))

        RGB_fusion = self.m * RGB_q + self.n * self.attn_rgb(RGB_q, torch.concat((NI_kv, TI_kv), dim=0), torch.concat((NI_kv, TI_kv), dim=0))[0]
        NI_fusion = self.m * NI_q + self.n * self.attn_nir(NI_q, torch.concat((RGB_kv, TI_kv), dim=0), torch.concat((RGB_kv, TI_kv), dim=0))[0]
        TI_fusion = self.m * TI_q + self.n * self.attn_tir(TI_q, torch.concat((RGB_kv, NI_kv), dim=0), torch.concat((RGB_kv, NI_kv), dim=0))[0]

        RGB_fusion = self.up_rgb(RGB_fusion)
        NI_fusion = self.up_nir(NI_fusion)
        TI_fusion = self.up_tir(TI_fusion)

        RGB = RGB + self.alpha * torch.concat((torch.zeros_like(RGB[0]).unsqueeze(0), RGB_fusion), dim=0)
        NI = NI + self.alpha * torch.concat((torch.zeros_like(NI[0]).unsqueeze(0), NI_fusion), dim=0)
        TI = TI + self.alpha * torch.concat((torch.zeros_like(TI[0]).unsqueeze(0), TI_fusion), dim=0)
        return RGB, NI, TI


class Nothing(nn.Module):
    def __init__(self, width):
        super().__init__()

    def forward(self, RGB, NIR, TIR):
        return RGB, NIR, TIR
