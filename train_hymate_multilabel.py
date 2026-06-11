# train_hymate_multilabel.py

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np
from mamba_ssm import Mamba
from einops import rearrange, repeat, einsum

from models import TimeSeriesModel

class CVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = int(np.sqrt(args.hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # bsz,max_len,hid_dim
        return x


class FusionAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = args.hid_dim
        self.W = nn.Parameter(torch.empty(args.hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # bsz,max_len
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz,max_len
        return att

class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim  # 768
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x ):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0])
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[1]) + \
                   self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[0]) + \
                   self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dimension per head

        # Linear layers for Q, K, V projections (for one layer/head_set)
        self.Wq = torch.nn.Parameter(torch.empty(num_heads, d_model, self.d_k))
        self.Wk = torch.nn.Parameter(torch.empty(num_heads, d_model, self.d_k))
        self.Wv = torch.nn.Parameter(torch.empty(num_heads, d_model, self.d_k))
        self.Wo = torch.nn.Parameter(torch.empty(num_heads * self.d_k, d_model))

        self.dropout = dropout_rate
        self.attention_dropout = attention_dropout_rate

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.Wq)
        torch.nn.init.xavier_uniform_(self.Wk)
        torch.nn.init.xavier_uniform_(self.Wv)
        torch.nn.init.xavier_uniform_(self.Wo)

    def forward(self, x, padding_mask):
        # x: bsz, max_len, d_model
        # padding_mask: bsz, max_len (True for real tokens, False for padding)

        bsz, max_len, d_model = x.size()
        h = self.num_heads

        attention_mask = padding_mask[:, :, None] * padding_mask[:, None, :] # (bsz, max_len, max_len)
        attention_mask = (1 - attention_mask).to(x.dtype) * torch.finfo(x.dtype).min # (bsz, max_len, max_len)
        attention_mask = attention_mask[:, None, :, :] # Add head dimension: (bsz, 1, max_len, max_len)


        # 2. Multi-Head Attention (MHA) calculations
        # Q, K, V projections
        # 'bld,hde->bhle': (batch, length, d_model) @ (heads, d_model, d_k) -> (batch, heads, length, d_k)
        q = torch.einsum('bld,hde->bhle', x, self.Wq)
        k = torch.einsum('bld,hde->bhle', x, self.Wk)
        v = torch.einsum('bld,hde->bhle', x, self.Wv)

        # Attention Scores (Q @ K_T)
        # 'bhle,bhke->bhlk': (batch, heads, length_q, d_k) @ (batch, heads, length_k, d_k) -> (batch, heads, length_q, length_k)
        A = torch.einsum('bhle,bhke->bhlk', q, k)

        # Apply dropout mask if training
        layer_mask = attention_mask # Start with padding mask
        if self.training:
            dropout_mask = (torch.rand_like(A) < self.attention_dropout).to(x.dtype) * torch.finfo(x.dtype).min
            layer_mask = attention_mask + dropout_mask # Combine padding and attention dropout masks

        A = A + layer_mask # Apply combined mask to scores
        A = torch.softmax(A, dim=-1) # Softmax to get attention probabilities

        v_out = torch.einsum('bhkl,bhle->bkhe', A, v)

        # Concatenate heads and apply final linear projection
        all_head_op = v_out.reshape((bsz, max_len, -1)) # Reshape to (bsz, max_len, num_heads * d_k)
        all_head_op = torch.matmul(all_head_op, self.Wo) # Final projection back to d_model
        all_head_op = F.dropout(all_head_op, self.dropout, self.training) # Dropout on output

        # 3. Residual connection and "normalization"
        x_out = (all_head_op + x) / 2 # Your custom averaging normalization

        return x_out

class Block_mamba(nn.Module):
    def __init__(self,
                 dim,
                 blocks,
                 mlp_ratio=1,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 cm_type='EinFFT' #'mlp'
                 ):
        super().__init__()
        self.blocks = blocks
        # self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # self.attn = MambaBlock(d_model=dim) # MambaLayer(dim)
        self.attn = MambaLayer(dim)
        if cm_type == 'EinFFT':
            self.mlp = EinFFT(dim)
        else:
            self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        for b in range(self.blocks):
            x = x + self.drop_path(self.attn(x))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HyMaTE(TimeSeriesModel):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.cve_time = CVE(args)
        self.cve_value = CVE(args)
        self.variable_emb = nn.Embedding(args.V, args.hid_dim)
        self.mamba = Block_mamba(args.hid_dim, args.num_blocks)
        self.self_att = SelfAttentionBlock(args.hid_dim, args.num_layers)
        self.fusion_att = FusionAtt(args)
        self.dropout = args.dropout
        self.V = args.V

    def forward(self, values, times, varis, obs_mask, demo,
                labels=None, forecast_values=None, forecast_mask=None):
        bsz, max_obs = values.size()
        device = values.device
        if self.training:
            with torch.no_grad():
                var_mask = (torch.rand((bsz, self.V), device=device) <= self.dropout).int()
                for v in range(self.V):
                    mask_pos = (varis == v).int() * var_mask[:, v:v + 1]
                    obs_mask = obs_mask * (1 - mask_pos)

        # demographics embedding
        demo_emb = self.demo_emb(demo) if self.args.model_type in ['strats', 'trimba'] \
            else demo
        # initial triplet embedding
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        vari_emb = self.variable_emb(varis)
        triplet_emb = time_emb + value_emb + vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)
        # triplet_emb = F.dropout(value_emb, self.dropout, self.training)
        # contextual triplet emb
        # contextual_emb = triplet_emb
        contextual_emb = self.mamba(triplet_emb)
        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, obs_mask)[:, :, None]
        # self.att_weights = attention_weights
        if self.args.model_type == 'istrats':
            ts_emb = (triplet_emb * attention_weights).sum(dim=1)
        else:
            ts_emb = (contextual_emb * attention_weights).sum(dim=1)
        self.att_weights = self.self_att( ts_emb, obs_mask )
        # ts_emb = ts_emb = (contextual_emb * att).sum(dim=1)
        # concat demo and ts_emb
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        # prediction/loss
        if self.pretrain:
            return self.forecast_final(ts_demo_emb, forecast_values, forecast_mask)
        self.embedding = ts_demo_emb
        logits = self.binary_head(self.forecast_head(ts_demo_emb))[:, 0] \
            if self.finetune else self.binary_head(ts_demo_emb)[:, 0]
        return self.binary_cls_final(logits, labels)