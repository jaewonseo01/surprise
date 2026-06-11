import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin')
os.add_dll_directory(os.path.dirname(__file__))

from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset

from transformer_conv import TransformerConv
from Ob_propagation import Observation_progation
import warnings
import numbers

# Gradient reversal layer for domain-invariant embeddings
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# Modules for Raindrop
class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe


class Raindrop(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None):
        super(Raindrop, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure

        d_pe = 36
        d_enc = 36

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model+36, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt' #  'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        num_types, num_relations = 36, 1
        nhead_HGT = 5

        self.edge_type_train = torch.ones([36*36*2], dtype= torch.int64).cuda()
        self.adj = torch.ones([36, 36]).cuda()

        self.dim = int(d_model/d_inp)

        self.transconv  = TransformerConv(in_channels=36, out_channels=36*self.dim, heads=1)

        d_final = 36*(self.dim+1) + d_model
        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.d_inp = d_inp
        self.d_model = d_model
        self.encoder = nn.Linear(d_inp, d_enc)
        self.emb = nn.Linear(d_static, d_model)

        self.MLP_replace_transformer = nn.Linear(72, 36)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :self.d_inp]
        maxlen, batch_size = src.shape[0], src.shape[1]

        src = self.encoder(src) * math.sqrt(self.d_model)

        pe = self.pos_encoder(times)
        src = self.dropout(src)
        emb = self.emb(static)

        withmask = False
        if withmask == True:
            x = torch.cat([src, missing_mask], dim=-1)
        else:
            x = src

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step2 = True
        if step2 == False:
            output = x
            distance = 0
        elif step2 ==True:
            adj = self.global_structure.cuda()
            adj[torch.eye(36).byte()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            output = torch.zeros([215, src.shape[1], 36*self.dim]).cuda()
            alpha_all = torch.zeros([edge_index.shape[1],  src.shape[1]]).cuda()
            for unit in range(0, x.shape[1]):
                stepdata = x[:, unit, :]
                stepdata, attentionweights = self.transconv(stepdata, edge_index=edge_index, edge_weights=edge_weights,
                                                            edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.reshape([-1, 36*self.dim]).unsqueeze(0)
                output[:, unit, :] = stepdata

                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        output = torch.cat([output, pe], dim=-1)

        step3 = True
        if step3 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step3 == False:
            r_out = output

        masked_agg = True
        if masked_agg == True:
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output, distance, None


class Raindrop_v2(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None, sensor_wise_mask=False, static=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        d_pe = 16
        d_enc = d_inp

        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        self.d_ob = int(d_model/d_inp)

        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :int(src.shape[2]/2)]
        n_sensor = self.d_inp

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(self.d_inp).byte()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda()

            use_beta = False
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]

                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output, distance, None

class Raindrop_Mod(nn.Module):
    """Modified Raindrop model to return embeddings, 
    Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes
    Returns:
        prediction
        distance
        output (embedding used for final prediction) 
    """
    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None, sensor_wise_mask=False, static=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        d_pe = 16
        d_enc = d_inp

        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        self.d_ob = int(d_model/d_inp)

        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 3]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :int(src.shape[2]/2)]
        n_sensor = self.d_inp

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(self.d_inp).bool()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda()

            use_beta = False
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]

                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        prediction = self.mlp_static(output)

        return prediction, distance, output

# Modules for STraTS
class ContinuousValueEmbedding(nn.Module):
    """
    Continuous Value Embedding for time and values.
    """
    def __init__(self, input_dim, embed_dim, activation='tanh'):
        super().__init__()
        self.hidden_dim = int(embed_dim ** 0.5)
        self.lin1 = nn.Linear(input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, embed_dim)
        self.activation = torch.tanh if activation == 'tanh' else nn.ReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        return self.lin2(x)

class TransformerBlock(nn.Module):
    """
    A single Transformer block for the model.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=padding_mask) # ignore "True"
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        return self.norm2(x + self.dropout(ff_output))

class FusionAttention(nn.Module):
    """
    Fusion Attention to aggregate contextual embeddings,
    with epsilon-stabilized softmax to avoid NaNs.
    """
    def __init__(self, embed_dim, eps: float = 1e-6):
        super().__init__()
        self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b = nn.Parameter(torch.zeros(embed_dim))
        self.u = nn.Parameter(torch.empty(embed_dim, 1))
        self.eps = eps
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)

    def forward(self, x, mask):
        # x: [B, L, D], mask: [B, L] float(1=keep, 0=mask)
        # 1) score pre‐activation
        att = torch.tanh(torch.matmul(x, self.W) + self.b)
        scores = torch.matmul(att, self.u).squeeze(-1)
        scores = scores + (1 - mask) * torch.finfo(scores.dtype).min

        # 3) shift for numeric stability
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        # 4) exponentiate and mask again
        exp_scores = torch.exp(scores) * mask         # zeros where mask=0

        # 5) normalize with ε-clamp
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        weights = exp_scores / denom                  

        return weights

class CLSHead(nn.Module):
    """
    Head for CLS token pooling.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)
        return self.activation(x)
    
class FrcstHead(nn.Module):
    """
    Head for masked value forecasting
    """
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()
        self.lin2 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        return x
        
class STraTSModel(nn.Module): # Single task
    """
    Main model definition for the STraTS task, 
    now using dataset-provided mask for pretrain.
    """
    def __init__(self, 
                num_features,
                embed_dim=32,
                static_dim=3, 
                num_heads=4, 
                num_blocks=2, 
                ff_dim=64,
                dropout=0.2, 
                time_activation='relu', 
                value_activation='tanh', 
                final_emb_type='balanced', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,
                ):
        super().__init__()

        # ------------------------------
        # (1) Transformer + Embeddings
        # ------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.fusion_attention = FusionAttention(embed_dim)
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features+1, embed_dim, padding_idx=num_features) # Feats + padding 1

        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(embed_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)


        # ------------------------------
        # (2) Downstream heads
        # ------------------------------

        self.downstream = FrcstHead(embed_dim+static_dim, 1)

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(nn.Linear(static_dim, embed_dim),
                                          nn.Tanh(),
                                          nn.Linear(embed_dim, static_dim)) # changed demo embedding to go from static->2*emb->emb
                                                                             # to static -> emb -> static

        # ------------------------------
        # (4) 설정
        # ------------------------------
 
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{self.final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type == 'balanced'
        else: 
            self.final_emb_type = final_emb_type
        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
   
    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):
        """
        - pretrain=True  => 마스킹 기반 시계열 예측 (forecast) 수행
        - pretrain_mask : (batch_size, seq_len) - bool (True=mask)
        - freeze_pretrained=True => pretrain 모듈 파라미터 고정 (Transformer, embeddings, etc)
        """

        # -----------------------------------------
        # 1) Freeze if requested (downstream only)
        # -----------------------------------------
        if freeze_pretrained and not pretrain:
            for param in self.transformer_blocks.parameters():
                param.requires_grad = False
            for param in self.fusion_attention.parameters():
                param.requires_grad = False
            for param in self.time_embed.parameters():
                param.requires_grad = False
            for param in self.value_embed.parameters():
                param.requires_grad = False
            for param in self.feature_embed.parameters():
                param.requires_grad = False
            for param in self.forecast_head.parameters():
                param.requires_grad = False
        else:
            for param in self.transformer_blocks.parameters():
                param.requires_grad = True
            for param in self.fusion_attention.parameters():
                param.requires_grad = True
            for param in self.time_embed.parameters():
                param.requires_grad = True
            for param in self.value_embed.parameters():
                param.requires_grad = True
            for param in self.feature_embed.parameters():
                param.requires_grad = True
            for param in self.forecast_head.parameters():
                param.requires_grad = True
        # -----------------------------------------
        # 2) Embeddings
        # -----------------------------------------
        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [bsz, 1, embed_dim]



        time_emb = self.time_embed(times.unsqueeze(-1))
        value_emb = self.value_embed(values.unsqueeze(-1))
        feature_emb = self.feature_embed(varis)

        # -----------------------------------------
        # 3) Pretrain => use pretrain_mask
        #    pretrain_mask[i,j] = True => 그 시점(mask)
        # -----------------------------------------
        if pretrain:
            # If pretrain : only use embedding of values that are not masked
            # mask=1 => "keep" in old logic, so we invert from dataset's bool 
            # dataset says True=> mask => zero out
            # in old code: mask=1 => unmasked => multiply => original
            # => we define: net_mask = 1 - pretrain_mask.float()
            # so if pretrain_mask=1 => net_mask=0 => zero out
            net_mask = 1 - pretrain_mask.float()  # shape: [bsz, seq_len]
            masked_value_emb = value_emb * net_mask.unsqueeze(-1)
            triplet_emb = time_emb + masked_value_emb + feature_emb
        else:
            # Use embedding of all obserbed value(input)
            triplet_emb = time_emb + value_emb + feature_emb

        triplet_emb = self.dropout(triplet_emb)
        # CLS concat
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)

        cls_pad = torch.ones((bsz,1), dtype=torch.bool, device=padding_mask.device)
        att_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)
        fus_padding_mask = torch.cat([torch.zeros_like(cls_pad), padding_mask], dim=1).float()

        # -----------------------------------------
        # 4) Transformer Blocks
        # -----------------------------------------
        for block in self.transformer_blocks:
            triplet_emb = block(triplet_emb, att_padding_mask)

        # CLS pooling
        cls_emb = self.cls_head(triplet_emb[:, 0, :])

        # -----------------------------------------
        # 5) Final Embedding (fusion vs cls etc.)
        # -----------------------------------------
        if self.final_emb_type == 'balanced':
            fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask) # Ignore when mask==1
            fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
            fusion_emb = self.layer_norm(fusion_emb)
            final_emb = self.fusion_emb_weight * fusion_emb + (1 - self.fusion_emb_weight) * cls_emb
        elif self.final_emb_type == 'cls':
            final_emb = cls_emb
        elif self.final_emb_type == 'fusion':
            fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
            fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
            fusion_emb = self.layer_norm(fusion_emb)
            final_emb = fusion_emb
        demo_emb = self.demo_emb(statics)
        # -----------------------------------------
        # 6) Pretrain => Forecast Head
        # -----------------------------------------
        if pretrain:
            # [bsz, seq_len+1, embed_dim]
            triplet_emb_seq = triplet_emb[:, 1:, :]  # CLS 제외
            seq_emb = (1 - self.final_emb_weight) * triplet_emb_seq + self.final_emb_weight * final_emb.unsqueeze(1)
            # seq_emb = torch.cat((seq_emb, demo_emb), dim=-1)
            forecast = self.forecast_head(seq_emb)    # [bsz, seq_len, num_features+1]
            gather_input = varis.unsqueeze(-1) 

            # 1) gather & squeeze
            forecast_selected = torch.gather(forecast, 2, gather_input)  # → [B, L, 1]
            forecast_selected = forecast_selected.squeeze(-1)                   # → [B, L]

            # 2) 준비된 mask (both are [B, L]) padding false, pre true인 mask들 계산
            net_mask = 1 - pretrain_mask.float()   # unmasked=1, masked=0
            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())      # masked=1, else=0

            # 3) loss 계산
            pred_masked = forecast_selected * masked_loss  # [B, L]
            gt_masked = values * masked_loss  # [B, L]
            diff = pred_masked - gt_masked
            sq_diff = diff ** 2
            # sum over all masked positions
            sum_sq = sq_diff.sum()
            cnt_masked = masked_loss.sum()
            mse_loss = sum_sq / cnt_masked if cnt_masked>0 else torch.tensor(0., device=values.device)

            return {
                'forecast': forecast_selected,
                'values': values,
                'varis': varis,
                'times': times,
                'mask': pretrain_mask,  # (bool)
                'loss': mse_loss
            }

        # -----------------------------------------
        # 7) Downstream => Death
        # -----------------------------------------
        else:
            final_emb = torch.cat((final_emb, demo_emb), dim=-1)
            pred = torch.sigmoid(self.downstream(final_emb)).squeeze(-1)
            #pred = self.downstream(final_emb)
            return pred, final_emb

# Models with gradient reversal

class STraTSModelGR(nn.Module): # Single task
    """
    Altered model definition for the STraTS task, 
    Gradient reversal is added to adapt DANN
    i.e. Domain Adversarial Nerual Network
    """
    def __init__(self, 
                num_features,
                embed_dim=32,
                static_dim=3, 
                num_heads=4, 
                num_blocks=2, 
                ff_dim=64,
                dropout=0.2, 
                time_activation='relu', 
                value_activation='tanh', 
                final_emb_type='balanced', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,
                domain_lambda=1.0
                ):
        super().__init__()

        # ------------------------------
        # (1) Transformer + Embeddings
        # ------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.fusion_attention = FusionAttention(embed_dim)
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features+1, embed_dim, padding_idx=num_features) # Feats + padding 1

        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(embed_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # ------------------------------
        # (2) Downstream heads
        # ------------------------------

        self.downstream_head = FrcstHead(2*embed_dim, 1)
        self.domain_head = FrcstHead(2*embed_dim, 1)
        self.domain_lambda = domain_lambda

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(nn.Linear(static_dim, embed_dim),
                                          nn.Tanh(),
                                          nn.Linear(embed_dim, embed_dim)) # changed demo embedding to go from static->2*emb->emb
                                                                             # to static -> emb -> static

        # ------------------------------
        # (4) 설정
        # ------------------------------
 
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{self.final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type == 'balanced'
        else: 
            self.final_emb_type = final_emb_type
        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
   
    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):
        """
        - pretrain=True  => 마스킹 기반 시계열 예측 (forecast) 수행
        - pretrain_mask : (batch_size, seq_len) - bool (True=mask)
        - freeze_pretrained=True => pretrain 모듈 파라미터 고정 (Transformer, embeddings, etc)
        """

        # -----------------------------------------
        # 1) Freeze if requested (downstream only)
        # -----------------------------------------
        if freeze_pretrained and not pretrain:
            for param in self.transformer_blocks.parameters():
                param.requires_grad = False
            for param in self.fusion_attention.parameters():
                param.requires_grad = False
            for param in self.time_embed.parameters():
                param.requires_grad = False
            for param in self.value_embed.parameters():
                param.requires_grad = False
            for param in self.feature_embed.parameters():
                param.requires_grad = False
            for param in self.forecast_head.parameters():
                param.requires_grad = False
        else:
            for param in self.transformer_blocks.parameters():
                param.requires_grad = True
            for param in self.fusion_attention.parameters():
                param.requires_grad = True
            for param in self.time_embed.parameters():
                param.requires_grad = True
            for param in self.value_embed.parameters():
                param.requires_grad = True
            for param in self.feature_embed.parameters():
                param.requires_grad = True
            for param in self.forecast_head.parameters():
                param.requires_grad = True
        # -----------------------------------------
        # 2) Embeddings
        # -----------------------------------------
        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [bsz, 1, embed_dim]

        time_emb = self.time_embed(times.unsqueeze(-1))
        value_emb = self.value_embed(values.unsqueeze(-1))
        feature_emb = self.feature_embed(varis)

        # -----------------------------------------
        # 3) Pretrain => use pretrain_mask
        #    pretrain_mask[i,j] = True => 그 시점(mask)
        # -----------------------------------------
        if pretrain:
            # If pretrain : only use embedding of values that are not masked
            # mask=1 => "keep" in old logic, so we invert from dataset's bool 
            # dataset says True=> mask => zero out
            # in old code: mask=1 => unmasked => multiply => original
            # => we define: net_mask = 1 - pretrain_mask.float()
            # so if pretrain_mask=1 => net_mask=0 => zero out
            net_mask = 1 - pretrain_mask.float()  # shape: [bsz, seq_len]
            masked_value_emb = value_emb * net_mask.unsqueeze(-1)
            triplet_emb = time_emb + masked_value_emb + feature_emb
        else:
            # Use embedding of all obserbed value(input)
            triplet_emb = time_emb + value_emb + feature_emb

        triplet_emb = self.dropout(triplet_emb)
        # CLS concat
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)

        cls_pad = torch.ones((bsz,1), dtype=torch.bool, device=padding_mask.device)
        att_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)
        # Final fusion mask does not include CLS token
        fus_keep = (~padding_mask).float()
        fus_padding_mask = torch.cat([torch.zeros_like(cls_pad), fus_keep], dim=1).float()

        # -----------------------------------------
        # 4) Transformer Blocks
        # -----------------------------------------
        for block in self.transformer_blocks:
            triplet_emb = block(triplet_emb, att_padding_mask)

        # CLS pooling
        cls_emb = self.cls_head(triplet_emb[:, 0, :])

        # -----------------------------------------
        # 5) Final Embedding (fusion vs cls etc.)
        # -----------------------------------------
        if self.final_emb_type == 'balanced':
            fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask) # Ignore when mask==1
            fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
            fusion_emb = self.layer_norm(fusion_emb)
            final_emb = self.fusion_emb_weight * fusion_emb + (1 - self.fusion_emb_weight) * cls_emb
        elif self.final_emb_type == 'cls':
            final_emb = cls_emb
        elif self.final_emb_type == 'fusion':
            fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
            fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
            fusion_emb = self.layer_norm(fusion_emb)
            final_emb = fusion_emb
        demo_emb = self.demo_emb(statics)
        # -----------------------------------------
        # 6) Pretrain => Forecast Head
        # -----------------------------------------
        if pretrain:
            # 1) prepare sequence embedding as before
            triplet_emb_seq = triplet_emb[:, 1:, :]  # drop CLS
            seq_emb = (1 - self.final_emb_weight) * triplet_emb_seq \
                      + self.final_emb_weight * final_emb.unsqueeze(1)

            # 2) time‑series forecast
            forecast = self.forecast_head(seq_emb)     # [B, L, F+1]
            forecast_selected = torch.gather(
                forecast, 2, varis.unsqueeze(-1)
            ).squeeze(-1)                              # [B, L]

            # 3) forecast loss
            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff   = forecast_selected * masked_loss - values * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            # 4) adversarial domain prediction on the _pooled_ final_emb
            #    (append demo_emb just like downstream)
            dom_input = torch.cat((final_emb, demo_emb), dim=-1)
            dom_rev = grad_reverse(dom_input, self.domain_lambda)
            dom_logit = self.domain_head(dom_rev).squeeze(-1)


            return {
                'forecast': forecast_selected,
                'values': values,
                'varis': varis,
                'times': times,
                'mask': pretrain_mask,
                'loss': mse_loss,
                'dom_logit': dom_logit
            }


        # -----------------------------------------
        # 7) Downstream => Outcome & Domain
        # -----------------------------------------
        else:
            # Outcome prediction
            final_emb = torch.cat((final_emb, demo_emb), dim=-1)
            pred = self.downstream_head(final_emb).squeeze(-1)
            # pred = torch.sigmoid(self.downstream_head(final_emb)).squeeze(-1)
            # pred = pred.clamp(min=1e-7, max=1-1e-7)

            # Domain prediction
            # reverse_grad = grad_reverse(final_emb, self.domain_lambda)
            # dom_logit = self.domain_head(reverse_grad).squeeze(-1)
            # pred_dom = torch.sigmoid(dom_logit)
            # pred_dom = pred_dom.clamp(min=1e-7, max=1-1e-7)
            #pred = self.downstream(final_emb)
            return {
                'pred' : pred,
                #'pred_domain' : pred_dom,
                'embs' : final_emb
            }


# Varwise model with gradient reversal

class STraTSGRVar(nn.Module): # Single task
    """
    Altered model definition for the STraTS task, 
    Gradient reversal is added to adapt DANN
    Variable-group wise embedding is added to ease interpretation
    as Dict[group_name(str) → List[int]]
    i.e. Domain Adversarial Nerual Network
    """
    def __init__(self, 
                num_features,
                var_groups,
                embed_dim=32,
                static_dim=3, 
                num_heads=4, 
                num_blocks=2, 
                ff_dim=64,
                dropout=0.2, 
                time_activation='relu', 
                value_activation='tanh', 
                final_emb_type='balanced', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,
                domain_lambda=1.0
                ):
        super().__init__()

        self.var_groups = var_groups
        # ------------------------------
        # (1) Transformer + Embeddings
        # ------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features+1, embed_dim, padding_idx=num_features) # Feats + padding 1

        self.group_atts = nn.ModuleDict({
            name: FusionAttention(embed_dim)
            for name in var_groups
        })
        
        final_emb_dim = embed_dim * (len(var_groups) + 1)
        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(embed_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    

        # ------------------------------
        # (2) Downstream heads
        # ------------------------------

        self.downstream_head = FrcstHead(final_emb_dim, 1)
        self.domain_head = FrcstHead(final_emb_dim, 1)
        self.domain_lambda = domain_lambda

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(nn.Linear(static_dim, embed_dim),
                                          nn.Tanh(),
                                          nn.Linear(embed_dim, embed_dim)) # changed demo embedding to go from static->2*emb->emb
                                                                             # to static -> emb -> emb

        # ------------------------------
        # (4) 설정
        # ------------------------------
 
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{self.final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type == 'balanced'
        else: 
            self.final_emb_type = final_emb_type
        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
   
    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):
        """
        - pretrain=True  => 마스킹 기반 시계열 예측 (forecast) 수행
        - pretrain_mask : (batch_size, seq_len) - bool (True=mask)
        - freeze_pretrained=True => pretrain 모듈 파라미터 고정 (Transformer, embeddings, etc)
        """
        # -----------------------------------------
        # 2) Embeddings
        # -----------------------------------------
        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [bsz, 1, embed_dim]

        time_emb = self.time_embed(times.unsqueeze(-1))
        value_emb = self.value_embed(values.unsqueeze(-1))
        feature_emb = self.feature_embed(varis)

        # 2) --- Pretrain mask 적용 (값만 0으로) ---
        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)      # [B, L, 1]
            value_emb = value_emb * keep

        # 3) 합산 + dropout + CLS
        triplet_emb = self.dropout(time_emb + value_emb + feature_emb)  # [B, L, D]
        # CLS concat
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)

        cls_pad = torch.ones((bsz,1), dtype=torch.bool, device=padding_mask.device)
        att_mask  = torch.cat([cls_pad, padding_mask], dim=1)

        # -----------------------------------------
        # 4) Transformer Blocks
        # -----------------------------------------
        for block in self.transformer_blocks:
            triplet_emb = block(triplet_emb, att_mask)

        # CLS pooling -> Deprecated : use fusion attention to summarize triplet embeddings
        # cls_emb = self.cls_head(triplet_emb[:, 0, :])

        # -----------------------------------------
        # 5) Groupwise Embedding 
        # -----------------------------------------
        group_embs = []
        for name, idx_list in self.var_groups.items():
            # a) var_mask: 해당 그룹 변수인지
            var_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=varis.device)
            for vid in idx_list:
                var_mask |= (varis == vid)

            # b) valid_mask: exclude padding and pretrain mask
            pad_pre_mask = padding_mask if not pretrain else (padding_mask | pretrain_mask)
            valid_mask = ~pad_pre_mask  # [B, L]
            event_mask = valid_mask & var_mask

            # c) Use cls tokens as fallback in case there are no observations
            cls_mask = torch.ones((bsz,1), dtype=torch.bool, device=event_mask.device)
            mask_evt = torch.cat([cls_mask, event_mask], dim=1).float()  # [B, L+1] 

            weights = self.group_atts[name](triplet_emb, mask_evt)
            emb = (triplet_emb * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
            group_embs.append(self.layer_norm(emb))
        demo_emb = self.demo_emb(statics)

 # -----------------------------------------
        # 6) Pretrain => Forecast Head + Domain Head
        # -----------------------------------------
        if pretrain:
            # (a) build the “embed_dim”‑sized final_emb used for seq prediction
            final_emb_small = demo_emb
            for g in group_embs:
                final_emb_small = final_emb_small + g                       # [B, embed_dim]

            # (b) masked‐forecast exactly as before
            triplet_emb_seq = triplet_emb[:, 1:, :]                         # drop CLS
            seq_emb = ((1 - self.final_emb_weight) * triplet_emb_seq
                       + self.final_emb_weight * final_emb_small.unsqueeze(1))
            forecast = self.forecast_head(seq_emb)                         # [B, L, F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1))  
            forecast_sel = forecast_sel.squeeze(-1)                         # [B, L]

            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            # # (c) now build the “full” final_emb (concat groupwise+demo) for domain
            # final_emb_full = torch.cat(group_embs + [demo_emb], dim=-1)     # [B, D_full]
            # rev = grad_reverse(final_emb_full, self.domain_lambda)
            # dom_logit = self.domain_head(rev).squeeze(-1)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
                # 'dom_logit': dom_logit
            }
        # -----------------------------------------
        # 7) Downstream => Outcome & Domain
        # -----------------------------------------
        else:
            final_emb = torch.cat(group_embs + [demo_emb], dim=-1) 

            # Downstream prediction
            pred = self.downstream_head(final_emb).squeeze(-1)
            # pred = torch.sigmoid(self.downstream_head(final_emb)).squeeze(-1)
            # pred = pred.clamp(min=1e-7, max=1-1e-7)

            # # Domain prediction
            reverse_grad = grad_reverse(final_emb, self.domain_lambda)
            dom_logit = self.domain_head(reverse_grad).squeeze(-1)
            # pred_dom = torch.sigmoid(dom_logit)
            # pred_dom = pred_dom.clamp(min=1e-7, max=1-1e-7)
            # #pred = self.downstream(final_emb)
            return {
                'pred' : pred,
                'pred_domain' : dom_logit,
                'embs' : final_emb
            }
        
# Varwise model with gradient reversal

class SurpriseSTraTSGRVar(nn.Module): # Single task
    """
    Altered model definition for the STraTS task,
    Surprise metric defined by cosine similarity of triplet embeddings 
    Gradient reversal is added to adapt DANN
    Variable-group wise embedding is added to ease interpretation
    as Dict[group_name(str) → List[int]]
    i.e. Domain Adversarial Nerual Network
    """
    def __init__(self, 
                num_features,
                var_groups,
                embed_dim=32,
                static_dim=3, 
                num_heads=4, 
                num_blocks=2, 
                ff_dim=64,
                dropout=0.2, 
                time_activation='relu', 
                value_activation='tanh', 
                final_emb_type='balanced', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,
                domain_lambda=1.0,
                sim_threshold=0.95
                ):
        super().__init__()

        self.var_groups = var_groups
        # ------------------------------
        # (1) Transformer + Embeddings
        # ------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features+1, embed_dim, padding_idx=num_features) # Feats + padding 1

        self.group_atts = nn.ModuleDict({
            name: FusionAttention(embed_dim)
            for name in var_groups
        })
        
        final_emb_dim = embed_dim * (len(var_groups) + 1)
        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(embed_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    

        # ------------------------------
        # (2) Downstream heads
        # ------------------------------

        self.downstream_head = FrcstHead(final_emb_dim, 1)
        self.domain_head = FrcstHead(final_emb_dim, 1)
        self.domain_lambda = domain_lambda

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(nn.Linear(static_dim, embed_dim),
                                          nn.Tanh(),
                                          nn.Linear(embed_dim, embed_dim)) # changed demo embedding to go from static->2*emb->emb
                                                                             # to static -> emb -> emb

        # ------------------------------
        # (4) 설정
        # ------------------------------
 
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{self.final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type == 'balanced'
        else: 
            self.final_emb_type = final_emb_type
        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
        self.sim_threshold = sim_threshold
            # ----- var_id → group_idx 맵핑 생성 (벡터화용) -----
        group_names = list(var_groups.keys())
        var_to_group = torch.full((num_features + 1,), -1, dtype=torch.long)  # padding 포함
        for g_idx, g_name in enumerate(group_names):
            for vid in var_groups[g_name]:
                var_to_group[vid] = g_idx
        # buffer로 등록해서 device 이동 자동화
        self.register_buffer("var_to_group", var_to_group, persistent=False)

   
    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):

        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]

        # --- Embeddings ---
        time_emb   = self.time_embed(times.unsqueeze(-1))     # [B, L, D]
        value_emb  = self.value_embed(values.unsqueeze(-1))   # [B, L, D]
        feature_emb= self.feature_embed(varis)                # [B, L, D]

        # pretrain: 값 마스킹(기존 동작 유지)
        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)  # [B, L, 1]
            value_emb = value_emb * keep

        # (A) Dropout 이전, triplet 합산 (base embedding)
        triplet_base = time_emb + value_emb + feature_emb     # [B, L, D]

        # ================== 벡터화 하드게이팅 시작 ==================
        triplet_norm = F.normalize(triplet_base, p=2, dim=-1)   # [B, L, D]
        sim_mat = torch.bmm(triplet_norm, triplet_norm.transpose(1, 2))  # [B, L, L]

        same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))    # [B, L, L]
        valid = (~padding_mask)
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)    # [B, L, L]
        # 과거(i<j)만 허용
        upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=varis.device),
                        diagonal=1)  # [L, L]
        upper = upper.unsqueeze(0)                                       # [1, L, L]

        pair_hit = (sim_mat >= self.sim_threshold) & same_var & upper & valid_pair
        redundant_mask = pair_hit.any(dim=1)  # [B, L], 오른쪽(j) 토큰이 True
        # ================== 벡터화 하드게이팅 끝 ==================

        # (C) 이후 전체 파이프라인에 쓸 새 padding 마스크 구성
        new_padding_mask = padding_mask | redundant_mask  # [B, L] (True=pad)

        # (D) 드롭아웃/CLS concat
        triplet_emb = self.dropout(triplet_base)          # [B, L, D]
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)  # [B, L+1, D]

        # (E) Transformer attn mask 업데이트 (CLS는 pad로 취급하지 않음)
        cls_pad = torch.ones((bsz,1), dtype=torch.bool, device=new_padding_mask.device) # 이거 zero로 하면 일반화 성능이 꼬라박음? (아마도? as of 250820 -> 아닌듯)
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)   # [B, L+1]

        # --- Transformer Blocks ---
        for block in self.transformer_blocks: # Ignores "True"
            triplet_emb = block(triplet_emb, att_mask)

        # --- Groupwise Embedding (FusionAttention) ---
        group_embs = []
        for name, idx_list in self.var_groups.items():
            # a) var_mask: 해당 그룹 변수인지
            var_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=varis.device)
            for vid in idx_list:
                var_mask |= (varis == vid)

            # b) valid_mask: padding + (pretrain 시 pretrain_mask) + redundancy를 제외
            if pretrain:
                pad_pre_mask = (new_padding_mask | pretrain_mask)  # 여기에 redundancy 포함됨
            else:
                pad_pre_mask = new_padding_mask
            valid_mask = ~pad_pre_mask                             # [B, L]
            event_mask = valid_mask & var_mask

            # c) CLS fallback
            cls_mask = torch.ones((bsz,1), dtype=torch.bool, device=event_mask.device)
            mask_evt = torch.cat([cls_mask, event_mask], dim=1).float()  # [B, L+1]

            weights = self.group_atts[name](triplet_emb, mask_evt) # Keeps "True"
            emb = (triplet_emb * weights.unsqueeze(-1)).sum(dim=1)       # [B, D]
            group_embs.append(self.layer_norm(emb))

        demo_emb = self.demo_emb(statics)

        # --- Pretrain (forecast) ---
        if pretrain:
            # (a) 최종 임베딩 (작은) 구성
            final_emb_small = demo_emb
            for g in group_embs:
                final_emb_small = final_emb_small + g                     # [B, D]

            # (b) forecast
            triplet_emb_seq = triplet_emb[:, 1:, :]                       # drop CLS
            seq_emb = ((1 - self.final_emb_weight) * triplet_emb_seq
                    + self.final_emb_weight * final_emb_small.unsqueeze(1))
            forecast = self.forecast_head(seq_emb)                        # [B, L, F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1)  # [B, L]

            # (c) Loss 계산 시: pretrain_mask & NOT(new_padding_mask)
            #     → redundancy로 마스킹된 토큰은 loss에서 제외
            masked_loss = pretrain_mask.float() * (1 - new_padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
            }

        # --- Downstream ---
        else:
            final_emb = torch.cat(group_embs + [demo_emb], dim=-1)        # [B, D_full]
            pred = self.downstream_head(final_emb).squeeze(-1)

            reverse_grad = grad_reverse(final_emb, self.domain_lambda)
            dom_logit = self.domain_head(reverse_grad).squeeze(-1)

            return {
                'pred' : pred,
                'pred_domain' : dom_logit,
                'embs' : final_emb
            }

    def check_padding(self,
                      times: torch.Tensor,        # [B, L]
                      varis: torch.Tensor,        # [B, L] (int64)
                      values: torch.Tensor,       # [B, L]
                      padding_mask: torch.Tensor  # [B, L] (bool, True=pad)
                      ):
        """
        Debug helper:
        - triplet embedding(time+value+feature)로 변환
        - 같은 variable-group 내 cosine similarity 기반 하드게이팅 마스크(redundant_mask) 계산
        - 원본 입력과 redundant_mask를 함께 반환
        Return dict: {'times', 'varis', 'values', 'mask'}  # mask = redundant_mask
        """
        bsz, seq_len = values.size()

        # 1) Triplet base embedding (dropout/Transformer 적용 이전과 동일)
        time_emb    = self.time_embed(times.unsqueeze(-1))     # [B, L, D]
        value_emb   = self.value_embed(values.unsqueeze(-1))   # [B, L, D]
        feature_emb = self.feature_embed(varis)                # [B, L, D]
        triplet_base = time_emb + value_emb + feature_emb      # [B, L, D]

        # 2) Cosine similarity 계산을 위한 정규화
        triplet_norm = torch.nn.functional.normalize(triplet_base, p=2, dim=-1)  # [B, L, D]

        # 3) 배치별 전쌍 cosine 유사도 행렬: [B, L, L]
        sim_mat = torch.bmm(triplet_norm, triplet_norm.transpose(1, 2))

        # 4) 같은 variable group 여부 (미매핑(-1) 제거 포함)
        group_idx = self.var_to_group[varis]                                    # [B, L]
        same_group = group_idx.unsqueeze(2).eq(group_idx.unsqueeze(1))          # [B, L, L]
        valid_group = (group_idx != -1)
        same_group = same_group & valid_group.unsqueeze(2) & valid_group.unsqueeze(1)

        # 5) 유효 토큰(패딩 제외) & 과거 토큰(아래 삼각) 마스크
        valid = (~padding_mask)                                                 # [B, L]
        lower = torch.tril(torch.ones(seq_len, seq_len,
                                    dtype=torch.bool, device=varis.device), diagonal=-1)  # [L, L]
        lower = lower.unsqueeze(0)                                              # [1, L, L]
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)                    # [B, L, L]

        # 6) 임계치 이상 & 같은 그룹 & 과거 & 유효 쌍 필터
        pair_hit = (sim_mat >= self.sim_threshold) & same_group & lower & valid_pair  # [B, L, L]

        # 7) 열 기준(any)로 현재 토큰이 게이팅 대상인지 결정
        redundant_mask = pair_hit.any(dim=1)  # [B, L], True면 이후 단계에서 padding 취급 예정

        return {
            'times':  times,
            'varis':  varis,
            'values': values,
            'mask':   redundant_mask,  # hard gating으로 제거되는 토큰 표시
        }

class Time2Vec(nn.Module):
    """
    T2V(t) = [w0 * t + b0,  sin(w1 * t + b1), ..., sin(wk * t + bk)]
    - 입력 t는 [B, L] (float). 0~1로 정규화되어 있다고 가정.
    - 출력 shape: [B, L, d_model]
    """
    def __init__(self, d_model: int, scale_to_2pi: bool = True, use_cos: bool = False):
        super().__init__()
        assert d_model >= 2, "Time2Vec: d_model >= 2 권장(선형 1 + 주기 d_model-1)."
        self.d_model = d_model
        self.scale_to_2pi = scale_to_2pi
        self.use_cos = use_cos

        # 선형 성분 (1차원)
        self.lin = nn.Linear(1, 1)                      # w0, b0
        # 주기 성분 (d_model-1 차원)
        self.per = nn.Linear(1, d_model - 1, bias=True) # w_i, b_i

        # 초기화(무난한 Xavier)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.xavier_uniform_(self.per.weight)
        nn.init.zeros_(self.per.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, L]
        x = t.unsqueeze(-1)  # [B, L, 1]
        if self.scale_to_2pi:
            x = x * (2 * math.pi)  # 0~1 -> 0~2π

        linear = self.lin(x)                      # [B, L, 1]
        periodic_arg = self.per(x)                # [B, L, d_model-1]
        periodic = torch.cos(periodic_arg) if self.use_cos else torch.sin(periodic_arg)
        return torch.cat([linear, periodic], dim=-1)  # [B, L, d_model]

class STraTS(nn.Module): # Single task
    """
    Altered model definition for the STraTS task,
    Surprise metric defined by cosine similarity of triplet embeddings 
    Gradient reversal is added to adapt DANN
    Variable-group wise embedding is added to ease interpretation
    as Dict[group_name(str) → List[int]]
    i.e. Domain Adversarial Nerual Network
    """
    def __init__(self, 
                num_features,
                embed_dim=32,
                static_dim=3, 
                num_heads=4, 
                num_blocks=2, 
                ff_dim=64,
                dropout=0.2, 
                time_activation='relu', 
                value_activation='tanh', 
                final_emb_type='balanced', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,

                ):
        super().__init__()

        # ------------------------------
        # (1) Transformer + Embeddings
        # ------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features+1, embed_dim, padding_idx=num_features) # Feats + padding 1

        self.fusion_attention = FusionAttention(embed_dim)
            
        
        final_emb_dim = embed_dim * 2
        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(final_emb_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_time   = nn.LayerNorm(embed_dim)     # for Time path
        self.ln_value  = nn.LayerNorm(embed_dim)     # for Value path
        self.ln_feat   = nn.LayerNorm(embed_dim)     # for Feature(nn.Embedding) path
        # ------------------------------
        # (2) Downstream heads
        # ------------------------------

        self.downstream_head = FrcstHead(final_emb_dim, 1)

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(nn.Linear(static_dim, embed_dim),
                                          nn.Tanh(),
                                          nn.Linear(embed_dim, embed_dim)) # changed demo embedding to go from static->2*emb->emb
                                                                             # to static -> emb -> emb

        # ------------------------------
        # (4) 설정
        # ------------------------------
 
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{self.final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type == 'balanced'
        else: 
            self.final_emb_type = final_emb_type
        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
   
    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None):

        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]

        # --- Embeddings ---
        time_emb = self.ln_time(self.time_embed(times.unsqueeze(-1)))
        value_emb = self.ln_value(self.value_embed(values.unsqueeze(-1)))
        feature_emb = self.ln_feat(self.feature_embed(varis))

        # pretrain: 값 마스킹(기존 동작 유지)
        # Consider changing this to fixed "MASK" token instead of zero
        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)  # [B, L, 1]
            value_emb = value_emb * keep

        # (A) Dropout 이전, triplet 합산 (base embedding)
        triplet_base = time_emb + value_emb + feature_emb     # [B, L, D]
        if pretrain:
            new_padding_mask = padding_mask | pretrain_mask
        else:
            # (C) 이후 전체 파이프라인에 쓸 새 padding 마스크 구성
            new_padding_mask = padding_mask

        # (D) 드롭아웃/CLS concat
        triplet_emb = self.dropout(triplet_base)          # [B, L, D]
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)  # [B, L+1, D]

        # (E) Transformer attn mask 업데이트 (CLS는 pad로 취급하지 않음)
        cls_pad = torch.zeros((bsz,1), dtype=torch.bool, device=new_padding_mask.device)
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)   # [B, L+1]

        # --- Transformer Blocks ---
        for block in self.transformer_blocks: # Ignores "True"
            triplet_emb = block(triplet_emb, att_mask)

        # Final fusion mask does not include CLS token
        # i. e. CLS token is used for computing the final fusion attention
        # This is a fallback in case all tokens are masked (by some bad luck)
        fus_keep = (~new_padding_mask).float()
        fus_padding_mask = torch.cat([torch.ones_like(cls_pad), fus_keep], dim=1).float()

        # -----------------------------------------
        # 5) Final Embedding (fusion vs cls etc.)
        # -----------------------------------------

        fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
        fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fusion_emb = self.layer_norm(fusion_emb)
        demo_emb = self.demo_emb(statics)

        # --- Pretrain (forecast) ---
        if pretrain:
            # [B, D]

            # (b) forecast
            seq_emb = ((1 - self.final_emb_weight) * triplet_base
                    + self.final_emb_weight * fusion_emb.unsqueeze(1))
            demo_seq = demo_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
            final_emb = torch.cat((seq_emb, demo_seq), dim=-1)
            
            forecast = self.forecast_head(final_emb) # [B, L, F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1) # [B, L]

            # (c) Loss 계산 시: pretrain_mask & NOT(padding_mask)
            #     → redundancy로 마스킹된 토큰은 loss에서 제외
            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
            }

        # --- Downstream ---
        else:
            final_emb = torch.cat((fusion_emb, demo_emb), dim=-1)        # [B, D_full]
            pred = self.downstream_head(final_emb).squeeze(-1)

            return {
                'pred' : pred,
                'embs' : final_emb
            }

@torch.jit.script
def surprise_redundant_mask_from_pairhit(
    pair_hit: torch.Tensor,  # [B, L, L] bool, (i<j) 후보만 True
    valid: torch.Tensor,     # [B, L]     bool
    window_W: int            # 0이면 제한 없음, >0이면 최근 W개 과거만 비교
) -> torch.Tensor:
    B = pair_hit.size(0)
    L = pair_hit.size(1)
    keep = torch.zeros((B, L), dtype=torch.bool, device=pair_hit.device)

    if L > 0:
        keep[:, 0] = valid[:, 0]

    for j in range(1, L):
        # 최근 W개 과거만 보게 범위 제한 (0이면 전체)
        if window_W > 0:
            start = j - window_W
            if start < 0:
                start = 0
        else:
            start = 0

        prev_keep = keep[:, start:j]                 # [B, J]
        col_hit   = pair_hit[:, start:j, j]          # [B, J], Threshold 넘는 유사도면 True
        blocked   = (col_hit & prev_keep).any(dim=1) # [B], Keep인 Token에 대해 Threshold를 넘는 유사도를 가지면 Block
        keep[:, j] = valid[:, j] & (~blocked)

    redundant = valid & (~keep) # Keep이 False고 Valid가 True여야만 Redundant하게 취급
    return redundant

class SurpriseSTraTS(nn.Module):  # per-variable gating + global FusionAttention
    """
    - Hard gating: same variable (per-var) 기준
    - FusionAttention: 그룹별이 아닌 전체 시퀀스에 대해 단일 attention으로 집약
    - var_groups 제거
    """
    def __init__(self, 
                 num_features,
                 embed_dim=32,
                 static_dim=3, 
                 num_heads=4, 
                 num_blocks=2, 
                 ff_dim=64,
                 dropout=0.2, 
                 time_activation='relu', 
                 value_activation='tanh', 
                 final_emb_type='balanced', 
                 fusion_emb_weight=0.5,
                 final_emb_weight=0.5,
                 domain_lambda=1.0,
                 sim_threshold=0.95):
        super().__init__()

        # (1) Transformer + Embeddings
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features + 1, embed_dim, padding_idx=num_features)  # Feats + padding 1

        # 단일 FusionAttention (전체 변수에 대해)
        self.fusion_attention = FusionAttention(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)


        # Downstream heads
        # 최종 임베딩: [vars_fused_emb, demo_emb] → 2 * D
        final_emb_dim = embed_dim * 2
        self.downstream_head = FrcstHead(final_emb_dim, 1)
        # pretrain용 head (토큰 임베딩에서 각 변수값 예측)
        self.forecast_head = FrcstHead(final_emb_dim, num_features + 1)
        self.domain_head = FrcstHead(final_emb_dim, 1)
        self.domain_lambda = domain_lambda

        # (3) CLS token & heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

        # (4) 설정
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type = 'balanced'
        else:
            self.final_emb_type = final_emb_type

        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
        self.sim_threshold = sim_threshold

        self.padding_idx = num_features

    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):

        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]

        # --- Embeddings ---
        time_emb = self.time_embed(times.unsqueeze(-1))
        value_emb = self.value_embed(values.unsqueeze(-1))
        feature_emb = self.feature_embed(varis)

        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)    # [B, L, 1]
            value_emb = value_emb * keep

        # (A) base embedding
        triplet_base = time_emb + value_emb + feature_emb       # [B, L, D]        
            
        # # ======== 하드게이팅: "같은 변수" 기준 ========
        # with torch.no_grad():  # 마스크 생성만 하므로 grad 불필요
        #     # [B, L, D] -> [B, L, L]
        #     triplet_norm = F.normalize(triplet_base, p=2, dim=-1)
        #     sim_mat = torch.bmm(triplet_norm, triplet_norm.transpose(1, 2))

        #     valid = (~padding_mask)                                        # [B, L]
        #     same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))           # [B, L, L]

        #     # 상삼각(과거 i<j만) 제한
        #     upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=varis.device), diagonal=1)
        #     upper = upper.unsqueeze(0)                                     # [1, L, L]

        #     # 후보 페어 전부 미리 계산: 전부 행렬 연산
        #     pair_hit = (sim_mat >= self.sim_threshold) & same_var & upper  # [B, L, L]

        #     # 좌->우 그리디 스캔: 이전 중 'keep'만 비교 대상으로 사용
        #     B, L = bsz, seq_len
        #     keep = torch.zeros(B, L, dtype=torch.bool, device=varis.device)

        #     # j=0 초기화
        #     if L > 0:
        #         keep[:, 0] = valid[:, 0]

        #     # j=1..L-1
        #     for j in range(1, L):
        #         # 이전 중 현재 j를 접을 수 있는 과거 토큰이 'keep' 상태로 남아있는가?
        #         # pair_hit[:, :j, j] : [B, j]  (과거 vs 현재 j)
        #         blocked = (pair_hit[:, :j, j] & keep[:, :j]).any(dim=1)    # [B]
        #         keep[:, j] = valid[:, j] & (~blocked)

        #     redundant_mask = valid & (~keep)        
        # padding_mask = padding_mask | redundant_mask

        # ===== Surprise masking (JIT scan, zero host sync) =====
        with torch.no_grad():
            # triplet_base는 아래에서 계속 쓰니까 역전파는 원본 유지,
            # 마스킹 계산만 detach된 복사로 진행하여 오버헤드/메모리 감소
            tb_det = triplet_base.detach()

            # 1) 유사도 행렬 (bmm 한 번) — AMP/TF32로 더 가볍게
            torch.backends.cuda.matmul.allow_tf32 = True
            triplet_norm = torch.nn.functional.normalize(tb_det, p=2, dim=-1)  # [B, L, D]
            sim_mat = torch.bmm(triplet_norm, triplet_norm.transpose(1, 2))    # [B, L, L]

            # 2) 마스크들 (전부 텐서 연산, CPU로 끌어오지 않음)
            B, L, _ = sim_mat.shape
            device = sim_mat.device
            valid = (~padding_mask)                                            # [B, L]
            same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))               # [B, L, L]

            # 과거만(i<j). 필요하면 밴드 윈도우로 치환.
            idx = torch.arange(L, device=device)
            upper = (idx[None, :] - idx[:, None]) > 0                          # [L, L]
            upper = upper.unsqueeze(0)                                         # [1, L, L]

            # 후보 페어 (유사도/변수/유효/과거)
            valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)               # [B, L, L]
            pair_hit = (sim_mat >= self.sim_threshold) & same_var & valid_pair & upper  # [B, L, L]

            # 3) 좌→우 그리디 스캔 (전부 텐서, TorchScript 내부 루프만 사용)
            #    window_W로 최근 W개만 비교하고 싶으면 정수로 넣기(예: 128). 정확히 하려면 0.
            window_W = 0
            redundant_mask = surprise_redundant_mask_from_pairhit(pair_hit, valid, window_W)  # [B, L]

        # 최종 패딩 합치기
        padding_mask = padding_mask | redundant_mask

        if pretrain:
            new_padding_mask = padding_mask | pretrain_mask
        else:
            # (C) 이후 전체 파이프라인에 쓸 새 padding 마스크 구성
            new_padding_mask = padding_mask

        # (D) 드롭아웃 / CLS concat
        triplet_emb = self.dropout(triplet_base)                # [B, L, D]
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)  # [B, L+1, D]

        # (E) 어텐션 마스크 (주의: 블록 내부의 mask 의미(True=pad)와 일치 필요)
        cls_pad = torch.zeros((bsz, 1), dtype=torch.bool, device=new_padding_mask.device)  # 원 로직 유지
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)  # [B, L+1]

        # --- Transformer Blocks ---
        for block in self.transformer_blocks: # Ignores True
            triplet_emb = block(triplet_emb, att_mask)

        fus_keep = (~new_padding_mask).float()
        fus_padding_mask = torch.cat([torch.ones_like(cls_pad), fus_keep], dim=1).float()

        # -----------------------------------------
        # 5) Final Embedding (fusion vs cls etc.)
        # -----------------------------------------

        fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
        fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fusion_emb = self.layer_norm(fusion_emb)
        demo_emb = self.demo_emb(statics)

        # --- Pretrain (Self-prediction) ---
        if pretrain:
            # [B, D]

            # Self-prediction
            seq_emb = ((1 - self.final_emb_weight) * triplet_base
                    + self.final_emb_weight * fusion_emb.unsqueeze(1))
            demo_seq = demo_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
            final_emb = torch.cat((seq_emb, demo_seq), dim=-1)
            
            forecast = self.forecast_head(final_emb) # [B, L, F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1) # [B, L]

            # Loss 계산 시: pretrain_mask & NOT(padding_mask)
            #     → redundancy로 마스킹된 토큰(Padding mask에 반영)은 loss에서 제외
            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
            }

        # --- Downstream ---
        else:
            final_emb = torch.cat([fusion_emb, demo_emb], dim=-1)  # [B, 2D]
            pred = self.downstream_head(final_emb).squeeze(-1)

            return {
                'pred': pred,
                'embs': final_emb
            }

    def check_padding(self,
                      times: torch.Tensor,        # [B, L]
                      varis: torch.Tensor,        # [B, L] (int64)
                      values: torch.Tensor,       # [B, L]
                      padding_mask: torch.Tensor  # [B, L] (bool, True=pad)
                      ):
        """
        Debug helper for per-variable hard gating.
        Return dict: {'times','varis','values','mask(redundant)'}
        """
        bsz, seq_len = values.size()

        time_emb = self.time_embed(times.unsqueeze(-1))
        value_emb = self.value_embed(values.unsqueeze(-1))
        feature_emb = self.feature_embed(varis)
        triplet_base = time_emb + value_emb + feature_emb       # [B, L, D]

        # ===== Surprise masking (JIT scan, zero host sync) =====
        with torch.no_grad():
            # triplet_base는 아래에서 계속 쓰니까 역전파는 원본 유지,
            # 마스킹 계산만 detach된 복사로 진행하여 오버헤드/메모리 감소
            tb_det = triplet_base.detach()

            # 1) 유사도 행렬 (bmm 한 번) — AMP/TF32로 더 가볍게
            torch.backends.cuda.matmul.allow_tf32 = True
            triplet_norm = torch.nn.functional.normalize(tb_det, p=2, dim=-1)  # [B, L, D]
            sim_mat = torch.bmm(triplet_norm, triplet_norm.transpose(1, 2))    # [B, L, L]

            # 2) 마스크들 (전부 텐서 연산, CPU로 끌어오지 않음)
            B, L, _ = sim_mat.shape
            device = sim_mat.device
            valid = (~padding_mask)                                            # [B, L]
            same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))               # [B, L, L]

            # 과거만(i<j). 필요하면 밴드 윈도우로 치환.
            idx = torch.arange(L, device=device)
            upper = (idx[None, :] - idx[:, None]) > 0                          # [L, L]
            upper = upper.unsqueeze(0)                                         # [1, L, L]

            # 후보 페어 (유사도/변수/유효/과거)
            valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)               # [B, L, L]
            pair_hit = (sim_mat >= self.sim_threshold) & same_var & valid_pair & upper  # [B, L, L]

            # 3) 좌→우 그리디 스캔 (전부 텐서, TorchScript 내부 루프만 사용)
            #    window_W로 최근 W개만 비교하고 싶으면 정수로 넣기(예: 128). 정확히 하려면 0.
            window_W = 0
            redundant_mask = surprise_redundant_mask_from_pairhit(pair_hit, valid, window_W)  # [B, L]
        

        return {
            'times':  times,
            'varis':  varis,
            'values': values,
            'mask':   redundant_mask,
        }




@torch.no_grad()
def surprise_mask(
    triplet_base: torch.Tensor,   # [B, L, D]
    varis: torch.Tensor,          # [B, L] (int64)
    padding_mask: torch.Tensor,   # [B, L] (bool, True=pad)
    sim_threshold: float,         # base threshold (e.g., 0.95)
    *,
    direction: str = "past",      # "past" | "future"
    window_W: int = 0,            # 최근 W개만 비교(0이면 전체)
    adaptive: bool = False,       # 유효 토큰 많을수록 threshold ↓
    adapt_alpha: float = 0.25,    # 임계값 내리는 강도(0~1 권장)
    min_threshold: float = 0.50,  # 임계값 하한
    normalize_by: str = "L",      # "L" | "valid_max"
    detach_embeddings: bool = True,
    use_tf32: bool = True,
) -> torch.Tensor:
    """
    Return:
        redundant_mask: [B, L] (bool, True=redundant)

    구현:
      - future는 입력을 좌우 flip해서 past 규칙으로 계산 후, 결과를 다시 flip.
      - adaptive=True면 위치 j의 '과거'(flip 시 future)에 축적된 유효 개수로 임계값을 낮춤.
    """
    if direction not in ("past", "future"):
        raise ValueError(f'direction must be "past" or "future", got {direction}')

    flip = (direction == "future")
    if flip:
        triplet_base = torch.flip(triplet_base, dims=[1])
        varis        = torch.flip(varis,        dims=[1])
        padding_mask = torch.flip(padding_mask, dims=[1])

    tb = triplet_base.detach() if detach_embeddings else triplet_base
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    B, L, D = tb.shape
    device = tb.device

    # 1) cosine similarity
    triplet_norm = F.normalize(tb, p=2, dim=-1)                      # [B,L,D]
    sim_mat = torch.bmm(triplet_norm, triplet_norm.transpose(1, 2))  # [B,L,L]

    # 2) masks
    valid = (~padding_mask)                                          # [B,L]
    same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))             # [B,L,L]

    # 3) 과거만(i<j). window_W>0이면 j-i<=W
    idx = torch.arange(L, device=device)
    dist = idx[None, :] - idx[:, None]                               # [L,L] (i<j → dist>0)
    if window_W and window_W > 0:
        tri = (dist > 0) & (dist <= window_W)
    else:
        tri = (dist > 0)
    tri = tri.unsqueeze(0)                                           # [1,L,L]

    # 4) 유효 토큰끼리만
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)             # [B,L,L]

    # 5) adaptive threshold (열 j별)
    if adaptive:
        vfloat = valid.float()                                       # [B,L]
        past_cnt = torch.cumsum(vfloat, dim=1) - vfloat              # [B,L]
        if normalize_by == "valid_max":
            denom = past_cnt.max(dim=1, keepdim=True).values.clamp_min(1.0)
        else:  # "L"
            denom = float(L)
        norm_cnt = past_cnt / denom
        thr_vec = sim_threshold - adapt_alpha * norm_cnt             # [B,L]
        thr_vec = torch.clamp(thr_vec, min=min_threshold, max=0.999)
    else:
        thr_vec = torch.full((B, L), float(sim_threshold), device=device)

    thr_col = thr_vec.unsqueeze(1).expand(B, L, L)                   # [B,L,L]

    # 6) 후보 페어
    pair_core = (sim_mat >= thr_col) & same_var & valid_pair & tri   # [B,L,L]

    # 7) 좌->우 그리디 스캔 (past 규칙)
    keep = torch.zeros((B, L), dtype=torch.bool, device=device)
    for j in range(L):
        if j == 0:
            blocked = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            blocked = (pair_core[:, :j, j] & keep[:, :j]).any(dim=1)
        keep[:, j] = valid[:, j] & (~blocked)

    redundant_mask = valid & (~keep)                                 # [B,L]

    if flip:
        redundant_mask = torch.flip(redundant_mask, dims=[1])

    return redundant_mask

class SurpSTraTSLnWa(nn.Module):  # per-variable weighted average + global FusionAttention
    """
    - LayerNorm 이후 time/value/feature 임베딩을 변수별 가중평균으로 결합
    - 각 변수 k마다 [w_t(k), w_v(k), w_f(k)]를 학습 (softmax로 합=1)
    - 나머지 블록(하드게이팅/퓨전/헤드)은 SurpriseSTraTSLn과 동일
    """
    def __init__(self, 
                 num_features,
                 embed_dim=32,
                 static_dim=3, 
                 num_heads=4, 
                 num_blocks=2, 
                 ff_dim=64,
                 dropout=0.2, 
                 time_activation='relu', 
                 value_activation='tanh', 
                 final_emb_type='balanced', 
                 fusion_emb_weight=0.5,
                 final_emb_weight=0.5,
                 domain_lambda=1.0,
                 sim_threshold=0.95,
                 direction='past'):
        super().__init__()

        # (1) Transformer + Embeddings
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features + 1, embed_dim, padding_idx=num_features)  # Feats + padding 1

        # 단일 FusionAttention (전체 변수에 대해)
        self.fusion_attention = FusionAttention(embed_dim)

        self.dropout   = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_time   = nn.LayerNorm(embed_dim)     # for Time path
        self.ln_value  = nn.LayerNorm(embed_dim)     # for Value path
        self.ln_feat   = nn.LayerNorm(embed_dim)     # for Feature(nn.Embedding) path

        # Downstream / heads
        final_emb_dim = embed_dim * 2
        self.downstream_head = FrcstHead(final_emb_dim, 1)
        self.forecast_head   = FrcstHead(final_emb_dim, num_features + 1)
        self.domain_head     = FrcstHead(final_emb_dim, 1)
        self.domain_lambda   = domain_lambda

        # CLS & misc
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 설정
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type "{final_emb_type}", using default "balanced".')
            self.final_emb_type = 'balanced'
        else:
            self.final_emb_type = final_emb_type

        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight  = final_emb_weight
        self.sim_threshold     = sim_threshold

        self.padding_idx = num_features
        self.direction = direction

        # === (NEW) 변수별 가중치 파라미터: [F+1, 3] -> softmax로 합 1 ===
        # 초기 0 => softmax는 균등(1/3,1/3,1/3)
        self.triplet_weight_logits = nn.Parameter(torch.zeros(num_features + 1, 3))

    # --------- helper: 변수별 가중치 가져오기 ---------
    def _triplet_weights(self, varis):
        """
        varis: [B, L] (int64)
        returns: weights [B, L, 3] with sum=1 across last dim
        order: [time, value, feature]
        """
        weights = F.softmax(self.triplet_weight_logits, dim=-1)     # [F+1, 3]
        w_sel = weights[varis]                                      # [B, L, 3]
        return w_sel

    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):

        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]

        # --- Embeddings ---
        time_emb    = self.ln_time(self.time_embed(times.unsqueeze(-1)))      # [B, L, D]
        value_emb   = self.ln_value(self.value_embed(values.unsqueeze(-1)))   # [B, L, D]
        feature_emb = self.ln_feat(self.feature_embed(varis))                 # [B, L, D]

        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)    # [B, L, 1]
            value_emb = value_emb * keep

        # === (NEW) 변수별 가중 평균 결합 ===
        w = self._triplet_weights(varis)                         # [B, L, 3]
        wt = w[..., 0].unsqueeze(-1)                             # [B, L, 1]
        wv = w[..., 1].unsqueeze(-1)
        wf = w[..., 2].unsqueeze(-1)

        triplet_base = wt * time_emb + wv * value_emb + wf * feature_emb   # [B, L, D]

        # ===== Surprise masking (모듈 함수 호출) =====
        redundant_mask = surprise_mask(
            triplet_base=triplet_base,
            varis=varis,
            padding_mask=padding_mask,
            sim_threshold=self.sim_threshold,
            direction=self.direction,   # or "past"
            window_W=0,
            adaptive=False,
            adapt_alpha=0.25,
            min_threshold=0.55,
            normalize_by="L",
            detach_embeddings=True,
            use_tf32=True,
        )

        padding_mask = padding_mask | redundant_mask
        new_padding_mask = padding_mask | pretrain_mask if pretrain else padding_mask

        # (D) 드롭아웃 / CLS concat
        triplet_emb = self.dropout(triplet_base)                                # [B, L, D]
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)               # [B, L+1, D]

        # (E) 어텐션 마스크 (True=pad)
        cls_pad = torch.zeros((bsz, 1), dtype=torch.bool, device=new_padding_mask.device)
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)                # [B, L+1]

        # --- Transformer Blocks ---
        for block in self.transformer_blocks:
            triplet_emb = block(triplet_emb, att_mask)

        fus_keep = (~new_padding_mask).float()
        fus_padding_mask = torch.cat([torch.ones_like(cls_pad), fus_keep], dim=1).float()

        # 5) Final Embedding
        fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
        fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fusion_emb = self.layer_norm(fusion_emb)
        demo_emb = self.demo_emb(statics)

        # --- Pretrain (Self-prediction) ---
        if pretrain:
            seq_emb = ((1 - self.final_emb_weight) * triplet_base
                       + self.final_emb_weight * fusion_emb.unsqueeze(1))
            demo_seq = demo_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
            final_emb = torch.cat((seq_emb, demo_seq), dim=-1)

            forecast = self.forecast_head(final_emb)                              # [B, L, F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1)  # [B, L]

            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
            }

        # --- Downstream ---
        else:
            final_emb = torch.cat([fusion_emb, demo_emb], dim=-1)  # [B, 2D]
            pred = self.downstream_head(final_emb).squeeze(-1)
            return {'pred': pred, 'embs': final_emb}

    # 디버그용: 마스크 확인
    def check_padding(self,
                      times: torch.Tensor,        # [B, L]
                      varis: torch.Tensor,        # [B, L] (int64)
                      values: torch.Tensor,       # [B, L]
                      padding_mask: torch.Tensor  # [B, L] (bool, True=pad)
                      ):
        bsz, seq_len = values.size()

        time_emb    = self.ln_time(self.time_embed(times.unsqueeze(-1)))
        value_emb   = self.ln_value(self.value_embed(values.unsqueeze(-1)))
        feature_emb = self.ln_feat(self.feature_embed(varis))

        # (NEW) 변수별 가중 결합
        w  = self._triplet_weights(varis)     # [B, L, 3]
        wt = w[..., 0].unsqueeze(-1)
        wv = w[..., 1].unsqueeze(-1)
        wf = w[..., 2].unsqueeze(-1)
        triplet_base = wt*time_emb + wv*value_emb + wf*feature_emb

        # 외부 모듈로 마스크 생성
        redundant_mask = surprise_mask(
            triplet_base=triplet_base,
            varis=varis,
            padding_mask=padding_mask,
            sim_threshold=self.sim_threshold,
            direction=self.direction,   # or "past"
            window_W=0,
            adaptive=False,
            adapt_alpha=0.25,
            min_threshold=0.55,
            normalize_by="L",
            detach_embeddings=True,
            use_tf32=True,
        )

        return {'times': times, 'varis': varis, 'values': values, 'mask': redundant_mask}
    
class SurpriseSTraTSLn(nn.Module):  # per-variable gating + global FusionAttention
    """
    - Hard gating: same variable (per-var) 기준
    - FusionAttention: 그룹별이 아닌 전체 시퀀스에 대해 단일 attention으로 집약
    - var_groups 제거
    """
    def __init__(self, 
                 num_features,
                 embed_dim=32,
                 static_dim=3, 
                 num_heads=4, 
                 num_blocks=2, 
                 ff_dim=64,
                 dropout=0.2, 
                 time_activation='relu', 
                 value_activation='tanh', 
                 final_emb_type='balanced', 
                 fusion_emb_weight=0.5,
                 final_emb_weight=0.5,
                 domain_lambda=1.0,
                 sim_threshold=0.95,
                 direction='past'):
        super().__init__()

        # (1) Transformer + Embeddings
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features + 1, embed_dim, padding_idx=num_features)  # Feats + padding 1

        # 단일 FusionAttention (전체 변수에 대해)
        self.fusion_attention = FusionAttention(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_time   = nn.LayerNorm(embed_dim)     # for Time path
        self.ln_value  = nn.LayerNorm(embed_dim)     # for Value path
        self.ln_feat   = nn.LayerNorm(embed_dim)     # for Feature(nn.Embedding) path

        # Downstream heads
        # 최종 임베딩: [vars_fused_emb, demo_emb] → 2 * D
        final_emb_dim = embed_dim * 2
        self.downstream_head = FrcstHead(final_emb_dim, 1)
        # pretrain용 head (토큰 임베딩에서 각 변수값 예측)
        self.forecast_head = FrcstHead(final_emb_dim, num_features + 1)
        self.domain_head = FrcstHead(final_emb_dim, 1)
        self.domain_lambda = domain_lambda

        # (3) CLS token & heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

        # (4) 설정
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type = 'balanced'
        else:
            self.final_emb_type = final_emb_type

        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight = final_emb_weight
        self.sim_threshold = sim_threshold
        self.direction = direction

        self.padding_idx = num_features

    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                freeze_pretrained=False):

        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]

        # --- Embeddings ---
        time_emb = self.ln_time(self.time_embed(times.unsqueeze(-1)))
        value_emb = self.ln_value(self.value_embed(values.unsqueeze(-1)))
        feature_emb = self.ln_feat(self.feature_embed(varis))

        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)    # [B, L, 1]
            value_emb = value_emb * keep

        # (A) base embedding
        triplet_base = time_emb + value_emb + feature_emb       # [B, L, D]        
            
        # ===== Surprise masking (모듈 함수 호출) =====
        redundant_mask = surprise_mask(
            triplet_base=triplet_base,
            varis=varis,
            padding_mask=padding_mask,
            sim_threshold=self.sim_threshold,
            direction=self.direction,   # or "past"
            window_W=0,
            adaptive=False,
            adapt_alpha=0.25,
            min_threshold=0.55,
            normalize_by="L",
            detach_embeddings=True,
            use_tf32=True,
        )

        padding_mask = padding_mask | redundant_mask
        new_padding_mask = padding_mask | pretrain_mask if pretrain else padding_mask

        # (D) 드롭아웃 / CLS concat
        triplet_emb = self.dropout(triplet_base)                # [B, L, D]
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)  # [B, L+1, D]

        # (E) 어텐션 마스크 (주의: 블록 내부의 mask 의미(True=pad)와 일치 필요)
        cls_pad = torch.zeros((bsz, 1), dtype=torch.bool, device=new_padding_mask.device)  # 원 로직 유지
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)  # [B, L+1]

        # --- Transformer Blocks ---
        for block in self.transformer_blocks: # Ignores True
            triplet_emb = block(triplet_emb, att_mask)

        fus_keep = (~new_padding_mask).float()
        fus_padding_mask = torch.cat([torch.ones_like(cls_pad), fus_keep], dim=1).float()

        # -----------------------------------------
        # 5) Final Embedding (fusion vs cls etc.)
        # -----------------------------------------

        fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
        fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fusion_emb = self.layer_norm(fusion_emb)
        demo_emb = self.demo_emb(statics)

        # --- Pretrain (Self-prediction) ---
        if pretrain:
            # [B, D]

            # Self-prediction
            seq_emb = ((1 - self.final_emb_weight) * triplet_base
                    + self.final_emb_weight * fusion_emb.unsqueeze(1))
            demo_seq = demo_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
            final_emb = torch.cat((seq_emb, demo_seq), dim=-1)
            
            forecast = self.forecast_head(final_emb) # [B, L, F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1) # [B, L]

            # Loss 계산 시: pretrain_mask & NOT(padding_mask)
            #     → redundancy로 마스킹된 토큰(Padding mask에 반영)은 loss에서 제외
            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
            }

        # --- Downstream ---
        else:
            final_emb = torch.cat([fusion_emb, demo_emb], dim=-1)  # [B, 2D]
            pred = self.downstream_head(final_emb).squeeze(-1)

            return {
                'pred': pred,
                'embs': final_emb
            }

    def check_padding(self,
                      times: torch.Tensor,        # [B, L]
                      varis: torch.Tensor,        # [B, L] (int64)
                      values: torch.Tensor,       # [B, L]
                      padding_mask: torch.Tensor  # [B, L] (bool, True=pad)
                      ):
        """
        Debug helper for per-variable hard gating.
        Return dict: {'times','varis','values','mask(redundant)'}
        """
        bsz, seq_len = values.size()

        time_emb = self.ln_time(self.time_embed(times.unsqueeze(-1)))
        value_emb = self.ln_value(self.value_embed(values.unsqueeze(-1)))
        feature_emb = self.ln_feat(self.feature_embed(varis))
        triplet_base = time_emb + value_emb + feature_emb       # [B, L, D]

        # ===== Surprise masking (모듈 함수 호출) =====
        redundant_mask = surprise_mask(
            triplet_base=triplet_base,
            varis=varis,
            padding_mask=padding_mask,
            sim_threshold=self.sim_threshold,
            direction=self.direction,   # or "past"
            window_W=0,
            adaptive=False,
            adapt_alpha=0.25,
            min_threshold=0.55,
            normalize_by="L",
            detach_embeddings=True,
            use_tf32=True,
        )

        return {
            'times':  times,
            'varis':  varis,
            'values': values,
            'mask':   redundant_mask,
        }

# Variable-wise value embedding

@torch.no_grad()
def surprise_mask_vt(
    value_emb: torch.Tensor,     # [B, L, D]
    time_emb: torch.Tensor,      # [B, L, D]
    varis: torch.Tensor,         # [B, L] (int64)
    padding_mask: torch.Tensor,  # [B, L] (bool, True=pad)
    sim_threshold: float,
    *,
    direction: str = "past",
    window_W: int = 0,
    adaptive: bool = False,
    adapt_alpha: float = 0.25,
    min_threshold: float = 0.50,
    normalize_by: str = "L",
    tau_v: float = 1.0,
    tau_t: float = 1.0,
    w_v: float = 1.0,
    w_t: float = 1.0,
    use_tf32: bool = True,

    # ==== NEW: gating 대상 변수 제어 ====
    gated_vars: list | None = None,         # 예: [0,3,7]  -> 이 변수들만 gating
    per_token_gate_mask: torch.Tensor | None = None,  # [B,L] bool. True면 그 토큰(열 j)에 게이팅 적용
    invert: bool = False,                   # True면 gated_vars를 블랙리스트로 사용
    padding_idx: int | None = None,         # 패딩 변수 id (기본 None이면 무시)
) -> torch.Tensor:
    """
    gated_vars/per_token_gate_mask로 '게이팅이 적용될 열 j'를 지정.
    - gated_vars: 변수 id 리스트 (열 j의 var가 이 리스트 안에 있을 때만 게이팅)
    - per_token_gate_mask: [B,L] bool. True인 열만 게이팅
    - invert=True: 리스트를 제외 목록으로 취급
    - 기본동작: 둘 다 None이면 '모든 실제 변수(패딩 제외)'에 게이팅 적용
    """
    assert direction in ("past", "future")
    flip = (direction == "future")
    if flip:
        value_emb    = torch.flip(value_emb, dims=[1])
        time_emb     = torch.flip(time_emb,  dims=[1])
        varis        = torch.flip(varis,     dims=[1])
        padding_mask = torch.flip(padding_mask, dims=[1])

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    vnorm = F.normalize(value_emb, p=2, dim=-1)
    tnorm = F.normalize(time_emb,  p=2, dim=-1)
    sim_v = torch.bmm(vnorm, vnorm.transpose(1, 2))  # [B,L,L]
    sim_t = torch.bmm(tnorm, tnorm.transpose(1, 2))  # [B,L,L]
    score = (w_v * (sim_v / max(tau_v, 1e-6))) + (w_t * (sim_t / max(tau_t, 1e-6)))

    B, L = varis.shape
    device = varis.device
    valid = (~padding_mask)                                        # [B,L]
    same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))           # [B,L,L]

    # --- 게이팅 대상 열 j 마스크 만들기 ---
    if per_token_gate_mask is not None:
        gate_col_mask = per_token_gate_mask.bool().to(device)      # [B,L]
    elif gated_vars is not None:
        gv = torch.tensor(gated_vars, device=device, dtype=varis.dtype)
        gate_col_mask = torch.isin(varis, gv)                      # [B,L]
        if invert:
            gate_col_mask = ~gate_col_mask
    else:
        # 기본: 모든 실제 변수(패딩 제외)에 적용
        if padding_idx is None:
            gate_col_mask = torch.ones_like(varis, dtype=torch.bool)
        else:
            gate_col_mask = (varis != padding_idx)

    # 과거만(i<j) + window
    idx = torch.arange(L, device=device)
    dist = idx[None, :] - idx[:, None]
    tri = (dist > 0) if (window_W == 0 or window_W is None) else ((dist > 0) & (dist <= window_W))
    tri = tri.unsqueeze(0)                                         # [1,L,L]

    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)           # [B,L,L]

    # Adaptive(정적)
    if adaptive:
        vfloat = valid.float()
        past_cnt = torch.cumsum(vfloat, dim=1) - vfloat
        denom = (float(L) if normalize_by == "L"
                 else past_cnt.max(dim=1, keepdim=True).values.clamp_min(1.0))
        thr_vec = sim_threshold - adapt_alpha * (past_cnt / denom)
        thr_vec = torch.clamp(thr_vec, min=min_threshold, max=0.999)
    else:
        thr_vec = torch.full((B, L), float(sim_threshold), device=device)
    thr_col = thr_vec.unsqueeze(1).expand(B, L, L)

    # 후보 페어
    pair_core = (score >= thr_col) & same_var & valid_pair & tri   # [B,L,L]

    # === 핵심: 열 j가 게이팅 대상일 때만 차단 로직을 허용 ===
    # gate_col_mask: [B,L] (열 j 기준). 이를 [B,L,L]로 브로드캐스트해서 열 방향에만 적용.
    pair_core = pair_core & gate_col_mask.unsqueeze(1)             # j 열만 게이팅 가능

    # 그리디 스캔
    keep = torch.zeros((B, L), dtype=torch.bool, device=device)
    for j in range(L):
        if j == 0:
            blocked = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            blocked = (pair_core[:, :j, j] & keep[:, :j]).any(dim=1)
        keep[:, j] = valid[:, j] & (~blocked)

    redundant_mask = valid & (~keep)

    # 게이팅 대상이 아닌 열은 절대 마스크하지 않도록 한 번 더 보수적으로 차단
    redundant_mask = redundant_mask & gate_col_mask

    if flip:
        redundant_mask = torch.flip(redundant_mask, dims=[1])
    return redundant_mask

class SurpriseSTraTSLnVT_SeparateValue(nn.Module):
    """
    - 변수마다 독립적인 ContinuousValueEmbedding 모듈을 사용 (FiLM 없음)
    - 마스킹: Value cosine + Time cosine 가중합 → threshold 비교
    - Transformer, Fusion, Heads는 기존 형태 유지(입력은 value+time 합)
    """
    def __init__(self,
                 num_features,
                 embed_dim=32,
                 static_dim=3,
                 num_heads=4,
                 num_blocks=2,
                 ff_dim=64,
                 dropout=0.2,
                 time_activation='relu',
                 value_activation='tanh',
                 final_emb_type='balanced',
                 fusion_emb_weight=0.5,
                 final_emb_weight=0.5,
                 domain_lambda=1.0,
                 # Masking params
                 sim_threshold=0.95,
                 direction='future',
                 window_W=0,
                 adaptive=False,
                 adapt_alpha=0.25,
                 min_threshold=0.55,
                 normalize_by='L',
                 tau_v=1.0,
                 tau_t=1.0,
                 w_v=1.0,
                 w_t=1.0,
                 use_tf32=True,
                 gated_vars=None):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.padding_idx = num_features
        self.gated_vars = gated_vars

        # --- Time embedding ---
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.ln_time = nn.LayerNorm(embed_dim)

        # --- Value embeddings (one per feature, +1 for padding idx) ---
        self.value_embeds = nn.ModuleList([
            ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
            for _ in range(num_features)
        ])
        # padding row: dummy module; we'll just zero out where varis==padding_idx
        self.value_pad_ln = nn.LayerNorm(embed_dim)  # kept for interface symmetry
        self.ln_value = nn.LayerNorm(embed_dim)

        # --- Transformer & Fusion ---
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.fusion_attention = FusionAttention(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)

        # Heads
        final_emb_dim = embed_dim + embed_dim  # fusion_emb + demo_emb
        self.downstream_head = FrcstHead(final_emb_dim, 1)
        self.forecast_head   = FrcstHead(final_emb_dim, num_features + 1)
        self.domain_head     = FrcstHead(final_emb_dim, 1)
        self.domain_lambda   = domain_lambda

        # Demo/static
        self.demo_emb = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Configs
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type "{final_emb_type}", using default "balanced".')
            self.final_emb_type = 'balanced'
        else:
            self.final_emb_type = final_emb_type

        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight  = final_emb_weight

        # Masking hyperparams
        self.sim_threshold = sim_threshold
        self.direction     = direction
        self.window_W      = window_W
        self.adaptive      = adaptive
        self.adapt_alpha   = adapt_alpha
        self.min_threshold = min_threshold
        self.normalize_by  = normalize_by
        self.tau_v         = tau_v
        self.tau_t         = tau_t
        self.w_v           = w_v
        self.w_t           = w_t
        self.use_tf32      = use_tf32

    # ---- helpers ----
    def _time_embed(self, times):
        return self.ln_time(self.time_embed(times.unsqueeze(-1)))  # [B,L,D]

    def _value_embed_per_feature(self, values, varis):
        """
        변수별 독립 임베더 선택 적용.
        values: [B,L], varis: [B,L] (int64, padding_idx = num_features)
        return: [B,L,D]
        """
        B, L = values.shape
        out = torch.zeros(B, L, self.embed_dim, device=values.device, dtype=values.dtype)

        # 각 feature에 대해 해당 위치만 계산해서 채워넣음
        for k in range(self.num_features):
            mask = (varis == k)
            if mask.any():
                v_k = values[mask].unsqueeze(-1)              # [Nk,1]
                emb_k = self.value_embeds[k](v_k)             # [Nk,D]
                out[mask] = emb_k

        # padding 위치는 0 (이미 0으로 초기화됨)
        out = self.ln_value(out)
        return out

    # ---- forward ----
    def forward(self,
                times, varis, values, statics,
                padding_mask,
                pretrain=False,
                pretrain_mask=None,
                freeze_pretrained=False):

        bsz, seq_len = values.size()

        time_emb  = self._time_embed(times)                       # [B,L,D]
        value_emb = self._value_embed_per_feature(values, varis)  # [B,L,D]

        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)
            value_emb = value_emb * keep

        # --- Surprise masking (Value+Time) ---
        redundant_mask = surprise_mask_vt(
            value_emb=value_emb,
            time_emb=time_emb,
            varis=varis,
            padding_mask=padding_mask,
            sim_threshold=self.sim_threshold,
            direction=self.direction,
            window_W=self.window_W,
            adaptive=self.adaptive,
            adapt_alpha=self.adapt_alpha,
            min_threshold=self.min_threshold,
            normalize_by=self.normalize_by,
            tau_v=self.tau_v, tau_t=self.tau_t,
            w_v=self.w_v,   w_t=self.w_t,
            use_tf32=self.use_tf32,
            # ==== 새 옵션 전달 ====
            gated_vars=self.gated_vars,          # ex) [2,5,9]
            per_token_gate_mask=None,            # 필요 시 텐서로 주입
            invert=False,
            padding_idx=self.padding_idx,
        )
        padding_mask = padding_mask | redundant_mask
        new_padding_mask = padding_mask | pretrain_mask if pretrain else padding_mask

        # Transformer 입력: value + time (feature id emb 사용 안 함)
        tok_emb = self.dropout(value_emb + time_emb)              # [B,L,D]
        cls_tokens = self.cls_token.expand(bsz, 1, -1)
        triplet_emb = torch.cat([cls_tokens, tok_emb], dim=1)     # [B,L+1,D]

        cls_pad = torch.zeros((bsz, 1), dtype=torch.bool, device=new_padding_mask.device)
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)

        for block in self.transformer_blocks:
            triplet_emb = block(triplet_emb, att_mask)

        fus_keep = (~new_padding_mask).float()
        fus_padding_mask = torch.cat([torch.ones_like(cls_pad), fus_keep], dim=1).float()

        fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
        fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fusion_emb = self.layer_norm(fusion_emb)

        demo_emb = self.demo_emb(statics)

        if pretrain:
            seq_emb = ((1 - self.final_emb_weight) * tok_emb
                       + self.final_emb_weight * fusion_emb.unsqueeze(1))
            demo_seq = demo_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
            final_emb = torch.cat((seq_emb, demo_seq), dim=-1)     # [B,L,2D]
            forecast = self.forecast_head(final_emb)               # [B,L,F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1)

            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {'forecast': forecast_sel, 'values': values, 'varis': varis,
                    'times': times, 'mask': pretrain_mask, 'loss': mse_loss}
        else:
            final_emb = torch.cat([fusion_emb, demo_emb], dim=-1)  # [B,2D]
            pred = self.downstream_head(final_emb).squeeze(-1)
            return {'pred': pred, 'embs': final_emb}

    # 디버그
    def check_padding(self, times, varis, values, padding_mask):
        time_emb  = self._time_embed(times)
        value_emb = self._value_embed_per_feature(values, varis)
        redundant_mask = surprise_mask_vt(
            value_emb=value_emb,
            time_emb=time_emb,
            varis=varis,
            padding_mask=padding_mask,
            sim_threshold=self.sim_threshold,
            direction=self.direction,
            window_W=self.window_W,
            adaptive=self.adaptive,
            adapt_alpha=self.adapt_alpha,
            min_threshold=self.min_threshold,
            normalize_by=self.normalize_by,
            tau_v=self.tau_v, tau_t=self.tau_t,
            w_v=self.w_v,   w_t=self.w_t,
            use_tf32=self.use_tf32,
            # ==== 새 옵션 전달 ====
            gated_vars=self.gated_vars,          # ex) [2,5,9]
            per_token_gate_mask=None,            # 필요 시 텐서로 주입
            invert=False,
            padding_idx=self.padding_idx,
        )
        return {'times': times, 'varis': varis, 'values': values, 'mask': redundant_mask}
    
class SepSTraTS(nn.Module):  # Single task
    """
    Separated embeddings, but no surprise gating
    - 변경점: 변수별 독립 Value embedding 사용 (feature_embed 제거)
    """
    def __init__(self, 
                 num_features,
                 embed_dim=32,
                 static_dim=3, 
                 num_heads=4, 
                 num_blocks=2, 
                 ff_dim=64,
                 dropout=0.2, 
                 time_activation='relu', 
                 value_activation='tanh', 
                 final_emb_type='balanced', 
                 fusion_emb_weight=0.5,
                 final_emb_weight=0.5):
        super().__init__()

        # ------------------------------
        # (1) Transformer + Embeddings
        # ------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        # Time embedding
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        # Value embeddings: feature마다 하나씩 (+ padding용 index는 값 임베딩 없음 → 0으로 둠)
        self.value_embeds = nn.ModuleList([
            ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
            for _ in range(num_features)
        ])
        self.padding_idx = num_features  # varis에서 패딩 토큰 id

        self.fusion_attention = FusionAttention(embed_dim)

        # heads
        final_emb_dim = embed_dim * 2
        self.forecast_head = FrcstHead(final_emb_dim, num_features+1)
        self.downstream_head = FrcstHead(final_emb_dim, 1)

        self.dropout   = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_time   = nn.LayerNorm(embed_dim)
        self.ln_value  = nn.LayerNorm(embed_dim)

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.demo_emb = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

        # ------------------------------
        # (4) 설정
        # ------------------------------
        if final_emb_type not in ['cls', 'fusion', 'balanced']:
            print(f'Invalid final_emb_type of "{final_emb_type}", using default "balanced" final embedding.')
            self.final_emb_type = 'balanced'
        else:
            self.final_emb_type = final_emb_type

        self.fusion_emb_weight = fusion_emb_weight
        self.final_emb_weight  = final_emb_weight

    # ---- helpers ----
    def _time_embed(self, times: torch.Tensor) -> torch.Tensor:
        # times: [B,L]
        return self.ln_time(self.time_embed(times.unsqueeze(-1)))  # [B,L,D]

    def _value_embed_per_feature(self, values: torch.Tensor, varis: torch.Tensor) -> torch.Tensor:
        """
        values: [B,L], varis: [B,L] (int64, padding_idx = num_features)
        feature k에 대해 ContinuousValueEmbedding_k를 적용하여 해당 위치만 채운다.
        나머지(패딩 포함)는 0으로 남고, 전체를 LN으로 정규화.
        """
        B, L = values.shape
        D = self.cls_token.shape[-1]
        out = torch.zeros(B, L, D, device=values.device, dtype=values.dtype)

        for k in range(len(self.value_embeds)):
            mask = (varis == k)
            if mask.any():
                v_k = values[mask].unsqueeze(-1)      # [Nk,1]
                emb_k = self.value_embeds[k](v_k)     # [Nk,D]
                out[mask] = emb_k

        out = self.ln_value(out)                      # [B,L,D]
        return out

    # ---- forward ----
    def forward(self, 
                times, varis, values, statics,
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None):

        bsz, seq_len = values.size()
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, D]

        # --- Embeddings ---
        time_emb  = self._time_embed(times)                         # [B,L,D]
        value_emb = self._value_embed_per_feature(values, varis)    # [B,L,D]

        # pretrain: 값 마스킹(기존 동작 유지)
        if pretrain:
            keep = (1 - pretrain_mask.float()).unsqueeze(-1)        # [B,L,1]
            value_emb = value_emb * keep

        # (A) base embedding (feature_emb 제거 → time + value)
        triplet_base = time_emb + value_emb                         # [B,L,D]

        # (B) padding mask
        new_padding_mask = (padding_mask | pretrain_mask) if pretrain else padding_mask

        # (D) 드롭아웃/CLS concat
        triplet_emb = self.dropout(triplet_base)                    # [B,L,D]
        triplet_emb = torch.cat([cls_tokens, triplet_emb], dim=1)   # [B,L+1,D]

        # (E) Transformer attn mask
        cls_pad = torch.zeros((bsz,1), dtype=torch.bool, device=new_padding_mask.device)
        att_mask = torch.cat([cls_pad, new_padding_mask], dim=1)    # [B,L+1]

        # --- Transformer Blocks ---
        for block in self.transformer_blocks:
            triplet_emb = block(triplet_emb, att_mask)

        # fusion
        fus_keep = (~new_padding_mask).float()
        fus_padding_mask = torch.cat([torch.ones_like(cls_pad), fus_keep], dim=1).float()

        fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
        fusion_emb = (triplet_emb * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fusion_emb = self.layer_norm(fusion_emb)
        demo_emb = self.demo_emb(statics)

        # --- Pretrain (forecast) ---
        if pretrain:
            seq_emb = ((1 - self.final_emb_weight) * triplet_base
                       + self.final_emb_weight * fusion_emb.unsqueeze(1))
            demo_seq = demo_emb.unsqueeze(1).expand(-1, seq_emb.size(1), -1)
            final_emb = torch.cat((seq_emb, demo_seq), dim=-1)      # [B,L,2D]
            
            forecast = self.forecast_head(final_emb)                # [B,L,F+1]
            forecast_sel = torch.gather(forecast, 2, varis.unsqueeze(-1)).squeeze(-1)  # [B,L]

            masked_loss = pretrain_mask.float() * (1 - padding_mask.float())
            diff = (forecast_sel - values) * masked_loss
            mse_loss = (diff**2).sum() / masked_loss.sum().clamp(min=1.0)

            return {
                'forecast': forecast_sel,
                'values': values,
                'varis': varis,
                'times': times,
                'mask':  pretrain_mask,
                'loss':  mse_loss,
            }

        # --- Downstream ---
        else:
            final_emb = torch.cat((fusion_emb, demo_emb), dim=-1)   # [B, 2D]
            pred = self.downstream_head(final_emb).squeeze(-1)

            return {
                'pred' : pred,
                'embs' : final_emb
            }
