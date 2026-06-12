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
        prediction = self.mlp_static(output)

        return prediction, distance, None

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

        return prediction, distance, None

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
            final file truncated for brevity...