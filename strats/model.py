import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [batch_size] or [batch_size, 1]  => 예측 확률 (sigmoid output)
        targets: [batch_size] (0 or 1)
        """
        # inputs는 sigmoid 확률 형태(0~1)라고 가정
        # 만약 logits(=전 sigmoid)이라면 F.binary_cross_entropy_with_logits와 유사하게 수정 필요
        eps = 1e-6
        inputs = torch.clamp(inputs, min=eps, max=1.0-eps)

        pt = torch.where(targets == 1, inputs, 1 - inputs)  # p_t
        focal_term = (1 - pt) ** self.gamma

        bce = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        loss = self.alpha * focal_term * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

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
        attn_mask = ~padding_mask.bool()
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feedforward(x)
        return self.norm2(x + self.dropout(ff_output))

class FusionAttention(nn.Module):
    """
    Fusion Attention to aggregate contextual embeddings.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b = nn.Parameter(torch.zeros(embed_dim))
        self.u = nn.Parameter(torch.empty(embed_dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)

    def forward(self, x, mask):
        att = torch.tanh(torch.matmul(x, self.W) + self.b)
        scores = torch.matmul(att, self.u).squeeze(-1)
        scores = scores + (1 - mask) * torch.finfo(scores.dtype).min
        weights = F.softmax(scores, dim=-1)
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


class STraTSModel_3task(nn.Module):
    """
    Main model definition for the STraTS task, 
    now using dataset-provided mask for pretrain.
    """
    def __init__(self, 
                num_features,
                embed_dim, 
                num_heads, 
                num_blocks, 
                ff_dim,
                dropout, 
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
        self.fusion_attention = FusionAttention(embed_dim)
        self.time_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=time_activation)
        self.value_embed = ContinuousValueEmbedding(input_dim=1, embed_dim=embed_dim, activation=value_activation)
        self.feature_embed = nn.Embedding(num_features + 1, embed_dim)

        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(embed_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    

        # ------------------------------
        # (2) Downstream heads
        # ------------------------------
        self.linear_saps = FrcstHead(embed_dim, 1)
        self.linear_sofa = nn.Linear(embed_dim, 1)
        self.linear_death = nn.Linear(embed_dim, 1)

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)

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
                times, varis, values, 
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                outcomes=None,
                freeze_pretrained=False):
        """
        - pretrain=True  => 마스킹 기반 시계열 예측 (forecast) 수행
        - pretrain_mask : (batch_size, seq_len) - bool (True=mask)
        - freeze_pretrained=True => pretrain 모듈 파라미터 고정 (Transformer, embeddings, etc)
        - outcomes=(batch_size, 4) => [hadm_id, saps, sofa, death] (downstream)
        """

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

        cls_pad = torch.ones((bsz, 1), device=padding_mask.device)
        att_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)
        fus_padding_mask = torch.cat([torch.zeros_like(cls_pad), padding_mask], dim=1)

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
            fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
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

        # -----------------------------------------
        # 6) Pretrain => Forecast Head
        # -----------------------------------------
        if pretrain:
            # [bsz, seq_len+1, embed_dim]
            triplet_emb_seq = triplet_emb[:, 1:, :]  # CLS 제외
            seq_emb = (1 - self.final_emb_weight) * triplet_emb_seq + self.final_emb_weight * final_emb.unsqueeze(1)

            forecast = self.forecast_head(seq_emb)    # [bsz, seq_len, num_features+1]
            gather_input = varis.unsqueeze(-1)        # [bsz, seq_len, 1]
            forecast_selected = torch.gather(forecast, 2, gather_input)  # => [bsz, seq_len, 1]
            forecast_selected = forecast_selected.squeeze(-1)            # => [bsz, seq_len]

            # 모델 내부적 "mask"는 net_mask=1 => unmasked, net_mask=0 => masked
            # loss는 "masked 지점"만 계산 => (1 - net_mask) = pretrain_mask
            masked_loss = pretrain_mask.float() * padding_mask  # [bsz, seq_len]
            pred_masked = forecast_selected * masked_loss
            gt_masked = values * masked_loss

            diff = pred_masked - gt_masked
            sq_diff = diff ** 2
            sum_sq_diff = sq_diff.sum()
            sum_masked_loss = masked_loss.sum()

            if sum_masked_loss == 0:
                mse_loss = torch.tensor(0.0, device=values.device)
            else:
                mse_loss = sum_sq_diff / sum_masked_loss

            return {
                'forecast': forecast_selected,
                'values': values,
                'varis': varis,
                'times': times,
                'mask': pretrain_mask,  # (bool)
                'loss': mse_loss
            }

        # -----------------------------------------
        # 7) Downstream => SAPS/SOFA/Death
        # -----------------------------------------
        pred_saps  = self.linear_saps(final_emb)       # [bsz, 1]
        pred_sofa  = self.linear_sofa(final_emb)       # [bsz, 1]
        pred_death = torch.sigmoid(self.linear_death(final_emb))

        if outcomes is not None:
            # outcomes=(bsz, 4) => [hadm_id, saps_label, sofa_label, death_label]
            hadm_id     = outcomes[:, 0]
            saps_label  = outcomes[:, 1]
            sofa_label  = outcomes[:, 2]
            death_label = outcomes[:, 3]

            # 예시로 death만 loss 계산
            loss_saps  = F.mse_loss(pred_saps.squeeze(-1), saps_label)
            loss_sofa  = F.mse_loss(pred_sofa.squeeze(-1), sofa_label)
            loss_death = F.binary_cross_entropy(pred_death.squeeze(-1), death_label)
            total_loss = loss_death  # 필요 시 세 개 더하거나 가중치 적용

            return {
                'hadm_id': hadm_id,
                'pred_saps':  pred_saps,
                'pred_sofa':  pred_sofa,
                'pred_death': pred_death,
                'saps': saps_label,
                'sofa': sofa_label,
                'death': death_label,
                'loss': total_loss,
                'loss_saps': loss_saps,
                'loss_sofa': loss_sofa,
                'loss_death': loss_death
            }
        else:
            # Inference only
            return {
                'pred_saps':  pred_saps,
                'pred_sofa':  pred_sofa,
                'pred_death': pred_death
            }

class STraTSModel(nn.Module): # Single task
    """
    Main model definition for the STraTS task, 
    now using dataset-provided mask for pretrain.
    """
    def __init__(self, 
                num_features,
                embed_dim=32, 
                num_heads=4, 
                num_blocks=2, 
                ff_dim=64,
                dropout=0.2, 
                time_activation='relu', 
                value_activation='tanh', 
                final_emb_type='balanced', 
                fusion_emb_weight=0.5,
                final_emb_weight=0.5,
                loss_type='bce'):
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
        self.feature_embed = nn.Embedding(num_features + 1, embed_dim)

        # pretrain 시계열 예측 head
        self.forecast_head = FrcstHead(embed_dim, num_features+1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    

        # ------------------------------
        # (2) Downstream heads
        # ------------------------------

        self.linear_death = FrcstHead(embed_dim, 1)

        # ------------------------------
        # (3) CLS token & heads
        # ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)
        self.cls_head = CLSHead(embed_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)

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
        self.loss_type = loss_type

    def forward(self, 
                times, varis, values, 
                padding_mask, 
                pretrain=False, 
                pretrain_mask=None, 
                outcomes=None,
                freeze_pretrained=False):
        """
        - pretrain=True  => 마스킹 기반 시계열 예측 (forecast) 수행
        - pretrain_mask : (batch_size, seq_len) - bool (True=mask)
        - freeze_pretrained=True => pretrain 모듈 파라미터 고정 (Transformer, embeddings, etc)
        - outcomes=(batch_size, 4) => [hadm_id, saps, sofa, death] (downstream)
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

        cls_pad = torch.ones((bsz, 1), device=padding_mask.device)
        att_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)
        fus_padding_mask = torch.cat([torch.zeros_like(cls_pad), padding_mask], dim=1)

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
            fusion_weights = self.fusion_attention(triplet_emb, fus_padding_mask)
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

        # -----------------------------------------
        # 6) Pretrain => Forecast Head
        # -----------------------------------------
        if pretrain:
            # [bsz, seq_len+1, embed_dim]
            triplet_emb_seq = triplet_emb[:, 1:, :]  # CLS 제외
            seq_emb = (1 - self.final_emb_weight) * triplet_emb_seq + self.final_emb_weight * final_emb.unsqueeze(1)

            forecast = self.forecast_head(seq_emb)    # [bsz, seq_len, num_features+1]
            gather_input = varis.unsqueeze(-1)        # [bsz, seq_len, 1]
            forecast_selected = torch.gather(forecast, 2, gather_input)  # => [bsz, seq_len, 1]
            forecast_selected = forecast_selected.squeeze(-1)            # => [bsz, seq_len]

            # 모델 내부적 "mask"는 net_mask=1 => unmasked, net_mask=0 => masked
            # loss는 "masked 지점"만 계산 => (1 - net_mask) = pretrain_mask
            masked_loss = pretrain_mask.float() * padding_mask  # [bsz, seq_len]
            pred_masked = forecast_selected * masked_loss
            gt_masked = values * masked_loss

            diff = pred_masked - gt_masked
            sq_diff = diff ** 2
            sum_sq_diff = sq_diff.sum()
            sum_masked_loss = masked_loss.sum()

            if sum_masked_loss == 0:
                mse_loss = torch.tensor(0.0, device=values.device)
            else:
                mse_loss = sum_sq_diff / sum_masked_loss

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
        pred_death = torch.sigmoid(self.linear_death(final_emb))

        if outcomes is not None:
            # outcomes=(bsz, 3) => [hadm_id, query_time, death_label]
            hadm_id = outcomes[:, 0]
            query_time = outcomes[:, 1]
            death_label = outcomes[:, 2].float()

            # 예시로 death만 loss 계산
            if self.loss_type == 'focal':
                focal_loss = FocalLoss(alpha = 0.5, gamma=1.5)
            
                loss = focal_loss(pred_death.squeeze(-1), death_label)
            else:
                loss = F.binary_cross_entropy(pred_death.squeeze(-1), death_label)

            return {
                'hadm_id': hadm_id,
                'pred_death': pred_death,
                'death': death_label,
                'query_time' : query_time,
                'loss': loss,
                'emb': final_emb
            }
        else:
            # Inference only
            return {
                'pred_death': pred_death
            }
