import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import datetime
import wandb
import json
import os

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.metrics._ranking import _binary_clf_curve

from utils_rd import getStats_fixed, getStats_static, tensorize_normalize
from models_rd import Raindrop_Mod
from typing import List

def precision_recall_curve(y_true, y_pred, pos_label=None,
                           sample_weight=None,pi0=None):
    """Compute precision-recall (with optional calibration) pairs for different probability thresholds
    This implementation is a modification of scikit-learn "precision_recall_curve" function that adds calibration
    Siblini W., Fréry J., He-Guelton L., Oblé F., Wang YQ. (2020) Master Your Metrics with Calibration. 
    In: Berthold M., Feelders A., Krempl G. (eds) Advances in Intelligent Data Analysis XVIII. IDA 2020. Lecture Notes in Computer Science, vol 12080. Springer, Cham

    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    
   
    
    
    if pi0 is not None:
        pi = np.sum(y_true)/float(np.array(y_true).shape[0])
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tps / (tps + ratio*fps)
    else:
        precision = tps / (tps + fps)
    
    precision[np.isnan(precision)] = 0
        
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
def average_precision_calibrated(y_true, y_pred, pos_label=1, sample_weight=None,pi0=0.5): # AUPRC_c from 
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, pi0=pi0)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
# This is for BucketSampler
def get_seq_lens_from_concat(cd):
    lens = []
    for d in cd.datasets:                       # cd: ConcatDataset
        for s in d.samples:                     # your dataset has .samples
            lens.append(len(s[0]))              # s[0] == times (1D Tensor/list)
    return lens

# This is to use BucketSampler on ConcatDataset
def concat_length_fn(dataset, idx):
    # dataset: ConcatDataset
    # idx: global index
    # dataset.cumulative_sizes와 dataset.datasets를 이용해서
    # 어떤 서브데이터셋과 로컬 인덱스인지 찾음
    for ds_idx, c in enumerate(dataset.cumulative_sizes):
        if idx < c:
            base = 0 if ds_idx == 0 else dataset.cumulative_sizes[ds_idx-1]
            local_idx = idx - base
            subds = dataset.datasets[ds_idx]
            return subds.get_event_len(local_idx)
    raise IndexError

class BucketBatchSampler(Sampler):
    """
    길이 기준 정렬 → batch_size로 묶기 → (선택적) 에폭마다 셔플
    """
    def __init__(self, dataset, batch_size, drop_last=False,
                 length_fn=None, base_seed=0, shuffle=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.base_seed  = base_seed
        self.shuffle    = shuffle
        self.epoch      = 0

        # 길이 함수
        if length_fn is None:
            if hasattr(dataset, "get_event_len"):
                self.length_fn = dataset.get_event_len
            else:
                raise ValueError("length_fn을 넘기거나 dataset.get_event_len을 구현하세요.")
        else:
            self.length_fn = length_fn

        # 길이 기준 정렬된 인덱스
        self.indices = list(range(len(dataset)))
        self.indices.sort(key=lambda idx: self.length_fn(idx))

    def set_epoch(self, epoch:int):
        self.epoch = epoch

    def __iter__(self):
        # 정렬된 인덱스를 batch로 쪼갬
        batches = []
        for i in range(0, len(self.indices), self.batch_size):
            batch = self.indices[i:i+self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # 훈련 시에만 에폭별 셔플
        if self.shuffle:
            rng = np.random.default_rng(self.base_seed + self.epoch)
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.indices)//self.batch_size if self.drop_last \
               else int(np.ceil(len(self.indices)/self.batch_size))

class FeatureCollator(object):
    def __init__(self, feat_len):
        self.feat_len = feat_len
    def __call__(self, batch):
        times, varis, values, pad_mask, pre_mask, yi, pid, Si, domain = zip(*batch)

        # pad_sequence 로 (max_len, ) → (batch, max_len)
        # 원래 times: list of [n_i], want [batch, max_n]
        times_padded = pad_sequence(times, batch_first=True, padding_value=0.0)
        varis_padded = pad_sequence(varis, batch_first=True, padding_value=self.feat_len)
        vals_padded = pad_sequence(values, batch_first=True, padding_value=0.0)
        pad_mask_p = pad_sequence(pad_mask, batch_first=True, padding_value=True).bool()
        pre_mask_p = pad_sequence(pre_mask, batch_first=True, padding_value=False)

        yi_tensor = torch.tensor(yi, dtype=torch.long)
        domain_tensor = torch.tensor(domain, dtype=torch.long)
        pid_list  = list(pid)
        Si_tensor = torch.stack(Si)

        return times_padded, varis_padded, vals_padded, pad_mask_p, pre_mask_p, yi_tensor, pid_list, Si_tensor, domain_tensor

def pad_collate(batch):
    times, varis, values, pad_mask, pre_mask, yi, pid, Si = zip(*batch)

    # pad_sequence 로 (max_len, ) → (batch, max_len)
    # 원래 times: list of [n_i], want [batch, max_n]
    times_padded = pad_sequence(times, batch_first=True, padding_value=0.0)
    varis_padded = pad_sequence(varis, batch_first=True, padding_value=-1)
    vals_padded = pad_sequence(values, batch_first=True, padding_value=0.0)
    pad_mask_p = pad_sequence(pad_mask, batch_first=True, padding_value=True).bool()
    pre_mask_p = pad_sequence(pre_mask, batch_first=True, padding_value=False)

    yi_tensor = torch.tensor(yi, dtype=torch.long)
    pid_list  = list(pid)
    Si_tensor = torch.stack(Si)

    return times_padded, varis_padded, vals_padded, pad_mask_p, pre_mask_p, yi_tensor, pid_list, Si_tensor

# def pad_collate_dom(batch):
#     times, varis, values, pad_mask, pre_mask, yi, pid, Si, domain = zip(*batch)

#     # pad_sequence 로 (max_len, ) → (batch, max_len)
#     # 원래 times: list of [n_i], want [batch, max_n]
#     times_padded = pad_sequence(times, batch_first=True, padding_value=0.0)
#     varis_padded = pad_sequence(varis, batch_first=True, padding_value=-1)
#     vals_padded = pad_sequence(values, batch_first=True, padding_value=0.0)
#     pad_mask_p = pad_sequence(pad_mask, batch_first=True, padding_value=True).bool()
#     pre_mask_p = pad_sequence(pre_mask, batch_first=True, padding_value=False)

#     yi_tensor = torch.tensor(yi, dtype=torch.long)
#     domain_tensor = torch.tensor(domain, dtype=torch.long)
#     pid_list  = list(pid)
#     Si_tensor = torch.stack(Si)

#     return times_padded, varis_padded, vals_padded, pad_mask_p, pre_mask_p, yi_tensor, pid_list, Si_tensor, domain_tensor


class STraTSDataset(Dataset):
    """
    P: Tensor[T, B, F_total]     (F_total = n_sensors + n_sensors_mask)
    P_time: Tensor[T, B]
    P_static: Tensor[B, D]
    y: Tensor[B]
    ids:   List[B]
    pretrain_mask_ratio: float (per‐time pretrain mask 비율)
    """
    def __init__(self, P, P_time, P_static, y, ids, pretrain_mask_ratio=0.2):
        self.P = P
        self.P_time = P_time
        self.P_static = P_static
        self.y = y.view(-1)
        self.ids = ids

        self.T, self.B, self.F_total = P.shape
        assert self.F_total % 2 == 0, "F_total은 센서수 * 2 여야 합니다"
        self.n_sensors = self.F_total // 2

        # per‐patient per‐time mask for pretrain
        self.pre_masks = [
            torch.rand(self.T) < pretrain_mask_ratio
            for _ in range(self.B)
        ]

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        # 1) 텐서 슬라이싱
        Xi = self.P[:, idx, :]            # [T, F_total]
        Ti = self.P_time[:, idx]          # [T]
        Si = self.P_static[idx]           # [D]
        yi = int(self.y[idx].item())      # scalar
        pid = self.ids[idx]

        # 2) 센서값(앞 절반) / 마스크(뒤 절반) 분리
        values_mat = Xi[:, :self.n_sensors]      # [T, n_sensors]
        mask_mat   = Xi[:, self.n_sensors:] > 0  # [T, n_sensors], 관측된 타임스텝만 True

        # 3) 실제 관측 이벤트만 flatten
        #    mask_obs: [T, n_sensors]
        mask_obs = values_mat != 0
        t_idx, f_idx = mask_obs.nonzero(as_tuple=True)

        Ti = Ti.float() / 48.0
        times  = Ti[t_idx].float()        # [n_obs]
        varis  = f_idx                           # [n_obs], 0 <= varis < n_sensors
        values = values_mat[t_idx, f_idx]        # [n_obs]

        # 4) 패딩 마스크: event‐level로는 padding 없음 → 모두 False
        pad_mask = torch.zeros_like(times, dtype=torch.bool)

        # 5) pretrain mask: 시간 단위로 만든 self.pre_masks 에서
        #    t_idx 시점만 가져옴
        pre_mask = self.pre_masks[idx][t_idx]    # [n_obs]

        return times, varis, values, pad_mask, pre_mask, yi, pid, Si

class STraTSDatasetGR(Dataset):
    # When running P12, MIMIC, eICU make sure 3 #s are deleted (to deal with 48h data)
    """
    P: Tensor[T, B, F_total]     (F_total = n_sensors + n_sensors_mask)
    P_time: Tensor[T, B]         (각 타임스텝의 offset, 단위: 시간, 0~48)
    P_static: Tensor[B, D]
    y: Tensor[B]
    ids:   List[B]
    pretrain_mask_ratio: float (전체 48h 중 마스킹할 비율)
    """
    def __init__(self,
                 P, P_time, P_static, y, ids,
                 domain, pretrain_mask_ratio=0.2):
        self.P = P
        self.P_time = P_time
        self.P_static = P_static
        self.y = y.view(-1)
        self.ids = ids

        self.T, self.B, self.F_total = P.shape
        assert self.F_total % 2 == 0, "F_total은 센서수 * 2 여야 합니다"
        self.n_sensors = self.F_total // 2

        # 마스킹할 시간 길이 (48h 중 ratio 만큼)
        mask_span = 48.0 * pretrain_mask_ratio
        #mask_span = pretrain_mask_ratio

        # per‐patient per‐time mask for pretrain
        self.pre_masks = []
        for b in range(self.B):
            Ti = P_time[:, b]  # Tensor[T], 0~48 사이
            # 0 ~ (48 - mask_span) 구간에서 랜덤 시작점
            start = random.uniform(0.0, 48.0 - mask_span)
            # start = random.uniform(0.0, 1.0 - mask_span)
            end   = start + mask_span
            # 해당 시간 구간 안에 들어있는 타임스텝만 True
            mask = (Ti >= start) & (Ti < end)
            self.pre_masks.append(mask)

        # domain label
        self.domain = domain

        # event length 계산 (옵션)
        self.ev_lens = []
        for b in range(self.B):
            Xi = P[:, b, :self.n_sensors]
            mask_obs = Xi != 0
            t_idx, _ = mask_obs.nonzero(as_tuple=True)
            self.ev_lens.append(len(t_idx))

    def get_event_len(self, idx: int) -> int:
        return self.ev_lens[idx]

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        Xi = self.P[:, idx, :]
        Ti = self.P_time[:, idx]
        Si = self.P_static[idx]
        yi = int(self.y[idx].item())
        pid = self.ids[idx]

        values_mat = Xi[:, :self.n_sensors]
        mask_obs = values_mat != 0
        t_idx, f_idx = mask_obs.nonzero(as_tuple=True)

        # event-level로 flatten
        Ti = Ti.float() / 48.0
        #Ti = Ti.float() # For running P19
        times  = Ti[t_idx]                    # [n_obs]
        varis  = f_idx                        # [n_obs]
        values = values_mat[t_idx, f_idx]     # [n_obs]

        pad_mask = torch.zeros_like(times, dtype=torch.bool)
        pre_mask = self.pre_masks[idx][t_idx]  # [n_obs]
        domain   = self.domain

        # ---------- 정렬: time ↑, 같은 time 내 varis ↑ ----------
        # 1) varis 오름차순(안정 정렬)
        idx_var = torch.argsort(varis, stable=True)
        times   = times[idx_var]
        varis   = varis[idx_var]
        values  = values[idx_var]
        pre_mask= pre_mask[idx_var]
        pad_mask= pad_mask[idx_var]

        # 2) times 오름차순(안정 정렬) → 같은 time 내 varis 순서 보존
        idx_time = torch.argsort(times, stable=True)
        times   = times[idx_time]
        varis   = varis[idx_time]
        values  = values[idx_time]
        pre_mask= pre_mask[idx_time]
        pad_mask= pad_mask[idx_time]
        # -------------------------------------------------------

        return times, varis, values, pad_mask, pre_mask, yi, pid, Si, domain

class TimeSeriesDataset(Dataset):
    # Dataset class for Raindrop
    def __init__(self, P, Tm, S, y, ids):
        """
        P:    Tensor [T, B, F]
        Tm:   Tensor [T, B]
        S:    Tensor [B, D]
        y:    Tensor [B] or [B,1]
        ids:  array-like length B
        """
        self.P   = P
        self.Tm  = Tm
        self.S   = S
        self.y   = y.view(-1)
        self.ids = ids

    def __len__(self):
        return self.P.shape[1]

    def __getitem__(self, idx):
        # 인덱싱하면
        #   P[:,idx,:]   -> [T, F]
        #   Tm[:,idx]    -> [T]
        #   S[idx]       -> [D]
        #   y[idx]       -> scalar
        #   ids[idx]     -> pid
        return (
            self.P[:, idx, :],      # [T, F]
            self.Tm[:, idx],        # [T]
            self.S[idx],            # [D]
            self.y[idx],            # []
            self.ids[idx]           # identifier
        )


def prepare_data(
    processed_data_dir: str = './processed_data',
    source: str = 'mimic',
    target: str = 'eicu',
    outcome_idx: int = 3,   # 1:AKI,2:CF,3:LoS,4:Mortality
    split_seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
):
    # source load & split
    print("🪄 Readying data")
    src_PT = np.load(f"{processed_data_dir}/PTdict_list_{source}.npy", allow_pickle=True)
    src_yarr = np.load(f"{processed_data_dir}/arr_outcomes_{source}.npy", allow_pickle=True)
    src_y = src_yarr[:, outcome_idx].reshape(-1,1)

    idx = np.arange(len(src_PT))
    np.random.seed(split_seed)
    np.random.shuffle(idx)
    n_train = int(train_frac * len(idx))
    n_val = int(val_frac * len(idx))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    Ptrain, Pval, Ptest = src_PT[train_idx], src_PT[val_idx], src_PT[test_idx]
    ytrain, yval, ytest = src_y[train_idx], src_y[val_idx], src_y[test_idx]

    # compute stats on source-train
    T, F = Ptrain[0]['arr'].shape
    D = len(Ptrain[0]['extended_static'])

    arr_stack = np.stack([p['arr'] for p in Ptrain])
    sta_stack = np.stack([p['extended_static'] for p in Ptrain])

    mf, stdf = getStats_fixed(arr_stack)
    ms, ss = getStats_static(sta_stack, 'Default')

    # helper for tensorize+normalize
    # This part also converts minutes to hours
    def tz(P_list, y_list):
        return tensorize_normalize(P_list, y_list, mf, stdf, ms, ss)

    # source splits
    Ptrain_t, Ptrain_s, Ptrain_tm, ytrain_t = tz(Ptrain, ytrain)
    Pval_t, Pval_s, Pval_tm, yval_t = tz(Pval, yval)
    Ptest_t, Ptest_s, Ptest_tm, ytest_t = tz(Ptest, ytest)
    print("  ✅ Source data prepared")

    # target load & split + normalize with source stats)
    tgt_PT = np.load(f"{processed_data_dir}/PTdict_list_{target}.npy", allow_pickle=True)
    tgt_yarr = np.load(f"{processed_data_dir}/arr_outcomes_{target}.npy", allow_pickle=True)
    tgt_y = tgt_yarr[:, outcome_idx].reshape(-1,1)

    idx = np.arange(len(tgt_PT))
    np.random.seed(split_seed)
    np.random.shuffle(idx)
    n_train = int(train_frac * len(idx))
    n_val = int(val_frac * len(idx))

    t_train_idx = idx[:n_train]
    t_val_idx = idx[n_train:n_train+n_val]
    t_test_idx = idx[n_train+n_val:]

    Ptrain_target, Pval_target, Ptest_target = tgt_PT[t_train_idx], tgt_PT[t_val_idx], tgt_PT[t_test_idx]
    ytrain_target, yval_target, ytest_target = tgt_y[t_train_idx], tgt_y[t_val_idx], tgt_y[t_test_idx]

    Ptarget_t, Ptarget_s, Ptarget_tm, ytarget_t = tz(Ptrain_target, ytrain_target)
    Pval_target_t, Pval_target_s, Pval_target_tm, yval_target_t = tz(Pval_target, yval_target)
    Ptest_target_t, Ptest_target_s, Ptest_target_tm, ytest_target_t = tz(Ptest_target, ytest_target)
    print("  ✅ Target data prepared")

    # Permute time dims
    def permute(X, TM):
        return X.permute(1,0,2), TM.squeeze(-1).permute(1,0)

    Ptrain_t, Ptrain_tm = permute(Ptrain_t, Ptrain_tm)
    Pval_t, Pval_tm = permute(Pval_t, Pval_tm)
    Ptest_t, Ptest_tm = permute(Ptest_t, Ptest_tm)
    Ptarget_t, Ptarget_tm = permute(Ptarget_t, Ptarget_tm)
    Pval_target_t, Pval_target_tm = permute(Pval_target_t, Pval_target_tm)
    Ptest_target_t, Ptest_target_tm = permute(Ptest_target_t, Ptest_target_tm)

    # Ids
    train_ids = [p['id'] for p in Ptrain]
    val_ids = [p['id'] for p in Pval]
    test_ids = [p['id'] for p in Ptest]
    target_ids = [p['id'] for p in Ptrain_target]
    val_target_ids = [p['id'] for p in Pval_target]
    test_target_ids = [p['id'] for p in Ptest_target]

    # ====== ID를 JSON 파일로 저장 ======
    split_dir = os.path.join(processed_data_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # source
    source_split_path = os.path.join(split_dir, f"{source}_seed{split_seed}.json")
    with open(source_split_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }, f, ensure_ascii=False, indent=2)

    # target
    target_split_path = os.path.join(split_dir, f"{target}_seed{split_seed}.json")
    with open(target_split_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": target_ids,
            "val": val_target_ids,
            "test": test_target_ids
        }, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved split IDs to:\n  {source_split_path}\n  {target_split_path}")

    # 8) build dicts
    source_dict = {
        'Ptrain': Ptrain_t,
        'Pval': Pval_t,
        'Ptest': Ptest_t,
        'Ptrain_time':Ptrain_tm,
        'Pval_time':Pval_tm,
        'Ptest_time':Ptest_tm,
        'Ptrain_static':Ptrain_s,
        'Pval_static':Pval_s,
        'Ptest_static':Ptest_s,
        'ytrain': ytrain_t,
        'yval': yval_t,
        'ytest': ytest_t,
        'shape': {'T':T,'F':F,'D':D},
        'train_ids':train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
    }
    target_dict = {
        'Ptrain': Ptarget_t,
        'Pval': Pval_target_t,
        'Ptest': Ptest_target_t,
        'Ptrain_time':Ptarget_tm,
        'Pval_time':Pval_target_tm,
        'Ptest_time':Ptest_target_tm,
        'Ptrain_static':Ptarget_s,
        'Pval_static':Pval_target_s,
        'Ptest_static':Ptest_target_s,
        'ytrain': ytarget_t,
        'yval': yval_target_t,
        'ytest': ytest_target_t,
        'shape': {'T':T,'F':F,'D':D},
        'train_ids':target_ids,
        'val_ids': val_target_ids,
        'test_ids': test_target_ids,
    }
    counts = torch.bincount(ytrain_t)
    class_weights = (1 - (counts.float() / counts.sum())).tolist()

    return source_dict, target_dict, class_weights

def train_model(run_name: str,
                data: dict,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                num_epochs: int = 5,
                batch_size: int = 16,
                patience: int = 5,
                device: torch.device = torch.device('cuda'),
                wandb=None):

    # 1) Dataset / DataLoader 생성
    train_ds = TimeSeriesDataset(
        data['Ptrain'], data['Ptrain_time'],
        data['Ptrain_static'], data['ytrain'], data['train_ids']
    )
    val_ds = TimeSeriesDataset(
        data['Pval'], data['Pval_time'],
        data['Pval_static'], data['yval'], data['val_ids']
    )
    test_ds = TimeSeriesDataset(
        data['Ptest'], data['Ptest_time'],
        data['Ptest_static'], data['ytest'], data['test_ids']
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True        
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0
    model_path = f"./rd_models/best_raindrop_model_{run_name}.pt"

    for epoch in range(num_epochs):
        epoch_start = datetime.datetime.now()

        # === 2) Training ===
        model.train()
        train_losses, all_preds, all_probs, all_labels = [], [], [], []
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Train", unit="batch")
        for X_b, Tm_b, S_b, y_b, _ in pbar:
            # X_b: [T, F] → DataLoader stacks to [B, T, F]
            # Tm_b: [T]   → stacks to [B, T]
            X = X_b.to(device, non_blocking=True).permute(1,0,2) # [B,T,F]→[T,B,F]
            Tm = Tm_b.to(device, non_blocking=True).permute(1,0) # [B,T]→[T,B]
            S = S_b.to(device, non_blocking=True) # [B,D]
            y = y_b.to(device, non_blocking=True) # [B]

            lengths = (Tm>0).sum(dim=0)        # [B]

            optimizer.zero_grad()
            out, _, _ = model(X, S, Tm, lengths)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            proba = torch.softmax(out, dim=1)[:,1].detach().cpu().numpy()
            pred = (proba>=0.5).astype(int)
            label = y.cpu().numpy()

            all_probs.extend(proba)
            all_preds.extend(pred)
            all_labels.extend(label)

            pbar.set_postfix(loss=loss.item())

        train_loss = sum(train_losses)/len(train_losses)
        train_acc = accuracy_score(all_labels, all_preds)
        train_auroc = roc_auc_score(all_labels, all_probs)
        train_auprc = average_precision_score(all_labels, all_probs)

        # === 3) Validation ===
        model.eval()
        val_losses, v_preds, v_probs, v_labels = [], [], [], []
        pbar = tqdm(val_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Valid", unit="batch")
        with torch.no_grad():
            for X_b, Tm_b, S_b, y_b, _ in pbar:
                X = X_b.to(device, non_blocking=True).permute(1,0,2) # [B,T,F]→[T,B,F]
                Tm = Tm_b.to(device, non_blocking=True).permute(1,0) # [B,T]→[T,B]
                S = S_b.to(device, non_blocking=True) # [B,D]
                y = y_b.to(device, non_blocking=True) # [B]

                lengths = (Tm>0).sum(dim=0)
                out, _, _ = model(X, S, Tm, lengths)
                loss = criterion(out, y)

                val_losses.append(loss.item())
                proba = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                pred = (proba>=0.5).astype(int)
                label = y.cpu().numpy()

                v_probs.extend(proba)
                v_preds.extend(pred)
                v_labels.extend(label)

        valid_loss = sum(val_losses)/len(val_losses)
        valid_acc = accuracy_score(v_labels, v_preds)
        valid_auroc = roc_auc_score(v_labels, v_probs)
        valid_auprc = average_precision_score(v_labels, v_probs)

        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        print(f"Epoch {epoch+1}: "
              f"Train loss {train_loss:.4f} | Valid loss {valid_loss:.4f} "
              f"({epoch_time:.1f}s)")

        # wandb 로깅
        if wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "train_auroc": train_auroc,
                "valid_auroc": valid_auroc,
                "train_auprc": train_auprc,
                "valid_auprc": valid_auprc,
                "epoch_time_s": epoch_time
            })

        # early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # === 4) Test 평가: metric만 계산 ===
    model.load_state_dict(torch.load(model_path))
    model.eval()
    t_preds, t_probs, t_labels = [], [], []
    pbar = tqdm(test_loader, desc="[Best model] Test", unit="batch")
    with torch.no_grad():
        for X_b, Tm_b, S_b, y_b, _ in pbar:
            X = X_b.to(device, non_blocking=True).permute(1,0,2) # [B,T,F]→[T,B,F]
            Tm = Tm_b.to(device, non_blocking=True).permute(1,0) # [B,T]→[T,B]
            S = S_b.to(device, non_blocking=True) # [B,D]
            y = y_b.to(device, non_blocking=True) # [B]

            lengths = (Tm>0).sum(dim=0)
            out, _, _ = model(X, S, Tm, lengths)

            proba = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            pred = (proba>=0.5).astype(int)
            label = y.cpu().numpy()

            t_probs.extend(proba)
            t_preds.extend(pred)
            t_labels.extend(label)

    test_loss = criterion(out, y).item()
    test_acc = accuracy_score(t_labels, t_preds)
    test_auroc = roc_auc_score(t_labels, t_probs)
    test_auprc = average_precision_score(t_labels, t_probs)

    if wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc
        })

    # 최종 모델 반환
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_split(model: nn.Module, 
                   device: torch.device,
                   loader: DataLoader,
                   data_type: str) -> pd.DataFrame:
    """
    DataLoader 로부터 (X_cpu, Tm_cpu, S_cpu, y_cpu, ids) 배치를 받아
    모델을 돌리고 pid, label, pred, proba, embs 컬럼이 있는 DataFrame 반환.
    """
    model.eval()
    rows = []
    pbar = tqdm(loader, desc=f"[Evaluating {data_type}]", unit="batch")
    with torch.no_grad():
        for X_cpu, Tm_cpu, S_cpu, y_cpu, ids_cpu in pbar:
            # (B, T, F) → (T, B, F)
            X = X_cpu.permute(1,0,2).to(device, non_blocking=True)
            # (B, T) → (T, B)
            Tm = Tm_cpu.permute(1,0).to(device, non_blocking=True)
            # (B, D)
            S = S_cpu.to(device, non_blocking=True)
            # (B,)
            y = y_cpu.to(device, non_blocking=True)

            lengths = (Tm > 0).sum(dim=0)
            out, _, embs = model(X, S, Tm, lengths)

            proba = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            preds = (proba >= 0.5).astype(int)
            trues = y_cpu.numpy()             # 아직 CPU 텐서
            embs_np = embs.cpu().numpy()      # (B, emb_dim)

            for pid, t, pr, p, emb in zip(ids_cpu, trues, preds, proba, embs_np):
                rows.append({
                    'pid': int(pid),
                    'label': int(t),
                    'pred': int(pr),
                    'proba': float(p),
                    'embs': emb.copy()
                })

    return pd.DataFrame(rows)

def get_embs(data, model, device, batch_size, name, data_type, metric = False):
    # 1) Dataset / DataLoader 생성

    print(f"😇 Getting embeddings... for batch size {batch_size} and data {data_type}")
    ds = TimeSeriesDataset(
        data[f'P{data_type}'], data[f'P{data_type}_time'],
        data[f'P{data_type}_static'], data[f'y{data_type}'], data[f'{data_type}_ids']
    )
    print("  ✅ Dataset Created")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    print("  ✅ DataLoader Created")

    df = evaluate_split(model, device, loader, data_type)
    print("    ✅ Split complete")
    emb_dim = df['embs'].iloc[0].shape[0]
    emb_cols = [f'emb_{i}' for i in range(emb_dim)]
    emb_df = pd.DataFrame(df['embs'].tolist(), columns=emb_cols)
    out = pd.concat([df.drop(columns='embs'), emb_df], axis=1)
    if metric==True:
        try:
            acc = accuracy_score(out['label'], out['pred'])
            auroc= roc_auc_score(out['label'], out['proba'])
            auprc= average_precision_score(out['label'], out['proba'])
            print(f"      ✅ [{data_type}] Acc:{acc:.4f}, AUROC:{auroc:.4f}, AUPRC:{auprc:.4f}")
        except:
            print("      ❗Failed to get metrics; Skipping")
    
    out.to_feather(f"./rd_results/predictions_{data_type}_{name}.feather")
    print("    ✅ Embeddings saved")

    
    print(f"✅ Job finished for batch size {batch_size} and data {data_type}")


# From here : STraTS
def train_model_STraTS(run_name: str,
                source: str,
                data: dict,
                model: torch.nn.Module,
                use_pretrain: bool,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                num_epochs: int = 5,
                batch_size: int = 16,
                patience: int = 5,
                device: torch.device = torch.device('cuda'),
                wandb=None):

    # 1) Dataset / DataLoader 생성
    train_ds = STraTSDataset(
        data['Ptrain'], data['Ptrain_time'],
        data['Ptrain_static'], data['ytrain'], data['train_ids'], pretrain_mask_ratio=0.2
    )
    val_ds = STraTSDataset(
        data['Pval'], data['Pval_time'],
        data['Pval_static'], data['yval'], data['val_ids']
    )
    test_ds = STraTSDataset(
        data['Ptest'], data['Ptest_time'],
        data['Ptest_static'], data['ytest'], data['test_ids']
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        pin_memory=True        
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, pin_memory=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    pretr_model_path = f"./rd_models/best_strats_model_{source}_pretrained.pt"
    model_path = f"./rd_models/best_strats_model_{run_name}.pt"

    # Pretraining loop
    if use_pretrain == True:
        try:
            model.load_state_dict(torch.load(pretr_model_path))
        except:
            # pretraining code
            for epoch in range(1, num_epochs+1):
                epoch_start = datetime.datetime.now()
                model.train()
                train_losses = []
                pbar = tqdm(train_loader, desc=f"[Pre:Epoch {epoch}/{num_epochs}]")
                for times, varis, values, pad_mask, pre_mask, outcomes, _, statics in pbar:
                    times = times.to(device, non_blocking=True)
                    varis = varis.to(device, non_blocking=True)
                    values = values.to(device, non_blocking=True)
                    statics = statics.to(device, non_blocking=True)
                    pad_mask = pad_mask.to(device, non_blocking=True)
                    pre_mask = pre_mask.to(device, non_blocking=True)
                    outcomes = outcomes.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    # (A) pretrain: forecast
                    out_pt = model(
                        times, varis, values, statics,
                        padding_mask=pad_mask,
                        pretrain=True,
                        pretrain_mask=pre_mask
                    )

                    # returns {
                    # 'forecast': forecast_selected,
                    # 'values': values,
                    # 'varis': varis,
                    # 'times': times,
                    # 'mask': pretrain_mask,  # (bool)
                    # 'loss': mse_loss
                    #  }

                    loss_pt = out_pt['loss']
                    loss_pt.backward()
                    optimizer.step()

                    train_losses.append(loss_pt.item())

        # === 3) Validation ===
                model.eval()
                val_losses = []
                pbar = tqdm(val_loader, desc=f"[Pre:Epoch {epoch+1}/{num_epochs}] Valid", unit="batch")
                with torch.no_grad():
                    for times, varis, values, pad_mask, pre_mask, outcomes, _, statics in pbar:
                        times = times.to(device, non_blocking=True)
                        varis = varis.to(device, non_blocking=True)
                        values = values.to(device, non_blocking=True)
                        statics = statics.to(device, non_blocking=True)
                        pad_mask = pad_mask.to(device, non_blocking=True)
                        pre_mask = pre_mask.to(device, non_blocking=True)
                        outcomes = outcomes.to(device, non_blocking=True)

                        optimizer.zero_grad()
                        # (A) pretrain: forecast
                        out_pt = model(
                            times, varis, values, statics,
                            padding_mask=pad_mask,
                            pretrain=True,
                            pretrain_mask=pre_mask
                        )
                        loss_pt = out_pt['loss']
                        val_losses.append(loss_pt.item())
                epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                print(f"[Pretraining] Epoch {epoch+1}: "
                    f"Train loss {train_loss:.4f} | Valid loss {val_loss:.4f} "
                    f"({epoch_time:.1f}s)")

                # wandb 로깅
                if wandb:
                    wandb.log({
                        "pre_epoch": epoch+1,
                        "train_pre_loss": train_loss,
                        "valid_pre_loss": val_loss,
                        "pre_epoch_time_s": epoch_time
                    })

                # early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), pretr_model_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[Pretraining] Early stopping at epoch {epoch+1}")
                        break
            model.load_state_dict(torch.load(pretr_model_path))
        
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        epoch_start = datetime.datetime.now()

        # === 2) Training ===
        model.train()
        train_losses, all_preds, all_probs, all_labels = [], [], [], []
        pbar = tqdm(train_loader, desc=f"[Down:Epoch {epoch+1}/{num_epochs}] Train", unit="batch")
        for times, varis, values, pad_mask, pre_mask, outcomes, _, statics in pbar:
            times = times.to(device, non_blocking=True)
            varis = varis.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            pre_mask = pre_mask.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            # (A) train: forecast
            out, _ = model(
                times, varis, values, statics,
                padding_mask=pad_mask,
            )
            loss = criterion(out, outcomes)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            #proba = torch.softmax(out, dim=1)[:,1].detach().cpu().numpy()
            proba = out.detach().cpu().numpy()
            pred = (proba>=0.5).astype(int)
            label = outcomes.cpu().numpy()

            all_probs.extend(proba)
            all_preds.extend(pred)
            all_labels.extend(label)

            pbar.set_postfix(loss=loss.item())

        train_loss = sum(train_losses)/len(train_losses)
        train_acc = accuracy_score(all_labels, all_preds)
        train_auroc = roc_auc_score(all_labels, all_probs)
        train_auprc = average_precision_score(all_labels, all_probs)

        # === 3) Validation ===
        model.eval()
        val_losses, v_preds, v_probs, v_labels = [], [], [], []
        pbar = tqdm(val_loader, desc=f"[Down:Epoch {epoch+1}/{num_epochs}] Valid", unit="batch")
        with torch.no_grad():
            for times, varis, values, pad_mask, pre_mask, outcomes, _, statics in pbar:
                times = times.to(device, non_blocking=True)
                varis = varis.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                statics = statics.to(device, non_blocking=True)
                pad_mask = pad_mask.to(device, non_blocking=True)
                pre_mask = pre_mask.to(device, non_blocking=True)
                outcomes = outcomes.to(device, non_blocking=True).float()

                optimizer.zero_grad()
                # (A) train: forecast
                out, _ = model(
                    times, varis, values, statics,
                    padding_mask=pad_mask,
                )
                loss = criterion(out, outcomes)

                val_losses.append(loss.item())
                #proba = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                proba = out.detach().cpu().numpy()
                pred = (proba>=0.5).astype(int)
                label = outcomes.cpu().numpy()

                v_probs.extend(proba)
                v_preds.extend(pred)
                v_labels.extend(label)

        valid_loss = sum(val_losses)/len(val_losses)
        valid_acc = accuracy_score(v_labels, v_preds)
        valid_auroc = roc_auc_score(v_labels, v_probs)
        valid_auprc = average_precision_score(v_labels, v_probs)

        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        print(f"[Downstream] Epoch {epoch+1}: "
              f"Train loss {train_loss:.4f} | Valid loss {valid_loss:.4f} "
              f"({epoch_time:.1f}s)")

        # wandb 로깅
        if wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "train_auroc": train_auroc,
                "valid_auroc": valid_auroc,
                "train_auprc": train_auprc,
                "valid_auprc": valid_auprc,
                "epoch_time_s": epoch_time
            })

        # early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # === 4) Test 평가: metric만 계산 ===
    model.load_state_dict(torch.load(model_path))
    model.eval()
    t_preds, t_probs, t_labels = [], [], []
    pbar = tqdm(test_loader, desc="[Best model] Test", unit="batch")
    with torch.no_grad():
        for times, varis, values, pad_mask, pre_mask, outcomes, _, statics in pbar:
            times = times.to(device, non_blocking=True)
            varis = varis.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            pre_mask = pre_mask.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            # (A) train: forecast
            out, _ = model(
                times, varis, values, statics,
                padding_mask=pad_mask,
            )
            #proba = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            proba = out.detach().cpu().numpy()
            pred = (proba>=0.5).astype(int)
            label = outcomes.cpu().numpy()

            t_probs.extend(proba)
            t_preds.extend(pred)
            t_labels.extend(label)
          

    test_loss = criterion(out, outcomes).item()
    test_acc = accuracy_score(t_labels, t_preds)
    test_auroc = roc_auc_score(t_labels, t_probs)
    test_auprc = average_precision_score(t_labels, t_probs)

    if wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc
        })

    # 최종 모델 반환
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_split_STraTS(model: nn.Module, 
                   device: torch.device,
                   loader: DataLoader,
                   data_type: str) -> pd.DataFrame:
    """
    DataLoader 로부터 (Time, Variable, Value, Padding mask, Pretrain mask, Outcome, ID, Static) 배치를 받아
    모델을 돌리고 pid, label, pred, proba, embs 컬럼이 있는 DataFrame 반환.
    """
    model.eval()
    rows = []
    pbar = tqdm(loader, desc=f"[Evaluating {data_type}]", unit="batch")
    with torch.no_grad():
        for times, varis, values, pad_mask, _, outcomes, ids, statics in pbar:
            times = times.to(device, non_blocking=True)
            varis = varis.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)


            # (A) train: forecast
            out, embs = model(
                times, varis, values, statics,
                padding_mask=pad_mask,
            )

            proba = out.detach().cpu().numpy()
            preds = (proba >= 0.5).astype(int)
            trues = outcomes.numpy()             # 아직 CPU 텐서
            embs_np = embs.cpu().numpy()      # (B, emb_dim)

            for pid, t, pr, p, emb in zip(ids, trues, preds, proba, embs_np):
                rows.append({
                    'pid': int(pid),
                    'label': int(t),
                    'pred': int(pr),
                    'proba': float(p),
                    'embs': emb.copy()
                })

    return pd.DataFrame(rows)

def get_embs_STraTS(data, model, device, batch_size, name, data_type, metric = False):
    # 1) Dataset / DataLoader 생성

    print(f"😇 Getting embeddings... for batch size {batch_size} and data {data_type}")
    ds = STraTSDataset(
        data[f'P{data_type}'], data[f'P{data_type}_time'],
        data[f'P{data_type}_static'], data[f'y{data_type}'], data[f'{data_type}_ids']
    )
    print("  ✅ Dataset Created")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, pin_memory=True)
    print("  ✅ DataLoader Created")

    df = evaluate_split_STraTS(model, device, loader, data_type)
    print("    ✅ Split complete")
    emb_dim = df['embs'].iloc[0].shape[0]
    emb_cols = [f'emb_{i}' for i in range(emb_dim)]
    emb_df = pd.DataFrame(df['embs'].tolist(), columns=emb_cols)
    out = pd.concat([df.drop(columns='embs'), emb_df], axis=1)
    if metric==True:
        try:
            acc = accuracy_score(out['label'], out['pred'])
            auroc= roc_auc_score(out['label'], out['proba'])
            auprc= average_precision_score(out['label'], out['proba'])
            print(f"      ✅ [{data_type}] Acc:{acc:.4f}, AUROC:{auroc:.4f}, AUPRC:{auprc:.4f}")
        except:
            print("      ❗Failed to get metrics; Skipping")
    
    out.to_feather(f"./rd_results/predictions_{data_type}_{name}.feather")
    print("    ✅ Embeddings saved")

    
    print(f"✅ Job finished for batch size {batch_size} and data {data_type}")

# From here : STraTSGR
# Added DANN to ensure homogeneity between source / target sets
def train_model_STraTSGR(run_name: str,
                source: str,
                data: dict,
                model: torch.nn.Module,
                use_pretrain: bool,
                use_dann: bool,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                num_epochs: int = 5,
                batch_size: int = 16,
                patience: int = 5,
                device: torch.device = torch.device('cuda'),
                wandb=None,
                seed: int = 42,
                varwise: bool=False,
                model_name: str = 'STraTS'):
    try:
        pad_collate_dom = FeatureCollator(data['shape']['F'])
    except:
        pad_collate_dom = FeatureCollator(data['source']['shape']['F'])
    # 1) Dataset / DataLoader 생성
    # Use combined datasets
    # combined = {
    # 'source': source_dict,
    # 'target': target_dict
    # }
    cri_dom = nn.BCELoss()
    # train_source_ds = STraTSDatasetGR(
    #     data['source']['Ptrain'], data['source']['Ptrain_time'],
    #     data['source']['Ptrain_static'], data['source']['ytrain'], data['source']['train_ids'], domain=0, pretrain_mask_ratio=0.2
    # )
    # train_target_ds = STraTSDatasetGR(
    #     data['target']['Ptrain'], data['target']['Ptrain_time'],
    #     data['target']['Ptrain_static'], data['target']['ytrain'], data['target']['train_ids'], 1
    # )
    # val_source_ds = STraTSDatasetGR(
    #     data['source']['Pval'], data['source']['Pval_time'],
    #     data['source']['Pval_static'], data['source']['yval'], data['source']['val_ids'], 0
    # )
    # val_target_ds = STraTSDatasetGR(
    #     data['target']['Pval'], data['target']['Pval_time'],
    #     data['target']['Pval_static'], data['target']['yval'], data['target']['val_ids'], 1
    # )
    # test_source_ds = STraTSDatasetGR(
    #     data['source']['Ptest'], data['source']['Ptest_time'],
    #     data['source']['Ptest_static'], data['source']['ytest'], data['source']['test_ids'], 0
    # )
    # test_target_ds = STraTSDatasetGR(
    #     data['target']['Ptest'], data['target']['Ptest_time'],
    #     data['target']['Ptest_static'], data['target']['ytest'], data['target']['test_ids'], 1
    # )
    # train_ds = torch.utils.data.ConcatDataset([train_source_ds, train_target_ds])
    # val_ds = torch.utils.data.ConcatDataset([val_source_ds, val_target_ds])
    # test_ds = torch.utils.data.ConcatDataset([test_source_ds, test_target_ds])

    train_ds = STraTSDatasetGR(
        data['Ptrain'], data['Ptrain_time'],
        data['Ptrain_static'], data['ytrain'], data['train_ids'], domain=0, pretrain_mask_ratio=0.2
    )

    val_ds = STraTSDatasetGR(
        data['Pval'], data['Pval_time'],
        data['Pval_static'], data['yval'], data['val_ids'], 0
    )

    test_ds = STraTSDatasetGR(
        data['Ptest'], data['Ptest_time'],
        data['Ptest_static'], data['ytest'], data['test_ids'], 0
    )

    # train_sampler = BucketBatchSampler(train_ds, batch_size=batch_size,
    #                             drop_last=False, base_seed=seed,
    #                             length_fn=lambda i: concat_length_fn(train_ds, i))

    # val_sampler = BucketBatchSampler(val_ds, batch_size=batch_size,
    #                         drop_last=False, base_seed=seed, shuffle=False,
    #                         length_fn=lambda i: concat_length_fn(val_ds, i))

    train_sampler = BucketBatchSampler(train_ds, batch_size=batch_size,
                                drop_last=False, base_seed=seed, shuffle=True)

    val_sampler = BucketBatchSampler(val_ds, batch_size=batch_size,
                            drop_last=False, base_seed=seed, shuffle=False,)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_dom, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=pad_collate_dom, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_dom, pin_memory=True)
    # train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=pad_collate_dom, pin_memory=True)
    #val_loader = DataLoader(val_ds, collate_fn=pad_collate_dom, pin_memory=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    if use_dann:
        dann ='_dann'
    else:
        dann = ''
    if varwise:
        va = 'Var'
    else:
        va = ''
    pretr_model_path = f"./rd_models/best{model_name}_model_{source}.pt"
    model_path = f"./rd_models/best_{run_name}.pt"

    # Pretraining loop
    if use_pretrain == True:
        try:
            model.load_state_dict(torch.load(pretr_model_path))
            print(f"Loaded pretrained model from {pretr_model_path}")
        except:
            # pretraining code
            for epoch in range(1, num_epochs+1):
                epoch_start = datetime.datetime.now()
                model.train()
                train_losses, train_dom_losses, train_pre_losses = [], [], []
                pbar = tqdm(train_loader, desc=f"[Pre:Epoch {epoch}/{num_epochs}]")
                for times, varis, values, pad_mask, pre_mask, outcomes, _, statics, domain in pbar:
                    
                    times = times.to(device, non_blocking=True)
                    varis = varis.to(device, non_blocking=True)
                    values = values.to(device, non_blocking=True)
                    statics = statics.to(device, non_blocking=True)
                    pad_mask = pad_mask.to(device, non_blocking=True)
                    pre_mask = pre_mask.to(device, non_blocking=True)
                    outcomes = outcomes.to(device, non_blocking=True)
                    domain = domain.to(device, non_blocking=True).float()

                    

                    optimizer.zero_grad()
                    # (A) pretrain: forecast
                    out_dict = model(
                        times, varis, values, statics,
                        padding_mask=pad_mask,
                        pretrain=True,
                        pretrain_mask=pre_mask,
                    )

                    # returns dictionary {
                    #     'forecast': forecast_sel,
                    #     'values': values,
                    #     'varis': varis,
                    #     'times': times,
                    #     'mask':  pretrain_mask,
                    #     'loss':  mse_loss,
                    #     'dom_logit': dom_logit
                    # }
                    cri_dom = nn.BCEWithLogitsLoss()
                    loss_pt = out_dict['loss']
                    # if use_dann:
                    #     loss_domain = cri_dom(out_dict['dom_logit'], domain)
                    #     loss_pt = loss_masked_pred + loss_domain
                    # else:
                    #     loss_domain = torch.tensor(0., device=values.device)
                    #     loss_pt = loss_masked_pred
                    loss_pt.backward()
                    optimizer.step()

                    train_losses.append(loss_pt.item())
                    # train_pre_losses.append(loss_masked_pred.item())
                    # train_dom_losses.append(loss_domain.item())

        # === 3) Validation ===
                model.eval()
                val_losses, val_dom_losses, val_pre_losses = [], [], []

                pbar = tqdm(val_loader, desc=f"[Pre:Epoch {epoch}/{num_epochs}] Valid", unit="batch")
                with torch.no_grad():
                    for times, varis, values, pad_mask, pre_mask, outcomes, _, statics, domain in pbar:
                        times = times.to(device, non_blocking=True)
                        varis = varis.to(device, non_blocking=True)
                        values = values.to(device, non_blocking=True)
                        statics = statics.to(device, non_blocking=True)
                        pad_mask = pad_mask.to(device, non_blocking=True)
                        pre_mask = pre_mask.to(device, non_blocking=True)
                        outcomes = outcomes.to(device, non_blocking=True)
                        domain = domain.to(device, non_blocking=True).float()

                        optimizer.zero_grad()
                        # (A) pretrain: forecast
                        out_dict = model(
                            times, varis, values, statics,
                            padding_mask=pad_mask,
                            pretrain=True,
                            pretrain_mask=pre_mask
                        )
                        cri_dom = nn.BCEWithLogitsLoss()
                        loss_pt = out_dict['loss']
                        # if use_dann:
                        #     loss_domain = cri_dom(out_dict['dom_logit'], domain)
                        #     loss_pt = loss_masked_pred + loss_domain
                        # else:
                        #     loss_domain = torch.tensor(0., device=values.device)
                        #     loss_pt = loss_masked_pred
                        

                        val_losses.append(loss_pt.item())
                        # val_pre_losses.append(loss_masked_pred.item())
                        # val_dom_losses.append(loss_domain.item())
                epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                # train_pre_loss = sum(train_pre_losses) / len(train_pre_losses)
                # val_pre_loss = sum(val_pre_losses) / len(val_pre_losses)
                # train_dom_loss = sum(train_dom_losses) / len(train_dom_losses)
                # val_dom_loss = sum(val_dom_losses) / len(val_dom_losses)
                print(f"[Pretraining] Epoch {epoch}: "
                    f"Train loss {train_loss:.4f} | Valid loss {val_loss:.4f} "
                    f"({epoch_time:.1f}s)")

                # wandb 로깅
                if wandb:
                    pt_log_dict = {
                        "pre_epoch": epoch,
                        "train_pre_loss": train_loss,
                        "valid_pre_loss": val_loss,
                        # "train_pre_pre_loss" : train_pre_loss,
                        # "valid_pre_pre_loss" : val_pre_loss,
                        "pre_epoch_time_s": epoch_time
                    }
                    

                    # if use_dann:
                    #     pt_log_dict.update({
                    #         "train_pre_dom_loss" : train_dom_loss,
                    #         "valid_pre_dom_loss" : val_dom_loss,
                    #     })
                    wandb.log(pt_log_dict)
                # early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), pretr_model_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"[Pretraining] Early stopping at epoch {epoch}")
                        break
            model.load_state_dict(torch.load(pretr_model_path))
        
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_sampler = BucketBatchSampler(train_ds, batch_size=batch_size,
                                drop_last=False, base_seed=seed,)
                                #length_fn=lambda i: concat_length_fn(train_ds, i))

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_dom, pin_memory=True)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch+30)
        epoch_start = datetime.datetime.now()

        # === 2) Training ===
        model.train()
        train_losses, train_losses_downstream, train_losses_domain, all_preds, all_probs, all_labels = [], [], [], [], [], []
        pbar = tqdm(train_loader, desc=f"[Down:Epoch {epoch+1}/{num_epochs}] Train", unit="batch")
        for times, varis, values, pad_mask, pre_mask, outcomes, _, statics, domain in pbar:
            
            times = times.to(device, non_blocking=True)
            varis = varis.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            pre_mask = pre_mask.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True).float()
            domain = domain.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            # (A) train: forecast
            out_dict = model(
                times, varis, values, statics,
                padding_mask=pad_mask,
            )
            # Logit returned
            loss_downstream = criterion(out_dict['pred'], outcomes)
            # loss = loss_downstream
            cri_dom = nn.BCEWithLogitsLoss()
            if use_dann:
                loss_domain = cri_dom(out_dict['pred_domain'], domain)
                loss = loss_downstream + loss_domain
            else:
                loss_domain = torch.tensor(0., device=values.device)
                loss = loss_downstream
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_losses_downstream.append(loss_downstream.item())
            train_losses_domain.append(loss_domain.item())
            # proba = torch.softmax(out, dim=1)[:,1].detach().cpu().numpy()
            proba = torch.sigmoid(out_dict['pred']).detach().cpu().numpy()
            pred = (proba>=0.5).astype(int)
            label = outcomes.cpu().numpy()

            all_probs.extend(proba)
            all_preds.extend(pred)
            all_labels.extend(label)

            pbar.set_postfix(loss=loss.item())

        train_loss = sum(train_losses)/len(train_losses)
        train_loss_downstream = sum(train_losses_downstream)/len(train_losses_downstream)
        train_loss_domain = sum(train_losses_domain)/len(train_losses_domain)
        train_acc = accuracy_score(all_labels, all_preds)
        train_auroc = roc_auc_score(all_labels, all_probs)
        train_auprc = average_precision_score(all_labels, all_probs)

        # === 3) Validation ===
        model.eval()
        val_losses, val_losses_downstream, val_losses_domain, v_preds, v_probs, v_labels = [], [], [], [], [], []
        pbar = tqdm(val_loader, desc=f"[Down:Epoch {epoch+1}/{num_epochs}] Valid", unit="batch")
        with torch.no_grad():
            for times, varis, values, pad_mask, pre_mask, outcomes, _, statics, domain in pbar:
                times = times.to(device, non_blocking=True)
                varis = varis.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                statics = statics.to(device, non_blocking=True)
                pad_mask = pad_mask.to(device, non_blocking=True)
                pre_mask = pre_mask.to(device, non_blocking=True)
                outcomes = outcomes.to(device, non_blocking=True).float()
                domain = domain.to(device, non_blocking=True).float()

                optimizer.zero_grad()
                # (A) train: forecast
                out_dict = model(
                    times, varis, values, statics,
                    padding_mask=pad_mask,
                )
                loss_downstream = criterion(out_dict['pred'], outcomes)
                # loss = loss_downstream
                cri_dom = nn.BCEWithLogitsLoss()
                if use_dann:
                    loss_domain = cri_dom(out_dict['pred_domain'], domain)
                    loss = loss_downstream + loss_domain
                else:
                    loss_domain = torch.tensor(0., device=values.device)
                    loss = loss_downstream

                val_losses.append(loss.item())
                val_losses_downstream.append(loss_downstream.item())
                val_losses_domain.append(loss_domain.item())
                # proba = torch.softmax(out, dim=1)[:,1].detach().cpu().numpy()
                proba = torch.sigmoid(out_dict['pred']).detach().cpu().numpy()
                pred = (proba>=0.5).astype(int)
                label = outcomes.cpu().numpy()

                v_probs.extend(proba)
                v_preds.extend(pred)
                v_labels.extend(label)

        valid_loss = sum(val_losses)/len(val_losses)
        val_loss_downstream = sum(val_losses_downstream)/len(val_losses_downstream)
        val_loss_domain = sum(val_losses_domain)/len(val_losses_domain)
        valid_acc = accuracy_score(v_labels, v_preds)
        valid_auroc = roc_auc_score(v_labels, v_probs)
        valid_auprc = average_precision_score(v_labels, v_probs)

        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        print(f"[Downstream] Epoch {epoch+1}: "
              f"Train loss {train_loss:.4f} | Valid loss {valid_loss:.4f} "
              f"({epoch_time:.1f}s)")

        # wandb 로깅
        if wandb:
            log_dict = {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_loss_downstream" : train_loss_downstream,
                "valid_loss_downstream" : val_loss_downstream,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "train_auroc": train_auroc,
                "valid_auroc": valid_auroc,
                "train_auprc": train_auprc,
                "valid_auprc": valid_auprc,
                "epoch_time_s": epoch_time
            }
            if use_dann:
                log_dict.update({
                    "train_loss_domain": train_loss_domain,
                    "valid_loss_domain": val_loss_domain,
                })

            wandb.log(log_dict)

        # early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # === 4) Test 평가: metric만 계산 ===
    model.load_state_dict(torch.load(model_path))
    model.eval()
    test_losses, t_preds, t_probs, t_labels = [], [], [], []
    pbar = tqdm(test_loader, desc="[Best model] Test", unit="batch")
    with torch.no_grad():
        for times, varis, values, pad_mask, pre_mask, outcomes, _, statics, domain in pbar:
            times = times.to(device, non_blocking=True)
            varis = varis.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            pre_mask = pre_mask.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True).float()
            domain = domain.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            # (A) train: forecast
            out_dict = model(
                times, varis, values, statics,
                padding_mask=pad_mask,
            )
            #proba = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            loss_downstream = criterion(out_dict['pred'], outcomes)
            # loss = loss_downstream
            if use_dann:
                loss_domain = cri_dom(out_dict['pred_domain'], domain)
                loss = loss_downstream + loss_domain
            else:
                loss_domain = torch.tensor(0., device=values.device)
                loss = loss_downstream

            test_losses.append(loss.item())
            proba = torch.sigmoid(out_dict['pred']).detach().cpu().numpy()
            pred = (proba>=0.5).astype(int)
            label = outcomes.cpu().numpy()

            test_losses.append(loss.item())

            t_probs.extend(proba)
            t_preds.extend(pred)
            t_labels.extend(label)

    test_loss = sum(test_losses)/len(test_losses)
    test_acc = accuracy_score(t_labels, t_preds)
    test_auroc = roc_auc_score(t_labels, t_probs)
    test_auprc = average_precision_score(t_labels, t_probs)

    if wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc
        })

    # 최종 모델 반환
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_split_STraTSGR(model: nn.Module, 
                   device: torch.device,
                   loader: DataLoader,
                   data_type: str) -> pd.DataFrame:
    """
    DataLoader 로부터 (Time, Variable, Value, Padding mask, Pretrain mask, Outcome, ID, Static) 배치를 받아
    모델을 돌리고 pid, label, pred, proba, embs 컬럼이 있는 DataFrame 반환.
    """
    model.eval()
    rows = []
    pbar = tqdm(loader, desc=f"[Evaluating {data_type}]", unit="batch")
    with torch.no_grad():
        for times, varis, values, pad_mask, _, outcomes, ids, statics, domain in pbar:
            times = times.to(device, non_blocking=True)
            varis = varis.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)

            # (A) train: forecast
            out_dict = model(
                times, varis, values, statics,
                padding_mask=pad_mask,
            )

            proba = torch.sigmoid(out_dict['pred']).detach().cpu().numpy()
            preds = (proba >= 0.5).astype(int)
            trues = outcomes.numpy()             # 아직 CPU 텐서
            embs_np = out_dict['embs'].cpu().numpy()      # (B, emb_dim)

            for pid, t, pr, p, emb, d in zip(ids, trues, preds, proba, embs_np, domain):
                rows.append({
                    'pid': pid,
                    'domain': int(d),
                    'label': int(t),
                    'pred': int(pr),
                    'proba': float(p),
                    'embs': emb.copy()
                })

    return pd.DataFrame(rows)


def evaluate_split_STraTSGR(model: nn.Module, 
                            device: torch.device,
                            loader: DataLoader,
                            data_type: str) -> pd.DataFrame:
    """
    DataLoader 로부터 (Time, Variable, Value, Padding mask, Pretrain mask, Outcome, ID, Static) 배치를 받아
    모델을 돌리고 pid, label, pred, proba, embs 컬럼이 있는 DataFrame 반환.
    """
    model.eval()

    pid_all, domain_all, label_all = [], [], []
    pred_all, proba_all, embs_all = [], [], []

    pbar = tqdm(loader, desc=f"[Evaluating {data_type}]", unit="batch")

    with torch.no_grad():  # <-- 요청한 no_grad 컨텍스트
        for times, varis, values, pad_mask, _, outcomes, ids, statics, domain in pbar:
            # 모델 입력만 GPU로
            times   = times.to(device, non_blocking=True)
            varis   = varis.to(device, non_blocking=True)
            values  = values.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            pad_mask= pad_mask.to(device, non_blocking=True)

            out = model(times, varis, values, statics, padding_mask=pad_mask)

            # 배치 단위로 한 번에 CPU로 내리기
            proba = torch.sigmoid(out['pred']).to('cpu', non_blocking=True).numpy()   # (B,)
            preds = (proba >= 0.5).astype(np.int32)                                   # (B,)
            trues = outcomes.cpu().numpy() if torch.is_tensor(outcomes) else np.asarray(outcomes)
            embs_np = out['embs'].to('cpu', non_blocking=True).numpy()                # (B, D)

            # 메타데이터 일괄 확장
            pid_batch = ids.cpu().tolist() if torch.is_tensor(ids) else list(ids)
            domain_batch = domain.cpu().tolist() if torch.is_tensor(domain) else list(domain)

            pid_all.extend(pid_batch)
            domain_all.extend([int(d) for d in domain_batch])
            label_all.extend([int(t) for t in trues])
            pred_all.extend(preds.tolist())
            proba_all.extend(proba.tolist())
            # 각 행의 임베딩 배열 저장 (원 코드 호환)
            embs_all.extend([e.copy() for e in embs_np])

    df = pd.DataFrame({
        'pid':    pid_all,
        'domain': domain_all,
        'label':  label_all,
        'pred':   pred_all,
        'proba':  proba_all,
    })
    df['embs'] = embs_all
    return df

class SubsetVarDataset(Dataset):
    """
    STraTSGRDataset을 감싸서, __getitem__에서 
    times/varis/values 등을 var_subset에 해당하는 이벤트만 남겨줍니다.
    """
    def __init__(self,
                 base_ds,
                 var_subset):
        self.base = base_ds
        self.var_subset = set(var_subset)

    def __len__(self):
        return len(self.base)  

    def get_event_len(self, idx: int) -> int:
        return self.base.ev_lens[idx]

    def __getitem__(self, idx):
        # 원본 dataset에서 꺼내오기
        times, varis, values, pad_mask, pre_mask, yi, pid, Si, domain = self.base[idx]

        # varis in var_subset 인 이벤트만 골라내기
        subset_vals = torch.tensor(list(self.var_subset), device=varis.device)
        keep = torch.isin(varis, subset_vals)   # [n_obs] bool mask
        times   = times[keep]
        varis   = varis[keep]
        values  = values[keep]
        pad_mask= pad_mask[keep]
        pre_mask= pre_mask[keep]

        # label, pid, Si, domain 은 그대로
        return times, varis, values, pad_mask, pre_mask, yi, pid, Si, domain

def get_embs_STraTSGR(data, model, device, batch_size, name, data_type, metric = False, wandb=None):
    try:
        pad_collate_dom = FeatureCollator(data['shape']['F'])
    except:
        pad_collate_dom = FeatureCollator(data['source']['shape']['F'])

    # 1) Dataset / DataLoader 생성

    print(f"😇 Getting embeddings... for batch size {batch_size} and data {data_type}")
    ds = STraTSDatasetGR(
        data['source'][f'P{data_type}'], data['source'][f'P{data_type}_time'],
        data['source'][f'P{data_type}_static'], data['source'][f'y{data_type}'], data['source'][f'{data_type}_ids'], domain=0
    )
    print("  ✅ Dataset Created")
    sampler = BucketBatchSampler(ds, batch_size=batch_size,
                                drop_last=False, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=pad_collate_dom)    
    print("  ✅ DataLoader Created")

    df = evaluate_split_STraTSGR(model, device, loader, data_type)
    print("    ✅ Split complete")
    emb_dim = df['embs'].iloc[0].shape[0]
    emb_cols = [f'emb_{i}' for i in range(emb_dim)]
    emb_df = pd.DataFrame(df['embs'].tolist(), columns=emb_cols)
    out = pd.concat([df.drop(columns='embs'), emb_df], axis=1)
    if metric==True and data_type=='test':
        try:
            acc = accuracy_score(out['label'], out['pred'])
            auroc = roc_auc_score(out['label'], out['proba'])
            auprc = average_precision_score(out['label'], out['proba'])
            auprc_c = average_precision_calibrated(out['label'], out['proba'])
            print(f"      ✅ [{data_type}] Acc:{acc:.4f}, AUROC:{auroc:.4f}, AUPRC:{auprc:.4f}, AUPRC_c:{auprc_c:.4f}")
            if wandb:
                wandb.log({
                    "source_acc": acc,
                    "source_auroc": auroc,
                    "source_auprc": auprc,
                    "source_auprc_c" : auprc_c
                })
        except:
            print("      ❗Failed to get metrics; Skipping")
    
    out.to_feather(f"./rd_results/predictions_source_{data_type}_{name}.feather")
    print("    ✅ Embeddings saved for source")

    # Target data
    ds = STraTSDatasetGR(
        data['target'][f'P{data_type}'], data['target'][f'P{data_type}_time'],
        data['target'][f'P{data_type}_static'], data['target'][f'y{data_type}'], data['target'][f'{data_type}_ids'],
        domain=1
    )
    print("  ✅ Dataset Created")
    sampler = BucketBatchSampler(ds, batch_size=batch_size,
                                drop_last=False, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=pad_collate_dom)
    torch.cuda.empty_cache() 
    print("  ✅ DataLoader Created")

    df = evaluate_split_STraTSGR(model, device, loader, data_type)
    print("    ✅ Split complete")
    emb_dim = df['embs'].iloc[0].shape[0]
    emb_cols = [f'emb_{i}' for i in range(emb_dim)]
    emb_df = pd.DataFrame(df['embs'].tolist(), columns=emb_cols)
    out_target = pd.concat([df.drop(columns='embs'), emb_df], axis=1)
    if metric==True and data_type=='test':
        try:
            acc = accuracy_score(out_target['label'], out_target['pred'])
            auroc= roc_auc_score(out_target['label'], out_target['proba'])
            auprc= average_precision_score(out_target['label'], out_target['proba'])
            auprc_c = average_precision_calibrated(out_target['label'], out_target['proba'])
            print(f"      ✅ [{data_type}] Acc:{acc:.4f}, AUROC:{auroc:.4f}, AUPRC:{auprc:.4f}, AUPRC_c:{auprc_c:.4f}")
            if wandb:
                wandb.log({
                    "target_acc": acc,
                    "target_auroc": auroc,
                    "target_auprc": auprc,
                    "target_auprc_c" : auprc_c
                })
        except:
            print("      ❗Failed to get metrics; Skipping")
    
    out_target.to_feather(f"./rd_results/predictions_target_{data_type}_{name}.feather")
    print("    ✅ Embeddings saved")
    
    print(f"✅ Job finished for batch size {batch_size} and data {data_type}")

    return out, out_target



def get_embs_STraTSGR_varwise(data, model, var_dict, device, batch_size, name, data_type):
    # 1) Dataset / DataLoader 생성
    try:
        pad_collate_dom = FeatureCollator(data['shape']['F'])
    except:
        pad_collate_dom = FeatureCollator(data['source']['shape']['F'])

    print(f"😇 Getting embeddings... for batch size {batch_size} and data {data_type}")
    source_df = pd.DataFrame(columns=['pid'])
    target_df = pd.DataFrame(columns=['pid'])
    count=0
    for var_group in var_dict.keys():
        ds = STraTSDatasetGR(
            data['source'][f'P{data_type}'], data['source'][f'P{data_type}_time'],
            data['source'][f'P{data_type}_static'], data['source'][f'y{data_type}'], data['source'][f'{data_type}_ids'],
            domain=0
        )
        print("  ✅ Varwise Dataset Created")
        sub_ds = SubsetVarDataset(ds, var_dict[var_group])

        sampler = BucketBatchSampler(sub_ds, batch_size=batch_size,
                                    drop_last=False, shuffle=False)
        loader = DataLoader(sub_ds, batch_sampler=sampler, collate_fn=pad_collate_dom, pin_memory=True)    
        print("  ✅ DataLoader Created")

        df = evaluate_split_STraTSGR(model, device, loader, data_type)
        print("    ✅ Split complete")
        emb_dim = df['embs'].iloc[0].shape[0]
        emb_cols = [f'emb_{i}' for i in range(emb_dim)]
        emb_df = pd.DataFrame(df['embs'].tolist(), columns=emb_cols)
        out = pd.concat([df.drop(columns='embs'), emb_df], axis=1)
        source_df = pd.merge(source_df, out, on=['pid'], how='outer', suffixes=('', f'_{var_group}'))
        
        print("    ✅ Embeddings saved for source")

        # Target data
        ds = STraTSDatasetGR(
            data['target'][f'P{data_type}'], data['target'][f'P{data_type}_time'],
            data['target'][f'P{data_type}_static'], data['target'][f'y{data_type}'], data['target'][f'{data_type}_ids'],
            domain=1
        )
        print("  ✅ Varwise Dataset Created")
        sub_ds = SubsetVarDataset(ds, var_dict[var_group])

        sampler = BucketBatchSampler(sub_ds, batch_size=batch_size,
                                    drop_last=False, shuffle=False)
        loader = DataLoader(sub_ds, batch_sampler=sampler, collate_fn=pad_collate_dom, pin_memory=True)    
        print("  ✅ DataLoader Created")

        df = evaluate_split_STraTSGR(model, device, loader, data_type)
        print("    ✅ Split complete")
        emb_dim = df['embs'].iloc[0].shape[0]
        emb_cols = [f'emb_{i}' for i in range(emb_dim)]
        emb_df = pd.DataFrame(df['embs'].tolist(), columns=emb_cols)
        out = pd.concat([df.drop(columns='embs'), emb_df], axis=1)
        target_df = pd.merge(target_df, out, on=['pid'], how='outer', suffixes=('', f'_{var_group}'))
        
    source_df.to_feather(f"./rd_results/predictions_source_{data_type}_{name}_varwise.feather")
    target_df.to_feather(f"./rd_results/predictions_target_{data_type}_{name}_varwise.feather")
    print("    ✅ Embeddings saved")
    
    print(f"✅ Job finished for batch size {batch_size} and data {data_type}")

def collect_gating_view(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
) -> pd.DataFrame:
    """
    DataLoader에서 첫/중간/마지막 배치만 골라
    model.check_padding(times, varis, values, padding_mask)를 실행.
    각 토큰 단위로 pid, times, varis, values, mask(redundant) 컬럼의 DataFrame 반환.
    
    DataLoader batch format:
      (times, varis, values, pad_mask, pretrain_mask, outcomes, ids, statics, domain)
    """

    model.eval()
    n_batches = len(loader)
    if n_batches == 0:
        return pd.DataFrame(columns=["pid", "times", "varis", "values", "mask"])

    # 선택할 배치 인덱스 집합 (중복 방지)
    if n_batches == 1:
        sel_idx: List[int] = [0]
    elif n_batches == 2:
        sel_idx = [0, 1]
    else:
        mid = n_batches // 2
        sel_idx = sorted(set([0, mid, n_batches - 1]))
    rows = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx not in sel_idx:
                continue

            # Unpack: (Time, Variable, Value, Padding mask, Pretrain mask, Outcome, ID, Static)
            times, varis, values, pad_mask, pretrain_mask, outcomes, ids, statics, domain = batch

            # move to device
            times   = times.to(device, non_blocking=True)
            varis   = varis.to(device, non_blocking=True)
            values  = values.to(device, non_blocking=True)
            pad_mask= pad_mask.to(device, non_blocking=True)
            # pretrain_mask, outcomes, ids는 이 함수에선 로직상 사용 X (ids만 CPU로 씀)

            # check_padding 호출 (redundant_mask 계산)
            out = model.check_padding(times, varis, values, pad_mask)
            # out: {'times','varis','values','mask'}

            # CPU로 꺼내기
            times_np = out['times'].cpu().numpy()
            varis_np = out['varis'].cpu().numpy()
            values_np = out['values'].cpu().numpy()
            mask_np = out['mask'].cpu().numpy().astype(bool)

            B, L = times_np.shape
            # 배치-토큰 단위로 평탄화하여 행 추가
            for b in range(B):
                pid = ids[b]
                for t in range(L):
                    rows.append({
                        "pid":   pid,
                        "times": float(times_np[b, t]),
                        "varis": int(varis_np[b, t]),
                        "values": float(values_np[b, t]),
                        "mask":  bool(mask_np[b, t]),   # True면 하드게이팅(패딩 취급)
                    })

    return pd.DataFrame(rows)

def get_masks_Surprise(data, model, device, batch_size, name):
    # 1) Dataset / DataLoader 생성
    try:
        pad_collate_dom = FeatureCollator(data['shape']['F'])
    except:
        pad_collate_dom = FeatureCollator(data['source']['shape']['F'])


    print(f"😇 Getting masks... for batch size {batch_size}")
    ds = STraTSDatasetGR(
        data['source'][f'Ptest'], data['source'][f'Ptest_time'],
        data['source'][f'Ptest_static'], data['source'][f'ytest'], data['source'][f'test_ids'], domain=0
    )
    print("  ✅ Dataset Created")
    sampler = BucketBatchSampler(ds, batch_size=batch_size,
                                drop_last=False, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=pad_collate_dom, pin_memory=True)    
    print("  ✅ DataLoader Created")

    df = collect_gating_view(model, device, loader)
    
    df.to_feather(f"./rd_results/masks_source_{name}.feather")
    print("    ✅ Masks saved for source")

    # Target data
    ds = STraTSDatasetGR(
        data['target'][f'Ptest'], data['target'][f'Ptest_time'],
        data['target'][f'Ptest_static'], data['target'][f'ytest'], data['target'][f'test_ids'],
        domain=1
    )
    print("  ✅ Dataset Created")
    sampler = BucketBatchSampler(ds, batch_size=batch_size,
                                drop_last=False, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=pad_collate_dom, pin_memory=True) 
    print("  ✅ DataLoader Created")

    df = collect_gating_view(model, device, loader)
    
    df.to_feather(f"./rd_results/masks_target_{name}.feather")
    print("    ✅ Masks saved for target")
    
    print(f"✅ Surprise masks per token retrieved")
