import pandas as pd
import numpy as np
import datetime
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

def collate_padding(batch, emb_idx):
    """
    batch: list of (hadm_id, times, varis, values, saps, sofa, death, mask_t)
    """
    (hadm_id_list, times_list, varis_list, values_list, 
     query_time_list, death_list, mask_list) = zip(*batch)

    # pad
    times = pad_sequence(times_list, batch_first=True, padding_value=0.0)
    varis = pad_sequence(varis_list, batch_first=True, padding_value=emb_idx)
    values = pad_sequence(values_list, batch_first=True, padding_value=0.0)
    masks = pad_sequence(mask_list, batch_first=True, padding_value=False)  # bool

    # padding_mask
    padding_mask = (varis != emb_idx).float()

    hadm_id = torch.stack(hadm_id_list)
    query_time = torch.stack(query_time_list)
    death = torch.stack(death_list)

    return hadm_id, times, varis, values, query_time, death, masks, padding_mask

class BucketBatchSampler(Sampler):
    """
    Dataset의 샘플들을 시퀀스 길이 기준으로 정렬한 후, 
    batch_size 개씩 묶어 배치별로 길이가 비슷하도록 하는 배치 샘플러.
    
    drop_last가 True이면 마지막 배치가 batch_size보다 작을 경우 버립니다.
    """
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # dataset.samples에서 각 샘플의 시퀀스 길이는 sample[2] (obs_window_df) 의 row 수라고 가정
        # 예: len(sample[2]) 를 사용
        self.indices = list(range(len(dataset)))
        self.indices.sort(key=lambda idx: len(dataset.samples[idx][2]))
    
    def __iter__(self):
        # 정렬된 인덱스를 배치 단위로 묶습니다.
        batches = []
        for i in range(0, len(self.indices), self.batch_size):
            batch = self.indices[i:i+self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        # 배치의 순서를 무작위로 섞어 전체 에포크의 순서를 섞습니다.
        np.random.shuffle(batches)
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return int(np.ceil(len(self.indices) / self.batch_size))

class MakeLoaders:
    """
    run_all() -> (scalers_dict, clip_bounds, train_loader, valid_loader, test_loader, emb_idx)
    Takes DFs and OutcomeDFs, and
    1) Detect binary variables
    2) Trim
    3) Scale
    4) Create Pytorch Dataset (should be configured for each inherited class)
    5) Create PyTorch DataLoader

    - Initialization saves "df_train, df_valid, df_test, outcome_train, outcome_valid, outcome_test"
    - run_all() -> (scalers_dict, clip_bounds, train_loader, valid_loader, test_loader, emb_idx)
    - Setting "trim_percentile" out of range will skip trimming
    - Setting "scale" to False will skip scaling
    """

    def __init__(self,
                 df_train: pd.DataFrame,
                 df_valid: pd.DataFrame,
                 df_test:  pd.DataFrame,
                 outcome_train: pd.DataFrame,
                 outcome_valid: pd.DataFrame,
                 outcome_test:  pd.DataFrame,
                 obs_window=1440,
                 batch_size=32,
                 mask_length=600,
                 mask_segment_count=4,
                 trim_percentile=0.01,
                 scale=True,
                 step=60,
                 min_window=10,
                 emb_idx=None,
                 upsample=False,
                 bucket_sampler=True):
        """
        Takes data, outcome, parameters and returns
        -> [scalers_dict, train_loader, valid_loader, test_loader, emb_idx]
        """
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test  = df_test
        self.outcome_train = outcome_train
        self.outcome_valid = outcome_valid
        self.outcome_test  = outcome_test

        self.obs_window = obs_window
        self.batch_size = batch_size
        self.mask_length = mask_length
        self.mask_segment_count = mask_segment_count
        self.trim_percentile = trim_percentile
        self.scale = scale
        self.step = step
        self.min_window = min_window

        self.binary_itemids = set()
        self.scalers_dict = {}
        self.clip_bounds = {}
        self.emb_idx = emb_idx
        self.upsample = upsample
        self.bucket_sampler = bucket_sampler

    def detect_binary_items(self):
        """
        이진 변수 식별
        """
        print("[Binary] Detecting binary itemids ...")
        df_train = self.df_train
        binary_itemids = set()

        for item_id, grp in df_train.groupby('itemid'):
            uniq_vals = set(grp['value'].dropna().unique())
            # subset of {0,1}?
            if uniq_vals.issubset({0,1}):
                binary_itemids.add(item_id)

        self.binary_itemids = binary_itemids
        print(f"  found {len(binary_itemids)} binary itemids => {binary_itemids}")

    def do_trimming(self, clip_bounds=None):
        """
        Trimming percentile
        - clip_bounds: if given, use it directly
          else compute from self.df_train
        returns the final clip_bounds
        """


        # (1) If no external clip_bounds is provided, build from train
        if clip_bounds is None:
            print(f"[Trimming] No clip bounds given")
            print(f"[Trimming] Generating clip bounds with percentile {self.trim_percentile}")
            clip_bounds = {}
            for item_id, grp in self.df_train.groupby('itemid'):
                if item_id in self.binary_itemids:
                    continue
                lower = grp['value'].quantile(self.trim_percentile)
                upper = grp['value'].quantile(1 - self.trim_percentile)
                clip_bounds[item_id] = (lower, upper)
        
        else:
            print(f"[Trimming] Using predefined clip bounds")


        # (2) apply to train/valid/test
        def trim_df(df, clip_dict):
            new_list = []
            for _item_id, subgrp in df.groupby('itemid'):
                if _item_id in self.binary_itemids:
                    new_list.append(subgrp)
                    continue
                if _item_id in clip_dict:
                    l, u = clip_dict[_item_id]
                    # clipped = subgrp.loc[(subgrp['value']>=l)&(subgrp['value']<=u)]
                    clipped = subgrp.loc[(subgrp['value']>l)&(subgrp['value']<u)] # 수정함!!
                    new_list.append(clipped)
                else:
                    new_list.append(subgrp)
            if len(new_list)>0:
                return pd.concat(new_list, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns)

        self.df_train = trim_df(self.df_train, clip_bounds)
        self.df_valid = trim_df(self.df_valid, clip_bounds)
        self.df_test  = trim_df(self.df_test, clip_bounds)

        print(f"  trimmed => train={len(self.df_train)}, valid={len(self.df_valid)}, test={len(self.df_test)}")

        # 추가: 각 DataFrame에서 hadm_id별 row 수 통계 확인
        def print_hadm_row_stats(df, name):
            if df.empty:
                print(f"  [{name}] is empty, skipping stats.")
                return
            group_counts = df.groupby('hadm_id').size()  # Series: hadm_id -> row_count
            mean_ = group_counts.mean()
            std_ = group_counts.std()
            min_ = group_counts.min()
            max_ = group_counts.max()
            median_ = group_counts.median()
            print(f"  [{name}] hadm_id row count => "
                  f"mean={mean_:.2f}, std={std_:.2f}, "
                  f"min={min_}, max={max_}, median={median_}")

        print_hadm_row_stats(self.df_train, 'train')
        print_hadm_row_stats(self.df_valid, 'valid')
        print_hadm_row_stats(self.df_test,  'test')

        return clip_bounds

    def do_scaling(self, scalers_dict=None):
        """
        Scaling (StandardScaler)
        - scalers_dict: if given, apply it directly
          else compute from self.df_train
        returns the final scalers_dict
        """
        if not self.scale:
            print("[Scale] Skipping scaling.")
            return scalers_dict if scalers_dict else {}

        print("[Scale] Scaling type : StandardScaler")
        if scalers_dict is None:
            scalers_dict = {}
            # train fit + transform
            for item_id, grp in self.df_train.groupby('itemid'):
                if item_id in self.binary_itemids:
                    continue
                scaler = StandardScaler()
                vals = grp['value'].values.reshape(-1,1)
                scaler.fit(vals)
                scaled = scaler.transform(vals)
                self.df_train.loc[grp.index, 'value'] = scaled
                scalers_dict[item_id] = scaler
        else:
            # if scalers_dict is given => just transform train
            for item_id, grp in self.df_train.groupby('itemid'):
                if item_id in self.binary_itemids:
                    continue
                if item_id in scalers_dict:
                    sc = scalers_dict[item_id]
                    scaled = sc.transform(grp['value'].values.reshape(-1,1))
                    self.df_train.loc[grp.index, 'value'] = scaled

        # valid/test transform
        for item_id, grpv in self.df_valid.groupby('itemid'):
            if item_id in scalers_dict:
                sc = scalers_dict[item_id]
                transformed = sc.transform(grpv['value'].values.reshape(-1,1))
                self.df_valid.loc[grpv.index, 'value'] = transformed

        for item_id, grpt in self.df_test.groupby('itemid'):
            if item_id in scalers_dict:
                sc = scalers_dict[item_id]
                transformed = sc.transform(grpt['value'].values.reshape(-1,1))
                self.df_test.loc[grpt.index, 'value'] = transformed

        print("  Scaling done.")
        return scalers_dict

    def create_datasets(self):
        """
        Creates Datasets from data, must be configured when inherited
        """
        pass

    def create_loaders(self):
        """
        Dataset => DataLoader, emb_idx
        """
        train_ds, valid_ds, test_ds = self.create_datasets()
        if self.emb_idx == None:
            emb_idx = self.df_train['itemid'].nunique()  # or union of train+valid+test
            self.emb_idx = emb_idx


        if self.bucket_sampler is False:
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=lambda b: collate_padding(b, emb_idx=self.emb_idx)
            )
            valid_loader = DataLoader(
                valid_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_padding(b, emb_idx=self.emb_idx)
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_padding(b, emb_idx=self.emb_idx)
            )
        else:
            # 여기서 BalancedBatchSampler 대신 BucketBatchSampler를 사용하여 배치 내 길이 차이를 줄임.
            train_sampler = BucketBatchSampler(train_ds, batch_size=self.batch_size, drop_last=False)
            valid_sampler = BucketBatchSampler(valid_ds, batch_size=self.batch_size, drop_last=False)
            test_sampler  = BucketBatchSampler(test_ds, batch_size=self.batch_size, drop_last=False)
            
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                collate_fn=lambda b: collate_padding(b, emb_idx=self.emb_idx)
            )
            valid_loader = DataLoader(
                valid_ds,
                batch_sampler=valid_sampler,
                collate_fn=lambda b: collate_padding(b, emb_idx=self.emb_idx)
            )
            test_loader = DataLoader(
                test_ds,
                batch_sampler=test_sampler,
                collate_fn=lambda b: collate_padding(b, emb_idx=self.emb_idx)
    )

        return train_loader, valid_loader, test_loader, self.emb_idx

    def run_all(self, clip_bounds=None, scalers_dict=None):
        """
        Run all the modules
        1) detect_binary_items
        2) do_trimming => returns final clip_bounds
        3) do_scaling => returns final scalers_dict
        4) create_loaders
        returns: (scalers_dict, clip_bounds, train_loader, valid_loader, test_loader, emb_idx)
        """
        print("[MakeLoaders] Beginning job... 😤")
        self.detect_binary_items()
        final_clip_bounds = self.do_trimming(clip_bounds=clip_bounds)
        final_scalers_dict = self.do_scaling(scalers_dict=scalers_dict)
        self.train_loader, self.valid_loader, self.test_loader, emb_idx = self.create_loaders()

        print("[MakeLoaders] Job complete! 😇")
        return final_scalers_dict, final_clip_bounds, self.train_loader, self.valid_loader, self.test_loader, emb_idx



class TimeSeriesDatasetFixedTimeMask(Dataset):
    def __init__(self, df, outcome_df, observation_window, step, mode='train',
                 mask_length=600):
        """
        mask_length: 무작위로 마스킹할 연속된 시간 구간 길이(분 단위)
        """
        self.samples = []
        self.observation_window = observation_window
        self.step = step
        self.mode = mode
        self.mask_length = mask_length

        for hadm_id, group in df.groupby('hadm_id'):
            outcome_row = outcome_df.loc[outcome_df['hadm_id'] == hadm_id, ['SAPS-I', 'SOFA', 'death']]
            saps, sofa, death = outcome_row.values[0]

            max_offset = group['offset'].max()
            
            # 여기서는 query_times를 [2880]로 고정 (예: 48*60분)
            # 필요에 따라 다르게 설정할 수 있음
            query_times = [2880]

            for query_time in query_times:
                obs_window = group[
                    (group['offset'] > query_time - self.observation_window) &
                    (group['offset'] <= query_time)
                ].copy()

                if not obs_window.empty and obs_window.shape[0] > 9:
                    # offset을 0 ~ self.observation_window 범위로 재조정
                    obs_window['offset'] = obs_window['offset'] - (query_time - self.observation_window)

                    # ------------------------------
                    # 1) 무작위 마스킹 구간 생성
                    # ------------------------------
                    # 예: 0 <= start <= observation_window - mask_length
                    possible_start = self.observation_window - self.mask_length
                    if possible_start < 0:
                        # 혹시 observation_window < mask_length인 경우 예외처리
                        possible_start = 0
                    
                    start = np.random.randint(0, possible_start + 1)
                    end = start + self.mask_length

                    # obs_window의 offset 값들에 대해
                    offset_arr = obs_window['offset'].values
                    # 해당 범위 내에 있으면 True, 아니면 False
                    mask_array = (offset_arr >= start) & (offset_arr < end)
                    
                    # sample에 mask_array까지 포함시켜 저장
                    self.samples.append(
                        (hadm_id, query_time, obs_window, saps, sofa, death, mask_array)
                    )

        print(f"[TimeSeriesDataset] Created {len(self.samples)} samples.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hadm_id, query_time, obs_window, saps, sofa, death, mask_array = self.samples[idx]

        times = torch.tensor(obs_window['offset'].values / self.observation_window, dtype=torch.float32)
        varis = torch.tensor(obs_window['itemid'].values, dtype=torch.long)
        values = torch.tensor(obs_window['value'].values, dtype=torch.float32)

        hadm_id_t = torch.tensor(hadm_id, dtype=torch.long)
        saps_t = torch.tensor(saps, dtype=torch.float32)
        sofa_t = torch.tensor(sofa, dtype=torch.float32)
        death_t = torch.tensor(death, dtype=torch.long)

        # mask_array는 bool 타입
        mask_t = torch.tensor(mask_array, dtype=torch.bool)  # shape = (sequence_length,)

        return hadm_id_t, times, varis, values, saps_t, sofa_t, death_t, mask_t


class PC2012TimeSeriesDataset(Dataset): # Fixed outcome (window length = sequence length)
    def __init__(self, df, outcome_df, observation_window, mode='train',
                 mask_length=720, mask_segment_count=4):
        """
        mask_length: 각 itemid별로 마스킹할 전체 길이(분)
        mask_segment_count: 마스킹 구간 개수
        => 각 구간 길이는 mask_length // mask_segment_count
        """
        self.samples = []
        self.observation_window = observation_window
        self.mode = mode
        self.mask_length = mask_length
        self.mask_segment_count = mask_segment_count

        for hadm_id, group in df.groupby('hadm_id'):
            outcome_row = outcome_df.loc[outcome_df['hadm_id'] == hadm_id, ['SAPS-I', 'SOFA', 'death']]
            if len(outcome_row) == 0:
                # outcome이 없는 경우 스킵
                continue
            saps, sofa, death = outcome_row.values[0]

            # 예: 단 하나의 query_time=2880 (48시간)
            query_times = [2880]

            for query_time in query_times:
                obs_window = group[
                    (group['offset'] > query_time - self.observation_window) &
                    (group['offset'] <= query_time)
                ].copy()

                if not obs_window.empty and obs_window.shape[0] > 9:
                    # 0~observation_window 범위로 offset 재조정
                    obs_window['offset'] = obs_window['offset'] - (query_time - self.observation_window)

                    # -------------------------
                    # (A) 아이템별로 마스킹 구간 생성
                    # -------------------------
                    # 1) 유니크 아이템 목록
                    unique_itemids = obs_window['itemid'].unique()

                    # 2) itemid -> 구간 목록 매핑
                    itemid2intervals = {}
                    segment_len = self.mask_length // self.mask_segment_count
                    possible_start = self.observation_window - segment_len
                    if possible_start < 0:
                        # 예외처리
                        possible_start = 0

                    for it in unique_itemids:
                        intervals = []
                        for seg_idx in range(self.mask_segment_count):
                            start = np.random.randint(0, possible_start+1)
                            end = start + segment_len
                            intervals.append((start, end))
                        itemid2intervals[it] = intervals

                    # -------------------------
                    # (B) obs_window 각 row별 mask 결정
                    # -------------------------
                    obs_window = obs_window.reset_index(drop=True)  # 인덱스 리셋
                    mask_array = np.zeros(obs_window.shape[0], dtype=bool)

                    for i, row in obs_window.iterrows():
                        offset_val = row['offset']   
                        itemid_val = row['itemid'] 
                        intervals = itemid2intervals[itemid_val]  # 해당 itemid의 구간 리스트

                        for (st, en) in intervals:
                            if st <= offset_val < en:
                                mask_array[i] = True
                                break  # 한 구간 충족 시 즉시 탈출

                    # 넘파이 배열 인덱스 정렬 순서 주의:
                    # obs_window.iterrows() 순서 vs obs_window.index
                    # row별 i가 obs_window.index일 수 있으므로, index -> position
                    # 간단히 reset_index 써서 0..n-1로 만드는 방법:
                    obs_window = obs_window.reset_index(drop=True)
                    # 이제 obs_window와 mask_array가 positional indexing으로 align

                    # sample 저장
                    self.samples.append((
                        hadm_id, query_time, obs_window, saps, sofa, death, mask_array
                    ))

        print(f"[TimeSeriesDataset] Created {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hadm_id, query_time, obs_window, saps, sofa, death, mask_array = self.samples[idx]

        # offset, itemid, value
        # offset은 0~1 범위로 스케일
        times = torch.tensor(obs_window['offset'].values / self.observation_window, dtype=torch.float32)
        varis = torch.tensor(obs_window['itemid'].values, dtype=torch.long)
        values = torch.tensor(obs_window['value'].values, dtype=torch.float32)

        hadm_id_t = torch.tensor(hadm_id, dtype=torch.long)
        saps_t = torch.tensor(saps, dtype=torch.float32)
        sofa_t = torch.tensor(sofa, dtype=torch.float32)
        death_t = torch.tensor(death, dtype=torch.long)

        # mask_array: (seq_len,) bool
        mask_t = torch.tensor(mask_array, dtype=torch.bool)

        return hadm_id_t, times, varis, values, saps_t, sofa_t, death_t, mask_t
    

class TimeSeriesDataset(Dataset): # Dynamic outcome (window length < sequence length)
    def __init__(self,
                 df, 
                 outcome_df, 
                 observation_window, 
                 binary_ids,
                 mask_length=360, 
                 mask_segment_count=4, 
                 query_type='dynamic', 
                 step=1440, 
                 min_window=10,
                 upsample=False,
                 upsample_seed=9871
                 ):
        self.samples = []
        self.observation_window = observation_window
        self.mask_length = mask_length
        self.mask_segment_count = mask_segment_count

        seq_lens = []  # 각 샘플의 sequence 길이를 저장할 리스트
        stay_length =[]

        for hadm_id, group in df.groupby('hadm_id'):
            # Fix here
            death_offset_series = outcome_df.loc[outcome_df['hadm_id'] == hadm_id, 'death_offset']
            death_offset = death_offset_series.iloc[0]
            max_offset = max(group['offset'])

            if max_offset < observation_window:
                continue

            # query_time 생성 로직
            if query_type == 'dynamic':
                query_times = [i for i in range(1440, max(group['offset']), step)]
            elif query_type == 'initial':
                query_times = [observation_window]
            else:
                query_times = [observation_window]

            for query_time in query_times:
                obs_window = group[
                    (group['offset'] > (query_time - self.observation_window)) &
                    (group['offset'] <= query_time)
                ].copy()

                death = 0
                if pd.isna(death_offset):
                    death = 0
                else:
                    if query_type == 'dynamic':
                        if death_offset - 1440 <= query_time:
                            death = 1
                    elif query_type == 'initial':
                        death = 1


                window_size = obs_window.shape[0] >= min_window

                if obs_window.empty or window_size==False:
                    continue
                # offset 재조정
                obs_window['offset'] = obs_window['offset'] - (query_time - self.observation_window)

                # 아이템별 마스킹 구간
                unique_itemids = obs_window['itemid'].unique()
                itemid2intervals = {}
                segment_len = self.mask_length // self.mask_segment_count
                possible_start = self.observation_window - segment_len
                if possible_start < 0:
                    possible_start = 0
                
                for it in unique_itemids:
                    if it in binary_ids:
                        continue
                    intervals = []
                    for seg_idx in range(self.mask_segment_count):
                        start = np.random.randint(0, possible_start+1)
                        end = start + segment_len
                        intervals.append((start, end))
                    itemid2intervals[it] = intervals

                # row별 마스크
                obs_window = obs_window.reset_index(drop=True)
                mask_array = np.zeros(obs_window.shape[0], dtype=bool)

                itemids_np = obs_window['itemid'].values
                offsets_np = obs_window['offset'].values

                for it in unique_itemids:
                    if it in binary_ids:
                        continue                        
                    intervals = itemid2intervals[it]
                    # 이 아이템의 행 인덱스
                    idxs = np.where(itemids_np == it)[0]
                    if len(idxs)==0:
                        continue

                    # 해당 offset
                    off_for_it = offsets_np[idxs]
                    combined_mask = np.zeros(len(idxs), dtype=bool)

                    for (st, en) in intervals:
                        combined_mask |= ((off_for_it >= st) & (off_for_it < en))

                    mask_array[idxs] = combined_mask

                # 샘플 저장
                self.samples.append((hadm_id, query_time, obs_window, death, mask_array))
                seq_lens.append(obs_window.shape[0])
            stay_length.append(max_offset)
        
        print(f"[TimeSeriesDatasetMortality_Outcome] Total {len(stay_length)} stays used")

        # Get stats of each stay length
        def get_stats(series: pd.Series):
            if series.empty:
                # 비어있으면 NaN으로 채우기
                return [0, np.nan, np.nan, np.nan, np.nan, np.nan]
            return [
                series.count(),       # count
                series.mean(),        # mean
                series.min(),         # min
                series.max(),         # max
                series.std(),         # std
                series.median()       # median
            ]

        stays_stats = get_stats(pd.Series(stay_length))

        stat_names = ['count','mean','min','max','std','median']
        df_stats = pd.DataFrame({
            'Statistic': stat_names,
            'Stay': stays_stats,

        })

        df_stats.set_index('Statistic', inplace=True)

        print("=== Stay-wise Statistics Table ===")
        print(df_stats)

        # Dataset 통계 정보 출력
        total_samples = len(self.samples)
        death_1_count = sum(1 for s in self.samples if s[3] == 1)
        print(f"[TimeSeriesDataset] Created {total_samples} samples.  Death=1: {death_1_count} samples.")

        if total_samples > 0:
            seq_lens = np.array(seq_lens)
            mean_ = seq_lens.mean()
            std_ = seq_lens.std()
            min_ = seq_lens.min()
            max_ = seq_lens.max()
            median_ = np.median(seq_lens)
            print(f"[TimeSeriesDataset] Sequence length stats: mean={mean_:.2f}, std={std_:.2f}, "
                  f"min={min_}, max={max_}, median={median_}")
        # ------------- Upsampling -------------
        if upsample:
            # 별도로 양성(CF=1)와 음성(CF=0) 샘플을 분리
            pos_samples = [s for s in self.samples if s[3] == 1]
            neg_samples = [s for s in self.samples if s[3] == 0]
            n_pos = len(pos_samples)
            n_neg = len(neg_samples)
            # 더 적은 샘플인 클래스의 수를 늘림 (중복 선택 허용)
            if n_pos > 0 and n_neg > 0:
                if n_pos > n_neg:
                    additional = random.choices(neg_samples, k=(n_pos - n_neg))
                    self.samples.extend(additional)
                elif n_neg > n_pos:
                    additional = random.choices(pos_samples, k=(n_neg - n_pos))
                    self.samples.extend(additional)
            # 재현성을 위해 seed 설정 후 전체 샘플 섞기
            random.seed(upsample_seed)
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hadm_id, query_time, obs_window, death, mask_array = self.samples[idx]

        # offset, itemid, value
        # offset은 0~1 범위로 스케일
        times = torch.tensor(obs_window['offset'].values / self.observation_window, dtype=torch.float32)
        varis = torch.tensor(obs_window['itemid'].values, dtype=torch.long)
        values = torch.tensor(obs_window['value'].values, dtype=torch.float32)

        query_time = torch.tensor(query_time, dtype=torch.long)
        hadm_id_t = torch.tensor(hadm_id, dtype=torch.long)
        death_t = torch.tensor(death, dtype=torch.long)

        # mask_array: (seq_len,) bool
        mask_t = torch.tensor(mask_array, dtype=torch.bool)

        return hadm_id_t, times, varis, values, query_time, death_t, mask_t
    
class TimeSeriesDatasetLoS(Dataset):
    def __init__(self,
                 df, 
                 outcome_df, 
                 observation_window, 
                 binary_ids,
                 mask_length=360, 
                 mask_segment_count=4, 
                 step=1440, 
                 min_window=10,
                 upsample=False,
                 upsample_seed=9871
                 ):
        self.samples = []
        self.observation_window = observation_window
        self.mask_length = mask_length
        self.mask_segment_count = mask_segment_count

        seq_lens = []  # 각 샘플의 sequence 길이를 저장할 리스트
        stay_length =[]

        for hadm_id, group in df.groupby('hadm_id'):
            # Fix here
            los = outcome_df.loc[outcome_df['hadm_id'] == hadm_id, 'los']
            los_value = los.iloc[0]
            max_offset = max(group['offset'])

            if max_offset < observation_window:
                continue

            query_times = [observation_window]

            for query_time in query_times:
                obs_window = group[
                    (group['offset'] > (query_time - self.observation_window)) &
                    (group['offset'] <= query_time)
                ].copy()

                label = 0
                if los_value <= 7:
                    label = 0
                else:
                    label = 1


                window_size = obs_window.shape[0] >= min_window

                if obs_window.empty or window_size==False:
                    continue
                # offset 재조정
                obs_window['offset'] = obs_window['offset'] - (query_time - self.observation_window)

                # 아이템별 마스킹 구간
                unique_itemids = obs_window['itemid'].unique()
                itemid2intervals = {}
                segment_len = self.mask_length // self.mask_segment_count
                possible_start = self.observation_window - segment_len
                if possible_start < 0:
                    possible_start = 0
                
                for it in unique_itemids:
                    if it in binary_ids:
                        continue
                    intervals = []
                    for seg_idx in range(self.mask_segment_count):
                        start = np.random.randint(0, possible_start+1)
                        end = start + segment_len
                        intervals.append((start, end))
                    itemid2intervals[it] = intervals

                # row별 마스크
                obs_window = obs_window.reset_index(drop=True)
                mask_array = np.zeros(obs_window.shape[0], dtype=bool)

                itemids_np = obs_window['itemid'].values
                offsets_np = obs_window['offset'].values

                for it in unique_itemids:
                    if it in binary_ids:
                        continue                        
                    intervals = itemid2intervals[it]
                    # 이 아이템의 행 인덱스
                    idxs = np.where(itemids_np == it)[0]
                    if len(idxs)==0:
                        continue

                    # 해당 offset
                    off_for_it = offsets_np[idxs]
                    combined_mask = np.zeros(len(idxs), dtype=bool)

                    for (st, en) in intervals:
                        combined_mask |= ((off_for_it >= st) & (off_for_it < en))

                    mask_array[idxs] = combined_mask

                # 샘플 저장
                self.samples.append((hadm_id, query_time, obs_window, label, mask_array))
                seq_lens.append(obs_window.shape[0])
            stay_length.append(max_offset)
        
        print(f"[TimeSeriesDatasetLengthOfStay_Outcome] Total {len(stay_length)} stays used")

        # Get stats of each stay length
        def get_stats(series: pd.Series):
            if series.empty:
                # 비어있으면 NaN으로 채우기
                return [0, np.nan, np.nan, np.nan, np.nan, np.nan]
            return [
                series.count(),       # count
                series.mean(),        # mean
                series.min(),         # min
                series.max(),         # max
                series.std(),         # std
                series.median()       # median
            ]

        stays_stats = get_stats(pd.Series(stay_length))

        stat_names = ['count','mean','min','max','std','median']
        df_stats = pd.DataFrame({
            'Statistic': stat_names,
            'Stay': stays_stats,

        })

        df_stats.set_index('Statistic', inplace=True)

        print("=== Stay-wise Statistics Table ===")
        print(df_stats)

        # Dataset 통계 정보 출력
        total_samples = len(self.samples)
        los_over_7_count = sum(1 for s in self.samples if s[3] == 1)
        print(f"[TimeSeriesDataset] Created {total_samples} samples.  Death=1: {los_over_7_count} samples.")

        if total_samples > 0:
            seq_lens = np.array(seq_lens)
            mean_ = seq_lens.mean()
            std_ = seq_lens.std()
            min_ = seq_lens.min()
            max_ = seq_lens.max()
            median_ = np.median(seq_lens)
            print(f"[TimeSeriesDataset] Sequence length stats: mean={mean_:.2f}, std={std_:.2f}, "
                  f"min={min_}, max={max_}, median={median_}")
        # ------------- Upsampling -------------
        if upsample:
            # 별도로 양성(CF=1)와 음성(CF=0) 샘플을 분리
            pos_samples = [s for s in self.samples if s[3] == 1]
            neg_samples = [s for s in self.samples if s[3] == 0]
            n_pos = len(pos_samples)
            n_neg = len(neg_samples)
            # 더 적은 샘플인 클래스의 수를 늘림 (중복 선택 허용)
            if n_pos > 0 and n_neg > 0:
                if n_pos > n_neg:
                    additional = random.choices(neg_samples, k=(n_pos - n_neg))
                    self.samples.extend(additional)
                elif n_neg > n_pos:
                    additional = random.choices(pos_samples, k=(n_neg - n_pos))
                    self.samples.extend(additional)
            # 재현성을 위해 seed 설정 후 전체 샘플 섞기
            random.seed(upsample_seed)
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hadm_id, query_time, obs_window, label, mask_array = self.samples[idx]

        # offset, itemid, value
        # offset은 0~1 범위로 스케일
        times = torch.tensor(obs_window['offset'].values / self.observation_window, dtype=torch.float32)
        varis = torch.tensor(obs_window['itemid'].values, dtype=torch.long)
        values = torch.tensor(obs_window['value'].values, dtype=torch.float32)

        query_time = torch.tensor(query_time, dtype=torch.long)
        hadm_id_t = torch.tensor(hadm_id, dtype=torch.long)
        label_t = torch.tensor(label, dtype=torch.long)

        # mask_array: (seq_len,) bool
        mask_t = torch.tensor(mask_array, dtype=torch.bool)

        return hadm_id_t, times, varis, values, query_time, label_t, mask_t



def pc2012_collate_padding(batch, emb_idx): # For when using three results such as in PC2012 : SAPS, SOFA, DEATH
    # Should fix later to take results 'dynamically'?
    """
    batch: list of (hadm_id, times, varis, values, saps, sofa, death, mask_t)
    """
    (hadm_id_list, times_list, varis_list, values_list, 
     saps_list, sofa_list, death_list, mask_list) = zip(*batch)

    # pad
    times = pad_sequence(times_list, batch_first=True, padding_value=0.0)
    varis = pad_sequence(varis_list, batch_first=True, padding_value=emb_idx)
    values = pad_sequence(values_list, batch_first=True, padding_value=0.0)
    masks = pad_sequence(mask_list, batch_first=True, padding_value=False)  # bool

    # padding_mask
    padding_mask = (varis != emb_idx).float()

    hadm_id = torch.stack(hadm_id_list)
    saps = torch.stack(saps_list)
    sofa = torch.stack(sofa_list)
    death = torch.stack(death_list)

    return hadm_id, times, varis, values, saps, sofa, death, masks, padding_mask

def pc2012_create_loaders(df_train : pd.DataFrame, 
                   df_valid : pd.DataFrame, 
                   df_test : pd.DataFrame, 
                   df_train_outcome : pd.DataFrame,
                   df_valid_outcome : pd.DataFrame, 
                   df_test_outcome : pd.DataFrame,
                   batch_size=32,
                   mask_length=600,
                   mask_segment_count=4
                   ):
    """
    Function to create loaders from train, valid, test dataframes. PC 2012 data has data and outcome linked by 'hadm_id'
    To use on other tasks such as 'Next 24H mortality'; new Dataset should be defined.

    Parameters
    ----------
    df_train, valid, test : pd.DataFrame
        Dataframe of training / validation / test data, columns ['hadm_id', 'offset', 'itemid', 'value']
    df_train / valid / test _outcome : pd.DataFrame
        Dataframe containing outcomes of corresponding data, columns ['hadm_id', 'SAPS-I', 'SOFA', 'death']
    batch_size : int
        Batch size of dataloader
    mask_length : int
        Length of total mask for each feature, should be set with respect to the length of each sample (currently 48*60)
    mask_segment_count : int
        Number of same length segments of mask
    
    Returns
    -------
    scaler_dict : dict
        Dictionary of scalers, itemid as key
    train / valid / test _loader: DataLoader
        Dataloader of each data
    emb_idx : int
        Padding index in 'itemid'

    """    
    scalers_dict = {}  

    for item_id, group in df_train.groupby('itemid'):
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(group['value'].values.reshape(-1, 1))
        
        # group에는 스케일링한 값을 '직접' 넣어봤자 원본은 바뀌지 않음
        # 대신 group.index를 활용해 원본 DF에 반영
        df_train.loc[group.index, 'value'] = scaled_values
        
        scalers_dict[item_id] = scaler

    for item_id, group in df_valid.groupby('itemid'):
        scaler = scalers_dict[item_id]  # 이미 학습된 scaler 사용
        scaled_values = scaler.transform(group['value'].values.reshape(-1, 1))
        df_valid.loc[group.index, 'value'] = scaled_values

    for item_id, group in df_test.groupby('itemid'):
        scaler = scalers_dict[item_id]  # 이미 학습된 scaler 사용
        scaled_values = scaler.transform(group['value'].values.reshape(-1, 1))
        df_test.loc[group.index, 'value'] = scaled_values

    print('Scaling complete!')

    train_dataset = PC2012TimeSeriesDataset(df_train, df_train_outcome, observation_window=48*60, 
                                    mask_length=mask_length, mask_segment_count=mask_segment_count, mode='train')
    valid_dataset = PC2012TimeSeriesDataset(df_valid, df_valid_outcome, observation_window=48*60,
                                    mask_length=mask_length, mask_segment_count=mask_segment_count, mode='test')
    test_dataset = PC2012TimeSeriesDataset(df_test, df_test_outcome, observation_window=48*60,
                                    mask_length=mask_length, mask_segment_count=mask_segment_count, mode='test')

    emb_idx = df_train['itemid'].nunique()
    print(f'Padded values have variable id {emb_idx}')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: pc2012_collate_padding(batch, emb_idx=emb_idx))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: pc2012_collate_padding(batch, emb_idx=emb_idx))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: pc2012_collate_padding(batch, emb_idx=emb_idx))

    return scalers_dict, train_loader, valid_loader, test_loader, emb_idx



class TimeSeriesDatasetCF(Dataset):
    """
    Circulatory Failure 라벨링:
      - 관찰 구간(window)에 이미 CF 있으면 샘플 생성 X
      - window 끝 시점 ~ 8시간 이내 CF가 있으면 label=1, 없으면 0
    """
    def __init__(self,
                 df: pd.DataFrame,
                 outcome_df: pd.DataFrame,
                 binary_ids,
                 obs_window=1440,
                 mask_length=360,
                 mask_segment_count=4,
                 step=60,
                 min_window=10,
                 upsample=False,
                 upsample_seed=9871):
        """
        df: [hadm_id, offset, itemid, value]
        outcome_df: [hadm_id, cf_offset]  (cf_offset=NaN if CF 없음)
        obs_window: 관찰 윈도우(분)
        step: query_time 간격(분)
        mask_length, mask_segment_count: 마스킹
        """
        super().__init__()
        seq_lens = []  # 각 샘플의 sequence 길이를 저장할 리스트

        self.samples = []
        self.observation_window = obs_window
        self.mask_length = mask_length
        self.mask_segment_count = mask_segment_count

        # outcome_df에서 hadm_id->cf_offset 딕셔너리 구성
        # cf_offset_nan이면 CF없음
        hadm2cf_offset = {}
        # for _, row in outcome_df.iterrows():
        #     hid = row['hadm_id']
        #     coff = row['cf_offset']
        #     hadm2cf_offset[hid] = coff
        
        for _, row in outcome_df.iterrows():
            hid = row['hadm_id']
            # aki_offset가 여러 onset의 리스트여야 함; 만약 NaN이면 빈 리스트로 설정
            coffs = row['cf_offset']
            if isinstance(coffs, (list, np.ndarray)):
                # 만약 배열 전체가 NaN이면 빈 리스트로 설정
                if np.all(pd.isna(coffs)):
                    coffs = []
            else:
                if pd.isna(coffs):
                    coffs = []
            hadm2cf_offset[hid] = coffs

        stay_length = []

        # hadm_id별로 group
        for hadm_id, group in df.groupby('hadm_id'):
            if group.empty:
                continue
            max_offset = group['offset'].max()
            cf_onsets = hadm2cf_offset.get(hadm_id, [])
            # flatten: 혹시 리스트 내에 리스트가 있다면
            if cf_onsets and isinstance(cf_onsets[0], list):
                flat = []
                for sub in cf_onsets:
                    flat.extend(sub)
                cf_onsets = flat
            # if group.empty:
            #     continue

            max_offset = group['offset'].max()
            # CF offset
            cf_offset = hadm2cf_offset.get(hadm_id, np.nan)

            # query_time 범위 결정: [24h, max_offset - 4h] etc.
            start_qtime = obs_window
            # end_qtime = max_offset - 240
            # if end_qtime <= start_qtime:
            #     continue

            # query_times = range(start_qtime, int(end_qtime)+1, step)
            query_times = [obs_window]

            for qtime in query_times:
                win_start = qtime - obs_window
                win_end   = qtime

                # 윈도우 내 데이터
                obs_window_df = group[
                    (group['offset']>win_start) & (group['offset']<=win_end)
                ].copy()

                window_size = obs_window_df.shape[0] >= min_window
                obs_window_df = group[(group['offset'] > win_start) & (group['offset'] <= win_end)].copy()
                if obs_window_df.empty or obs_window_df.shape[0] < min_window or max_offset < obs_window:
                    continue
                # (A) 관찰 윈도우 내에 **어떤** AKI onset이 존재하면 sample skip
                # if any((onset > win_start) and (onset <= win_end) for onset in aki_onsets):
                #     continue
                # if any((onset <= win_start) for onset in aki_onsets):
                #     continue
                # (B) Query range 내에 AKI onset이 있으면 label=1, 없으면 0
                cf_label = 0
                if any((onset > win_end) and (onset <= win_end + 480) for onset in cf_onsets):
                    cf_label = 1

                # if obs_window_df.empty or window_size==False or max_offset < obs_window:
                #     continue

                # # (A) 윈도우 내 CF 발생 => skip
                # # cf_offset in (win_start, win_end]
                # # Do not create windows after CF
                # skip_sample = False
                # if not np.isnan(cf_offset):
                #     if (cf_offset > win_start) and (cf_offset <= win_end):
                #         skip_sample = True
                #     if cf_offset < win_start:
                #         skip_sample = True

                # if skip_sample:
                #     continue

                # # (B) 윈도우 끝~8h 내 CF => label=1
                # # cf_offset in (win_end, win_end+480]
                # cf_label = 0
                # if not np.isnan(cf_offset):
                #     if (cf_offset > win_end) and (cf_offset <= win_end+480):
                #         cf_label = 1

                # offset 재조정 => [0, obs_window]
                obs_window_df['offset'] = obs_window_df['offset'] - win_start

                # 마스킹 구간
                seg_len = self.mask_length // self.mask_segment_count
                possible_start = self.observation_window - seg_len
                if possible_start<0:
                    possible_start=0

                itemid2intervals = {}
                unique_itemids = obs_window_df['itemid'].unique()
                for it in unique_itemids:
                    if it in binary_ids:
                        continue
                    intervals=[]
                    for _ in range(self.mask_segment_count):
                        st = np.random.randint(0, possible_start+1)
                        en = st + seg_len
                        intervals.append((st,en))
                    itemid2intervals[it] = intervals

                obs_window_df = obs_window_df.reset_index(drop=True)
                mask_array = np.zeros(obs_window_df.shape[0], dtype=bool)

                itemids_np = obs_window_df['itemid'].values
                offsets_np = obs_window_df['offset'].values

                for it in unique_itemids:
                    if it in binary_ids:
                        continue
                    intervals = itemid2intervals[it]
                    # 이 아이템의 행 인덱스
                    idxs = np.where(itemids_np == it)[0]
                    if len(idxs)==0:
                        continue

                    # 해당 offset
                    off_for_it = offsets_np[idxs]
                    combined_mask = np.zeros(len(idxs), dtype=bool)

                    for (st, en) in intervals:
                        combined_mask |= ((off_for_it >= st) & (off_for_it < en))

                    mask_array[idxs] = combined_mask

                self.samples.append((hadm_id, qtime, obs_window_df, cf_label, mask_array))
                seq_lens.append(obs_window_df.shape[0])
            stay_length.append(max_offset)
        
        print(f"[TimeSeriesDatasetCF_Outcome] Total {len(stay_length)} stays used")

        # Get stats of each stay length
        def get_stats(series: pd.Series):
            if series.empty:
                # 비어있으면 NaN으로 채우기
                return [0, np.nan, np.nan, np.nan, np.nan, np.nan]
            return [
                series.count(),       # count
                series.mean(),        # mean
                series.min(),         # min
                series.max(),         # max
                series.std(),         # std
                series.median()       # median
            ]

        stays_stats = get_stats(pd.Series(stay_length))

        stat_names = ['count','mean','min','max','std','median']
        df_stats = pd.DataFrame({
            'Statistic': stat_names,
            'Stay': stays_stats,

        })

        df_stats.set_index('Statistic', inplace=True)

        print("=== Stay-wise Statistics Table ===")
        print(df_stats)

        total_count = len(self.samples)
        pos_count = sum(1 for s in self.samples if s[3]==1)
        print(f"[TimeSeriesDatasetCF_Outcome] Created {total_count} samples. CF=1: {pos_count}")

        if total_count > 0:
            seq_lens = np.array(seq_lens)
            mean_ = seq_lens.mean()
            std_ = seq_lens.std()
            min_ = seq_lens.min()
            max_ = seq_lens.max()
            median_ = np.median(seq_lens)
            print(f"[TimeSeriesDatasetCF_Outcome] Sequence length stats: mean={mean_:.2f}, std={std_:.2f}, "
                  f"min={min_}, max={max_}, median={median_}")

        # ------------- Upsampling -------------
        if upsample:
            # 별도로 양성(CF=1)와 음성(CF=0) 샘플을 분리
            pos_samples = [s for s in self.samples if s[3] == 1]
            neg_samples = [s for s in self.samples if s[3] == 0]
            n_pos = len(pos_samples)
            n_neg = len(neg_samples)
            # 더 적은 샘플인 클래스의 수를 늘림 (중복 선택 허용)
            if n_pos > 0 and n_neg > 0:
                if n_pos > n_neg:
                    additional = random.choices(neg_samples, k=(n_pos - n_neg))
                    self.samples.extend(additional)
                elif n_neg > n_pos:
                    additional = random.choices(pos_samples, k=(n_neg - n_pos))
                    self.samples.extend(additional)
            # 재현성을 위해 seed 설정 후 전체 샘플 섞기
            random.seed(upsample_seed)
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hadm_id, query_time, obs_window_df, cf_label, mask_array = self.samples[idx]

        times = torch.tensor(obs_window_df['offset'].values / self.observation_window, dtype=torch.float32)
        varis = torch.tensor(obs_window_df['itemid'].values, dtype=torch.long)
        vals  = torch.tensor(obs_window_df['value'].values, dtype=torch.float32)

        hid_t = torch.tensor(hadm_id, dtype=torch.long)
        qt_t  = torch.tensor(query_time, dtype=torch.long)
        cf_t  = torch.tensor(cf_label, dtype=torch.long)
        mask_t= torch.tensor(mask_array, dtype=torch.bool)

        return hid_t, times, varis, vals, qt_t, cf_t, mask_t
    
class TimeSeriesDatasetAKI(Dataset):
    """
    AKI Outcome Dataset: 한 환자에서 여러 AKI onset을 고려하여 샘플을 생성함.
    
    샘플 생성 논리:
      - 관찰 윈도우(obs_window): [qtime - obs_window, qtime]
         → 이 기간 내에 **어떤 AKI onset이라도 존재하면 해당 sample을 skip**
      - Query range: (qtime, qtime + query_horizon]
         → 이 구간 내에 하나라도 AKI onset이 존재하면 label=1, 없으면 0
    """
    def __init__(self,
                 df: pd.DataFrame,
                 outcome_df: pd.DataFrame,
                 binary_ids: set,
                 obs_window=1440,
                 mask_length=360,
                 mask_segment_count=4,
                 step=60,
                 min_window=10,
                 query_horizon=480,
                 upsample=False,
                 upsample_seed=9871):
        super().__init__()
        self.samples = []
        self.observation_window = obs_window
        self.mask_length = mask_length
        self.mask_segment_count = mask_segment_count
        self.query_horizon = query_horizon

        seq_lens = []
        # hadm_id별로 AKI onset 리스트 구성
        hadm2aki_offsets = {}
        for _, row in outcome_df.iterrows():
            hid = row['hadm_id']
            # aki_offset가 여러 onset의 리스트여야 함; 만약 NaN이면 빈 리스트로 설정
            aoffs = row['aki_offset']
            if isinstance(aoffs, (list, np.ndarray)):
                # 만약 배열 전체가 NaN이면 빈 리스트로 설정
                if np.all(pd.isna(aoffs)):
                    aoffs = []
            else:
                if pd.isna(aoffs):
                    aoffs = []
            hadm2aki_offsets[hid] = aoffs


        stay_length = []
        for hadm_id, group in df.groupby('hadm_id'):
            if group.empty:
                continue
            max_offset = group['offset'].max()
            aki_onsets = hadm2aki_offsets.get(hadm_id, [])
            # flatten: 혹시 리스트 내에 리스트가 있다면
            if aki_onsets and isinstance(aki_onsets[0], list):
                flat = []
                for sub in aki_onsets:
                    flat.extend(sub)
                aki_onsets = flat
            start_qtime = obs_window
            # end_qtime = max_offset - self.query_horizon
            # if end_qtime <= start_qtime:
            #     continue
            # query_times = range(start_qtime, int(end_qtime)+1, step)
            query_times = [obs_window]
            for qtime in query_times:
                win_start = qtime - obs_window
                win_end   = qtime
                obs_window_df = group[(group['offset'] > win_start) & (group['offset'] <= win_end)].copy()
                if obs_window_df.empty or obs_window_df.shape[0] < min_window or max_offset < obs_window:
                    continue
                # (A) 관찰 윈도우 내에 **어떤** AKI onset이 존재하면 sample skip
                # if any((onset > win_start) and (onset <= win_end) for onset in aki_onsets):
                #     continue
                # if any((onset <= win_start) for onset in aki_onsets):
                #     continue
                # (B) Query range 내에 AKI onset이 있으면 label=1, 없으면 0
                aki_label = 0
                if any((onset > win_end) and (onset <= win_end + self.query_horizon) for onset in aki_onsets):
                    aki_label = 1
                obs_window_df['offset'] = obs_window_df['offset'] - win_start
                seg_len = self.mask_length // self.mask_segment_count
                possible_start = self.observation_window - seg_len
                if possible_start < 0:
                    possible_start = 0
                itemid2intervals = {}
                unique_itemids = obs_window_df['itemid'].unique()
                for it in unique_itemids:
                    if it in binary_ids:
                        continue
                    intervals = []
                    for _ in range(self.mask_segment_count):
                        st = np.random.randint(0, possible_start+1)
                        en = st + seg_len
                        intervals.append((st,en))
                    itemid2intervals[it] = intervals
                obs_window_df = obs_window_df.reset_index(drop=True)
                mask_array = np.zeros(obs_window_df.shape[0], dtype=bool)
                itemids_np = obs_window_df['itemid'].values
                offsets_np = obs_window_df['offset'].values
                for it in unique_itemids:
                    if it in binary_ids:
                        continue
                    intervals = itemid2intervals[it]
                    idxs = np.where(itemids_np == it)[0]
                    if len(idxs)==0:
                        continue
                    off_for_it = offsets_np[idxs]
                    combined_mask = np.zeros(len(idxs), dtype=bool)
                    for (st, en) in intervals:
                        combined_mask |= ((off_for_it >= st) & (off_for_it < en))
                    mask_array[idxs] = combined_mask
                self.samples.append((hadm_id, qtime, obs_window_df, aki_label, mask_array))
                seq_lens.append(obs_window_df.shape[0])
            stay_length.append(max_offset)
        print(f"[TimeSeriesDatasetAKI_Outcome] Total {len(stay_length)} stays used")
        total_count = len(self.samples)
        pos_count = sum(1 for s in self.samples if s[3]==1)
        print(f"[TimeSeriesDatasetAKI_Outcome] Created {total_count} samples. AKI=1: {pos_count}")
        if total_count > 0:
            seq_lens = np.array(seq_lens)
            print(f"[TimeSeriesDatasetAKI_Outcome] Sequence length stats: mean={seq_lens.mean():.2f}, std={seq_lens.std():.2f}, min={seq_lens.min()}, max={seq_lens.max()}, median={np.median(seq_lens)}")

                # ------------- Upsampling -------------
        if upsample:
            # 별도로 양성(CF=1)와 음성(CF=0) 샘플을 분리
            pos_samples = [s for s in self.samples if s[3] == 1]
            neg_samples = [s for s in self.samples if s[3] == 0]
            n_pos = len(pos_samples)
            n_neg = len(neg_samples)
            # 더 적은 샘플인 클래스의 수를 늘림 (중복 선택 허용)
            if n_pos > 0 and n_neg > 0:
                if n_pos > n_neg:
                    additional = random.choices(neg_samples, k=(n_pos - n_neg))
                    self.samples.extend(additional)
                elif n_neg > n_pos:
                    additional = random.choices(pos_samples, k=(n_neg - n_pos))
                    self.samples.extend(additional)
            # 재현성을 위해 seed 설정 후 전체 샘플 섞기
            random.seed(upsample_seed)
            random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        hadm_id, query_time, obs_window_df, aki_label, mask_array = self.samples[idx]
        times = torch.tensor(obs_window_df['offset'].values / self.observation_window, dtype=torch.float32)
        varis = torch.tensor(obs_window_df['itemid'].values, dtype=torch.long)
        vals  = torch.tensor(obs_window_df['value'].values, dtype=torch.float32)
        hid_t = torch.tensor(hadm_id, dtype=torch.long)
        qt_t  = torch.tensor(query_time, dtype=torch.long)
        aki_t = torch.tensor(aki_label, dtype=torch.long)
        mask_t = torch.tensor(mask_array, dtype=torch.bool)
        return hid_t, times, varis, vals, qt_t, aki_t, mask_t
    

class MakeLoadersCF(MakeLoaders):
    """
    Circulatory Failure as task!
    - Override create_datasets()
    """


    def create_datasets(self):
        """
        TimeSeriesDatasetCF (with outcome_df) => train/valid/test
        """
        print("[MakeLoadersCF] Creating circulatory failure datasets ...")

        # self.df_train, self.outcome_train 등은 부모 클래스 MakeLoaders에서 상속됨
        # 예: TimeSeriesDatasetCF는 다음과 같은 시그니처:
        # TimeSeriesDatasetCF(df, outcome_df, obs_window, mask_length, mask_segment_count, step)
        train_dataset = TimeSeriesDatasetCF(
            df=self.df_train,
            outcome_df=self.outcome_train,
            binary_ids=self.binary_itemids,
            obs_window=self.obs_window,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
            upsample=self.upsample
        )
        valid_dataset = TimeSeriesDatasetCF(
            df=self.df_valid,
            outcome_df=self.outcome_valid,
            binary_ids=self.binary_itemids,
            obs_window=self.obs_window,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
        )
        test_dataset = TimeSeriesDatasetCF(
            df=self.df_test,
            outcome_df=self.outcome_test,
            binary_ids=self.binary_itemids,
            obs_window=self.obs_window,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
        )
        return train_dataset, valid_dataset, test_dataset

class MakeLoadersMor(MakeLoaders):
    """
    Mortality as task!
    """
    def __init__(self,
                 df_train, df_valid, df_test,
                 outcome_train, outcome_valid, outcome_test,
                 obs_window=1440,
                 batch_size=32,
                 mask_length=600,
                 mask_segment_count=4,
                 trim_percentile=0.01,
                 scale=True,
                 step=60,
                 min_window=10,
                 emb_idx=None,
                 query_type='initial',
                 upsample=False,
                 bucket_sampler=True):
        """
        Added query_type, upsample as additional param
        """
        # 1) 부모 init 호출
        super().__init__(
            df_train, df_valid, df_test,
            outcome_train, outcome_valid, outcome_test,
            obs_window=obs_window,
            batch_size=batch_size,
            mask_length=mask_length,
            mask_segment_count=mask_segment_count,
            trim_percentile=trim_percentile,
            scale=scale,
            step=step,
            min_window=min_window,
            emb_idx = emb_idx,
            upsample = upsample,
            bucket_sampler= bucket_sampler
        )
        # 2) 자식 클래스만의 추가 초기화
        self.query_type = query_type

    def create_datasets(self):
        print("[MakeLoadersMor] Creating mortality datasets ...")
   
        train_dataset = TimeSeriesDataset(
            self.df_train, self.outcome_train,
            observation_window=self.obs_window,
            binary_ids=self.binary_itemids, 
            mask_length=self.mask_length, 
            mask_segment_count=self.mask_segment_count, 
            query_type=self.query_type,
            step=self.step,
            min_window=self.min_window,
            upsample=self.upsample
        )
        valid_dataset = TimeSeriesDataset(
            self.df_valid, self.outcome_valid,
            observation_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length, 
            mask_segment_count=self.mask_segment_count, 
            query_type=self.query_type,
            step=self.step,
            min_window=self.min_window,
        )
        test_dataset = TimeSeriesDataset(
            self.df_test, self.outcome_test,
            observation_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length, 
            mask_segment_count=self.mask_segment_count, 
            query_type=self.query_type,
            step=self.step,
            min_window=self.min_window,
        )

        return train_dataset, valid_dataset, test_dataset


class MakeLoadersAKI(MakeLoaders):
    """
    Acute Kidney Injury as task!
    - Override create_datasets()
    """

    def create_datasets(self):
        """
        TimeSeriesDatasetAKI (with outcome_df) => train/valid/test
        """
        print("[MakeLoadersAKI] Creating acute kidney injury datasets ...")

        # self.df_train, self.outcome_train 등은 부모 클래스 MakeLoaders에서 상속됨
        # 예: TimeSeriesDatasetCF는 다음과 같은 시그니처:
        # TimeSeriesDatasetCF(df, outcome_df, obs_window, mask_length, mask_segment_count, step)
        train_dataset = TimeSeriesDatasetAKI(
            df=self.df_train,
            outcome_df=self.outcome_train,
            obs_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
            upsample=self.upsample
        )
        valid_dataset = TimeSeriesDatasetAKI(
            df=self.df_valid,
            outcome_df=self.outcome_valid,
            obs_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
        )
        test_dataset = TimeSeriesDatasetAKI(
            df=self.df_test,
            outcome_df=self.outcome_test,
            obs_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
        )
        return train_dataset, valid_dataset, test_dataset
    
class MakeLoadersLoS(MakeLoaders):
    """
    Acute Kidney Injury as task!
    - Override create_datasets()
    """

    def create_datasets(self):
        """
        TimeSeriesDatasetLoS (with outcome_df) => train/valid/test
        """
        print("[MakeLoadersLoS] Creating acute kidney injury datasets ...")

        # self.df_train, self.outcome_train 등은 부모 클래스 MakeLoaders에서 상속됨
        # 예: TimeSeriesDatasetCF는 다음과 같은 시그니처:
        # TimeSeriesDatasetCF(df, outcome_df, obs_window, mask_length, mask_segment_count, step)
        train_dataset = TimeSeriesDatasetLoS(
            df=self.df_train,
            outcome_df=self.outcome_train,
            observation_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
            upsample=self.upsample
        )
        valid_dataset = TimeSeriesDatasetLoS(
            df=self.df_valid,
            outcome_df=self.outcome_valid,
            observation_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
            upsample=self.upsample
        )
        test_dataset = TimeSeriesDatasetLoS(
            df=self.df_test,
            outcome_df=self.outcome_test,
            observation_window=self.obs_window,
            binary_ids=self.binary_itemids,
            mask_length=self.mask_length,
            mask_segment_count=self.mask_segment_count,
            step=self.step,
            min_window=self.min_window,
            upsample=self.upsample
        )
        return train_dataset, valid_dataset, test_dataset