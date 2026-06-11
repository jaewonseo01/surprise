import pandas as pd
import numpy as np
import torch

import datetime
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def mask_rows_by_proportion(df: pd.DataFrame, mask_ratio: float, seed: int = 42) -> pd.DataFrame:
    """
    주어진 DataFrame (컬럼: ['offset', 'hadm_id', 'itemid', 'value'])에 대해,
    각 hadm_id별로 전체 관측치 중 일정 비율(mask_ratio)을 무작위로 masking 처리합니다.
    (mask_ratio 예: 0.3이면 각 hadm_id 그룹의 30% 행의 value를 NaN으로 설정)
    
    Parameters:
      df: DataFrame with columns ['offset', 'hadm_id', 'itemid', 'value']
      mask_ratio: masking할 비율 (예: 0.3)
      seed: 재현성을 위한 난수 시드
      
    Returns:
      df_masked: 원본 DataFrame과 동일한 구조이며, 각 hadm_id별로 mask_ratio에 해당하는 행의 'value'가 NaN 처리됨.
    """
    df_masked = df.copy()
    np.random.seed(seed)
    
    # 각 hadm_id 그룹별로 처리
    for hadm_id, group in df_masked.groupby('hadm_id'):
        idxs = group.index
        n_rows = len(idxs)
        n_to_mask = int(np.floor(n_rows * mask_ratio))
        if n_to_mask > 0:
            mask_idxs = np.random.choice(idxs, size=n_to_mask, replace=False)
            df_masked.loc[mask_idxs, 'value'] = np.nan
            
    return df_masked

def find_aki_onset_for_patient(patient_df, creat_itemids):
    """
    환자 한 명(patient_df)에 대해
    - 크레아티닌(itemid ∈ creat_itemids)만 필터
    - offset 오름차순으로 보면서 AKI 기준을 만족하는 모든 시점을 리스트로 반환
      AKI 기준:
        (A) 최근 48시간(=2880분) 내의 측정치들과 비교하여, 
            현재 값과의 차이가 >= 0.3 mg/dL인 경우
        (B) 전체 측정치의 baseline(최소 creatinine) 대비 1.5배 이상 증가한 경우
    반환: 리스트 (예: [off1, off2, ...]), 조건을 만족하는 시점이 없으면 빈 리스트 []
    """
    # 1) 해당 환자의 크레아티닌 값만 필터
    cdf = patient_df[patient_df['itemid'].isin(creat_itemids)].copy()
    if cdf.empty:
        return []
    # 2) baseline = 전체 측정치 중 최소값
    baseline = cdf['value'].min()
    # 3) 시간 순 정렬
    cdf = cdf.sort_values('offset').reset_index(drop=True)
    recent_measure = []  # 최근 48시간 내 측정 (offset, value)
    onsets = []
    for i, row in cdf.iterrows():
        off = row['offset']
        val = row['value']
        # 최근 48시간 내 측정을 유지 (off-2880분 이후)
        lower_bound = off - 2880
        recent_measure = [(poff, pval) for (poff, pval) in recent_measure if poff >= lower_bound]
        recent_measure.append((off, val))
        # Criterion (B): baseline 대비 1.5배 이상
        if val >= 1.5 * baseline:
            onsets.append(off)
            continue  # 추가 onset 검사를 위해 계속 진행
        # Criterion (A): 최근 48시간 내 어느 측정과 비교해도 val - prev_val >= 0.3
        for (prev_off, prev_val) in recent_measure:
            if val - prev_val >= 0.3:
                onsets.append(off)
                break
    if onsets == []:
        onsets = np.nan
    return onsets


def build_aki_outcome(train_df, valid_df, test_df,
                      creat_itemids, clip_bounds):
    """
    - data: [hadm_id, offset, itemid, value]
    - creat_itemids: creatinine에 해당하는 itemid 집합
    - trim_percentile: 상하위 X% outlier 제거 (train 기준)

    반환:
      outcome_train, outcome_valid, outcome_test
      => [hadm_id, aki_offset], aki_offset=NaN이면 AKI 없음
    """
    # 0) creatinine itemid만 필터
    #    (다른 변수 필요 없다고 가정)
    print(f'[AKI Outcomes] 🪄 Starting process at {datetime.datetime.now()}...')

    orig_train_ids = train_df['hadm_id'].unique()
    orig_valid_ids = valid_df['hadm_id'].unique()
    orig_test_ids  = test_df['hadm_id'].unique()

    train_df = train_df[train_df['itemid'].isin([creat_itemids])].copy()
    valid_df = valid_df[valid_df['itemid'].isin([creat_itemids])].copy()
    test_df  = test_df[test_df['itemid'].isin([creat_itemids])].copy()

    # 1) binary item detect? 여기서는 크레아티닌이라 이진은 아님 -> pass
    #    (사용자가 필요하면 추가)
    # 2) trim outliers
    if clip_bounds is not None:

        # (b) Train, Valid, Test에서 실제로 제거
        def trim_df(df, clip_dict):
            new_list = []
            for _item_id, group in df.groupby('itemid'):
                if _item_id in clip_dict:
                    l, u = clip_dict[_item_id]
                    clipped = group.loc[(group['value'] > l) & (group['value'] < u)]
                    new_list.append(clipped)
                else:
                    # clip_dict에 없는 itemid (train에 드물거나 없었을 수 있음)
                    # 그대로 두거나, 스킵하는 등 정책 결정
                    new_list.append(group)

            if len(new_list) > 0:
                return pd.concat(new_list, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns)

        train_df = trim_df(train_df, clip_bounds)
        valid_df = trim_df(valid_df, clip_bounds)
        test_df  = trim_df(test_df, clip_bounds)

        print(f"[Trimming] Completed. Remaining train size = {len(train_df)}, "
              f"valid size = {len(valid_df)}, test size = {len(test_df)}")
    else:
        print("[Trimming] Skipped or trim_percentile=0")

    # 3) hadm_id별 aki onset
    #    train
    train_results=[]
    for hadm_id, group in train_df.groupby('hadm_id'):
        onset = find_aki_onset_for_patient(group, [creat_itemids])
        train_results.append((hadm_id, onset))
    outcome_train = pd.DataFrame(train_results, columns=['hadm_id','aki_offset'])
    print(f"   ✅ [Train] AKI onsets in {outcome_train.dropna(subset=['aki_offset'])['hadm_id'].nunique()} ids")

    #    valid
    valid_results=[]
    for hadm_id, group in valid_df.groupby('hadm_id'):
        onset = find_aki_onset_for_patient(group, [creat_itemids])
        valid_results.append((hadm_id, onset))
    outcome_valid = pd.DataFrame(valid_results, columns=['hadm_id','aki_offset'])
    print(f"   ✅ [Valid] AKI onsets in {outcome_valid.dropna(subset=['aki_offset'])['hadm_id'].nunique()} ids")

    #    test
    test_results=[]
    for hadm_id, group in test_df.groupby('hadm_id'):
        onset = find_aki_onset_for_patient(group, [creat_itemids])
        test_results.append((hadm_id, onset))
    outcome_test = pd.DataFrame(test_results, columns=['hadm_id','aki_offset'])
    print(f"   ✅ [Test] AKI onsets in {outcome_test.dropna(subset=['aki_offset'])['hadm_id'].nunique()} ids")

    # 1) 각 outcome_df에 'set' 열을 추가하여 구분
    outcome_train['set'] = 'Train'
    outcome_valid['set'] = 'Valid'
    outcome_test['set'] = 'Test'


    print(f'[AKI Outcomes] 🪄 Process over at {datetime.datetime.now()}...')


    return outcome_train, outcome_valid, outcome_test

def find_cf_onsets_for_patient(patient_df, map_itemids, lactate_itemids, c01_itemids):
    """
    환자 한 명(patient_df)에 대해, offset 오름차순으로 CF onset 시점을 모두 찾는다.
    
    CF 기준:
      - Condition 1: MAP <= 65 (for map_itemids) 또는 c01 == 1 (for c01_itemids)
      - Condition 2: lactate > 2 (for lactate_itemids)
      두 조건이 서로 120분 이내에 발생하면 해당 offset을 CF onset으로 판단.
    
    반환: 리스트 (예: [off1, off2, ...]); 조건을 만족하는 시점이 없으면 np.nan
    """
    # offset 기준으로 오름차순 정렬
    patient_df = patient_df.sort_values('offset').reset_index(drop=True)
    
    cond1_list = []  # MAP<=65 또는 c01==1을 만족하는 offset 저장
    lactate_list = []  # lactate > 2를 만족하는 offset 저장
    onsets = []  # 두 조건이 모두 만족된 CF onset들을 저장
    
    for i, row in patient_df.iterrows():
        off = row['offset']
        it  = row['itemid']
        val = row['value']
        
        # 먼저, 리스트에서 최근 120분 이전의 이벤트는 버림
        cond1_list = [c for c in cond1_list if c >= off - 120]
        lactate_list = [l for l in lactate_list if l >= off - 120]
        
        # Condition 1: MAP 조건 또는 c01 조건
        if (it == map_itemids and val <= 65) or (it == c01_itemids and val == 1):
            cond1_list.append(off)
            # 조건 1이 발생한 시점에서, 만약 이미 lactate 이벤트가 기록되어 있으면 조건 2도 만족한 것으로 간주
            for l_off in lactate_list:
                if abs(off - l_off) <= 120:
                    onsets.append(off)
                    break  # 한 번 만족하면 추가하고 넘어감
        
        # Condition 2: lactate > 2
        if it == lactate_itemids and val > 2:
            lactate_list.append(off)
            # 조건 2 발생 시, 이미 cond1_list에 있는 이벤트 중 120분 이내에 있는 것이 있다면 onset 기록
            for c_off in cond1_list:
                if abs(off - c_off) <= 120:
                    onsets.append(off)
                    break
        
    if not onsets:
        return np.nan
    return onsets

def find_cf_onset_for_patient(patient_df, map_itemids, lactate_itemids, c01_itemids):
    # Deprecated
    """
    환자 한 명(patient_df)에 대해
    offset 오름차순으로 보면서 "최초 CF 발생 시점"을 찾는다.

    - data columns: [hadm_id, offset, itemid, value]
    - return: float(cf_offset) or np.nan
    """
    # offset 기준으로 정렬
    patient_df = patient_df.sort_values('offset', ascending=True)
    patient_df = patient_df.reset_index(drop=True)

    # 방법 1) 각 offset에서 "up to that offset"을 window로 보고 check?
    #         => 연속적으로 커지는 window이므로 비효율적
    #
    # 방법 2) row 단위로 MAP<=65를 바로 체크
    #         + lactate>2 / c01=1는 "120분 이내"여부를 확인 => 과거 offset과 비교
    #
    # 여기서는 방법 2 시나리오:
    #   MAP<=65인 row가 나오면 해당 offset이 CF
    #   lactate>2 row 발생시 "앞뒤 120분 내 c01=1"이 있는지,
    #   c01=1 row 발생시 "앞뒤 120분 내 lactate>2"가 있는지
    #
    # =>past_rows, or a queue/two-pointer

    # 2-pointer-ish approach:
    # keep track of lactate>2 offsets in a list
    # keep track of c01=1 offsets in a list
    # each new row -> check MAP <=65 or check lactate/c01 condition
    lactate_list = []
    c01_list = []

    for i, row in patient_df.iterrows():
        off = row['offset']
        it  = row['itemid']
        val = row['value']

        # (A) MAP <=65
        if it == map_itemids and val <=65:
            return off  # CF onset at offset=off

        # (B) lactate>2 / c01=1 => check pairing
        if it == lactate_itemids and val>2:
            # check existing c01_list offsets within [off-120, off+120]
            for c_off in c01_list:
                if abs(off-c_off) <=120:
                    return off  # or min(off, c_off)?
            # add lactate offset to list
            lactate_list.append(off)

        if it == c01_itemids and val==1:
            # check lactate_list offsets
            for l_off in lactate_list:
                if abs(off-l_off) <=120:
                    return off
            c01_list.append(off)

        # (C) 오래된 offset(현재 offset에서 120보다 더 떨어진) 제거해서 관리 (optional)
        #     => if c01_list or lactate_list is large, we can pop old offsets
        # e.g. c01_list=[ x for x in c01_list if x>=off-120 ]

    # if we never returned => no CF
    return np.nan



def build_cf_outcome(train_df, valid_df, test_df,
                     map_itemids, lactate_itemids, c01_itemids, clip_bounds):
    """
    data: [hadm_id, offset, itemid, value]
    return outcome_df: [hadm_id, cf_offset] (NaN if none)
    """
    print(f'[CF Outcomes] 🪄 Starting process at {datetime.datetime.now()}...')

    orig_train_ids = train_df['hadm_id'].unique()
    orig_valid_ids = valid_df['hadm_id'].unique()
    orig_test_ids  = test_df['hadm_id'].unique()
    
    train_df = train_df.loc[train_df['itemid'].isin([map_itemids, lactate_itemids, c01_itemids]), :]
    valid_df = valid_df.loc[valid_df['itemid'].isin([map_itemids, lactate_itemids, c01_itemids]), :]
    test_df = test_df.loc[test_df['itemid'].isin([map_itemids, lactate_itemids, c01_itemids]), :]

    print("[Auto-detect Binary ItemIDs] Start...")
    binary_itemids = set()

    for item_id, group in train_df.groupby('itemid'):
        # set of unique values
        unique_vals = set(group['value'].unique())
        # 혹시 NaN이 있을 수 있으므로 제거
        if np.nan in unique_vals:
            unique_vals.discard(np.nan)
        # 만약 unique_vals가 {0} or {1} or {0,1} 이라면 binary로 판단
        if unique_vals.issubset({0,1}):
            binary_itemids.add(item_id)

    print(f"[Auto-detect Binary ItemIDs] Found {len(binary_itemids)} itemids:", binary_itemids)

    # -------------------------------
    # 1) Trim outliers based on train distribution
    # -------------------------------
    if clip_bounds is not None:
        # (a) Train, Valid, Test에서 실제로 제거
        def trim_df(df, clip_dict):
            new_list = []
            for _item_id, grp in df.groupby('itemid'):
                # binary itemid면 trimming 없이 그대로
                if _item_id in binary_itemids:
                    new_list.append(grp)
                    continue

                if _item_id in clip_dict:
                    l, u = clip_dict[_item_id]
                    clipped = grp.loc[(grp['value'] > l) & (grp['value'] < u)]
                    new_list.append(clipped)
                else:
                    # clip_dict에 없는 itemid (train에 드물거나 없었을 수 있음)
                    # 그대로 두거나, 스킵하는 등 정책 결정
                    new_list.append(grp)

            if len(new_list) > 0:
                return pd.concat(new_list, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns)

        train_df = trim_df(train_df, clip_bounds)
        valid_df = trim_df(valid_df, clip_bounds)
        test_df  = trim_df(test_df, clip_bounds)

        print(f"[Trimming] Completed. Remaining train size = {len(train_df)}, "
              f"valid size = {len(valid_df)}, test size = {len(test_df)}")
    else:
        print("[Trimming] Skipped or trim_percentile=0")

   
    results = []
    for hadm_id, group in train_df.groupby('hadm_id'):
        cf_onset = find_cf_onsets_for_patient(group, map_itemids, lactate_itemids, c01_itemids)
        results.append((hadm_id, cf_onset))    
    outcome_train = pd.DataFrame(results, columns=['hadm_id','cf_offset'])
    print(f"   ✅ [Train] CF onsets in {outcome_train.loc[outcome_train['cf_offset'].notna(),'hadm_id'].nunique()} ids")

    results = []

    for hadm_id, group in valid_df.groupby('hadm_id'):
        cf_onset = find_cf_onsets_for_patient(group, map_itemids, lactate_itemids, c01_itemids)
        results.append((hadm_id, cf_onset))    
    outcome_valid = pd.DataFrame(results, columns=['hadm_id','cf_offset'])
    print(f"   ✅ [Valid] CF onsets in {outcome_valid.loc[outcome_valid['cf_offset'].notna(),'hadm_id'].nunique()} ids")

    results = []

    for hadm_id, group in test_df.groupby('hadm_id'):
        cf_onset = find_cf_onsets_for_patient(group, map_itemids, lactate_itemids, c01_itemids)
        results.append((hadm_id, cf_onset))    
    outcome_test = pd.DataFrame(results, columns=['hadm_id','cf_offset'])    
    print(f"   ✅ [Test] CF onsets in {outcome_test.loc[outcome_test['cf_offset'].notna(),'hadm_id'].nunique()} ids")
    

    # # -----------------------------------------------
    # # This is only necessary when using stays without CF in their occurence
    # # 6) 누락된 hadm_id를 다시 추가 -> cf_offset=NaN
    # #    => outer join or left join using the original hadm_id set
    # def reindex_outcome(outcome_df, orig_ids):
    #     all_ids_df = pd.DataFrame({'hadm_id': orig_ids})
    #     merged = all_ids_df.merge(outcome_df, on='hadm_id', how='left')
    #     return merged  # cf_offset=NaN for missing

    # outcome_train = reindex_outcome(outcome_train, orig_train_ids)
    # outcome_valid = reindex_outcome(outcome_valid, orig_valid_ids)
    # outcome_test  = reindex_outcome(outcome_test,  orig_test_ids)

    # # 1) 각 outcome_df에 'set' 열을 추가하여 구분
    # outcome_train['set'] = 'Train'
    # outcome_valid['set'] = 'Valid'
    # outcome_test['set'] = 'Test'

    # # 2) 세 DataFrame을 하나로 합치고, cf_offset이 결측인 행은 제거
    # df_plot = pd.concat([outcome_train, outcome_valid, outcome_test], ignore_index=True)
    # df_plot = df_plot.dropna(subset=['cf_offset'])  # cf_offset=NaN 제거

    # # 3) Seaborn KDE plot
    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(data=df_plot, x='cf_offset', hue='set', fill=True)
    # plt.title("Distribution of CF Onset (cf_offset)")
    # plt.show()

    # # 1) 각 set별 cf_offset 시리즈 (결측 제외)
    # train_offset = outcome_train['cf_offset'].dropna()
    # valid_offset = outcome_valid['cf_offset'].dropna()
    # test_offset  = outcome_test['cf_offset'].dropna()

    # # 2) 통계량 계산을 위한 함수
    # def get_stats(series: pd.Series):
    #     if series.empty:
    #         # 비어있으면 NaN으로 채우기
    #         return [0, np.nan, np.nan, np.nan, np.nan, np.nan]
    #     return [
    #         series.count(),       # count
    #         series.mean(),        # mean
    #         series.min(),         # min
    #         series.max(),         # max
    #         series.std(),         # std
    #         series.median()       # median
    #     ]

    # # 3) 각 세트별 통계량
    # train_stats = get_stats(train_offset)
    # valid_stats = get_stats(valid_offset)
    # test_stats  = get_stats(test_offset)

    # # 4) 하나의 표(열=Train/Valid/Test, 행=통계량 이름)로 정리
    # stat_names = ['count','mean','min','max','std','median']
    # df_stats = pd.DataFrame({
    #     'Statistic': stat_names,
    #     'Train': train_stats,
    #     'Valid': valid_stats,
    #     'Test':  test_stats
    # })

    # # 5) 보기 좋게 Statistic 열을 인덱스로 설정
    # df_stats.set_index('Statistic', inplace=True)

    # print("=== CF Offset Statistics Table ===")
    # print(df_stats)
    
    print(f'[CF Outcomes] 🪄 Process over at {datetime.datetime.now()}...')

    
    return outcome_train, outcome_valid, outcome_test