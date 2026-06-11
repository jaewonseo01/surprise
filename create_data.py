import pandas as pd
import numpy as np
import tqdm
import math
from typing import List, Tuple, Iterable, Dict, Optional

def agg_window_fast(window_df: pd.DataFrame):
    """
    window_df: columns=['offset','itemid','value']
    pivot -> shape=(#unique offset, #unique itemid)
    Then for each column(itemid), compute stats (mean, max, ...)
    """
    if window_df.empty:
        return {}

    pivoted = window_df.pivot(index='offset', columns='itemid', values='value')
    # pivoted: offset x itemid
    # col별 => mean, max, min, ...
    stats_dict = {}
    col_names = pivoted.columns
    means = pivoted.mean(axis=0)
    maxs  = pivoted.max(axis=0)
    mins  = pivoted.min(axis=0)
    stds  = pivoted.std(axis=0)
    meds  = pivoted.median(axis=0)
    q25s  = pivoted.quantile(0.25, axis=0)
    q75s  = pivoted.quantile(0.75, axis=0)
    counts = pivoted.count(axis=0)

    for item in col_names:
        # skip if all NaN?
        # or do if counts[item]==0 => skip
        rng = maxs[item] - mins[item] if counts[item]>0 else np.nan
        stats_dict[f'{item}_mean'] = means[item]
        stats_dict[f'{item}_max']  = maxs[item]
        stats_dict[f'{item}_min']  = mins[item]
        stats_dict[f'{item}_std']  = stds[item]
        stats_dict[f'{item}_median'] = meds[item]
        stats_dict[f'{item}_q25'] = q25s[item]
        stats_dict[f'{item}_q75'] = q75s[item]
        stats_dict[f'{item}_range'] = rng
        stats_dict[f'{item}_count'] = counts[item]

    return stats_dict


def get_clip_bounds_from_train(train_df: pd.DataFrame, binary_itemids: set, trim_percentile=0.01):
    """
    Train 데이터로부터 각 itemid별 (lower, upper) clip bound를 계산.
    binary_itemids는 trimming 제외.
    
    반환:
      clip_bounds: dict(itemid -> (lower, upper))
    """

    clip_bounds = {}
    for item_id, grp in train_df.groupby('itemid'):
        if item_id in binary_itemids:
            continue
        lower = grp['value'].quantile(trim_percentile)
        upper = grp['value'].quantile(1 - trim_percentile)
        clip_bounds[item_id] = (lower, upper)
    
    return clip_bounds

def apply_clip_bounds(df: pd.DataFrame, binary_itemids: set, clip_bounds: dict):
    """
    df를 itemid별로 나누어, clip_bounds가 있는 경우 그 범위 내로 trimming.
    binary itemid는 trimming 없이 그대로.
    """

    trimmed_parts = []
    for item_id, grp in df.groupby('itemid'):
        if item_id in binary_itemids:
            trimmed_parts.append(grp)
        else:
            if item_id in clip_bounds:
                l, u = clip_bounds[item_id]
                clipped = grp[(grp['value']>=l)&(grp['value']<=u)]
                trimmed_parts.append(clipped)
            else:
                trimmed_parts.append(grp)
    return pd.concat(trimmed_parts, ignore_index=True)


def aggregate_time_series_features_los_stepwise(
    df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    obs_window: int = 48*60,   # default 48시간
    step: int = 60,           # default 1시간 간격
    trim_percentile: float = 0.01
):
    """
    1) df ([pid, offset, itemid, value])에 대해
       pid별로 max_offset까지 step 간격으로 query_time을 생성.
       - window = [query_time, query_time + obs_window)
       - window 내 itemid별 통계를 계산( mean, max, min, etc... ).
    2) outcome_df ([pid, los_offset])를 이용해 los 라벨링:
       - los_offset이 window 안([wstart, wend]) 혹은 이전이면 => 해당 행 skip
       - los_offset이 (wend, wend+480] 내에 있으면 => los_label=1, else=0

    Returns:
      DataFrame: 각 (pid, query_time) 별로 itemid 통계가 열이 되고,
                 los_label 열이 추가됨.

    [주의] Trimming, 이진 변수 식별 등은 외부에서 해주세요.
    """

    # ------------------------
    # 0) Data 복사
    # ------------------------
    df_filtered = df.copy()

    # ------------------------
    # E) pid별로 step 윈도우 + los 라벨링
    # ------------------------
    results = []

    hadm_groups = df_filtered.groupby('pid')
    for pid, group in tqdm(hadm_groups, desc="[aggregate los stepwise]"):
        if group.empty:
            continue

        # label offset
        los_series = outcome_df.loc[outcome_df['pid'] == pid, 'los']
        los = los_series.iloc[0]

        max_offset = group['offset'].max()
        if max_offset < obs_window:
            continue
        qtimes = [obs_window]



        for qtime in qtimes:
            win_start = qtime - obs_window
            win_end   = qtime 
            # 1) 윈도우 dataframe
            obs_window_df = group[(group['offset']>win_start) & (group['offset']<=win_end)].copy()
            if obs_window_df.empty or obs_window_df.shape[0]<10:
                continue

            if los <= 7:
                label = 0
            else:
                label = 1

            # 4) itemid별로 통계 계산
            row_dict = {
                'pid': pid,
                'query_time': qtime,
                'outcome': label
            }

            stats_dict = agg_window_fast(obs_window_df[['offset','itemid','value']])
            row_dict.update(stats_dict)

            results.append(row_dict)

    # ------------------------
    # F) 결과 DataFrame
    # ------------------------
    final_df = pd.DataFrame(results)
    if final_df.empty:
        print("[aggregate_time_series_features_los_stepwise] No data generated.")
        return final_df

    # 편의상 pid, query_time, label 순으로 정렬
    base_cols = ['pid', 'query_time', 'outcome']
    other_cols = [c for c in final_df.columns if c not in base_cols]
    final_cols = base_cols + other_cols
    final_df = final_df[final_cols]

    return final_df

def create_inputs_from_data(data, max_len = 1000):
    if data=='mimic' or data=='eicu':
        print("✅ Starting data processing")
        # =====================
        # 1) 데이터 불러오기
        # =====================
        df = pd.read_feather(f'./data/{data}_data.feather')

        # =====================
        # 2) 변수 이름 리스트 생성
        # =====================
        param_list = sorted(df['itemid'].unique())
        print(f"Total unique parameters: {len(param_list)}")
        np.save(f'./processed_data/ts_params_{data}.npy', param_list)
        print('ts_params.npy saved')

        df_static = pd.read_feather(f'./data/{data}_data_static.feather')

        static_dict = {}
        for idx, row in df_static.iterrows():
            pid = row['pid']
            gender_onehot = row['gender']
            static_array = [row['age']] + [row['gender']] + [row['height']]
            static_dict[pid] = static_array

        print('static_dict ready, example:', list(static_dict.items())[:3])

        # 정적 변수 있음
        static_param_list = ['age', 'gender', 'height']
        d_static = 3
        np.save(f'./processed_data/static_params_{data}.npy', static_param_list)

        # =====================
        # 3) 환자별 그룹화 및 PTdict_list 생성
        # =====================
        max_len = max_len # Max no. of unique timestamps
        max_offset = 48 * 60  # 48시간
        F = len(param_list)

        PTdict_list = []

        for pid, group in df.groupby('pid'):
            group = group.sort_values(by='offset')
            ts = group[['offset', 'itemid', 'value']].values

            unq_offsets = []
            for sample in ts:
                offset = sample[0]
                if (offset not in unq_offsets) and (offset <= max_offset):
                    unq_offsets.append(offset)
            unq_offsets = np.array(unq_offsets)
            length = len(unq_offsets)

            # 시계열 및 타임스탬프 배열 생성
            Parr = np.zeros((max_len, F))
            Tarr = np.zeros((max_len, 1))

            for sample in ts:
                offset = sample[0]
                param = sample[1]
                value = sample[2]
                if offset <= max_offset:
                    try:
                        time_id = np.where(offset == unq_offsets)[0][0]
                        param_id = np.where(param_list == param)[0][0]
                        Parr[time_id, param_id] = value
                        Tarr[time_id, 0] = offset
                    except:
                        continue
            
            if pid in static_dict:
                static_array = static_dict[pid]
            else:
                static_array = [0] * d_static  # fallback

            my_dict = {
                'id': pid,
                'static': static_array,
                'extended_static': static_array,
                'arr': Parr,
                'time': Tarr,
                'length': length
            }
            PTdict_list.append(my_dict)

        print(f"Total patients processed: {len(PTdict_list)}")

        np.save(f'./processed_data/PTdict_list_{data}.npy', PTdict_list)
        print('PTdict_list.npy saved, ready for Raindrop training')
        # =====================
        # 4) Outcomes 저장
        # =====================

        outcomes = pd.read_feather(f'./data/{data}_outcomes.feather') # AKI, CF, LoS, Mor
        arr_outcomes = outcomes.values
        np.save(f'./processed_data/arr_outcomes_{data}.npy', arr_outcomes)
    else:
        print("❗Invalid dataset. Try 'eicu' or 'mimic' ")

def prepare_dataset_samples(
    data: pd.DataFrame,
    static: pd.DataFrame,
    *,
    sort_obs_by: Iterable[str] = ("offset", "itemid"),
    dropna_in_obs_value: bool = True,
    cast_itemid_to_int: bool = True,
    dtypes: Tuple[str, str] = ("float32", "float32"),  # (obs dtype, static dtype)
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Prepare variable-length observation matrices per pid for a PyTorch Dataset.

    Returns
    -------
    samples : list of (pid, obs_matrix, static_vec)
        - pid: int
        - obs_matrix: np.ndarray shape (Ni, 3) with rows [itemid, offset, value]
        - static_vec: np.ndarray shape (3,) = [age, gender, height]
      Ni differs per pid. dtypes = (obs_dtype, static_dtype)

    Inputs
    ------
    data : DataFrame with columns ['pid','itemid','offset','value']
    static : DataFrame with columns ['pid','age','gender','height', ...]
             (gender is already binary; we keep only age, gender, height)
    """
    obs_dtype, static_dtype = dtypes

    # --- sanity checks
    req_data = {"pid", "itemid", "offset", "value"}
    req_static = {"pid", "age", "gender", "height"}
    md = req_data - set(data.columns)
    ms = req_static - set(static.columns)
    if md:
        raise ValueError(f"`data` missing columns: {sorted(md)}")
    if ms:
        raise ValueError(f"`static` missing columns: {sorted(ms)}")

    # --- static: keep only needed, ensure numeric
    static_slim = static.loc[:, ["pid", "age", "gender", "height"]].copy()
    for col in ["age", "gender", "height"]:
        static_slim[col] = pd.to_numeric(static_slim[col], errors="coerce")
    # gender is already binary; optional clamp for robustness
    static_slim["gender"] = static_slim["gender"].clip(0, 1)

    # drop bad static rows
    static_slim = static_slim.dropna(subset=["age", "gender", "height"])
    static_slim = static_slim.drop_duplicates(subset=["pid"], keep="first")

    # build lookup: pid -> [age, gender, height]
    static_vec_lookup = {
        int(pid): np.array([row.age, row.gender, row.height], dtype=static_dtype)
        for pid, row in static_slim.set_index("pid").iterrows()
    }

    # --- data: clean & sort
    data_clean = data.loc[:, ["pid", "itemid", "offset", "value"]].copy()
    if cast_itemid_to_int:
        data_clean["itemid"] = pd.to_numeric(data_clean["itemid"], errors="coerce")
    data_clean["offset"] = pd.to_numeric(data_clean["offset"], errors="coerce")
    data_clean["value"]  = pd.to_numeric(data_clean["value"],  errors="coerce")
    data_clean["pid"]    = pd.to_numeric(data_clean["pid"],    errors="coerce")

    if dropna_in_obs_value:
        data_clean = data_clean.dropna(subset=["pid", "itemid", "offset", "value"])

    data_clean["pid"] = data_clean["pid"].astype(int)
    sort_cols = ["pid"] + list(sort_obs_by)
    data_clean = data_clean.sort_values(sort_cols, kind="mergesort")  # stable

    # --- group and assemble outputs
    samples: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for pid, df_pid in data_clean.groupby("pid", sort=False):
        # if static is unexpectedly missing for this pid, skip quietly (preprocessing should’ve ensured presence)
        static_vec = static_vec_lookup.get(pid, None)
        if static_vec is None:
            continue

        obs_mat = df_pid.loc[:, ["itemid", "offset", "value"]].to_numpy(dtype=obs_dtype)
        if obs_mat.shape[0] == 0:
            continue

        samples.append((int(pid), obs_mat, static_vec))

    return samples


Sample = Tuple[int, np.ndarray, np.ndarray]

def compute_norm_stats(
    reference_samples: List[Sample],
    categorical_itemids: Iterable[float | int] = ()
) -> Dict:
    """
    Build normalization stats from reference samples (e.g., train).
    - value: per-item mean/std for NON-categorical items only
    - static: age, height mean/std (gender untouched)
    - categorical_itemids are excluded entirely from value stats.

    categorical_itemids: iterable of itemids (int/float). These will be excluded
      from stats and later left unnormalized.
    """
    cat_set = {float(x) for x in categorical_itemids}

    per_item_sums, per_item_sq, per_item_cnt = {}, {}, {}
    global_sum = 0.0
    global_sq  = 0.0
    global_cnt = 0

    ages, heights = [], []

    for _, obs, st in reference_samples:
        ages.append(float(st[0]))
        heights.append(float(st[2]))

        if obs.size == 0:
            continue

        itemids = obs[:, 0]
        values  = obs[:, 2]
        for it, v in zip(itemids, values):
            it = float(it); v = float(v)
            if it in cat_set:
                # skip categorical values from stats
                continue
            per_item_sums[it] = per_item_sums.get(it, 0.0) + v
            per_item_sq[it]   = per_item_sq.get(it, 0.0)   + v*v
            per_item_cnt[it]  = per_item_cnt.get(it, 0)    + 1
            global_sum += v; global_sq += v*v; global_cnt += 1

    def mean_std(sum_, sq_, cnt_):
        if cnt_ <= 1:
            m = (sum_/cnt_) if cnt_ > 0 else 0.0
            return m, 1.0
        m = sum_ / cnt_
        var = max(sq_/cnt_ - m*m, 0.0)
        s = math.sqrt(var) if var > 1e-12 else 1.0
        return m, s

    per_item_stats = {it: mean_std(per_item_sums[it], per_item_sq[it], per_item_cnt[it])
                      for it in per_item_cnt.keys()}

    # global fallback (for unseen non-categorical items)
    g_mean, g_std = mean_std(global_sum, global_sq, global_cnt)

    def arr_mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size <= 1:
            return (float(arr.mean()) if arr.size > 0 else 0.0, 1.0)
        m = float(arr.mean()); s = float(arr.std(ddof=0))
        return (m, 1.0) if s < 1e-12 else (m, s)

    age_stats    = arr_mean_std(ages)
    height_stats = arr_mean_std(heights)

    return {
        "value": {
            "per_item": per_item_stats,
            "global": (g_mean, g_std),
            "categorical": list(cat_set),  # keep for downstream
        },
        "static": {
            "age": age_stats,
            "height": height_stats,
        }
    }


def _normalize_single_samples(
    samples: List[Sample],
    stats: Dict,
    *,
    clamp_std_min: float = 1e-12
) -> List[Sample]:
    """
    Apply normalization:
      - value: z-score per itemid (skip if categorical)
      - static: z on age, height; gender unchanged
    """
    per_item = stats["value"]["per_item"]
    g_mean, g_std = stats["value"]["global"]
    cat_set = set(stats["value"].get("categorical", []))

    if g_std < clamp_std_min: g_std = 1.0
    age_m, age_s       = stats["static"]["age"]
    height_m, height_s = stats["static"]["height"]
    if age_s   < clamp_std_min:   age_s   = 1.0
    if height_s < clamp_std_min:  height_s = 1.0

    out: List[Sample] = []
    for pid, obs, st in samples:
        st2 = st.astype(np.float32, copy=True)
        # st = [age, gender, height]
        st2[0] = (st2[0] - age_m)    / age_s      # age
        # st2[1] = gender (keep as is)
        st2[2] = (st2[2] - height_m) / height_s   # height

        if obs.size > 0:
            obs2 = obs.astype(np.float32, copy=True)
            for i in range(obs2.shape[0]):
                it = float(obs2[i, 0])
                if it in cat_set:
                    # leave categorical as-is
                    continue
                m, s = per_item.get(it, (g_mean, g_std))
                if s < clamp_std_min: s = 1.0
                obs2[i, 2] = (obs2[i, 2] - m) / s
        else:
            obs2 = obs

        out.append((pid, obs2, st2))
    return out


def normalize_many(
    reference_samples: List[Sample],
    * ,
    categorical_itemids: Iterable[float | int] = (),
    **datasets: Dict[str, List[Sample]]
) -> Tuple[Dict[str, List[Sample]], Dict]:
    """
    Normalize multiple datasets using stats from `reference_samples`.
    Pass `categorical_itemids` to exclude those itemids from normalization.

    Usage:
      normalized, stats = normalize_many(
          train_samples, categorical_itemids={19,20,21},  # example
          train=train_samples, valid=valid_samples, test=test_samples
      )
    """
    stats = compute_norm_stats(reference_samples, categorical_itemids=categorical_itemids)
    normalized = {name: _normalize_single_samples(samples, stats)
                  for name, samples in datasets.items()}
    return normalized, stats

# Use in form
# normalized, stats = normalize_many(
#     train_samples,
#     categorical_itemids=cat_items,
#     train=train_samples, valid=valid_samples, test=test_samples, target=target_samples
# )

if __name__=="__main__":
    create_inputs_from_data('eicu')
    create_inputs_from_data('mimic')