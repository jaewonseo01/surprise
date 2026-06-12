# runner_raindrop.py
import os
import numpy as np
import pandas as pd
import torch
import json

from utils_rd import getStats_fixed, getStats_static, tensorize_normalize_multilabel

# These itemids are binary (even though "categorical"), so EXCLUDE from normalization.
categorical_itemids = [i for i in range(19, 35)]  # 19..34 inclusive

# ✅ conventions
MISSING_VALUE = 0.0   # feature missing
MISSING_TIME  = -1.0  # time padding (minutes)
DROP_OUTCOME_COLS = {"aki_label", "cf_label"}


def split_data_dir(base_data_dir: str, split: int) -> str:
    if not (1 <= split <= 5):
        raise ValueError(f"split must be in [1..5], got {split}")
    return os.path.join(base_data_dir, f"split{split}")


def split_suffix(split: int) -> str:
    if not (1 <= split <= 5):
        raise ValueError(f"split must be in [1..5], got {split}")
    return f"_{split}"


def _aug_tag(use_aug: bool, aug_suffix: str) -> str:
    return aug_suffix if use_aug else "orig"


def feather_path(
    data_dir: str,
    domain: str,
    kind: str,
    split_name: str,
    split: int,
    *,
    use_aug: bool = False,
    aug_suffix: str = "aug",
) -> str:
    """
    kind: "data" | "outcomes"
    split_name: "train" | "valid" | "test"

    Original data:
      {domain}_data_{split_name}_{split}.feather
      e.g. eicu_data_train_1.feather

    Augmented data:
      {domain}_data_{aug_suffix}_{split_name}_{split}.feather
      e.g. eicu_data_aug_train_1.feather
           eicu_data_aug_p2_123x3_train_1.feather

    Static / outcomes are always original.
    """
    suf = split_suffix(split)

    if kind == "data":
        if use_aug:
            return os.path.join(data_dir, f"{domain}_data_{aug_suffix}_{split_name}{suf}.feather")
        return os.path.join(data_dir, f"{domain}_data_{split_name}{suf}.feather")

    elif kind == "outcomes":
        return os.path.join(data_dir, f"{domain}_outcomes_{split_name}{suf}.feather")

    else:
        raise ValueError(f"Invalid kind={kind}")


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_inputs_from_data(
    domain: str,
    *,
    split: int,
    data_dir: str = "./data",
    processed_root: str = "./data_rd_template",
    max_len: int = 800,
    max_offset: int = 48 * 60,
    use_aug: bool = False,
    aug_suffix: str = "aug",
):
    assert domain in {"mimic", "eicu"}, f"Invalid domain={domain}"

    # input folder: ./data/split{i}
    data_dir = split_data_dir(data_dir, split)

    # processed dir: augmentation setting까지 분리해서 캐시 충돌 방지
    aug_tag = _aug_tag(use_aug, aug_suffix)
    processed_dir = os.path.join(processed_root, f"split{split}", f"{domain}_{aug_tag}")
    os.makedirs(processed_dir, exist_ok=True)

    required_files = [
        f"ts_params_{domain}.npy",
        f"categorical_idx_{domain}.npy",
        f"static_params_{domain}.npy",
        f"PTdict_list_{domain}_train.npy",
        f"PTdict_list_{domain}_valid.npy",
        f"PTdict_list_{domain}_test.npy",
        f"arr_outcomes_{domain}_train.npy",
        f"arr_outcomes_{domain}_valid.npy",
        f"arr_outcomes_{domain}_test.npy",
        f"outcome_cols_{domain}.npy",
    ]

    missing = [fname for fname in required_files if not os.path.exists(os.path.join(processed_dir, fname))]
    if len(missing) == 0:
        print(
            f"⏭️  create_inputs_from_data skipped for {domain} split{split} "
            f"(aug={use_aug}, suffix={aug_suffix})"
        )
        return processed_dir
    else:
        print(
            f"⚠️  Rebuilding {domain} split{split} "
            f"(aug={use_aug}, suffix={aug_suffix}, missing {len(missing)} files)"
        )
        for m in missing:
            print(f"    - {m}")

    # ---- train data for feature params ----
    train_data_path = feather_path(
        data_dir, domain, "data", "train", split,
        use_aug=use_aug, aug_suffix=aug_suffix
    )
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Missing data feather: {train_data_path}")

    df_train = pd.read_feather(train_data_path)
    param_list = np.array(sorted(df_train["itemid"].unique()))
    F = len(param_list)
    np.save(os.path.join(processed_dir, f"ts_params_{domain}.npy"), param_list)

    # categorical/binary indices
    cat_mask = np.isin(param_list, np.array(categorical_itemids))
    cat_idx = np.where(cat_mask)[0].astype(np.int64)
    np.save(os.path.join(processed_dir, f"categorical_idx_{domain}.npy"), cat_idx)

    # ---- static (always original) ----
    static_train_path = os.path.join(data_dir, f"{domain}_data_static_train_{split}.feather")
    static_valid_path = os.path.join(data_dir, f"{domain}_data_static_valid_{split}.feather")
    static_test_path  = os.path.join(data_dir, f"{domain}_data_static_test_{split}.feather")

    for p in [static_train_path, static_valid_path, static_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing static feather: {p}")

    df_static_train = pd.read_feather(static_train_path)
    df_static_valid = pd.read_feather(static_valid_path)
    df_static_test  = pd.read_feather(static_test_path)

    df_static = pd.concat([df_static_train, df_static_valid, df_static_test], ignore_index=True)

    static_param_list = ["age", "gender", "height"]
    np.save(os.path.join(processed_dir, f"static_params_{domain}.npy"), np.array(static_param_list, dtype=object))

    d_static = 3
    static_dict = {}
    for _, row in df_static.iterrows():
        pid = row["pid"]
        static_dict[pid] = [row.get("age", 0), row.get("gender", 0), row.get("height", 0)]

    def build_ptdict(df: pd.DataFrame):
        PTdict_list = []

        for pid, group in df.groupby("pid"):
            group = group.sort_values(by=["offset", "itemid"])
            ts = group[["offset", "itemid", "value"]].values

            # 시간축은 unique offset 기준
            unq_offsets = []
            seen = set()
            for offset, _, _ in ts:
                offset = float(offset)
                if offset <= max_offset and offset not in seen:
                    unq_offsets.append(offset)
                    seen.add(offset)

            unq_offsets = np.array(unq_offsets, dtype=np.float32)
            length = len(unq_offsets)

            Parr = np.full((max_len, F), MISSING_VALUE, dtype=np.float32)
            Tarr = np.full((max_len, 1), MISSING_TIME, dtype=np.float32)

            if length > 0:
                # 빠른 조회용 dict
                offset_to_t = {float(off): i for i, off in enumerate(unq_offsets[:max_len])}
                item_to_f = {float(item): i for i, item in enumerate(param_list)}

                for offset, itemid, value in ts:
                    offset = float(offset)
                    itemid = float(itemid)

                    if offset > max_offset:
                        continue

                    t = offset_to_t.get(offset, None)
                    if t is None or t >= max_len:
                        continue

                    f = item_to_f.get(itemid, None)
                    if f is None:
                        continue

                    v = float(value)

                    if itemid in categorical_itemids:
                        if v == 0.0:
                            v = -1.0
                        elif v == 1.0:
                            v = 1.0

                    Parr[t, f] = v
                    Tarr[t, 0] = offset

            static_array = static_dict.get(pid, [0] * d_static)

            PTdict_list.append({
                "id": pid,
                "static": static_array,
                "extended_static": static_array,
                "arr": Parr,
                "time": Tarr,
                "length": int(min(length, max_len)),
            })

        return PTdict_list

    def load_outcomes(split_name: str, pids_in_order):
        out_path = feather_path(data_dir, domain, "outcomes", split_name, split)
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Missing outcomes feather: {out_path}")

        out = pd.read_feather(out_path)
        if "pid" in out.columns:
            out = out.set_index("pid")

        drop_cols = [c for c in out.columns if c in DROP_OUTCOME_COLS]
        if len(drop_cols) > 0:
            out = out.drop(columns=drop_cols)

        outcome_cols = list(out.columns)

        if isinstance(out.index, pd.Index) and out.index.name == "pid":
            y = out.loc[pids_in_order].to_numpy()
        else:
            y = out.to_numpy()

        return y.astype(np.float32), outcome_cols

    saved_cols = False
    for split_name in ["train", "valid", "test"]:
        df_path = feather_path(
            data_dir, domain, "data", split_name, split,
            use_aug=use_aug, aug_suffix=aug_suffix
        )
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Missing data feather: {df_path}")

        df = pd.read_feather(df_path)
        PTdict_list = build_ptdict(df)

        np.save(
            os.path.join(processed_dir, f"PTdict_list_{domain}_{split_name}.npy"),
            np.array(PTdict_list, dtype=object)
        )

        pids_in_order = [d["id"] for d in PTdict_list]
        y, cols = load_outcomes(split_name, pids_in_order)
        np.save(os.path.join(processed_dir, f"arr_outcomes_{domain}_{split_name}.npy"), y)

        if (not saved_cols) and (split_name == "train"):
            np.save(
                os.path.join(processed_dir, f"outcome_cols_{domain}.npy"),
                np.array(cols, dtype=object),
            )

            json_path = os.path.join(processed_dir, f"outcome_cols_{domain}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cols, f, ensure_ascii=False, indent=2)

            saved_cols = True

    meta = {
        "domain": domain,
        "split": int(split),
        "use_aug": bool(use_aug),
        "aug_suffix": str(aug_suffix),
        "processed_dir": processed_dir,
        "data_dir": data_dir,
    }
    with open(os.path.join(processed_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ create_inputs_from_data done for {domain} split{split}. Saved to {processed_dir}.")
    print(f"   - aug={use_aug}, aug_suffix={aug_suffix}")
    print(f"   - F={F}, binary categorical excluded from normalization: {len(cat_idx)}")
    return processed_dir


def prepare_data(
    *,
    processed_root: str = "./data_rd_template",
    split_source: int = 1,
    split_target_test: int = 1,
    source: str = "mimic",
    target: str = "eicu",
    use_source_aug: bool = False,
    source_aug_suffix: str = "aug",
    use_target_aug: bool = False,
    target_aug_suffix: str = "aug",
):
    assert 1 <= split_source <= 5
    assert 1 <= split_target_test <= 5
    assert source in {"mimic", "eicu"}
    assert target in {"mimic", "eicu"}
    assert source != target

    processed_dir_src = os.path.join(
        processed_root,
        f"split{split_source}",
        f"{source}_{_aug_tag(use_source_aug, source_aug_suffix)}",
    )
    processed_dir_tgt = os.path.join(
        processed_root,
        f"split{split_target_test}",
        f"{target}_{_aug_tag(use_target_aug, target_aug_suffix)}",
    )

    # ---- load source splits ----
    Ptrain = np.load(os.path.join(processed_dir_src, f"PTdict_list_{source}_train.npy"), allow_pickle=True)
    Pval   = np.load(os.path.join(processed_dir_src, f"PTdict_list_{source}_valid.npy"), allow_pickle=True)
    Ptest  = np.load(os.path.join(processed_dir_src, f"PTdict_list_{source}_test.npy"), allow_pickle=True)

    ytrain = np.load(os.path.join(processed_dir_src, f"arr_outcomes_{source}_train.npy"), allow_pickle=True).astype(np.float32)
    yval   = np.load(os.path.join(processed_dir_src, f"arr_outcomes_{source}_valid.npy"), allow_pickle=True).astype(np.float32)
    ytest  = np.load(os.path.join(processed_dir_src, f"arr_outcomes_{source}_test.npy"), allow_pickle=True).astype(np.float32)

    T, F_full = Ptrain[0]["arr"].shape
    D = len(Ptrain[0]["extended_static"])

    # categorical indices: SOURCE 기준
    cat_idx = np.load(os.path.join(processed_dir_src, f"categorical_idx_{source}.npy"), allow_pickle=True).astype(np.int64)
    cat_idx = cat_idx[(cat_idx >= 0) & (cat_idx < F_full)]

    # stats on source train only
    arr_stack = np.stack([p["arr"] for p in Ptrain])                 # [N,T,F]
    sta_stack = np.stack([p["extended_static"] for p in Ptrain])     # [N,D]
    mf, stdf = getStats_fixed(arr_stack, missing_value=MISSING_VALUE)
    ms, ss   = getStats_static(sta_stack, "Default")

    mf = mf.copy()
    stdf = stdf.copy()
    mf[cat_idx] = 0.0
    stdf[cat_idx] = 1.0

    def tz(P_list, y_arr):
        return tensorize_normalize_multilabel(
            P_list, y_arr, mf, stdf, ms, ss,
            missing_value=MISSING_VALUE,
            missing_time=MISSING_TIME,
        )

    # ---- tensorize source ----
    Ptrain_t, Ptrain_s, Ptrain_tm, ytrain_t = tz(Ptrain, ytrain)
    Pval_t,   Pval_s,   Pval_tm,   yval_t   = tz(Pval,   yval)
    Ptest_t,  Ptest_s,  Ptest_tm,  ytest_t  = tz(Ptest,  ytest)

    # ---- target test only ----
    Ptest_target = np.load(os.path.join(processed_dir_tgt, f"PTdict_list_{target}_test.npy"), allow_pickle=True)
    ytest_target = np.load(os.path.join(processed_dir_tgt, f"arr_outcomes_{target}_test.npy"), allow_pickle=True).astype(np.float32)
    Ptest_target_t, Ptest_target_s, Ptest_target_tm, ytest_target_t = tz(Ptest_target, ytest_target)

    def permute(X, TM):
        return X.permute(1, 0, 2), TM.squeeze(-1).permute(1, 0)

    Ptrain_t, Ptrain_tm = permute(Ptrain_t, Ptrain_tm)
    Pval_t,   Pval_tm   = permute(Pval_t,   Pval_tm)
    Ptest_t,  Ptest_tm  = permute(Ptest_t,  Ptest_tm)
    Ptest_target_t, Ptest_target_tm = permute(Ptest_target_t, Ptest_target_tm)

    source_dict = {
        "Ptrain": Ptrain_t, "Pval": Pval_t, "Ptest": Ptest_t,
        "Ptrain_time": Ptrain_tm, "Pval_time": Pval_tm, "Ptest_time": Ptest_tm,
        "Ptrain_static": Ptrain_s, "Pval_static": Pval_s, "Ptest_static": Ptest_s,
        "ytrain": ytrain_t, "yval": yval_t, "ytest": ytest_t,
        "shape": {"T": T, "F": F_full, "D": D},
    }

    C = ytrain_t.shape[1]
    empty_P  = torch.zeros((T, 0, Ptest_target_t.shape[2]), dtype=Ptest_target_t.dtype)
    empty_TM = torch.zeros((T, 0), dtype=Ptest_target_tm.dtype)
    empty_S  = torch.zeros((0, D), dtype=Ptest_target_s.dtype)
    empty_Y  = torch.zeros((0, C), dtype=ytest_target_t.dtype)

    target_dict = {
        "Ptrain": empty_P, "Pval": empty_P, "Ptest": Ptest_target_t,
        "Ptrain_time": empty_TM, "Pval_time": empty_TM, "Ptest_time": Ptest_target_tm,
        "Ptrain_static": empty_S, "Pval_static": empty_S, "Ptest_static": Ptest_target_s,
        "ytrain": empty_Y, "yval": empty_Y, "ytest": ytest_target_t,
        "shape": {"T": T, "F": F_full, "D": D},
    }

    outcome_cols = np.load(os.path.join(processed_dir_src, f"outcome_cols_{source}.npy"), allow_pickle=True).tolist()
    source_dict["outcome_cols"] = outcome_cols
    target_dict["outcome_cols"] = outcome_cols

    return source_dict, target_dict
