# ConstructImage.py
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

"""
Requirements:
1) Generate images for:
   - SOURCE domain: train / valid / test from split{source_split}
   - TARGET domain: test from split1 (fixed)
   in a single run.

2) Scaling (y-limits) computed ONLY from SOURCE train (split{source_split}) and reused for all.

3) Output path:
   - source original:
       out_dir/split{source_split}/source_<source>_images/
   - source augmented:
       out_dir/split{source_split}/source_<source>_aug_images/

   Inside that:
   - source train/valid/test:
       <source>_train_images / <source>_valid_images / <source>_test_images
   - target test:
       <target>_test_images          (target original)
       <target>_test_aug_images      (target augmented)

4) Metadata saved:
   out_dir/split{source_split}/source_<source>[_aug]_images/ImageDict_list_<domain>_<split>.npy
   For target augmented test, metadata filename is:
       ImageDict_list_<target>_test_aug.npy

Input conventions:
- Processed files exist under processed_root/split{k}/{domain}_{orig|aug_suffix}/:
    PTdict_list_<domain>_<split>.npy
    arr_outcomes_<domain>_<split>.npy
    outcome_cols_<domain>.json (optional)
- Missing value in arr is 0.0
- Time padding is -1.0 minutes in PTdict['time']
"""

# ----------------------------
# Your 35 variables
# ----------------------------
var_mappings = {
    "gcs": 0,  # vital
    "heart_rate": 1,
    "map": 2,
    "resp_rate": 3,
    "temperature": 4,
    "weight": 5,
    "albumin": 6,  # lab
    "bilirubin": 7,
    "creatinine": 8,
    "fio2": 9,
    "glucose": 10,
    "hematocrit": 11,
    "lactate": 12,
    "pao2": 13,
    "ph": 14,
    "platelets": 15,
    "sodium": 16,
    "urine": 17,
    "wbc": 18,
    "a10": 19,  # treatments
    "a_drug": 20,
    "a_supplements": 21,
    "b": 22,
    "c01": 23,
    "c01_etc": 24,
    "c_else": 25,
    "h": 26,
    "l": 27,
    "m": 28,
    "n": 29,
    "r": 30,
    "v": 31,
    "antibiotic": 32,
    "fluid": 33,
    "ventilator": 34,
}

ts_params: List[str] = [None] * len(var_mappings)
for k, v in var_mappings.items():
    ts_params[v] = k

MISSING_VALUE = 0.0
MISSING_TIME_MIN = -1.0


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _aug_name(use_aug: bool, base: str) -> str:
    return f"{base}_aug" if use_aug else base


def _processed_domain_dir(domain: str, use_aug: bool, aug_suffix: str) -> str:
    tag = aug_suffix if use_aug else "orig"
    return f"{domain}_{tag}"


def _metadata_suffix(split: str, use_aug: bool) -> str:
    if split == "test" and use_aug:
        return "test_aug"
    return split


def _as_1d_time_minutes(TT) -> np.ndarray:
    TT = np.asarray(TT, dtype=np.float32)
    if TT.ndim == 2 and TT.shape[1] == 1:
        TT = TT[:, 0]
    return TT


def _valid_length_from_time_minutes(tt_min: np.ndarray) -> int:
    pad_pos = np.where(tt_min == MISSING_TIME_MIN)[0]
    return int(pad_pos[0]) if pad_pos.size > 0 else int(tt_min.shape[0])


def construct_demogr_description(static_demogr, length_hours: float) -> str:
    if static_demogr is None:
        return ""

    static_demogr = np.asarray(static_demogr).astype(float).tolist()
    desc = []

    if len(static_demogr) > 0 and static_demogr[0] not in [None, 0, 0.0]:
        desc.append(f"{int(static_demogr[0])} years old")

    if len(static_demogr) > 1:
        try:
            g = int(static_demogr[1])
            if g == 0:
                desc.append("female")
            elif g == 1:
                desc.append("male")
        except Exception:
            pass

    if len(static_demogr) > 2 and static_demogr[2] not in [None, 0, 0.0]:
        desc.append(f"height {int(static_demogr[2])} cm")

    desc.append(f"has physiological data within {length_hours:.1f} hours")
    return "A patient is " + ", ".join(desc) + "."


def _make_colors(num_vars: int):
    base = plt.get_cmap("tab20").colors
    return [base[i % len(base)] for i in range(num_vars)]


def _load_outcome_cols(processed_dir: str, domain: str) -> Optional[List[str]]:
    jpath = os.path.join(processed_dir, f"outcome_cols_{domain}.json")
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            cols = json.load(f)
        return cols

    npypath = os.path.join(processed_dir, f"outcome_cols_{domain}.npy")
    if os.path.exists(npypath):
        cols = np.load(npypath, allow_pickle=True).tolist()
        return cols

    return None


def _compute_scales_from_train(Ptrain_list, num_vars: int, outlier: Optional[str]):
    all_vals: List[List[np.ndarray]] = [[] for _ in range(num_vars)]

    for p in tqdm(Ptrain_list, desc="Collect source-train stats", leave=False):
        X = np.asarray(p["arr"], dtype=np.float32)
        tt = _as_1d_time_minutes(p["time"])
        L = _valid_length_from_time_minutes(tt)
        X = X[:L]

        for f in range(num_vars):
            v = X[:, f]
            v = v[v != MISSING_VALUE]
            if v.size:
                all_vals[f].append(v)

    scales = np.zeros((num_vars, 2), dtype=np.float32)

    for f in range(num_vars):
        if len(all_vals[f]) == 0:
            scales[f] = np.array([-0.5, 0.5], dtype=np.float32)
            continue

        vals = np.concatenate(all_vals[f]).astype(np.float32)
        if vals.size == 0:
            scales[f] = np.array([-0.5, 0.5], dtype=np.float32)
            continue

        if outlier is None:
            lo, hi = float(vals.min()), float(vals.max())
        elif outlier == "iqr":
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            lo = float(q1 - 1.5 * iqr)
            hi = float(q3 + 1.5 * iqr)
        elif outlier == "sd":
            med = float(np.median(vals))
            sd = float(np.std(vals))
            lo = med - 3.0 * sd
            hi = med + 3.0 * sd
        elif outlier == "mzs":
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            lo = float((-3.5 / 0.6745) * mad + med)
            hi = float((3.5 / 0.6745) * mad + med)
        else:
            raise ValueError(f"Unknown outlier mode: {outlier}")

        if lo == hi:
            lo -= 0.5
            hi += 0.5

        scales[f] = np.array([lo, hi], dtype=np.float32)

    return scales


def _compute_order_by_density_source_train(Ptrain_list, num_vars: int) -> List[int]:
    counts = np.zeros((num_vars,), dtype=np.int64)
    for p in Ptrain_list:
        X = np.asarray(p["arr"], dtype=np.float32)
        tt = _as_1d_time_minutes(p["time"])
        L = _valid_length_from_time_minutes(tt)
        X = X[:L]
        counts += (X != MISSING_VALUE).sum(axis=0)
    return list(np.argsort(counts)[::-1].tolist())


def draw_image(
    *,
    pid: str,
    domain: str,
    split: str,
    out_base_dir: str,
    split_dir_name: Optional[str] = None,
    ts_orders: List[int],
    ts_values: np.ndarray,
    ts_times_min: np.ndarray,
    ts_scales: np.ndarray,
    override: bool,
    interpolation: bool,
    image_size: Optional[Tuple[int, int]],
    grid_layout: Tuple[int, int],
    linestyle: str,
    linewidth: float,
    marker: str,
    markersize: float,
    ts_color_mapping: Dict[str, Tuple[float, float, float, float]],
) -> str:
    grid_h, grid_w = grid_layout
    if image_size is None:
        cell_h, cell_w = 64, 64
        img_h, img_w = grid_h * cell_h, grid_w * cell_w
    else:
        img_h, img_w = image_size

    dpi = 100
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.figsize"] = (img_w / dpi, img_h / dpi)
    plt.rcParams["figure.frameon"] = False

    if split_dir_name is None:
        split_dir_name = f"{domain}_{split}_images"

    split_dir = os.path.join(out_base_dir, split_dir_name)
    _safe_mkdir(split_dir)

    img_path = os.path.join(split_dir, f"{pid}.png")
    if os.path.exists(img_path) and (not override):
        return img_path

    ts_times_hr = ts_times_min / 60.0
    L = _valid_length_from_time_minutes(ts_times_min)
    ts_times_hr = ts_times_hr[:L]
    ts_values = ts_values[:L]

    x_max = float(np.nanmax(ts_times_hr)) if ts_times_hr.size else 1.0
    x_min = 0.0

    plt.figure()

    for subplot_i, f_idx in enumerate(ts_orders):
        param = ts_params[f_idx]
        v = ts_values[:, f_idx].astype(np.float32)

        lo, hi = float(ts_scales[f_idx, 0]), float(ts_scales[f_idx, 1])
        keep = (v != MISSING_VALUE) & (v >= lo) & (v <= hi)
        t = ts_times_hr.copy()

        if interpolation:
            t_plot = t[keep]
            v_plot = v[keep]
        else:
            v_plot = v.copy()
            t_plot = t.copy()
            v_plot[~keep] = np.nan
            t_plot[~keep] = np.nan

        plt.subplot(grid_h, grid_w, subplot_i + 1)
        plt.plot(
            t_plot,
            v_plot,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            marker=marker,
            color=ts_color_mapping[param],
        )

        plt.xlim([x_min, max(1e-6, x_max)])
        plt.ylim([lo, hi])
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_path, pad_inches=0)
    plt.close()

    return img_path


def _load_pt_and_y(processed_dir: str, domain: str, split: str):
    p_path = os.path.join(processed_dir, f"PTdict_list_{domain}_{split}.npy")
    y_path = os.path.join(processed_dir, f"arr_outcomes_{domain}_{split}.npy")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    if not os.path.exists(y_path):
        raise FileNotFoundError(y_path)

    P = np.load(p_path, allow_pickle=True)
    Y = np.load(y_path, allow_pickle=True)
    return P, Y


def construct_source_target_images(
    *,
    source_domain: str,
    target_domain: str,
    processed_root: str,
    source_split: int,
    out_dir: str,

    use_source_aug: bool = False,
    source_aug_suffix: str = "aug",
    use_target_aug: bool = False,
    target_aug_suffix: str = "aug",

    linestyle: str = "-",
    linewidth: float = 1.0,
    marker: str = "*",
    markersize: float = 2.0,

    override: bool = False,
    outlier: Optional[str] = None,        # None | "iqr" | "sd" | "mzs"
    interpolation: bool = True,
    order_by_density: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
    grid_layout: Optional[Tuple[int, int]] = None,
):
    assert source_domain in {"mimic", "eicu"}
    assert target_domain in {"mimic", "eicu"}
    assert source_domain != target_domain
    assert source_split in {1, 2, 3, 4, 5}
    assert outlier in {None, "iqr", "sd", "mzs"}

    num_vars = len(ts_params)
    assert num_vars == 35

    if grid_layout is None:
        grid_layout = (5, 7)
    grid_h, grid_w = grid_layout
    if grid_h * grid_w < num_vars:
        raise ValueError(f"grid_layout {grid_layout} too small for {num_vars} variables")

    # processed dirs
    src_proc = os.path.join(
        processed_root,
        f"split{source_split}",
        _processed_domain_dir(source_domain, use_source_aug, source_aug_suffix),
    )
    tgt_proc = os.path.join(
        processed_root,
        "split1",
        _processed_domain_dir(target_domain, use_target_aug, target_aug_suffix),
    )

    # load SOURCE splits from split{source_split}
    P_src_train, Y_src_train = _load_pt_and_y(src_proc, source_domain, "train")
    P_src_valid, Y_src_valid = _load_pt_and_y(src_proc, source_domain, "valid")
    P_src_test,  Y_src_test  = _load_pt_and_y(src_proc, source_domain, "test")

    # load TARGET test from split1
    P_tgt_test,  Y_tgt_test  = _load_pt_and_y(tgt_proc, target_domain, "test")

    # outcome cols
    outcome_cols_src = _load_outcome_cols(src_proc, source_domain)
    if outcome_cols_src is None:
        outcome_cols_src = _load_outcome_cols(tgt_proc, source_domain)

    if outcome_cols_src is None:
        C = int(np.asarray(Y_src_train).shape[1]) if np.asarray(Y_src_train).ndim == 2 else 1
        outcome_cols_src = [f"task_{i}" for i in range(C)]

    outcome_cols_tgt = _load_outcome_cols(tgt_proc, target_domain)
    if outcome_cols_tgt is None:
        outcome_cols_tgt = outcome_cols_src

    # compute scales + ordering from SOURCE train only
    ts_scales = _compute_scales_from_train(
        Ptrain_list=P_src_train,
        num_vars=num_vars,
        outlier=outlier,
    )

    ts_orders = (
        _compute_order_by_density_source_train(P_src_train, num_vars)
        if order_by_density else list(range(num_vars))
    )

    # color mapping
    colors = _make_colors(num_vars)
    ts_color_mapping = {ts_params[i]: colors[i] for i in range(num_vars)}
    ts_idx_mapping = {ts_params[i]: i for i in range(num_vars)}

    # output base dir includes source + source aug status
    source_folder_name = f"source_{_aug_name(use_source_aug, source_domain)}_images"
    out_base_dir = os.path.join(out_dir, f"split{source_split}", source_folder_name)
    _safe_mkdir(out_base_dir)

    # save metadata
    run_meta = {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "source_split": int(source_split),
        "target_test_split": 1,
        "use_source_aug": bool(use_source_aug),
        "source_aug_suffix": source_aug_suffix,
        "use_target_aug": bool(use_target_aug),
        "target_aug_suffix": target_aug_suffix,
        "outlier": outlier,
        "interpolation": bool(interpolation),
        "order_by_density": bool(order_by_density),
        "grid_layout": list(grid_layout),
        "image_size": None if image_size is None else list(image_size),
        "source_processed_dir": src_proc,
        "target_processed_dir": tgt_proc,
        "output_base_dir": out_base_dir,
    }
    with open(os.path.join(out_base_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    with open(os.path.join(out_base_dir, "param_idx_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(ts_idx_mapping, f, indent=2)
    with open(os.path.join(out_base_dir, "param_color_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({k: list(map(float, v)) for k, v in ts_color_mapping.items()}, f, indent=2)
    with open(os.path.join(out_base_dir, "param_list.json"), "w", encoding="utf-8") as f:
        json.dump(ts_params, f, indent=2)
    with open(os.path.join(out_base_dir, "scales_source_train.json"), "w", encoding="utf-8") as f:
        json.dump(ts_scales.tolist(), f)
    with open(os.path.join(out_base_dir, "var_order_source_train.json"), "w", encoding="utf-8") as f:
        json.dump(ts_orders, f, indent=2)

    with open(os.path.join(out_base_dir, f"outcome_cols_{source_domain}.json"), "w", encoding="utf-8") as f:
        json.dump(outcome_cols_src, f, indent=2)
    with open(os.path.join(out_base_dir, f"outcome_cols_{target_domain}.json"), "w", encoding="utf-8") as f:
        json.dump(outcome_cols_tgt, f, indent=2)

    src_train_dir_name = f"{source_domain}_train_images"
    src_valid_dir_name = f"{source_domain}_valid_images"
    src_test_dir_name  = f"{source_domain}_test_images"
    tgt_test_dir_name  = f"{target_domain}_test_aug_images" if use_target_aug else f"{target_domain}_test_images"

    jobs = [
        (source_domain, "train", src_train_dir_name, False, P_src_train, Y_src_train, outcome_cols_src),
        (source_domain, "valid", src_valid_dir_name, False, P_src_valid, Y_src_valid, outcome_cols_src),
        (source_domain, "test",  src_test_dir_name,  False, P_src_test,  Y_src_test,  outcome_cols_src),
        (target_domain, "test",  tgt_test_dir_name,  use_target_aug, P_tgt_test, Y_tgt_test, outcome_cols_tgt),
    ]

    for dom, sp, split_dir_name, is_aug_job, P_list, Y_arr, out_cols in jobs:
        Y_arr = np.asarray(Y_arr)
        ImageDict_list = []

        print(
            f"Generating images | src={source_domain} (split{source_split}, source_aug={use_source_aug}) "
            f"| dom={dom} sp={sp} aug={is_aug_job} "
            f"| tgt_test_split=1 "
            f"| N={len(P_list)}"
        )

        for i, p in tqdm(enumerate(P_list), total=len(P_list), desc=f"Draw {dom}:{sp}", leave=False):
            pid_raw = str(p.get("id", f"idx_{i}"))
            pid = pid_raw.split("\\")[-1].split("/")[-1].replace(".psv", "").strip()

            X = np.asarray(p["arr"], dtype=np.float32)
            tt_min = _as_1d_time_minutes(p["time"])
            L = _valid_length_from_time_minutes(tt_min)
            length_hours = float(np.nanmax(tt_min[:L]) / 60.0) if L > 0 else 0.0

            static_demogr = p.get("extended_static", None)
            demogr_desc = construct_demogr_description(static_demogr, length_hours)

            img_path = draw_image(
                pid=pid,
                domain=dom,
                split=sp,
                out_base_dir=out_base_dir,
                split_dir_name=split_dir_name,
                ts_orders=ts_orders,
                ts_values=X,
                ts_times_min=tt_min,
                ts_scales=ts_scales,
                override=override,
                interpolation=interpolation,
                image_size=image_size,
                grid_layout=grid_layout,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                ts_color_mapping=ts_color_mapping,
            )

            label = int(Y_arr[i]) if Y_arr.ndim == 1 else Y_arr[i].astype(float).tolist()

            ImageDict_list.append(
                {
                    "id": pid,
                    "id_raw": pid_raw,
                    "split": sp,
                    "split_dir_name": split_dir_name,
                    "domain": dom,
                    "is_augmented": bool(is_aug_job),
                    "source_split_for_source_domain": int(source_split) if dom == source_domain else 1,
                    "source_domain_for_scaling": source_domain,
                    "text": demogr_desc,
                    "image_path": img_path,
                    "label": label,
                    "outcome_cols": out_cols,
                    "param_num": num_vars,
                    "param_list": ts_params,
                }
            )

        meta_suffix = _metadata_suffix(sp, is_aug_job)
        out_path = os.path.join(out_base_dir, f"ImageDict_list_{dom}_{meta_suffix}.npy")
        np.save(out_path, np.array(ImageDict_list, dtype=object))
        print(f"✅ Saved metadata: {out_path} (N={len(ImageDict_list)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="mimic", choices=["mimic", "eicu"])
    parser.add_argument("--target", type=str, default=None, choices=["mimic", "eicu"])

    parser.add_argument(
        "--processed_root",
        type=str,
        default="../data_rd_template",
        help="Root containing split{1..5}/{domain}_{orig|aug_suffix}/ with PTdict_list_*.npy and arr_outcomes_*.npy",
    )
    parser.add_argument("--source_split", type=int, default=1, choices=[1, 2, 3, 4, 5])

    parser.add_argument(
        "--out_dir",
        type=str,
        default="../data_rd_template",
        help="Root output directory. Will create out_dir/split{source_split}/source_<source>[_aug]_images/...",
    )

    # aug args
    parser.add_argument("--use_source_aug", action="store_true")
    parser.add_argument("--source_aug_suffix", type=str, default="aug")
    parser.add_argument("--use_target_aug", action="store_true")
    parser.add_argument("--target_aug_suffix", type=str, default="aug")

    # plotting style
    parser.add_argument("--linestyle", type=str, default="-")
    parser.add_argument("--linewidth", type=float, default=1.0)
    parser.add_argument("--marker", type=str, default="*")
    parser.add_argument("--markersize", type=float, default=2.0)

    parser.add_argument("--override", action="store_true")
    parser.add_argument("--outlier", type=str, default="none", choices=["none", "iqr", "sd", "mzs"])
    parser.add_argument("--no_interpolation", action="store_true")
    parser.add_argument("--no_order_by_density", action="store_true")

    parser.add_argument("--grid_h", type=int, default=5)
    parser.add_argument("--grid_w", type=int, default=7)

    parser.add_argument("--img_h", type=int, default=0)
    parser.add_argument("--img_w", type=int, default=0)

    args = parser.parse_args()

    if args.target is None:
        args.target = "eicu" if args.source == "mimic" else "mimic"
    if args.target == args.source:
        raise ValueError("target must be different from source")

    outlier = None if args.outlier == "none" else args.outlier
    interpolation = not args.no_interpolation
    order_by_density = not args.no_order_by_density

    image_size = None
    if args.img_h > 0 and args.img_w > 0:
        image_size = (args.img_h, args.img_w)

    construct_source_target_images(
        source_domain=args.source,
        target_domain=args.target,
        processed_root=args.processed_root,
        source_split=args.source_split,
        out_dir=args.out_dir,

        use_source_aug=args.use_source_aug,
        source_aug_suffix=args.source_aug_suffix,
        use_target_aug=args.use_target_aug,
        target_aug_suffix=args.target_aug_suffix,

        linestyle=args.linestyle,
        linewidth=args.linewidth,
        marker=args.marker,
        markersize=args.markersize,
        override=args.override,
        outlier=outlier,
        interpolation=interpolation,
        order_by_density=order_by_density,
        image_size=image_size,
        grid_layout=(args.grid_h, args.grid_w),
    )


if __name__ == "__main__":
    main()