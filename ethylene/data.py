"""Shared data layer: dataset container, scaler, firmware-format CSV, windows, splits.

The CSV format is the firmware's log line (`CSV_COLUMNS`), optionally followed by a
`label` column (training target) and, for multi-run files written by `save_csv`, a
`run` column. `adc_mq` and `idx` are not part of a `Dataset`: `load_csv` ignores them.
"""
import csv
from dataclasses import dataclass

import numpy as np

from ethylene.physics import FEATURE_NAMES, SensorConfig

CSV_COLUMNS = ("t_ms", "adc_mq", "rs_r0", "c_comp", "ldr_raw", "ldr_blk", "temp_c", "rh", "idx", "flags")

FLAG_WARMUP = 1
FLAG_DHT_FAIL = 2
FLAG_MQ_FAULT = 4
FLAG_VENT = 8
FLAG_SCRUB = 16

STD_FLOOR = 1e-6


@dataclass
class Dataset:
    """Row-aligned arrays: X (N,6) float32 in FEATURE_NAMES order, y (N,) float32 or None,
    run (N,) int32, t_s (N,) float32, flags (N,) int32."""

    X: np.ndarray
    y: np.ndarray | None
    run: np.ndarray
    t_s: np.ndarray
    flags: np.ndarray

    def _take(self, mask) -> "Dataset":
        return Dataset(
            X=self.X[mask],
            y=None if self.y is None else self.y[mask],
            run=self.run[mask],
            t_s=self.t_s[mask],
            flags=self.flags[mask],
        )


class Scaler:
    """Per-feature standardiser. NaN cells are ignored when fitting and stay NaN."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X) -> "Scaler":
        X = np.asarray(X)
        self.mean = np.nanmean(X, axis=0, dtype=np.float64)
        self.std = np.maximum(np.nanstd(X, axis=0, dtype=np.float64), STD_FLOOR)
        return self

    def transform(self, X) -> np.ndarray:
        """Standardise over the last axis, so (N,6) rows and (M,W,6) windows both work."""
        if self.mean is None:
            raise RuntimeError("Scaler is not fitted")
        return ((np.asarray(X) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


def save_csv(ds: Dataset, path) -> None:
    """Write `ds` as one CSV line per sample.

    `adc_mq` is cosmetic: the MQ divider inverted with R0 = RL, rounded and kept in
    [0, adc_max] (NaN rs_r0 writes 0). `idx` is always -1. A `label` column is added
    when `ds.y` is set and a `run` column when there is more than one run.
    """
    cfg = SensorConfig()
    rs = ds.X[:, FEATURE_NAMES.index("rs_r0")].astype(np.float64) * cfg.rl_ohm
    with np.errstate(divide="ignore", invalid="ignore"):
        adc = cfg.adc_max * cfg.rl_ohm / (rs + cfg.rl_ohm)
    adc = np.rint(np.nan_to_num(np.clip(adc, 0, cfg.adc_max), nan=0.0)).astype(np.int64)
    t_ms = np.rint(ds.t_s.astype(np.float64) * 1000).astype(np.int64)

    with_label = ds.y is not None
    with_run = np.unique(ds.run).size > 1
    header = list(CSV_COLUMNS) + (["label"] if with_label else []) + (["run"] if with_run else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for i in range(len(ds.X)):
            row = [t_ms[i], adc[i], *ds.X[i], -1, ds.flags[i]]
            if with_label:
                row.append(ds.y[i])
            if with_run:
                row.append(ds.run[i])
            w.writerow(row)


def _parse(cell: str, name: str, line: int, kind=float):
    try:
        return kind(cell)
    except ValueError:
        raise ValueError(
            f"line {line}: column '{name}': cannot parse {cell!r} as {kind.__name__}"
        ) from None


def load_csv(path) -> Dataset:
    """Read a firmware or `save_csv` file. Columns are found by header name.

    Raises ValueError naming a missing column, or the 1-based line number (header is
    line 1) of a row with a non-numeric cell or the wrong number of fields.
    `adc_mq` and `idx` are ignored; `nan` and `idx == -1` are accepted. `label` and
    `run` are read when present; without `run` every row gets run id 0.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path}: empty file, expected a header line") from None
        missing = [c for c in CSV_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")
        col = {name: i for i, name in enumerate(header)}

        t_ms, X, y, run, flags = [], [], [], [], []
        for row in reader:
            if not row:
                continue  # blank line
            line = reader.line_num
            if len(row) != len(header):
                raise ValueError(f"line {line}: expected {len(header)} fields, got {len(row)}")
            t_ms.append(_parse(row[col["t_ms"]], "t_ms", line))
            X.append([_parse(row[col[n]], n, line) for n in FEATURE_NAMES])
            flags.append(_parse(row[col["flags"]], "flags", line, int))
            if "label" in col:
                y.append(_parse(row[col["label"]], "label", line))
            if "run" in col:
                run.append(_parse(row[col["run"]], "run", line, int))

    n = len(t_ms)
    return Dataset(
        X=np.array(X, dtype=np.float32).reshape(n, len(FEATURE_NAMES)),
        y=np.array(y, dtype=np.float32) if "label" in col else None,
        run=np.array(run, dtype=np.int32) if "run" in col else np.zeros(n, dtype=np.int32),
        t_s=(np.array(t_ms, dtype=np.float64) / 1000.0).astype(np.float32),
        flags=np.array(flags, dtype=np.int32),
    )


def make_windows(ds: Dataset, width: int):
    """Sliding windows of `width` consecutive rows, never crossing runs.

    Returns (W (M,width,6) float32, y (M,) float32, run (M,) int32); y is the label of
    the window's last row. Rows are taken in array order within each run (no sorting);
    runs appear in order of first appearance; runs shorter than `width` give none.
    """
    if ds.y is None:
        raise ValueError("make_windows needs labels, but ds.y is None")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    ws, ys, runs = [], [], []
    _, first = np.unique(ds.run, return_index=True)
    for r in ds.run[np.sort(first)]:
        idx = np.flatnonzero(ds.run == r)
        m = len(idx) - width + 1
        if m <= 0:
            continue
        view = np.lib.stride_tricks.sliding_window_view(ds.X[idx], width, axis=0)  # (m, 6, width)
        ws.append(view.transpose(0, 2, 1))
        ys.append(ds.y[idx[width - 1 :]])
        runs.append(np.full(m, r, dtype=np.int32))
    if not ws:
        return (
            np.empty((0, width, ds.X.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return (
        np.concatenate(ws).astype(np.float32, copy=False),
        np.concatenate(ys).astype(np.float32, copy=False),
        np.concatenate(runs),
    )


def split_by_run(ds: Dataset, val_frac: float, test_frac: float, seed: int):
    """Split into (train, val, test) Datasets disjoint by run id.

    Run ids are shuffled with `np.random.default_rng(seed)` and dealt out whole; rows
    keep their original order. With n_runs >= 3 every split gets at least one run
    (counts are max(1, round(n * frac)), trimmed so train keeps one); with fewer runs
    the split is best-effort and some of val/test may be empty.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError(f"need val_frac, test_frac >= 0 and val_frac + test_frac < 1, got {val_frac}, {test_frac}")
    ids = np.random.default_rng(seed).permutation(np.unique(ds.run))
    n = len(ids)
    n_test = min(max(1, round(n * test_frac)), max(n - 2, 0))
    n_val = min(max(1, round(n * val_frac)), max(n - 1 - n_test, 0))
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return tuple(ds._take(np.isin(ds.run, part)) for part in (train_ids, val_ids, test_ids))
