import json

import numpy as np
import pytest

from ethylene.data import (
    CSV_COLUMNS,
    FLAG_DHT_FAIL,
    FLAG_MQ_FAULT,
    FLAG_SCRUB,
    FLAG_VENT,
    FLAG_WARMUP,
    Dataset,
    Scaler,
    load_csv,
    make_windows,
    save_csv,
    split_by_run,
)
from ethylene.physics import FEATURE_NAMES, SensorConfig

HEADER = ",".join(CSV_COLUMNS)


def make_ds(run_lengths, labelled=True, seed=0):
    """Dataset whose X[:, 0] and y encode the global row index, other features random."""
    rng = np.random.default_rng(seed)
    n = sum(run_lengths)
    X = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    X[:, 0] = np.arange(n)
    run = np.repeat(np.arange(len(run_lengths)), run_lengths).astype(np.int32)
    t_s = np.concatenate([np.arange(k) * 0.25 for k in run_lengths]).astype(np.float32)
    flags = (np.arange(n) % 4).astype(np.int32)
    y = (np.arange(n) * 10).astype(np.float32) if labelled else None
    return Dataset(X=X, y=y, run=run, t_s=t_s, flags=flags)


def write(tmp_path, text, name="d.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_constants():
    assert CSV_COLUMNS == (
        "t_ms", "adc_mq", "rs_r0", "c_comp", "ldr_raw", "ldr_blk",
        "temp_c", "rh", "idx", "flags",
    )
    assert (FLAG_WARMUP, FLAG_DHT_FAIL, FLAG_MQ_FAULT, FLAG_VENT, FLAG_SCRUB) == (1, 2, 4, 8, 16)


# ---- windows -----------------------------------------------------------------


def test_windows_never_cross_runs():
    ds = make_ds([5, 5])
    W, y, run = make_windows(ds, 3)
    assert W.shape == (6, 3, 6)
    assert y.shape == (6,) and run.shape == (6,)
    assert W.dtype == np.float32 and y.dtype == np.float32 and run.dtype == np.int32
    rows = W[:, :, 0].astype(int)  # global row indices contained in each window
    assert (np.diff(rows, axis=1) == 1).all()  # consecutive rows
    for i in range(6):
        assert (ds.run[rows[i]] == run[i]).all()  # one run id per window
    assert run.tolist() == [0, 0, 0, 1, 1, 1]
    # y comes from the last row of each window
    assert rows[:, -1].tolist() == [2, 3, 4, 7, 8, 9]
    np.testing.assert_array_equal(y, ds.y[rows[:, -1]])


def test_windows_skip_runs_shorter_than_width():
    ds = make_ds([2, 4])
    W, y, run = make_windows(ds, 3)
    assert W.shape == (2, 3, 6)
    assert run.tolist() == [1, 1]


def test_windows_when_nothing_fits_returns_empty_arrays():
    W, y, run = make_windows(make_ds([2, 2]), 3)
    assert W.shape == (0, 3, 6) and y.shape == (0,) and run.shape == (0,)


def test_windows_require_labels():
    with pytest.raises(ValueError, match="label"):
        make_windows(make_ds([5], labelled=False), 3)


# ---- split -------------------------------------------------------------------


def test_split_by_run_is_disjoint():
    ds = make_ds([4] * 10)
    train, val, test = split_by_run(ds, 0.2, 0.2, seed=7)
    ids = [set(np.unique(d.run).tolist()) for d in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert ids[0] | ids[1] | ids[2] == set(range(10))
    assert [len(i) for i in ids] == [6, 2, 2]
    assert sum(len(d.X) for d in (train, val, test)) == len(ds.X)
    for d in (train, val, test):  # every field stays aligned with X
        n = len(d.X)
        assert len(d.y) == len(d.run) == len(d.t_s) == len(d.flags) == n


def test_split_by_run_is_seeded_and_keeps_row_order():
    ds = make_ds([3] * 8)
    a = split_by_run(ds, 0.25, 0.25, seed=1)
    b = split_by_run(ds, 0.25, 0.25, seed=1)
    c = split_by_run(ds, 0.25, 0.25, seed=2)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.run, y.run)
    assert any(not np.array_equal(x.run, z.run) for x, z in zip(a, c))
    for d in a:  # global row index (column 0) stays ascending: original order kept
        assert (np.diff(d.X[:, 0]) > 0).all()


@pytest.mark.parametrize(
    "n_runs, fracs",
    [
        (3, (0.2, 0.2)),
        (3, (0.0, 0.0)),
        (3, (0.45, 0.45)),
        (3, (0.05, 0.9)),  # test alone would round to all 3 runs
        (4, (0.4, 0.4)),
        (4, (0.49, 0.49)),
    ],
)
def test_split_by_run_fills_every_split_from_three_runs(n_runs, fracs):
    # (4, 0.4/0.4) would round to 2 + 2 runs and starve train without the trim
    train, val, test = split_by_run(make_ds([3] * n_runs), *fracs, seed=0)
    assert all(len(d.X) > 0 for d in (train, val, test))


def test_split_by_run_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_by_run(make_ds([3] * 4), 0.6, 0.5, seed=0)
    with pytest.raises(ValueError):
        split_by_run(make_ds([3] * 4), -0.1, 0.2, seed=0)


def test_split_by_run_keeps_missing_labels_missing():
    train, val, test = split_by_run(make_ds([2] * 5, labelled=False), 0.2, 0.2, seed=0)
    assert train.y is None and val.y is None and test.y is None


# ---- scaler ------------------------------------------------------------------


def test_scaler_roundtrip():
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(500, 6)) * [1, 5, 100, 3, 10, 0.1] + [0, 10, 500, 20, 25, 50]).astype(
        np.float32
    )
    sc = Scaler().fit(X)
    d = sc.to_dict()
    sc2 = Scaler.from_dict(json.loads(json.dumps(d)))  # survives JSON
    Z = sc.transform(X)
    np.testing.assert_array_equal(Z, sc2.transform(X))
    assert Z.dtype == np.float32
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-4)
    assert sc.mean.shape == sc.std.shape == (6,)


def test_scaler_std_floor_and_nan_tolerance():
    X = np.array([[1, 5, 2, 7, 3, 4], [1, np.nan, 4, 7, 3, 6], [1, 9, 6, 7, 3, 5]], dtype=np.float32)
    sc = Scaler().fit(X)
    assert sc.std[0] == 1e-6 and sc.std[3] == 1e-6  # constant columns floored
    assert np.isfinite(sc.mean).all() and np.isfinite(sc.std).all()  # NaN cell ignored
    Z = sc.transform(X)
    assert (Z[:, 0] == 0).all()
    assert np.isnan(Z[1, 1])  # NaN stays NaN


def test_scaler_transforms_windows_too():
    X = np.random.default_rng(1).normal(size=(40, 6)).astype(np.float32)
    sc = Scaler().fit(X)
    Wt = sc.transform(X.reshape(4, 10, 6))
    np.testing.assert_allclose(Wt.reshape(40, 6), sc.transform(X), atol=1e-6)


def test_scaler_unfitted_transform_raises():
    with pytest.raises(RuntimeError):
        Scaler().transform(np.zeros((2, 6), dtype=np.float32))


# ---- csv ---------------------------------------------------------------------


def test_csv_roundtrip(tmp_path):
    ds = make_ds([5, 4, 3])
    ds.X[2, 1] = np.nan
    ds.X[7, 4] = 0.1  # float32 value that is not exactly representable in decimal
    p = tmp_path / "a.csv"
    save_csv(ds, p)
    back = load_csv(p)
    np.testing.assert_array_equal(back.X, ds.X)  # NaN compares equal here
    np.testing.assert_array_equal(back.y, ds.y)
    np.testing.assert_array_equal(back.run, ds.run)
    np.testing.assert_array_equal(back.t_s, ds.t_s)
    np.testing.assert_array_equal(back.flags, ds.flags)
    assert back.X.dtype == np.float32 and back.y.dtype == np.float32
    assert back.run.dtype == np.int32 and back.flags.dtype == np.int32
    assert back.t_s.dtype == np.float32


def test_csv_without_labels_single_run_is_the_firmware_format(tmp_path):
    ds = make_ds([6], labelled=False)
    p = tmp_path / "b.csv"
    save_csv(ds, p)
    lines = p.read_text().splitlines()
    assert lines[0] == HEADER  # no label, no run column
    assert len(lines) == 1 + 6
    back = load_csv(p)
    assert back.y is None
    assert (back.run == 0).all()
    np.testing.assert_array_equal(back.X, ds.X)


def test_csv_labelled_multi_run_header_and_fixed_columns(tmp_path):
    ds = make_ds([3, 3])
    p = tmp_path / "c.csv"
    save_csv(ds, p)
    lines = p.read_text().splitlines()
    assert lines[0] == HEADER + ",label,run"
    cfg = SensorConfig()
    for i, line in enumerate(lines[1:]):
        cells = line.split(",")
        assert len(cells) == 12
        assert cells[8] == "-1"  # idx
        adc = float(cells[1])  # adc_mq: MQ divider inverted with R0 = RL, always in range
        assert 0 <= adc <= cfg.adc_max
        assert adc == pytest.approx(cfg.adc_max / (ds.X[i, 0] + 1), abs=0.5)


def test_csv_adc_mq_stays_in_range_for_odd_rs(tmp_path):
    ds = make_ds([4], labelled=False)
    ds.X[:, 0] = [np.nan, 0.0, -1.0, 1e9]
    p = tmp_path / "e.csv"
    save_csv(ds, p)
    for line in p.read_text().splitlines()[1:]:
        assert 0 <= float(line.split(",")[1]) <= SensorConfig().adc_max


def test_load_csv_converts_ms_to_seconds_and_reads_columns_by_name(tmp_path):
    # columns shuffled and an unknown extra column: lookup is by header name
    p = write(
        tmp_path,
        "flags,extra,rh,temp_c,ldr_blk,ldr_raw,c_comp,rs_r0,adc_mq,t_ms,idx\n"
        "5,zzz,60,25,300,310,2.5,1.5,400,123456,-1\n",
    )
    ds = load_csv(p)
    assert ds.t_s.tolist() == pytest.approx([123.456])
    assert ds.X[0].tolist() == [1.5, 2.5, 310, 300, 25, 60]  # FEATURE_NAMES order
    assert ds.flags.tolist() == [5]


def test_load_csv_missing_column_names_it(tmp_path):
    cols = [c for c in CSV_COLUMNS if c != "ldr_blk"]
    p = write(tmp_path, ",".join(cols) + "\n" + ",".join("1" for _ in cols) + "\n")
    with pytest.raises(ValueError, match="ldr_blk"):
        load_csv(p)


def test_load_csv_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_csv(write(tmp_path, ""))


@pytest.mark.parametrize("bad_line", [2, 4])
def test_load_csv_non_numeric_reports_line(tmp_path, bad_line):
    good = "1000,500,1.2,3.4,300,310,25,50,-1,0"
    bad = "1000,500,1.2,oops,300,310,25,50,-1,0"
    rows = [bad if n == bad_line else good for n in (2, 3, 4)]
    p = write(tmp_path, HEADER + "\n" + "\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=rf"line {bad_line}\b") as e:
        load_csv(p)
    assert "c_comp" in str(e.value)


def test_load_csv_line_number_counts_blank_lines(tmp_path):
    p = write(
        tmp_path,
        HEADER + "\n1000,500,1.2,3.4,300,310,25,50,-1,0\n\n1000,500,1.2,3.4,300,310,25,nope,-1,0\n",
    )
    with pytest.raises(ValueError, match=r"line 4\b"):
        load_csv(p)


def test_load_csv_wrong_field_count_reports_line(tmp_path):
    p = write(tmp_path, HEADER + "\n1000,500,1.2,3.4,300,310,25,50,-1,0\n1000,500,1.2\n")
    with pytest.raises(ValueError, match=r"line 3\b"):
        load_csv(p)


def test_load_csv_accepts_idx_minus1_and_nan(tmp_path):
    p = write(
        tmp_path,
        HEADER + "\n"
        "1000,512,nan,nan,300,310,25.5,50.0,-1,1\n"
        "2000,513,1.5,2.5,301,311,25.6,50.1,-1,0\n",
    )
    ds = load_csv(p)
    assert ds.X.shape == (2, 6) and ds.X.dtype == np.float32
    assert np.isnan(ds.X[0, 0]) and np.isnan(ds.X[0, 1])
    assert ds.X[1, 0] == pytest.approx(1.5)
    assert ds.t_s.tolist() == [1.0, 2.0]
    assert ds.flags.tolist() == [1, 0]
    assert ds.y is None and (ds.run == 0).all()


def test_load_csv_reads_trailing_label_column(tmp_path):
    p = write(
        tmp_path,
        HEADER + ",label\n1000,512,1.0,2.0,300,310,25,50,-1,0,0.7\n2000,512,1.0,2.0,300,310,25,50,-1,0,nan\n",
    )
    ds = load_csv(p)
    assert ds.y[0] == pytest.approx(0.7) and np.isnan(ds.y[1])
    assert ds.y.dtype == np.float32


def test_load_csv_header_only_gives_empty_dataset(tmp_path):
    ds = load_csv(write(tmp_path, HEADER + "\n"))
    assert ds.X.shape == (0, 6) and ds.t_s.shape == (0,) and ds.y is None


def test_load_csv_skips_comments_and_starts_new_run_on_repeated_header(tmp_path):
    header = ",".join(CSV_COLUMNS)
    rows = ["# mlp_us=812345", "2000,500,0.8,250,600,650,24,60,0.5,0", "4000,500,0.8,251,600,650,24,60,0.5,0",
            header, "# mlp_us=812000", "2000,500,0.8,252,600,650,24,60,0.5,0"]
    path = tmp_path / "raw_serial.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    ds = load_csv(path)
    assert len(ds.X) == 3
    assert ds.run.tolist() == [0, 0, 1]
