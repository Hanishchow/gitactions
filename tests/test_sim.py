import numpy as np
import pytest

from ethylene.data import Dataset
from ethylene.physics import FEATURE_NAMES, SensorConfig, compensate, concentration
from ethylene.sim import FAULT_KINDS, STAGE_MEANS, inject_faults, simulate

COL = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Columns each fault is allowed to alter (and, for the first three, must alter).
FAULT_COLUMNS = {
    "rs_drift": {"rs_r0"},
    "ldr_dead": {"ldr_raw"},
    "dht_zero": {"temp_c", "rh"},
    "c_comp_stuck": {"c_comp"},
}


@pytest.fixture(scope="module")
def ds():
    return simulate(n_runs=40, steps=120, seed=7)


@pytest.fixture
def sample():
    """100 rows of healthy data (one run)."""
    return simulate(n_runs=1, steps=100, seed=3).X


def test_constants():
    assert STAGE_MEANS == (0.05, 0.22, 0.42, 0.62, 0.82, 0.93)
    assert FAULT_KINDS == ("rs_drift", "ldr_dead", "dht_zero", "c_comp_stuck")
    assert set(FAULT_COLUMNS) == set(FAULT_KINDS)


def test_simulate_deterministic():
    a = simulate(5, 30, seed=11)
    b = simulate(5, 30, seed=11)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert not np.array_equal(a.X, simulate(5, 30, seed=12).X)


def test_shapes_and_ranges(ds):
    n_runs, steps = 40, 120
    assert isinstance(ds, Dataset)
    assert ds.X.shape == (n_runs * steps, 6)
    assert ds.X.dtype == np.float32
    assert np.isfinite(ds.X).all()
    assert ds.y.shape == (n_runs * steps,) and ds.y.dtype == np.float32
    assert ds.y.min() >= 0 and ds.y.max() <= 1
    assert ds.run.dtype == np.int32 and ds.flags.dtype == np.int32 and ds.t_s.dtype == np.float32
    assert np.array_equal(ds.run, np.repeat(np.arange(n_runs), steps))
    assert np.array_equal(ds.t_s, np.tile(2.0 * np.arange(steps), n_runs).astype(np.float32))
    assert not ds.flags.any()
    blk = ds.X[:, COL["ldr_blk"]].reshape(n_runs, steps)
    assert (blk == blk[:, :1]).all()  # one baseline per run
    assert 500 <= blk.min() and blk.max() <= 700
    assert 5 <= ds.X[:, COL["temp_c"]].min() and ds.X[:, COL["temp_c"]].max() <= 35
    assert 50 <= ds.X[:, COL["rh"]].min() and ds.X[:, COL["rh"]].max() <= 110


def test_dht_values_are_integers(ds):
    for name in ("temp_c", "rh"):
        v = ds.X[:, COL[name]]
        assert np.array_equal(v, np.round(v))


def test_c_comp_tracks_ripeness(ds):
    assert np.corrcoef(ds.X[:, COL["c_comp"]], ds.y)[0, 1] > 0.95


def test_ripeness_rises_within_run(ds):
    for r in np.unique(ds.run):
        m = ds.run == r
        assert np.corrcoef(ds.y[m], ds.t_s[m])[0, 1] > 0.9


def test_runs_ripen_at_different_speeds(ds):
    mid = ds.y.reshape(40, 120)[:, 60]
    assert mid.std() > 0.03
    first, last = ds.y.reshape(40, 120)[:, 0], ds.y.reshape(40, 120)[:, -1]
    assert abs(first.mean() - STAGE_MEANS[0]) < 0.02
    assert abs(last.mean() - STAGE_MEANS[-1]) < 0.02


def test_c_comp_matches_physics(ds):
    """c_comp is the physics-compensated concentration implied by rs_r0 and the DHT reading."""
    for row in ds.X[::97]:
        rs_r0, c_comp, _, _, t, rh = (float(v) for v in row)
        expected = compensate(concentration(rs_r0), t, rh)
        assert c_comp == pytest.approx(expected, rel=1e-4)


def test_c_comp_close_to_true_concentration(ds):
    """Compensation cancels the T/RH dependence, leaving only the ~2 % sensor noise."""
    cfg = SensorConfig()
    ratio = ds.X[:, COL["c_comp"]] / (cfg.A * (1 + 3.0 * ds.y)) - 1
    assert abs(ratio.mean()) < 0.005
    assert 0.01 < ratio.std() < 0.04


def test_cfg_and_g_are_used():
    base = simulate(3, 40, seed=5)
    cfg = SensorConfig(A=50.0)
    scaled = simulate(3, 40, seed=5, cfg=cfg)
    assert np.allclose(scaled.X[:, COL["c_comp"]], base.X[:, COL["c_comp"]] * 50.0 / SensorConfig().A, rtol=1e-4)
    assert np.array_equal(scaled.y, base.y)
    steeper = simulate(3, 40, seed=5, g=6.0)
    assert (steeper.X[:, COL["c_comp"]] > base.X[:, COL["c_comp"]]).mean() > 0.9


@pytest.mark.parametrize("n_runs,steps", [(0, 10), (3, 0), (-1, 10)])
def test_simulate_rejects_empty_sizes(n_runs, steps):
    with pytest.raises(ValueError):
        simulate(n_runs, steps, seed=0)


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_each_fault_changes_input(sample, kind):
    before = sample.copy()
    out = inject_faults(sample, kind, np.random.default_rng(0))
    assert out is not sample
    assert np.array_equal(sample, before)  # input untouched
    assert out.shape == sample.shape and out.dtype == sample.dtype
    changed = {FEATURE_NAMES[j] for j in range(6) if not np.array_equal(out[:, j], sample[:, j])}
    assert changed, f"{kind} changed nothing"
    assert changed <= FAULT_COLUMNS[kind]


def test_fault_effects(sample):
    rng = np.random.default_rng(1)
    drift = inject_faults(sample, "rs_drift", rng)
    ratio = drift[:, COL["rs_r0"]] / sample[:, COL["rs_r0"]]
    assert ratio.min() >= 1.5 - 1e-4 and ratio.max() <= 2.5 + 1e-4
    assert np.array_equal(drift[:, COL["c_comp"]], sample[:, COL["c_comp"]])

    assert (inject_faults(sample, "ldr_dead", rng)[:, COL["ldr_raw"]] == 0).all()

    dht = inject_faults(sample, "dht_zero", rng)
    assert (dht[:, COL["temp_c"]] == 0).all() and (dht[:, COL["rh"]] == 0).all()

    stuck = inject_faults(sample, "c_comp_stuck", rng)[:, COL["c_comp"]]
    assert (stuck == np.float32(np.median(sample[:, COL["c_comp"]]))).all()


def test_rs_drift_is_random_but_seeded(sample):
    a = inject_faults(sample, "rs_drift", np.random.default_rng(4))
    b = inject_faults(sample, "rs_drift", np.random.default_rng(4))
    c = inject_faults(sample, "rs_drift", np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_unknown_fault_raises_value_error(sample):
    with pytest.raises(ValueError, match="bogus"):
        inject_faults(sample, "bogus", np.random.default_rng(0))
