import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ethylene import sim
from ethylene.coverage import coverage_report
from ethylene.data import Scaler, split_by_run


@pytest.fixture(scope="module")
def ds():
    return sim.simulate(120, 80, seed=3)


def test_deterministic_and_shapes(ds):
    again = sim.simulate(120, 80, seed=3)
    assert np.array_equal(ds.X, again.X) and not np.array_equal(ds.X, sim.simulate(120, 80, seed=4).X)
    assert ds.X.shape == (120 * 80, 6) and ds.X.dtype == np.float32
    assert ds.y.min() >= 0 and ds.y.max() <= 1 and ds.group.shape == ds.y.shape
    assert np.isfinite(ds.X).all()


def test_measurements_look_like_the_hardware(ds):
    rs_r0, c_comp, ldr_raw, ldr_blk, temp_c, rh = ds.X.T
    assert np.array_equal(temp_c, np.rint(temp_c)) and temp_c.min() >= 0 and temp_c.max() <= 50
    assert np.array_equal(rh, np.rint(rh)) and rh.min() >= 0 and rh.max() <= 100
    assert ldr_raw.min() >= 0 and ldr_raw.max() <= 1023 and ldr_blk.min() >= 0 and ldr_blk.max() <= 1023
    assert (rs_r0 > 0).all() and (c_comp > 0).all()


def test_not_trivially_predictable(ds):
    """The old generator gave R2 = 0.996 for a line on c_comp; here the label must stay genuinely ambiguous."""
    tr, _, te = split_by_run(ds, 0.15, 0.15, 0)
    line = LinearRegression().fit(tr.X[:, [1]], tr.y)
    r2 = r2_score(te.y, line.predict(te.X[:, [1]]))
    assert 0.3 < r2 < 0.9


def test_old_generator_is_the_trivial_one():
    old = sim.simulate_idealised(120, 80, seed=3)
    line = LinearRegression().fit(old.X[:, [1]], old.y)
    assert r2_score(old.y, line.predict(old.X[:, [1]])) > 0.98


def test_cultivar_params_are_deterministic_and_distinct():
    assert sim.cultivar_params(3) == sim.cultivar_params(3)
    assert sim.cultivar_params(3) != sim.cultivar_params(4)


def test_ood_profiles_shift_the_conditions(ds):
    unseen = sim.simulate(60, 60, 9, profile=sim.OOD_PROFILES["unseen_cultivars"])
    cold = sim.simulate(60, 60, 9, profile=sim.OOD_PROFILES["cold_chain"])
    assert ds.group.max() < 10 and unseen.group.min() >= 10
    assert cold.X[:, 4].mean() < ds.X[:, 4].mean() - 8


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        sim.simulate(2, 5, 0, profile=sim.SimProfile(cultivars=(5, 5)))
    with pytest.raises(ValueError):
        sim.simulate(2, 5, 0, profile=sim.SimProfile(cultivars=(0, sim.N_CULTIVARS + 1)))


def test_coverage_report_flags_extrapolation(ds):
    scaler = Scaler().fit(ds.X)
    rep = coverage_report(ds.X, scaler)
    assert set(rep) == {"rs_r0", "c_comp", "ldr_raw", "ldr_blk", "temp_c", "rh"}
    assert all(e["frac_outside_physical"] == 0 and e["frac_far"] < 0.01 for e in rep.values())
    broken = ds.X.copy()
    broken[:, 5] = 0.0
    assert coverage_report(broken, scaler)["rh"]["frac_far"] == 1.0
    broken[0, 4] = 99.0
    assert coverage_report(broken, scaler)["temp_c"]["frac_outside_physical"] > 0
