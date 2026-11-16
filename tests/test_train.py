import dataclasses

import numpy as np
import pytest

from ethylene import sim, train
from ethylene.data import FLAG_WARMUP


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    return out, train.main(str(out), fast=True)


def test_choose_primary_rule():
    base = {"mlp": {"mae": 0.10}}
    assert train.choose_primary({**base, "transformer": {"mae": 0.07}}) == "transformer"
    assert train.choose_primary({**base, "transformer": {"mae": 0.09}}) == "mlp"


def test_main_fast_writes_artifacts_and_metrics(fast_run):
    out, metrics = fast_run
    for name in ("mlp.pt", "ae.pt", "transformer.pt", "scaler.json", "metrics.json"):
        assert (out / name).exists()
    assert metrics["data"]["kind"] == "synthetic"
    assert metrics["primary"] in ("mlp", "transformer")
    assert metrics["mlp"]["params"] == 45057


def test_fast_mlp_quality(fast_run):
    _, metrics = fast_run
    assert metrics["mlp"]["r2"] >= 0.9
    assert metrics["mlp"]["monotonic_violation_rate"] <= 0.05


def test_autoencoder_detects_faults(fast_run):
    _, metrics = fast_run
    ae = metrics["autoencoder"]
    assert ae["fault_detection_rate"] >= 0.8
    assert ae["false_positive_rate"] <= 0.05
    assert set(ae["per_kind"]) == set(sim.FAULT_KINDS)


def test_warmup_rows_excluded():
    ds = sim.simulate(10, 30, seed=1)
    full = sum(len(s.X) for s in train.make_splits(fast=True, dataset=ds))
    flagged = ds.flags.copy()
    flagged[::3] = FLAG_WARMUP
    fewer = sum(len(s.X) for s in train.make_splits(fast=True, dataset=dataclasses.replace(ds, flags=flagged)))
    assert full == len(ds.X) and fewer == len(ds.X) - len(np.arange(0, len(ds.X), 3))


def test_load_artifacts_roundtrip(fast_run):
    out, metrics = fast_run
    art = train.load_artifacts(out)
    assert art.threshold == pytest.approx(metrics["autoencoder"]["threshold"])
    assert not art.mlp.training and not art.transformer.training
