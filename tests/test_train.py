import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ethylene import sim, train
from ethylene.data import FLAG_WARMUP

COMMITTED = Path(__file__).resolve().parent.parent / "artifacts" / "metrics.json"


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
    assert metrics["mlp"]["architecture"] == [32, 16] and metrics["mlp"]["params"] == 769
    assert set(metrics["ood"]) == set(sim.OOD_PROFILES)
    assert set(metrics["baselines"]) == {"linear_c_comp_only", "ridge_all_features", "gradient_boosting"}
    assert set(metrics["baselines"]["ridge_all_features"]) == {"id_test", *sim.OOD_PROFILES}
    assert "train" in metrics["coverage"] and set(metrics["coverage"]["ood"]) == set(sim.OOD_PROFILES)


def test_fast_models_beat_predicting_the_mean(fast_run):
    _, metrics = fast_run
    assert metrics["mlp"]["r2"] > 0.2 and metrics["transformer"]["r2"] > 0.2


def test_autoencoder_gate_detects_faults(fast_run):
    _, metrics = fast_run
    ae = metrics["autoencoder"]
    assert ae["fault_detection_rate"] >= 0.7
    assert ae["false_positive_rate"] <= 0.08
    assert set(ae["per_kind"]) == set(sim.FAULT_KINDS)
    assert ae["per_kind"]["dht_zero"] >= 0.9  # the range gate catches T=RH=0 that the autoencoder alone passes
    assert ae["fault_detection_rate"] >= ae["ae_only"]["fault_detection_rate"]


def test_anomaly_mask_flags_inputs_far_outside_training(fast_run):
    out, _ = fast_run
    art = train.load_artifacts(out)
    typical = np.array([[0.65, 300.0, 430.0, 530.0, 20.0, 78.0]], dtype=np.float32)
    far = typical.copy()
    far[0, 5] = 0.0  # RH reading of 0 %
    assert not train.anomaly_mask(art.ae, art.scaler, art.threshold, typical)[0][0]
    assert train.anomaly_mask(art.ae, art.scaler, art.threshold, far)[0][0]


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
    assert sum(p.numel() for p in art.mlp.parameters()) == metrics["mlp"]["params"]


def test_committed_artifacts_are_honest():
    """The shipped model must beat the one-feature baseline and not collapse under shifted conditions."""
    m = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert m["data"]["kind"] == "synthetic"
    assert m["mlp"]["mae"] <= m["baselines"]["linear_c_comp_only"]["id_test"]["mae"]
    assert all(v["mlp"]["mae"] <= 2.0 * m["mlp"]["mae"] for v in m["ood"].values())
    assert m["mlp"]["monotonic_violation_rate"] <= 0.1
    assert m["autoencoder"]["false_positive_rate"] <= 0.03
