"""Train and benchmark the MLP, autoencoder and transformer on synthetic ripening data.

``python -m ethylene.train [--fast]`` writes mlp.pt, ae.pt, transformer.pt, scaler.json and
metrics.json to ``artifacts/``. Every metric is on SYNTHETIC data, not field accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

from ethylene import sim
from ethylene.data import FLAG_WARMUP, Dataset, Scaler, make_windows, split_by_run
from ethylene.models import Autoencoder, ConstrainedMLP, TinyTransformer, composite_loss

C_IDX = 1  # c_comp column in the standardised feature vector
MONOTONIC_STEP = 0.1  # standardised units
MONOTONIC_TOL = 1e-4
WINDOW = 16


@dataclass(frozen=True)
class TrainConfig:
    n_runs: int
    steps: int
    epochs_mlp: int
    epochs_tf: int
    epochs_ae: int
    seed: int = 0
    patience: int = 15
    batch: int = 256
    lam_phys: float = 1.0


FULL = TrainConfig(n_runs=300, steps=120, epochs_mlp=150, epochs_tf=60, epochs_ae=80)
FAST = TrainConfig(n_runs=80, steps=60, epochs_mlp=15, epochs_tf=8, epochs_ae=60, patience=8)


@dataclass
class Artifacts:
    mlp: ConstrainedMLP
    ae: Autoencoder
    transformer: TinyTransformer
    scaler: Scaler
    threshold: float
    metrics: dict


def make_splits(fast: bool = False, dataset: Dataset | None = None) -> tuple[Dataset, Dataset, Dataset]:
    """Deterministic train/val/test split by run, with warm-up rows dropped."""
    cfg = FAST if fast else FULL
    ds = dataset if dataset is not None else sim.simulate(cfg.n_runs, cfg.steps, cfg.seed)
    ds = ds._take((ds.flags & FLAG_WARMUP) == 0)
    return split_by_run(ds, val_frac=0.15, test_frac=0.15, seed=cfg.seed)


def choose_primary(metrics: dict) -> str:
    """The transformer wins only if its MAE is at most 0.8x the MLP's; otherwise the cheaper MLP."""
    return "transformer" if metrics["transformer"]["mae"] <= 0.8 * metrics["mlp"]["mae"] else "mlp"


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


def _batches(n: int, batch: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, batch):
        yield order[i:i + batch]


def _fit(model, tensors_train, tensors_val, loss_fn, val_fn, epochs, cfg, lr=1e-3):
    """Generic Adam loop with early stopping on the validation metric; restores the best weights."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for idx in _batches(len(tensors_train[0]), cfg.batch, rng):
            opt.zero_grad()
            loss_fn(model, *(t[idx] for t in tensors_train)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = float(val_fn(model, *tensors_val))
        if score < best - 1e-7:
            best, best_state, bad = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def train_mlp(train: Dataset, val: Dataset, scaler: Scaler, cfg: TrainConfig) -> ConstrainedMLP:
    torch.manual_seed(cfg.seed)
    xt, yt = _t(scaler.transform(train.X)), _t(train.y)
    xv, yv = _t(scaler.transform(val.X)), _t(val.y)
    model = ConstrainedMLP()
    return _fit(
        model, (xt, yt), (xv, yv),
        loss_fn=lambda m, x, y: composite_loss(m, x, y, lam_phys=cfg.lam_phys),
        val_fn=lambda m, x, y: torch.mean((m(x) - y) ** 2),
        epochs=cfg.epochs_mlp, cfg=cfg,
    )


def _windows(ds: Dataset, scaler: Scaler):
    scaled = dataclasses.replace(ds, X=scaler.transform(ds.X))
    w, y, _ = make_windows(scaled, WINDOW)
    return _t(w), _t(y)


def train_transformer(train: Dataset, val: Dataset, scaler: Scaler, cfg: TrainConfig) -> TinyTransformer:
    torch.manual_seed(cfg.seed)
    model = TinyTransformer(width=WINDOW)
    return _fit(
        model, _windows(train, scaler), _windows(val, scaler),
        loss_fn=lambda m, w, y: torch.mean((m(w) - y) ** 2),
        val_fn=lambda m, w, y: torch.mean((m(w) - y) ** 2),
        epochs=cfg.epochs_tf, cfg=cfg,
    )


def train_autoencoder(train: Dataset, val: Dataset, scaler: Scaler, cfg: TrainConfig) -> Autoencoder:
    """Fit on fault-free rows only."""
    torch.manual_seed(cfg.seed)
    xt, xv = _t(scaler.transform(train.X)), _t(scaler.transform(val.X))
    model = Autoencoder()
    return _fit(
        model, (xt,), (xv,),
        loss_fn=lambda m, x: torch.mean(m.score(x)),
        val_fn=lambda m, x: torch.mean(m.score(x)),
        epochs=cfg.epochs_ae, cfg=cfg, lr=3e-3,
    )


def _latency_ms(fn, example, calls: int = 200) -> float:
    with torch.no_grad():
        for _ in range(10):
            fn(example)
        times = []
        for _ in range(calls):
            t0 = time.perf_counter()
            fn(example)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _regression_metrics(pred: np.ndarray, y: np.ndarray, model, example) -> dict:
    params = sum(p.numel() for p in model.parameters())
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(np.mean(np.abs(pred - y))),
        "params": params,
        "fp32_kb": params * 4 / 1024,
        "int8_kb": params / 1024,
        "latency_ms": _latency_ms(model, example),
    }


def _monotonic_violation_rate(model, x: torch.Tensor) -> float:
    x2 = x.clone()
    x2[:, C_IDX] += MONOTONIC_STEP
    with torch.no_grad():
        return float(torch.mean(((model(x2) - model(x)) < -MONOTONIC_TOL).float()))


def _autoencoder_metrics(ae: Autoencoder, scaler: Scaler, val: Dataset, test: Dataset, seed: int) -> dict:
    with torch.no_grad():
        threshold = float(np.percentile(ae.score(_t(scaler.transform(val.X))).numpy(), 99))
        fpr = float(np.mean(ae.score(_t(scaler.transform(test.X))).numpy() > threshold))
        rng = np.random.default_rng(seed)
        per_kind = {
            kind: float(np.mean(ae.score(_t(scaler.transform(sim.inject_faults(test.X, kind, rng)))).numpy() > threshold))
            for kind in sim.FAULT_KINDS
        }
    return {
        "threshold": threshold,
        "fault_detection_rate": float(np.mean(list(per_kind.values()))),
        "false_positive_rate": fpr,
        "per_kind": per_kind,
    }


def main(out_dir: str = "artifacts", fast: bool = False, dataset: Dataset | None = None) -> dict:
    cfg = FAST if fast else FULL
    train, val, test = make_splits(fast=fast, dataset=dataset)
    scaler = Scaler().fit(train.X)

    mlp = train_mlp(train, val, scaler, cfg)
    transformer = train_transformer(train, val, scaler, cfg)
    ae = train_autoencoder(train, val, scaler, cfg)

    x_test = _t(scaler.transform(test.X))
    with torch.no_grad():
        mlp_pred = mlp(x_test).numpy()
    w_test, y_w = _windows(test, scaler)
    with torch.no_grad():
        tf_pred = transformer(w_test).numpy()

    metrics = {
        "data": {"kind": "synthetic", "n_train": len(train.X), "n_val": len(val.X), "n_test": len(test.X)},
        "mlp": {**_regression_metrics(mlp_pred, test.y, mlp, x_test[:1]),
                "monotonic_violation_rate": _monotonic_violation_rate(mlp, x_test)},
        "transformer": _regression_metrics(tf_pred, y_w.numpy(), transformer, w_test[:1]),
        "autoencoder": _autoencoder_metrics(ae, scaler, val, test, cfg.seed),
    }
    metrics["primary"] = choose_primary(metrics)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(mlp.state_dict(), out / "mlp.pt")
    torch.save(ae.state_dict(), out / "ae.pt")
    torch.save(transformer.state_dict(), out / "transformer.pt")
    (out / "scaler.json").write_text(json.dumps(scaler.to_dict()), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_artifacts(directory: str | Path = "artifacts") -> Artifacts:
    d = Path(directory)
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    scaler = Scaler.from_dict(json.loads((d / "scaler.json").read_text(encoding="utf-8")))
    mlp, ae, tf = ConstrainedMLP(), Autoencoder(), TinyTransformer(width=WINDOW)
    for model, name in ((mlp, "mlp"), (ae, "ae"), (tf, "transformer")):
        model.load_state_dict(torch.load(d / f"{name}.pt", map_location="cpu", weights_only=True))
        model.eval()
    return Artifacts(mlp, ae, tf, scaler, float(metrics["autoencoder"]["threshold"]), metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="small, quick configuration")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    result = main(args.out, fast=args.fast)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
    print("NOTE: all metrics are on synthetic data, not field accuracy.")
