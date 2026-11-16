"""Train and benchmark the MLP, autoencoder and transformer on synthetic ripening data.

``python -m ethylene.train [--fast] [--hidden 32,16]`` writes mlp.pt, ae.pt, transformer.pt, scaler.json and
metrics.json to ``artifacts/``. Every metric is on SYNTHETIC data, not field accuracy. Models are scored on a
held-out in-distribution test set and on out-of-distribution sets (unseen cultivars, cold-chain conditions,
shifted sensor units) so the numbers show how the models behave when conditions change.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ethylene import sim
from ethylene.coverage import Z_FAR, coverage_report
from ethylene.data import FLAG_WARMUP, Dataset, Scaler, make_windows, split_by_run
from ethylene.models import Autoencoder, ConstrainedMLP, TinyTransformer, composite_loss

C_IDX = 1  # c_comp column in the standardised feature vector
MONOTONIC_STEP = 0.1  # standardised units
MONOTONIC_TOL = 1e-4
WINDOW = 16
DEFAULT_HIDDEN = (32, 16)  # compact net: 769 parameters
PAPER_HIDDEN = (256, 128, 64, 32)  # the paper's 6-256-128-64-32-1
MEGA_MS_PER_MAC = 663.072 / 44_576  # measured on the Mega 2560: 663 ms for the paper net's 44,576 MACs


@dataclass(frozen=True)
class TrainConfig:
    n_runs: int
    steps: int
    epochs_mlp: int
    epochs_tf: int
    epochs_ae: int
    ood_runs: int = 60
    seed: int = 0
    patience: int = 15
    batch: int = 256
    lam_phys: float = 1.0
    noise_sigma: float = 0.05  # input-noise augmentation, standardised units


FULL = TrainConfig(n_runs=300, steps=120, epochs_mlp=150, epochs_tf=60, epochs_ae=80)
FAST = TrainConfig(n_runs=80, steps=60, epochs_mlp=15, epochs_tf=8, epochs_ae=60, ood_runs=15, patience=8)


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


def make_ood_sets(fast: bool = False) -> dict[str, Dataset]:
    """Fresh batches generated under shifted conditions (never seen in training)."""
    cfg = FAST if fast else FULL
    return {
        name: sim.simulate(cfg.ood_runs, cfg.steps, cfg.seed + 100 + i, profile=profile)
        for i, (name, profile) in enumerate(sim.OOD_PROFILES.items())
    }


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


def train_mlp(train: Dataset, val: Dataset, scaler: Scaler, cfg: TrainConfig,
              hidden: tuple[int, ...] = DEFAULT_HIDDEN, seed: int | None = None) -> ConstrainedMLP:
    """MSE + physics penalty, with Gaussian input-noise augmentation (sensor noise) when cfg.noise_sigma > 0."""
    torch.manual_seed(cfg.seed if seed is None else seed)
    xt, yt = _t(scaler.transform(train.X)), _t(train.y)
    xv, yv = _t(scaler.transform(val.X)), _t(val.y)
    model = ConstrainedMLP(hidden)

    def loss_fn(m, x, y):
        if cfg.noise_sigma > 0:
            x = x + cfg.noise_sigma * torch.randn_like(x)
        return composite_loss(m, x, y, lam_phys=cfg.lam_phys)

    return _fit(model, (xt, yt), (xv, yv), loss_fn=loss_fn,
                val_fn=lambda m, x, y: torch.mean((m(x) - y) ** 2), epochs=cfg.epochs_mlp, cfg=cfg)


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


def n_macs(model: ConstrainedMLP) -> int:
    return sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, torch.nn.Linear))


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


def score(pred: np.ndarray, y: np.ndarray) -> dict:
    return {"r2": float(r2_score(y, pred)), "mae": float(np.mean(np.abs(pred - y)))}


def eval_mlp(model, scaler: Scaler, sets: dict[str, Dataset]) -> dict:
    with torch.no_grad():
        return {name: score(model(_t(scaler.transform(ds.X))).numpy(), ds.y) for name, ds in sets.items()}


def eval_transformer(model, scaler: Scaler, sets: dict[str, Dataset]) -> dict:
    out = {}
    with torch.no_grad():
        for name, ds in sets.items():
            w, y = _windows(ds, scaler)
            out[name] = score(model(w).numpy(), y.numpy())
    return out


def baselines(train: Dataset, sets: dict[str, Dataset]) -> dict:
    """Simple non-neural models trained on the same data: if the network cannot beat these, it adds nothing."""
    specs = {
        "linear_c_comp_only": (make_pipeline(StandardScaler(), LinearRegression()), [C_IDX]),
        "ridge_all_features": (make_pipeline(StandardScaler(), Ridge(1.0)), None),
        "gradient_boosting": (HistGradientBoostingRegressor(random_state=0), None),
    }
    out = {}
    for name, (model, cols) in specs.items():
        pick = (lambda X: X[:, cols]) if cols else (lambda X: X)
        model.fit(pick(train.X), train.y)
        out[name] = {s: score(model.predict(pick(ds.X)), ds.y) for s, ds in sets.items()}
    return out


def _regression_metrics(pred: np.ndarray, y: np.ndarray, model, example, macs: int | None = None) -> dict:
    params = sum(p.numel() for p in model.parameters())
    out = {**score(pred, y), "params": params, "fp32_kb": params * 4 / 1024, "int8_kb": params / 1024,
           "latency_ms": _latency_ms(model, example)}
    if macs is not None:
        out["macs"] = macs
        out["est_mega_ms"] = macs * MEGA_MS_PER_MAC
    return out


def _monotonic_violation_rate(model, x: torch.Tensor) -> float:
    x2 = x.clone()
    x2[:, C_IDX] += MONOTONIC_STEP
    with torch.no_grad():
        return float(torch.mean(((model(x2) - model(x)) < -MONOTONIC_TOL).float()))


def anomaly_mask(ae: Autoencoder, scaler: Scaler, threshold: float, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flagged, ae_score) for rows of X. A row is flagged when the autoencoder's reconstruction error exceeds
    `threshold` OR any feature lies more than Z_FAR standard deviations outside the training data (an input
    the models were never trained on, which independent features such as T/RH would otherwise pass through)."""
    xs = scaler.transform(X)
    with torch.no_grad():
        score = ae.score(_t(xs)).numpy()
    far = (np.abs(xs) > Z_FAR).any(axis=-1)
    return (score > threshold) | far, score


def _autoencoder_metrics(ae: Autoencoder, scaler: Scaler, val: Dataset, test: Dataset, seed: int) -> dict:
    with torch.no_grad():
        threshold = float(np.percentile(ae.score(_t(scaler.transform(val.X))).numpy(), 99))
    rng = np.random.default_rng(seed)
    faulty = {kind: sim.inject_faults(test.X, kind, rng) for kind in sim.FAULT_KINDS}
    per_kind = {k: float(np.mean(anomaly_mask(ae, scaler, threshold, X)[0])) for k, X in faulty.items()}
    ae_only = {k: float(np.mean(anomaly_mask(ae, scaler, np.inf, X)[1] > threshold)) for k, X in faulty.items()}
    return {
        "threshold": threshold,
        "fault_detection_rate": float(np.mean(list(per_kind.values()))),
        "false_positive_rate": float(np.mean(anomaly_mask(ae, scaler, threshold, test.X)[0])),
        "per_kind": per_kind,
        "ae_only": {"fault_detection_rate": float(np.mean(list(ae_only.values()))), "per_kind": ae_only},
    }


def _coverage(train: Dataset, scaler: Scaler, oods: dict[str, Dataset]) -> dict:
    return {
        "train": coverage_report(train.X),
        "ood": {name: {feat: {"frac_far": e["frac_far"], "frac_outside_physical": e["frac_outside_physical"]}
                       for feat, e in coverage_report(ds.X, scaler).items()} for name, ds in oods.items()},
    }


def main(out_dir: str = "artifacts", fast: bool = False, dataset: Dataset | None = None,
         hidden: tuple[int, ...] | None = None) -> dict:
    cfg = FAST if fast else FULL
    hidden = tuple(hidden) if hidden else DEFAULT_HIDDEN
    train, val, test = make_splits(fast=fast, dataset=dataset)
    oods = make_ood_sets(fast=fast)
    scaler = Scaler().fit(train.X)

    mlp = train_mlp(train, val, scaler, cfg, hidden)
    transformer = train_transformer(train, val, scaler, cfg)
    ae = train_autoencoder(train, val, scaler, cfg)

    x_test = _t(scaler.transform(test.X))
    with torch.no_grad():
        mlp_pred = mlp(x_test).numpy()
    w_test, y_w = _windows(test, scaler)
    with torch.no_grad():
        tf_pred = transformer(w_test).numpy()

    sets = {"id_test": test, **oods}
    metrics = {
        "data": {"kind": "synthetic", "generator": "realistic: cultivars, sensor-unit spread, interferents, drift, ADC",
                 "n_train": len(train.X), "n_val": len(val.X), "n_test": len(test.X), "ood_sets": list(oods)},
        "mlp": {**_regression_metrics(mlp_pred, test.y, mlp, x_test[:1], macs=n_macs(mlp)),
                "architecture": list(hidden), "monotonic_violation_rate": _monotonic_violation_rate(mlp, x_test)},
        "transformer": _regression_metrics(tf_pred, y_w.numpy(), transformer, w_test[:1]),
        "autoencoder": _autoencoder_metrics(ae, scaler, val, test, cfg.seed),
        "ood": {
            name: {"mlp": m, "transformer": t}
            for (name, m), t in zip(eval_mlp(mlp, scaler, oods).items(), eval_transformer(transformer, scaler, oods).values())
        },
        "baselines": baselines(train, sets),
        "coverage": _coverage(train, scaler, oods),
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
    hidden = tuple(metrics["mlp"].get("architecture", PAPER_HIDDEN))
    mlp, ae, tf = ConstrainedMLP(hidden), Autoencoder(), TinyTransformer(width=WINDOW)
    for model, name in ((mlp, "mlp"), (ae, "ae"), (tf, "transformer")):
        model.load_state_dict(torch.load(d / f"{name}.pt", map_location="cpu", weights_only=True))
        model.eval()
    return Artifacts(mlp, ae, tf, scaler, float(metrics["autoencoder"]["threshold"]), metrics)


def selected_hidden(directory: str | Path = "artifacts") -> tuple[int, ...]:
    """Architecture picked by `python -m ethylene.benchmark`, else the compact default."""
    path = Path(directory) / "benchmark.json"
    if path.exists():
        return tuple(json.loads(path.read_text(encoding="utf-8"))["selected_hidden"])
    return DEFAULT_HIDDEN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="small, quick configuration")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--hidden", help="comma-separated hidden sizes, e.g. 32,16 (default: benchmark's pick)")
    args = parser.parse_args()
    chosen = tuple(int(v) for v in args.hidden.split(",")) if args.hidden else selected_hidden(args.out)
    result = main(args.out, fast=args.fast, hidden=chosen)
    print(json.dumps({k: result[k] for k in ("mlp", "transformer", "autoencoder", "ood", "baselines", "primary")}, indent=2))
    print("NOTE: all metrics are on synthetic data, not field accuracy.")
