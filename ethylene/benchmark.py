"""Compare candidate networks and baselines on in-distribution and shifted data, then pick the architecture.

``python -m ethylene.benchmark [--fast]`` writes ``artifacts/benchmark.json``. Selection rule: among the MLP
candidates, take the smallest network whose mean MAE over all test sets (in-distribution plus the three
shifted-condition sets) is within 5% of the best one, so extra size has to earn its place. Everything here is
SYNTHETIC data, not field accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from ethylene import train
from ethylene.data import Scaler

CANDIDATES = {
    "mlp_paper": train.PAPER_HIDDEN,
    "mlp_mid": (64, 32),
    "mlp_small": (32, 16),
    "mlp_tiny": (16, 8),
}
SELECTION_TOLERANCE = 1.05
ENSEMBLE_SIZE = 5


def _summary(model, hidden, per_set: dict) -> dict:
    macs = train.n_macs(model)
    maes = [s["mae"] for s in per_set.values()]
    return {
        "hidden": list(hidden), "params": sum(p.numel() for p in model.parameters()), "macs": macs,
        "est_mega_ms": macs * train.MEGA_MS_PER_MAC, "sets": per_set,
        "id_mae": per_set["id_test"]["mae"], "mean_mae": float(np.mean(maes)),
    }


def run(out_dir: str = "artifacts", fast: bool = False) -> dict:
    cfg = train.FAST if fast else train.FULL
    tr, va, te = train.make_splits(fast=fast)
    oods = train.make_ood_sets(fast=fast)
    sets = {"id_test": te, **oods}
    scaler = Scaler().fit(tr.X)
    small = CANDIDATES["mlp_small"]

    results = {}
    for name, hidden in CANDIDATES.items():
        model = train.train_mlp(tr, va, scaler, cfg, hidden)
        results[name] = _summary(model, hidden, train.eval_mlp(model, scaler, sets))
    for name, variant in (("mlp_small_no_noise_aug", dataclasses.replace(cfg, noise_sigma=0.0)),
                          ("mlp_small_no_physics_prior", dataclasses.replace(cfg, lam_phys=0.0))):
        model = train.train_mlp(tr, va, scaler, variant, small)
        results[name] = _summary(model, small, train.eval_mlp(model, scaler, sets))

    # Deep ensemble of small nets: better mean, and the spread is an uncertainty estimate.
    members = [train.train_mlp(tr, va, scaler, cfg, small, seed=cfg.seed + 1 + i) for i in range(ENSEMBLE_SIZE)]
    ensemble = {"members": ENSEMBLE_SIZE, "params": sum(p.numel() for m in members for p in m.parameters()), "sets": {}}
    with torch.no_grad():
        for name, ds in sets.items():
            preds = np.stack([m(train._t(scaler.transform(ds.X))).numpy() for m in members])
            mean, std = preds.mean(axis=0), preds.std(axis=0)
            err = np.abs(mean - ds.y)
            ensemble["sets"][name] = {**train.score(mean, ds.y), "mean_std": float(std.mean()),
                                      "std_error_corr": float(np.corrcoef(std, err)[0, 1])}
    ensemble["mean_mae"] = float(np.mean([s["mae"] for s in ensemble["sets"].values()]))

    transformer = train.train_transformer(tr, va, scaler, cfg)
    tf_sets = train.eval_transformer(transformer, scaler, sets)
    transformer_summary = {"params": sum(p.numel() for p in transformer.parameters()), "sets": tf_sets,
                           "mean_mae": float(np.mean([s["mae"] for s in tf_sets.values()]))}

    nn_names = list(CANDIDATES)
    best = min(results[n]["mean_mae"] for n in nn_names)
    eligible = [n for n in nn_names if results[n]["mean_mae"] <= SELECTION_TOLERANCE * best]
    selected = min(eligible, key=lambda n: results[n]["params"])

    out = {
        "data": "synthetic (realistic generator); OOD sets: " + ", ".join(oods),
        "rule": f"smallest MLP whose mean MAE over all sets is within {int((SELECTION_TOLERANCE - 1) * 100)}% of the best",
        "selected": selected, "selected_hidden": list(CANDIDATES[selected]),
        "candidates": results, "ensemble_small": ensemble, "transformer": transformer_summary,
        "baselines": train.baselines(tr, sets),
    }
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def print_table(out: dict) -> None:
    sets = list(next(iter(out["candidates"].values()))["sets"])
    print(f"{'model':30s} {'params':>7s} {'Mega ms':>8s} " + " ".join(f"{s[:14]:>14s}" for s in sets) + "   mean MAE")
    rows = [(n, c["params"], c["est_mega_ms"], c["sets"], c["mean_mae"]) for n, c in out["candidates"].items()]
    e = out["ensemble_small"]
    rows.append((f"ensemble x{e['members']} (small)", e["params"], None, e["sets"], e["mean_mae"]))
    t = out["transformer"]
    rows.append(("transformer (16-step window)", t["params"], None, t["sets"], t["mean_mae"]))
    for name, s in out["baselines"].items():
        rows.append((name, None, None, s, float(np.mean([v["mae"] for v in s.values()]))))
    for name, params, ms, per_set, mean_mae in rows:
        cells = " ".join(f"{per_set[s]['mae']:.3f}/{per_set[s]['r2']:.2f}".rjust(14) for s in sets)
        print(f"{name:30s} {str(params or '-'):>7s} {('%.0f' % ms) if ms else '-':>8s} {cells}   {mean_mae:.4f}")
    print("cells: MAE/R2.  selected:", out["selected"], out["selected_hidden"], "-", out["rule"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    result = run(args.out, fast=args.fast)
    print_table(result)
    print("NOTE: all metrics are on synthetic data, not field accuracy.")
