import numpy as np
import pytest
import torch

from ethylene import export, train

ARTIFACTS = "artifacts"


@pytest.fixture(scope="module")
def art():
    return train.load_artifacts(ARTIFACTS)


@pytest.fixture(scope="module")
def q(art):
    return export.quantize_mlp(art.mlp, art.scaler)


@pytest.fixture(scope="module")
def test_x():
    return train.make_splits()[2].X[:500]


def test_quantized_close_to_float(art, q, test_x):
    with torch.no_grad():
        ref = art.mlp(torch.as_tensor(art.scaler.transform(test_x))).numpy()
    got = np.array([export.quantized_forward(q, row) for row in test_x])
    err = np.abs(got - ref)
    assert err.mean() < 0.01 and err.max() < 0.03


def test_header_roundtrip(q, test_x, tmp_path):
    path = tmp_path / "model_weights.h"
    export.write_header(q, path)
    back = export.read_header(path)
    for row in test_x[:20]:
        assert export.quantized_forward(back, row) == export.quantized_forward(q, row)
    assert all(np.array_equal(a, b) for a, b in zip(q.W, back.W))


def test_flash_budget(q):
    assert export.flash_bytes(q) <= 60_000


def test_header_contains_progmem_arrays(q, tmp_path):
    path = tmp_path / "model_weights.h"
    export.write_header(q, path)
    text = path.read_text()
    for token in ("PROGMEM", "const int8_t W1[1536]", "const int8_t W2A[16384]", "const int8_t W2B[16384]",
                  "const float SCALER_STD[6]", "#define MLP_L1 256"):
        assert token in text


def test_committed_weights_header_matches_artifacts(q, test_x):
    committed = export.read_header(export.HEADER_PATH)
    for row in test_x[:3]:
        assert export.quantized_forward(committed, row) == pytest.approx(export.quantized_forward(q, row), abs=1e-6)
