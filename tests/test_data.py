from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from pathfrontier import data
from pathfrontier.data import (
    TileRecord,
    iter_split,
    load_split,
    make_synthetic,
    records_to_parquet,
    select_rows,
    split_path,
)
from pathfrontier.schema import CLASSES


def _features(records: list[TileRecord]) -> np.ndarray:
    """Cheap hand features: mean/std per channel + fraction of dark pixels."""
    feats = []
    for r in records:
        a = np.asarray(r.image).astype(np.float32) / 255.0
        feats.append(np.concatenate([a.mean((0, 1)), a.std((0, 1)), [(a.mean(2) < 0.4).mean()]]))
    return np.array(feats)


def test_make_synthetic_shapes_and_labels(synthetic_records):
    assert len(synthetic_records) == 64
    for r in synthetic_records:
        assert isinstance(r.image, Image.Image)
        assert r.image.size == (96, 96) and r.image.mode == "RGB"
        assert r.tile_id.startswith("pcam:test:")
        assert r.label in (0, 1)
    labels = [r.label for r in synthetic_records]
    assert labels.count(0) == labels.count(1) == 32


def test_make_synthetic_deterministic():
    a = make_synthetic("pcam", n=8, seed=3)
    b = make_synthetic("pcam", n=8, seed=3)
    c = make_synthetic("pcam", n=8, seed=4)
    assert [r.label for r in a] == [r.label for r in b]
    assert all(np.array_equal(np.asarray(x.image), np.asarray(y.image)) for x, y in zip(a, b))
    assert not all(np.array_equal(np.asarray(x.image), np.asarray(y.image)) for x, y in zip(a, c))


def test_make_synthetic_crc_all_classes(synthetic_records_crc):
    labels = [r.label for r in synthetic_records_crc]
    assert sorted(set(labels)) == list(range(len(CLASSES["crc"])))
    assert all(r.image.size == (96, 96) for r in synthetic_records_crc)
    assert synthetic_records_crc[0].tile_id.startswith("crc:test:")


def test_synthetic_classes_are_separable(synthetic_records):
    X = _features(synthetic_records)
    y = np.array([r.label for r in synthetic_records])
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=4).mean()
    assert acc > 0.9, f"synthetic pcam classes not separable: acc={acc:.2f}"


def test_synthetic_crc_classes_separable(synthetic_records_crc):
    X = _features(synthetic_records_crc)
    y = np.array([r.label for r in synthetic_records_crc])
    acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=3).mean()
    assert acc > 0.5, f"synthetic crc 9-class accuracy too low: acc={acc:.2f} (chance 0.11)"


def test_make_synthetic_rejects_bad_dataset():
    with pytest.raises(ValueError):
        make_synthetic("foo")


# ---------------------------------------------------------------------------------------------
# select_rows: seeded shuffle, prefix property, balance
# ---------------------------------------------------------------------------------------------
def test_select_rows_prefix_and_balance():
    labels = np.array([0] * 70 + [1] * 30)
    rows = np.arange(100)
    a = select_rows(labels, rows, limit=10, seed=0, balanced=False, n_classes=2)
    b = select_rows(labels, rows, limit=40, seed=0, balanced=False, n_classes=2)
    assert list(a) == list(b[:10])  # prefix of the same permutation
    c = select_rows(labels, rows, limit=10, seed=1, balanced=False, n_classes=2)
    assert list(a) != list(c)
    full = select_rows(labels, rows, limit=None, seed=0, balanced=False, n_classes=2)
    assert list(full) == list(rows)
    bal = select_rows(labels, rows, limit=20, seed=0, balanced=True, n_classes=2)
    assert len(bal) == 20 and np.bincount(labels[bal]).tolist() == [10, 10]
    bal_all = select_rows(labels, rows, limit=None, seed=0, balanced=True, n_classes=2)
    assert np.bincount(labels[bal_all]).tolist() == [30, 30]


# ---------------------------------------------------------------------------------------------
# parquet round trip through load_split / iter_split (no network: files pre-written in tmp)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def fake_data_dir(tmp_path, monkeypatch):
    recs = make_synthetic("pcam", n=40, seed=7)
    records_to_parquet(recs, tmp_path / "pcam" / "test.parquet")
    crc = make_synthetic("crc", n=45, seed=7)
    records_to_parquet(crc, tmp_path / "crc" / "train.parquet")
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    return tmp_path, recs, crc


def test_load_split_roundtrip(fake_data_dir):
    tmp, recs, _ = fake_data_dir
    assert split_path("pcam", "test") == tmp / "pcam" / "test.parquet"
    out = load_split("pcam", "test")
    assert len(out) == 40
    assert [r.tile_id for r in out] == [f"pcam:test:{i}" for i in range(40)]
    assert [r.label for r in out] == [r.label for r in recs]
    assert np.array_equal(np.asarray(out[5].image), np.asarray(recs[5].image))
    assert out[0].image.mode == "RGB"


def test_load_split_limit_seed_balanced(fake_data_dir):
    a = load_split("pcam", "test", limit=8, seed=0)
    b = load_split("pcam", "test", limit=16, seed=0)
    assert [r.tile_id for r in a] == [r.tile_id for r in b[:8]]
    assert [r.tile_id for r in a] != [f"pcam:test:{i}" for i in range(8)]  # shuffled
    bal = load_split("pcam", "test", limit=10, seed=0, balanced=True)
    assert sorted(r.label for r in bal) == [0] * 5 + [1] * 5
    streamed = list(iter_split("pcam", "test", limit=16, seed=0))
    assert sorted(r.tile_id for r in streamed) == sorted(r.tile_id for r in b)
    ids = [int(r.tile_id.split(":")[-1]) for r in streamed]
    assert ids == sorted(ids)  # iter_split yields in file order


def test_crc_valid_is_holdout_of_train(fake_data_dir):
    train = load_split("crc", "train")
    valid = load_split("crc", "valid")
    assert len(valid) == round(45 * data.CRC_HOLDOUT_FRAC)
    assert len(train) + len(valid) == 45
    t_rows = {r.tile_id.split(":")[-1] for r in train}
    v_rows = {r.tile_id.split(":")[-1] for r in valid}
    assert not (t_rows & v_rows)
    assert all(r.tile_id.startswith("crc:valid:") for r in valid)
    assert load_split("crc", "valid")[0].tile_id == valid[0].tile_id  # deterministic


def test_bad_args():
    with pytest.raises(ValueError):
        load_split("nope", "test")
    with pytest.raises(ValueError):
        load_split("pcam", "dev")
