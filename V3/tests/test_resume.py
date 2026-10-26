"""Resume-from-checkpoint: state restore, bit-identical continuation, CLI resolution.

CPU-only, tiny synthetic batches; no dataset, no real training run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from v3.config import TimingConfig, V3Config
from v3.data import Split
from v3.experiment import load_checkpoint_payload, resume_state_from_payload
from v3.model import VectorNeuronPopulation
from v3.train import fit, set_seed


class _ToyStore:
    """Deterministic store: fixed dense tensors + labels, no HDF5."""

    def __init__(self, n: int = 24, n_classes: int = 3, n_bins: int = 4, n_inputs: int = 5,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.x = (rng.random((n, n_bins, n_inputs)) < 0.25).astype(np.float32)
        self.labels = rng.integers(0, n_classes, n).astype(np.int64)
        self.speakers = np.zeros(n, dtype=np.int64)

    def batch(self, indices):
        return self.x[np.asarray(indices, dtype=np.int64)]

    def labels_of(self, indices):
        return self.labels[np.asarray(indices, dtype=np.int64)]


def _cfg(**kw) -> V3Config:
    base = dict(
        seed=7, n_neurons=4, state_dim=6, mix_rank=3, n_inputs=5, n_classes=3,
        timing=TimingConfig(8.0, 2.0),          # T = 4
        device="cpu", dtype="float32", amp=False,
        batch_size=6, epochs=1, val_fraction=0.0, shuffle_train=True,
        lr_schedule="none", max_train_samples=0, grad_checkpoint_chunks=2,
        tag="toy",
    )
    base.update(kw)
    return V3Config(**base)


def _split(store: _ToyStore) -> Split:
    idx = np.arange(store.labels.size)
    return Split("fit", idx, store.labels[idx], store.speakers[idx])


def _train(cfg: V3Config, store: _ToyStore, model=None, **kwargs):
    if model is None:
        set_seed(cfg.seed)
        model = VectorNeuronPopulation(cfg)
    sink = {}
    res = fit(model, store, _split(store), None, cfg, torch.device("cpu"),
              show_progress=False, state_sink=sink, **kwargs)
    return model, res, sink


def test_resume_reproduces_an_uninterrupted_two_epoch_run():
    """epoch 1 -> checkpoint -> resume == one 2-epoch run, bit for bit."""
    store = _ToyStore()
    cfg1, cfg2 = _cfg(epochs=1), _cfg(epochs=2)

    full, _res_full, _ = _train(cfg2, store)
    part, res1, sink = _train(cfg1, store)

    assert sink["epoch"] == 1
    assert sink["optimizer"] and sink["rng"]
    state = {
        **sink,
        "state_dict": {k: v.detach().clone() for k, v in part.state_dict().items()},
        "history": res1.history,
        "best_val_accuracy": res1.best_val_accuracy,
        "best_epoch": res1.best_epoch,
        "best_val_state_dict": res1.state_dict,
    }
    resumed, res2, sink2 = _train(cfg2, store, resume_state=state)

    for key, want in full.state_dict().items():
        assert torch.equal(want, resumed.state_dict()[key]), key
    assert [r["epoch"] for r in res2.history] == [1, 2]
    assert sink2["epoch"] == 2
    # carried-over row of epoch 1 must be the original one
    assert res2.history[0]["loss"] == res1.history[0]["loss"]


def test_checkpoint_payload_round_trip_through_disk(tmp_path):
    store = _ToyStore()
    cfg = _cfg(epochs=1)
    model, res, sink = _train(cfg, store)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "schema": "v3_checkpoint/v2",
            "config": cfg.to_dict(),
            "state_dict": model.state_dict(),
            "history": res.history,
            **sink,
        },
        path,
    )

    payload = load_checkpoint_payload(path)
    state = resume_state_from_payload(payload)
    assert state["epoch"] == 1
    assert state["optimizer"] and state["scheduler"] is None
    assert state["rng"]["torch"] is not None
    assert state["history"] == res.history

    cfg2 = _cfg(epochs=2)
    resumed, res2, sink2 = _train(cfg2, store, resume_state=state)
    assert sink2["epoch"] == 2 and len(res2.history) == 2


def test_load_checkpoint_payload_rejects_foreign_file(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(ValueError):
        load_checkpoint_payload(path)


def test_resume_config_suffixes_tag_and_requires_more_epochs():
    from run_experiment import resume_config

    payload = {"config": _cfg(epochs=1).to_dict(), "epoch": 1}
    cfg = resume_config(payload, ["epochs=2"], quick=False)
    assert cfg.tag == "toy_resumed" and cfg.epochs == 2
    # a resumed config never grows a second suffix
    twice = {"config": _cfg(epochs=1, tag="toy_resumed").to_dict(), "epoch": 1}
    assert resume_config(twice, ["epochs=3"], quick=False).tag == "toy_resumed"
    with pytest.raises(SystemExit):          # nothing to do: already at the last epoch
        resume_config(payload, [], quick=False)
    with pytest.raises(SystemExit):          # --quick would change the model shape
        resume_config(payload, ["epochs=2"], quick=True)
