"""Rewired recurrent-network control.

The learned recurrent matrix ``W_rec`` (``W_rec[i, j]`` = weight from neuron ``j``
to neuron ``i``) carries two kinds of information:

* **marginal / distributional** information - the multiset of recurrent weights,
  and possibly each neuron's incoming or outgoing weight distribution;
* **relational** information - *which specific* presynaptic neuron connects to
  *which specific* postsynaptic neuron.

This control destroys the relational structure while preserving the weight
distribution as much as is reasonably possible, so that a surviving
representation-function relationship cannot be attributed to the specific
learned wiring. Three rewiring modes are provided:

=================  ============================  ============================
mode               preserved                     destroyed
=================  ============================  ============================
``global``         the exact multiset of all      all per-neuron in/out
                   recurrent weights              marginals **and** all
                                                   neuron-to-neuron relations
``rowwise``        the exact multiset **and**     each neuron's **outgoing**
                   each neuron's **incoming**     marginal, and all specific
                   weight multiset                neuron-to-neuron relations
``columnwise``     the exact multiset **and**     each neuron's **incoming**
                   each neuron's **outgoing**     marginal, and all specific
                   weight multiset                neuron-to-neuron relations
=================  ============================  ============================

In every mode the diagonal (self-connections) is left as-is; if self-connections
are structurally disabled (``zero_self_connection`` with a zeroed self-mask) the
diagonal is re-zeroed so the architecture is unchanged.

Why this matters scientifically
-------------------------------
The recurrent representation blocks are marginal summaries
(``recurrent_in`` = statistics of the row ``W_rec[j, :]``; ``recurrent_out`` =
statistics of the column ``W_rec[:, j]``). Under a ``rowwise`` shuffle the
``recurrent_in`` block is **exactly unchanged** while the network dynamics change,
so any *loss* of the representation-function relationship there must come from the
function side (the relational structure that produced the function), not from a
change in the representation. Under a ``global`` shuffle both sides change.

The rewired network is a **different** network with a different function, so both
its representation and its functional fingerprint are recomputed and compared
within the rewired condition - exactly like the before/after-learning comparison.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

REWIRE_MODES = ("global", "rowwise", "columnwise")

_PRESERVED_DESTROYED: dict[str, tuple[str, str]] = {
    "global": (
        "exact multiset of all n_hidden^2 recurrent weights (and their total sum)",
        "every per-neuron incoming/outgoing weight marginal and all specific "
        "neuron-to-neuron (relational) structure",
    ),
    "rowwise": (
        "exact multiset of all weights AND every neuron's exact incoming weight "
        "multiset (so the recurrent_in block is unchanged)",
        "each neuron's outgoing weight marginal and all specific neuron-to-neuron "
        "(relational) structure",
    ),
    "columnwise": (
        "exact multiset of all weights AND every neuron's exact outgoing weight "
        "multiset (so the recurrent_out block is unchanged)",
        "each neuron's incoming weight marginal and all specific neuron-to-neuron "
        "(relational) structure",
    ),
}


def rewire_recurrent(model: Any, *, mode: str = "global", seed: int = 0) -> Any:
    """Return a copy of ``model`` with its recurrent matrix rewired.

    ``mode`` is one of :data:`REWIRE_MODES`. Only ``w_rec`` is changed; every
    other parameter is copied verbatim, so the input/readout connectivity and the
    architecture are identical to the original.
    """
    import torch

    if mode not in REWIRE_MODES:
        raise ValueError(f"Unknown rewire mode {mode!r}; choose from {list(REWIRE_MODES)}")

    w = model.w_rec.detach().cpu().numpy().astype(np.float64).copy()
    n = int(w.shape[0])
    rng = np.random.default_rng(int(seed))

    if mode == "global":
        w_new = w.reshape(-1)[rng.permutation(n * n)].reshape(n, n)
    elif mode == "rowwise":
        w_new = np.empty_like(w)
        for i in range(n):
            w_new[i, :] = w[i, rng.permutation(n)]
    else:  # columnwise
        w_new = np.empty_like(w)
        for j in range(n):
            w_new[:, j] = w[rng.permutation(n), j]

    # Preserve a structurally-disabled self-connection (a constant, not learned).
    self_connection_disabled = False
    mask = getattr(model, "self_mask", None)
    if mask is not None:
        m = mask.detach().cpu().numpy()
        if float(np.min(np.diag(m))) == 0.0:
            w_new = w_new * m
            self_connection_disabled = True

    new_model = copy.deepcopy(model)
    with torch.no_grad():
        new_model.w_rec.copy_(
            torch.as_tensor(w_new, dtype=model.w_rec.dtype, device=model.w_rec.device)
        )
    return new_model


def rewiring_report(original: Any, rewired: Any, *, mode: str) -> dict[str, Any]:
    """Document exactly what a rewiring preserved and destroyed.

    Purely descriptive: compares the original and rewired recurrent matrices.
    """
    a = original.w_rec.detach().cpu().numpy().astype(np.float64)
    b = rewired.w_rec.detach().cpu().numpy().astype(np.float64)
    n = int(a.shape[0])

    def _row_multisets_equal() -> bool:
        for i in range(n):
            if not np.allclose(np.sort(a[i, :]), np.sort(b[i, :])):
                return False
        return True

    def _col_multisets_equal() -> bool:
        for j in range(n):
            if not np.allclose(np.sort(a[:, j]), np.sort(b[:, j])):
                return False
        return True

    preserved, destroyed = _PRESERVED_DESTROYED[mode]
    return {
        "mode": mode,
        "n_hidden": n,
        "n_recurrent_weights": int(n * n),
        "fraction_positions_changed": float(np.mean(a != b)),
        "weight_multiset_preserved": bool(np.allclose(np.sort(a.ravel()), np.sort(b.ravel()))),
        "row_multisets_preserved": bool(_row_multisets_equal()),
        "column_multisets_preserved": bool(_col_multisets_equal()),
        "row_sums_preserved": bool(np.allclose(a.sum(axis=1), b.sum(axis=1))),
        "column_sums_preserved": bool(np.allclose(a.sum(axis=0), b.sum(axis=0))),
        "self_connection_disabled": bool(
            getattr(original, "cfg", None) is not None
            and float(getattr(original, "self_mask", None).detach().cpu().numpy().diagonal().min()) == 0.0
        ) if getattr(original, "self_mask", None) is not None else False,
        "preserved_summary": preserved,
        "destroyed_summary": destroyed,
    }