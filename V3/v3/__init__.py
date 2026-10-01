"""V3: a population of genuinely ``D``-dimensional vector-valued spiking neurons.

See ``V3/README.md``.  The package is deliberately small:

* :mod:`v3.config` -- the single place where every experiment parameter lives
* :mod:`v3.data`   -- SHD event loading / binning and the FIT/VAL/TEST splits
* :mod:`v3.model`  -- the vector-neuron population itself
* :mod:`v3.train`  -- training / evaluation loops
* :mod:`v3.plots`  -- the training and test figures
"""

import os

# The 250-step BPTT graph with chunked gradient checkpointing allocates and frees
# many different-sized tensors.  Expandable segments keep the CUDA caching
# allocator's reserved pool at the true peak (~1.2 GiB at N=64, D=1000, batch 128)
# instead of letting fragmentation grow it towards the 6 GiB device limit.
# Must be set before the first CUDA allocation; setdefault() leaves a user's own
# PYTORCH_CUDA_ALLOC_CONF alone.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import V3Config, describe, parameter_groups  # noqa: E402,F401

__all__ = ["V3Config", "describe", "parameter_groups"]
