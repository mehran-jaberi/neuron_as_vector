"""V3: a population of genuinely ``D``-dimensional vector-valued spiking neurons.

See ``V3/README.md``.  The package is deliberately small:

* :mod:`v3.config` -- the single place where every experiment parameter lives
* :mod:`v3.data`   -- SHD event loading / binning and the FIT/VAL/TEST splits
* :mod:`v3.model`  -- the vector-neuron population itself
* :mod:`v3.train`  -- training / evaluation loops
* :mod:`v3.plots`  -- the training and test figures
"""

from .config import V3Config, describe, parameter_groups  # noqa: F401

__all__ = ["V3Config", "describe", "parameter_groups"]
