"""SystemNone.

Shadow today's classifier (an LLM or a System One model), label the logged inputs
with a judge you are allowed to train on, distill the judge's soft labels into a
small encoder, calibrate it, and serve it behind a confidence cascade.
"""

__version__ = "0.1.0"

from .config import Config, TaskSpec, load_config  # noqa: F401
from .shadow import ShadowLogger, shadow  # noqa: F401
