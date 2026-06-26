"""commoncalls — phylogenetic age estimation of common animal calls.

See docs/data-format.md for the input contract. The pipeline turns audio
embeddings + a dated phylogeny into an estimated age for each "reference call",
via ancestral-state reconstruction.
"""

from commoncalls.config import PipelineConfig, ReconstructionMode

__all__ = ["PipelineConfig", "ReconstructionMode"]
__version__ = "0.1.0"
