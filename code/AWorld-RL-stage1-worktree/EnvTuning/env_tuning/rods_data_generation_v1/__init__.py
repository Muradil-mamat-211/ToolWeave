"""RODS Data-Generation Branch V1.

The package is deliberately independent from the frozen Training Branch.  It
consumes ``rods_boundary_seed.v1`` records and emits only fully validated
``rods_validated_candidate.v1`` records.
"""

from .config import GeneratorConfig
from .models import PipelineResult, SeedRecord
from .pipeline import RODSDataGenerationPipeline

__all__ = [
    "GeneratorConfig",
    "PipelineResult",
    "RODSDataGenerationPipeline",
    "SeedRecord",
]
