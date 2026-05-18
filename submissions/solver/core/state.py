"""Minimal placement state container for inter-stage handoff."""

from dataclasses import dataclass, field

import torch


@dataclass
class PlacementState:
    """Thin container coupling a placement tensor to its pipeline stage.

    Fields
    ------
    positions : torch.Tensor
        Shape [num_macros, 2], center coordinates in microns.
    stage : str
        Informal label for the pipeline stage that produced this placement.
        Suggested values: ``"raw"`` (straight from placer), ``"legal"``.
    """

    positions: torch.Tensor
    stage: str = "raw"
