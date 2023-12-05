from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class RayBundle:
    """A bundle of ray parameters."""

    origins: Tensor  # "*batch 3"
    """Ray origins (XYZ)"""
    directions: Tensor  # "*batch 3"]
    """Unit ray direction vector"""
    rays_index: Tensor = None
    pixel_area: Tensor = None  # "*batch 1"]
    """Projected area of pixel a distance 1 away from origin"""
    camera_indices: Optional[Tensor] = None  # , "*batch 1"
    """Camera indices"""
    nears: Optional[Tensor] = None  # , "*batch 1"
    """Distance along ray to start sampling"""
    fars: Optional[Tensor] = None  # "*batch 1"
    """Rays Distance along ray to stop sampling"""
    # metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(
    #     default_factory=dict
    # )
    # """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    # times: Optional[Float[Tensor, "*batch 1"]] = None
    # """Times at which rays are sampled"""


@dataclass
class RaySample:
    """Sample points on rays"""

    # sample points
    points: Tensor = None  # N x 3

    # points position to the origin  t_positions = (t_starts + t_ends) / 2.0
    positions: Tensor = None  # N x 3

    # sample intervals:   t_intervals = t_ends - t_starts
    intervals: Tensor = None

    # ray direction
    directions: Tensor = None
