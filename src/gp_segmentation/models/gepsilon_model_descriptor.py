from dataclasses import dataclass
from typing import List

from gp_segmentation.models.parameter_descriptor import ParameterDescriptor


@dataclass
class GepsilonModelDescriptor:
    name: str
    parameters: List[ParameterDescriptor]
