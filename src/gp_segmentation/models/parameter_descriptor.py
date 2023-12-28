from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ParameterDescriptor:
    name: str
    description: str
    validator: Callable[[Any], bool]
    required: bool
    default_value: Any
