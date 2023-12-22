from abc import abstractmethod, ABC
from typing import Dict, Any

from gp_segmentation.exception import UnknownParam, MissingRequiredParam, ValidationFailed
from gp_segmentation.models.segmantation_output import SegmentationOutput, GepsilonSource
from gp_segmentation.models.gepsilon_model_descriptor import GepsilonModelDescriptor
from gp_segmentation.models.parameter_descriptor import ParameterDescriptor




class GepsilonModel(ABC):
    """
    Gepsilon model represent an abstraction of an ML model using in Gepsilon.
    It provides a common API to many ML gp_segmentation.
    """

    def __init__(self, descriptor: GepsilonModelDescriptor):
        self.descriptor = descriptor
        self.params_map: Dict[str, ParameterDescriptor] = {param.name: param for param in descriptor.parameters}

    def resolve(self, param: str, params: Dict[str, Any]) -> Any:
        """
        Resolves a param for the model based on the descriptor and params provided.
        :param param: name of the ML model parameter
        :param params: Mapping of parameter names and values
        :return: the value for the parameter named 'param'
        """
        if not param in self.params_map:
            raise UnknownParam()

        value = None
        parameter_descriptor = self.params_map[param]
        if not param in params:
            if parameter_descriptor.required and parameter_descriptor.default_value is None:
                raise MissingRequiredParam()
            elif parameter_descriptor.required:
                value = parameter_descriptor.default_value

        value = params[param]
        if parameter_descriptor.validator(value):
            return value

        raise ValidationFailed()

    @abstractmethod
    def predict(self, source: GepsilonSource, params: Dict[str, Any]) -> SegmentationOutput:
        pass

    @abstractmethod
    def train(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        pass

    @abstractmethod
    def eval(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        pass
