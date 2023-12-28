from gp_segmentation.exception import UnknownModel
from gp_segmentation.gepsilon_model import GepsilonModel




class ModelRepository:
    def __init__(self):
        self.gepsilon_models = {}

    def register(self, gepsilon_model: GepsilonModel):
        self.gepsilon_models[gepsilon_model.descriptor.name] = gepsilon_model

    def resolve(self, model_name: str) -> GepsilonModel:
        if model_name in self.gepsilon_models:
            return self.gepsilon_models[model_name]

        raise UnknownModel()
