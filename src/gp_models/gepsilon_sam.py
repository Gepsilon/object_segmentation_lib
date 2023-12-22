from typing import Dict, Any

import cv2

from gp_segmentation.exception import UnsupportedSource
from gp_segmentation.gepsilon_model import GepsilonModel
from gp_segmentation.models.segmantation_output import SegmentationOutput, Segmantation, LocalImageSource, Box, GepsilonSource
from gp_segmentation.models.gepsilon_model_descriptor import GepsilonModelDescriptor
from gp_segmentation.models.parameter_descriptor import ParameterDescriptor
from gp_utils import flat_map
from gp_utils.validators import SkipValidation
import torch

class GepsilonSam(GepsilonModel):
    @staticmethod
    def model_name():
        return 'segment-anything'

    def __init__(self):
        super().__init__(
            GepsilonModelDescriptor(
                name=GepsilonSam.model_name(),
                parameters=[
                    ParameterDescriptor(name='weights', description='', validator=SkipValidation, required=True,
                                        default_value='sam_vit_l_0b3195.pth')
                ]
            )
        )

    def predict(self, source: GepsilonSource, params: Dict[str, Any]) -> SegmentationOutput:


        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        weights = self.resolve('weights', params)

        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h" if weights == "sam_vit_h_4b8939.pth" else "vit_l"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=weights).to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)


        data_set = ''
        if isinstance(source, LocalImageSource):
            data_set = source.img_path
        else:
            raise UnsupportedSource()

        image_bgr = cv2.imread(data_set)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_result = mask_generator.generate(image_rgb)  # bbox ,

        return SegmentationOutput(
            detections=[Segmantation(
                source=LocalImageSource(img_path=data_set),
                confidence=80,
                label='name',
                box=Box.from_xyxy(sam_result[0].get('bbox'))
            )]
        )

    def train(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self

    def eval(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self