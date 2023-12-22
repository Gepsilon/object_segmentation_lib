from typing import Dict, Any

import cv2
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import zipfile
import os
from gp_segmentation.exception import UnsupportedSource
from gp_segmentation.gepsilon_model import GepsilonModel
from gp_segmentation.models.segmantation_output import SegmentationOutput, Segmantation, LocalImageSource, Box, GepsilonSource
from gp_segmentation.models.gepsilon_model_descriptor import GepsilonModelDescriptor
from gp_segmentation.models.parameter_descriptor import ParameterDescriptor
from gp_utils import flat_map
from gp_utils.validators import SkipValidation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

class EfficientSam(GepsilonModel):
    @staticmethod
    def model_name():
        return 'efficient_sam'

    def __init__(self):
        super().__init__(
            GepsilonModelDescriptor(
                name=EfficientSam.model_name(),
                parameters=[
                    ParameterDescriptor(name='weights', description='', validator=SkipValidation, required=True,
                                        default_value='efficient_sam_vitt.pt')
                ]
            )
        )

    def predict(self, source: GepsilonSource, params: Dict[str, Any]) -> SegmentationOutput:
        from efficient_sam import build_efficient_sam_vitt
        os.chdir('EfficientSam')



        models = {}

        # Build the EfficientSAM-Ti model.
        models['efficientsam_ti'] = build_efficient_sam_vitt()


        # load an image
        data_set = ''
        if isinstance(source, LocalImageSource):
            data_set = source.img_path
        else:
            raise UnsupportedSource()
        sample_image_np = np.array(Image.open(data_set))
        sample_image_tensor = transforms.ToTensor()(sample_image_np)
        # Feed a few (x,y) points in the mask as input.

        input_points = torch.tensor([[[[300, 400], [650, 350]]]])
        input_labels = torch.tensor([[[1, 1]]])

        # Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
        for model_name, model in models.items():
            print('Running inference using ', model_name)
            predicted_logits, predicted_iou = model(
                sample_image_tensor[None, ...],
                input_points,
                input_labels,
            )
            # The masks are already sorted by their predicted IOUs.
            # The first dimension is the batch size (we have a single image. so it is 1).
            # The second dimension is the number of masks we want to generate (in this case, it is only 1)
            # The third dimension is the number of candidate masks output by the model.
            # For this demo we use the first mask.
            mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
            masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:, :, None]
            Image.fromarray(masked_image_np).save(f"D:/object_segmentation_lib/data_sample/{model_name}k.png")

        return masked_image_np

    def train(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self

    def eval(self, source: GepsilonSource, params: Dict[str, Any]) -> 'GepsilonModel':
        # TODO: implement me!
        return self