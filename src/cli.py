import argparse
from os import path
from gp_segmentation.model_repository import ModelRepository
from gp_segmentation.models.segmantation_output import LocalImageSource
from gp_models.gepsilon_yolo_v8 import GepsilonYoloV8
from gp_models.gepsilon_sam import GepsilonSam
from gp_models.efficientSAM import EfficientSam


def create_model_repository():
    model_resolver = ModelRepository()
    model_resolver.register(GepsilonYoloV8())
    model_resolver.register(GepsilonSam())
    model_resolver.register(EfficientSam())
    return model_resolver


def main(weights, source, model):
    while True:

        repository = create_model_repository()
        gepsilon_model = repository.resolve(model)
        output = gepsilon_model.predict(LocalImageSource(path.abspath(f'data_sample/{source}')), {
            'weights': path.abspath(f'D:/object_segmentation_lib/checkpoints/{weights[0]}')
        })

        return output


def parser():
    parser = argparse.ArgumentParser()
    #  parser.add_argument('--mode', type=str, default='predict', help='')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov8n-seg.pt', help='')
    parser.add_argument('--source', type=str, default='k.jpg', help='')
    parser.add_argument('--model', type=str, default='yolo_v8', help='')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parser()
    main(**vars(opt))

