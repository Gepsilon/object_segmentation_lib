import sys, os
import torch
import cv2
import supervision as sv



def path_to_weights(chkp):
    return os.path.abspath(f'model_zoo/{chkp}')


def is_python_module(path):
    return os.path.isfile(os.path.join(path, '__init__.py'))


def load_python_modules(path):
    [sys.path.append(os.path.abspath(x[0])) for x in os.walk(path) if is_python_module(x[0])]


def inference(chosen_model, data_set, weight):
    if chosen_model == "YOLOv8":
        from ultralytics import YOLO
        model = YOLO(weight)
        results = model(data_set, save=True)
        return results
    if chosen_model == "YOLOv5":
        from yolov5.segment.predict import run
        run(weights=weight, source=data_set)

    if chosen_model == "FastSAM":

        FastSAM_path = 'FastSAM'
        sys.path.append(os.path.abspath(FastSAM_path))
        load_python_modules(FastSAM_path)
        os.chdir(FastSAM_path)
        print(os.getcwd())
        os.system(f'python Inference.py --model_path {weight} --img_path {data_set} --output ./runs/ ')

    if chosen_model == "SAM":
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        # -------------------- Load Model----------------------------
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h" if weight == "sam_vit_h_4b8939.pth" else "vit_l"
        check_point_path = path_to_weights(weight)
        sam = sam_model_registry[MODEL_TYPE](checkpoint=check_point_path).to(device=DEVICE)

        mask_generator = SamAutomaticMaskGenerator(sam)

        image_bgr = cv2.imread(data_set)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        sam_result = mask_generator.generate(image_rgb)
        #    --------------------------------------------
        #    Results visualisation with Supervision
        #    --------------------------------------------
        mask_annotator = sv.MaskAnnotator(color_map='index')

        detections = sv.Detections.from_sam(sam_result=sam_result)

        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        sv.plot_images_grid(
            images=[image_bgr, annotated_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )


if __name__ == '__main__':
    inference("FastSAM","D:/object_segmentation_lib/data_set/u.jpg",path_to_weights('FastSAM-x.pt'))
