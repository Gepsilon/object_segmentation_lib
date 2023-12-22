from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import List

import cv2 as cv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

@dataclass
class GepsilonSource(ABC):

    @abstractmethod
    def plot(self):
        pass


@dataclass
class LocalImageSource(GepsilonSource):
    img_path: str

    def plot(self):
        # image = cv2.imread(self.img_path)
        image = Image.open(self.img_path)
        image = np.array(image, dtype=np.uint8)
        fig = plt.figure(figsize=(16, 16))
        plt.axis("off")
        plt.imshow(image)


@dataclass
class Box:
    x: float
    y: float
    h: float
    w: float


    @staticmethod
    def from_xyxy(xyxy: List[float]):
        return xyxy

    @staticmethod
    def from_xywh(xywh: List[float]):
        return Box(
            x=xywh[0],
            y=xywh[1],
            w=xywh[2],
            h=xywh[3]
        )


@dataclass
class Segmantation:
    source: GepsilonSource
    box: Box
    confidence: float
    label: str


@dataclass
class SegmentationOutput:
    detections: List[Segmantation]

    def save(self):
        pass

    def save_img(self, path):
        self.show()
        plt.savefig(path)

    def show(self, line_width=2, color='lawngreen'):
        for source, detections_by_source in groupby(self.detections, lambda x: x.source):
            source.plot()
            ax = plt.gca()
            for detection in detections_by_source:
                text = "{}: {:.2f}".format(detection.label, detection.confidence)
                patch = plt.Rectangle(
                    [detection.box.x, detection.box.y], detection.box.w, detection.box.h, fill=False, edgecolor=color,
                    linewidth=line_width
                )
                ax.add_patch(patch)
                ax.text(
                    detection.box.x,
                    detection.box.y,
                    text,
                    bbox={"facecolor": color, "alpha": 0.8},
                    clip_box=ax.clipbox,
                    clip_on=True,
                )
        plt.interactive(True)
        plt.show()






