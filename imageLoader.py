import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum

samplePicsDir = "samplePhotos/"
sampleWoodPicsDir = "woodSamples/"
outputPicsDir = "outputPhotos/"
class SamplePicture(Enum):
    ZION = samplePicsDir + "Zion 1.jpg"
    GIRL_FACE = samplePicsDir + "Girl Face 1.jpg"
    PARROTS = samplePicsDir + "parrots.jpg"
    JUNGLE_BOOK = samplePicsDir + "jungle-book.jpg"
    JAKE_EM = samplePicsDir + "jake-em.jpg"


class WoodType(Enum):
    ALDER = sampleWoodPicsDir + "alder.jpg"
    CHERRY = sampleWoodPicsDir + "black-cherry.jpg"
    BLACK_WALNUT = sampleWoodPicsDir + "black-walnut.jpg"
    MAPLE = sampleWoodPicsDir + "hard-maple.jpg"
    PADAUK = sampleWoodPicsDir + "padauk.jpg"
    POPLAR = sampleWoodPicsDir + "poplar.jpg"
    PURPLEHEART = sampleWoodPicsDir + "purpleheart.jpg"
    WALNUT = sampleWoodPicsDir + "walnut.jpg"

def wood_type_from_name(name):
    return next(name for name, value in vars(WoodType).items() if value == 1)


def read_file(filename, display_img=False):
    img = cv2.imread(filename.value)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if display_img:
        plt.imshow(img)
        plt.show()
    return img

def write_output_picture(img, filename):
    output_path = outputPicsDir + filename
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)

