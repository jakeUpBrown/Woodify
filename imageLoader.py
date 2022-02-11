import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum

samplePicsDir = "samplePhotos/"
sampleWoodPicsDir = "woodSamples/"
outputPicsDir = "outputPhotos/"


class SamplePicture(Enum):
    ZION = "Zion 1.jpg"
    GIRL_FACE = "Girl Face 1.jpg"
    PARROTS = "parrots.jpg"
    JUNGLE_BOOK = "jungle-book.jpg"
    JAKE_EM = "jake-em.jpg"
    JAKE_EM_FILTER = "jake-em-filter.jpg"


class WoodType(Enum):
    ALDER = "alder.jpg"
    CHERRY = "black-cherry.jpg"
    BLACK_WALNUT = "black-walnut.jpg"
    MAPLE = "hard-maple.jpg"
    PADAUK = "padauk.jpg"
    POPLAR = "poplar.jpg"
    PURPLEHEART = "purpleheart.jpg"
    WALNUT = "walnut.jpg"


def wood_type_from_name(name):
    return next(name for name, value in vars(WoodType).items() if value == 1)


def read_sample_photo(filename):
    return read_file(samplePicsDir + filename.value)


def read_sample_wood_pic(filename):
    return read_file(sampleWoodPicsDir + filename.value)


def read_file(filepath, display_img=False):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if display_img:
        plt.imshow(img)
        plt.show()
    return img

def write_output_picture(img, filename):
    output_path = outputPicsDir + filename
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)

