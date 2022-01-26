import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

samplePicsDir = "samplePhotos/"
outputPicsDir = "outputPhotos/"
class SamplePicture(Enum):
    ZION = samplePicsDir + "Zion 1.jpg"
    GIRL_FACE = samplePicsDir + "Girl Face 1.jpg"
    PARROTS = samplePicsDir + "parrots.jpg"
    JUNGLE_BOOK = samplePicsDir + "jungle-book.jpg"


def read_file(filename):
    img = cv2.imread(filename.value)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img

def write_output_picture(img, filename):
    output_path = outputPicsDir + filename
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)
