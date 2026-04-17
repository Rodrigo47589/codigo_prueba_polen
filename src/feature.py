import cv2
import numpy as np
from pathlib import Path


def extraer_histograma_hsv(ruta_imagen, bins=(8, 8, 8)):
    img = cv2.imread(str(ruta_imagen))
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()