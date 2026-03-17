import numpy as np
import cv2
from PIL import Image


def detect_image(image):

    # convert PIL image to numpy
    img = np.array(image)

    # resize for processing
    img = cv2.resize(img, (128, 128))

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate noise / sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # simple logic for prototype
    if laplacian_var < 50:
        result = "Fake"
        confidence = 80 + int(np.random.randint(5, 15))
    else:
        result = "Real"
        confidence = 80 + int(np.random.randint(5, 15))

    if laplacian_var < 50:
        result = "Fake"
        confidence = 85
        reason = "Low texture / blur detected"
    else:
        result = "Real"
        confidence = 87
        reason = "Normal texture"

    return result, confidence, reason
