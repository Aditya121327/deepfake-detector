import numpy as np
import cv2


def detect_image(image):

    img = np.array(image)

    img = cv2.resize(img, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    mean_val = np.mean(gray)

    if laplacian_var < 50 or mean_val < 60:
        result = "Fake"
        confidence = 85
        reason = "Low texture / abnormal lighting detected"
    else:
        result = "Real"
        confidence = 88
        reason = "Normal texture detected"

    return result, confidence, reason
