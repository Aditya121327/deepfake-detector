import random

def detect_image(image):

    result = random.choice(["Real", "Fake"])
    confidence = random.randint(70, 99)

    return result, confidence
