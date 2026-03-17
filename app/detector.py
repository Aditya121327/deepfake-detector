import numpy as np
import cv2


# ---------- PREPROCESS ----------

def preprocess(image):

    img = np.array(image)

    img = cv2.resize(img, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


# ---------- FEATURE 1 : BLUR ----------

def blur_score(gray):

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


# ---------- FEATURE 2 : EDGES ----------

def edge_score(gray):

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    return np.mean(edges), edges


# ---------- FEATURE 3 : BRIGHTNESS ----------

def brightness_score(gray):

    return np.mean(gray)


# ---------- FEATURE 4 : NOISE ----------

def noise_score(gray):

    noise = np.std(gray)

    return noise


# ---------- FEATURE 5 : COLOR VARIATION ----------

def color_score(img):

    b, g, r = cv2.split(img)

    return (
        np.std(b)
        + np.std(g)
        + np.std(r)
    ) / 3


# ---------- FEATURE 6 : FACE DETECTION ----------

def face_score(gray):

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    return len(faces)


# ---------- PREDICTION ----------

def predict(
    blur,
    edge,
    bright,
    noise,
    color,
    faces
):

    score = 0

    if blur < 40:
        score += 1

    if edge < 15:
        score += 1

    if bright < 50:
        score += 1

    if noise < 20:
        score += 1

    if color < 30:
        score += 1

    if faces == 0:
        score += 1

    confidence = 70 + score * 5

    if score >= 3:
        result = "Fake"
    else:
        result = "Real"

    return result, confidence, score


# ---------- MAIN ----------

def detect_image(image):

    img, gray = preprocess(image)

    blur = blur_score(gray)

    edge, edges = edge_score(gray)

    bright = brightness_score(gray)

    noise = noise_score(gray)

    color = color_score(img)

    faces = face_score(gray)

    result, confidence, score = predict(
        blur,
        edge,
        bright,
        noise,
        color,
        faces
    )

    heatmap = cv2.applyColorMap(
        edges,
        cv2.COLORMAP_JET
    )

    reason = (
        f"Blur={blur:.1f}, "
        f"Edge={edge:.1f}, "
        f"Noise={noise:.1f}, "
        f"Color={color:.1f}, "
        f"Faces={faces}"
    )

    return result, confidence, reason, heatmap
