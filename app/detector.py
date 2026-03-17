import numpy as np
import cv2


def preprocess(image):

    img = np.array(image)

    img = cv2.resize(img, (224, 224))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


def extract_features(gray):

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 50, 150)

    edge_score = np.mean(edges)

    brightness = np.mean(gray)

    return blur, edge_score, brightness, edges


def predict(blur, edge_score, brightness):

    score = 0

    if blur < 50:
        score += 1

    if edge_score < 20:
        score += 1

    if brightness < 60:
        score += 1

    confidence = 80 + score * 5

    if score >= 2:
        result = "Fake"
    else:
        result = "Real"

    return result, confidence


def detect_image(image):

    img, gray = preprocess(image)

    blur, edge_score, brightness, edges = extract_features(gray)

    result, confidence = predict(
        blur,
        edge_score,
        brightness
    )

    heatmap = cv2.applyColorMap(
        edges,
        cv2.COLORMAP_JET
    )

    reason = (
        f"Blur={blur:.1f}, "
        f"Edges={edge_score:.1f}, "
        f"Brightness={brightness:.1f}"
    )

    return result, confidence, reason, heatmap
