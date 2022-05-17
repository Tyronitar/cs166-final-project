from tkinter import W
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
import dlib
from utils import shape2np, perpendicular_vector

landmark_dict = {
    "left_eye": np.arange(36, 42),
    "left_eyebrow": np.arange(17, 22),
    "right_eye": np.arange(42, 48),
    "right_eyebrow": np.arange(22, 27),
    "nose": np.arange(31, 36),
    "nose_bridge": np.arange(27, 31),
    "lips_inner": np.arange(60, 68),
    "lips_outer": np.arange(48, 60),
    "face": np.arange(0, 17),
    # "test": np.array([27, 28])  # Single line for testing algorithm
}

# Description of parameters
"""If a is barely greater than zero, then if the distance from the line to the pixel
is zero, the strength is nearly infinite. With this value for a, the user knows that
pixels on the line will go exactly where he wants them. Values larger than that will
yield a more smooth warping, but with less precise control."""

"""The variable b determines how the relative strength of different lines Falls off
with distance. If it is large, then every pixel will be affected only by the line
nearest it. Ifb is zero, then each pixel will be affected by all lines equally. Values
of b in the range [0.5, 2] are the most useful. """

"""The value of p is typically in the range [0, 1]; if it is zero, then all lines have
the same weight. if it is one, then longer lines have a greater relative weight than
shorter lines."""
PARAMETERS = {
    'a': 1,
    'b': 2.0,
    'p': 0.5
}


def display_landmarks_and_lines(image, landmarks, lines, fname='img\\out\\temp.png'):
    img = image.copy()

    thickness = max(1, img.shape[1] // 250)
    for start, end in lines:
        cv2.line(img, start, end, (255, 0, 0), thickness)
    for x, y in landmarks:
        cv2.circle(img, (x, y), thickness, (0, 0, 255), -1)
    
    cv2.imwrite(fname, img)


def detect_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bbox = detector(gray)[0]

    shape = predictor(gray, bbox)
    landmarks = shape2np(shape)

    return landmarks


def get_landmark_lines():
    lines = []
    for pts in landmark_dict.values():
        lines.extend(np.stack([pts[:-1], pts[1:]]).T)
    return np.array(lines)


def detect_features(image, detector, predictor):
    landmarks = detect_landmarks(image, detector, predictor)
    line_ids = get_landmark_lines()
    
    # Define lines PQ
    PQ = np.zeros((len(line_ids), 2, 2), dtype=int)
    for i, (start, end) in enumerate(line_ids):
        PQ[i, 0, :] = landmarks[start]
        PQ[i, 1, :] = landmarks[end]

    return landmarks, PQ


def visualize(image, detector, predictor, fname='img\\out\\temp.png'):
    landmarks = detect_landmarks(image, detector, predictor)
    line_ids = get_landmark_lines()
    lines = []
    for start, end in line_ids:
        lines.append([landmarks[start], landmarks[end]])
    
    landmarks, PQ = detect_features(image, detector, predictor)

    display_landmarks_and_lines(image, landmarks, PQ, fname=fname)


def morph(I0, I1, detector, predictor, p=0.5, a=1.0, b=1.0):
    # Get featue lines for both faces
    _, PQ = detect_features(I1, detector, predictor)
    _, PQ_ = detect_features(I0, detector, predictor)
    P, Q = PQ[:, 0], PQ[:, 1]
    P_, Q_ = PQ_[:, 0], PQ_[:, 1]

    # Output "destination" image
    dst = np.zeros((I0.shape[0] * I0.shape[1], I0.shape[2]))

    # All pixel coordinates
    x, y = np.meshgrid(np.arange(I0.shape[0]), np.arange(I0.shape[1]))
    X = np.dstack([x, y])

    # Directed vectors, their norms, and perpendicular vectors
    pq_vec = Q - P
    pq_vec_ = Q_ - P_
    pq_norm = np.sqrt(np.sum(pq_vec * pq_vec, axis=1))
    pq_norm_ = np.sqrt(np.sum(pq_vec_ * pq_vec_, axis=1))
    pq_perp = perpendicular_vector(pq_vec)
    pq_perp_ = perpendicular_vector(pq_vec_)

    # Change X's shape to allow it to operate with lists of vectors
    X = X.reshape(-1, 1, 2)

    # Compute u, v using PQ
    u = np.sum((X - P) * pq_vec, axis=-1) / (pq_norm ** 2)
    v = np.sum((X - P) * pq_perp, axis=-1) / pq_norm

    # Compute X' from u, v, and PQ'
    X_ = P_[np.newaxis, ...] +\
         u[..., np.newaxis] * pq_vec_[np.newaxis, ...] +\
         v[..., np.newaxis] * pq_perp_[np.newaxis, ...] / pq_norm_.reshape(1, -1, 1)

    # Compute Displacement 
    D = X_ - X

    # Compute distance from the line using trick from the paper based on value of u
    # u \ in [0, 1] => abs(v), u < 0 => distance from P, u > 1 => distance from Q
    use_P_dist = u < 0
    p_dist = np.sqrt(np.sum((X - P) * (X - P), axis=-1))
    use_Q_dist = u > 1
    q_dist = np.sqrt(np.sum((X - Q) * (X - Q), axis=-1))
    use_v = np.logical_and(0 <= u, u <= 1)
    dist = use_P_dist * p_dist + use_Q_dist * q_dist + use_v * np.abs(v)

    # Compute weight of each displacement
    weight = ((pq_norm ** p) / (a + dist)) ** b

    # Position to use in source image is weighted sum of displacements added to position
    DSUM = np.sum(D * weight[..., np.newaxis], axis=1)
    weightsum = np.sum(weight, axis=1)
    X_ = X.squeeze() + DSUM / weightsum[..., np.newaxis]

    # Get the morphed image using the positions in X_
    for j in range(I0.shape[2]):
        dst[..., j] = map_coordinates(I0[..., j], X_[:, ::-1].T, mode='nearest')
    dst = dst.reshape(I0.shape)
    
    return dst
