from tkinter import W
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
import dlib
from utils import shape2np, perpendicular_vector

landmark_dict = {
    # "left_eye": np.arange(36, 42),
    # "left_eyebrow": np.arange(17, 22),
    # "right_eye": np.arange(42, 48),
    # "right_eyebrow": np.arange(22, 27),
    # "nose": np.arange(31, 36),
    # "nose_bridge": np.arange(27, 31),
    # "lips_inner": np.arange(60, 68),
    # "lips_outer": np.arange(48, 60),
    # "face": np.arange(0, 17),
    "test": np.array([27, 28])  # Single line for testing algorithm
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

"""The value ofp is typically in the range [0, 1]; if it is zero, then all lines have
the same weight. if it is one, then longer lines have a greater relative weight than
shorter lines."""
PARAMETERS = {
    'a': 1e-3,
    'b': 1.0,
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


def morph(I0, I1, detector, predictor):
    _, PQ = detect_features(I1, detector, predictor)
    _, PQ_ = detect_features(I0, detector, predictor)

    dst = np.zeros(I0.shape)
    print(I0.shape)

    for x in range(dst.shape[0]):
        print(f"{x / dst.shape[0] * 100:.2f}%\r", end="")
        for y in range(dst.shape[1]):
            DSUM = np.zeros(2)
            weightsum = 0
            X = np.array([x, y])
            for i in range(len(PQ)):
                P, Q = PQ[i]
                P_, Q_ = PQ_[i]
                pq_norm = np.linalg.norm(Q - P)
                pq_norm_ = np.linalg.norm(Q_ - P_)
                u = (X - P).dot((Q - P)) / (pq_norm ** 2)
                v = (X - P).dot(perpendicular_vector(Q - P)) / pq_norm

                X_i = P_ + u * (Q_ - P_) + v * perpendicular_vector(Q_ - P_) / pq_norm_

                Di = X_i - X
                dist = np.linalg.norm(X - P) if u < 0 else np.linalg.norm(X - Q) if u > 1\
                    else abs(v)
                # dist = np.linalg.norm(np.cross(Q - P, P - X)) / pq_norm

                weight = ((pq_norm ** PARAMETERS['p']) / (PARAMETERS['a'] + dist)) ** PARAMETERS['b']
                DSUM += Di * weight
                weightsum += weight
            X_ = X + DSUM / weightsum
            # print(X_)
            # print(X_.astype(int))
            for j in range(I0.shape[2]):
                dst[x, y, j] = map_coordinates(I0[..., j], [[X_[0]], [X_[1]]], mode='nearest')
            # dst[x, y] = I0[X_[0] % I0.shape[0], X_[1] % I0.shape[1]]
    
    return dst
                