from tkinter import W
import numpy as np
import cv2
import dlib
from utils import shape2np

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


def display_landmarks_and_lines(image, landmarks, lines, fname='img\\out\\temp.png'):
    img = image.copy()

    thickness = img.shape[1] // 250
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


def get_feature_lines(image, detector, predictor):
    landmarks = detect_landmarks(image, detector, predictor)
    line_ids = get_landmark_lines()
    
    # Define lines PQ
    P = []
    Q = []
    for start, end in line_ids:
        P.append(landmarks[start])
        Q.append(landmarks[end])

    return np.array(P), np.array(Q)


def visualize(image, detector, predictor):
    landmarks = detect_landmarks(image, detector, predictor)
    line_ids = get_landmark_lines()
    lines = []
    for start, end in line_ids:
        lines.append([landmarks[start], landmarks[end]])

    display_landmarks_and_lines(image, landmarks, lines)


