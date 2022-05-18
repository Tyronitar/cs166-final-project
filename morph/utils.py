import numpy as np
import cv2
import dlib

def shape2np(shape):
    shape_np = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    return shape_np


def resize(I, size):
    return cv2.resize(I, size, interpolation=cv2.INTER_AREA)


def perpendicular_vector(v):
    return v[:, ::-1] * np.array([1, -1])


def bgr2rgb(I):
    B = I[..., 0]
    G = I[..., 1]
    R = I[..., 2]
    return np.dstack([R, G, B])