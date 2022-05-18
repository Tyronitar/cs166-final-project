import cv2
import dlib
import numpy as np


def shape2np(shape: dlib.full_object_detection) -> np.ndarray:
    """Convert dlib's full_object_detection object to a numpy array."""
    shape_np = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    return shape_np


def resize(I: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image to specified size."""
    return cv2.resize(I, size, interpolation=cv2.INTER_AREA)


def perpendicular_vector(v: np.ndarray):
    """Find perpendicular vectors for every 2D vector in `v`."""
    return v[:, ::-1] * np.array([1, -1])


def bgr2rgb(I: np.ndarray):
    """Convert an image in BGR format to RGB."""
    B = I[..., 0]
    G = I[..., 1]
    R = I[..., 2]
    return np.dstack([R, G, B])
