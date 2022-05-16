import numpy as np
import cv2
import dlib

def shape2np(shape):
    shape_np = np.zeros((shape.num_parts, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    return shape_np