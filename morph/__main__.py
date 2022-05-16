import numpy as np
import cv2
import dlib
import argparse
from morphing import visualize

parser = argparse.ArgumentParser(description='Morph one face into another.')
parser.add_argument("image", type=str, help='The input image to find landmarks')

args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

image = cv2.imread(args.image, 1)
visualize(image, detector, predictor)
quit()

# cap = cv2.VideoCapture(1)

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# while True:
#     success, frame = cap.read()
#     assert success

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect Faces
#     faces = detector(gray)

#     # Detect landmarks for each face
#     for face in faces:
#         shape = predictor(gray, face)

#         shape_np = np.zeros((68, 2), dtype="int")
#         for i in range(0, 68):
#             shape_np[i] = (shape.part(i).x, shape.part(i).y)
#         shape = shape_np

#         # Display the landmarks
#         for i, (x, y) in enumerate(shape):
#         # Draw the circle to mark the keypoint 
#             cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

#     cv2.imshow('video', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
