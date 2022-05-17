import numpy as np
import cv2
import dlib
import argparse
from morphing import visualize, morph
from utils import perpendicular_vector, resize

parser = argparse.ArgumentParser(description='Morph one face into another.')
parser.add_argument("image0", type=str, help='The first image')
parser.add_argument("image1", type=str, help='The second image')
parser.add_argument("--output_name", "-o", type=str, help='Output file destination',
                    default="img\\out\\temp.png")

args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

imsize = (500, 500)

I0 = resize(cv2.imread(args.image0, 1), imsize)
I1 = resize(cv2.imread(args.image1, 1), imsize)
visualize(I0, detector, predictor, fname='img\\out\\temp0.png')
visualize(I1, detector, predictor, fname='img\\out\\temp1.png')
morphed = morph(I0, I1, detector, predictor)
cv2.imwrite(args.output_name, morphed)
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
