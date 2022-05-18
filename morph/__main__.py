import numpy as np
import cv2
import dlib
import argparse
from morphing import metamorphosis
from utils import resize

parser = argparse.ArgumentParser(description='Morph one face into another.')
parser.add_argument("image0", type=str, help='The first image')
parser.add_argument("image1", type=str, help='The second image')
parser.add_argument("--output_name", "-o", type=str, help='Output file destination',
                    default="img\\out\\temp.gif")
parser.add_argument("--duration", "-d", type=float, help='Output gif duration',
                    default=5.0)
parser.add_argument("--framerate", "-f", type=float, help='The framerate of output gif',
                    default=24)
parser.add_argument("--size", "-s", type=int, help='Image size in pixels. Images will be'\
    ' resized to this size.',
                    default=250)

args = parser.parse_args()

imsize = (args.size, args.size)

I0 = resize(cv2.imread(args.image0, 1), imsize)
I1 = resize(cv2.imread(args.image1, 1), imsize)
metamorphosis(I0, I1, fname=args.output_name, duration=args.duration, framerate=args.framerate)
