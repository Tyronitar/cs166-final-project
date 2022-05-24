import argparse

import cv2

from morphing import metamorphosis
from utils import resize

parser = argparse.ArgumentParser(
    description='Morph one face into another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("image0", type=str, help='The first image')
parser.add_argument("image1", type=str, help='The second image')
parser.add_argument("--output_name", "-o", type=str, help='Output file destination',
                    default="img\\out\\temp.gif")
parser.add_argument("--load0", type=str, help='File to load landmarks from. If empty, '
                    'detects them maually.', default='')
parser.add_argument("--load1", type=str, help='File to load landmarks from. If empty, '
                    'detects them maually.', default='')
parser.add_argument("--duration", "-d", type=float, help='Output gif duration',
                    default=5.0)
parser.add_argument("--framerate", "-f", type=float, help='The framerate of output gif',
                    default=24)
parser.add_argument("--size", "-s", type=int, help='Image size in pixels. Images will be'\
    ' resized to this size.',
                    default=250)

args = parser.parse_args()

imsize = (args.size, args.size)

print('Loading and resizing images...')
I0 = cv2.imread(args.image0, 1)
if I0 is None:
    raise IOError(f'Problem opening "{args.image0}". Make sure the file exists.')
size0 = I0.shape
I0 = resize(I0, imsize)

I1 = cv2.imread(args.image1, 1)
if I1 is None:
    raise IOError(f'Problem opening "{args.image1}". Make sure the file exists.')
size1 = I1.shape
I1 = resize(I1, imsize)

metamorphosis(
    I0,
    I1,
    size0,
    size1,
    fname=args.output_name,
    duration=args.duration,
    framerate=args.framerate,
    load0=args.load0,
    load1=args.load1)
