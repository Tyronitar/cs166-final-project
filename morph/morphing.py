import cv2
import dlib
import imageio
import numpy as np
from scipy.ndimage import map_coordinates
import tqdm

from utils import shape2np, perpendicular_vector, bgr2rgb

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
}


def display_landmarks_and_lines(image, landmarks, lines, fname='img\\out\\temp.png'):
    img = image.copy()

    thickness = max(1, img.shape[1] // 250)
    for start, end in lines:
        cv2.line(img, start, end, (255, 0, 0), thickness)
    for x, y in landmarks:
        cv2.circle(img, (x, y), thickness, (0, 0, 255), -1)
    
    cv2.imwrite(fname, img)


def detect_landmarks(
    image: np.ndarray,
    detector,
    predictor: dlib.shape_predictor) -> np.ndarray:
    """Find the landmarks in a face.

    Args:
        image (np.ndarray): Image to find landmarks in
        detector (_type_): Detector to find faces.
        predictor (dlib.shape_predictor): Shape predictor to find landmarks.

    Returns:
        np.ndarray: List of landmarks in [x, y] format.

    Raises:
        ValueError: If no faces are found in input image.
    """
    # Convert image to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) < 1:
        raise ValueError("Image does not contain faces.")
    bbox = faces[0]

    # Find landmarks in first face
    shape = predictor(gray, bbox)
    landmarks = shape2np(shape)

    return landmarks


def get_landmark_lines() -> np.ndarray:
    """Get landmark indices corresponding to feature lines.

    Returns:
        np.ndarray: List of feature line endpoints as landmark indices.
    """
    lines = []
    for pts in landmark_dict.values():
        lines.extend(np.stack([pts[:-1], pts[1:]]).T)
    return np.array(lines)


def detect_features(
    image: np.ndarray,
    detector,
    predictor: dlib.shape_predictor) -> tuple[np.ndarray, np.ndarray]:
    """Find landmarks and feature lines in an image.

    Args:
        image (np.ndarray): Image to find landmarks in
        detector (_type_): Detector to find faces.
        predictor (dlib.shape_predictor): Shape predictor to find landmarks.

    Returns:
        tuple[np.ndarray, np.ndarray]: Array of the landmarks, and the endpoints of the
            feature lines in form [[x_start, y_start], [x_end, y_end]].
    """
    landmarks = detect_landmarks(image, detector, predictor)
    line_ids = get_landmark_lines()
    
    # Define lines PQ
    PQ = np.zeros((len(line_ids), 2, 2), dtype=int)
    for i, (start, end) in enumerate(line_ids):
        PQ[i, 0, :] = landmarks[start]
        PQ[i, 1, :] = landmarks[end]

    return landmarks, PQ


def load_landmarks(path: str, size: tuple[int, int], old_size: tuple[int, int]) -> np.ndarray:
    """Load the landmakrs for an image stored in a .pts file.

    The format of the file is in each line, two integers separated by a space, indicating
    the x, y position of the landmark in the image. There should be 68 such lines, where
    the landmark order follows the specification used by dlib.

    Args:
        path (str): Path to load the landmarks from.
        size (tuple[int, int]): The current size of the image (H x W)
        old_size (tuple[int, int]): The original size of the image (H x W)

    Returns:
        np.ndarray: List of landmarks in [x, y] format.
    """
    items = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            items.append(np.zeros(2))
            items[i][0] = size[0] / old_size[1] * int(parts[0])
            items[i][1] = size[1] / old_size[0] * int(parts[1])
    
    items = np.array(items)
    if len(items) != 68:
        raise ValueError(f'Expected 68 landmark points in "{path}". Found {len(items)}')
    return items


def load_features(path: str, size: tuple[int, int], old_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load landmarks and features for an image.

    Args:
        path (str): Path to load the landmarks from.
        size (tuple[int, int]): The current size of the image (H x W)
        old_size (tuple[int, int]): The original size of the image (H x W)

    Returns:
        tuple[np.ndarray, np.ndarray]: Array of the landmarks, and the endpoints of the
            feature lines in form [[x_start, y_start], [x_end, y_end]].
    """
    landmarks = load_landmarks(path, size, old_size)
    line_ids = get_landmark_lines()
    
    # Define lines PQ
    PQ = np.zeros((len(line_ids), 2, 2), dtype=int)
    for i, (start, end) in enumerate(line_ids):
        PQ[i, 0, :] = landmarks[start]
        PQ[i, 1, :] = landmarks[end]

    return landmarks, PQ


def visualize(image, fname='img\\out\\temp.png', do_landmarks=True, do_lines=True):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    
    landmarks, PQ = detect_features(image, detector, predictor)
    if not do_landmarks:
        landmarks = []
    if not do_lines:
        PQ = []
    display_landmarks_and_lines(image, landmarks, PQ, fname=fname)


def morph(
    I: np.ndarray,
    PQ_: np.ndarray,
    PQ: np.ndarray,
    p: float=0.5,
    a: float=1.0,
    b: float=1.0) -> np.ndarray:
    """Compute Transformed Image using method in "Feature-Based Image Metamorphosis".

    The value of p is typically in the range [0, 1]; if it is zero, then all lines have
    the same weight. if it is one, then longer lines have a greater relative weight than
    shorter lines.

    If a is barely greater than zero, then if the distance from the line to the pixel
    is zero, the strength is nearly infinite. With this value for a, the user knows that
    pixels on the line will go exactly where he wants them. Values larger than that will
    yield a more smooth warping, but with less precise control.

    The variable b determines how the relative strength of different lines Falls off
    with distance. If it is large, then every pixel will be affected only by the line
    nearest it. Ifb is zero, then each pixel will be affected by all lines equally. Values
    of b in the range [0.5, 2] are the most useful.


    Args:
        I (np.ndarray): Source image. Needs three dimensions.
        PQ_ (np.ndarray): Pixel coordinates of feature lines of source image.
        PQ (np.ndarray): Pixel coordinates of feature lines of destination image.
        p (float, optional): Relative weighting based on line length. Defaults to 0.5.
        a (float, optional): Transformation strength based on distance. Defaults to 1.0.
        b (float, optional): Relative weighting based on proximity. Defaults to 1.0.

    Returns:
        np.ndarray: The destination image. 
    """
    # Get featue lines for both faces
    P, Q = PQ[:, 0], PQ[:, 1]
    P_, Q_ = PQ_[:, 0], PQ_[:, 1]

    # Output "destination" image
    dst = np.zeros((I.shape[0] * I.shape[1], I.shape[2]))

    # All pixel coordinates
    x, y = np.meshgrid(np.arange(I.shape[0]), np.arange(I.shape[1]))
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
    for j in range(I.shape[2]):
        dst[..., j] = map_coordinates(I[..., j], X_[:, ::-1].T, mode='nearest')
    dst = dst.reshape(I.shape)
    
    return dst


def metamorphosis(I0: np.ndarray,
    I1: np.ndarray,
    size0: tuple[int, int],
    size1: tuple[int, int],
    fname: str='img\\out\\temp.gif',
    duration: float=5.0,
    framerate: int=24,
    load0: str='',
    load1: str='',
    intermediates: bool=False):
    """Generate metamorphosis sequence from I0 to I1.

    Args:
        I0 (np.ndarray): First image
        I1 (np.ndarray): Second image 
        size0 (tuple of ints): The original H/W of I0 (before resizing)
        size1 (tuple of ints): The original H/W of I1 (before resizing)
        fname (str, optional): Path to save the gif to. Defaults to 'img\\out\\temp.gif'.
        duration (float, optional): Duration of the gif in seconds. Defaults to 5.0.
        framerate (int, optional): Framerate of the gif (FPS). Defaults to 24.
        load0 (str, optional): The path to load landmarks from. If empty, detects instead.
            Defaults to ''.
        load1 (str, optional): The path to load landmarks from. If empty, detects instead.
            Defaults to ''.
        intrmediates (bool, optional): Whether to save intermediate photos as well.
            Defaults to False.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    size = I0.shape[:2]

    if load0 != '':
        print("Loading features for I0...")
        _, PQ0 = load_features(load0, size, size0)
    else:
        print("Detecting features for I0...")
        _, PQ0 = detect_features(I0, detector, predictor)
    print('...done!')
    if load1 != '':
        print("Loading features for I1...")
        _, PQ1 = load_features(load1, size, size1)
    else:
        print("Detecting features for I1...")
        _, PQ1 = detect_features(I1, detector, predictor)
    print('...done!')

    frames = int(duration * framerate)
    with imageio.get_writer(fname, mode='I', duration=1/framerate) as writer:
        pbar = tqdm.tqdm(np.linspace(0, 1, frames)[::-1])
        pbar.set_description('Generating morph sequence')
        for p in pbar:
            # Interpolate feature lines and morph both images to intermediate image
            PQ_inter = p * PQ0 + (1 - p) * PQ1
            I0_inter = morph(I0, PQ0, PQ_inter)
            I1_inter = morph(I1, PQ1, PQ_inter)

            # Intermediate image is weighted average of both morphed images
            I_inter = bgr2rgb(p * I0_inter + (1 - p) * I1_inter)
            if intermediates:
                imageio.imwrite(f"{fname[:-4]}-{p:.2f}.png", I_inter)
                imageio.imwrite(f"{fname[:-4]}-i0-{p:.2f}.png", bgr2rgb(I0_inter))
                imageio.imwrite(f"{fname[:-4]}-i1-{p:.2f}.png", bgr2rgb(I1_inter))
            
            writer.append_data(I_inter.astype(np.uint8))
