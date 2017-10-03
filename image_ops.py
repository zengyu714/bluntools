import cv2
import numpy as np
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels
from skimage import measure
from skimage.exposure import adjust_gamma
from skimage.transform import warp, AffineTransform


def random_gamma(*inputs, low=0.9, high=1.1):
    """Adjust image intensity"""

    # `xs` has shape [height, width] with value in [0, 1].
    gamma = np.random.uniform(low=low, high=high)
    outputs = [adjust_gamma(item, gamma) for item in inputs]

    return outputs if len(inputs) > 1 else outputs[0]


def random_hsv(image,
               hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_affine(image, label,
                  scale_limit=(0.9, 1.1), translation=(-0.0625, 0.0625), shear=None, rotation=None):
    scale = np.random.uniform(*scale_limit, size=2)

    if translation is not None:
        translation = np.random.uniform(*np.multiply(image.shape[:-1], translation), size=2)
    if shear is not None:
        shear = np.random.uniform(*np.deg2rad(shear))
    if rotation is not None:
        rotation = np.random.uniform(*np.deg2rad(rotation))

    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear, translation=translation)
    return [warp(item, tform, mode='edge', preserve_range=True) for item in [image, label]]


def random_hflip(image, label, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)

    return image, label


def dense_crf(im, mask):
    h, w = mask.shape[:2]

    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, n-classes
    U = unary_from_labels(mask, 2, gt_prob=0.95, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=30, compat=3)
    d.addPairwiseBilateral(sxy=3, srgb=20, rgbim=im, compat=10)

    Q = d.inference(3)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


def shrink_to_bbox(*inputs):
    """
    Reduce volume/area to the size of bounding box

    Argument:
        Volumes:
        + modality_1, modality_2(optional), label

    Return:
        Reduced volumes
        + modality_1, modality_2(optional), label
        + bbox
    """
    ndim = inputs[0].ndim
    labeled = measure.label(inputs[0] > 0, connectivity=ndim)

    # bbox: (min_x, min_y, max_z, max_x, max_y, max_z)
    bb = measure.regionprops(labeled)[0].bbox

    # 3-d slice: (bb[0]: bb[3], bb[1]: bb[4], bb[2]: bb[5])
    # 2-d slice: (bb[0]: bb[2], bb[1]: bb[3])
    sl = [slice(bb[i], bb[i + ndim]) for i in range(ndim)]
    outputs = [item[sl] for item in inputs]

    return outputs + [bb]


def normalize(im):
    """
    Normalize volume's intensity to range [0, 1], for suing image processing
    Compute global maximum and minimum cause cerebrum is relatively homogeneous
    """
    _max = np.max(im)
    _min = np.min(im)
    # `im` with dtype float64
    return (im - _min) / (_max - _min)
