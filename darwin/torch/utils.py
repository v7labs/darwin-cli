from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from darwin.types import ComplexPolygon, Polygon

try:
    import accimage
except ImportError:
    accimage = None


def load_pil_image(path: Path):
    """
    Loads a PIL image and converts it into RGB.

    Parameters
    ----------
    path: Path
        Path to the image file

    Returns
    -------
    PIL Image
        Values between 0 and 255
    """
    pic = Image.open(path)
    if pic.mode == "RGB":
        pass
    elif pic.mode in ("CMYK", "RGBA", "P"):
        pic = pic.convert("RGB")
    elif pic.mode == "I":
        img = (np.divide(np.array(pic, np.int32), 2 ** 16 - 1) * 255).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    elif pic.mode == "I;16":
        img = (np.divide(np.array(pic, np.int16), 2 ** 8 - 1) * 255).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    elif pic.mode == "L":
        img = np.array(pic).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    else:
        raise TypeError(f"unsupported image type {pic.mode}")
    return pic


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def convert_polygon_to_mask(segmentations: List[float], height: int, width: int):
    """
    Converts a polygon represented as a sequence of coordinates into a mask.

    Input:
        segmentations: list of float values -> [x1, y1, x2, y2, ..., xn, yn]
        height: image's height
        width: image's width

    Output:
        torch.tensor
    """
    polygons = [
        np.array(list(zip(map(int, map(round, s[0::2])), map(int, map(round, s[1::2])))))
        for contour in segmentations
        for s in contour
    ]
    zeros = torch.zeros((1080, 1920)).numpy().astype(np.uint8)
    mask = torch.from_numpy(cv2.drawContours(zeros, polygons, -1, 1, cv2.FILLED))
    return mask.unsqueeze(0)


def convert_polygons_to_sequences(polygons: Union[Polygon, ComplexPolygon]) -> List[np.ndarray]:
    """
    Converts a list of polygons, encoded as a list of dictionaries of into a list of nd.arrays
    of coordinates.

    Parameters
    ----------
    polygons: list
        List of coordinates in the format [{x: x1, y:y1}, ..., {x: xn, y:yn}] or a list of them
        as  [[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]].

    Returns
    -------
    sequences: list[ndarray[float]]
        List of arrays of coordinates in the format [[x1, y1, x2, y2, ..., xn, yn], ...,
        [x1, y1, x2, y2, ..., xn, yn]]
    """
    if not polygons:
        raise ValueError("No polygons provided")
    # If there is a single polygon composing the instance the format is going to be
    # polygons = [{x: x1, y:y1}, ..., {x: xn, y:yn}]
    if isinstance(polygons[0], dict):
        path = []
        for point in polygons:
            path.append(point["x"])
            path.append(point["y"])
        return [np.array(path)]  # List type is used for backward compatibility
    # If there are multiple polygons composing the instance the format is going to be
    # polygons =  [[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]]
    if isinstance(polygons[0], list) and isinstance(polygons[0][0], dict):
        sequences = []
        for polygon in polygons:
            path = []
            for point in polygon:
                path.append(point["x"])
                path.append(point["y"])
            sequences.append(np.array(path))
        return sequences
    raise ValueError("Unknown input format")


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns the area of the input polygon, represented with two numpy arrays
    for x and y coordinates.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
