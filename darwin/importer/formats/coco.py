from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import deprecation
import orjson as json
from upolygon import find_contours, rle_decode

import darwin.datatypes as dt
from darwin.exceptions import UnrecognizableFileEncoding
from darwin.path_utils import deconstruct_full_path
from darwin.utils import attempt_decode
from darwin.version import __version__

DEPRECATION_MESSAGE = """

This function is going to be turned into private. This means that breaking 
changes in its interface and implementation are to be expected. We encourage using ``parse_annotation`` 
instead of calling this low-level function directly.
"""


def parse_path(path: Path) -> Optional[List[dt.AnnotationFile]]:
    """
    Parses the given ``coco`` file and returns a ``List[dt.AnnotationFile]`` with the parsed
    information.

    Parameters
    ----------
    path : Path
        The ``Path`` to the ``coco`` file.

    Returns
    -------
    Optional[List[dt.AnnotationFile]]
        Returns ``None`` if the given file is not in ``json`` format, or ``List[dt.AnnotationFile]``
        otherwise.
    """
    if path.suffix != ".json":
        return None
    data = attempt_decode(path)
    return list(parse_json(path, data))


def parse_json(path: Path, data: Dict[str, Any]) -> Iterator[dt.AnnotationFile]:
    """
    Parses the given ``json`` structure into an ``Iterator[dt.AnnotationFile]``.

    Parameters
    ----------
    path : Path
        The ``Path`` where file containing the ``data`` is.
    data : Dict[str, Any]
        The ``json`` data to process.

    Returns
    -------
    Iterator[dt.AnnotationFile]
        An iterator of all parsed annotation files.
    """
    annotations = data["annotations"]
    image_lookup_table = {image["id"]: image for image in data["images"]}
    category_lookup_table = {category["id"]: category for category in data["categories"]}
    tag_categories = data.get("tag_categories") or []
    tag_category_lookup_table = {category["id"]: category for category in tag_categories}
    image_annotations: Dict[str, Any] = {}

    for image in data["images"]:
        image_id = image["id"]
        tag_ids = image.get("tag_ids") or []

        if image_id not in image_annotations:
            image_annotations[image_id] = []

        for tag_id in tag_ids:
            tag = tag_category_lookup_table[tag_id]
            image_annotations[image_id].append(dt.make_tag(tag["name"]))

    for annotation in annotations:
        image_id = annotation["image_id"]
        annotation["category_id"]
        annotation["segmentation"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(parse_annotation(annotation, category_lookup_table))

    for image_id in image_annotations.keys():
        image = image_lookup_table[int(image_id)]
        annotations = list(filter(None, image_annotations[image_id]))
        annotation_classes = set([annotation.annotation_class for annotation in annotations])
        remote_path, filename = deconstruct_full_path(image["file_name"])
        yield dt.AnnotationFile(path, filename, annotation_classes, annotations, remote_path=remote_path)


def parse_annotation(annotation: Dict[str, Any], category_lookup_table: Dict[str, Any]) -> Optional[dt.Annotation]:
    """
    Parses the given ``json`` dictionary into a darwin ``Annotation`` if possible.

    Parameters
    ----------
    annotation : Dict[str, Any]
        The ``json`` dictionary to parse.
    category_lookup_table : Dict[str, Any]
        Dictionary with all the categories from the ``coco`` file.

    Returns
    -------
    Optional[dt.Annotation]
        A darwin ``Annotation`` if the parse was successful, or ``None`` otherwise.
    """
    category = category_lookup_table[annotation["category_id"]]
    segmentation = annotation["segmentation"]

    if len(segmentation) == 0 and len(annotation["bbox"]) == 4:
        x, y, w, h = map(int, annotation["bbox"])
        return dt.make_bounding_box(category["name"], x, y, w, h)
    elif len(segmentation) == 0 and len(annotation["bbox"]) == 1 and len(annotation["bbox"][0]) == 4:
        x, y, w, h = map(int, annotation["bbox"][0])
        return dt.make_bounding_box(category["name"], x, y, w, h)
    elif isinstance(segmentation, dict):
        print("warning, converting complex coco rle mask to polygon, could take some time")
        if isinstance(segmentation["counts"], list):
            mask = rle_decode(segmentation["counts"], segmentation["size"][::-1])
        else:
            counts = decode_binary_rle(segmentation["counts"])
            mask = rle_decode(counts, segmentation["size"][::-1])

        _labels, external, _internal = find_contours(mask)
        paths = []
        for external_path in external:
            new_path = []
            # skip paths with less than 2 points
            if len(external_path) // 2 <= 2:
                continue
            points = iter(external_path)
            while True:
                try:
                    x, y = next(points), next(points)
                    new_path.append({"x": x, "y": y})
                except StopIteration:
                    break
            paths.append(new_path)

        for internal_path in _internal:
            new_path = []
            # skip paths with less than 2 points
            if len(internal_path) // 2 <= 2:
                continue
            points = iter(internal_path)
            while True:
                try:
                    x, y = next(points), next(points)
                    new_path.append({"x": x, "y": y})
                except StopIteration:
                    break
            paths.append(new_path)
        return dt.make_complex_polygon(category["name"], paths)
    elif isinstance(segmentation, list):
        path = []
        points = iter(segmentation[0] if isinstance(segmentation[0], list) else segmentation)
        while True:
            try:
                x, y = next(points), next(points)
                path.append({"x": x, "y": y})
            except StopIteration:
                break
        return dt.make_polygon(category["name"], path)
    else:
        return None


def _decode_file(current_encoding: str, path: Path):
    if current_encoding == "system_default":
        with path.open() as f:
            data = json.loads(f.read())
    else:
        with path.open(encoding=current_encoding) as f:
            data = json.loads(f.read())
    return list(parse_json(path, data))


@deprecation.deprecated(
    deprecated_in="0.7.12",
    removed_in="0.8.0",
    current_version=__version__,
    details=DEPRECATION_MESSAGE,
)
def decode_binary_rle(data: str) -> List[int]:
    """
    Decodes binary rle to integer list rle.
    """
    m = len(data)
    counts = [0] * m
    h = 0
    p = 0
    while p < m:
        x = 0
        k = 0
        more = 1
        while more > 0:
            c = ord(data[p]) - 48
            x |= (c & 0x1F) << 5 * k
            more = c & 0x20
            p = p + 1
            k = k + 1
            if more == 0 and (c & 0x10) != 0:
                x |= -1 << 5 * k
        if h > 2:
            x += counts[h - 2]
        counts[h] = x
        h += 1
    return counts[0:h]
