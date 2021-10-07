import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

import darwin.datatypes as dt


def parse_file(path: Path) -> Union[dt.AnnotationFile, None]:
    if path.suffix != ".xml":
        return None

    tree = ET.parse(str(path))
    root = tree.getroot()

    filename = _find_text_value(root, "filename")

    annotations: List[dt.Annotation] = list(filter(None, map(_parse_annotation, root.findall("object"))))
    annotation_classes = set([annotation.annotation_class for annotation in annotations])

    return dt.AnnotationFile(path, filename, annotation_classes, annotations, remote_path="/")


# Private
def _parse_annotation(annotation_object: ET.Element) -> dt.Annotation:
    class_name = _find_text_value(annotation_object, "name")

    bndbox = _find_element(annotation_object, "bndbox")
    xmin = int(float(_find_text_value(bndbox, "xmin")))
    xmax = int(float(_find_text_value(bndbox, "xmax")))
    ymin = int(float(_find_text_value(bndbox, "ymin")))
    ymax = int(float(_find_text_value(bndbox, "ymax")))

    return dt.make_bounding_box(class_name, xmin, ymin, xmax - xmin, ymax - ymin)


# Private
def _find_element(source: ET.Element, name: str) -> ET.Element:
    element = source.find(name)
    if element is None:
        raise ValueError(f"Could not find {name} element in annotation file")
    return element


# Private
def _find_text_value(source: ET.Element, name: str) -> str:
    element = _find_element(source, name)
    if element.text is None or not element.text.strip():
        raise ValueError(f"{name} element does not have a text value")
    return element.text
