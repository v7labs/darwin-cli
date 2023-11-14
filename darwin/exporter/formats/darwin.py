from typing import Any, Dict, List

import deprecation

import darwin.datatypes as dt
from darwin.version import __version__

DEPRECATION_MESSAGE = """

This function is going to be turned into private. This means that breaking 
changes in its interface and implementation are to be expected. We encourage using ``build_image_annotation`` 
instead of calling this low-level function directly.

"""


def build_image_annotation(annotation_file: dt.AnnotationFile) -> Dict[str, Any]:
    """
    Builds and returns a dictionary with the annotations present in the given file in Darwin v2 format.

    Parameters
    ----------
    annotation_file: AnnotationFile
        File with the image annotations to extract.
        For schema, see: https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json

    Returns
    -------
    Dict[str, Any]
        A dictionary with the annotations in Darwin v2 format.
    """
    annotations_list: List[Dict[str, Any]] = []

    for annotation in annotation_file.annotations:
        annotation_data = _build_v2_annotation_data(annotation)
        annotations_list.append(annotation_data)

    slots_data = _build_slots_data(annotation_file.slots)
    item = _build_item_data(annotation_file)
    item["slots"] = slots_data

    return {
        "version": "2.0",
        "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
        "item": item,
        "annotations": annotations_list,
    }


def _build_v2_annotation_data(annotation: dt.Annotation) -> Dict[str, Any]:
    annotation_data = {"id": annotation.id, "name": annotation.annotation_class.name}

    if annotation.annotation_class.annotation_type == "bounding_box":
        annotation_data["bounding_box"] = _build_bounding_box_data(annotation.data)
    elif annotation.annotation_class.annotation_type == "tag":
        annotation_data["tag"] = {}
    elif annotation.annotation_class.annotation_type == "polygon":
        annotation_data["polygon"] = _build_polygon_data(annotation.data)

    return annotation_data


def _build_bounding_box_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "h": data.get("h"),
        "w": data.get("w"),
        "x": data.get("x"),
        "y": data.get("y"),
    }


def _build_polygon_data(
    data: Dict[str, Any]
) -> Dict[str, List[List[Dict[str, float]]]]:
    """
    Builds the polygon data for Darwin v2 format.

    Parameters
    ----------
    data : Dict[str, Any]
        The original data for the polygon annotation.

    Returns
    -------
    Dict[str, List[List[Dict[str, float]]]]
        The polygon data in the format required for Darwin v2 annotations.
    """
    # Assuming the data contains a 'paths' key that is a list of lists of points,
    # where each point is a dictionary with 'x' and 'y' keys.
    paths = data.get("paths", [])
    v2_paths = []

    for path in paths:
        v2_path = []
        for point in path:
            v2_point = {"x": point.get("x"), "y": point.get("y")}
            v2_path.append(v2_point)
        v2_paths.append(v2_path)

    return {"paths": v2_paths}


def _build_item_data(annotation_file: dt.AnnotationFile) -> Dict[str, Any]:
    """
    Constructs the 'item' section of the Darwin v2 format annotation.

    Parameters
    ----------
    annotation_file: dt.AnnotationFile
        The AnnotationFile object containing annotation data.

    Returns
    -------
    Dict[str, Any]
        The 'item' section of the Darwin v2 format annotation.
    """
    return {
        "name": annotation_file.filename,
        "path": annotation_file.remote_path or "/",
        "source_info": {
            "dataset": {
                "name": annotation_file.dataset_name,
                "slug": annotation_file.dataset_name.lower().replace(" ", "-")
                if annotation_file.dataset_name
                else None,
            },
            "item_id": annotation_file.item_id or "unknown-item-id",
            "team": {
                "name": None,  # TODO Replace with actual team name
                "slug": None,  # TODO Replace with actual team slug
            },
            "workview_url": annotation_file.workview_url,
        },
    }


def _build_slots_data(slots: List[dt.Slot]) -> List[Dict[str, Any]]:
    """
    Constructs the 'slots' data for the Darwin v2 format annotation.

    Parameters
    ----------
    slots: List[Slot]
        A list of Slot objects from the AnnotationFile.

    Returns
    -------
    List[Dict[str, Any]]
        The 'slots' data for the Darwin v2 format annotation.
    """
    slots_data = []
    for slot in slots:
        slot_data = {
            "type": slot.type,
            "slot_name": slot.name,
            "width": slot.width,
            "height": slot.height,
            "thumbnail_url": slot.thumbnail_url,
            "source_files": slot.source_files,
        }
        slots_data.append(slot_data)

    return slots_data


@deprecation.deprecated(
    deprecated_in="0.7.8",
    removed_in="0.8.0",
    current_version=__version__,
    details=DEPRECATION_MESSAGE,
)
def build_annotation_data(annotation: dt.Annotation) -> Dict[str, Any]:
    if annotation.annotation_class.annotation_type == "complex_polygon":
        return {"path": annotation.data["paths"]}

    if annotation.annotation_class.annotation_type == "polygon":
        return dict(
            filter(lambda item: item[0] != "bounding_box", annotation.data.items())
        )

    return dict(annotation.data)


def _build_annotation_data(annotation: dt.Annotation) -> Dict[str, Any]:
    if annotation.annotation_class.annotation_type == "complex_polygon":
        return {"path": annotation.data["paths"]}

    if annotation.annotation_class.annotation_type == "polygon":
        return dict(
            filter(lambda item: item[0] != "bounding_box", annotation.data.items())
        )

    return dict(annotation.data)
