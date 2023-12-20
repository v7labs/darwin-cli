from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import orjson as json

from darwin.datatypes import (
    Annotation,
    AnnotationAuthor,
    AnnotationClass,
    AnnotationFile,
    DictFreeForm,
    SubAnnotation,
    VideoAnnotation,
)
from darwin.exceptions import (
    ExportException,
    ExportException_CouldNotAssembleOutputPath,
    ExportException_CouldNotBuildOutput,
    ExportException_CouldNotWriteFile,
)


def export(annotation_files: Iterable[AnnotationFile], output_dir: Path) -> None:
    errors: List[Exception] = []

    for id, annotation_file in enumerate(annotation_files):
        try:
            _export_file(annotation_file, id, output_dir)
        except Exception as e:
            errors.append(e)

    if errors:
        raise ExportException.from_multiple_exceptions(errors)


def _export_file(annotation_file: AnnotationFile, _: int, output_dir: Path) -> None:
    try:
        filename = annotation_file.path.parts[-1]
        output_file_path = (output_dir / filename).with_suffix(".json")
    except Exception as e:
        raise ExportException_CouldNotAssembleOutputPath(
            f"Could not export file {annotation_file.path} to {output_dir}"
        ) from e

    try:
        output: DictFreeForm = _build_json(annotation_file)
    except Exception as e:
        raise ExportException_CouldNotBuildOutput(
            f"Could not build output for {annotation_file.path}"
        ) from e

    try:
        with open(output_file_path, "w") as f:
            op = json.dumps(
                output,
                option=json.OPT_INDENT_2
                | json.OPT_SERIALIZE_NUMPY
                | json.OPT_NON_STR_KEYS,
            ).decode("utf-8")
            f.write(op)
    except Exception as e:
        raise ExportException_CouldNotWriteFile(
            f"Could not write output for {annotation_file.path}"
        ) from e


def _build_json(annotation_file: AnnotationFile) -> DictFreeForm:
    if annotation_file.is_video:
        return _build_video_json(annotation_file)
    else:
        return _build_image_json(annotation_file)


def _build_image_json(annotation_file: AnnotationFile) -> DictFreeForm:
    return {
        "image": {
            "seq": annotation_file.seq,
            "width": annotation_file.image_width,
            "height": annotation_file.image_height,
            "filename": annotation_file.filename,
            "original_filename": annotation_file.filename,
            "url": annotation_file.image_url,
            "thumbnail_url": annotation_file.image_thumbnail_url,
            "path": annotation_file.remote_path,
            "workview_url": annotation_file.workview_url,
            **_build_metadata(annotation_file),
        },
        "annotations": list(map(_build_annotation, annotation_file.annotations)),
        "dataset": str(annotation_file.dataset_name),
    }


def _build_video_json(annotation_file: AnnotationFile) -> DictFreeForm:
    return {
        "image": {
            "seq": annotation_file.seq,
            "frame_urls": annotation_file.frame_urls,
            "frame_count": len(annotation_file.frame_urls or []),
            "width": annotation_file.image_width,
            "height": annotation_file.image_height,
            "filename": annotation_file.filename,
            "original_filename": annotation_file.filename,
            "thumbnail_url": annotation_file.image_thumbnail_url,
            "url": annotation_file.image_url,
            "path": annotation_file.remote_path,
            "workview_url": annotation_file.workview_url,
            **_build_metadata(annotation_file),
        },
        "annotations": list(map(_build_annotation, annotation_file.annotations)),
        "dataset": str(annotation_file.dataset_name),
    }


def _build_annotation(annotation: Union[Annotation, VideoAnnotation]) -> DictFreeForm:
    if isinstance(annotation, VideoAnnotation):
        return _build_video_annotation(annotation)
    else:
        return _build_image_annotation(annotation)


def _build_author(author: AnnotationAuthor) -> DictFreeForm:
    return {"full_name": author.name, "email": author.email}


def _build_sub_annotation(sub: SubAnnotation) -> DictFreeForm:
    if sub.annotation_type == "instance_id":
        return {sub.annotation_type: {"value": sub.data}}
    elif sub.annotation_type == "attributes":
        return {sub.annotation_type: sub.data}
    elif sub.annotation_type == "text":
        return {sub.annotation_type: {"text": sub.data}}
    else:
        return {sub.annotation_type: sub.data}


def _build_authorship(annotation: Union[VideoAnnotation, Annotation]) -> DictFreeForm:
    annotators = {}
    if annotation.annotators:
        annotators = {
            "annotators": [
                _build_author(annotator) for annotator in annotation.annotators
            ]
        }

    reviewers = {}
    if annotation.reviewers:
        reviewers = {
            "annotators": [_build_author(reviewer) for reviewer in annotation.reviewers]
        }

    return {**annotators, **reviewers}


def _build_video_annotation(annotation: VideoAnnotation) -> DictFreeForm:
    return {
        **annotation.get_data(
            only_keyframes=False,
            post_processing=lambda annotation, _: _build_image_annotation(
                annotation, skip_slots=True
            ),
        ),
        "name": annotation.annotation_class.name,
        "slot_names": annotation.slot_names,
        **_build_authorship(annotation),
    }


def _build_image_annotation(
    annotation: Annotation, skip_slots: bool = False
) -> DictFreeForm:
    json_subs = {}
    for sub in annotation.subs:
        json_subs.update(_build_sub_annotation(sub))

    base_json = {
        **json_subs,
        **_build_authorship(annotation),
        **_build_legacy_annotation_data(annotation.annotation_class, annotation.data),
        "name": annotation.annotation_class.name,
    }

    if skip_slots:
        return base_json
    else:
        return {**base_json, "slot_names": annotation.slot_names}


def _build_legacy_annotation_data(
    annotation_class: AnnotationClass, data: DictFreeForm
) -> DictFreeForm:
    v1_data = {}
    polygon_annotation_mappings = {"complex_polygon": "paths", "polygon": "path"}

    if annotation_class.annotation_type in polygon_annotation_mappings:
        key = polygon_annotation_mappings[annotation_class.annotation_type]
        v1_data[annotation_class.annotation_type] = {"path": data.get(key)}

    elif annotation_class.annotation_type == "tag":
        v1_data["tag"] = {}

    elif annotation_class.annotation_type == "bounding_box":
        v1_data[annotation_class.annotation_type] = data

    if "bounding_box" in data and annotation_class.annotation_type != "bounding_box":
        # Poygons and complex polygons usually have attached bounding_box annotations
        v1_data["bounding_box"] = data["bounding_box"]

    return v1_data


def _build_metadata(annotation_file: AnnotationFile) -> DictFreeForm:
    if len(annotation_file.slots) > 0 and annotation_file.slots[0].metadata:
        return {"metadata": annotation_file.slots[0].metadata}
    else:
        return {}


def build_image_annotation(annotation_file: AnnotationFile) -> Dict[str, Any]:
    """
    Builds and returns a dictionary with the annotations present in the given file.

    Parameters
    ----------
    annotation_file: dt.AnnotationFile
        File with the image annotations to extract.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the annotation from the given file. Has the following structure:

        .. code-block:: python

            {
                "annotations": [
                    {
                        "annotation_type": { ... }, # annotation_data
                        "name": "annotation class name",
                        "bounding_box": { ... } # Optional parameter, only present if the file has a bounding box as well
                    }
                ],
                "image": {
                    "filename": "a_file_name.json",
                    "height": 1000,
                    "width": 2000,
                    "url": "https://www.darwin.v7labs.com/..."
                }
            }
    """
    annotations: List[Dict[str, Any]] = []
    for annotation in annotation_file.annotations:
        payload = {
            annotation.annotation_class.annotation_type: _build_annotation_data(
                annotation
            ),
            "name": annotation.annotation_class.name,
        }

        if (
            annotation.annotation_class.annotation_type == "complex_polygon"
            or annotation.annotation_class.annotation_type == "polygon"
        ) and "bounding_box" in annotation.data:
            payload["bounding_box"] = annotation.data["bounding_box"]

        annotations.append(payload)

    return {
        "annotations": annotations,
        "image": {
            "filename": annotation_file.filename,
            "height": annotation_file.image_height,
            "width": annotation_file.image_width,
            "url": annotation_file.image_url,
        },
    }


def _build_annotation_data(annotation: Annotation) -> Dict[str, Any]:
    if annotation.annotation_class.annotation_type == "complex_polygon":
        return {"path": annotation.data["paths"]}

    if annotation.annotation_class.annotation_type == "polygon":
        return dict(
            filter(lambda item: item[0] != "bounding_box", annotation.data.items())
        )

    return dict(annotation.data)
