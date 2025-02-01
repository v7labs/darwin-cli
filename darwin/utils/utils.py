"""
Contains several unrelated utility functions used across the SDK.
"""

import platform
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import json_stream
import numpy as np
import orjson as json
import requests
from json_stream.base import PersistentStreamingJSONList, PersistentStreamingJSONObject
from jsonschema import validators
from natsort import natsorted
from requests import Response
from rich.progress import ProgressType, track
from upolygon import draw_polygon

import darwin.datatypes as dt
from darwin.config import Config
from darwin.exceptions import (
    MissingSchema,
    OutdatedDarwinJSONFormat,
    UnrecognizableFileEncoding,
    UnsupportedFileType,
)
from darwin.future.data_objects.properties import SelectedProperty

if TYPE_CHECKING:
    from darwin.client import Client


SUPPORTED_IMAGE_EXTENSIONS = [
    ".png",
    ".jpeg",
    ".jpg",
    ".jfif",
    ".tif",
    ".tiff",
    ".qtiff",
    ".bmp",
    ".svs",
    ".webp",
    ".JPEG",
    ".JPG",
]
SUPPORTED_VIDEO_EXTENSIONS = [
    ".avi",
    ".bpm",
    ".dcm",
    ".mov",
    ".mp4",
    ".mkv",
    ".hevc",
    ".pdf",
    ".nii",
    ".nii.gz",
    ".ndpi",
    ".rvg",
]
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS

# Define incompatible `item_merge_mode` arguments
PRESERVE_FOLDERS_KEY = "preserve_folders"
AS_FRAMES_KEY = "as_frames"
EXTRACT_VIEWS_KEY = "extract_views"

# Define reasons for blocking slot uploads
BLOCKED_UPLOAD_ERROR_ALREADY_EXISTS = "ALREADY_EXISTS"
BLOCKED_UPLOAD_ERROR_FILE_UPLOAD_TIMEOUT = "FILE_UPLOAD_TIMEOUT"
BLOCKED_UPLOAD_ERROR_FILE_UPLOAD_FAILED = "FILE_UPLOAD_FAILED"
BLOCKED_UPLOAD_ERROR_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
BLOCKED_UPLOAD_ERROR_ITEM_COUNT_LIMIT_EXCEEDED = "ITEM_COUNT_LIMIT_EXCEEDED"

SLOTS_GRID_MAP = {
    1: [[["0"]]],
    2: [[["0"]], [["1"]]],
    3: [[["0"]], [["1"]], [["2"]]],
    4: [[["0"], ["2"]], [["1"], ["3"]]],
    5: [[["0"], ["3"]], [["1"], ["4"]], [["2"]]],
    6: [[["0"], ["3"]], [["1"], ["4"]], [["2"], ["5"]]],
    7: [[["0"], ["3"], ["6"]], [["1"], ["4"]], [["2"], ["5"]]],
    8: [[["0"], ["3"], ["6"]], [["1"], ["4"], ["7"]], [["2"], ["5"]]],
    9: [[["0"], ["3"], ["6"]], [["1"], ["4"], ["7"]], [["2"], ["5"], ["8"]]],
    10: [[["0"], ["4"], ["8"]], [["1"], ["5"], ["9"]], [["2"], ["6"]], [["3"], ["7"]]],
    11: [
        [["0"], ["4"], ["8"]],
        [["1"], ["5"], ["9"]],
        [["2"], ["6"], ["10"]],
        [["3"], ["7"]],
    ],
    12: [
        [["0"], ["4"], ["8"]],
        [["1"], ["5"], ["9"]],
        [["2"], ["6"], ["10"]],
        [["3"], ["7"], ["11"]],
    ],
    13: [
        [["0"], ["4"], ["8"], ["12"]],
        [["1"], ["5"], ["9"]],
        [["2"], ["6"], ["10"]],
        [["3"], ["7"], ["11"]],
    ],
    14: [
        [["0"], ["4"], ["8"], ["12"]],
        [["1"], ["5"], ["9"], ["13"]],
        [["2"], ["6"], ["10"]],
        [["3"], ["7"], ["11"]],
    ],
    15: [
        [["0"], ["4"], ["8"], ["12"]],
        [["1"], ["5"], ["9"], ["13"]],
        [["2"], ["6"], ["10"], ["14"]],
        [["3"], ["7"], ["11"]],
    ],
    16: [
        [["0"], ["4"], ["8"], ["12"]],
        [["1"], ["5"], ["9"], ["13"]],
        [["2"], ["6"], ["10"], ["14"]],
        [["3"], ["7"], ["11"], ["15"]],
    ],
}


_darwin_schema_cache = {}


def is_extension_allowed_by_filename(filename: str) -> bool:
    """
    Returns whether or not the given video or image extension is allowed.

    Parameters
    ----------
    filename : str
        The filename.

    Returns
    -------
    bool
        Whether or not the given extension of the filename is allowed.
    """
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def is_image_extension_allowed_by_filename(filename: str) -> bool:
    """
    Returns whether or not the given image extension is allowed.

    Parameters
    ----------
    filename : str
        The image extension.

    Returns
    -------
    bool
        Whether or not the given extension is allowed.
    """
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)


def is_file_extension_allowed(filename: str) -> bool:
    """
    Returns whether or not the given image extension is allowed.

    Parameters
    ----------
    filename : str
        The name of the file.

    Returns
    -------
    bool
        Whether or not the given extension is allowed.
    """
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def urljoin(*parts: str) -> str:
    """
    Take as input an unpacked list of strings and joins them to form an URL.

    Parameters
    ----------
    parts : str
        The list of strings to form the url.

    Returns
    -------
    str
        The url.
    """
    return "/".join(part.strip("/") for part in parts)


def is_project_dir(project_path: Path) -> bool:
    """
    Verifies if the directory is a project from Darwin by inspecting its structure.

    Parameters
    ----------
    project_path : Path
        Directory to examine

    Returns
    -------
    bool
        Is the directory a project from Darwin?
    """
    return (project_path / "releases").exists() and (project_path / "images").exists()


def get_progress_bar(
    array: List[dt.AnnotationFile], description: Optional[str] = None
) -> Iterable[ProgressType]:
    """
    Get a rich a progress bar for the given list of annotation files.

    Parameters
    ----------
    array : List[dt.AnnotationFile]
        The list of annotation files.
    description : Optional[str], default: None
        A description to show above the progress bar.

    Returns
    -------
    Iterable[ProgressType]
        An iterable of ``ProgressType`` to show a progress bar.
    """
    if description:
        return track(array, description=description)
    return track(array)


def prompt(msg: str, default: Optional[str] = None) -> str:
    """
    Prompt the user on a CLI to input a message.

    Parameters
    ----------
    msg : str
        Message to print.
    default : Optional[str], default: None
        Default values which is put between [] when the user is prompted.

    Returns
    -------
    str
        The input from the user or the default value provided as parameter if user does not provide
        one.
    """
    if default:
        msg = f"{msg} [{default}]: "
    else:
        msg = f"{msg}: "
    result = input(msg)
    if not result and default:
        return default
    return result


def find_files(
    files: List[dt.PathLike],
    *,
    files_to_exclude: List[dt.PathLike] = [],
    recursive: bool = True,
    sort: bool = False,
) -> List[Path]:
    """
    Retrieve a list of all files belonging to supported extensions. The exploration can be made
    recursive and a list of files can be excluded if desired.

    Parameters
    ----------
    files: List[dt.PathLike]
        List of files that will be filtered with the supported file extensions and returned.
    files_to_exclude : List[dt.PathLike]
        List of files to exclude from the search.
    recursive : bool
        Flag for recursive search.
    sort : bool
        Flag for sorting the files naturally, i.e. file2.txt will come before file10.txt.
    Returns
    -------
    List[Path]
        List of all files belonging to supported extensions. Can't return None.
    """

    found_files: List[Path] = []
    pattern = "**/*" if recursive else "*"

    for f in files:
        path = Path(f)
        if path.is_dir():
            found_files.extend(
                [
                    path_object
                    for path_object in path.glob(pattern)
                    if is_extension_allowed_by_filename(str(path_object))
                ]
            )
        elif is_extension_allowed_by_filename(str(path)):
            found_files.append(path)
        else:
            raise UnsupportedFileType(path)

    files_to_exclude_full_paths = [str(Path(f)) for f in files_to_exclude]
    filtered_files = [
        f for f in found_files if str(f) not in files_to_exclude_full_paths
    ]
    if sort:
        return natsorted(filtered_files)
    return filtered_files


def secure_continue_request() -> bool:
    """
    Asks for explicit approval from the user. Empty string not accepted.

    Returns
    -------
    bool
        True if the user wishes to continue, False otherwise.
    """
    return input("Do you want to continue? [y/N] ") in ["Y", "y"]


def persist_client_configuration(
    client: "Client",
    default_team: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Config:
    """
    Authenticate user against the server and creates a configuration file for him/her.

    Parameters
    ----------
    client : Client
        Client to take the configurations from.
    default_team : Optional[str], default: None
        The default team for the user.
    config_path : Optional[Path], default: None
        Specifies where to save the configuration file.

    Returns
    -------
    Config
        A configuration object to handle YAML files.
    """
    if not config_path:
        config_path = Path.home() / ".darwin" / "config.yaml"
        config_path.parent.mkdir(exist_ok=True)

    team_config: Optional[dt.Team] = client.config.get_default_team()
    if not team_config:
        raise ValueError("Unable to get default team.")

    config: Config = Config(config_path)
    config.set_team(
        team=team_config.slug,
        api_key=team_config.api_key,
        datasets_dir=team_config.datasets_dir,
    )
    config.set_global(
        api_endpoint=client.url, base_url=client.base_url, default_team=default_team
    )

    return config


def _get_local_filename(metadata: Dict[str, Any]) -> str:
    if "original_filename" in metadata:
        return metadata["original_filename"]
    else:
        return metadata["filename"]


def _get_schema(data: dict) -> Optional[dict]:
    version = _parse_version(data)
    schema_url = data.get("schema_ref") or _default_schema(version)
    if not schema_url:
        return None
    if schema_url not in _darwin_schema_cache:
        response = requests.get(schema_url)
        response.raise_for_status()
        schema = response.json()
        _darwin_schema_cache[schema_url] = schema
    return _darwin_schema_cache[schema_url]


def validate_file_against_schema(path: Path) -> List:
    data, _ = load_data_from_file(path)
    return validate_data_against_schema(data)


def validate_data_against_schema(data) -> List:
    try:
        schema = _get_schema(data)
    except requests.exceptions.RequestException as e:
        raise MissingSchema(f"Error retrieving schema from url: {e}")
    if not schema:
        raise MissingSchema("Schema not found")
    validator = validators.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    return errors


def attempt_decode(path: Path) -> dict:
    try:
        with path.open() as infile:
            data = json.loads(infile.read())
        return data
    except Exception:
        pass
    encodings = ["utf-8", "utf-16", "utf-32", "ascii"]
    for encoding in encodings:
        try:
            with path.open(encoding=encoding) as infile:
                data = json.loads(infile.read())
            return data
        except Exception:
            continue
    raise UnrecognizableFileEncoding(
        f"Unable to load file {path} with any encodings: {encodings}"
    )


def load_data_from_file(path: Path) -> Tuple[dict, dt.AnnotationFileVersion]:
    data = attempt_decode(path)
    version = _parse_version(data)
    return data, version


def parse_darwin_json(
    path: Path, count: Optional[int] = None
) -> Optional[dt.AnnotationFile]:
    """
    Parses the given JSON file in v7's darwin proprietary format. Works for images, split frame
    videos (treated as images) and playback videos.

    Parameters
    ----------
    path : Path
        Path to the file to parse.
    count : Optional[int]
        Optional count parameter. Used only if the 's image sequence is None.

    Returns
    -------
    Optional[dt.AnnotationFile]
        An AnnotationFile with the information from the parsed JSON file, or None, if there were no
        annotations in the JSON.

    Raises
    ------
    OutdatedDarwinJSONFormat
        If the given darwin video JSON file is missing the 'width' and 'height' keys in the 'image'
        dictionary.
    """

    path = Path(path)

    data, version = load_data_from_file(path)
    if "annotations" not in data:
        return None

    return _parse_darwin_v2(path, data)


def stream_darwin_json(path: Path) -> PersistentStreamingJSONObject:
    """
    Returns a Darwin JSON file as a persistent stream. This allows for parsing large files without
    loading them entirely into memory.

    Parameters
    ----------
    path : Path
        Path to the file to parse.

    Returns
    -------
    PersistentStreamingJSONObject
        A stream of the JSON file.
    """

    with path.open() as infile:
        return json_stream.load(infile, persistent=True)


def get_image_path_from_stream(
    darwin_json: PersistentStreamingJSONObject,
    images_dir: Path,
    annotation_filepath: Path,
    with_folders: bool = True,
) -> Path:
    """
    Returns the path to the image file associated with the given darwin json file.
    Compatible with Darwin JSON V2, as well as releases in folders and flat structures.

    Parameters
    ----------
    darwin_json : PersistentStreamingJSONObject
        A stream of the JSON file.
    images_dir : Path
        Path to the directory containing the images.
    with_folders: bool
        Flag to determine if the release was pulled with or without folders.
    annotation_filepath : Path
        Path to the annotation file. Used if loading the JSON as a stream fails.

    Returns
    -------
    Path
        Path to the image file.
    """
    try:
        item_name_stem = Path(darwin_json["item"]["name"]).stem
        source_name_suffix = Path(
            darwin_json["item"]["slots"][0]["source_files"][0]["file_name"]
        ).suffix
        local_file_name = Path(item_name_stem + source_name_suffix)
        if not with_folders:
            return images_dir / local_file_name
        else:
            return (
                images_dir
                / (Path(darwin_json["item"]["path"].lstrip("/\\")))
                / local_file_name
            )
    except OSError:
        # Load in the JSON as normal
        darwin_json = parse_darwin_json(path=annotation_filepath)
        if not with_folders:
            return images_dir / Path(darwin_json.filename)
        else:
            return images_dir / Path(darwin_json.full_path.lstrip("/\\"))


def is_stream_list_empty(json_list: PersistentStreamingJSONList) -> bool:
    try:
        json_list[0]
    except IndexError:
        return True

    return False


def _parse_darwin_v2(path: Path, data: Dict[str, Any]) -> dt.AnnotationFile:
    item = data["item"]
    item_source = item.get("source_info", {})
    slots: List[dt.Slot] = list(
        filter(None, map(_parse_darwin_slot, item.get("slots", [])))
    )
    annotations: List[Union[dt.Annotation, dt.VideoAnnotation]] = _data_to_annotations(
        data
    )
    annotation_classes: Set[dt.AnnotationClass] = {
        annotation.annotation_class for annotation in annotations
    }

    if len(slots) == 0:
        annotation_file = dt.AnnotationFile(
            version=_parse_version(data),
            path=path,
            filename=item["name"],
            item_id=item.get("source_info", {}).get("item_id", None),
            dataset_name=item.get("source_info", {})
            .get("dataset", {})
            .get("name", None),
            annotation_classes=annotation_classes,
            annotations=annotations,
            is_video=False,
            image_width=None,
            image_height=None,
            image_url=None,
            image_thumbnail_url=None,
            workview_url=item_source.get("workview_url", None),
            seq=0,
            frame_urls=None,
            remote_path=item["path"],
            slots=slots,
            item_properties=data.get("properties", []),
        )
    else:
        slot = slots[0]
        annotation_file = dt.AnnotationFile(
            version=_parse_version(data),
            path=path,
            filename=item["name"],
            item_id=item.get("source_info", {}).get("item_id", None),
            dataset_name=item.get("source_info", {})
            .get("dataset", {})
            .get("name", None),
            annotation_classes=annotation_classes,
            annotations=annotations,
            is_video=slot.frame_urls is not None or slot.frame_manifest is not None,
            image_width=slot.width,
            image_height=slot.height,
            image_url=(
                None if len(slot.source_files or []) == 0 else slot.source_files[0].url
            ),
            image_thumbnail_url=slot.thumbnail_url,
            workview_url=item_source.get("workview_url", None),
            seq=0,
            frame_urls=slot.frame_urls,
            remote_path=item["path"],
            slots=slots,
            frame_count=slot.frame_count,
            item_properties=data.get("properties", []),
        )

    return annotation_file


def _parse_darwin_slot(data: Dict[str, Any]) -> dt.Slot:
    source_files_data = data.get("source_files", [])
    source_files = [
        dt.SourceFile(file_name=source_file["file_name"], url=source_file.get("url"))
        for source_file in source_files_data
    ]

    return dt.Slot(
        name=data["slot_name"],
        type=data["type"],
        width=data.get("width"),
        height=data.get("height"),
        source_files=source_files,
        thumbnail_url=data.get("thumbnail_url"),
        frame_count=data.get("frame_count"),
        frame_urls=data.get("frame_urls"),
        fps=data.get("fps"),
        metadata=data.get("metadata"),
        segments=data.get("segments", []),
        frame_manifest=data.get("frame_manifests"),
    )


def _parse_darwin_image(
    path: Path, data: Dict[str, Any], count: Optional[int]
) -> dt.AnnotationFile:
    annotations: List[Union[dt.Annotation, dt.VideoAnnotation]] = _data_to_annotations(
        data
    )
    annotation_classes: Set[dt.AnnotationClass] = {
        annotation.annotation_class for annotation in annotations
    }

    slot = dt.Slot(
        name=None,
        type="image",
        source_files=[
            dt.SourceFile(
                file_name=_get_local_filename(data["image"]),
                url=data["image"].get("url"),
            )
        ],
        thumbnail_url=data["image"].get("thumbnail_url"),
        width=data["image"].get("width"),
        height=data["image"].get("height"),
        metadata=data["image"].get("metadata"),
    )

    annotation_file = dt.AnnotationFile(
        path=path,
        filename=_get_local_filename(data["image"]),
        annotation_classes=annotation_classes,
        annotations=annotations,
        is_video=False,
        image_width=data["image"].get("width"),
        image_height=data["image"].get("height"),
        image_url=data["image"].get("url"),
        workview_url=data["image"].get("workview_url"),
        seq=data["image"].get("seq", count),
        frame_urls=None,
        remote_path=data["image"].get("path", "/"),
        slots=[],
        image_thumbnail_url=data["image"].get("thumbnail_url"),
    )
    annotation_file.slots.append(slot)
    return annotation_file


def _parse_darwin_video(
    path: Path, data: Dict[str, Any], count: Optional[int]
) -> dt.AnnotationFile:
    annotations: List[Union[dt.Annotation, dt.VideoAnnotation]] = _data_to_annotations(
        data
    )
    annotation_classes: Set[dt.AnnotationClass] = {
        annotation.annotation_class for annotation in annotations
    }

    if "width" not in data["image"] or "height" not in data["image"]:
        raise OutdatedDarwinJSONFormat(
            "Missing width/height in video, please re-export"
        )

    slot = dt.Slot(
        name=None,
        type="video",
        source_files=[
            dt.SourceFile(
                file_name=_get_local_filename(data["image"]),
                url=data["image"].get("url"),
            )
        ],
        thumbnail_url=data["image"].get("thumbnail_url"),
        width=data["image"].get("width"),
        height=data["image"].get("height"),
        frame_count=data["image"].get("frame_count"),
        frame_urls=data["image"].get("frame_urls"),
        fps=data["image"].get("fps"),
        metadata=data["image"].get("metadata"),
    )
    annotation_file = dt.AnnotationFile(
        path=path,
        filename=_get_local_filename(data["image"]),
        annotation_classes=annotation_classes,
        annotations=annotations,
        is_video=True,
        image_width=data["image"].get("width"),
        image_height=data["image"].get("height"),
        image_url=data["image"].get("url"),
        workview_url=data["image"].get("workview_url"),
        seq=data["image"].get("seq", count),
        frame_urls=data["image"].get("frame_urls"),
        remote_path=data["image"].get("path", "/"),
        slots=[],
        image_thumbnail_url=data["image"].get("thumbnail_url"),
    )
    annotation_file.slots.append(slot)

    return annotation_file


def _parse_darwin_annotation(
    annotation: Dict[str, Any],
    only_keyframes: bool = False,
    annotation_type: Optional[str] = None,
    annotation_data: Optional[Dict] = None,
) -> Optional[dt.Annotation]:
    slot_names = parse_slot_names(annotation)
    name: str = annotation["name"].strip()
    main_annotation: Optional[dt.Annotation] = None

    # Darwin JSON 2.0 representation of polygons
    if "polygon" in annotation and "paths" in annotation["polygon"]:
        bounding_box = annotation.get("bounding_box")
        paths = annotation["polygon"]["paths"]
        main_annotation = dt.make_polygon(
            name, paths, bounding_box, slot_names=slot_names
        )

    elif "polygon" in annotation and "path" in annotation["polygon"]:
        bounding_box = annotation.get("bounding_box")
        path = annotation["polygon"]["path"]
        main_annotation = dt.make_polygon(
            name, path, bounding_box, slot_names=slot_names
        )

    elif "bounding_box" in annotation:
        bounding_box = annotation["bounding_box"]
        main_annotation = dt.make_bounding_box(
            name,
            bounding_box["x"],
            bounding_box["y"],
            bounding_box["w"],
            bounding_box["h"],
            slot_names=slot_names,
        )
    elif "tag" in annotation:
        main_annotation = dt.make_tag(name, slot_names=slot_names)
    elif "line" in annotation:
        main_annotation = dt.make_line(
            name, annotation["line"]["path"], slot_names=slot_names
        )
    elif "keypoint" in annotation:
        main_annotation = dt.make_keypoint(
            name,
            annotation["keypoint"]["x"],
            annotation["keypoint"]["y"],
            slot_names=slot_names,
        )
    elif "ellipse" in annotation:
        main_annotation = dt.make_ellipse(
            name, annotation["ellipse"], slot_names=slot_names
        )
    elif "cuboid" in annotation:
        main_annotation = dt.make_cuboid(
            name, annotation["cuboid"], slot_names=slot_names
        )
    elif "skeleton" in annotation:
        main_annotation = dt.make_skeleton(
            name, annotation["skeleton"]["nodes"], slot_names=slot_names
        )
    elif "table" in annotation:
        main_annotation = dt.make_table(
            name,
            annotation["table"]["bounding_box"],
            annotation["table"]["cells"],
            slot_names=slot_names,
        )
    elif "simple_table" in annotation:
        main_annotation = dt.make_simple_table(
            name,
            annotation["simple_table"]["bounding_box"],
            annotation["simple_table"]["col_offsets"],
            annotation["simple_table"]["row_offsets"],
            slot_names=slot_names,
        )
    elif "string" in annotation:
        main_annotation = dt.make_string(
            name, annotation["string"]["sources"], slot_names=slot_names
        )
    elif "graph" in annotation:
        main_annotation = dt.make_graph(
            name,
            annotation["graph"]["nodes"],
            annotation["graph"]["edges"],
            slot_names=slot_names,
        )
    elif "mask" in annotation:
        main_annotation = dt.make_mask(name, slot_names=slot_names)
    elif "raster_layer" in annotation:
        raster_layer = annotation["raster_layer"]
        main_annotation = dt.make_raster_layer(
            name,
            raster_layer["mask_annotation_ids_mapping"],
            raster_layer["total_pixels"],
            raster_layer["dense_rle"],
            slot_names=slot_names,
        )
    elif only_keyframes:
        main_annotation = make_keyframe_annotation(
            annotation_type, annotation_data, name, slot_names
        )

    if not main_annotation:
        print(f"[WARNING] Unsupported annotation type: '{annotation.keys()}'")
        return None

    if "id" in annotation:
        main_annotation.id = annotation["id"]
    if "instance_id" in annotation:
        main_annotation.subs.append(
            dt.make_instance_id(annotation["instance_id"]["value"])
        )
    if "attributes" in annotation:
        main_annotation.subs.append(dt.make_attributes(annotation["attributes"]))
    if "text" in annotation:
        main_annotation.subs.append(dt.make_text(annotation["text"]["text"]))
    if "inference" in annotation:
        main_annotation.subs.append(
            dt.make_opaque_sub("inference", annotation["inference"])
        )
    if "directional_vector" in annotation:
        main_annotation.subs.append(
            dt.make_opaque_sub("directional_vector", annotation["directional_vector"])
        )
    if "measures" in annotation:
        main_annotation.subs.append(
            dt.make_opaque_sub("measures", annotation["measures"])
        )
    if "auto_annotate" in annotation:
        main_annotation.subs.append(
            dt.make_opaque_sub("auto_annotate", annotation["auto_annotate"])
        )

    if annotation.get("annotators") is not None:
        main_annotation.annotators = _parse_annotators(annotation["annotators"])

    if annotation.get("reviewers") is not None:
        main_annotation.reviewers = _parse_annotators(annotation["reviewers"])

    if "properties" in annotation:
        main_annotation.properties = _parse_properties(annotation["properties"])

    return main_annotation


def make_keyframe_annotation(
    annotation_type: Optional[str],
    annotation_data: Optional[Dict],
    name: str,
    slot_names: List[str],
) -> dt.Annotation:
    if annotation_type == "polygon":
        return dt.make_polygon(
            name, annotation_data["paths"], annotation_data["bounding_box"]
        )
    elif annotation_type == "bounding_box":
        return dt.make_bounding_box(
            name,
            annotation_data["x"],
            annotation_data["y"],
            annotation_data["w"],
            annotation_data["h"],
        )
    elif annotation_type == "tag":
        return dt.make_tag(name)
    elif annotation_type == "line":
        return dt.make_line(name, annotation_data["path"])
    elif annotation_type == "keypoint":
        return dt.make_keypoint(name, annotation_data["x"], annotation_data["y"])
    elif annotation_type == "ellipse":
        return dt.make_ellipse(name, annotation_data)
    elif annotation_type == "cuboid":
        return dt.make_cuboid(name, annotation_data)
    elif annotation_type == "skeleton":
        return dt.make_skeleton(name, annotation_data["nodes"])
    elif annotation_type == "table":
        return dt.make_table(
            name, annotation_data["bounding_box"], annotation_data["cells"]
        )
    elif annotation_type == "simple_table":
        return dt.make_simple_table(
            name,
            annotation_data["bounding_box"],
            annotation_data["col_offsets"],
            annotation_data["row_offsets"],
        )
    elif annotation_type == "string":
        return dt.make_string(name, annotation_data["sources"])
    elif annotation_type == "graph":
        return dt.make_graph(name, annotation_data["nodes"], annotation_data["edges"])
    elif annotation_type == "mask":
        return dt.make_mask(name)
    elif annotation_type == "raster_layer":
        return dt.make_raster_layer(
            name,
            annotation_data["mask_annotation_ids_mapping"],
            annotation_data["total_pixels"],
            annotation_data["dense_rle"],
        )
    else:
        raise ValueError(f"Unsupported annotation type: '{annotation_type}'")


def update_annotation_data(
    main_annotation_data: Dict[str, Any],
    annotation_type: Optional[str],
    annotation_data: Optional[Dict],
) -> Tuple[Optional[str], Optional[Dict]]:
    if annotation_type == "polygon":
        bounding_box = main_annotation_data.get("bounding_box")
        paths = main_annotation_data["paths"]
        annotation_data = {"paths": paths, "bounding_box": bounding_box}
    elif annotation_type == "bounding_box":
        annotation_data = {
            "x": main_annotation_data["x"],
            "y": main_annotation_data["y"],
            "w": main_annotation_data["w"],
            "h": main_annotation_data["h"],
        }
    elif annotation_type == "tag":
        annotation_data = {}
    elif annotation_type == "line":
        annotation_data = {"path": main_annotation_data["path"]}
    elif annotation_type == "keypoint":
        annotation_data = {
            "x": main_annotation_data["x"],
            "y": main_annotation_data["y"],
        }
    elif annotation_type == "ellipse":
        annotation_data = {
            "angle": main_annotation_data["angle"],
            "center": main_annotation_data["center"],
            "radius": main_annotation_data["radius"],
        }
    elif annotation_type == "cuboid":
        annotation_data = {
            "back": main_annotation_data["back"],
            "front": main_annotation_data["front"],
        }
    elif annotation_type == "skeleton":
        annotation_data = {"nodes": main_annotation_data["nodes"]}
    elif annotation_type == "table":
        annotation_type = "table"
        annotation_data = {
            "bounding_box": main_annotation_data["table"]["bounding_box"],
            "cells": main_annotation_data["table"]["cells"],
        }
    elif annotation_type == "string":
        annotation_data = {"sources": main_annotation_data["string"]["sources"]}
    elif annotation_type == "graph":
        annotation_data = {
            "nodes": main_annotation_data["graph"]["nodes"],
            "edges": main_annotation_data["graph"]["edges"],
        }
    elif annotation_type == "mask":
        annotation_data = {}
    elif annotation_type == "raster_layer":
        annotation_data = {
            "dense_rle": main_annotation_data["dense_rle"],
            "mask_annotation_ids_mapping": main_annotation_data[
                "mask_annotation_ids_mapping"
            ],
            "total_pixels": main_annotation_data["total_pixels"],
        }

    return annotation_data


def _parse_darwin_video_annotation(annotation: dict) -> Optional[dt.VideoAnnotation]:
    name = annotation["name"].strip()
    frame_annotations = {}
    keyframes: Dict[int, bool] = {}
    frames = {**annotation.get("frames", {}), **annotation.get("sections", {})}
    only_keyframes = annotation.get("only_keyframes", False)
    annotation_type, annotation_data = None, None
    if only_keyframes:
        for f, frame in frames.items():
            annotation_type, annotation_data = get_annotation_type_and_data(
                frame, annotation_type, annotation_data
            )
            if annotation_type:
                break
    for f, frame in frames.items():
        frame_annotations[int(f)] = _parse_darwin_annotation(
            {**frame, **{"name": name, "id": annotation.get("id", None)}},
            only_keyframes,
            annotation_type,
            annotation_data,
        )
        # If we hit a keyframe, we need to update annotation_data for frames later on that may be missing a main type
        if only_keyframes:
            annotation_data = update_annotation_data(
                frame_annotations[int(f)].data, annotation_type, annotation_data
            )
        keyframes[int(f)] = frame.get("keyframe", False)

    if not frame_annotations or None in frame_annotations.values():
        return None
    main_annotation = dt.make_video_annotation(
        frame_annotations,
        keyframes,
        annotation.get("ranges", annotation.get("segments", [])),
        annotation.get("interpolated", False),
        slot_names=parse_slot_names(annotation),
        properties=_parse_properties(annotation.get("properties", [])),
        hidden_areas=annotation.get("hidden_areas", []),
    )

    if "id" in annotation:
        main_annotation.id = annotation["id"]

    if "annotators" in annotation:
        main_annotation.annotators = _parse_annotators(annotation["annotators"])

    if annotation.get("reviewers") is not None:
        main_annotation.reviewers = _parse_annotators(annotation["reviewers"])

    return main_annotation


def get_annotation_type_and_data(
    frame: Dict, annotation_type: str, annotation_data: Dict
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Returns the type of a given video annotation and its data.
    """

    if "polygon" in frame:
        if frame["polygon"]["paths"]:
            bounding_box = frame.get("bounding_box")
            paths = frame["polygon"]["paths"]
            annotation_type = "polygon"
            annotation_data = {"paths": paths, "bounding_box": bounding_box}
        else:
            bounding_box = frame.get("bounding_box")
            path = frame["polygon"]["paths"]
            annotation_type = "polygon"
            annotation_data = {"paths": path, "bounding_box": bounding_box}
    elif "bounding_box" in frame:
        bounding_box = frame["bounding_box"]
        annotation_type = "bounding_box"
        annotation_data = {
            "x": bounding_box["x"],
            "y": bounding_box["y"],
            "w": bounding_box["w"],
            "h": bounding_box["h"],
        }
    elif "tag" in frame:
        annotation_type = "tag"
        annotation_data = {}
    elif "line" in frame:
        annotation_type = "line"
        annotation_data = {"path": frame["line"]["path"]}
    elif "keypoint" in frame:
        annotation_type = "keypoint"
        annotation_data = {
            "x": frame["keypoint"]["x"],
            "y": frame["keypoint"]["y"],
        }
    elif "ellipse" in frame:
        annotation_type = "ellipse"
        annotation_data = frame["ellipse"]
    elif "cuboid" in frame:
        annotation_type = "cuboid"
        annotation_data = frame["cuboid"]
    elif "skeleton" in frame:
        annotation_type = "skeleton"
        annotation_data = {"nodes": frame["skeleton"]["nodes"]}
    elif "table" in frame:
        annotation_type = "table"
        annotation_data = {
            "bounding_box": frame["table"]["bounding_box"],
            "cells": frame["table"]["cells"],
        }
    elif "string" in frame:
        annotation_type = "string"
        annotation_data = {"sources": frame["string"]["sources"]}
    elif "graph" in frame:
        annotation_type = "graph"
        annotation_type = {
            "nodes": frame["graph"]["nodes"],
            "edges": frame["graph"]["edges"],
        }
    elif "mask" in frame:
        annotation_type = "mask"
        annotation_data = {}
    elif "raster_layer" in frame:
        raster_layer = frame["raster_layer"]
        annotation_type = "raster_layer"
        annotation_data = {
            "dense_rle": raster_layer["dense_rle"],
            "mask_annotation_ids_mapping": raster_layer["mask_annotation_ids_mapping"],
            "total_pixels": raster_layer["total_pixels"],
        }

    return annotation_type, annotation_data


def _parse_darwin_raster_annotation(annotation: dict) -> Optional[dt.Annotation]:
    if not annotation.get("raster_layer"):
        raise ValueError("Raster annotation must have a 'raster_layer' field")

    id: Optional[str] = annotation.get("id")
    name: Optional[str] = annotation.get("name")
    raster_layer: Optional[dt.JSONFreeForm] = annotation.get("raster_layer")
    slot_names: Optional[List[str]] = parse_slot_names(annotation)

    if not id or not name or not raster_layer:
        raise ValueError(
            "Raster annotation must have an 'id', 'name' and 'raster_layer' field"
        )

    dense_rle, mask_annotation_ids_mapping, total_pixels = (
        raster_layer.get("dense_rle", None),
        raster_layer.get("mask_annotation_ids_mapping", None),
        raster_layer.get("total_pixels", None),
    )

    if not dense_rle or not mask_annotation_ids_mapping or not total_pixels:
        raise ValueError(
            "Raster annotation must have a 'dense_rle', 'mask_annotation_ids_mapping' and 'total_pixels' field"
        )

    new_annotation = dt.Annotation(
        dt.AnnotationClass(name.strip(), "raster_layer"),
        {
            "dense_rle": dense_rle,
            "mask_annotation_ids_mapping": mask_annotation_ids_mapping,
            "total_pixels": total_pixels,
        },
        slot_names=slot_names or [],
        id=id,
    )

    return new_annotation


def _parse_darwin_mask_annotation(annotation: dict) -> Optional[dt.Annotation]:
    id: Optional[str] = annotation.get("id")
    name: Optional[str] = annotation.get("name")
    mask: Optional[dt.JSONFreeForm] = annotation.get("mask")
    slot_names: Optional[List[str]] = parse_slot_names(annotation)

    if not id or not name or mask is None:
        raise ValueError("Mask annotation must have an 'id', 'name' and 'mask' field")

    if ("sparse_rle" in mask) and (mask["sparse_rle"] is not None):
        raise ValueError("Mask annotation field 'sparse_rle' must contain a null value")

    new_annotation = dt.Annotation(
        dt.AnnotationClass(name.strip(), "mask"),
        mask,
        slot_names=slot_names or [],
        id=id,
    )

    return new_annotation


def _parse_annotators(annotators: List[Dict[str, Any]]) -> List[dt.AnnotationAuthor]:
    if not (hasattr(annotators, "full_name") or not hasattr(annotators, "email")):
        raise AttributeError(
            "JSON file must contain annotators with 'full_name' and 'email' fields"
        )

    return [
        dt.AnnotationAuthor(annotator["full_name"], annotator["email"])
        for annotator in annotators
    ]


def _parse_properties(
    properties: List[Dict[str, Any]],
) -> Optional[List[SelectedProperty]]:
    selected_properties = []
    for property in properties:
        frame_index = property.get("frame_index")
        selected_properties.append(
            SelectedProperty(
                frame_index=frame_index if frame_index is not None else "global",
                name=property.get("name", None),
                value=property.get("value", None),
            )
        )

    return selected_properties or None


def split_video_annotation(annotation: dt.AnnotationFile) -> List[dt.AnnotationFile]:
    """
    Splits the given video ``AnnotationFile`` into several video ``AnnotationFile``s, one for each
    ``frame_url``.

    Parameters
    ----------
    annotation : dt.AnnotationFile
        The video ``AnnotationFile`` we want to split.

    Returns
    -------
    List[dt.AnnotationFile]
        A list with the split video ``AnnotationFile``\\s.

    Raises
    ------
    AttributeError
        If the given ``AnnotationFile`` is not a video annotation, or if the given annotation has
        no ``frame_url`` attribute.
    """
    if not annotation.is_video:
        raise AttributeError("This is not a video annotation")

    # changes here from annotation.frame_urls to annotation.frame_count with frame_urls as backup
    # due to addition of long videos feature, where frame_urls is no longer available.
    # frame_count should be available for both, however existing annotations will not have this
    if not annotation.frame_count and not annotation.frame_urls:
        raise AttributeError("This Annotation has no frames")
    urls = annotation.frame_urls or [None] * (annotation.frame_count or 1)
    frame_annotations = []
    for i, frame_url in enumerate(urls):
        print(i)
        annotations = [
            a.frames[i]
            for a in annotation.annotations
            if isinstance(a, dt.VideoAnnotation) and i in a.frames
        ]
        annotation_classes: Set[dt.AnnotationClass] = {
            annotation.annotation_class for annotation in annotations
        }
        filename: str = f"{Path(annotation.filename).stem}/{i:07d}.png"
        frame_annotations.append(
            dt.AnnotationFile(
                annotation.path,
                filename,
                annotation_classes,
                annotations,
                [],
                False,
                annotation.image_width,
                annotation.image_height,
                frame_url,
                annotation.workview_url,
                annotation.seq,
                dataset_name=annotation.dataset_name,
                item_id=annotation.item_id,
                slots=annotation.slots,
                remote_path=annotation.remote_path,
            )
        )

    return frame_annotations


def parse_slot_names(annotation: dict) -> List[str]:
    return annotation.get("slot_names", [])


def ispolygon(annotation: dt.AnnotationClass) -> bool:
    """
    Returns whether or not the given ``AnnotationClass`` is a polygon.

    Parameters
    ----------
    annotation : AnnotationClass
        The ``AnnotationClass`` to evaluate.

    Returns
    -------
    ``True`` is the given ``AnnotationClass`` is a polygon, ``False`` otherwise.
    """
    return annotation.annotation_type == "polygon"


def convert_polygons_to_sequences(
    polygons: List[Union[dt.Polygon, List[dt.Polygon]]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    rounding: bool = True,
) -> List[List[Union[int, float]]]:
    """
    Converts a list of polygons, encoded as a list of dictionaries of into a list of nd.arrays
    of coordinates.

    Parameters
    ----------
    polygons : Iterable[dt.Polygon]
        Non empty list of coordinates in the format ``[{x: x1, y:y1}, ..., {x: xn, y:yn}]`` or a
        list of them as ``[[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]]``.
    height : Optional[int], default: None
        Maximum height for a polygon coordinate.
    width : Optional[int], default: None
        Maximum width for a polygon coordinate.
    rounding : bool, default: True
        Whether or not to round values when creating sequences.

    Returns
    -------
    sequences: List[ndarray[float]]
        List of arrays of coordinates in the format [[x1, y1, x2, y2, ..., xn, yn], ...,
        [x1, y1, x2, y2, ..., xn, yn]]

    Raises
    ------
    ValueError
        If the given list is a falsy value (such as ``[]``) or if it's structure is incorrect.
    """
    if not polygons:
        raise ValueError("No polygons provided")
    # If there is a single polygon composing the instance then this is
    # transformed to polygons = [[{x: x1, y:y1}, ..., {x: xn, y:yn}]]
    list_polygons: List[dt.Polygon] = []
    if isinstance(polygons[0], list):
        list_polygons = cast(List[dt.Polygon], polygons)
    else:
        list_polygons = cast(List[dt.Polygon], [polygons])

    if not isinstance(list_polygons[0], list) or not isinstance(
        list_polygons[0][0], dict
    ):
        raise ValueError("Unknown input format")

    sequences: List[List[Union[int, float]]] = []
    for polygon in list_polygons:
        path: List[Union[int, float]] = []
        for point in polygon:
            # Clip coordinates to the image size
            x = max(min(point["x"], width - 1) if width else point["x"], 0)
            y = max(min(point["y"], height - 1) if height else point["y"], 0)
            if rounding:
                path.append(round(x))
                path.append(round(y))
            else:
                path.append(x)
                path.append(y)
        sequences.append(path)
    return sequences


def convert_xyxy_to_bounding_box(box: List[Union[int, float]]) -> dt.BoundingBox:
    """
    Converts a list of xy coordinates representing a bounding box into a dictionary.
    This is used by in-platform model training.

    Parameters
    ----------
    box : List[Union[int, float]]
        List of arrays of coordinates in the format [x1, y1, x2, y2]

    Returns
    -------
    BoundingBox
        Bounding box in the format ``{x: x1, y: y1, h: height, w: width}``.

    Raises
    ------
    ValueError
        If ``box`` has an incorrect format.
    """
    if not isinstance(box[0], float) and not isinstance(box[0], int):
        raise ValueError("Unknown input format")

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return {"x": x1, "y": y1, "w": width, "h": height}


def convert_polygons_to_mask(
    polygons: List, height: int, width: int, value: Optional[int] = 1
) -> np.ndarray:
    """
    Converts a list of polygons, encoded as a list of dictionaries into an ``nd.array`` mask.

    Parameters
    ----------
    polygons: list
        List of coordinates in the format ``[{x: x1, y:y1}, ..., {x: xn, y:yn}]`` or a list of them
        as  ``[[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]]``.
    height : int
        The maximum height for the created mask.
    width : int
        The maximum width for the created mask.
    value : Optional[int], default: 1
        The drawing value for ``upolygon``.

    Returns
    -------
    ndarray
        ``ndarray`` mask of the polygon(s).
    """
    sequence = convert_polygons_to_sequences(polygons, height=height, width=width)
    mask = np.zeros((height, width)).astype(np.uint8)
    draw_polygon(mask, sequence, value)
    return mask


def chunk(items: List[Any], size: int) -> Iterator[Any]:
    """
    Splits the given list into chunks of the given size and yields them.

    Parameters
    ----------
    items : List[Any]
        The list of items to be split.
    size : int
        The size of each split.

    Yields
    ------
    Iterator[Any]
        A chunk of the of the given size.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


def is_unix_like_os() -> bool:
    """
    Returns ``True`` if the executing OS is Unix-based (Ubuntu or MacOS, for example) or ``False``
    otherwise.

    Returns
    --------
    bool
        True for Unix-based systems, False otherwise.
    """
    return platform.system() != "Windows"


def has_json_content_type(response: Response) -> bool:
    """
    Returns ``True`` if response has application/json content type or ``False``
    otherwise.

    Returns
    --------
    bool
        True for application/json content type, False otherwise.
    """
    return "application/json" in response.headers.get("content-type", "")


def get_response_content(response: Response) -> Any:
    """
    Returns json content if response has application/json content-type, otherwise returns text.

    Returns
    --------
    Any
        Json or text content.
    """
    if has_json_content_type(response):
        return response.json()
    else:
        return response.text


def _parse_version(data: dict) -> dt.AnnotationFileVersion:
    version_string = data.get("version", "1.0")
    major, minor, suffix = re.findall(r"^(\d+)\.(\d+)(.*)$", version_string)[0]
    return dt.AnnotationFileVersion(int(major), int(minor), suffix)


def _data_to_annotations(
    data: Dict[str, Any],
) -> List[Union[dt.Annotation, dt.VideoAnnotation]]:
    raw_image_annotations = filter(
        lambda annotation: (
            ("frames" not in annotation)
            and ("raster_layer" not in annotation)
            and ("mask" not in annotation)
        ),
        data["annotations"],
    )
    raw_video_annotations = filter(
        lambda annotation: "frames" in annotation, data["annotations"]
    )
    raw_raster_annotations = filter(
        lambda annotation: "raster_layer" in annotation, data["annotations"]
    )
    raw_mask_annotations = filter(
        lambda annotation: "mask" in annotation, data["annotations"]
    )
    image_annotations: List[dt.Annotation] = list(
        filter(None, map(_parse_darwin_annotation, raw_image_annotations))
    )
    video_annotations: List[dt.VideoAnnotation] = list(
        filter(None, map(_parse_darwin_video_annotation, raw_video_annotations))
    )
    raster_annotations: List[dt.Annotation] = list(
        filter(None, map(_parse_darwin_raster_annotation, raw_raster_annotations))
    )
    mask_annotations: List[dt.Annotation] = list(
        filter(None, map(_parse_darwin_mask_annotation, raw_mask_annotations))
    )

    return [
        *image_annotations,
        *video_annotations,
        *raster_annotations,
        *mask_annotations,
    ]


def _supported_schema_versions() -> Dict[Tuple[int, int, str], str]:
    return {
        (
            2,
            0,
            "",
        ): "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json"
    }


def _default_schema(version: dt.AnnotationFileVersion) -> Optional[str]:
    return _supported_schema_versions().get(
        (version.major, version.minor, version.suffix)
    )


def get_annotation_files_from_dir(path: Path) -> Iterator[str]:
    """
    Returns an iterator of all the JSON annotation files in the given directory.
    Ignores the .v7/metadata.json properties manifest file if present.

    Parameters
    ----------
    path : Path
        The directory to search for JSON annotation files.

    Returns
    -------
    Iterator[str]
        An iterator of all the JSON annotation files in the given directory.
    """
    return (
        str(filepath)
        for filepath in sorted(path.glob("**/*.json"))
        if "/.v7/" not in str(filepath) and "\\.v7\\" not in str(filepath)
    )


def convert_sequences_to_polygons(
    sequences: List[Union[List[int], List[float]]],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Dict[str, List[dt.Polygon]]:
    """
    Converts a list of polygons, encoded as a list of dictionaries of into a list of nd.arrays
    of coordinates. This is used by the backend.

    Parameters
    ----------
    sequences : List[Union[List[int], List[float]]]
        List of arrays of coordinates in the format ``[x1, y1, x2, y2, ..., xn, yn]`` or as a list
        of them as ``[[x1, y1, x2, y2, ..., xn, yn], ..., [x1, y1, x2, y2, ..., xn, yn]]``.
    height : Optional[int], default: None
        Maximum height for a polygon coordinate.
    width : Optional[int], default: None
        Maximum width for a polygon coordinate.

    Returns
    -------
    Dict[str, List[dt.Polygon]]
        Dictionary with the key ``path`` containing a list of coordinates in the format of
        ``[[{x: x1, y:y1}, ..., {x: xn, y:yn}], ..., [{x: x1, y:y1}, ..., {x: xn, y:yn}]]``.

    Raises
    ------
    ValueError
        If sequences is a falsy value (such as ``[]``) or if it is in an incorrect format.
    """
    if not sequences:
        raise ValueError("No sequences provided")
    # If there is a single sequences composing the instance then this is
    # transformed to polygons = [[x1, y1, ..., xn, yn]]
    if not isinstance(sequences[0], list):
        sequences = [sequences]

    if not isinstance(sequences[0][0], (int, float)):
        raise ValueError("Unknown input format")

    def grouped(iterable, n):
        return zip(*[iter(iterable)] * n)

    polygons = []
    for sequence in sequences:
        path = []
        for x, y in grouped(sequence, 2):
            # Clip coordinates to the image size
            x = max(min(x, width - 1) if width else x, 0)
            y = max(min(y, height - 1) if height else y, 0)
            path.append({"x": x, "y": y})
        polygons.append(path)
    return {"path": polygons}


def get_primary_plane_from_nifti(affine: np.ndarray) -> str:
    """
    Determine the primary orientation of a NIfTI image from its affine matrix.

    Parameters
    ----------
    affine : numpy.ndarray
        4x4 affine matrix from NIfTI file

    Returns
    -------
    str
        The primary anatomical plane - 'AXIAL', 'SAGITTAL', or 'CORONAL'
    """
    # Extract the rotation/scaling matrix (first 3x3 elements)
    rotation_scale = affine[:3, :3]

    # Get absolute values to determine magnitude of each direction
    abs_rotation = np.abs(rotation_scale)

    # Find which axis has the largest magnitude for each dimension
    primary_directions = np.argmax(abs_rotation, axis=1)

    # Find which dimension has the largest overall magnitude
    magnitudes = [abs_rotation[i, primary_directions[i]] for i in range(3)]
    primary_plane = np.argmax(magnitudes)
    if primary_plane == 0:
        return "SAGITTAL"
    elif primary_plane == 1:
        return "CORONAL"
    else:
        return "AXIAL"
