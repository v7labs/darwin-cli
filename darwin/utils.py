"""
Contains several unrelated utility functions used across the SDK.
"""

import json
import platform
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
    Union,
    cast,
)

import deprecation
import numpy as np
from rich.progress import ProgressType, track
from upolygon import draw_polygon

import darwin.datatypes as dt
from darwin.config import Config
from darwin.exceptions import OutdatedDarwinJSONFormat, UnsupportedFileType
from darwin.version import __version__

if TYPE_CHECKING:
    from darwin.client import Client


SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".jfif", ".tif", ".tiff", ".bmp", ".svs"]
SUPPORTED_VIDEO_EXTENSIONS = [".avi", ".bpm", ".dcm", ".mov", ".mp4", ".pdf", ".ndpi"]
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS


def is_extension_allowed(extension: str) -> bool:
    """
    Returns whether or not the given video or image extension is allowed.

    Parameters
    ----------
    extension : str
        The extension.

    Returns
    -------
    bool
        Whether or not the given extension is allowed.
    """
    return extension.lower() in SUPPORTED_EXTENSIONS


def is_image_extension_allowed(extension: str) -> bool:
    """
    Returns whether or not the given image extension is allowed.

    Parameters
    ----------
    extension : str
        The image extension.

    Returns
    -------
    bool
        Whether or not the given extension is allowed.
    """
    return extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_video_extension_allowed(extension: str) -> bool:
    """
    Returns whether or not the given video extension is allowed.

    Parameters
    ----------
    extension : str
        The video extension.

    Returns
    -------
    bool
        Whether or not the given extension is allowed.
    """
    return extension.lower() in SUPPORTED_VIDEO_EXTENSIONS


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


def get_progress_bar(array: List[dt.AnnotationFile], description: Optional[str] = None) -> Iterable[ProgressType]:
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
    files: List[dt.PathLike], *, files_to_exclude: List[dt.PathLike] = [], recursive: bool = True
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
            found_files.extend([f for f in path.glob(pattern) if is_extension_allowed(f.suffix)])
        elif is_extension_allowed(path.suffix):
            found_files.append(path)
        else:
            raise UnsupportedFileType(path)

    return [f for f in found_files if f not in map(Path, files_to_exclude)]


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
    client: "Client", default_team: Optional[str] = None, config_path: Optional[Path] = None
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
    config.set_team(team=team_config.slug, api_key=team_config.api_key, datasets_dir=team_config.datasets_dir)
    config.set_global(api_endpoint=client.url, base_url=client.base_url, default_team=default_team)

    return config


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Access the dictionary's key directly",
)
def get_local_filename(metadata: Dict[str, Any]) -> str:
    """
    Returns the value of the ``filename`` key from the given dictionary.

    Parameters
    ----------
    metadata : Dict[str, Any]
        A Dictionary with a ``filename`` key.

    Returns
    -------
    str
        The value  of the ``filename`` key.
    """
    return metadata["filename"]


def _get_local_filename(metadata: Dict[str, Any]) -> str:
    return metadata["filename"]


def parse_darwin_json(path: Path, count: Optional[int]) -> Optional[dt.AnnotationFile]:
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
    with path.open() as f:
        data = json.load(f)
        if "annotations" not in data:
            return None
        if "fps" in data["image"] or "frame_count" in data["image"]:
            return _parse_darwin_video(path, data, count)
        else:
            return _parse_darwin_image(path, data, count)


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Use 'parse_darwin_json' instead",
)
def parse_darwin_image(path: Path, data: Dict[str, Any], count: Optional[int]) -> dt.AnnotationFile:
    """
    Parses the given JSON file in v7's darwin proprietary format. Works only for images.

    Parameters
    ----------
    path : Path
        Path to the file to parse.
    data : Dict[str, Any]
        The decoded JSON file in Python format.
    count : Optional[int]
        Optional count parameter. Used only if the 's image sequence is None.

    Returns
    -------
    dt.AnnotationFile
        An AnnotationFile with the information from the parsed JSON file.
    """

    annotations: List[dt.Annotation] = list(filter(None, map(parse_darwin_annotation, data["annotations"])))
    annotation_classes: Set[dt.AnnotationClass] = set([annotation.annotation_class for annotation in annotations])
    return dt.AnnotationFile(
        path,
        _get_local_filename(data["image"]),
        annotation_classes,
        annotations,
        False,
        data["image"].get("width"),
        data["image"].get("height"),
        data["image"].get("url"),
        data["image"].get("workview_url"),
        data["image"].get("seq", count),
        None,
        data["image"].get("path", "/"),
    )


def _parse_darwin_image(path: Path, data: Dict[str, Any], count: Optional[int]) -> dt.AnnotationFile:
    annotations: List[dt.Annotation] = list(filter(None, map(_parse_darwin_annotation, data["annotations"])))
    annotation_classes: Set[dt.AnnotationClass] = set([annotation.annotation_class for annotation in annotations])
    return dt.AnnotationFile(
        path,
        _get_local_filename(data["image"]),
        annotation_classes,
        annotations,
        False,
        data["image"].get("width"),
        data["image"].get("height"),
        data["image"].get("url"),
        data["image"].get("workview_url"),
        data["image"].get("seq", count),
        None,
        data["image"].get("path", "/"),
    )


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Use 'parse_darwin_json' instead",
)
def parse_darwin_video(path: Path, data: Dict[str, Any], count: Optional[int]) -> dt.AnnotationFile:
    """
    Parses the given JSON file in v7's darwin proprietary format. Works for playback videos.

    Parameters
    ----------
    path : Path
        Path to the file to parse.
    data : Dict[str, Any]
        The decoded JSON file in Python format.
    count : Optional[int]
        Optional count parameter. Used only if the data["image"]["seq"] sequence is None.

    Returns
    -------
    dt.AnnotationFile
        An AnnotationFile with the information from the parsed JSON file.
    """

    annotations: List[dt.VideoAnnotation] = list(filter(None, map(parse_darwin_video_annotation, data["annotations"])))
    annotation_classes: Set[dt.AnnotationClass] = set([annotation.annotation_class for annotation in annotations])

    if "width" not in data["image"] or "height" not in data["image"]:
        raise OutdatedDarwinJSONFormat("Missing width/height in video, please re-export")

    return dt.AnnotationFile(
        path,
        _get_local_filename(data["image"]),
        annotation_classes,
        annotations,
        True,
        data["image"].get("width"),
        data["image"].get("height"),
        data["image"].get("url"),
        data["image"].get("workview_url"),
        data["image"].get("seq", count),
        data["image"].get("frame_urls"),
        data["image"].get("path", "/"),
    )


def _parse_darwin_video(path: Path, data: Dict[str, Any], count: Optional[int]) -> dt.AnnotationFile:
    annotations: List[dt.VideoAnnotation] = list(filter(None, map(_parse_darwin_video_annotation, data["annotations"])))
    annotation_classes: Set[dt.AnnotationClass] = set([annotation.annotation_class for annotation in annotations])

    if "width" not in data["image"] or "height" not in data["image"]:
        raise OutdatedDarwinJSONFormat("Missing width/height in video, please re-export")

    return dt.AnnotationFile(
        path,
        _get_local_filename(data["image"]),
        annotation_classes,
        annotations,
        True,
        data["image"].get("width"),
        data["image"].get("height"),
        data["image"].get("url"),
        data["image"].get("workview_url"),
        data["image"].get("seq", count),
        data["image"].get("frame_urls"),
        data["image"].get("path", "/"),
    )


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Use 'parse_darwin_json' instead",
)
def parse_darwin_annotation(annotation: Dict[str, Any]) -> Optional[dt.Annotation]:
    name: str = annotation["name"]
    main_annotation: Optional[dt.Annotation] = None
    if "polygon" in annotation:
        bounding_box = annotation.get("bounding_box")
        if "additional_paths" in annotation["polygon"]:
            paths = [annotation["polygon"]["path"]] + annotation["polygon"]["additional_paths"]
            main_annotation = dt.make_complex_polygon(name, paths, bounding_box)
        else:
            main_annotation = dt.make_polygon(name, annotation["polygon"]["path"], bounding_box)
    elif "complex_polygon" in annotation:
        bounding_box = annotation.get("bounding_box")
        if "additional_paths" in annotation["complex_polygon"]:
            paths = annotation["complex_polygon"]["path"] + annotation["complex_polygon"]["additional_paths"]
            main_annotation = dt.make_complex_polygon(name, paths, bounding_box)
        else:
            main_annotation = dt.make_complex_polygon(name, annotation["complex_polygon"]["path"], bounding_box)
    elif "bounding_box" in annotation:
        bounding_box = annotation["bounding_box"]
        main_annotation = dt.make_bounding_box(
            name, bounding_box["x"], bounding_box["y"], bounding_box["w"], bounding_box["h"]
        )
    elif "tag" in annotation:
        main_annotation = dt.make_tag(name)
    elif "line" in annotation:
        main_annotation = dt.make_line(name, annotation["line"]["path"])
    elif "keypoint" in annotation:
        main_annotation = dt.make_keypoint(name, annotation["keypoint"]["x"], annotation["keypoint"]["y"])
    elif "ellipse" in annotation:
        main_annotation = dt.make_ellipse(name, annotation["ellipse"])
    elif "cuboid" in annotation:
        main_annotation = dt.make_cuboid(name, annotation["cuboid"])
    elif "skeleton" in annotation:
        main_annotation = dt.make_skeleton(name, annotation["skeleton"]["nodes"])

    if not main_annotation:
        print(f"[WARNING] Unsupported annotation type: '{annotation.keys()}'")
        return None

    if "instance_id" in annotation:
        main_annotation.subs.append(dt.make_instance_id(annotation["instance_id"]["value"]))
    if "attributes" in annotation:
        main_annotation.subs.append(dt.make_attributes(annotation["attributes"]))
    if "text" in annotation:
        main_annotation.subs.append(dt.make_text(annotation["text"]["text"]))

    return main_annotation


def _parse_darwin_annotation(annotation: Dict[str, Any]) -> Optional[dt.Annotation]:
    name: str = annotation["name"]
    main_annotation: Optional[dt.Annotation] = None
    if "polygon" in annotation:
        bounding_box = annotation.get("bounding_box")
        if "additional_paths" in annotation["polygon"]:
            paths = [annotation["polygon"]["path"]] + annotation["polygon"]["additional_paths"]
            main_annotation = dt.make_complex_polygon(name, paths, bounding_box)
        else:
            main_annotation = dt.make_polygon(name, annotation["polygon"]["path"], bounding_box)
    elif "complex_polygon" in annotation:
        bounding_box = annotation.get("bounding_box")
        if "additional_paths" in annotation["complex_polygon"]:
            paths = annotation["complex_polygon"]["path"] + annotation["complex_polygon"]["additional_paths"]
            main_annotation = dt.make_complex_polygon(name, paths, bounding_box)
        else:
            main_annotation = dt.make_complex_polygon(name, annotation["complex_polygon"]["path"], bounding_box)
    elif "bounding_box" in annotation:
        bounding_box = annotation["bounding_box"]
        main_annotation = dt.make_bounding_box(
            name, bounding_box["x"], bounding_box["y"], bounding_box["w"], bounding_box["h"]
        )
    elif "tag" in annotation:
        main_annotation = dt.make_tag(name)
    elif "line" in annotation:
        main_annotation = dt.make_line(name, annotation["line"]["path"])
    elif "keypoint" in annotation:
        main_annotation = dt.make_keypoint(name, annotation["keypoint"]["x"], annotation["keypoint"]["y"])
    elif "ellipse" in annotation:
        main_annotation = dt.make_ellipse(name, annotation["ellipse"])
    elif "cuboid" in annotation:
        main_annotation = dt.make_cuboid(name, annotation["cuboid"])
    elif "skeleton" in annotation:
        main_annotation = dt.make_skeleton(name, annotation["skeleton"]["nodes"])

    if not main_annotation:
        print(f"[WARNING] Unsupported annotation type: '{annotation.keys()}'")
        return None

    if "instance_id" in annotation:
        main_annotation.subs.append(dt.make_instance_id(annotation["instance_id"]["value"]))
    if "attributes" in annotation:
        main_annotation.subs.append(dt.make_attributes(annotation["attributes"]))
    if "text" in annotation:
        main_annotation.subs.append(dt.make_text(annotation["text"]["text"]))

    return main_annotation


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Use 'parse_darwin_json' instead",
)
def parse_darwin_video_annotation(annotation: dict) -> dt.VideoAnnotation:
    name = annotation["name"]
    frame_annotations = {}
    keyframes: Dict[int, bool] = {}
    for f, frame in annotation["frames"].items():
        frame_annotations[int(f)] = parse_darwin_annotation({**frame, **{"name": name}})
        keyframes[int(f)] = frame.get("keyframe", False)

    return dt.make_video_annotation(
        frame_annotations, keyframes, annotation["segments"], annotation.get("interpolated", False)
    )


def _parse_darwin_video_annotation(annotation: dict) -> dt.VideoAnnotation:
    name = annotation["name"]
    frame_annotations = {}
    keyframes: Dict[int, bool] = {}
    for f, frame in annotation["frames"].items():
        frame_annotations[int(f)] = _parse_darwin_annotation({**frame, **{"name": name}})
        keyframes[int(f)] = frame.get("keyframe", False)

    return dt.make_video_annotation(
        frame_annotations, keyframes, annotation["segments"], annotation.get("interpolated", False)
    )


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
        A list with the split video ``AnnotationFile``s.

    Raises
    ------
    AttributeError
        If the given ``AnnotationFile`` is not a video annotation, or if the given annotation has
        no ``frame_url`` attribute.
    """
    if not annotation.is_video:
        raise AttributeError("this is not a video annotation")

    if not annotation.frame_urls:
        raise AttributeError("This Annotation has no frame urls")

    frame_annotations = []
    for i, frame_url in enumerate(annotation.frame_urls):
        annotations = [
            a.frames[i] for a in annotation.annotations if isinstance(a, dt.VideoAnnotation) and i in a.frames
        ]
        annotation_classes: Set[dt.AnnotationClass] = set([annotation.annotation_class for annotation in annotations])
        filename: str = f"{Path(annotation.filename).stem}/{i:07d}.png"

        frame_annotations.append(
            dt.AnnotationFile(
                annotation.path,
                filename,
                annotation_classes,
                annotations,
                False,
                annotation.image_width,
                annotation.image_height,
                frame_url,
                annotation.workview_url,
                annotation.seq,
            )
        )
    return frame_annotations


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
    return annotation.annotation_type in ["polygon", "complex_polygon"]


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

    if not isinstance(list_polygons[0], list) or not isinstance(list_polygons[0][0], dict):
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


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Do not use.",
)
def convert_sequences_to_polygons(
    sequences: List[Union[List[int], List[float]]], height: Optional[int] = None, width: Optional[int] = None
) -> Dict[str, List[dt.Polygon]]:
    """
    Converts a list of polygons, encoded as a list of dictionaries of into a list of nd.arrays
    of coordinates.

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


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Do not use.",
)
def convert_xyxy_to_bounding_box(box: List[Union[int, float]]) -> dt.BoundingBox:
    """
    Converts a list of xy coordinates representing a bounding box into a dictionary.

    Parameters
    ----------
    box : ist[Union[int, float]]
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


@deprecation.deprecated(
    deprecated_in="0.7.5",
    removed_in="0.8.0",
    current_version=__version__,
    details="Do not use.",
)
def convert_bounding_box_to_xyxy(box: dt.BoundingBox) -> List[float]:
    """
    Converts dictionary representing a bounding box into a list of xy coordinates.

    Parameters
    ----------
    box : BoundingBox
        Bounding box in the format ``{x: x1, y: y1, h: height, w: width}``.

    Returns
    -------
    List[float]
        List of arrays of coordinates in the format ``[x1, y1, x2, y2]``.
    """

    x2 = box["x"] + box["width"]
    y2 = box["y"] + box["height"]
    return [box["x"], box["y"], x2, y2]


def convert_polygons_to_mask(polygons: List, height: int, width: int, value: Optional[int] = 1) -> np.ndarray:
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
