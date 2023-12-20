import sys
import uuid
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rich.console import Console

from darwin.utils import attempt_decode

console = Console()
try:
    import cc3d
    import nibabel as nib
except ImportError:
    import_fail_string = """
    You must install darwin-py with pip install nibabel connected-components-3d
    in order to import with using nifti format
    """
    console.print(import_fail_string)
    sys.exit(1)
import numpy as np
from jsonschema import validate
from upolygon import find_contours

import darwin.datatypes as dt
from darwin.importer.formats.nifti_schemas import nifti_import_schema


def parse_path(path: Path) -> Optional[List[dt.AnnotationFile]]:
    """
    Parses the given ``nifti`` file and returns a ``List[dt.AnnotationFile]`` with the parsed
    information.

    Parameters
    ----------
    path : Path
        The ``Path`` to the ``nifti`` file.

    Returns
    -------
    Optional[List[dt.AnnotationFile]]
        Returns ``None`` if the given file is not in ``json`` format, or ``List[dt.AnnotationFile]``
        otherwise.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if path.suffix != ".json":
        console.print(
            "Skipping file: {} (not a json file)".format(path), style="bold yellow"
        )
        return None
    data = attempt_decode(path)
    try:
        validate(data, schema=nifti_import_schema)
    except Exception:
        console.print(
            "Skipping file: {} (invalid json file, see schema for details)".format(
                path
            ),
            style="bold yellow",
        )
        return None
    nifti_annotations = data.get("data")
    if nifti_annotations is None or nifti_annotations == []:
        console.print(
            "Skipping file: {} (no data found)".format(path), style="bold yellow"
        )
        return None
    annotation_files = []
    for nifti_annotation in nifti_annotations:
        annotation_file = _parse_nifti(
            Path(nifti_annotation["label"]),
            nifti_annotation["image"],
            path,
            class_map=nifti_annotation.get("class_map"),
            mode=nifti_annotation.get("mode", "image"),
            slot_names=nifti_annotation.get("slot_names", []),
            is_mpr=nifti_annotation.get("is_mpr", False),
        )
        annotation_files.append(annotation_file)
    return annotation_files


def _parse_nifti(
    nifti_path: Path,
    filename: Path,
    json_path: Path,
    class_map: Dict,
    mode: str,
    slot_names: List[str],
    is_mpr: bool,
) -> dt.AnnotationFile:
    img, pixdims = process_nifti(nib.load(nifti_path))

    processed_class_map = process_class_map(class_map)
    video_annotations = []
    if mode == "instances":  # For each instance produce a video annotation
        for class_name, class_idxs in processed_class_map.items():
            if class_name == "background":
                continue
            class_img = np.isin(img, class_idxs).astype(np.uint8)
            cc_img, num_labels = cc3d.connected_components(class_img, return_N=True)
            for instance_id in range(1, num_labels):
                _video_annotations = get_polygon_video_annotations(
                    cc_img,
                    class_idxs=[instance_id],
                    class_name=class_name,
                    slot_names=slot_names,
                    is_mpr=is_mpr,
                    pixdims=pixdims,
                )
                if _video_annotations:
                    video_annotations += _video_annotations
    elif mode == "video":  # For each class produce a single video annotation
        for class_name, class_idxs in processed_class_map.items():
            if class_name == "background":
                continue
            _video_annotations = get_polygon_video_annotations(
                img,
                class_idxs=class_idxs,
                class_name=class_name,
                slot_names=slot_names,
                is_mpr=is_mpr,
                pixdims=pixdims,
            )
            if _video_annotations is None:
                continue
            video_annotations += _video_annotations
    elif mode == "mask":
        video_annotations = get_mask_video_annotations(
            img,
            processed_class_map,
            slot_names,
        )
    if mode in ["video", "instances"]:
        annotation_classes = {
            dt.AnnotationClass(class_name, "polygon", "polygon")
            for class_name in class_map.values()
        }
    elif mode == "mask":
        annotation_classes = {
            dt.AnnotationClass(class_name, "mask", "mask")
            for class_name in class_map.values()
        }

    return dt.AnnotationFile(
        path=json_path,
        filename=str(filename),
        remote_path="/",
        annotation_classes=annotation_classes,
        annotations=video_annotations,
        slots=[
            dt.Slot(
                name=slot_name,
                type="dicom",
                source_files=[{"url": None, "file_name": str(filename)}],
            )
            for slot_name in slot_names
        ],
    )


def get_polygon_video_annotations(
    volume: np.ndarray,
    class_name: str,
    class_idxs: List[int],
    slot_names: List[str],
    is_mpr: bool,
    pixdims: Tuple[float],
) -> Optional[List[dt.VideoAnnotation]]:
    if not is_mpr:
        return nifti_to_video_polygon_annotation(
            volume,
            class_name,
            class_idxs,
            slot_names,
            view_idx=2,
            pixdims=pixdims,
        )
    elif is_mpr and len(slot_names) == 3:
        video_annotations = []
        for view_idx, slot_name in enumerate(slot_names):
            _video_annotations = nifti_to_video_polygon_annotation(
                volume,
                class_name,
                class_idxs,
                [slot_name],
                view_idx=view_idx,
                pixdims=pixdims,
            )
            video_annotations += _video_annotations
        return video_annotations
    else:
        raise Exception("If is_mpr is True, slot_names must be of length 3")


def get_mask_video_annotations(
    volume: np.ndarray, processed_class_map: Dict, slot_names: List[str]
) -> Optional[List[dt.VideoAnnotation]]:
    """
    The function takes a volume and a class map and returns a list of video annotations

    We write a single raster layer for the volume but K mask annotations, where K is the number of classes.
    """
    frame_annotations = OrderedDict()
    all_mask_annotations = defaultdict(lambda: OrderedDict())
    # This is a dictionary of class_names to generated mask_annotation_ids
    mask_annotation_ids = {
        class_name: str(uuid.uuid4()) for class_name in processed_class_map.keys()
    }
    # We need to create a new mapping dictionary where the keys are the mask_annotation_ids
    # and the values are the new integers which we use in the raster layer
    mask_annotation_ids_mapping = {}
    map_from_nifti_idx_to_raster_idx = {0: 0}  # 0 is the background class
    for i in range(volume.shape[2]):
        slice_mask = volume[:, :, i].astype(np.uint8)
        for raster_idx, (class_name, class_idxs) in enumerate(
            processed_class_map.items()
        ):
            if class_name == "background":
                continue
            class_mask = np.isin(slice_mask, class_idxs).astype(np.uint8).copy()
            if class_mask.sum() == 0:
                continue
            all_mask_annotations[class_name][i] = dt.make_mask(
                class_name, subs=None, slot_names=slot_names
            )
            all_mask_annotations[class_name][i].id = mask_annotation_ids[class_name]
            mask_annotation_ids_mapping[mask_annotation_ids[class_name]] = (
                raster_idx + 1
            )
            for class_idx in class_idxs:
                map_from_nifti_idx_to_raster_idx[class_idx] = raster_idx + 1
    # Now that we've created all the mask annotations, we need to create the raster layer
    # We only map the mask_annotation_ids which appear in any given frame.
    for i in range(volume.shape[2]):
        slice_mask = volume[:, :, i].astype(
            np.uint8
        )  # Product requirement: We only support 255 classes!
        # We need to convert from nifti_idx to raster_idx
        slice_mask = np.vectorize(
            lambda key: map_from_nifti_idx_to_raster_idx.get(key, 0)
        )(slice_mask)
        dense_rle = convert_to_dense_rle(slice_mask)
        raster_annotation = dt.make_raster_layer(
            class_name="__raster_layer__",
            mask_annotation_ids_mapping=mask_annotation_ids_mapping,
            total_pixels=class_mask.size,
            dense_rle=dense_rle,
            slot_names=slot_names,
        )
        frame_annotations[i] = raster_annotation
    all_frame_ids = list(frame_annotations.keys())
    if not all_frame_ids:
        return None
    if len(all_frame_ids) == 1:
        segments = [[all_frame_ids[0], all_frame_ids[0] + 1]]
    elif len(all_frame_ids) > 1:
        segments = [[min(all_frame_ids), max(all_frame_ids)]]
    raster_video_annotation = dt.make_video_annotation(
        frame_annotations,
        keyframes={f_id: True for f_id in all_frame_ids},
        segments=segments,
        interpolated=False,
        slot_names=slot_names,
    )
    mask_video_annotations = []
    for class_name, mask_annotations in all_mask_annotations.items():
        mask_video_annotation = dt.make_video_annotation(
            mask_annotations,
            keyframes={f_id: True for f_id in all_frame_ids},
            segments=segments,
            interpolated=False,
            slot_names=slot_names,
        )
        mask_video_annotation.id = mask_annotation_ids[class_name]
        mask_video_annotations.append(mask_video_annotation)

    return [raster_video_annotation] + mask_video_annotations


def nifti_to_video_polygon_annotation(
    volume: np.ndarray,
    class_name: str,
    class_idxs: List[int],
    slot_names: List[str],
    view_idx: int = 2,
    pixdims: Tuple[int, int, int] = (1, 1, 1),
) -> Optional[List[dt.VideoAnnotation]]:
    frame_annotations = OrderedDict()
    for i in range(volume.shape[view_idx]):
        if view_idx == 2:
            slice_mask = volume[:, :, i].astype(np.uint8)
            _pixdims = [pixdims[0], pixdims[1]]
        elif view_idx == 1:
            slice_mask = volume[:, i, :].astype(np.uint8)
            _pixdims = [pixdims[0], pixdims[2]]
        elif view_idx == 0:
            slice_mask = volume[i, :, :].astype(np.uint8)
            _pixdims = [pixdims[1], pixdims[2]]
        class_mask = np.isin(slice_mask, class_idxs).astype(np.uint8).copy()
        if class_mask.sum() == 0:
            continue
        polygon = mask_to_polygon(
            mask=class_mask, class_name=class_name, pixdims=_pixdims
        )
        if polygon is None:
            continue
        frame_annotations[i] = polygon
    all_frame_ids = list(frame_annotations.keys())
    if not all_frame_ids:
        return None
    if len(all_frame_ids) == 1:
        segments = [[all_frame_ids[0], all_frame_ids[0] + 1]]
    elif len(all_frame_ids) > 1:
        segments = [[min(all_frame_ids), max(all_frame_ids)]]
    video_annotation = dt.make_video_annotation(
        frame_annotations,
        keyframes={f_id: True for f_id in all_frame_ids},
        segments=segments,
        interpolated=False,
        slot_names=slot_names,
    )
    return [video_annotation]


def mask_to_polygon(
    mask: np.ndarray, class_name: str, pixdims: List[float]
) -> Optional[dt.Annotation]:
    def adjust_for_pixdims(x, y, pixdims):
        if pixdims[1] > pixdims[0]:
            return {"x": y, "y": x * pixdims[1] / pixdims[0]}
        elif pixdims[1] < pixdims[0]:
            return {"x": y * pixdims[0] / pixdims[1], "y": x}
        else:
            return {"x": y, "y": x}

    _labels, external_paths, _internal_paths = find_contours(mask)
    if len(external_paths) > 1:
        paths = []
        for external_path in external_paths:
            # skip paths with less than 2 points
            if len(external_path) // 2 <= 2:
                continue
            path = [
                adjust_for_pixdims(x, y, pixdims)
                for x, y in zip(external_path[0::2], external_path[1::2])
            ]
            paths.append(path)
        if len(paths) > 1:
            polygon = dt.make_complex_polygon(class_name, paths)
        elif len(paths) == 1:
            polygon = dt.make_polygon(
                class_name,
                point_path=paths[0],
            )
        else:
            return None
    elif len(external_paths) == 1:
        external_path = external_paths[0]
        if len(external_path) < 6:
            return None
        polygon = dt.make_polygon(
            class_name,
            point_path=[
                adjust_for_pixdims(x, y, pixdims)
                for x, y in zip(external_path[0::2], external_path[1::2])
            ],
        )
    else:
        return None
    return polygon


def process_class_map(class_map):
    """
    This function takes a class_map and returns a dictionary with the class names as keys and
    all the corresponding class indexes as values.
    """
    processed_class_map = defaultdict(list)
    for key, value in class_map.items():
        processed_class_map[value].append(int(key))
    return processed_class_map


def rectify_header_sform_qform(img_nii):
    """
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py

    Args:
        img_nii: nifti image object
    """
    d = img_nii.header["dim"][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = affine_to_spacing(sform, r=d)
    norm_qform = affine_to_spacing(qform, r=d)
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header["sform_code"] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header["qform_code"] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm = affine_to_spacing(img_nii.affine, r=d)
    warnings.warn(f"Modifying image pixdim from {pixdim} to {norm}")

    img_nii.header.set_zooms(norm)
    return img_nii


def affine_to_spacing(
    affine: np.ndarray, r: int = 3, dtype=float, suppress_zeros: bool = True
) -> np.ndarray:
    """
    Copied over from monai.data.utils - https://docs.monai.io/en/stable/_modules/monai/data/utils.html

    Computing the current spacing from the affine matrix.

    Args:
        affine: a d x d affine matrix.
        r: indexing based on the spatial rank, spacing is computed from `affine[:r, :r]`.
        dtype: data type of the output.
        suppress_zeros: whether to surpress the zeros with ones.

    Returns:
        an `r` dimensional vector of spacing.
    """
    spacing = np.sqrt(np.sum(affine[:r, :r] * affine[:r, :r], axis=0))
    if suppress_zeros:
        spacing[spacing == 0] = 1.0
    return spacing


def correct_nifti_header_if_necessary(img_nii):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img_nii: nifti image object
    """
    if img_nii.header.get("dim") is None:
        return img_nii  # not nifti?
    dim = img_nii.header["dim"][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = affine_to_spacing(img_nii.affine, r=dim)
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, "get_sform"):
        return rectify_header_sform_qform(img_nii)
    return img_nii


def process_nifti(
    input_data: Union[Sequence[nib.nifti1.Nifti1Image], nib.nifti1.Nifti1Image]
):
    """
    Function which takes in a single nifti path or a list of nifti paths
    and returns the pixel_array, affine and pixdim
    """
    if isinstance(input_data, nib.nifti1.Nifti1Image):
        img = correct_nifti_header_if_necessary(input_data)
        img = nib.funcs.as_closest_canonical(img)
        nib.orientations.aff2axcodes(img.affine)
        # TODO: Future feature to pass custom ornt could go here.
        ornt = [[0.0, -1.0], [1.0, -1.0], [1.0, -1.0]]
        data_array = nib.orientations.apply_orientation(img.get_fdata(), ornt)
        pixdims = img.header.get_zooms()
        return data_array, pixdims


def convert_to_dense_rle(raster: np.ndarray) -> List[int]:
    dense_rle, prev_val, cnt = [], None, 0
    for val in raster.T.flat:
        if val == prev_val:
            cnt += 1
        else:
            if prev_val is not None:
                dense_rle.extend([int(prev_val), int(cnt)])
            prev_val, cnt = val, 1
    dense_rle.extend([int(prev_val), int(cnt)])
    return dense_rle
