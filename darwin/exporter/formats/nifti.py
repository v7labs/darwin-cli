import ast
import json as native_json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from rich.console import Console
from rich.theme import Theme


def _console_theme() -> Theme:
    return Theme(
        {
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "info": "bold deep_sky_blue1",
        }
    )


console = Console(theme=_console_theme())
try:
    import nibabel as nib
except ImportError:
    import_fail_string = r"""
    You must install darwin-py with pip install darwin-py\[medical]
    in order to export using nifti forma
    """
    console.print(import_fail_string)
    exit()
import numpy as np

import darwin.datatypes as dt
from darwin.utils import convert_polygons_to_mask


class Plane(Enum):
    XY = 0
    XZ = 1
    YZ = 2


@dataclass
class Volume:
    pixel_array: np.ndarray
    affine: Optional[np.ndarray]
    original_affine: Optional[np.ndarray]
    dims: List
    pixdims: List
    class_name: str
    series_instance_uid: str
    from_raster_layer: bool
    primary_plane: str


def export(
    annotation_files: Iterable[dt.AnnotationFile],
    output_dir: Path,
) -> None:
    """
    Exports the given ``AnnotationFile``\\s into nifti format inside of the given
    ``output_dir``. Deletes everything within ``output_dir/masks`` before writting to it.

    Parameters
    ----------
    annotation_files : Iterable[dt.AnnotationFile]
        The ``AnnotationFile``\\s to be exported.
    output_dir : Path
        The folder where the new instance mask files will be.

    Returns
    -------
    sends output volumes, image_id and output_dir to the write_output_volume_to_disk function

    """
    video_annotations = list(annotation_files)
    for video_annotation in video_annotations:
        slot_name = video_annotation.slots[0].name
        try:
            medical_metadata = video_annotation.slots[0].metadata
            legacy = not medical_metadata.get("handler") == "MONAI"  # type: ignore
            plane_map = medical_metadata.get("plane_map", {slot_name: "AXIAL"})
            primary_plane = plane_map.get(slot_name, "AXIAL")
        except (KeyError, AttributeError):
            legacy = True
            primary_plane = "AXIAL"

        image_id = check_for_error_and_return_imageid(video_annotation, output_dir)
        if not isinstance(image_id, str):
            continue
        polygon_class_names = [
            ann.annotation_class.name
            for ann in video_annotation.annotations
            if ann.annotation_class.annotation_type == "polygon"
        ]
        # Check if there are any rasters in the annotation, these are created with a _m suffix
        # in addition to those created from polygons.
        annotation_types = [
            a.annotation_class.annotation_type for a in video_annotation.annotations
        ]
        mask_present = "raster_layer" in annotation_types and "mask" in annotation_types
        output_volumes = build_output_volumes(
            video_annotation,
            class_names_to_export=polygon_class_names,
            from_raster_layer=False,
            mask_present=mask_present,
            primary_plane=primary_plane,
        )
        slot_map = {slot.name: slot for slot in video_annotation.slots}
        polygon_annotations = [
            ann
            for ann in video_annotation.annotations
            if ann.annotation_class.annotation_type == "polygon"
        ]
        if polygon_annotations:
            populate_output_volumes_from_polygons(
                polygon_annotations, slot_map, output_volumes, legacy=legacy
            )
        write_output_volume_to_disk(
            output_volumes,
            image_id=image_id,
            output_dir=output_dir,
            legacy=legacy,
            filename=video_annotation.filename,
        )
        # Need to map raster layers to SeriesInstanceUIDs
        if mask_present:
            mask_id_to_classname = {
                ann.id: ann.annotation_class.name
                for ann in video_annotation.annotations
                if ann.annotation_class.annotation_type == "mask"
            }
            raster_output_volumes = build_output_volumes(
                video_annotation,
                class_names_to_export=list(mask_id_to_classname.values()),
                from_raster_layer=True,
                primary_plane=primary_plane,
            )

            # This assumes only one raster_layer annotation. If we allow multiple raster layers per annotation file we need to change this.
            raster_layer_annotation = [
                ann
                for ann in video_annotation.annotations
                if ann.annotation_class.annotation_type == "raster_layer"
            ][0]
            if raster_layer_annotation:
                populate_output_volumes_from_raster_layer(
                    annotation=raster_layer_annotation,
                    mask_id_to_classname=mask_id_to_classname,
                    slot_map=slot_map,
                    output_volumes=raster_output_volumes,
                    primary_plane=primary_plane,
                )
            write_output_volume_to_disk(
                raster_output_volumes,
                image_id=image_id,
                output_dir=output_dir,
                legacy=legacy,
                filename=video_annotation.filename,
            )


def build_output_volumes(
    video_annotation: dt.AnnotationFile,
    from_raster_layer: bool = False,
    class_names_to_export: List[str] = None,
    mask_present: Optional[bool] = False,
    primary_plane: str = "AXIAL",
) -> Dict:
    """
    This is a function to create the output volumes based on the whole annotation file

    Parameters
    ----------
    video_annotation : dt.AnnotationFile
        The ``AnnotationFile``\\s to be exported.
    from_raster_layer : bool
        Whether the output volumes are being built from raster layers or not
    class_names_to_export : List[str]
        The list of class names to export
    mask_present: bool
        If mask annotations are present in the annotation
    primary_plane: str
        The primary plane of the annotation
    Returns
    -------
    output_volumes: Dict
        The output volume built per class

    """
    # Builds a map of class to integer, if its a polygon we use the class name as is
    # for the mask annotations we append a suffix _m to ensure backwards compatibility

    output_volumes = {}
    for slot in video_annotation.slots:
        slot_metadata = slot.metadata
        assert slot_metadata is not None
        series_instance_uid = slot_metadata.get(
            "SeriesInstanceUID", "SeriesIntanceUIDNotProvided"
        )
        # Builds output volumes per class
        volume_dims, pixdims, affine, original_affine = process_metadata(slot.metadata)
        if not mask_present and not class_names_to_export:
            class_names_to_export = [
                ""
            ]  # If there are no annotations to export, we still need to create an empty volume

        output_volumes[series_instance_uid] = {
            class_name: Volume(
                pixel_array=np.zeros(volume_dims),
                affine=affine,
                original_affine=original_affine,
                dims=volume_dims,
                pixdims=pixdims,
                series_instance_uid=series_instance_uid,
                class_name=class_name,
                from_raster_layer=from_raster_layer,
                primary_plane=primary_plane,
            )
            for class_name in class_names_to_export
        }
    return output_volumes


def check_for_error_and_return_imageid(
    video_annotation: dt.AnnotationFile, output_dir: Path
) -> Union[str, bool]:
    """Given the video_annotation file and the output directory, checks for a range of errors and
    returns messages accordingly.

    Parameters
    ----------
    video_annotation : dt.AnnotationFile
        The ``AnnotationFile``\\s to be exported.
    output_dir : Path
        The folder where the new instance mask files will be.

    Returns
    -------
    Union[str, bool]
        Returns the image_id if no errors are found, otherwise returns False
    """
    # Check if all item slots have the correct file-extension
    for slot in video_annotation.slots:
        for source_file in slot.source_files:
            filename = Path(source_file.file_name)
            if not (
                filename.name.lower().endswith(".nii.gz")
                or filename.name.lower().endswith(".nii")
                or filename.name.lower().endswith(".dcm")
            ):
                return create_error_message_json(
                    "Misconfigured filename, not ending in .nii, .nii.gz or .dcm. Are you sure this is medical data?",
                    output_dir,
                    str(filename),
                )

    filename = Path(video_annotation.filename)
    if filename.name.lower().endswith(".nii.gz"):
        image_id = re.sub(r"(?i)\.nii\.gz$", "", str(filename))
    elif filename.name.lower().endswith(".nii"):
        image_id = re.sub(r"(?i)\.nii$", "", str(filename))
    elif filename.name.lower().endswith(".dcm"):
        image_id = re.sub(r"(?i)\.dcm$", "", str(filename))
    else:
        image_id = str(filename)

    if video_annotation is None:
        return create_error_message_json(
            "video_annotation not found", output_dir, image_id
        )
    if video_annotation is None:
        return create_error_message_json(
            "video_annotation not found", output_dir, image_id
        )

    for slot in video_annotation.slots:
        # Pick the first slot to take the metadata from. We assume that all slots have the same metadata.
        metadata = slot.metadata
        if metadata is None:
            return create_error_message_json(
                f"No metadata found for {str(filename)}, are you sure this is medical data?",
                output_dir,
                image_id,
            )

        volume_dims, pixdim, affine, _ = process_metadata(metadata)
        if affine is None or pixdim is None or volume_dims is None:
            return create_error_message_json(
                f"Missing one of affine, pixdim or shape in metadata for {str(filename)}, try reuploading file",
                output_dir,
                image_id,
            )
    return image_id


def update_pixel_array(
    volume: Dict,
    annotation_class_name: str,
    im_mask: np.ndarray,
    primary_plane: str,
    frame_idx: int,
) -> Dict:
    """Updates the pixel array of the given volume with the given mask.

    Parameters
    ----------
    volume : Dict
        Volume with pixel array to be updated
    annotation_class_name : str
        Name of the annotation class
    im_mask : np.ndarray
        Mask to be added to the pixel array
    primary_plane : str
        Plane of the mask
    frame_idx : int
        Frame index of the mask

    Returns
    -------
    Dict
        Updated volume
    """
    plane_to_slice = {
        "AXIAL": np.s_[:, :, frame_idx],
        "CORONAL": np.s_[:, frame_idx, :],
        "SAGITTAL": np.s_[frame_idx, :, :],
    }
    if primary_plane in plane_to_slice:
        slice_ = plane_to_slice[primary_plane]
        volume[annotation_class_name].pixel_array[slice_] = np.logical_or(
            im_mask,
            volume[annotation_class_name].pixel_array[slice_],
        )
    return volume


def populate_output_volumes_from_polygons(
    annotations: List[Union[dt.Annotation, dt.VideoAnnotation]],
    slot_map: Dict,
    output_volumes: Dict,
    legacy: bool = False,
):
    """
    Populates the output volumes with the given polygon annotations. The annotations are converted into masks
    and added to the corresponding volume based on the series instance UID.

    Parameters
    ----------
    annotations : List[Union[dt.Annotation, dt.VideoAnnotation]]
        List of polygon annotations used to populate the volume with
    slot_map : Dict
        Dictionary of the different slots within the annotation file
    output_volumes : Dict
        Volumes created from the build_output_volumes file
    legacy : bool, default=False
        If ``True``, the exporter will use the legacy calculation.
        If ``False``, the exporter will use the new calculation by dividing with pixdims.
    """
    for annotation in annotations:
        slot_name = annotation.slot_names[0]
        slot = slot_map[slot_name]
        series_instance_uid = slot.metadata.get(
            "SeriesInstanceUID", "SeriesIntanceUIDNotProvided"
        )
        volume = output_volumes.get(series_instance_uid)
        frames = annotation.frames

        for frame_idx in frames.keys():
            primary_plane = volume[annotation.annotation_class.name].primary_plane
            dims = volume[annotation.annotation_class.name].dims
            if primary_plane == "AXIAL":
                height, width = dims[0], dims[1]
            elif primary_plane == "CORONAL":
                height, width = dims[0], dims[2]
            elif primary_plane == "SAGITTAL":
                height, width = dims[1], dims[2]
            pixdims = volume[annotation.annotation_class.name].pixdims
            frame_data = frames[frame_idx].data
            if "paths" in frame_data:
                # Dealing with a complex polygon
                polygons = [
                    shift_polygon_coords(
                        polygon_path,
                        pixdims,
                        primary_plane=volume[
                            annotation.annotation_class.name
                        ].primary_plane,
                        legacy=legacy,
                    )
                    for polygon_path in frame_data["paths"]
                ]
            else:
                continue
            im_mask = convert_polygons_to_mask(polygons, height=height, width=width)
            volume = update_pixel_array(
                output_volumes[series_instance_uid],
                annotation.annotation_class.name,
                im_mask,
                primary_plane,
                frame_idx,
            )


def populate_output_volumes_from_raster_layer(
    annotation: dt.Annotation,
    mask_id_to_classname: Dict,
    slot_map: Dict,
    output_volumes: Dict,
    primary_plane: str,
) -> Dict:
    """
    Populates the output volumes provided with the raster layer annotations

    Parameters
    ----------
    annotation : Union[dt.Annotation, dt.VideoAnnotation]
        The Union of these two files used to populate the volume with
    mask_id_to_classname : Dict
        Map from mask id to class names
    slot_map: Dict
        Dictionary of the different slots within the annotation file
    output_volumes : Dict
        volumes created from the build_output_volumes file
    primary_plane: str
        The primary plane of the volume containing the annotation
    Returns
    -------
    volume : dict
        Returns dict of volumes with class names as keys and volumes as values
    """
    slot_name = annotation.slot_names[0]
    slot = slot_map[slot_name]
    series_instance_uid = slot.metadata.get(
        "SeriesInstanceUID", "SeriesIntanceUIDNotProvided"
    )
    volume = output_volumes.get(series_instance_uid)
    frames = annotation.frames
    mask_annotation_ids_mapping = {}
    multilabel_volume = np.zeros(slot.metadata["shape"][1:])
    for frame_idx in sorted(frames.keys()):
        frame_idx = int(frame_idx)
        frame_data = annotation.frames[frame_idx]
        dense_rle = frame_data.data["dense_rle"]
        mask_2d = decode_rle(dense_rle, slot.width, slot.height)
        if primary_plane == "AXIAL":
            multilabel_volume[:, :, frame_idx] = mask_2d.T
        elif primary_plane == "CORONAL":
            multilabel_volume[:, frame_idx, :] = mask_2d.T
        elif primary_plane == "SAGITTAL":
            multilabel_volume[frame_idx, :, :] = mask_2d.T
        mask_annotation_ids_mapping.update(
            frame_data.data["mask_annotation_ids_mapping"]
        )
    # Now we convert this multilabel array into this dictionary of output volumes
    # in order to re-use the write_output_volume_to_disk function.
    for mask_id, class_name in mask_id_to_classname.items():
        volume = output_volumes[series_instance_uid]
        mask_int_id = mask_annotation_ids_mapping[mask_id]
        # We want to create a binary mask for each class
        volume[class_name].pixel_array = np.where(
            multilabel_volume == int(mask_int_id), 1, volume[class_name].pixel_array
        )
    return volume


def write_output_volume_to_disk(
    output_volumes: Dict,
    image_id: str,
    output_dir: Union[str, Path],
    legacy: bool = False,
    filename: str = None,
) -> None:
    """Writes the given output volumes to disk.

    Parameters
    ----------
    output_volumes : Dict
        Output volumes to be written to disk
    image_id : str
        The specific image id
    output_dir : Union[str, Path]
        The output directory to write the volumes to
    legacy : bool, default=False
        If ``True``, the exporter will use the legacy calculation.
        If ``False``, the exporter will use the new calculation by dividing with pixdims.
    filename: str
        The filename of the dataset item

    Returns
    -------
    None
    """

    # volumes are the values of this nested dict
    def unnest_dict_to_list(d: Dict) -> List:
        result = []
        for value in d.values():
            if isinstance(value, dict):
                result.extend(unnest_dict_to_list(value))
            else:
                result.append(value)
        return result

    volumes = unnest_dict_to_list(output_volumes)
    for volume in volumes:
        img = nib.Nifti1Image(
            dataobj=volume.pixel_array.astype(np.int16),
            affine=volume.affine,
        )
        img = _get_reoriented_nifti_image(img, volume, legacy, filename)
        if volume.from_raster_layer:
            output_path = Path(output_dir) / f"{image_id}_{volume.class_name}_m.nii.gz"
        else:
            output_path = Path(output_dir) / f"{image_id}_{volume.class_name}.nii.gz"
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        nib.save(img=img, filename=output_path)


def _get_reoriented_nifti_image(
    img: nib.Nifti1Image, volume: Dict, legacy: bool, filename: str
) -> nib.Nifti1Image:
    """
    Reorients the given NIfTI image based on the affine of the originally uploaded file.

    Files that were uploaded before the `MED_2D_VIEWER` feature are `legacy`. Non-legacy
    files are uploaded and re-oriented to the `LPI` orientation. Legacy NifTI
    files were treated differently. These files were re-oriented to `LPI`, but their
    affine was stored as `RAS`, which is the opposite orientation. We therefore need to
    flip the axes of these images to ensure alignment.

    Parameters
    ----------
    img: nib.Nifti1Image
        The NIfTI image to be reoriented
    volume: Dict
        The volume containing the affine and original affine
    legacy: bool
        If ``True``, the exporter will flip all axes of the image if the dataset item
        is not a DICOM
        If ``False``, the exporter will not flip the axes
    filename: str
        The filename of the dataset item
    """
    if volume.original_affine is not None:
        img_ax_codes = nib.orientations.aff2axcodes(volume.affine)
        orig_ax_codes = nib.orientations.aff2axcodes(volume.original_affine)
        img_ornt = nib.orientations.axcodes2ornt(img_ax_codes)
        orig_ornt = nib.orientations.axcodes2ornt(orig_ax_codes)
        transform = nib.orientations.ornt_transform(img_ornt, orig_ornt)
        img = img.as_reoriented(transform)
        is_dicom = filename.lower().endswith(".dcm")
        if legacy and not is_dicom:
            img = nib.Nifti1Image(
                np.flip(img.get_fdata(), (0, 1, 2)).astype(np.int16), img.affine
            )
    return img


def shift_polygon_coords(
    polygon: List[Dict[str, float]],
    pixdim: List[float],
    primary_plane: str,
    legacy: bool = False,
) -> List[Dict[str, float]]:
    """
    Shifts input polygon coordinates based on the primary plane and the pixdim of the volume
    the polygon belongs to.

    If the volume is a legacy volume, we perform isotropic scaling

    Parameters
    ----------
    polygon : List[Dict[str, float]]
        The polygon to be shifted
    pixdim : List[float]
        The (x, y, z) pixel dimensons of the image
    primary_plane : str
        The primary plane of the volume that the polygon belongs to
    legacy : bool
        Whether this polygon is being exported from a volume that requires legacy NifTI scaling

    Returns
    -------
    List[Dict[str, Number]]
        The shifted polygon
    """
    if primary_plane == "AXIAL":
        pixdim = [pixdim[0], pixdim[1]]
    elif primary_plane == "CORONAL":
        pixdim = [pixdim[0], pixdim[2]]
    elif primary_plane == "SAGITTAL":
        pixdim = [pixdim[1], pixdim[2]]
    if legacy:
        # Need to make it clear that we flip x/y because we need to take the transpose later.
        if pixdim[1] > pixdim[0]:
            return [{"x": p["y"], "y": p["x"] * pixdim[1] / pixdim[0]} for p in polygon]
        elif pixdim[1] < pixdim[0]:
            return [{"x": p["y"] * pixdim[0] / pixdim[1], "y": p["x"]} for p in polygon]
        else:
            return [{"x": p["y"], "y": p["x"]} for p in polygon]
    else:
        return [{"x": p["y"] // pixdim[1], "y": p["x"] // pixdim[0]} for p in polygon]


def get_view_idx(frame_idx: int, groups: List) -> int:
    """Returns the view index for the given frame index and groups.

    Parameters
    ----------
    frame_idx : int
        Frame index
    groups : List
        List of groups

    Returns
    -------
    int
        View index
    """
    if groups is None:
        return 0
    for view_idx, group in enumerate(groups):
        if frame_idx in group:
            return view_idx


def process_metadata(metadata: Dict) -> Tuple:
    """Processes the metadata and returns the volume dimensions, pixel dimensions, affine and original affine.

    Parameters
    ----------
    metadata : Dict
        Metadata to be processed

    Returns
    -------
    Tuple
        Tuple containing volume dimensions, pixel dimensions, affine and original affine
    """
    volume_dims = metadata.get("shape")
    pixdim = metadata.get("pixdim")
    affine = process_affine(metadata.get("affine"))
    original_affine = process_affine(metadata.get("original_affine"))
    # If the original affine is in the medical payload of metadata then use it
    if isinstance(pixdim, str):
        pixdim = ast.literal_eval(pixdim)
        if isinstance(pixdim, tuple) or isinstance(pixdim, list):
            if len(pixdim) == 4:
                pixdim = pixdim[1:]
            if len(pixdim) != 3:
                pixdim = None
        else:
            pixdim = None
    if isinstance(volume_dims, list):
        if volume_dims:
            if volume_dims[0] == 1:  # remove first singleton dimension
                volume_dims = volume_dims[1:]
        else:
            volume_dims = None
    return volume_dims, pixdim, affine, original_affine


def process_affine(affine: Union[str, List, np.ndarray]) -> Optional[np.ndarray]:
    """Converts affine to numpy array if it is not already.

    Parameters
    ----------
    affine : Union[str, List, np.ndarray]
        affine object to be converted

    Returns
    -------
    Optional[np.ndarray]
        affine as numpy array
    """
    if isinstance(affine, str):
        affine = np.squeeze(
            np.array([ast.literal_eval(lst) for lst in affine.split("\n")])
        )
    elif isinstance(affine, list):
        affine = np.array(affine).astype(float)
    else:
        return
    if isinstance(affine, np.ndarray):
        return affine


def create_error_message_json(
    error_message: str, output_dir: Union[str, Path], image_id: str
) -> bool:
    """Creates a json file with the given error message.

    Parameters
    ----------
    error_message : str
        Error message to be written to the file
    output_dir : Union[str, Path]
        Output directory
    image_id : str
        Associated image id

    Returns
    -------
    bool
        Always returns False
    """
    output_path = Path(output_dir) / f"{image_id}_error.json"
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    with open(output_path, "w") as f:
        native_json.dump({"error": error_message}, f)

    return False


def decode_rle(rle_data: List[int], width: int, height: int) -> np.ndarray:
    """Decodes run-length encoding (RLE) data into a mask array.

    Parameters
    ----------
    rle_data : List[int]
        List of RLE data
    width : int
        Width of the data
    height : int
        Height of the data

    Returns
    -------
    np.ndarray
        RLE data
    """
    total_pixels = width * height
    mask = np.zeros(total_pixels, dtype=np.uint8)
    pos = 0
    for i in range(0, len(rle_data), 2):
        value = rle_data[i]
        length = rle_data[i + 1]
        mask[pos : pos + length] = value
        pos += length
    return mask.reshape(height, width)
