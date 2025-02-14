import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import nibabel as nib
import numpy as np

from darwin.exporter.exporter import darwin_to_dt_gen
from darwin.exporter.formats import nifti
from tests.fixtures import *


def test_video_annotation_nifti_export_single_slot(team_slug_darwin_json_v2: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            annotations_dir = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti/releases/latest/annotations"
            )
            video_annotation_filepaths = [annotations_dir / "hippocampus_001.nii.json"]
            video_annotations = list(
                darwin_to_dt_gen(video_annotation_filepaths, False)
            )
            nifti.export(video_annotations, output_dir=tmpdir)
            export_im = nib.load(
                annotations_dir / "hippocampus_001_hippocampus.nii.gz"
            ).get_fdata()
            expected_im = nib.load(
                annotations_dir / "hippocampus_001_hippocampus.nii.gz"
            ).get_fdata()
            assert np.allclose(export_im, expected_im)


def test_video_annotation_nifti_export_multi_slot(team_slug_darwin_json_v2: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            annotations_dir = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti/releases/latest/annotations"
            )
            video_annotation_filepaths = [
                annotations_dir / "hippocampus_multislot.nii.json"
            ]
            video_annotations = list(
                darwin_to_dt_gen(video_annotation_filepaths, False)
            )
            nifti.export(video_annotations, output_dir=tmpdir)
            names = ["1", "2", "3", "4", "5"]
            for slotname in names:
                export_im = nib.load(
                    annotations_dir
                    / f"hippocampus_multislot_{slotname}_test_hippo.nii.gz"
                ).get_fdata()
                expected_im = nib.load(
                    annotations_dir
                    / f"hippocampus_multislot_{slotname}_test_hippo.nii.gz"
                ).get_fdata()
                assert np.allclose(export_im, expected_im)


def test_video_annotation_nifti_export_mpr(team_slug_darwin_json_v2: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            annotations_dir = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti/releases/latest/annotations"
            )
            video_annotation_filepaths = [
                annotations_dir / "hippocampus_multislot_001_mpr.json"
            ]
            video_annotations = list(
                darwin_to_dt_gen(video_annotation_filepaths, False)
            )
            nifti.export(video_annotations, output_dir=Path(tmpdir))
            export_im = nib.load(
                annotations_dir / "hippocampus_001_mpr_1_test_hippo.nii.gz"
            ).get_fdata()
            expected_im = nib.load(
                annotations_dir / "hippocampus_001_mpr_1_test_hippo.nii.gz"
            ).get_fdata()
            assert np.allclose(export_im, expected_im)


def test_export_calls_populate_output_volumes_from_polygons(
    team_slug_darwin_json_v2: str,
):
    with patch(
        "darwin.exporter.formats.nifti.populate_output_volumes_from_polygons"
    ) as mock:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile("tests/data.zip") as zfile:
                zfile.extractall(tmpdir)
            annotations_dir = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti/releases/latest/annotations"
            )
            video_annotation_filepaths = [annotations_dir / "polygon_only.json"]
            video_annotations = list(
                darwin_to_dt_gen(video_annotation_filepaths, False)
            )
            nifti.export(video_annotations, output_dir=Path(tmpdir))
            mock.assert_called()


def test_export_calls_populate_output_volumes_from_raster_layer(
    team_slug_darwin_json_v2: str,
):
    with patch(
        "darwin.exporter.formats.nifti.populate_output_volumes_from_raster_layer"
    ) as mock:
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile("tests/data.zip") as zfile:
                zfile.extractall(tmpdir)
            annotations_dir = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti/releases/latest/annotations"
            )
            video_annotation_filepaths = [annotations_dir / "mask_only.json"]
            video_annotations = list(
                darwin_to_dt_gen(video_annotation_filepaths, False)
            )
            nifti.export(video_annotations, output_dir=Path(tmpdir))
            mock.assert_called()


def test_export_creates_file_for_polygons_and_masks(
    team_slug_darwin_json_v2: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmpdir)
            annotations_dir = (
                Path(tmpdir)
                / team_slug_darwin_json_v2
                / "nifti/releases/latest/annotations"
            )
            video_annotation_files = {
                "mask_only.json": ["hippocampus_multislot_3_test_hippo_LOIN_m.nii.gz"],
                "polygon_only.json": [
                    "hippocampus_multislot_3_test_hippo_create_class_1.nii.gz"
                ],
                "polygon_and_mask.json": [
                    "hippocampus_multislot_3_test_hippo_create_class_1.nii.gz",
                    "hippocampus_multislot_3_test_hippo_LOIN_m.nii.gz",
                ],
                "empty.json": ["hippocampus_multislot_3_test_hippo_.nii.gz"],
            }
            for video_annotation_file in video_annotation_files:
                video_annotation_filepaths = [annotations_dir / video_annotation_file]
                video_annotations = list(
                    darwin_to_dt_gen(video_annotation_filepaths, False)
                )
                nifti.export(video_annotations, output_dir=Path(tmpdir))
                for output_file in video_annotation_files[video_annotation_file]:
                    assert (
                        Path(tmpdir) / output_file
                    ).exists(), (
                        f"Expected file {output_file} does not exist in {tmpdir}"
                    )
                # Empty the directory for the next test
                for output_file in video_annotation_files[video_annotation_file]:
                    (Path(tmpdir) / output_file).unlink()


def test_shift_polygon_coords_no_scaling():
    """Test polygon coordinate shifting where all pixdims values are 1.0."""
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [1.0, 1.0, 1.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "AXIAL", legacy=False)
    expected = [{"x": 20.0, "y": 10.0}, {"x": 40.0, "y": 30.0}, {"x": 60.0, "y": 50.0}]
    assert result == expected


def test_shift_polygon_coords_axial_plane():
    """Test polygon coordinate shifting in axial plane."""
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [0.25, 0.5, 2.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "AXIAL", legacy=False)
    expected = [
        {"x": 40.0, "y": 40.0},
        {"x": 80.0, "y": 120.0},
        {"x": 120.0, "y": 200.0},
    ]
    assert result == expected


def test_shift_polygon_coords_coronal_plane():
    """Test polygon coordinate shifting in coronal plane."""
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [0.25, 0.5, 2.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "CORONAL", legacy=False)
    expected = [
        {"x": 10.0, "y": 40.0},
        {"x": 20.0, "y": 120.0},
        {"x": 30.0, "y": 200.0},
    ]
    assert result == expected


def test_shift_polygon_coords_sagittal_plane():
    """Test polygon coordinate shifting in sagittal plane."""
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [0.25, 0.5, 2.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "SAGITTAL", legacy=False)
    expected = [
        {"x": 10.0, "y": 20.0},
        {"x": 20.0, "y": 60.0},
        {"x": 30.0, "y": 100.0},
    ]
    assert result == expected


def test_shift_polygon_coords_legacy():
    """Test polygon coordinate shifting with legacy mode."""
    # Test case where pixdim[1] > pixdim[0]
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [0.25, 0.5, 2.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "AXIAL", legacy=True)
    expected = [{"x": 20.0, "y": 20.0}, {"x": 40.0, "y": 60.0}, {"x": 60.0, "y": 100.0}]
    assert result == expected

    # Test case where pixdim[1] < pixdim[0]
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [0.5, 0.25, 2.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "AXIAL", legacy=True)
    expected = [{"x": 40.0, "y": 10.0}, {"x": 80.0, "y": 30.0}, {"x": 120.0, "y": 50.0}]
    assert result == expected

    # Test case where pixdim[1] == pixdim[0]
    polygon = [{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}, {"x": 50.0, "y": 60.0}]
    pixdim = [0.5, 0.5, 2.0]
    result = nifti.shift_polygon_coords(polygon, pixdim, "AXIAL", legacy=True)
    expected = [{"x": 20.0, "y": 10.0}, {"x": 40.0, "y": 30.0}, {"x": 60.0, "y": 50.0}]
    assert result == expected
