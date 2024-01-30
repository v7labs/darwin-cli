import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from darwin.config import Config
from darwin.torch.dataset import (
    ClassificationDataset,
    InstanceSegmentationDataset,
    ObjectDetectionDataset,
    SemanticSegmentationDataset,
    get_dataset,
)
from tests.fixtures import *  # noqa: F403


def generic_dataset_test(ds, n, size):
    weights = ds.measure_weights()
    img = ds[0][0]
    assert img.shape[-2] == size[0] and img.shape[-1] == size[1]
    assert len(weights) == len(ds.classes)
    assert np.isclose(np.sum(weights), 1)
    assert len(ds) == n


class TestClassificationDataset:
    def test_should_correctly_create_a_single_label_dataset(
        self, team_slug_darwin_json_v2: str, team_extracted_dataset_path: Path
    ) -> None:
        root = team_extracted_dataset_path / team_slug_darwin_json_v2 / "sl"
        ds = ClassificationDataset(dataset_path=root, release_name="latest")

        generic_dataset_test(ds, n=20, size=(50, 50))
        assert not ds.is_multi_label

    def test_should_correctly_create_a_multi_label_dataset(
        self, team_slug_darwin_json_v2: str, team_extracted_dataset_path: Path
    ) -> None:
        root = team_extracted_dataset_path / team_slug_darwin_json_v2 / "ml"
        ds = ClassificationDataset(dataset_path=root, release_name="latest")

        generic_dataset_test(ds, n=20, size=(50, 50))
        assert ds.is_multi_label


class TestInstanceSegmentationDataset:
    def test_should_correctly_create_a_instance_seg_dataset(
        self, team_slug_darwin_json_v2: str, team_extracted_dataset_path: Path
    ) -> None:
        root = team_extracted_dataset_path / team_slug_darwin_json_v2 / "coco"
        ds = InstanceSegmentationDataset(dataset_path=root, release_name="latest")

        generic_dataset_test(ds, n=20, size=(50, 50))
        assert isinstance(ds[0][1], dict)


class TestSemanticSegmentationDataset:
    def test_should_correctly_create_a_semantic_seg_dataset(
        self, team_slug_darwin_json_v2: str, team_extracted_dataset_path: Path
    ) -> None:
        root = team_extracted_dataset_path / team_slug_darwin_json_v2 / "coco"
        ds = SemanticSegmentationDataset(dataset_path=root, release_name="latest")

        generic_dataset_test(ds, n=20, size=(50, 50))
        assert isinstance(ds[0][1], dict)


class TestObjectDetectionDataset:
    def test_should_correctly_create_a_object_detection_dataset(
        self, team_slug_darwin_json_v2: str, team_extracted_dataset_path: Path
    ) -> None:
        root = team_extracted_dataset_path / team_slug_darwin_json_v2 / "coco"
        ds = ObjectDetectionDataset(dataset_path=root, release_name="latest")

        generic_dataset_test(ds, n=20, size=(50, 50))
        assert isinstance(ds[0][1], dict)
        img, target = ds[0]

        for bbox in target["boxes"]:
            assert bbox.shape[-1] == 4
            assert torch.all(bbox > 0)
            # check if xs are > than the width
            assert torch.all(bbox[::2] < img.shape[-1])
            # check if yts are > than the height
            assert torch.all(bbox[1::2] < img.shape[-2])


class TestGetDataset:
    def test_exits_when_dataset_not_supported(
        self, team_slug_darwin_json_v2: str, local_config_file: Config
    ) -> None:
        with patch.object(sys, "exit") as exception:
            get_dataset(f"{team_slug_darwin_json_v2}/test", "unknown")
            exception.assert_called_once_with(1)

    def test_exits_when_dataset_does_not_exist_locally(
        self, team_slug_darwin_json_v2: str, local_config_file: Config
    ) -> None:
        with patch.object(sys, "exit") as exception:
            get_dataset(f"{team_slug_darwin_json_v2}/test", "classification")
            exception.assert_called_once_with(1)

    def test_loads_classification_dataset(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(f"{team_slug_darwin_json_v2}/sl", "classification")
        assert isinstance(dataset, ClassificationDataset)
        assert len(dataset) == 20

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)
        assert label.item() == 0

    def test_loads_multi_label_classification_dataset(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(f"{team_slug_darwin_json_v2}/ml", "classification")
        assert isinstance(dataset, ClassificationDataset)
        assert len(dataset) == 20
        assert dataset.is_multi_label

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)
        assert _maybe_tensor_to_list(label) == [1, 0, 1]

    def test_loads_object_detection_dataset_from_bounding_box_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(f"{team_slug_darwin_json_v2}/bb", "object-detection")
        assert isinstance(dataset, ObjectDetectionDataset)
        assert len(dataset) == 1

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: v.numpy().tolist() for k, v in label.items()}

        assert label == {
            "boxes": [
                [4, 33, 17, 16]
            ],  # we need to account for xywh format and clamping
            "area": [612],
            "labels": [1],
            "image_id": [0],
            "iscrowd": [0],
        }

    def test_loads_object_detection_dataset_from_polygon_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(f"{team_slug_darwin_json_v2}/coco", "object-detection")
        assert isinstance(dataset, ObjectDetectionDataset)
        assert len(dataset) == 20

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: v.numpy().tolist() for k, v in label.items()}
        assert label == {
            "boxes": [
                [4, 33, 17, 16]
            ],  # we need to account for xywh format and clamping
            "area": [612],
            "labels": [1],
            "image_id": [0],
            "iscrowd": [0],
        }

    def test_loads_object_detection_dataset_from_complex_polygon_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(
            f"{team_slug_darwin_json_v2}/complex_polygons", "object-detection"
        )
        assert isinstance(dataset, ObjectDetectionDataset)
        assert len(dataset) == 1

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: v.numpy().tolist() for k, v in label.items()}
        assert label == {
            "boxes": [[1, 1, 39, 48]],
            "area": [1911],
            "labels": [1],
            "image_id": [0],
            "iscrowd": [0],
        }

    def test_loads_instance_segmentation_dataset_from_bounding_box_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        # You can load an instance segmentation dataset from an export that only has bounding boxes.
        # But it will ignore all the annotations, so you'll end up with 0 annotations.
        dataset = get_dataset(
            f"{team_slug_darwin_json_v2}/bb",
            "instance-segmentation",
        )
        assert isinstance(dataset, InstanceSegmentationDataset)
        assert len(dataset) == 1

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: _maybe_tensor_to_list(v) for k, v in label.items()}

        assert label["boxes"] == []
        assert label["area"] == []
        assert label["labels"] == []
        assert label["image_id"] == [0]
        assert label["iscrowd"] == []
        assert label["height"] == 50
        assert label["image_path"] == str(dataset.dataset_path / "images" / "0.png")
        assert label["width"] == 50

    def test_loads_instance_segmentation_dataset_from_polygon_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(
            f"{team_slug_darwin_json_v2}/coco", "instance-segmentation"
        )
        assert isinstance(dataset, InstanceSegmentationDataset)
        assert len(dataset) == 20

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: _maybe_tensor_to_list(v) for k, v in label.items()}

        assert label["boxes"] == [[4.0, 33.0, 36.0, 16.0]]
        assert label["area"] == [576.0]
        assert label["labels"] == [1]
        assert label["image_id"] == [0]
        assert label["iscrowd"] == [0]
        assert label["height"] == 50
        assert label["image_path"] == str(dataset.dataset_path / "images" / "0.png")
        assert label["width"] == 50

    def test_loads_instance_segmentation_dataset_from_complex_polygon_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(
            f"{team_slug_darwin_json_v2}/complex_polygons", "instance-segmentation"
        )
        assert isinstance(dataset, InstanceSegmentationDataset)
        assert len(dataset) == 1

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: _maybe_tensor_to_list(v) for k, v in label.items()}

        assert label["boxes"] == [[1.0, 1.0, 39.0, 48.0]]
        assert label["area"] == [592.0]
        assert label["labels"] == [1]
        assert label["image_id"] == [0]
        assert label["iscrowd"] == [0]
        assert label["height"] == 50
        assert label["image_path"] == str(dataset.dataset_path / "images" / "0.png")
        assert label["width"] == 50

    def test_loads_semantic_segmentation_dataset_from_polygon_annotations(
        self,
        team_slug_darwin_json_v2: str,
        local_config_file: Config,
        team_extracted_dataset_path: Path,
    ) -> None:
        dataset = get_dataset(
            f"{team_slug_darwin_json_v2}/coco", "semantic-segmentation"
        )
        assert isinstance(dataset, SemanticSegmentationDataset)
        assert len(dataset) == 20
        assert "__background__" in dataset.classes

        image, label = dataset[0]
        assert image.size() == (3, 50, 50)

        label = {k: _maybe_tensor_to_list(v) for k, v in label.items()}

        assert label["image_id"] == [0]
        assert isinstance(label["mask"][0], list)
        assert label["height"] == 50
        assert label["width"] == 50


def _maybe_tensor_to_list(arg: Any) -> Any:
    if isinstance(arg, torch.Tensor):
        return arg.numpy().tolist()
    return arg
