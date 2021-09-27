from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from darwin.cli_functions import _error, _load_client
from darwin.dataset import LocalDataset
from darwin.dataset.identifier import DatasetIdentifier
from darwin.torch.transforms import (
    Compose,
    ConvertPolygonsToInstanceMasks,
    ConvertPolygonsToSemanticMask,
)
from darwin.torch.utils import polygon_area
from darwin.utils import convert_polygons_to_sequences

import torch


def get_dataset(
    dataset_slug: str,
    dataset_type: str,
    partition: Optional[str] = None,
    split: str = "default",
    split_type: str = "random",
    transform: Optional[List] = None,
):
    """
    Creates and returns a dataset

    Parameters
    ----------
    dataset_slug: str
        Slug of the dataset to retrieve
    dataset_type: str
        The type of dataset [classification, instance-segmentation, semantic-segmentation]
    partition: str
        Selects one of the partitions [train, val, test, None]. (Default: None)
    split: str
        Selects the split that defines the percentages used. (Default: 'default')
    split_type: str
        Heuristic used to do the split [random, stratified]. (Default: 'random')
    transform : list[torchvision.transforms]
        List of PyTorch transforms
    """
    dataset_functions = {
        "classification": ClassificationDataset,
        "instance-segmentation": InstanceSegmentationDataset,
        "semantic-segmentation": SemanticSegmentationDataset,
    }
    dataset_function = dataset_functions.get(dataset_type)
    if not dataset_function:
        list_of_types = ", ".join(dataset_functions.keys())
        _error(f"dataset_type needs to be one of '{list_of_types}'")

    identifier = DatasetIdentifier.parse(dataset_slug)
    client = _load_client(offline=True)

    for p in client.list_local_datasets(team=identifier.team_slug):
        if identifier.dataset_slug == p.name:
            return dataset_function(
                dataset_path=p,
                partition=partition,
                split=split,
                split_type=split_type,
                release_name=identifier.version,
                transform=transform,
            )

    _error(
        f"Dataset '{identifier.dataset_slug}' does not exist locally. "
        f"Use 'darwin dataset remote' to see all the available datasets, "
        f"and 'darwin dataset pull' to pull them."
    )


class ClassificationDataset(LocalDataset):
    def __init__(self, transform: Optional[Callable | List] = None, **kwargs):
        """
        See class `LocalDataset` for documentation
        """
        super().__init__(annotation_type="tag", **kwargs)

        self.transform = transform
        if self.transform is not None and isinstance(self.transform, list):
            self.transform = Compose(self.transform)

        self.is_multi_label = False
        self.check_if_multi_label()

    def __getitem__(self, index: int):
        """
        See superclass for documentation

        Notes
        -----
        The return value is a dict with the following fields:
            image_id: int
                The index of the image in the split
            image_path: str
                The path to the image on the file system
            category_id : int
                The single label of the image selected
        """
        img = self.get_image(index)
        if self.transform is not None:
            img = self.transform(img)

        target = self.get_target(index)

        return img, target

    def get_target(self, index: int):
        """
        Returns the classification target
        """

        target = self.parse_json(index)
        annotations = target.pop("annotations")
        tags = [a["name"] for a in annotations if "tag" in a]

        if self.is_multi_label:
            target = torch.zeros(len(self.classes))
            # one hot encode all the targets
            for tag in tags:
                idx = self.classes.index(tag)
                target[idx] = 1
        else:
            target = torch.tensor([self.classes.index(tags[0])])

        return target

    def check_if_multi_label(self) -> None:
        """
        This function loops over all the .json files and check if we have more than one tags in at least one file, if yes we assume the dataset is for multi label classification.
        """
        for idx in range(len(self)):
            target = self.parse_json(idx)
            annotations = target.pop("annotations")
            tags = [a["name"] for a in annotations if "tag" in a]

            if len(tags) > 1:
                self.is_multi_label = True
                break

    def get_class_idx(self, index: int):
        target = self.get_target(index)
        return target["category_id"]

    def measure_weights(self, **kwargs) -> np.ndarray:
        """
        Computes the class balancing weights (not the frequencies!!) given the train loader
        Get the weights proportional to the inverse of their class frequencies.
        The vector sums up to 1

        Returns
        -------
        class_weights : ndarray[double]
            Weight for each class in the train set (one for each class) as a 1D array normalized
        """
        # Collect all the labels by iterating over the whole dataset
        labels = []
        for i, _filename in enumerate(self.images_path):
            target = self.get_target(i)
            labels.append(target["category_id"])
        return self._compute_weights(labels)


class InstanceSegmentationDataset(LocalDataset):
    def __init__(self, transform: Optional[List] = None, **kwargs):
        """
        See `LocalDataset` class for documentation
        """
        super().__init__(annotation_type="polygon", **kwargs)

        self.transform = transform
        if self.transform is not None and isinstance(self.transform, list):
            self.transform = Compose(self.transform)

        self.convert_polygons = ConvertPolygonsToInstanceMasks()

    def __getitem__(self, index: int):
        """
        Notes
        -----
        The return value is a dict with the following fields:
            image_id : int
                Index of the image inside the dataset
            image_path: str
                The path to the image on the file system
            labels : tensor(n)
                The class label of each one of the instances
            masks : tensor(n, H, W)
                Segmentation mask of each one of the instances
            boxes : tensor(n, 4)
                Coordinates of the bounding box enclosing the instances as [x, y, x, y]
            area : float
                Area in pixels of each one of the instances
        """
        img = self.get_image(index)
        target = self.get_target(index)

        img, target = self.convert_polygons(img, target)
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def get_target(self, index: int):
        """
        Returns the instance segmentation target
        """
        target = self.parse_json(index)

        annotations = []
        for annotation in target["annotations"]:
            if "polygon" not in annotation and "complex_polygon" not in annotation:
                print(f"Warning: missing polygon in annotation {self.annotations_path[index]}")
            # Extract the sequences of coordinates from the polygon annotation
            annotation_type: str = "polygon" if "polygon" in annotation else "complex_polygon"
            sequences = convert_polygons_to_sequences(
                annotation[annotation_type]["path"],
                height=target["height"],
                width=target["width"],
            )
            # Compute the bbox of the polygon
            x_coords = [s[0::2] for s in sequences]
            y_coords = [s[1::2] for s in sequences]
            min_x = np.min([np.min(x_coord) for x_coord in x_coords])
            min_y = np.min([np.min(y_coord) for y_coord in y_coords])
            max_x = np.max([np.max(x_coord) for x_coord in x_coords])
            max_y = np.max([np.max(y_coord) for y_coord in y_coords])
            w = max_x - min_x + 1
            h = max_y - min_y + 1
            # Compute the area of the polygon
            # TODO fix with addictive/subtractive paths in complex polygons
            poly_area = np.sum([polygon_area(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords)])

            # Create and append the new entry for this annotation
            annotations.append(
                {
                    "category_id": self.classes.index(annotation["name"]),
                    "segmentation": sequences,
                    "bbox": [min_x, min_y, w, h],
                    "area": poly_area,
                }
            )
        target["annotations"] = annotations

        return target

    def measure_weights(self, **kwargs):
        """
        Computes the class balancing weights (not the frequencies!!) given the train loader
        Get the weights proportional to the inverse of their class frequencies.
        The vector sums up to 1

        Returns
        -------
        class_weights : ndarray[double]
            Weight for each class in the train set (one for each class) as a 1D array normalized
        """
        # Collect all the labels by iterating over the whole dataset
        labels = []
        for i, _ in enumerate(self.images_path):
            target = self.get_target(i)
            labels.extend([a["category_id"] for a in target["annotations"]])
        return self._compute_weights(labels)


class SemanticSegmentationDataset(LocalDataset):
    def __init__(self, transform: Optional[List] = None, **kwargs):
        """
        See `LocalDataset` class for documentation
        """
        super().__init__(annotation_type="polygon", **kwargs)

        self.transform = transform
        if self.transform is not None and isinstance(self.transform, list):
            self.transform = Compose(self.transform)

        self.convert_polygons = ConvertPolygonsToSemanticMask()

    def __getitem__(self, index: int):
        """
        See superclass for documentation

        Notes
        -----
        The return value is a dict with the following fields:
            image_id : int
                Index of the image inside the dataset
            image_path: str
                The path to the image on the file system
            mask : tensor(H, W)
                Segmentation mask where each pixel encodes a class label
        """
        img = self.get_image(index)
        target = self.get_target(index)

        img, target = self.convert_polygons(img, target)
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def get_target(self, index: int):
        """
        Returns the semantic segmentation target
        """
        target = self.parse_json(index)

        annotations = []
        for obj in target["annotations"]:
            sequences = convert_polygons_to_sequences(
                obj["polygon"]["path"],
                height=target["height"],
                width=target["width"],
            )
            # Discard polygons with less than three points
            sequences[:] = [s for s in sequences if len(s) >= 6]
            if not sequences:
                continue
            annotations.append({"category_id": self.classes.index(obj["name"]), "segmentation": sequences})
        target["annotations"] = annotations

        return target

    def measure_weights(self, **kwargs):
        """
        Computes the class balancing weights (not the frequencies!!) given the train loader
        Get the weights proportional to the inverse of their class frequencies.
        The vector sums up to 1

        Returns
        -------
        class_weights : ndarray[double]
            Weight for each class in the train set (one for each class) as a 1D array normalized
        """
        # Collect all the labels by iterating over the whole dataset
        labels = []
        for i, _ in enumerate(self.images_path):
            target = self.get_target(i)
            labels.extend([a["category_id"] for a in target["annotations"]])
        return self._compute_weights(labels)
