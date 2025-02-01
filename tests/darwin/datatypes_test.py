import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from darwin.client import Client
from darwin.config import Config
from darwin.dataset.remote_dataset_v2 import RemoteDatasetV2
from darwin.datatypes import (
    ObjectStore,
    Point,
    make_polygon,
    parse_property_classes,
    split_paths_by_metadata,
    Annotation,
    AnnotationClass,
)


class TestMakePolygon:
    def test_it_returns_annotation_with_default_params(self):
        class_name: str = "class_name"
        points: List[Point] = [[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}]]
        annotation = make_polygon(class_name, points)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

    def test_it_returns_annotation_with_bounding_box(self):
        class_name: str = "class_name"
        points: List[Point] = [[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}]]
        bbox: Dict[str, float] = {"x": 1, "y": 2, "w": 2, "h": 2}
        annotation = make_polygon(class_name, points, bbox)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

        class_bbox = annotation.data.get("bounding_box")
        assert class_bbox == bbox


class TestMakeComplexPolygon:
    def test_it_returns_annotation_with_default_params(self):
        class_name: str = "class_name"
        points: List[List[Point]] = [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}],
            [{"x": 4, "y": 5}, {"x": 6, "y": 7}, {"x": 4, "y": 5}],
        ]
        annotation = make_polygon(class_name, points)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

    def test_it_returns_annotation_with_bounding_box(self):
        class_name: str = "class_name"
        points: List[List[Point]] = [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}],
            [{"x": 4, "y": 5}, {"x": 6, "y": 7}, {"x": 4, "y": 5}],
        ]
        bbox: Dict[str, float] = {"x": 1, "y": 2, "w": 2, "h": 2}
        annotation = make_polygon(class_name, points, bbox)

        assert_annotation_class(annotation, class_name, "polygon", "polygon")

        paths = annotation.data.get("paths")
        assert paths == points

        class_bbox = annotation.data.get("bounding_box")
        assert class_bbox == bbox


def assert_annotation_class(annotation, name, type, internal_type=None) -> None:
    assert annotation.annotation_class.name == name
    assert annotation.annotation_class.annotation_type == type
    assert annotation.annotation_class.annotation_internal_type == internal_type


@pytest.mark.parametrize(
    ("filename", "property_class_n", "properties_n"),
    (
        ("metadata.json", 1, [2]),
        ("metadata_nested_properties.json", 3, [0, 0, 1]),
        ("metadata_empty_properties.json", 3, [0, 0, 0]),
    ),
)
def test_parse_properties(filename, property_class_n, properties_n):
    metadata_path = Path(__file__).parent / f"data/{filename}"

    with open(metadata_path) as f:
        metadata = json.load(f)

    property_classes = parse_property_classes(metadata)
    assert len(property_classes) == property_class_n
    assert [
        len(property_class.properties or []) for property_class in property_classes
    ] == properties_n


@pytest.mark.parametrize(
    ("filename", "property_class_n", "properties_n", "is_properties_enabled"),
    (
        ("metadata.json", 1, [2], True),
        ("metadata_nested_properties.json", 3, [0, 0, 1], True),
        ("metadata_empty_properties.json", 0, [], False),
    ),
)
def test_split_paths_by_manifest(
    filename, property_class_n, properties_n, is_properties_enabled
):
    manifest_path = Path(__file__).parent / f"data/{filename}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmpdir_v7 = tmpdir / ".v7"
        tmpdir_v7.mkdir(exist_ok=True)
        shutil.copy(manifest_path, tmpdir_v7)

        _path, property_classes = split_paths_by_metadata(tmpdir, filename=filename)

        is_path_file = _path.is_file()
        assert is_path_file == is_properties_enabled
        assert len(property_classes or []) == property_class_n
        assert [
            len(property_class.properties or [])
            for property_class in property_classes or []
        ] == properties_n


class TestObjectStore:
    @pytest.fixture
    def object_store(self):
        return ObjectStore(
            name="test",
            prefix="test_prefix",
            readonly=False,
            provider="aws",
            default=True,
        )

    @pytest.fixture
    def darwin_client(
        darwin_config_path: Path,
        darwin_datasets_path: Path,
        team_slug_darwin_json_v2: str,
    ) -> Client:
        config = Config(darwin_config_path)
        config.put(["global", "api_endpoint"], "http://localhost/api")
        config.put(["global", "base_url"], "http://localhost")
        config.put(["teams", team_slug_darwin_json_v2, "api_key"], "mock_api_key")
        config.put(
            ["teams", team_slug_darwin_json_v2, "datasets_dir"],
            str(darwin_datasets_path),
        )
        return Client(config)

    @pytest.fixture
    def remote_dataset_v2(self):
        return RemoteDatasetV2(
            client=self.darwin_client,
            team="test_team",
            name="Test dataset",
            slug="test-dataset",
            dataset_id=1,
        )

    def test_init(self, object_store):
        assert object_store.name == "test"
        assert object_store.prefix == "test_prefix"
        assert object_store.readonly is False
        assert object_store.provider == "aws"
        assert object_store.default is True

    def test_str(self, object_store):
        assert (
            str(object_store)
            == "Storage configuration:\n- Name: test\n- Prefix: test_prefix\n- Readonly: False\n- Provider: aws\n- Default: True"
        )

    def test_repr(self, object_store):
        assert (
            repr(object_store)
            == "ObjectStore(name=test, prefix=test_prefix, readonly=False, provider=aws)"
        )


class TestAnnotation:
    def test_scale_coordinates_bounding_box(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_bb", "bounding_box"),
            data={"x": 10, "y": 20, "w": 30, "h": 40},
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {"x": 20, "y": 30, "w": 60, "h": 60}

    def test_scale_coordinates_polygon(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_polygon", "polygon"),
            data={
                "paths": [
                    [
                        {"x": 0, "y": 0},
                        {"x": 100, "y": 0},
                        {"x": 100, "y": 100},
                        {"x": 0, "y": 100},
                        {"x": 0, "y": 0},
                    ],
                    [
                        {"x": 100, "y": 0},
                        {"x": 200, "y": 0},
                        {"x": 200, "y": 100},
                        {"x": 100, "y": 100},
                        {"x": 100, "y": 0},
                    ],
                ]
            },
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {
            "paths": [
                [
                    {"x": 0, "y": 0},
                    {"x": 200, "y": 0},
                    {"x": 200, "y": 150},
                    {"x": 0, "y": 150},
                    {"x": 0, "y": 0},
                ],
                [
                    {"x": 200, "y": 0},
                    {"x": 400, "y": 0},
                    {"x": 400, "y": 150},
                    {"x": 200, "y": 150},
                    {"x": 200, "y": 0},
                ],
            ]
        }

    def test_scale_coordinates_ellipse(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_ellipse", "ellipse"),
            data={
                "center": {"x": 0, "y": 0},
                "radius": {"x": 100, "y": 50},
            },
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {
            "center": {"x": 0, "y": 0},
            "radius": {"x": 200, "y": 75},
        }

    def test_scale_coordinates_line(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_line", "line"),
            data={"path": [{"x": 0, "y": 0}, {"x": 100, "y": 100}]},
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {"path": [{"x": 0, "y": 0}, {"x": 200, "y": 150}]}

    def test_scale_coordinates_keypoint(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_keypoint", "keypoint"),
            data={"x": 50, "y": 100},
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {"x": 100, "y": 150}

    def test_scale_coordinates_skeleton(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_skeleton", "skeleton"),
            data={
                "nodes": [
                    {"x": 0, "y": 0},
                    {"x": 100, "y": 100},
                ]
            },
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {"nodes": [{"x": 0, "y": 0}, {"x": 200, "y": 150}]}

    def test_scale_coordinates_skips_raster_layer(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("__raster_layer__", "raster_layer"),
            data={"dense_rle": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {
            "dense_rle": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }

    def test_scale_coordinates_skips_tag(self):
        annotation = Annotation(
            annotation_class=AnnotationClass("test_tag", "tag"),
            data={},
        )
        annotation.scale_coordinates(x_scale=2, y_scale=1.5)
        assert annotation.data == {}
