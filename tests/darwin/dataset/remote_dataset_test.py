import shutil
import tempfile
import types
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import orjson as json
import pytest
import responses
from pydantic import ValidationError
from darwin.client import Client
from darwin.config import Config
from darwin.dataset import RemoteDataset
from darwin.dataset.download_manager import (
    _download_image_from_json_annotation,
    download_all_images_from_annotations,
)
from darwin.dataset.release import Release, ReleaseStatus
from darwin.dataset.remote_dataset_v2 import (
    RemoteDatasetV2,
    _find_files_to_upload_as_single_file_items,
    _find_files_to_upload_as_multi_file_items,
)
from darwin.dataset.upload_manager import (
    ItemMergeMode,
    LocalFile,
    UploadHandlerV2,
)
from darwin.utils.utils import SLOTS_GRID_MAP
from darwin.datatypes import ManifestItem, ObjectStore, SegmentManifest
from darwin.exceptions import UnsupportedExportFormat, UnsupportedFileType
from darwin.item import DatasetItem
from tests.fixtures import *


@pytest.fixture
def mock_is_file_extension_allowed():
    with patch("darwin.dataset.download_manager.is_file_extension_allowed") as mock:
        yield mock


@pytest.fixture
def annotation_name() -> str:
    return "test_video.json"


@pytest.fixture
def annotation_content() -> Dict[str, Any]:
    # return {
    #     "image": {
    #         "width": 1920,
    #         "height": 1080,
    #         "filename": "test_video.mp4",
    #         "fps": 20.0,
    #         "frame_urls": ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"],
    #     },
    #     "annotations": [
    #         {
    #             "frames": {
    #                 "0": {
    #                     "polygon": {
    #                         "path": [
    #                             {"x": 0, "y": 0},
    #                             {"x": 1, "y": 1},
    #                             {"x": 1, "y": 0},
    #                         ]
    #                     }
    #                 },
    #                 "2": {
    #                     "polygon": {
    #                         "path": [
    #                             {"x": 5, "y": 5},
    #                             {"x": 6, "y": 6},
    #                             {"x": 6, "y": 5},
    #                         ]
    #                     }
    #                 },
    #                 "4": {
    #                     "polygon": {
    #                         "path": [
    #                             {"x": 9, "y": 9},
    #                             {"x": 8, "y": 8},
    #                             {"x": 8, "y": 9},
    #                         ]
    #                     }
    #                 },
    #             },
    #             "name": "test_class",
    #             "segments": [[0, 3]],
    #         }
    #     ],
    # }
    return {
        "version": "2.0",
        "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
        "item": {
            "name": "test_video.mp4",
            "path": "/",
            "source_info": {
                "dataset": {
                    "name": "v7-darwin-json-v2",
                    "slug": "v7-darwin-json-v2",
                    "dataset_management_url": "test_url",
                },
                "item_id": "test_item_id",
                "team": {"name": "Test team", "slug": "test-team"},
                "workview_url": "test_url",
            },
            "slots": [
                {
                    "type": "video",
                    "slot_name": "0",
                    "width": 1920,
                    "height": 1080,
                    "fps": 20.0,
                    "thumbnail_url": "",
                    "source_files": [{"file_name": "test_video.mp4", "url": ""}],
                    "frame_count": 3,
                    "frame_urls": ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"],
                }
            ],
        },
        "annotations": [
            {
                "name": "test_class",
                "slot_names": ["0"],
                "ranges": [[0, 3]],
                "hidden_areas": [[1, 2]],
                "id": "test_id",
                "frames": {
                    "0": {
                        "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1},
                        "polygon": {
                            "paths": [
                                [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 1, "y": 0}]
                            ],
                        },
                        "keyframe": True,
                    },
                    "2": {
                        "bounding_box": {"x": 5, "y": 5, "w": 1, "h": 1},
                        "polygon": {
                            "paths": [
                                [{"x": 5, "y": 5}, {"x": 6, "y": 6}, {"x": 6, "y": 5}]
                            ],
                        },
                        "keyframe": True,
                    },
                    "4": {
                        "bounding_box": {"x": 8, "y": 8, "w": 1, "h": 1},
                        "polygon": {
                            "paths": [
                                [{"x": 9, "y": 9}, {"x": 8, "y": 8}, {"x": 8, "y": 9}]
                            ],
                        },
                        "keyframe": True,
                    },
                },
            }
        ],
    }


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
        ["teams", team_slug_darwin_json_v2, "datasets_dir"], str(darwin_datasets_path)
    )
    return Client(config)


@pytest.fixture
def create_annotation_file(
    darwin_datasets_path: Path,
    team_slug_darwin_json_v2: str,
    dataset_slug: str,
    release_name: str,
    annotation_name: str,
    annotation_content: dict,
):
    annotations: Path = (
        darwin_datasets_path
        / team_slug_darwin_json_v2
        / dataset_slug
        / "releases"
        / release_name
        / "annotations"
    )
    annotations.mkdir(exist_ok=True, parents=True)

    with (annotations / annotation_name).open("w") as f:
        op = json.dumps(annotation_content).decode("utf-8")
        f.write(op)


@pytest.fixture()
def files_content() -> Dict[str, Any]:
    return {
        "items": [
            {
                "id": "018c6826-766c-d596-44b3-46159c7c23bc",
                "name": "segment_1.mp4",
                "priority": 0,
                "status": "new",
                "path": "/",
                "tags": [],
                "cursor": "018c6826-766c-d596-44b3-46159c7c23bc",
                "layout": {"type": "simple", "version": 1, "slots": ["0"]},
                "uploads": [],
                "slots": [
                    {
                        "id": "daf0b44e-b328-4d6b-8148-e7f348cd16f5",
                        "type": "video",
                        "metadata": {
                            "height": 1920,
                            "native_fps": 30,
                            "segment_index": [
                                "#EXTM3U",
                                "#EXT-X-VERSION:3",
                                "#EXT-X-TARGETDURATION:11",
                                "#EXT-X-MEDIA-SEQUENCE:0",
                                "#EXTINF:11.500000,",
                                "data/teams/3961/partition_53/018c6826-766c-d596-44b3-46159c7c23bc/uploads/dc647e0e-917f-4586-8b51-2ebc37613884.mp4/segments/000000000.ts",
                                "#EXT-X-ENDLIST",
                                "",
                            ],
                            "width": 1080,
                        },
                        "file_name": "segment_1.mp4",
                        "fps": 0.58,
                        "slot_name": "0",
                        "total_sections": 7,
                        "sectionless": False,
                        "upload_id": "dc647e0e-917f-4586-8b51-2ebc37613884",
                        "size_bytes": 12220902,
                        "is_external": False,
                        "streamable": True,
                    }
                ],
                "inserted_at": "2023-12-14T11:46:40Z",
                "updated_at": "2023-12-14T11:46:40Z",
                "dataset_id": 611387,
                "archived": False,
                "processing_status": "complete",
                "workflow_status": "new",
                "slot_types": ["video"],
            },
            {
                "id": "018cf7e3-a43d-8d2b-cc04-375004360f51",
                "name": "hang_-_30902 (540p).mp4",
                "priority": 0,
                "status": "new",
                "path": "/",
                "tags": [],
                "cursor": "018cf7e3-a43d-8d2b-cc04-375004360f51",
                "layout": {"type": "simple", "version": 1, "slots": ["0"]},
                "uploads": [],
                "slots": [
                    {
                        "id": "8d8ebdd0-e405-4ff4-9899-ecc91f39322c",
                        "type": "video",
                        "metadata": {
                            "frames_manifests": [
                                {
                                    "total_frames": 611,
                                    "url": "https://darwin.v7labs.com/s/data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/frames_manifest.txt?token=SFMyNTY.eyJleHAiOjE3MDcwNjkxOTcsImtleV9wcmVmaXgiOiJkYXRhL3RlYW1zLzM5NjEvcGFydGl0aW9uXzUzLzAxOGNmN2UzLWE0M2QtOGQyYi1jYzA0LTM3NTAwNDM2MGY1MS8ifQ.A5lUGz5VFnzEs6NUi4vYw9mw17kqGSzu0FVoBBt1oXE",
                                    "visible_frames": 25,
                                }
                            ],
                            "height": 540,
                            "native_fps": 25,
                            "segment_index": [
                                "#EXTM3U",
                                "#EXT-X-VERSION:3",
                                "#EXT-X-TARGETDURATION:3",
                                "#EXT-X-MEDIA-SEQUENCE:0",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000000.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000001.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000002.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000003.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000004.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000005.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000006.ts",
                                "#EXTINF:3.040000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000007.ts",
                                "#EXTINF:0.120000,",
                                "data/teams/3961/partition_53/018cf7e3-a43d-8d2b-cc04-375004360f51/uploads/fc994bd0-61a7-4c1f-b9d2-0715b3c51e13.mp4/segments/000000008.ts",
                                "#EXT-X-ENDLIST",
                                "",
                            ],
                            "width": 960,
                        },
                        "file_name": "hang_-_30902 (540p).mp4",
                        "fps": 1,
                        "slot_name": "0",
                        "total_sections": 25,
                        "sectionless": True,
                        "upload_id": "fc994bd0-61a7-4c1f-b9d2-0715b3c51e13",
                        "size_bytes": 5754208,
                        "is_external": False,
                        "streamable": True,
                    }
                ],
                "inserted_at": "2024-01-11T09:39:00Z",
                "updated_at": "2024-01-25T23:05:35.454727Z",
                "dataset_id": 611387,
                "archived": False,
                "processing_status": "complete",
                "workflow_status": "new",
                "slot_types": ["video"],
            },
        ],
        "page": {
            "count": 2,
            "next": None,
            "previous": "018c6826-766c-d596-44b3-46159c7c23bc",
        },
    }


@pytest.fixture()
def unprocessed_file_content() -> Dict[str, Any]:
    return {
        "items": [
            {
                "id": "018c6826-766c-d596-44b3-46159c7c23bc",
                "name": "segment_1.mp4",
                "priority": 0,
                "status": "processing",
                "path": "/",
                "tags": [],
                "cursor": "018c6826-766c-d596-44b3-46159c7c23bc",
                "uploads": [
                    {
                        "type": "video",
                        "file_name": "segment_1.mp4",
                        "processing_status": "processing",
                        "slot_name": "0",
                        "fps": 25,
                        "upload_id": "5404aafe-6434-44cb-8bc2-4b3758466bbf",
                        "as_frames": False,
                    }
                ],
                "slots": [],
                "inserted_at": "2023-12-14T11:46:40Z",
                "updated_at": "2023-12-14T11:46:40Z",
                "dataset_id": 611387,
                "archived": False,
                "processing_status": "processing",
                "workflow_status": "new",
                "slot_types": ["video"],
            }
        ],
        "page": {
            "count": 2,
            "next": None,
            "previous": "018c6826-766c-d596-44b3-46159c7c23bc",
        },
    }


# This test was never actually running
# TODO: Fix this test
# class TestDatasetCreation:
#     def test_should_set_id_correctly_from_id(self, darwin_client: Client):
#         dataset_id = "team_slug/dataset_name:test_release"
#         dataset = darwin_client.get_remote_dataset(dataset_id)

#         assert dataset.slug == "team_slug"
#         assert dataset.name == "dataset_name"
#         assert dataset.release == "test_release"

#     def test_should_work_without_a_release(self, darwin_client: Client):
#         dataset_id = "team_slug/dataset_name"
#         dataset = darwin_client.get_remote_dataset(dataset_id)

#         assert dataset.slug == "team_slug"
#         assert dataset.name == "dataset_name"
#         assert dataset.release == None


@pytest.fixture()
def setup_zip():
    zip_path = Path("tests/darwin/data/push_test_dir.zip")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        yield Path(tmpdir)


@pytest.mark.usefixtures("file_read_write_test", "create_annotation_file")
class TestSplitVideoAnnotations:
    def test_works_on_videos(
        self,
        darwin_client: Client,
        darwin_datasets_path: Path,
        dataset_name: str,
        dataset_slug: str,
        release_name: str,
        team_slug_darwin_json_v2: str,
    ):
        remote_dataset = RemoteDatasetV2(
            client=darwin_client,
            team=team_slug_darwin_json_v2,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )

        remote_dataset.split_video_annotations()

        video_path = (
            darwin_datasets_path
            / team_slug_darwin_json_v2
            / dataset_slug
            / "releases"
            / release_name
            / "annotations"
            / "test_video"
        )
        assert video_path.exists()
        assert (video_path / "0000000.json").exists()
        assert (video_path / "0000001.json").exists()
        assert (video_path / "0000002.json").exists()
        assert not (video_path / "0000003.json").exists()

        with (video_path / "0000000.json").open() as f:
            assert json.loads(f.read()) == {
                "version": "2.0",
                "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
                "item": {
                    "name": "test_video/0000000.png",
                    "path": "/test_video",
                    "source_info": {
                        "dataset": {
                            "name": "v7-darwin-json-v2",
                            "slug": "v7-darwin-json-v2",
                        },
                        "item_id": "test_item_id",
                        "team": {
                            "name": "v7-darwin-json-v2",
                            "slug": "v7-darwin-json-v2",
                        },
                        "workview_url": "test_url",
                    },
                    "slots": [
                        {
                            "type": "video",
                            "slot_name": "0",
                            "width": 1920,
                            "height": 1080,
                            "thumbnail_url": "",
                            "source_files": [
                                {"file_name": "test_video.png", "url": ""}
                            ],
                        }
                    ],
                },
                "annotations": [
                    {
                        "id": "test_id",
                        "name": "test_class",
                        "polygon": {
                            "paths": [
                                [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 1, "y": 0}]
                            ]
                        },
                        "bounding_box": {"h": 1, "w": 1, "x": 0, "y": 0},
                    }
                ],
            }

        with (video_path / "0000001.json").open() as f:
            assert json.loads(f.read()) == {
                "version": "2.0",
                "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
                "item": {
                    "name": "test_video/0000001.png",
                    "path": "/test_video",
                    "source_info": {
                        "dataset": {
                            "name": "v7-darwin-json-v2",
                            "slug": "v7-darwin-json-v2",
                        },
                        "item_id": "test_item_id",
                        "team": {
                            "name": "v7-darwin-json-v2",
                            "slug": "v7-darwin-json-v2",
                        },
                        "workview_url": "test_url",
                    },
                    "slots": [
                        {
                            "type": "video",
                            "slot_name": "0",
                            "width": 1920,
                            "height": 1080,
                            "thumbnail_url": "",
                            "source_files": [
                                {"file_name": "test_video.png", "url": ""}
                            ],
                        }
                    ],
                },
                "annotations": [],
            }

        with (video_path / "0000002.json").open() as f:
            assert json.loads(f.read()) == {
                "version": "2.0",
                "schema_ref": "https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json/2.0/schema.json",
                "item": {
                    "name": "test_video/0000002.png",
                    "path": "/test_video",
                    "source_info": {
                        "dataset": {
                            "name": "v7-darwin-json-v2",
                            "slug": "v7-darwin-json-v2",
                        },
                        "item_id": "test_item_id",
                        "team": {
                            "name": "v7-darwin-json-v2",
                            "slug": "v7-darwin-json-v2",
                        },
                        "workview_url": "test_url",
                    },
                    "slots": [
                        {
                            "type": "video",
                            "slot_name": "0",
                            "width": 1920,
                            "height": 1080,
                            "thumbnail_url": "",
                            "source_files": [
                                {"file_name": "test_video.png", "url": ""}
                            ],
                        }
                    ],
                },
                "annotations": [
                    {
                        "id": "test_id",
                        "name": "test_class",
                        "polygon": {
                            "paths": [
                                [{"x": 5, "y": 5}, {"x": 6, "y": 6}, {"x": 6, "y": 5}]
                            ]
                        },
                        "bounding_box": {"h": 1, "w": 1, "x": 5, "y": 5},
                    }
                ],
            }


@pytest.mark.usefixtures("files_content", "file_read_write_test")
class TestFetchRemoteFiles:
    @responses.activate
    def test_works(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug_darwin_json_v2: str,
        files_content: dict,
    ):
        remote_dataset = RemoteDatasetV2(
            client=darwin_client,
            team=team_slug_darwin_json_v2,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )
        url = "http://localhost/api/v2/teams/v7-darwin-json-v2/items?page%5Bsize%5D=500&include_workflow_data=true&dataset_ids%5B%5D=1"
        responses.add(
            responses.GET,
            url,
            json=files_content,
            status=200,
        )

        actual = remote_dataset.fetch_remote_files()

        assert isinstance(actual, types.GeneratorType)

        (item_1, item_2) = list(actual)

        assert responses.assert_call_count(url, 1) is True

        assert item_1.id == "018c6826-766c-d596-44b3-46159c7c23bc"
        assert item_2.id == "018cf7e3-a43d-8d2b-cc04-375004360f51"

    @responses.activate
    def test_fetches_files_with_commas(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug_darwin_json_v2: str,
        files_content: dict,
    ):
        remote_dataset = RemoteDatasetV2(
            client=darwin_client,
            team=team_slug_darwin_json_v2,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )
        url = "http://localhost/api/v2/teams/v7-darwin-json-v2/items?item_names%5B%5D=example%2Cwith%2C+comma.mp4&page%5Bsize%5D=500&include_workflow_data=true&dataset_ids%5B%5D=1"
        responses.add(
            responses.GET,
            url,
            json=files_content,
            status=200,
        )

        filters = {"item_names": ["example,with, comma.mp4"]}

        list(remote_dataset.fetch_remote_files(filters))

        assert (
            responses.calls[0].request.params["item_names[]"]
            == "example,with, comma.mp4"
        )

    @responses.activate
    def test_returns_unprocessed_files(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug_darwin_json_v2: str,
        unprocessed_file_content: dict,
    ):
        remote_dataset = RemoteDatasetV2(
            client=darwin_client,
            team=team_slug_darwin_json_v2,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )
        url = "http://localhost/api/v2/teams/v7-darwin-json-v2/items?item_names%5B%5D=example%2Cwith%2C+comma.mp4&page%5Bsize%5D=500&include_workflow_data=true&dataset_ids%5B%5D=1"
        responses.add(
            responses.GET,
            url,
            json=unprocessed_file_content,
            status=200,
        )
        filters = {"item_names": ["example,with, comma.mp4"]}

        list(remote_dataset.fetch_remote_files(filters))

        assert (
            responses.calls[0].request.params["item_names[]"]
            == "example,with, comma.mp4"
        )


@pytest.mark.usefixtures("file_read_write_test")
class TestFetchRemoteClasses:
    def setup_method(self):
        self.mock_classes = [
            {
                "name": "class1",
                "datasets": [{"id": 1}],
                "annotation_types": ["type1"],
            },
            {
                "name": "class2",
                "datasets": [{"id": 2}],
                "annotation_types": ["type2"],
            },
            {
                "name": "raster_class",
                "datasets": [],
                "annotation_types": ["raster_layer"],
            },
        ]

    def create_remote_dataset(
        self, darwin_client, dataset_name, dataset_slug, team_slug_darwin_json_v2
    ):
        return RemoteDatasetV2(
            client=darwin_client,
            team=team_slug_darwin_json_v2,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )

    @responses.activate
    def test_fetch_remote_classes_team_wide(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug_darwin_json_v2: str,
    ):
        remote_dataset = self.create_remote_dataset(
            darwin_client, dataset_name, dataset_slug, team_slug_darwin_json_v2
        )
        with patch.object(
            remote_dataset.client,
            "fetch_remote_classes",
            return_value=self.mock_classes,
        ):
            result = remote_dataset.fetch_remote_classes(team_wide=True)
            assert len(result) == 3
            assert any(cls["name"] == "raster_class" for cls in result)

    @responses.activate
    def test_fetch_remote_classes_local_to_dataset(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug_darwin_json_v2: str,
    ):
        remote_dataset = self.create_remote_dataset(
            darwin_client, dataset_name, dataset_slug, team_slug_darwin_json_v2
        )
        with patch.object(
            remote_dataset.client,
            "fetch_remote_classes",
            return_value=self.mock_classes,
        ):
            result = remote_dataset.fetch_remote_classes(team_wide=False)
            assert len(result) == 2
            assert any(cls["name"] == "raster_class" for cls in result)


@pytest.fixture
def remote_dataset(
    darwin_client: Client,
    dataset_name: str,
    dataset_slug: str,
    team_slug_darwin_json_v2: str,
):
    return RemoteDatasetV2(
        client=darwin_client,
        team=team_slug_darwin_json_v2,
        name=dataset_name,
        slug=dataset_slug,
        dataset_id=1,
    )


@pytest.mark.usefixtures("file_read_write_test")
class TestPush:
    def test_raises_if_files_are_not_provided(self, remote_dataset: RemoteDataset):
        with pytest.raises(ValueError):
            remote_dataset.push(None)

    def test_raises_if_both_path_and_local_files_are_given(
        self, remote_dataset: RemoteDataset
    ):
        with pytest.raises(ValueError):
            remote_dataset.push([LocalFile("test.jpg")], path="test")

    def test_raises_if_both_fps_and_local_files_are_given(
        self, remote_dataset: RemoteDataset
    ):
        with pytest.raises(ValueError):
            remote_dataset.push([LocalFile("test.jpg")], fps=2)

    def test_raises_if_both_as_frames_and_local_files_are_given(
        self, remote_dataset: RemoteDataset
    ):
        with pytest.raises(ValueError):
            remote_dataset.push([LocalFile("test.jpg")], as_frames=True)

    def test_works_with_local_files_list(self, remote_dataset: RemoteDataset):
        with patch.object(remote_dataset, "fetch_remote_files", return_value=[]):
            assert_upload_mocks_are_correctly_called(
                remote_dataset, [LocalFile("test.jpg")]
            )

    def test_works_with_path_list(self, remote_dataset: RemoteDataset):
        with patch.object(remote_dataset, "fetch_remote_files", return_value=[]):
            assert_upload_mocks_are_correctly_called(remote_dataset, [Path("test.jpg")])

    def test_works_with_str_list(self, remote_dataset: RemoteDataset):
        with patch.object(remote_dataset, "fetch_remote_files", return_value=[]):
            assert_upload_mocks_are_correctly_called(remote_dataset, ["test.jpg"])

    def test_works_with_supported_files(self, remote_dataset: RemoteDataset):
        supported_extensions = [
            ".png",
            ".jpeg",
            ".jpg",
            ".jfif",
            ".tif",
            ".tiff",
            ".bmp",
            ".svs",
            ".avi",
            ".bpm",
            ".dcm",
            ".mov",
            ".mp4",
            ".pdf",
            ".ndpi",
        ]
        filenames = [f"test{extension}" for extension in supported_extensions]
        with patch.object(remote_dataset, "fetch_remote_files", return_value=[]):
            assert_upload_mocks_are_correctly_called(remote_dataset, filenames)

    def test_raises_with_unsupported_files(self, remote_dataset: RemoteDataset):
        with pytest.raises(UnsupportedFileType):
            remote_dataset.push(["test.txt"])

    def test_raises_if_invalid_item_merge_mode(self, remote_dataset: RemoteDataset):
        with pytest.raises(ValueError):
            remote_dataset.push(["path/to/dir"], item_merge_mode="invalid")

    def test_raises_if_incompatible_args_with_item_merge_mode(
        self, remote_dataset: RemoteDataset
    ):
        incompatible_args = [
            {"preserve_folders": True},
            {"as_frames": True},
            {"extract_views": True},
        ]
        for args in incompatible_args:
            with pytest.raises(TypeError):
                remote_dataset.push(
                    ["path/to/dir"],
                    item_merge_mode="slots",
                    **args,  # type: ignore
                )

    @pytest.mark.usefixtures("setup_zip")
    def test_files_in_root_push_directoies_are_given_correct_remote_path(
        self, setup_zip
    ):
        directory = setup_zip / "push_test_dir" / "topdir" / "subdir_1"
        local_files = _find_files_to_upload_as_single_file_items(
            [directory],
            [],
            [],
            path="",
            fps=1,
            as_frames=False,
            extract_views=False,
            preserve_folders=True,
        )
        assert local_files[0].data["path"] == "/"
        assert local_files[1].data["path"] == "/"


@pytest.mark.usefixtures("setup_zip")
class TestPushMultiSlotItem:
    def test_different_numbers_of_input_files(self, setup_zip):
        base_path = setup_zip / "push_test_dir" / "num_files_tests"
        directories = [d for d in base_path.iterdir() if d.is_dir()]
        for directory in directories:
            if directory.name == "0":
                with pytest.raises(
                    ValueError,
                    match="No valid folders to upload after searching the passed directories for files",
                ):
                    _find_files_to_upload_as_multi_file_items(
                        [directory], [], 0, item_merge_mode="slots"
                    )
                continue

            local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
                [directory], [], 0, item_merge_mode="slots"
            )
            num_local_files = len(local_files)
            expected_num_files = int(directory.name)
            num_viewports = min(num_local_files, 16)
            assert len(multi_file_items) == 1
            assert num_local_files == expected_num_files
            assert multi_file_items[0].merge_mode == ItemMergeMode.SLOTS
            assert multi_file_items[0].files == local_files
            assert multi_file_items[0].directory == directory
            assert multi_file_items[0].name == directory.name
            assert multi_file_items[0].slot_names == [
                str(i) for i in range(num_local_files)
            ]
            assert multi_file_items[0].layout == {
                "version": 3,
                "slots_grid": SLOTS_GRID_MAP.get(num_viewports),
            }

    def test_does_not_recursively_search(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "topdir"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="slots"
        )
        num_local_files = len(local_files)
        num_viewports = min(num_local_files, 16)
        assert len(multi_file_items) == 1
        assert len(local_files) == 2
        assert multi_file_items[0].merge_mode == ItemMergeMode.SLOTS
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout == {
            "version": 3,
            "slots_grid": SLOTS_GRID_MAP.get(num_viewports),
        }

    def test_dicoms(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "dicom_tests" / "dicoms"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="slots"
        )
        num_local_files = len(local_files)
        num_viewports = min(num_local_files, 16)
        assert len(multi_file_items) == 1
        assert len(local_files) == 5
        assert multi_file_items[0].merge_mode == ItemMergeMode.SLOTS
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout == {
            "version": 3,
            "slots_grid": SLOTS_GRID_MAP.get(num_viewports),
        }

    def test_dicoms_and_other_files(self, setup_zip):
        directory = (
            setup_zip / "push_test_dir" / "dicom_tests" / "dicoms_and_other_files"
        )
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="slots"
        )
        num_local_files = len(local_files)
        num_viewports = min(num_local_files, 16)
        assert len(multi_file_items) == 1
        assert len(local_files) == 10
        assert multi_file_items[0].merge_mode == ItemMergeMode.SLOTS
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout == {
            "version": 3,
            "slots_grid": SLOTS_GRID_MAP.get(num_viewports),
        }

    def test_multiple_file_types(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "multiple_file_types"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="slots"
        )
        num_local_files = len(local_files)
        num_viewports = min(num_local_files, 16)
        assert len(multi_file_items) == 1
        assert len(local_files) == 12
        assert multi_file_items[0].merge_mode == ItemMergeMode.SLOTS
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout == {
            "version": 3,
            "slots_grid": SLOTS_GRID_MAP.get(num_viewports),
        }


@pytest.mark.usefixtures("setup_zip")
class TestPushDICOMSeries:
    def test_dicoms(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "dicom_tests" / "dicoms"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="series"
        )
        assert len(multi_file_items) == 1
        assert len(local_files) == 1
        assert multi_file_items[0].merge_mode == ItemMergeMode.SERIES
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].slot_names == ["0"]
        assert multi_file_items[0].layout is None

    def test_dicoms_and_other_files(self, setup_zip):
        directory = (
            setup_zip / "push_test_dir" / "dicom_tests" / "dicoms_and_other_files"
        )
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="series"
        )
        assert len(multi_file_items) == 1
        assert len(local_files) == 1
        assert multi_file_items[0].merge_mode == ItemMergeMode.SERIES
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout is None
        assert multi_file_items[0].slot_names == ["0"]

    def test_multiple_file_types(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "multiple_file_types"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="series"
        )
        assert len(multi_file_items) == 1
        assert len(local_files) == 1
        assert multi_file_items[0].merge_mode == ItemMergeMode.SERIES
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout is None
        assert multi_file_items[0].slot_names == ["0"]


@pytest.mark.usefixtures("setup_zip")
class TestPushMultiChannelItem:
    def test_different_numbers_of_input_files(self, setup_zip):
        base_path = setup_zip / "push_test_dir" / "num_files_tests"
        directories = [d for d in base_path.iterdir() if d.is_dir()]
        for directory in directories:
            if directory.name == "0":
                with pytest.raises(
                    ValueError,
                    match="No valid folders to upload after searching the passed directories for files",
                ):
                    _find_files_to_upload_as_multi_file_items(
                        [directory], [], 0, item_merge_mode="channels"
                    )
                continue

            if directory.name == "17":
                with pytest.raises(
                    ValueError,
                    match="No multi-channel item can have more than 16 files. The following directory has 17 files: ",
                ):
                    _find_files_to_upload_as_multi_file_items(
                        [directory], [], 0, item_merge_mode="channels"
                    )
                continue

            local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
                [directory], [], 0, item_merge_mode="channels"
            )
            num_local_files = len(local_files)
            expected_num_files = int(directory.name)
            assert len(multi_file_items) == 1
            assert num_local_files == expected_num_files
            assert multi_file_items[0].merge_mode == ItemMergeMode.CHANNELS
            assert multi_file_items[0].files == local_files
            assert multi_file_items[0].directory == directory
            assert multi_file_items[0].name == directory.name
            assert multi_file_items[0].layout == {
                "version": 3,
                "slots_grid": [[[file.local_path.name for file in local_files]]],
            }

    def test_does_not_recursively_search(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "topdir"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="channels"
        )
        assert len(multi_file_items) == 1
        assert len(local_files) == 2
        assert multi_file_items[0].merge_mode == ItemMergeMode.CHANNELS
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout == {
            "version": 3,
            "slots_grid": [[[file.local_path.name for file in local_files]]],
        }

    def test_multiple_file_types(self, setup_zip):
        directory = setup_zip / "push_test_dir" / "multiple_file_types"
        local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
            [directory], [], 0, item_merge_mode="channels"
        )
        assert len(multi_file_items) == 1
        assert len(local_files) == 5
        assert multi_file_items[0].merge_mode == ItemMergeMode.CHANNELS
        assert multi_file_items[0].files == local_files
        assert multi_file_items[0].directory == directory
        assert multi_file_items[0].name == directory.name
        assert multi_file_items[0].layout == {
            "version": 3,
            "slots_grid": [[[file.local_path.name for file in local_files]]],
        }


@pytest.mark.usefixtures("file_read_write_test")
class TestPull:
    @patch("platform.system", return_value="Linux")
    def test_gets_latest_release_when_not_given_one(
        self, system_mock: MagicMock, remote_dataset: RemoteDataset
    ):
        stub_release_response = Release(
            "dataset-slug",
            "team-slug",
            "0.1.0",
            "release-name",
            ReleaseStatus("complete"),
            "http://darwin-fake-url.com",
            datetime.now(),
            None,
            None,
            True,
            True,
            "json",
        )

        def fake_download_zip(self, path):
            zip: Path = Path("tests/dataset.zip")
            shutil.copy(zip, path)
            return path

        with patch.object(
            RemoteDataset, "get_release", return_value=stub_release_response
        ) as get_release_stub:
            with patch.object(Release, "download_zip", new=fake_download_zip):
                remote_dataset.pull(only_annotations=True)
                get_release_stub.assert_called_once()

    @patch("platform.system", return_value="Windows")
    def test_does_not_create_symlink_on_windows(
        self, mocker: MagicMock, remote_dataset: RemoteDataset
    ):
        stub_release_response = Release(
            "dataset-slug",
            "team-slug",
            "0.1.0",
            "release-name",
            ReleaseStatus("complete"),
            "http://darwin-fake-url.com",
            datetime.now(),
            None,
            None,
            True,
            True,
            "json",
        )

        def fake_download_zip(self, path):
            zip: Path = Path("tests/dataset.zip")
            shutil.copy(zip, path)
            return path

        latest: Path = remote_dataset.local_releases_path / "latest"

        with patch.object(
            RemoteDataset, "get_release", return_value=stub_release_response
        ):
            with patch.object(Release, "download_zip", new=fake_download_zip):
                remote_dataset.pull(only_annotations=True)
                assert not latest.is_symlink()

    @patch("platform.system", return_value="Linux")
    def test_continues_if_symlink_creation_fails(
        self, system_mock: MagicMock, remote_dataset: RemoteDataset
    ):
        stub_release_response = Release(
            "dataset-slug",
            "team-slug",
            "0.1.0",
            "release-name",
            ReleaseStatus("complete"),
            "http://darwin-fake-url.com",
            datetime.now(),
            None,
            None,
            True,
            True,
            "json",
        )

        def fake_download_zip(self, path):
            zip: Path = Path("tests/dataset.zip")
            shutil.copy(zip, path)
            return path

        latest: Path = remote_dataset.local_releases_path / "latest"

        with patch.object(Path, "symlink_to") as mock_symlink_to:
            with patch.object(
                RemoteDataset, "get_release", return_value=stub_release_response
            ):
                with patch.object(Release, "download_zip", new=fake_download_zip):
                    mock_symlink_to.side_effect = OSError()
                    remote_dataset.pull(only_annotations=True)
                    assert not latest.is_symlink()

    @patch("platform.system", return_value="Linux")
    def test_raises_if_release_format_is_not_json(
        self, system_mock: MagicMock, remote_dataset: RemoteDataset
    ):
        a_release = Release(
            remote_dataset.slug,
            remote_dataset.team,
            "0.1.0",
            "release-name",
            ReleaseStatus("complete"),
            "http://darwin-fake-url.com",
            datetime.now(),
            None,
            None,
            True,
            True,
            "xml",
        )

        with pytest.raises(UnsupportedExportFormat):
            remote_dataset.pull(release=a_release)

    @patch("platform.system", return_value="Linux")
    def test_moves_properties_metadata_file(
        self, system_mock: MagicMock, remote_dataset: RemoteDataset
    ):
        stub_release_response = Release(
            "dataset-slug",
            "team-slug",
            "0.1.0",
            "release-name",
            ReleaseStatus("complete"),
            "http://darwin-fake-url.com",
            datetime.now(),
            None,
            None,
            True,
            True,
            "json",
        )

        def fake_download_zip(self, path):
            zip: Path = Path("tests/dataset_with_properties.zip")
            shutil.copy(zip, path)
            return path

        with patch.object(
            RemoteDataset, "get_release", return_value=stub_release_response
        ):
            with patch.object(Release, "download_zip", new=fake_download_zip):
                remote_dataset.pull(only_annotations=True)
                metadata_path = (
                    remote_dataset.local_path
                    / "releases"
                    / "latest"
                    / "annotations"
                    / ".v7"
                    / "metadata.json"
                )
                assert metadata_path.exists()

    @patch("time.sleep", return_value=None)
    def test_num_retries(self, mock_sleep, remote_dataset, pending_release):
        with patch.object(remote_dataset, "get_release", return_value=pending_release):
            with pytest.raises(ValueError):
                remote_dataset.pull(release=pending_release, retry=True)
            assert (
                mock_sleep.call_count == 60
            )  # Default values of 600 seconds / 10 seconds interval

    @patch("time.sleep", return_value=None)
    def test_raises_after_max_retry_duration(
        self, mock_sleep, remote_dataset, pending_release
    ):
        with patch.object(remote_dataset, "get_release", return_value=pending_release):
            with pytest.raises(ValueError, match="is still processing"):
                remote_dataset.pull(release=pending_release, retry=True)

    def test_raises_error_if_timeout_less_than_interval(self, remote_dataset):
        with pytest.raises(ValueError):
            remote_dataset.pull(retry=True, retry_timeout=5, retry_interval=10)


class TestPullNamingConvention:
    @pytest.mark.usefixtures("file_read_write_test")
    def _test_pull_naming_convention(
        self,
        file_name,
        use_folders,
        video_frames,
        force_slots,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile("tests/data.zip") as zfile:
                zfile.extractall(temp_dir)
                file_path = (
                    Path(temp_dir) / "v7-darwin-json-v2/pull_naming_tests" / file_name
                )
                download_func = _download_image_from_json_annotation(
                    client=darwin_client,
                    annotation_path=file_path,
                    image_path=Path("dataset_dir_path"),
                    use_folders=use_folders,
                    video_frames=video_frames,
                    force_slots=force_slots,
                    ignore_slots=False,
                )
                return download_func

    def test_single_slotted_image_flat_structure(self):
        file_name = "single_slotted_image_flat.json"
        expected_paths = [Path("dataset_dir_path/single_slotted_image_flat.png")]
        download_funcs = self._test_pull_naming_convention(
            file_name,
            use_folders=False,
            video_frames=False,
            force_slots=False,
        )
        assert download_funcs[0].args[2] == expected_paths[0]

    def test_single_slotted_video_flat_structure(self):
        file_name = "single_slotted_video_flat.json"
        expected_paths = [Path("dataset_dir_path/single_slotted_video_flat.mp4")]
        download_funcs = self._test_pull_naming_convention(
            file_name,
            use_folders=False,
            video_frames=False,
            force_slots=False,
        )
        assert download_funcs[0].args[2] == expected_paths[0]

    def test_single_slotted_image_folder_structure(self):
        file_name = "single_slotted_image_folder.json"
        expected_paths = [Path("dataset_dir_path/dir1/single_slotted_image_folder.png")]
        download_funcs = self._test_pull_naming_convention(
            file_name,
            use_folders=True,
            video_frames=False,
            force_slots=False,
        )
        assert download_funcs[0].args[2] == expected_paths[0]

    def test_single_slotted_video_folder_structure(self):
        file_name = "single_slotted_video_folder.json"
        expected_paths = [Path("dataset_dir_path/dir1/single_slotted_video_folder.mp4")]
        download_funcs = self._test_pull_naming_convention(
            file_name,
            use_folders=True,
            video_frames=False,
            force_slots=False,
        )
        assert download_funcs[0].args[2] == expected_paths[0]

    def test_multi_slotted_item_flat_structure(self):
        file_name = "multi_slotted_item_flat.json"
        expected_paths = [
            Path("dataset_dir_path/multi_slotted_item_flat/0/000000580654.jpg"),
            Path("dataset_dir_path/multi_slotted_item_flat/1/000000580676.jpg"),
            Path("dataset_dir_path/multi_slotted_item_flat/2/000000580703.jpg"),
        ]
        download_funcs = self._test_pull_naming_convention(
            file_name,
            use_folders=False,
            video_frames=False,
            force_slots=True,
        )
        assert download_funcs[0].args[2] == expected_paths[0]
        assert download_funcs[1].args[2] == expected_paths[1]
        assert download_funcs[2].args[2] == expected_paths[2]

    def test_multi_slotted_item_folder_structure(self):
        file_name = "multi_slotted_item_folder.json"
        expected_paths = [
            Path("dataset_dir_path/dir1/multi_slotted_item_folder/0/000000580654.jpg"),
            Path("dataset_dir_path/dir1/multi_slotted_item_folder/1/000000580676.jpg"),
            Path("dataset_dir_path/dir1/multi_slotted_item_folder/2/000000580703.jpg"),
        ]
        downloads_funcs = self._test_pull_naming_convention(
            file_name,
            use_folders=True,
            video_frames=False,
            force_slots=True,
        )
        assert downloads_funcs[0].args[2] == expected_paths[0]
        assert downloads_funcs[1].args[2] == expected_paths[1]
        assert downloads_funcs[2].args[2] == expected_paths[2]

    def test_single_slotted_video_item_with_frames(self):
        file_name = "single_slotted_video_frames.json"
        base_path = "dataset_dir_path/single_slotted_video_frames/"
        expected_paths = [Path(base_path + f"{i:07d}.png") for i in range(8)]
        download_funcs = self._test_pull_naming_convention(
            file_name, use_folders=False, video_frames=True, force_slots=False
        )
        for i, func in enumerate(download_funcs):
            assert func.args[1] == expected_paths[i]

    def test_multi_slotted_video_item_with_frames(self):
        file_name = "multi_slotted_video_frames.json"
        base_path = "dataset_dir_path/multi_slotted_video_frames/"
        download_funcs = self._test_pull_naming_convention(
            file_name, use_folders=False, video_frames=True, force_slots=True
        )

        for i in range(30):
            slot_name = i // 10
            frame_index = i % 10
            expected_path = Path(f"{base_path}{slot_name}/{frame_index:07d}.png")
            assert download_funcs[i].args[1] == expected_path

    def test_single_slotted_item_multiple_source_files(self):
        file_name = "single_slot_multiple_source_files.json"
        expected_paths = [
            Path("dataset_dir_path/single_slot_multiple_source_files/0/slice_1.dcm"),
            Path("dataset_dir_path/single_slot_multiple_source_files/0/slice_2.dcm"),
        ]
        download_funcs = self._test_pull_naming_convention(
            file_name, use_folders=False, video_frames=False, force_slots=True
        )
        assert download_funcs[0].args[2] == expected_paths[0]
        assert download_funcs[1].args[2] == expected_paths[1]

    def test_single_slotted_long_video(self):
        file_name = "single_slotted_long_video.json"
        expected_paths = [Path("dataset_dir_path/single_slotted_long_video.mp4")]
        download_funcs = self._test_pull_naming_convention(
            file_name, use_folders=False, video_frames=False, force_slots=False
        )
        assert download_funcs[0].args[2] == expected_paths[0]

    def test_single_slotted_long_video_with_frames(self):
        file_name = "single_slotted_long_video_frames.json"
        expected_paths = [
            Path("dataset_dir_path/single_slotted_long_video_frames/.0000000.ts")
        ]

        with patch(
            "darwin.dataset.download_manager.get_segment_manifests"
        ) as mock_get_segment_manifests:
            mock_get_segment_manifests.return_value = [
                SegmentManifest(
                    slot="0",
                    segment=0,
                    total_frames=5,
                    items=[
                        ManifestItem(
                            frame=0,
                            absolute_frame=0,
                            segment=0,
                            visibility=True,
                            timestamp=0,
                            visible_frame=0,
                        ),
                        ManifestItem(
                            frame=1,
                            absolute_frame=1,
                            segment=0,
                            visibility=True,
                            timestamp=0.04,
                            visible_frame=1,
                        ),
                        ManifestItem(
                            frame=2,
                            absolute_frame=2,
                            segment=0,
                            visibility=True,
                            timestamp=0.08,
                            visible_frame=2,
                        ),
                        ManifestItem(
                            frame=3,
                            absolute_frame=3,
                            segment=0,
                            visibility=True,
                            timestamp=0.12,
                            visible_frame=3,
                        ),
                        ManifestItem(
                            frame=4,
                            absolute_frame=4,
                            segment=0,
                            visibility=True,
                            timestamp=0.16,
                            visible_frame=4,
                        ),
                    ],
                )
            ]

            download_funcs = self._test_pull_naming_convention(
                file_name, use_folders=False, video_frames=True, force_slots=False
            )
            assert download_funcs[0].args[2] == expected_paths[0]

    def test_multi_slotted_item_multiple_source_files(self):
        file_name = "multiple_slots_multiple_source_files.json"
        expected_paths = [
            Path(
                "dataset_dir_path/multiple_slots_multiple_source_files/0.1/slice_1.dcm"
            ),
            Path(
                "dataset_dir_path/multiple_slots_multiple_source_files/0.1/slice_2.dcm"
            ),
            Path(
                "dataset_dir_path/multiple_slots_multiple_source_files/1.1/slice_3.dcm"
            ),
            Path(
                "dataset_dir_path/multiple_slots_multiple_source_files/1.1/slice_1.dcm"
            ),
            Path(
                "dataset_dir_path/multiple_slots_multiple_source_files/1.1/slice_2.dcm"
            ),
        ]
        download_funcs = self._test_pull_naming_convention(
            file_name, use_folders=False, video_frames=False, force_slots=True
        )
        assert download_funcs[0].args[2] == expected_paths[0]
        assert download_funcs[1].args[2] == expected_paths[1]
        assert download_funcs[2].args[2] == expected_paths[2]
        assert download_funcs[3].args[2] == expected_paths[3]
        assert download_funcs[4].args[2] == expected_paths[4]

    def test_single_slotted_nifti(self):
        file_name = "single_slotted_nifti.json"
        expected_paths = [Path("dataset_dir_path/single_slotted_nifti.nii.gz")]
        download_funcs = self._test_pull_naming_convention(
            file_name, use_folders=False, video_frames=False, force_slots=False
        )
        assert download_funcs[0].args[2] == expected_paths[0]


@pytest.fixture
def dataset_item(dataset_slug: str) -> DatasetItem:
    return DatasetItem(
        id=1,
        filename="test.jpg",
        status="new",
        archived=False,
        filesize=1,
        dataset_id=1,
        dataset_slug=dataset_slug,
        seq=1,
        current_workflow_id=None,
        current_workflow=None,
        path="/",
        slots=[],
        layout={"type": "grid", "version": 3, "slots": ["0", "1"]},
    )


@pytest.mark.usefixtures("file_read_write_test")
class TestArchive:
    def test_calls_put(
        self,
        remote_dataset: RemoteDatasetV2,
        dataset_item: DatasetItem,
        team_slug_darwin_json_v2: str,
        dataset_slug: str,
    ):
        with patch.object(RemoteDatasetV2, "archive", return_value={}) as stub:
            remote_dataset.archive([dataset_item])
            stub.assert_called_once_with([dataset_item])


@pytest.mark.usefixtures("file_read_write_test")
class TestMoveToNew:
    def test_calls_put(
        self,
        remote_dataset: RemoteDatasetV2,
        dataset_item: DatasetItem,
        team_slug_darwin_json_v2: str,
        dataset_slug: str,
    ):
        with patch.object(RemoteDatasetV2, "move_to_new", return_value={}) as stub:
            remote_dataset.move_to_new([dataset_item])
            stub.assert_called_once_with([dataset_item])


@pytest.mark.usefixtures("file_read_write_test")
class TestRestoreArchived:
    def test_calls_put(
        self,
        remote_dataset: RemoteDatasetV2,
        dataset_item: DatasetItem,
        team_slug_darwin_json_v2: str,
        dataset_slug: str,
    ):
        with patch.object(RemoteDatasetV2, "restore_archived", return_value={}) as stub:
            remote_dataset.restore_archived([dataset_item])
            stub.assert_called_once_with([dataset_item])


@pytest.mark.usefixtures("file_read_write_test")
class TestDeleteItems:
    def test_calls_delete(
        self,
        remote_dataset: RemoteDatasetV2,
        dataset_item: DatasetItem,
        team_slug_darwin_json_v2: str,
        dataset_slug: str,
    ):
        with patch.object(RemoteDatasetV2, "delete_items", return_value={}) as stub:
            remote_dataset.delete_items([dataset_item])
            stub.assert_called_once_with([dataset_item])


def assert_upload_mocks_are_correctly_called(remote_dataset: RemoteDataset, *args):
    with patch.object(
        UploadHandlerV2, "_request_upload", return_value=([], [])
    ) as request_upload_mock:
        with patch.object(UploadHandlerV2, "upload") as upload_mock:
            remote_dataset.push(*args)

            request_upload_mock.assert_called_once()
            upload_mock.assert_called_once_with(
                multi_threaded=True,
                progress_callback=None,
                file_upload_callback=None,
                max_workers=None,
            )


@pytest.mark.usefixtures("file_read_write_test")
class TestExportDataset:
    def test_honours_include_authorship(self, remote_dataset: RemoteDatasetV2):
        with patch.object(RemoteDatasetV2, "export", return_value={}) as stub:
            remote_dataset.export(
                "example",
                annotation_class_ids=[],
                include_url_token=False,
                include_authorship=True,
            )
            stub.assert_called_once_with(
                "example",
                annotation_class_ids=[],
                include_url_token=False,
                include_authorship=True,
            )


@pytest.mark.usefixtures("file_read_write_test")
class TestRegister:
    def test_raises_if_storage_keys_not_list_of_strings(
        self, remote_dataset: RemoteDatasetV2
    ):
        with pytest.raises(ValueError):
            remote_dataset.register(
                ObjectStore(
                    name="test",
                    prefix="test_prefix",
                    readonly=False,
                    provider="aws",
                    default=True,
                ),
                [1, 2, 3],
            )

    def test_raises_if_unsupported_file_type(self, remote_dataset: RemoteDatasetV2):
        with pytest.raises(TypeError):
            remote_dataset.register(
                ObjectStore(
                    name="test",
                    prefix="test_prefix",
                    readonly=False,
                    provider="aws",
                    default=True,
                ),
                ["unsupported_file.xyz"],
            )

    @responses.activate
    def test_register_files(self, remote_dataset: RemoteDatasetV2):
        responses.add(
            responses.POST,
            "http://localhost/api/v2/teams/v7-darwin-json-v2/items/register_existing",
            json={
                "items": [{"id": "1", "name": "test.jpg"}],
                "blocked_items": [],
            },
            status=200,
        )
        result = remote_dataset.register(
            ObjectStore(
                name="test",
                prefix="test_prefix",
                readonly=False,
                provider="aws",
                default=True,
            ),
            ["test.jpg"],
        )
        assert len(result["registered"]) == 1
        assert len(result["blocked"]) == 0

    @responses.activate
    def test_register_files_with_blocked_items(self, remote_dataset: RemoteDatasetV2):
        responses.add(
            responses.POST,
            "http://localhost/api/v2/teams/v7-darwin-json-v2/items/register_existing",
            json={
                "items": [],
                "blocked_items": [
                    {"name": "test.jpg", "slots": [{"reason": "test reason"}]}
                ],
            },
            status=200,
        )
        result = remote_dataset.register(
            ObjectStore(
                name="test",
                prefix="test_prefix",
                readonly=False,
                provider="aws",
                default=True,
            ),
            ["test.jpg"],
        )
        assert len(result["registered"]) == 0
        assert len(result["blocked"]) == 1


@pytest.mark.usefixtures("file_read_write_test")
class TestRegisterMultiSlotted:
    def test_raises_if_storage_keys_not_dictionary(
        self, remote_dataset: RemoteDatasetV2
    ):
        with pytest.raises(ValidationError):
            remote_dataset.register(
                ObjectStore(
                    name="test",
                    prefix="test_prefix",
                    readonly=False,
                    provider="aws",
                    default=True,
                ),
                {"item1": [1, 2, 3]},
                multi_slotted=True,
            )

    def test_raises_if_unsupported_file_type(self, remote_dataset: RemoteDatasetV2):
        with pytest.raises(TypeError):
            remote_dataset.register(
                ObjectStore(
                    name="test",
                    prefix="test_prefix",
                    readonly=False,
                    provider="aws",
                    default=True,
                ),
                {"item1": ["unsupported_file.xyz"]},
                multi_slotted=True,
            )

    @responses.activate
    def test_register_files(self, remote_dataset: RemoteDatasetV2):
        responses.add(
            responses.POST,
            "http://localhost/api/v2/teams/v7-darwin-json-v2/items/register_existing",
            json={
                "items": [{"id": "1", "name": "test.jpg"}],
                "blocked_items": [],
            },
            status=200,
        )
        result = remote_dataset.register(
            ObjectStore(
                name="test",
                prefix="test_prefix",
                readonly=False,
                provider="aws",
                default=True,
            ),
            {"item1": ["test.jpg"]},
            multi_slotted=True,
        )
        assert len(result["registered"]) == 1
        assert len(result["blocked"]) == 0

    @responses.activate
    def test_register_files_with_blocked_items(self, remote_dataset: RemoteDatasetV2):
        responses.add(
            responses.POST,
            "http://localhost/api/v2/teams/v7-darwin-json-v2/items/register_existing",
            json={
                "items": [],
                "blocked_items": [
                    {"name": "test.jpg", "slots": [{"reason": "test reason"}]}
                ],
            },
            status=200,
        )
        result = remote_dataset.register(
            ObjectStore(
                name="test",
                prefix="test_prefix",
                readonly=False,
                provider="aws",
                default=True,
            ),
            {"item1": ["test.jpg"]},
            multi_slotted=True,
        )
        assert len(result["registered"]) == 0
        assert len(result["blocked"]) == 1


@pytest.mark.usefixtures("file_read_write_test")
class TestGetReleases:
    @patch("darwin.backend_v2.BackendV2.get_exports")
    def test_returns_unavailable_releases_when_retry_is_true(
        self, mock_get_exports, remote_dataset, releases_api_response
    ):
        mock_get_exports.return_value = releases_api_response
        releases = remote_dataset.get_releases(include_unavailable=True)
        assert len(releases) == 2
        assert isinstance(releases[0], Release)
        assert isinstance(releases[1], Release)

    @patch("darwin.backend_v2.BackendV2.get_exports")
    def test_omits_unavailable_releases_when_retry_is_false(
        self, mock_get_exports, remote_dataset, releases_api_response
    ):
        mock_get_exports.return_value = releases_api_response
        releases = remote_dataset.get_releases(include_unavailable=False)
        assert len(releases) == 1
        assert isinstance(releases[0], Release)


@pytest.mark.usefixtures("file_read_write_test")
def test_force_slots_true(
    mock_is_file_extension_allowed, darwin_client, file_read_write_test
):
    mock_is_file_extension_allowed.return_value = True
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmp_dir)
            annotations_path = Path(tmp_dir) / "v7-darwin-json-v2/force_slots"
            images_path = Path("images")
            generator, count = download_all_images_from_annotations(
                annotations_path=annotations_path,
                client=darwin_client,
                images_path=images_path,
                force_slots=True,
                force_replace=True,
            )

    download_funcs = list(generator())
    planned_paths = [download_funcs[i].args[2] for i in range(count)]
    expected_paths = [
        Path("images/dir1/single_slotted_video_flat/0/mini_uct.mp4"),
        Path("images/dir1/multiple_slots_multiple_source_files/0.1/slice_1.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/0.1/slice_2.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/1.1/slice_1.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/1.1/slice_2.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/1.1/slice_3.dcm"),
        Path("images/dir1/single_slotted_image_flat/0/001.png"),
        Path("images/dir1/multi_slotted_item_flat/0/000000580654.jpg"),
        Path("images/dir1/multi_slotted_item_flat/1/000000580676.jpg"),
        Path("images/dir1/multi_slotted_item_flat/2/000000580703.jpg"),
    ]
    assert count == 10
    for expected_path in expected_paths:
        assert expected_path in planned_paths


@pytest.mark.usefixtures("file_read_write_test")
def test_force_slots_false(
    mock_is_file_extension_allowed, darwin_client, file_read_write_test
):
    mock_is_file_extension_allowed.return_value = True
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile("tests/data.zip") as zfile:
            zfile.extractall(tmp_dir)
            annotations_path = Path(tmp_dir) / "v7-darwin-json-v2/force_slots"
            images_path = Path("images")
            generator, count = download_all_images_from_annotations(
                annotations_path=annotations_path,
                client=darwin_client,
                images_path=images_path,
                force_slots=False,
                force_replace=True,
            )

    download_funcs = list(generator())
    planned_paths = [download_funcs[i].args[2] for i in range(count)]
    expected_paths = [
        Path("images/dir1/single_slotted_video_flat.mp4"),
        Path("images/dir1/multiple_slots_multiple_source_files/0.1/slice_1.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/0.1/slice_2.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/1.1/slice_1.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/1.1/slice_2.dcm"),
        Path("images/dir1/multiple_slots_multiple_source_files/1.1/slice_3.dcm"),
        Path("images/dir1/single_slotted_image_flat.png"),
        Path("images/dir1/multi_slotted_item_flat/0/000000580654.jpg"),
        Path("images/dir1/multi_slotted_item_flat/1/000000580676.jpg"),
        Path("images/dir1/multi_slotted_item_flat/2/000000580703.jpg"),
    ]
    assert count == 10
    for expected_path in expected_paths:
        assert expected_path in planned_paths


@pytest.mark.usefixtures("file_read_write_test")
class TestGetRemoteFilesThatRequireLegacyScaling:
    @pytest.fixture
    def mock_remote_files(self):
        return [
            DatasetItem(
                id=1,
                filename="filename",
                status="new",
                archived=False,
                filesize=1024,  # Example size in bytes
                dataset_id=1,
                dataset_slug="test-dataset",
                seq=1,
                current_workflow_id=None,
                path="/path/to/file",
                slots=[
                    {
                        "slot_name": "0",
                        "metadata": {
                            "medical": {"handler": "MONAI", "affine": [1, 0, 0, 0]}
                        },
                    }
                ],
                layout={},
                current_workflow=None,
            ),
            DatasetItem(
                id=2,
                filename="filename",
                status="new",
                archived=False,
                filesize=2048,  # Example size in bytes
                dataset_id=1,
                dataset_slug="test-dataset",
                seq=2,
                current_workflow_id=None,
                path="/path/to/file",
                slots=[
                    {
                        "slot_name": "0",
                        "metadata": {
                            "medical": {
                                "affine": [
                                    [-1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1],
                                ]
                            }
                        },
                    }
                ],
                layout={},
                current_workflow=None,
            ),
        ]
