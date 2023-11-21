import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import orjson as json
import pytest
import responses

from darwin.client import Client
from darwin.config import Config
from darwin.dataset import RemoteDataset
from darwin.dataset.release import Release
from darwin.dataset.remote_dataset_v1 import RemoteDatasetV1
from darwin.dataset.upload_manager import LocalFile, UploadHandlerV1
from darwin.exceptions import UnsupportedExportFormat, UnsupportedFileType
from darwin.item import DatasetItem
from tests.fixtures import *


@pytest.fixture
def annotation_name() -> str:
    return "test_video.json"


@pytest.fixture
def annotation_content() -> Dict[str, Any]:
    return {
        "image": {
            "width": 1920,
            "height": 1080,
            "filename": "test_video.mp4",
            "fps": 20.0,
            "frame_urls": ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"],
        },
        "annotations": [
            {
                "frames": {
                    "0": {
                        "polygon": {
                            "path": [
                                {"x": 0, "y": 0},
                                {"x": 1, "y": 1},
                                {"x": 1, "y": 0},
                            ]
                        }
                    },
                    "2": {
                        "polygon": {
                            "path": [
                                {"x": 5, "y": 5},
                                {"x": 6, "y": 6},
                                {"x": 6, "y": 5},
                            ]
                        }
                    },
                    "4": {
                        "polygon": {
                            "path": [
                                {"x": 9, "y": 9},
                                {"x": 8, "y": 8},
                                {"x": 8, "y": 9},
                            ]
                        }
                    },
                },
                "name": "test_class",
                "segments": [[0, 3]],
            }
        ],
    }


@pytest.fixture
def darwin_client(
    darwin_config_path: Path, darwin_datasets_path: Path, team_slug: str
) -> Client:
    config = Config(darwin_config_path)
    config.put(["global", "api_endpoint"], "http://localhost/api")
    config.put(["global", "base_url"], "http://localhost")
    config.put(["teams", team_slug, "api_key"], "mock_api_key")
    config.put(["teams", team_slug, "datasets_dir"], str(darwin_datasets_path))
    return Client(config)


@pytest.fixture
def create_annotation_file(
    darwin_datasets_path: Path,
    team_slug: str,
    dataset_slug: str,
    release_name: str,
    annotation_name: str,
    annotation_content: dict,
):
    annotations: Path = (
        darwin_datasets_path
        / team_slug
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
                "archived": False,
                "archived_reason": None,
                "current_workflow": {
                    "current_stage_number": 1,
                    "current_workflow_stage_template_id": 1258,
                    "dataset_item_id": 386074,
                    "id": 34533,
                    "stages": {
                        "1": [
                            {
                                "assignee_id": 172,
                                "completed": False,
                                "completes_at": None,
                                "dataset_item_id": 386074,
                                "id": 106630,
                                "metadata": {},
                                "number": 1,
                                "skipped": False,
                                "skipped_reason": None,
                                "template_metadata": {
                                    "assignable_to": "manual",
                                    "base_sampling_rate": 1.0,
                                    "user_sampling_rate": 1.0,
                                },
                                "type": "annotate",
                                "workflow_id": 34533,
                                "workflow_stage_template_id": 1258,
                            }
                        ],
                        "2": [
                            {
                                "assignee_id": None,
                                "completed": False,
                                "completes_at": None,
                                "dataset_item_id": 386074,
                                "id": 106631,
                                "metadata": {},
                                "number": 2,
                                "skipped": False,
                                "skipped_reason": None,
                                "template_metadata": {
                                    "assignable_to": "any_user",
                                    "base_sampling_rate": 1.0,
                                    "readonly": False,
                                    "user_sampling_rate": 1.0,
                                },
                                "type": "review",
                                "workflow_id": 34533,
                                "workflow_stage_template_id": 1259,
                            }
                        ],
                        "3": [
                            {
                                "assignee_id": None,
                                "completed": False,
                                "completes_at": None,
                                "dataset_item_id": 386074,
                                "id": 106632,
                                "metadata": {},
                                "number": 3,
                                "skipped": False,
                                "skipped_reason": None,
                                "template_metadata": {},
                                "type": "complete",
                                "workflow_id": 34533,
                                "workflow_stage_template_id": 1260,
                            }
                        ],
                    },
                    "status": "annotate",
                    "workflow_template_id": 455,
                },
                "current_workflow_id": 34533,
                "dataset_id": 312,
                "dataset_image": {
                    "dataset_id": 312,
                    "dataset_video_id": None,
                    "id": 192905,
                    "image": {
                        "external": False,
                        "height": 3024,
                        "id": 171674,
                        "key": "data/datasets/312/originals/00000006.jpg",
                        "original_filename": "dan-gold-Q_2p94h8rjI-unsplash.jpg",
                        "thumbnail_url": "https://localhost/data/datasets/312/thumbnails/00000006.jpg?Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTYyNjUxMDE5Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vc3RhZ2luZy52N2xhYnMuY29tL2RhdGEvZGF0YXNldHMvMzEyL3RodW1ibmFpbHMvMDAwMDAwMDYuanBnIn1dfQ==&Signature=iVrFk5qiDohQnr5UUgBAFsJtXC3G8rBSNmQTFeIjP2M4HE5QASII/rikRLDbMvRtG2QopWIpohclGp8tFEi2W1moo5LOQ69S+wmEulfr38ZWz4BHinzVesmC/oNeU0hGNeFKkkKlezDE2kOZADWx5fbgRBmRcsqXWM5aTpxn97G7GhmhQtzgKJB3uY4HSpMLw+/6R3m5g86c5mlzogBa6wdisN8AWNs8ftyQrFQiucHKfV0NyHgsFr8+zzSDbh6qp1A62d++IvDn3NWMMZju3bJMvmHGsuW2BqL4JbXHICQsIQSnpkLvCuqNsxqSrMzkeBgpjrT3E0YX7RVAseLAPA==&Key-Pair-Id=APKAIQLX6XUIH32V3QKA",
                        "uploaded": True,
                        "url": "https://localhost/data/datasets/312/originals/00000006.jpg?Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTYyNjUxMDE5Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vc3RhZ2luZy52N2xhYnMuY29tL2RhdGEvZGF0YXNldHMvMzEyL29yaWdpbmFscy8wMDAwMDAwNi5qcGcifV19&Signature=TYxoSOGeANEgjiGsG2krf4m0D3Xev/1w47pvwXL3kVhP50xTkgg7Zhy3XUg6bxQCWaJwsBgwxf6txqUzKUQxCzUHw131bZ4+il6tu9d8xUmoVcx/GpviNDbOmdTxJlPqqggR5xxgFTFj6EQ+kvR02MNbhLstHJpNJNf00TzYeQLhTTa/8XC99keuJ3wlZVuVz3yny3zTlAfYWd9t5SkTkeqQtn7T0Vm8IYrk3khOdJbI4kp65iHGu/3uuNsDKZI57D2A3jRMGOIiAKXNP4ZZfL3oBkYf3nn8oCdiOQ/dik5SBYutgif0QcJWH/dZ9wziKEV1k+tnlX+dZ1NiUwT2hQ==&Key-Pair-Id=APKAIQLX6XUIH32V3QKA",
                        "width": 4032,
                    },
                    "seq": 6,
                    "set": 1625492879,
                },
                "dataset_image_id": 192905,
                "dataset_video": None,
                "dataset_video_id": None,
                "file_size": 2911814,
                "filename": "dan-gold-Q_2p94h8rjI-unsplash.jpg",
                "height": 3024,
                "id": 386074,
                "inserted_at": "2021-07-05T13:47:59",
                "labels": [875],
                "path": "/",
                "priority": 0,
                "seq": 6,
                "set": 1625492879,
                "status": "annotate",
                "type": "image",
                "updated_at": "2021-07-06T14:07:24",
                "width": 4032,
            },
            {
                "archived": False,
                "archived_reason": None,
                "current_workflow": {
                    "current_stage_number": 2,
                    "current_workflow_stage_template_id": 1259,
                    "dataset_item_id": 386073,
                    "id": 34532,
                    "stages": {
                        "1": [
                            {
                                "assignee_id": 172,
                                "completed": True,
                                "completes_at": None,
                                "dataset_item_id": 386073,
                                "id": 106627,
                                "metadata": {},
                                "number": 1,
                                "skipped": False,
                                "skipped_reason": None,
                                "template_metadata": {
                                    "assignable_to": "manual",
                                    "base_sampling_rate": 1.0,
                                    "user_sampling_rate": 1.0,
                                },
                                "type": "annotate",
                                "workflow_id": 34532,
                                "workflow_stage_template_id": 1258,
                            }
                        ],
                        "2": [
                            {
                                "assignee_id": 172,
                                "completed": False,
                                "completes_at": None,
                                "dataset_item_id": 386073,
                                "id": 106628,
                                "metadata": {"previous_stage_number": 1},
                                "number": 2,
                                "skipped": False,
                                "skipped_reason": None,
                                "template_metadata": {
                                    "assignable_to": "any_user",
                                    "base_sampling_rate": 1.0,
                                    "readonly": False,
                                    "user_sampling_rate": 1.0,
                                },
                                "type": "review",
                                "workflow_id": 34532,
                                "workflow_stage_template_id": 1259,
                            }
                        ],
                        "3": [
                            {
                                "assignee_id": None,
                                "completed": False,
                                "completes_at": None,
                                "dataset_item_id": 386073,
                                "id": 106629,
                                "metadata": {},
                                "number": 3,
                                "skipped": False,
                                "skipped_reason": None,
                                "template_metadata": {},
                                "type": "complete",
                                "workflow_id": 34532,
                                "workflow_stage_template_id": 1260,
                            }
                        ],
                    },
                    "status": "review",
                    "workflow_template_id": 455,
                },
                "current_workflow_id": 34532,
                "dataset_id": 312,
                "dataset_image": {
                    "dataset_id": 312,
                    "dataset_video_id": None,
                    "id": 192904,
                    "image": {
                        "external": False,
                        "height": 3344,
                        "id": 171673,
                        "key": "data/datasets/312/originals/00000005.jpg",
                        "original_filename": "dan-gold-N7RiDzfF2iw-unsplash.jpg",
                        "thumbnail_url": "https://localhost/data/datasets/312/thumbnails/00000005.jpg?Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTYyNjUxMDE5Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vc3RhZ2luZy52N2xhYnMuY29tL2RhdGEvZGF0YXNldHMvMzEyL3RodW1ibmFpbHMvMDAwMDAwMDUuanBnIn1dfQ==&Signature=issN9nvtEYfIQWiK5K1o+zOOOPkUb6aIbehuI/JqG/Yytq5UxGsnWzot880FlFF2yIQ6nsbRexvWCc7EO41oJGVx8qMRISbDMvbDkmj//uGlh1bjE7W6GntcBVmNh71JWgzDyNKUq8H8sScQpv1DQ9B6LOs1bPmPor3nfm3RFmobAJo5Yh5qeGJ0nSlpNH1+DUqI3fnLC7vV/w+tFdQVyswHIIKYKNEUk1indVbsLazLjpUpr5E9Vv7yUjq1adw2uXyGrPbWobxgvMkFK7lpHJVtTq3FTCpwMso7xbkb6VppSEkKnH+FLfa661U35rUKnH1DYBOnv3Q7HGDUGeKEDQ==&Key-Pair-Id=APKAIQLX6XUIH32V3QKA",
                        "uploaded": True,
                        "url": "https://localhost/data/datasets/312/originals/00000005.jpg?Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTYyNjUxMDE5Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vc3RhZ2luZy52N2xhYnMuY29tL2RhdGEvZGF0YXNldHMvMzEyL29yaWdpbmFscy8wMDAwMDAwNS5qcGcifV19&Signature=HgKTEtl7nK2dKCf5jzECx+p/TdiICQkXw8sTGiLUFotn9iI5e46PCF+ShTvBXrVG9uhvIv0ifrmGmjSapA9vOXGHvyFRo/+RkcVjvQGhvg5B7JCS6ii3nolLZraqr5kHR4otNKwxs0+oynsliJSmffK+o7EPpYlrZ4Xqx/nXG5W9qSk4ndvSrC822VulzbARjPupC4lGMoHA+AUALnC8y9JXPmouexGeRBcQ+y8Bg7WD0hEbbPe20JvzGDc8JwJ6mu9wCZfbFC/RS3AWCudUXvXbl1X3PWt9DQveTO60zO9/xB+ubKu6Cj9np9ol45TJUGEfrLsdT5CkFL2+J8ZgTg==&Key-Pair-Id=APKAIQLX6XUIH32V3QKA",
                        "width": 5943,
                    },
                    "seq": 5,
                    "set": 1625492879,
                },
                "dataset_image_id": 192904,
                "dataset_video": None,
                "dataset_video_id": None,
                "file_size": 2613529,
                "filename": "dan-gold-N7RiDzfF2iw-unsplash.jpg",
                "height": 3344,
                "id": 386073,
                "inserted_at": "2021-07-05T13:47:59",
                "labels": [875],
                "path": "/",
                "priority": 0,
                "seq": 5,
                "set": 1625492879,
                "status": "review",
                "type": "image",
                "updated_at": "2021-07-06T14:06:02",
                "width": 5943,
            },
        ],
        "metadata": {"next": None, "previous": "2021-07-06 14:07:24,6"},
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


@pytest.mark.usefixtures("file_read_write_test", "create_annotation_file")
class TestSplitVideoAnnotations:
    def test_works_on_videos(
        self,
        darwin_client: Client,
        darwin_datasets_path: Path,
        dataset_name: str,
        dataset_slug: str,
        release_name: str,
        team_slug: str,
    ):
        remote_dataset = RemoteDatasetV1(
            client=darwin_client,
            team=team_slug,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )

        remote_dataset.split_video_annotations()

        video_path = (
            darwin_datasets_path
            / team_slug
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
                "annotations": [
                    {
                        "name": "test_class",
                        "polygon": {
                            "path": [
                                {"x": 0, "y": 0},
                                {"x": 1, "y": 1},
                                {"x": 1, "y": 0},
                            ]
                        },
                    }
                ],
                "image": {
                    "filename": "test_video/0000000.png",
                    "height": 1080,
                    "url": "frame_1.jpg",
                    "width": 1920,
                },
            }

        with (video_path / "0000001.json").open() as f:
            assert json.loads(f.read()) == {
                "annotations": [],
                "image": {
                    "filename": "test_video/0000001.png",
                    "height": 1080,
                    "url": "frame_2.jpg",
                    "width": 1920,
                },
            }

        with (video_path / "0000002.json").open() as f:
            assert json.loads(f.read()) == {
                "annotations": [
                    {
                        "name": "test_class",
                        "polygon": {
                            "path": [
                                {"x": 5, "y": 5},
                                {"x": 6, "y": 6},
                                {"x": 6, "y": 5},
                            ]
                        },
                    }
                ],
                "image": {
                    "filename": "test_video/0000002.png",
                    "height": 1080,
                    "url": "frame_3.jpg",
                    "width": 1920,
                },
            }


@pytest.mark.usefixtures("files_content", "file_read_write_test")
class TestFetchRemoteFiles:
    @responses.activate
    def test_works(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug: str,
        files_content: dict,
    ):
        remote_dataset = RemoteDatasetV1(
            client=darwin_client,
            team=team_slug,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )
        url = "http://localhost/api/datasets/1/items?page%5Bsize%5D=500"
        responses.add(
            responses.POST,
            url,
            json=files_content,
            status=200,
        )

        actual = remote_dataset.fetch_remote_files()

        assert isinstance(actual, types.GeneratorType)

        (item_1, item_2) = list(actual)

        assert responses.assert_call_count(url, 1) is True

        assert item_1.id == 386074
        assert item_2.id == 386073

    @responses.activate
    def test_fetches_files_with_commas(
        self,
        darwin_client: Client,
        dataset_name: str,
        dataset_slug: str,
        team_slug: str,
        files_content: dict,
    ):
        remote_dataset = RemoteDatasetV1(
            client=darwin_client,
            team=team_slug,
            name=dataset_name,
            slug=dataset_slug,
            dataset_id=1,
        )
        url = "http://localhost/api/datasets/1/items?page%5Bsize%5D=500"
        responses.add(
            responses.POST,
            url,
            json=files_content,
            status=200,
        )

        list(
            remote_dataset.fetch_remote_files(
                {"filenames": ["example,with, comma.mp4"]}
            )
        )

        request_body = json.loads(responses.calls[0].request.body)

        assert request_body["filter"]["filenames"] == ["example,with, comma.mp4"]


@pytest.fixture
def remote_dataset(
    darwin_client: Client, dataset_name: str, dataset_slug: str, team_slug: str
):
    return RemoteDatasetV1(
        client=darwin_client,
        team=team_slug,
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
        assert_upload_mocks_are_correctly_called(
            remote_dataset, [LocalFile("test.jpg")]
        )

    def test_works_with_path_list(self, remote_dataset: RemoteDataset):
        assert_upload_mocks_are_correctly_called(remote_dataset, [Path("test.jpg")])

    def test_works_with_str_list(self, remote_dataset: RemoteDataset):
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
        assert_upload_mocks_are_correctly_called(remote_dataset, filenames)

    def test_raises_with_unsupported_files(self, remote_dataset: RemoteDataset):
        with pytest.raises(UnsupportedFileType):
            remote_dataset.push(["test.txt"])


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
    )


@pytest.mark.usefixtures("file_read_write_test")
class TestArchive:
    def test_calls_client_put(
        self,
        remote_dataset: RemoteDataset,
        dataset_item: DatasetItem,
        team_slug: str,
        dataset_slug: str,
    ):
        with patch.object(Client, "archive_item", return_value={}) as stub:
            remote_dataset.archive([dataset_item])
            stub.assert_called_once_with(
                dataset_slug, team_slug, {"filter": {"dataset_item_ids": [1]}}
            )


@pytest.mark.usefixtures("file_read_write_test")
class TestMoveToNew:
    def test_calls_client_put(
        self,
        remote_dataset: RemoteDataset,
        dataset_item: DatasetItem,
        team_slug: str,
        dataset_slug: str,
    ):
        with patch.object(Client, "move_item_to_new", return_value={}) as stub:
            remote_dataset.move_to_new([dataset_item])
            stub.assert_called_once_with(
                dataset_slug, team_slug, {"filter": {"dataset_item_ids": [1]}}
            )


@pytest.mark.usefixtures("file_read_write_test")
class TestReset:
    def test_calls_client_put(
        self,
        remote_dataset: RemoteDataset,
        dataset_item: DatasetItem,
        team_slug: str,
        dataset_slug: str,
    ):
        with patch.object(Client, "reset_item", return_value={}) as stub:
            remote_dataset.reset([dataset_item])
            stub.assert_called_once_with(
                dataset_slug, team_slug, {"filter": {"dataset_item_ids": [1]}}
            )


@pytest.mark.usefixtures("file_read_write_test")
class TestRestoreArchived:
    def test_calls_client_put(
        self,
        remote_dataset: RemoteDataset,
        dataset_item: DatasetItem,
        team_slug: str,
        dataset_slug: str,
    ):
        with patch.object(Client, "restore_archived_item", return_value={}) as stub:
            remote_dataset.restore_archived([dataset_item])
            stub.assert_called_once_with(
                dataset_slug, team_slug, {"filter": {"dataset_item_ids": [1]}}
            )


@pytest.mark.usefixtures("file_read_write_test")
class TestDeleteItems:
    def test_calls_client_delete(
        self,
        remote_dataset: RemoteDataset,
        dataset_item: DatasetItem,
        team_slug: str,
        dataset_slug: str,
    ):
        with patch.object(Client, "delete_item", return_value={}) as stub:
            remote_dataset.delete_items([dataset_item])
            stub.assert_called_once_with(
                "test-dataset", team_slug, {"filter": {"dataset_item_ids": [1]}}
            )


def assert_upload_mocks_are_correctly_called(remote_dataset: RemoteDataset, *args):
    with patch.object(
        UploadHandlerV1, "_request_upload", return_value=([], [])
    ) as request_upload_mock:
        with patch.object(UploadHandlerV1, "upload") as upload_mock:
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
    def test_honours_include_authorship(self, remote_dataset: RemoteDataset):
        with patch.object(Client, "create_export", return_value={}) as stub:
            remote_dataset.export("example", None, False, True)
            stub.assert_called_once_with(
                remote_dataset.dataset_id,
                {
                    "annotation_class_ids": [],
                    "name": "example",
                    "include_export_token": False,
                    "include_authorship": True,
                },
                remote_dataset.team,
            )

    def test_default_values_have_negative_includes(self, remote_dataset: RemoteDataset):
        with patch.object(Client, "create_export", return_value={}) as stub:
            remote_dataset.export("example")
            stub.assert_called_once_with(
                remote_dataset.dataset_id,
                {
                    "annotation_class_ids": [],
                    "name": "example",
                    "include_export_token": False,
                    "include_authorship": False,
                },
                remote_dataset.team,
            )
