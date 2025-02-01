from pathlib import Path
from typing import List

import pytest
import responses

from darwin.client import Client
from darwin.config import Config
from darwin.dataset.remote_dataset import RemoteDataset
from darwin.dataset.remote_dataset_v2 import RemoteDatasetV2
from darwin.datatypes import Feature, JSONFreeForm, ObjectStore
from darwin.exceptions import NameTaken, NotFound
from darwin.future.data_objects.properties import FullProperty
from darwin.future.tests.core.fixtures import *  # noqa: F401, F403
from tests.fixtures import *  # noqa: F401, F403


from unittest.mock import Mock, patch
from requests import Response, HTTPError
from darwin.client import MAX_RETRIES
from tenacity import RetryError


@pytest.fixture
def darwin_client(
    darwin_config_path: Path, darwin_datasets_path: Path, team_slug_darwin_json_v2: str
) -> Client:
    config = Config(darwin_config_path)
    config.put(["global", "api_endpoint"], "http://localhost/api")
    config.put(["global", "base_url"], "http://localhost")
    config.put(["teams", team_slug_darwin_json_v2, "api_key"], "mock_api_key")
    config.put(
        ["teams", team_slug_darwin_json_v2, "datasets_dir"], str(darwin_datasets_path)
    )
    return Client(config)


@pytest.mark.usefixtures("file_read_write_test")
class TestListRemoteDatasets:
    @responses.activate
    def test_returns_list_of_datasets(self, darwin_client: Client) -> None:
        team_slug: str = "v7"
        endpoint: str = "/datasets"
        json_response: List[JSONFreeForm] = [
            {
                "name": "dataset-name-1",
                "slug": "dataset-slug-1",
                "id": 1,
                "num_images": 1,
                "num_videos": 0,
                "progress": 0,
            },
            {
                "name": "dataset-name-2",
                "slug": "dataset-slug-2",
                "id": 2,
                "num_images": 2,
                "num_videos": 0,
                "progress": 0,
            },
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=json_response, status=200
        )

        remote_datasets = list(darwin_client.list_remote_datasets(team_slug))

        expected_dataset_1 = RemoteDatasetV2(
            team=team_slug,
            name="dataset-name-1",
            slug="dataset-slug-1",
            dataset_id=1,
            item_count=1,
            client=darwin_client,
        )
        expected_dataset_2 = RemoteDatasetV2(
            team=team_slug,
            name="dataset-name-2",
            slug="dataset-slug-2",
            dataset_id=2,
            item_count=2,
            client=darwin_client,
        )

        assert_dataset(remote_datasets[0], expected_dataset_1)
        assert_dataset(remote_datasets[1], expected_dataset_2)

    @responses.activate
    def test_coalesces_null_item_counts_to_zeroes(self, darwin_client: Client) -> None:
        team_slug: str = "v7"
        endpoint: str = "/datasets"
        json_response: List[JSONFreeForm] = [
            {
                "name": "dataset-name-1",
                "slug": "dataset-slug-1",
                "id": 1,
                "num_items": None,  # As this is None, num_images and num_videos should be used
                "num_images": None,  # Should be coalesced to 0
                "num_videos": None,  # Should be coalesced to 0
                "progress": 4,
            },
            {
                "name": "dataset-name-2",
                "slug": "dataset-slug-2",
                "id": 2,
                "num_items": None,  # As this is None, num_images and num_videos should be used
                "num_images": 2,  # Should be used
                "num_videos": 3,  # Should be used
                "progress": 32,
            },
            {
                "name": "dataset-name-3",
                "slug": "dataset-slug-3",
                "id": 3,
                "num_items": 11,  # Should be used
                "num_images": 2,  # Should be ignored, as num_items is present
                "num_videos": 3,  # Should be ignored, as num_items is present
                "progress": 32,
            },
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=json_response, status=200
        )

        remote_datasets = list(darwin_client.list_remote_datasets(team_slug))

        expected_item_count = [0, 5, 11]
        for i, ds in enumerate(remote_datasets):
            assert ds.item_count == expected_item_count[i]


@pytest.mark.usefixtures("file_read_write_test")
class TestGetRemoteDataset:
    @responses.activate
    def test_raises_if_dataset_is_not_found(self, darwin_client: Client) -> None:
        endpoint: str = "/datasets"
        json_response = [
            {
                "name": "dataset-name-1",
                "slug": "dataset-slug-1",
                "id": 1,
                "num_images": 1,
                "num_videos": 0,
                "progress": 0,
            }
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=json_response, status=200
        )

        with pytest.raises(NotFound):
            darwin_client.get_remote_dataset("v7/dataset-slug-2")

    @responses.activate
    def test_returns_the_dataset(self, darwin_client: Client) -> None:
        endpoint: str = "/datasets"
        json_response = [
            {
                "name": "dataset-name-1",
                "slug": "dataset-slug-1",
                "id": 1,
                "num_images": 1,
                "num_videos": 0,
                "progress": 0,
            }
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=json_response, status=200
        )

        actual_dataset = darwin_client.get_remote_dataset("v7/dataset-slug-1")
        expected_dataset = RemoteDatasetV2(
            team="v7",
            name="dataset-name-1",
            slug="dataset-slug-1",
            dataset_id=1,
            item_count=1,
            client=darwin_client,
        )

        assert_dataset(actual_dataset, expected_dataset)


@pytest.mark.usefixtures("file_read_write_test")
class TestCreateDataset:
    @responses.activate
    def test_returns_the_created_dataset(self, darwin_client: Client) -> None:
        endpoint: str = "/datasets"
        json_response: JSONFreeForm = {
            "name": "my-dataset",
            "slug": "my-dataset",
            "id": 1,
            "num_images": 1,
            "num_videos": 0,
            "progress": 0,
        }

        responses.add(
            responses.POST, darwin_client.url + endpoint, json=json_response, status=200
        )

        actual_dataset = darwin_client.create_dataset("my-dataset", "v7")
        expected_dataset = RemoteDatasetV2(
            team="v7",
            name="my-dataset",
            slug="my-dataset",
            dataset_id=1,
            item_count=1,
            client=darwin_client,
        )

        assert_dataset(actual_dataset, expected_dataset)

    @responses.activate
    def test_raises_if_name_is_taken(self, darwin_client: Client) -> None:
        endpoint: str = "/datasets"
        json_response: JSONFreeForm = {"errors": {"name": ["has already been taken"]}}

        responses.add(
            responses.POST,
            darwin_client.url + endpoint,
            json=json_response,
            status=422,
            adding_headers={"content-type": "utf-8"},
        )

        with pytest.raises(NameTaken):
            darwin_client.create_dataset("my-dataset", "v7")


@pytest.mark.usefixtures("file_read_write_test")
class TestFetchRemoteClasses:
    @responses.activate
    def test_returns_remote_classes(
        self, team_slug_darwin_json_v2: str, darwin_client: Client
    ) -> None:
        endpoint: str = (
            f"/teams/{team_slug_darwin_json_v2}/annotation_classes?include_tags=true"
        )
        response: JSONFreeForm = {
            "annotation_classes": [
                {
                    "annotation_class_image_url": None,
                    "annotation_types": ["tag"],
                    "dataset_id": 215,
                    "datasets": [{"id": 215}, {"id": 265}],
                    "description": " Tag 2",
                    "id": 345,
                    "images": [],
                    "inserted_at": "2021-01-25T02:27:10",
                    "metadata": {"_color": "rgba(0,255,0,1.0)", "tag": {}},
                    "name": " Tag 2",
                    "team_id": 2,
                    "updated_at": "2021-01-25T02:27:10",
                }
            ]
        }

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=response, status=200
        )

        result: List[JSONFreeForm] = darwin_client.fetch_remote_classes(
            team_slug_darwin_json_v2
        )
        annotation_class: JSONFreeForm = result[0]

        assert annotation_class["annotation_class_image_url"] is None
        assert annotation_class["annotation_types"] == ["tag"]
        assert annotation_class["dataset_id"] == 215
        assert annotation_class["datasets"] == [{"id": 215}, {"id": 265}]
        assert annotation_class["id"] == 345


@pytest.mark.usefixtures("file_read_write_test")
class TestGetTeamFeatures:
    @responses.activate
    def test_returns_list_of_features(
        self, team_slug_darwin_json_v2: str, darwin_client: Client
    ) -> None:
        endpoint: str = f"/teams/{team_slug_darwin_json_v2}/features"
        json_response = [
            {"enabled": False, "name": "WORKFLOW_V2"},
            {"enabled": True, "name": "BLIND_STAGE"},
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=json_response, status=200
        )

        assert darwin_client.get_team_features(team_slug_darwin_json_v2) == [
            Feature(name="WORKFLOW_V2", enabled=False),
            Feature(name="BLIND_STAGE", enabled=True),
        ]


def assert_dataset(dataset_1: RemoteDataset, dataset_2: RemoteDataset) -> None:
    assert dataset_1.name == dataset_2.name
    assert dataset_1.team == dataset_2.team
    assert dataset_1.dataset_id == dataset_2.dataset_id
    assert dataset_1.slug == dataset_2.slug
    assert dataset_1.item_count == dataset_2.item_count


# Tests for _get_item_count
def test__get_item_count_defaults_to_num_items_if_present() -> None:
    dataset_return = {
        "num_images": 2,  # Should be ignored
        "num_videos": 3,  # Should be ignored
        "num_items": 5,  # Should get this one
    }

    assert Client._get_item_count(dataset_return) == 5


def test__get_item_count_returns_sum_of_others_if_num_items_not_present() -> None:
    dataset_return = {
        "num_images": 7,  # Should be summed
        "num_videos": 3,  # Should be summed
    }

    assert Client._get_item_count(dataset_return) == 10


def test__get_item_count_should_tolerate_missing_members() -> None:
    assert (
        Client._get_item_count(
            {
                "num_videos": 3,  # Should be ignored
            }
        )
        == 3
    )

    assert (
        Client._get_item_count(
            {
                "num_images": 2,
            }
        )
        == 2
    )


@pytest.mark.usefixtures("file_read_write_test")
class TestGetTeamProperties:
    @responses.activate
    def test_get_team_properties(self, darwin_client: Client) -> None:
        responses.add(
            responses.GET,
            "http://localhost/apiv2/teams/v7-darwin-json-v2/properties?include_values=true",
            json={
                "properties": [
                    {
                        "annotation_class_id": 2558,
                        "description": "test question",
                        "id": "d7368686-d087-4d92-bfd9-8ae776c3ed3a",
                        "name": "property question",
                        "property_values": [
                            {
                                "color": "rgba(143,255,0,1.0)",
                                "id": "3e40c575-41dc-43c9-87f9-7dc2b625650d",
                                "type": "string",
                                "value": "answer 1",
                            },
                            {
                                "color": "rgba(173,255,0,1.0)",
                                "id": "18ebaad0-c22a-49db-b6c5-1de0da986f4e",
                                "type": "string",
                                "value": "answer 2",
                            },
                            {
                                "color": "rgba(82,255,0,1.0)",
                                "id": "b67ae529-f612-4c9a-a175-5b98f1d81a6e",
                                "type": "string",
                                "value": "answer 3",
                            },
                        ],
                        "required": False,
                        "granularity": "section",
                        "slug": "property-question",
                        "team_id": 128,
                        "type": "multi_select",
                    },
                ]
            },
            status=200,
        )
        team_slug = "v7-darwin-json-v2"
        assert len(darwin_client.get_team_properties(team_slug)) == 1


@pytest.mark.usefixtures("file_read_write_test")
class TestCreateProperty:
    @responses.activate
    def test_create_property(
        self, darwin_client: Client, base_property_object: FullProperty
    ) -> None:
        responses.add(
            responses.POST,
            "http://localhost/apiv2/teams/v7-darwin-json-v2/properties",
            json=base_property_object.model_dump(mode="json"),
            status=200,
        )
        _property = darwin_client.create_property(
            team_slug="v7-darwin-json-v2", params=base_property_object
        )
        assert isinstance(_property, FullProperty)
        assert _property == base_property_object

    @responses.activate
    def test_create_property_from_json(
        self, darwin_client: Client, base_property_object: FullProperty
    ) -> None:
        responses.add(
            responses.POST,
            "http://localhost/apiv2/teams/v7-darwin-json-v2/properties",
            json=base_property_object.model_dump(mode="json"),
            status=200,
        )
        _property = darwin_client.create_property(
            team_slug="v7-darwin-json-v2",
            params=base_property_object.model_dump(mode="json"),
        )
        assert isinstance(_property, FullProperty)
        assert _property == base_property_object


@pytest.mark.usefixtures("file_read_write_test")
class TestUpdateProperty:
    @responses.activate
    def test_update_property(
        self, darwin_client: Client, base_property_object: FullProperty
    ) -> None:
        property_id = base_property_object.id
        responses.add(
            responses.PUT,
            f"http://localhost/apiv2/teams/v7-darwin-json-v2/properties/{property_id}",
            json=base_property_object.model_dump(mode="json"),
            status=200,
        )
        _property = darwin_client.update_property(
            team_slug="v7-darwin-json-v2", params=base_property_object
        )
        assert isinstance(_property, FullProperty)
        assert _property == base_property_object

    @responses.activate
    def test_update_property_from_json(
        self, darwin_client: Client, base_property_object: FullProperty
    ) -> None:
        property_id = base_property_object.id
        responses.add(
            responses.PUT,
            f"http://localhost/apiv2/teams/v7-darwin-json-v2/properties/{property_id}",
            json=base_property_object.model_dump(mode="json"),
            status=200,
        )
        _property = darwin_client.update_property(
            team_slug="v7-darwin-json-v2",
            params=base_property_object.model_dump(mode="json"),
        )
        assert isinstance(_property, FullProperty)
        assert _property == base_property_object


@pytest.mark.usefixtures("file_read_write_test")
class TestGetExternalStorage:
    @responses.activate
    def test_returns_external_storage(self, darwin_client: Client) -> None:
        team_slug: str = "v7"
        endpoint: str = f"/teams/{team_slug}/storage"
        response: List[JSONFreeForm] = [
            {
                "name": "storage-name-1",
                "prefix": "storage-prefix-1",
                "readonly": False,
                "provider": "aws",
                "default": True,
            }
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=response, status=200
        )

        actual_storage = darwin_client.get_external_storage(team_slug, "storage-name-1")
        expected_storage = ObjectStore(
            name="storage-name-1",
            prefix="storage-prefix-1",
            readonly=False,
            provider="aws",
            default=True,
        )

        assert actual_storage.name == expected_storage.name
        assert actual_storage.prefix == expected_storage.prefix
        assert actual_storage.readonly == expected_storage.readonly
        assert actual_storage.provider == expected_storage.provider
        assert actual_storage.default == expected_storage.default


@pytest.mark.usefixtures("file_read_write_test")
class TestListExternalStorageConnections:
    @responses.activate
    def test_returns_list_of_external_storage_connections(
        self, darwin_client: Client
    ) -> None:
        team_slug: str = "v7"
        endpoint: str = f"/teams/{team_slug}/storage"
        json_response: List[JSONFreeForm] = [
            {
                "name": "storage-name-1",
                "prefix": "storage-prefix-1",
                "readonly": False,
                "provider": "aws",
                "default": True,
            },
            {
                "name": "storage-name-2",
                "prefix": "storage-prefix-2",
                "readonly": True,
                "provider": "gcp",
                "default": False,
            },
        ]

        responses.add(
            responses.GET, darwin_client.url + endpoint, json=json_response, status=200
        )

        actual_storages = list(
            darwin_client.list_external_storage_connections(team_slug)
        )

        expected_storage_1 = ObjectStore(
            name="storage-name-1",
            prefix="storage-prefix-1",
            readonly=False,
            provider="aws",
            default=True,
        )
        expected_storage_2 = ObjectStore(
            name="storage-name-2",
            prefix="storage-prefix-2",
            readonly=True,
            provider="gcp",
            default=False,
        )

        assert actual_storages[0].name == expected_storage_1.name
        assert actual_storages[0].prefix == expected_storage_1.prefix
        assert actual_storages[0].readonly == expected_storage_1.readonly
        assert actual_storages[0].provider == expected_storage_1.provider
        assert actual_storages[0].default == expected_storage_1.default

        assert actual_storages[1].name == expected_storage_2.name
        assert actual_storages[1].prefix == expected_storage_2.prefix
        assert actual_storages[1].readonly == expected_storage_2.readonly
        assert actual_storages[1].provider == expected_storage_2.provider
        assert actual_storages[1].default == expected_storage_2.default


class TestClientRetry:
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=Config)

        # Set up the mock to return different values based on the key
        def get_side_effect(key, default=None):
            if key == "global/api_endpoint":
                return "https://darwin.v7labs.com/api/"
            if key == "global/payload_compression_level":
                return "0"
            return default

        config.get.side_effect = get_side_effect
        config.get_team.return_value = Mock(api_key="test-key", slug="test-team")
        return config

    @pytest.fixture
    def client(self, mock_config):
        return Client(config=mock_config, default_team="test-team")

    @patch("time.sleep", return_value=None)
    def test_get_retries_on_429(self, mock_sleep, client):
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch("requests.Session.get") as mock_get:
            mock_get.return_value = mock_response

            with pytest.raises(RetryError):
                client._get("/test-endpoint")

            assert mock_get.call_count == MAX_RETRIES

    @patch("time.sleep", return_value=None)
    def test_post_retries_on_429(self, mock_sleep, client):
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RetryError):
                client._post("/test-endpoint", {"test": "data"})

            assert mock_post.call_count == MAX_RETRIES

    @patch("time.sleep", return_value=None)
    def test_put_retries_on_429(self, mock_sleep, client):
        # Create a mock response that simulates a 429 status
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)

        with patch("requests.Session.put") as mock_put:
            mock_put.return_value = mock_response

            with pytest.raises(RetryError):
                client._put("/test-endpoint", {"test": "data"})

            assert mock_put.call_count == MAX_RETRIES

    @patch("time.sleep", return_value=None)
    def test_request_succeeds_after_retries(self, mock_sleep, client):
        mock_429_response = Mock(spec=Response)
        mock_429_response.status_code = 429
        mock_429_response.headers = {}

        mock_success_response = Mock(spec=Response)
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"success": True}
        mock_success_response.headers = {}

        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = [
                HTTPError(response=mock_429_response),
                HTTPError(response=mock_429_response),
                mock_success_response,
            ]

            result = client._get("/test-endpoint")

            assert result == {"success": True}
            assert mock_get.call_count == 3

    def test_no_retry_on_other_errors(self, client):
        mock_response = Mock(spec=Response)
        mock_response.status_code = 404

        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = HTTPError(response=mock_response)

            with pytest.raises(HTTPError):
                client._get("/test-endpoint")

            assert mock_get.call_count == 1

    @patch("time.sleep", return_value=None)
    def test_retry_respects_rate_limit_headers(self, mock_sleep, client):
        mock_response = Mock(spec=Response)
        mock_response.status_code = 429

        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = HTTPError(response=mock_response)

            with pytest.raises(RetryError):
                client._get("/test-endpoint")

            assert mock_get.call_count == MAX_RETRIES
            assert mock_sleep.called
