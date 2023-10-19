import asyncio
from pathlib import Path
from typing import Coroutine, Dict, Generator, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import responses

import darwin.future.core.items.uploads as uploads
from darwin.future.core.client import ClientCore
from darwin.future.data_objects.item import Item, ItemLayoutV1, ItemLayoutV2, ItemSlot
from darwin.future.exceptions import DarwinException
from darwin.future.tests.core.fixtures import *  # noqa: F401,F403

from .fixtures import *  # noqa: F401,F403


class TestBuildSlots:
    BUILD_SLOT_RETURN_TYPE = List[Dict]

    items_and_expectations: List[Tuple[Item, BUILD_SLOT_RETURN_TYPE]] = []

    # Test empty slots
    items_and_expectations.append((Item(name="name_with_no_slots", slots=[]), []))

    # Test Simple slot with no non-required fields
    items_and_expectations.append(
        (
            Item(
                name="name_with_simple_slot",
                slots=[
                    ItemSlot(
                        slot_name="slot_name_simple",
                        file_name="file_name",
                        storage_key="storage_key",
                    )
                ],
            ),
            [
                {
                    "slot_name": "slot_name_simple",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "type": "image",
                    "fps": 0,
                }
            ],
        )
    )

    # Test with multiple slots
    items_and_expectations.append(
        (
            Item(
                name="name_with_multiple_slots",
                slots=[
                    ItemSlot(
                        slot_name="slot_name1",
                        file_name="file_name1",
                        storage_key="storage_key1",
                    ),
                    ItemSlot(
                        slot_name="slot_name2",
                        file_name="file_name2",
                        storage_key="storage_key2",
                    ),
                ],
            ),
            [
                {
                    "slot_name": "slot_name1",
                    "file_name": "file_name1",
                    "storage_key": "storage_key1",
                    "type": "image",
                    "fps": 0,
                },
                {
                    "slot_name": "slot_name2",
                    "file_name": "file_name2",
                    "storage_key": "storage_key2",
                    "type": "image",
                    "fps": 0,
                },
            ],
        )
    )

    # Test with `as_frames` optional field
    items_and_expectations.append(
        (
            Item(
                name="name_testing_as_frames",
                slots=[
                    ItemSlot(
                        slot_name="slot_name1",
                        file_name="file_name",
                        storage_key="storage_key",
                        as_frames=True,
                    ),
                    ItemSlot(
                        slot_name="slot_name2",
                        file_name="file_name",
                        storage_key="storage_key",
                        as_frames=False,
                    ),
                    ItemSlot(
                        slot_name="slot_name3",
                        file_name="file_name",
                        storage_key="storage_key",
                    ),
                ],
            ),
            [
                {
                    "slot_name": "slot_name1",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                    "as_frames": True,
                },
                {
                    "slot_name": "slot_name2",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                    "as_frames": False,
                },
                {
                    "slot_name": "slot_name3",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                },
            ],
        )
    )

    # Test with `extract_views` optional field
    items_and_expectations.append(
        (
            Item(
                name="name_testing_extract_views",
                slots=[
                    ItemSlot(
                        slot_name="slot_name1",
                        file_name="file_name",
                        storage_key="storage_key",
                        extract_views=True,
                    ),
                    ItemSlot(
                        slot_name="slot_name2",
                        file_name="file_name",
                        storage_key="storage_key",
                        extract_views=False,
                    ),
                    ItemSlot(
                        slot_name="slot_name3",
                        file_name="file_name",
                        storage_key="storage_key",
                    ),
                ],
            ),
            [
                {
                    "slot_name": "slot_name1",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                    "extract_views": True,
                },
                {
                    "slot_name": "slot_name2",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                    "extract_views": False,
                },
                {
                    "slot_name": "slot_name3",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                },
            ],
        )
    )

    # Test with `fps` semi-optional field - field defaults to 0 if not provided
    items_and_expectations.append(
        (
            Item(
                name="name_with_simple_slot",
                slots=[
                    ItemSlot(
                        slot_name="slot_name25",
                        file_name="file_name",
                        storage_key="storage_key",
                        fps=25,  # Testing int
                    ),
                    ItemSlot(
                        slot_name="slot_name29.997",
                        file_name="file_name",
                        storage_key="storage_key",
                        fps=29.997,  # Testing float
                    ),
                    ItemSlot(
                        slot_name="slot_namenative",
                        file_name="file_name",
                        storage_key="storage_key",
                        fps="native",  # Testing literal
                    ),
                    ItemSlot(
                        slot_name="slot_name",
                        file_name="file_name",
                        storage_key="storage_key",
                    ),
                ],
            ),
            [
                {
                    "slot_name": "slot_name25",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "type": "image",
                    "fps": 25,
                },
                {
                    "slot_name": "slot_name29.997",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "type": "image",
                    "fps": 29.997,
                },
                {
                    "slot_name": "slot_namenative",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "type": "image",
                    "fps": "native",
                },
                {
                    "slot_name": "slot_name",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "type": "image",
                    "fps": 0,
                },
            ],
        )
    )

    # Test with `metadata` optional field
    items_and_expectations.append(
        (
            Item(
                name="name_with_simple_slot",
                slots=[
                    ItemSlot(
                        slot_name="slot_name",
                        file_name="file_name",
                        storage_key="storage_key",
                        metadata={"key": "value"},
                    ),
                    ItemSlot(
                        slot_name="slot_name",
                        file_name="file_name",
                        storage_key="storage_key",
                        metadata=None,
                    ),
                    ItemSlot(
                        slot_name="slot_name",
                        file_name="file_name",
                        storage_key="storage_key",
                    ),
                ],
            ),
            [
                {
                    "slot_name": "slot_name",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                    "metadata": {"key": "value"},
                },
                {
                    "slot_name": "slot_name",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                },
                {
                    "slot_name": "slot_name",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "fps": 0,
                    "type": "image",
                },
            ],
        )
    )

    # Test with `tags` optional field
    items_and_expectations.append(
        (
            Item(
                name="name_testing_tags",
                slots=[
                    ItemSlot(
                        slot_name="slot_name_with_string_list",
                        file_name="file_name",
                        storage_key="storage_key",
                        tags=["tag1", "tag2"],
                    ),
                    ItemSlot(
                        slot_name="slot_name_with_kv_pairs",
                        file_name="file_name",
                        storage_key="storage_key",
                        tags={"key": "value"},
                    ),
                ],
            ),
            [
                {
                    "slot_name": "slot_name_with_string_list",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "tags": ["tag1", "tag2"],
                    "fps": 0,
                    "type": "image",
                },
                {
                    "slot_name": "slot_name_with_kv_pairs",
                    "file_name": "file_name",
                    "storage_key": "storage_key",
                    "tags": {"key": "value"},
                    "fps": 0,
                    "type": "image",
                },
            ],
        )
    )

    @pytest.mark.parametrize("item,expected", [(item, expected) for item, expected in items_and_expectations])
    def test_build_slots(self, item: Item, expected: List[Dict]) -> None:
        result = asyncio.run(uploads._build_slots(item))
        assert result == expected


class TestBuildLayout:
    @pytest.mark.parametrize(
        "item, expected",
        [
            (
                Item(
                    name="test_item",
                    layout=ItemLayoutV1(version=1, type="grid", slots=["slot1", "slot2"]),
                ),
                {
                    "slots": ["slot1", "slot2"],
                    "type": "grid",
                    "version": 1,
                },
            ),
            (
                Item(
                    name="test_item",
                    layout=ItemLayoutV2(
                        version=2,
                        type="grid",
                        slots=["slot1", "slot2"],
                        layout_shape=[3, 4],
                    ),
                ),
                {
                    "slots": ["slot1", "slot2"],
                    "type": "grid",
                    "version": 2,
                    "layout_shape": [3, 4],
                },
            ),
        ],
    )
    def test_build_layout(self, item: Item, expected: Dict) -> None:
        assert asyncio.run(uploads._build_layout(item)) == expected


class TestBuildPayloadItems:
    @pytest.mark.parametrize(
        "items_and_paths, expected",
        [
            (
                [
                    (
                        Item(
                            name="test_item",
                            slots=[
                                ItemSlot(
                                    slot_name="slot_name_with_string_list",
                                    file_name="file_name",
                                    storage_key="storage_key",
                                    tags=["tag1", "tag2"],
                                ),
                                ItemSlot(
                                    slot_name="slot_name_with_kv_pairs",
                                    file_name="file_name",
                                    storage_key="storage_key",
                                    tags={"key": "value"},
                                ),
                            ],
                        ),
                        Path("test_path"),
                    )
                ],
                [
                    {
                        "name": "test_item",
                        "path:": "test_path",
                        "tags": [],
                        "slots": [
                            {
                                "slot_name": "slot_name_with_string_list",
                                "file_name": "file_name",
                                "storage_key": "storage_key",
                                "tags": ["tag1", "tag2"],
                                "fps": 0,
                                "type": "image",
                            },
                            {
                                "slot_name": "slot_name_with_kv_pairs",
                                "file_name": "file_name",
                                "storage_key": "storage_key",
                                "tags": {"key": "value"},
                                "fps": 0,
                                "type": "image",
                            },
                        ],
                    }
                ],
            )
        ],
    )
    def test_build_payload_items(self, items_and_paths: List[Tuple[Item, Path]], expected: List[Dict]) -> None:
        result = asyncio.run(uploads._build_payload_items(items_and_paths))
        assert result == expected


class SetupTests:
    @pytest.fixture
    def default_url(self, base_client: ClientCore) -> str:
        return f"{base_client.config.base_url}api/v2/teams/my-team/items"


class TestRegisterUpload(SetupTests):
    @responses.activate()
    @patch.object(uploads, "_build_payload_items")
    def test_async_register_uploads_accepts_tuple_or_list_of_tuples(
        self,
        mock_build_payload_items: MagicMock,
        base_client: ClientCore,
        default_url: str,
    ) -> None:
        mock_build_payload_items.return_value = []

        item = Item(name="name", path="path", slots=[])

        item_and_path = (item, Path("path"))
        items_and_paths = [item_and_path]

        responses.add(
            "post",
            f"{default_url}/register_upload",
            status=200,
            json={
                "dataset_slug": "dataset_slug",
                "items": [],
                "options": {"force_tiling": False, "handle_as_slices": False, "ignore_dicom_layout": False},
            },
        )

        responses.add(
            "post",
            f"{default_url}/register_upload",
            status=200,
            json={
                "dataset_slug": "dataset_slug",
                "items": [],
                "options": {"force_tiling": False, "handle_as_slices": False, "ignore_dicom_layout": False},
            },
        )

        tasks: List[Coroutine] = [
            uploads.async_register_upload(
                base_client,
                "team_slug",
                "dataset_slug",
                items_and_paths,
            ),
            uploads.async_register_upload(
                base_client,
                "team_slug",
                "dataset_slug",
                item_and_path,
            ),
        ]
        try:
            outputs = asyncio.run(tasks[0]), asyncio.run(tasks[1])
        except Exception as e:
            print(e)
            pytest.fail()

        print(outputs)


class TestCreateSignedUploadUrl(SetupTests):
    def test_async_create_signed_upload_url(self, default_url: str, base_config: DarwinConfig) -> None:
        with responses.RequestsMock() as rsps:
            # Mock the API response
            expected_response = {"upload_url": "https://signed.url"}
            rsps.add(
                rsps.POST,
                f"{default_url}/uploads/1/sign",
                json=expected_response,
            )

            # Call the function with mocked arguments
            api_client = ClientCore(base_config)
            actual_response = asyncio.run(uploads.async_create_signed_upload_url(api_client, "1", "my-team"))

            # Check that the response matches the expected response
            if not actual_response:
                pytest.fail("Response was None")

            assert actual_response == expected_response


class TestRegisterAndCreateSignedUploadUrl:
    @pytest.fixture
    def mock_async_register_upload(self) -> Generator:
        with patch.object(uploads, "async_register_upload") as mock:
            yield mock

    @pytest.fixture
    def mock_async_create_signed_upload_url(self) -> Generator:
        with patch.object(uploads, "async_create_signed_upload_url") as mock:
            yield mock

    def test_async_register_and_create_signed_upload_url(
        self,
        mock_async_register_upload: MagicMock,
        mock_async_create_signed_upload_url: MagicMock,
    ) -> None:
        # Set up mock responses
        mock_async_register_upload.return_value = {"id": "123"}
        mock_signed_url_response = {"upload_url": "https://signed.url"}
        mock_async_create_signed_upload_url.return_value = mock_signed_url_response

        # Set up mock API client
        mock_api_client = MagicMock()

        # Call the function with mocked arguments
        actual_response = asyncio.run(
            uploads.async_register_and_create_signed_upload_url(
                mock_api_client,
                "my-team",
                "my-dataset",
                [(Mock(), Mock())],
                False,
                False,
                False,
            )
        )

        # Check that the function called the correct sub-functions with the correct arguments
        mock_async_register_upload.assert_called_once()
        mock_async_create_signed_upload_url.assert_called_once_with(
            mock_api_client,
            "my-team",
            "123",
        )

        # Check that the response matches the expected response
        assert actual_response == mock_signed_url_response

    def test_async_register_and_create_signed_upload_url_raises(
        self,
        mock_async_register_upload: MagicMock,
        mock_async_create_signed_upload_url: MagicMock,
    ) -> None:
        # Set up mock responses
        mock_async_register_upload.return_value = {"id": "123", "errors": ["error"]}
        mock_signed_url_response = {"upload_url": "https://signed.url"}
        mock_async_create_signed_upload_url.return_value = mock_signed_url_response

        # Set up mock API client
        mock_api_client = MagicMock()

        # Check that the response matches the expected response
        with pytest.raises(DarwinException):
            asyncio.run(
                uploads.async_register_and_create_signed_upload_url(
                    mock_api_client,
                    "my-team",
                    "my-dataset",
                    [(Mock(), Mock())],
                    False,
                    False,
                    False,
                )
            )


class TestConfirmUpload(SetupTests):
    @responses.activate
    def test_async_confirm_upload(self, base_client: ClientCore, default_url: str) -> None:
        # Call the function with mocked arguments
        responses.add(
            "POST",
            f"{default_url}/uploads/123/confirm",
            status=200,
            json={},
        )

        actual_response = asyncio.run(
            uploads.async_confirm_upload(
                base_client,
                "my-team",
                "123",
            )
        )

        # Check that the response matches the expected response
        assert actual_response == {}

    def test_async_confirm_upload_raises(self, base_client: ClientCore) -> None:
        base_client.post = MagicMock()  # type: ignore
        base_client.post.side_effect = DarwinException("Error")

        with pytest.raises(DarwinException):
            asyncio.run(uploads.async_confirm_upload(base_client, "team", "123"))


class TestSynchronousMethods:
    @pytest.fixture
    def mock_async_register_upload(self) -> Generator:
        with patch.object(uploads, "async_register_upload") as mock:
            yield mock

    @pytest.fixture
    def mock_async_create_signed_upload_url(self) -> Generator:
        with patch.object(uploads, "async_create_signed_upload_url") as mock:
            yield mock

    @pytest.fixture
    def mock_async_register_and_create_signed_upload_url(self) -> Generator:
        with patch.object(uploads, "async_register_and_create_signed_upload_url") as mock:
            yield mock

    @pytest.fixture
    def mock_async_confirm_upload(self) -> Generator:
        with patch.object(uploads, "async_confirm_upload") as mock:
            yield mock

    def test_register_upload(
        self,
        mock_async_register_upload: MagicMock,
        base_client: ClientCore,
    ) -> None:
        uploads.register_upload(base_client, "team", "dataset", [(Mock(), Mock())])

        mock_async_register_upload.assert_called_once()

    def test_create_signed_upload_url(
        self,
        mock_async_create_signed_upload_url: MagicMock,
        base_client: ClientCore,
    ) -> None:
        uploads.create_signed_upload_url(base_client, "team", "123")

        mock_async_create_signed_upload_url.assert_called_once()

    def test_register_and_create_signed_upload_url(
        self,
        mock_async_register_and_create_signed_upload_url: MagicMock,
        base_client: ClientCore,
    ) -> None:
        uploads.register_and_create_signed_upload_url(base_client, "team", "dataset", [(Mock(), Mock())])

        mock_async_register_and_create_signed_upload_url.assert_called_once()

    def test_confirm_upload(
        self,
        mock_async_confirm_upload: MagicMock,
        base_client: ClientCore,
    ) -> None:
        uploads.confirm_upload(base_client, "team", "123")

        mock_async_confirm_upload.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
