from typing import List
from uuid import UUID, uuid4

import responses

from darwin.future.core.client import ClientCore
from darwin.future.core.items import get_item_ids, get_item_ids_stage
from darwin.future.core.items.get import get_item, list_folders, list_items
from darwin.future.core.types.common import QueryString
from darwin.future.data_objects.item import Folder, Item
from darwin.future.tests.core.fixtures import *
from darwin.future.tests.core.items.fixtures import *


def test_get_item_ids(
    UUIDs: List[UUID], UUIDs_str: List[str], base_client: ClientCore
) -> None:
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            base_client.config.api_endpoint + "v2/teams/default-team/items/ids"
            "?not_statuses=archived,error&sort[id]=desc&dataset_ids=1337",
            json={"item_ids": UUIDs_str},
            status=200,
        )
        item_ids = get_item_ids(base_client, "default-team", "1337")
        assert item_ids == UUIDs


def test_get_item_ids_stage(
    UUIDs: List[UUID], UUIDs_str: List[str], base_client: ClientCore
) -> None:
    stage_id = str(uuid4())
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            base_client.config.api_endpoint + "v2/teams/default-team/items/ids"
            f"?workflow_stage_ids={stage_id}&dataset_ids=1337",
            json={"item_ids": UUIDs_str},
            status=200,
        )
        item_ids = get_item_ids_stage(base_client, "default-team", "1337", stage_id)
        assert item_ids == UUIDs


def test_get_item(
    base_items_json: List[dict], base_items: List[Item], base_client: ClientCore
) -> None:
    uuid = str(base_items[0].id)
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            base_client.config.api_endpoint + f"v2/teams/default-team/items/{uuid}",
            json=base_items_json[0],
            status=200,
        )
        item = get_item(base_client, "default-team", uuid)
        assert item == base_items[0]


def test_list_items(
    base_items_json: List[dict], base_items: List[Item], base_client: ClientCore
) -> None:
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            base_client.config.api_endpoint
            + "v2/teams/default-team/items?dataset_ids=1337",
            json={"items": base_items_json},
            status=200,
        )
        items = list_items(
            base_client, "default-team", QueryString({"dataset_ids": "1337"})
        )
        for item, comparator in zip(items, base_items):
            assert item == comparator


def test_list_folders(
    base_folders_json: List[dict], base_folders: List[Folder], base_client: ClientCore
) -> None:
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            base_client.config.api_endpoint
            + "v2/teams/default-team/items/folders?dataset_ids=1337",
            json={"folders": base_folders_json},
            status=200,
        )
        folders = list_folders(
            base_client, "default-team", QueryString({"dataset_ids": "1337"})
        )
        for folder, comparator in zip(folders, base_folders):
            assert folder == comparator
