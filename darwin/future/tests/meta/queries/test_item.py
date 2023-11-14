from typing import List

import pytest
import responses
from responses.matchers import json_params_matcher, query_param_matcher

from darwin.future.core.client import ClientCore
from darwin.future.meta.objects.item import Item
from darwin.future.meta.queries.item import ItemQuery
from darwin.future.tests.core.fixtures import *
from darwin.future.tests.meta.fixtures import *
from darwin.future.tests.meta.objects.fixtures import *


@pytest.fixture
def item_query(base_client: ClientCore) -> ItemQuery:
    return ItemQuery(
        client=base_client, meta_params={"team_slug": "test", "dataset_id": 1}
    )


def test_item_query_collect(item_query: ItemQuery, items_json: List[dict]) -> None:
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            item_query.client.config.api_endpoint + "v2/teams/test/items",
            match=[
                query_param_matcher(
                    {"page[offset]": "0", "page[size]": "500", "dataset_ids": "1"}
                )
            ],
            json={"items": items_json, "errors": []},
        )
        items = item_query.collect_all()
        assert len(items) == 5
        for i in range(5):
            assert items[i].name == f"item_{i}"


def test_delete(
    item_query: ItemQuery, items_json: List[dict], items: List[Item]
) -> None:
    with responses.RequestsMock() as rsps:
        rsps.add(
            rsps.GET,
            item_query.client.config.api_endpoint + "v2/teams/test/items",
            match=[
                query_param_matcher(
                    {"page[offset]": "0", "page[size]": "500", "dataset_ids": "1"}
                )
            ],
            json={"items": items_json, "errors": []},
        )
        team_slug = items[0].meta_params["team_slug"]
        dataset_id = items[0].meta_params["dataset_id"]
        rsps.add(
            rsps.DELETE,
            items[0].client.config.api_endpoint + f"v2/teams/{team_slug}/items",
            status=200,
            match=[
                json_params_matcher(
                    {
                        "filters": {
                            "item_ids": [str(item.id) for item in items],
                            "dataset_ids": [dataset_id],
                        }
                    }
                )
            ],
            json={},
        )
        item_query.delete()
