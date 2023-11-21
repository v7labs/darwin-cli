from functools import reduce
from typing import Dict

from darwin.future.core.items.get import get_item_ids
from darwin.future.core.types.common import QueryString
from darwin.future.core.types.query import PaginatedQuery
from darwin.future.meta.objects.v7_id import V7ID


class ItemIDQuery(PaginatedQuery[V7ID]):
    def _collect(self) -> Dict[int, V7ID]:
        if "team_slug" not in self.meta_params:
            raise ValueError("Must specify team_slug to query item ids")
        if (
            "dataset_ids" not in self.meta_params
            and "dataset_id" not in self.meta_params
        ):
            raise ValueError("Must specify dataset_id to query item ids")
        team_slug: str = self.meta_params["team_slug"]
        dataset_ids: int = (
            self.meta_params["dataset_ids"]
            if "dataset_ids" in self.meta_params
            else self.meta_params["dataset_id"]
        )
        params: QueryString = reduce(
            lambda s1, s2: s1 + s2,
            [
                self.page.to_query_string(),
                *[QueryString(f.to_dict()) for f in self.filters],
            ],
        )
        uuids = get_item_ids(self.client, team_slug, dataset_ids, params)

        results = {
            i
            + self.page.offset: V7ID(
                client=self.client, element=uuid, meta_params=self.meta_params
            )
            for i, uuid in enumerate(uuids)
        }
        return results
