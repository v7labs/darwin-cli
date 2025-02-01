from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib import parse

from darwin.datatypes import ItemId


def inject_default_team_slug(method: Callable) -> Callable:
    """
    Injects team_slug if not specified
    """

    def wrapper(self, *args, **kwargs) -> Callable:
        if "team_slug" not in kwargs:
            kwargs["team_slug"] = self._default_team
        return method(self, *args, **kwargs)

    return wrapper


class BackendV2:
    def __init__(self, client: "Client", default_team):  # noqa F821
        self._client = client
        self._default_team = default_team

    @inject_default_team_slug
    def register_data(
        self,
        dataset_slug: str,
        payload: Dict[str, Any],
        *,
        team_slug: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload["dataset_slug"] = dataset_slug
        response = self._client._post(
            endpoint=f"v2/teams/{team_slug}/items/register_upload",
            payload=payload,
            team_slug=team_slug,
        )
        return response

    @inject_default_team_slug
    def sign_upload(
        self, dataset_slug: str, upload_id: str, *, team_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._client._get(
            f"v2/teams/{team_slug}/items/uploads/{upload_id}/sign", team_slug=team_slug
        )

    @inject_default_team_slug
    def confirm_upload(
        self, dataset_slug: str, upload_id: str, *, team_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._client._post(
            f"v2/teams/{team_slug}/items/uploads/{upload_id}/confirm",
            payload={},
            team_slug=team_slug,
        )

    @inject_default_team_slug
    def fetch_items(
        self,
        dataset_id: int,
        cursor: Union[Dict[str, Any], List[Tuple[str, Any]]],
        *,
        team_slug: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch the remote items from the given dataset.

        Parameters
        ----------
        dataset_id: int
            Id of the dataset the file belong to.
        cursor: Dict[str, Any]
            Number of items per page and page number. Defaults to {"page[size]": 500, "page[from]": 0}.
        payload: Dict[str, Any]
            Filter and sort parameters.
        team_slug: str
            The team slug of the dataset.

        Returns
        -------
         Dict[str, Any]
            A response dictionary with the file information.
        """
        if isinstance(cursor, dict):
            cursor = list(cursor.items())

        cursor.append(("dataset_ids[]", dataset_id))

        return self._client._get(
            f"/v2/teams/{team_slug}/items?{parse.urlencode(cursor, True)}", team_slug
        )

    @inject_default_team_slug
    def archive_items(
        self, payload: Dict[str, Any], *, team_slug: Optional[str] = None
    ) -> None:
        """
        Archives the item from the given dataset.

        Parameters
        ----------
        team_slug: str
            The slug of the team.
        payload: Dict[str, Any]
            A filter Dictionary that defines the items to be archived.
        """
        self._client._post(f"v2/teams/{team_slug}/items/archive", payload, team_slug)

    @inject_default_team_slug
    def restore_archived_items(
        self, payload: Dict[str, Any], *, team_slug: Optional[str] = None
    ) -> None:
        """
        Restores the archived item from the given dataset.

        Parameters
        ----------
        team_slug: str
            The slug of the team.
        payload: Dict[str, Any]
            A filter Dictionary that defines the items to be restored.
        """
        self._client._post(f"v2/teams/{team_slug}/items/restore", payload, team_slug)

    @inject_default_team_slug
    def move_to_stage(
        self,
        filters: Dict[str, Any],
        stage_id: str,
        workflow_id: str,
        *,
        team_slug: Optional[str] = None,
    ) -> None:
        """
        Moves the given items to the specified stage

        Parameters
        ----------
        dataset_slug: str
            The slug of the dataset.
        team_slug: str
            The slug of the team.
        payload: Dict[str, Any]
            A filter Dictionary that defines the items to have the 'new' status.
        """
        payload = {"filters": filters, "stage_id": stage_id, "workflow_id": workflow_id}
        self._client._post_raw(f"v2/teams/{team_slug}/items/stage", payload, team_slug)

    @inject_default_team_slug
    def get_dataset(
        self, id: str, *, team_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._client._get(f"datasets/{id}", team_slug)

    @inject_default_team_slug
    def get_workflow(
        self, id: str, *, team_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._client._get(f"v2/teams/{team_slug}/workflows/{id}", team_slug)

    @inject_default_team_slug
    def delete_items(self, filters, *, team_slug: Optional[str] = None):
        self._client._delete(
            f"v2/teams/{team_slug}/items", {"filters": filters}, team_slug
        )

    @inject_default_team_slug
    def export_dataset(
        self,
        name,
        format,
        include_authorship,
        include_token,
        dataset_slug,
        filters,
        annotation_class_ids,
        *,
        team_slug: Optional[str] = None,
    ):
        payload = {
            "include_authorship": include_authorship,
            "include_export_token": include_token,
            "name": name,
            "annotation_filters": {},
        }
        if format:
            payload["format"] = format

        if annotation_class_ids:
            payload["annotation_filters"] = {
                "annotation_class_ids": list(map(int, annotation_class_ids))
            }
        if filters is not None:
            # Backend assumes default filters only if those are completely missing.
            payload["filters"] = filters

        return self._client._post(
            f"v2/teams/{team_slug}/datasets/{dataset_slug}/exports", payload, team_slug
        )

    def get_exports(self, dataset_slug, *, team_slug: Optional[str] = None):
        return self._client._get(
            f"v2/teams/{team_slug}/datasets/{dataset_slug}/exports", team_slug
        )

    @inject_default_team_slug
    def post_comment(
        self, item_id, text, x, y, w, h, slot_name, team_slug: Optional[str] = None
    ):
        payload = {
            "bounding_box": {"h": h, "w": w, "x": x, "y": y},
            "comments": [{"body": text}],
            "slot_name": slot_name,
        }
        return self._client._post(
            f"v2/teams/{team_slug}/items/{item_id}/comment_threads", payload, team_slug
        )

    @inject_default_team_slug
    def import_annotation(
        self, item_id: ItemId, payload: Dict[str, Any], team_slug: Optional[str] = None
    ) -> None:
        """
        Imports the annotation for the item with the given id.

        Parameters
        ----------
        item_id: ItemId
            Identifier of the Item that we are import the annotation to.
        payload: Dict[str, Any]
            A dictionary with the annotation to import. The default format is:
            `{"annotations": serialized_annotations, "overwrite": "false"}`
        """

        return self._client._post_raw(
            f"v2/teams/{team_slug}/items/{item_id}/import", payload=payload
        )

    @inject_default_team_slug
    def register_items(self, payload: Dict[str, Any], team_slug: str) -> None:
        """
        Register items from external storage.

        Parameters
        ----------
        payload: JSONDict
            The payload to register items from external storage.
        team_slug: str
            The team slug.
        """
        return self._client._post_raw(
            f"/v2/teams/{team_slug}/items/register_existing", payload
        )

    def _get_remote_annotations(
        self,
        item_id: str,
        team_slug: str,
    ) -> List:
        """
        Returns the annotations currently present on a remote dataset item.

        Parameters
        ----------
        item_id: str
            The UUID of the item.
        team_slug: str
            The team slug.
        """
        return self._client._get(f"v2/teams/{team_slug}/items/{item_id}/annotations")

    def _get_properties_state_for_item(
        self, item_id: str, team_slug: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Returns the state of property values for the specified item.

        Parameters
        ----------
        item_id: str
            The ID of the item to get properties for.
        team_slug: str
            The slug of the team to get.
        """
        return self._client._get(f"/v2/teams/{team_slug}/items/{item_id}/properties")
