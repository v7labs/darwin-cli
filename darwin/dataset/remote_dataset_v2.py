import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pydantic import ValidationError
from requests.models import Response

from darwin.dataset import RemoteDataset
from darwin.dataset.release import Release
from darwin.dataset.upload_manager import (
    FileUploadCallback,
    ItemMergeMode,
    LocalFile,
    MultiFileItem,
    ProgressCallback,
    UploadHandler,
    UploadHandlerV2,
)
from darwin.dataset.utils import (
    chunk_items,
    get_external_file_name,
    get_external_file_type,
    is_relative_to,
    parse_external_file_path,
)
from darwin.datatypes import (
    AnnotationFile,
    ItemId,
    ObjectStore,
    PathLike,
    StorageKeyDictModel,
    StorageKeyListModel,
)
from darwin.exceptions import NotFound, UnknownExportVersion
from darwin.exporter.formats.darwin import build_image_annotation
from darwin.item import DatasetItem
from darwin.item_sorter import ItemSorter
from darwin.utils import (
    SUPPORTED_EXTENSIONS,
    PRESERVE_FOLDERS_KEY,
    AS_FRAMES_KEY,
    EXTRACT_VIEWS_KEY,
    find_files,
    urljoin,
)

if TYPE_CHECKING:
    from darwin.client import Client


class RemoteDatasetV2(RemoteDataset):
    """
    Manages the remote and local versions of a dataset hosted on Darwin.
    It allows several dataset management operations such as syncing between
    remote and local, pulling a remote dataset, removing the local files, ...

    Parameters
    ----------
    client : Client
        Client to use for interaction with the server.
    team : str
        Team the dataset belongs to.
    name : str
        Name of the datasets as originally displayed on Darwin.
        It may contain white spaces, capital letters and special characters, e.g. `Bird Species!`.
    slug : str
        This is the dataset name with everything lower-case, removed specials characters and
        spaces are replaced by dashes, e.g., `bird-species`. This string is unique within a team.
    dataset_id : int
        Unique internal reference from the Darwin backend.
    item_count : int, default: 0
        Dataset size (number of items).
    progress : float, default: 0
        How much of the dataset has been annotated 0.0 to 1.0 (1.0 == 100%).

    Attributes
    ----------
    client : Client
        Client to use for interaction with the server.
    team : str
        Team the dataset belongs to.
    name : str
        Name of the datasets as originally displayed on Darwin.
        It may contain white spaces, capital letters and special characters, e.g. `Bird Species!`.
    slug : str
        This is the dataset name with everything lower-case, removed specials characters and
        spaces are replaced by dashes, e.g., `bird-species`. This string is unique within a team.
    dataset_id : int
        Unique internal reference from the Darwin backend.
    item_count : int, default: 0
        Dataset size (number of items).
    progress : float, default: 0
        How much of the dataset has been annotated 0.0 to 1.0 (1.0 == 100%).
    """

    def __init__(
        self,
        *,
        client: "Client",
        team: str,
        name: str,
        slug: str,
        dataset_id: int,
        item_count: int = 0,
        progress: float = 0,
    ):
        super().__init__(
            client=client,
            team=team,
            name=name,
            slug=slug,
            dataset_id=dataset_id,
            item_count=item_count,
            progress=progress,
            version=2,
        )

    def get_releases(self, include_unavailable: bool = False) -> List["Release"]:
        """
        Get a sorted list of releases with the most recent first.

        Parameters
        ----------
        include_unavailable : bool, default: False
            If True, return  all releases, including those that are not available.

        Returns
        -------
        List["Release"]
            Returns a sorted list of available ``Release``\\s with the most recent first.
        """
        try:
            releases_json: List[Dict[str, Any]] = self.client.api_v2.get_exports(
                self.slug, team_slug=self.team
            )
        except NotFound:
            return []

        releases = [
            Release.parse_json(self.slug, self.team, payload)
            for payload in releases_json
        ]

        return sorted(
            (
                releases
                if include_unavailable
                else filter(lambda x: x.available, releases)
            ),
            key=lambda x: x.version,
            reverse=True,
        )

    def push(
        self,
        files_to_upload: Optional[Sequence[Union[PathLike, LocalFile]]],
        *,
        blocking: bool = True,
        multi_threaded: bool = True,
        max_workers: Optional[int] = None,
        fps: int = 0,
        as_frames: bool = False,
        extract_views: bool = False,
        handle_as_slices: Optional[bool] = False,
        files_to_exclude: Optional[List[PathLike]] = None,
        path: Optional[str] = None,
        preserve_folders: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        file_upload_callback: Optional[FileUploadCallback] = None,
        item_merge_mode: Optional[str] = None,
    ) -> UploadHandler:
        """
        Uploads a local dataset (images ONLY) in the datasets directory.

        Parameters
        ----------
        files_to_upload : Optional[List[Union[PathLike, LocalFile]]]
            List of files to upload. These can be folders.
            If `item_merge_mode` is set, these paths must be folders.
        blocking : bool, default: True
            If False, the dataset is not uploaded and a generator function is returned instead.
        multi_threaded : bool, default: True
            Uses multiprocessing to upload the dataset in parallel.
            If blocking is False this has no effect.
        max_workers : int, default: None
            Maximum number of workers to use for parallel upload.
        fps : int, default: 0
            When the uploading file is a video, specify its framerate.
        as_frames: bool, default: False
            When the uploading file is a video, specify whether it's going to be uploaded as a list of frames.
        extract_views: bool, default: False
            When the uploading file is a volume, specify whether it's going to be split into orthogonal views.
        handle_as_slices: Optioonal[bool], default: False
            Whether to upload DICOM files as slices
        files_to_exclude : Optional[PathLike]], default: None
            Optional list of files to exclude from the file scan. These can be folders.
        path: Optional[str], default: None
            Optional path to store the files in.
        preserve_folders : bool, default: False
            Specify whether or not to preserve folder paths when uploading
        progress_callback: Optional[ProgressCallback], default: None
            Optional callback, called every time the progress of an uploading files is reported.
        file_upload_callback: Optional[FileUploadCallback], default: None
            Optional callback, called every time a file chunk is uploaded.
        item_merge_mode : Optional[str]
            If set, each file path passed to `files_to_upload` behaves as follows:
            - Paths pointing directly to individual files are ignored
            - Paths pointing to folders of files will be uploaded according to the following mode rules.
            Note that folders will not be recursively searched, so only files in the first level of the folder will be uploaded:
                - "slots": Each file in the folder will be uploaded to a different slot of the same item.
                - "series": All `.dcm` files in the folder will be concatenated into a single slot. All other files are ignored.
                - "channels": Each file in the folder will be uploaded to a different channel of the same item.
        Returns
        -------
        handler : UploadHandler
            Class for handling uploads, progress and error messages.

        Raises
        ------
        ValueError
            - If ``files_to_upload`` is ``None``.
            - If a path is specified when uploading a LocalFile object.
            - If there are no files to upload (because path is wrong or the exclude filter excludes everything).
        """
        merge_incompatible_args = {
            PRESERVE_FOLDERS_KEY: preserve_folders,
            AS_FRAMES_KEY: as_frames,
            EXTRACT_VIEWS_KEY: extract_views,
        }

        if files_to_exclude is None:
            files_to_exclude = []

        if files_to_upload is None:
            raise ValueError("No files or directory specified.")

        if item_merge_mode:
            try:
                ItemMergeMode(item_merge_mode)
            except ValueError:
                raise ValueError(
                    f"Invalid item merge mode: {item_merge_mode}. Valid options are: 'slots', 'series', 'channels'"
                )
            incompatible_args = [
                arg for arg, value in merge_incompatible_args.items() if value
            ]

            if incompatible_args:
                incompatible_args_str = ", ".join(incompatible_args)
                raise TypeError(
                    f"`item_merge_mode` does not support the following incompatible arguments: {incompatible_args_str}."
                )

        # Folder paths
        search_files = [
            item for item in files_to_upload if not isinstance(item, LocalFile)
        ]

        if item_merge_mode:
            local_files, multi_file_items = _find_files_to_upload_as_multi_file_items(
                search_files, files_to_exclude, fps, item_merge_mode
            )
            handler = UploadHandlerV2(
                self, local_files, multi_file_items, handle_as_slices=handle_as_slices
            )
        else:
            local_files = _find_files_to_upload_as_single_file_items(
                search_files,
                files_to_upload,
                files_to_exclude,
                path,
                fps,
                as_frames,
                extract_views,
                preserve_folders,
            )
            handler = UploadHandlerV2(
                self, local_files, handle_as_slices=handle_as_slices
            )
        if blocking:
            handler.upload(
                max_workers=max_workers,
                multi_threaded=multi_threaded,
                progress_callback=progress_callback,
                file_upload_callback=file_upload_callback,
            )
        else:
            handler.prepare_upload()

        return handler

    def fetch_remote_files(
        self,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        sort: Optional[Union[str, ItemSorter]] = None,
    ) -> Iterator[DatasetItem]:
        """
        Fetch and lists all files on the remote dataset.

        Parameters
        ----------
        filters : Optional[Dict[str, Union[str, List[str]]]], default: None
            The filters to use. Files excluded by the filter won't be fetched.
        sort : Optional[Union[str, ItemSorter]], default: None
            A sorting direction. It can be a string with the values 'asc', 'ascending', 'desc',
            'descending' or an ``ItemSorter`` instance.

        Yields
        -------
        Iterator[DatasetItem]
            An iterator of ``DatasetItem``.
        """
        post_filters: List[Tuple[str, Any]] = []
        post_sort: Dict[str, str] = {}

        if filters:
            # Backward compatibility with V1 filter parameter
            if "filenames" in filters:
                filters["item_names"] = filters["filenames"]
                del filters["filenames"]

            for list_type in [
                "item_names",
                "statuses",
                "item_ids",
                "slot_types",
                "item_paths",
            ]:
                if list_type in filters:
                    if isinstance(filters[list_type], list):
                        for value in filters[list_type]:
                            post_filters.append(("{}[]".format(list_type), value))
                    else:
                        post_filters.append((list_type, str(filters[list_type])))

        if sort:
            item_sorter = ItemSorter.parse(sort)
            post_sort[f"sort[{item_sorter.field}]"] = item_sorter.direction.value
        cursor = {"page[size]": 500, "include_workflow_data": "true"}
        while True:
            query = post_filters + list(post_sort.items()) + list(cursor.items())
            response = self.client.api_v2.fetch_items(
                self.dataset_id, query, team_slug=self.team
            )
            yield from [
                DatasetItem.parse(item, dataset_slug=self.slug)
                for item in response["items"]
            ]

            if response["page"]["next"]:
                cursor["page[from]"] = response["page"]["next"]
            else:
                return

    def archive(self, items: Iterator[DatasetItem]) -> None:
        """
        Archives (soft-deletion) the given ``DatasetItem``\\s belonging to this ``RemoteDataset``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be archived.
        """
        payload: Dict[str, Any] = {
            "filters": {
                "item_ids": [item.id for item in items],
                "dataset_ids": [self.dataset_id],
            }
        }
        self.client.api_v2.archive_items(payload, team_slug=self.team)

    def restore_archived(self, items: Iterator[DatasetItem]) -> None:
        """
        Restores the archived ``DatasetItem``\\s that belong to this ``RemoteDataset``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be restored.
        """
        payload: Dict[str, Any] = {
            "filters": {
                "item_ids": [item.id for item in items],
                "dataset_ids": [self.dataset_id],
            }
        }
        self.client.api_v2.restore_archived_items(payload, team_slug=self.team)

    def move_to_new(self, items: Iterator[DatasetItem]) -> None:
        """
        Changes the given ``DatasetItem``\\s status to ``new``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s whose status will change.
        """

        (workflow_id, stages) = self._fetch_stages("dataset")
        if not stages:
            raise ValueError("Dataset's workflow is missing a dataset stage")

        self.client.api_v2.move_to_stage(
            {"item_ids": [item.id for item in items], "dataset_ids": [self.dataset_id]},
            stages[0]["id"],
            workflow_id,
            team_slug=self.team,
        )

    def reset(self, items: Iterator[DatasetItem]) -> None:
        """
        Deprecated
        Resets the  given ``DatasetItem``\\s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be resetted.
        """
        raise ValueError("Reset is deprecated for version 2 datasets")

    def complete(self, items: Iterator[DatasetItem]) -> None:
        """
        Completes the given ``DatasetItem``\\s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be completed.
        """
        (workflow_id, stages) = self._fetch_stages("complete")
        if not stages:
            raise ValueError("Dataset's workflow is missing a complete stage")

        self.client.api_v2.move_to_stage(
            {"item_ids": [item.id for item in items], "dataset_ids": [self.dataset_id]},
            stages[0]["id"],
            workflow_id,
            team_slug=self.team,
        )

    def delete_items(self, items: Iterator[DatasetItem]) -> None:
        """
        Deletes the given ``DatasetItem``\\s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be deleted.
        """
        self.client.api_v2.delete_items(
            {"dataset_ids": [self.dataset_id], "item_ids": [item.id for item in items]},
            team_slug=self.team,
        )

    def export(
        self,
        name: str,
        annotation_class_ids: Optional[List[str]] = None,
        include_url_token: bool = False,
        include_authorship: bool = False,
        version: Optional[str] = None,
    ) -> None:
        """
        Create a new release for this ``RemoteDataset``.

        Parameters
        ----------
        name : str
            Name of the release.
        annotation_class_ids : Optional[List[str]], default: None
            List of the classes to filter.
        include_url_token : bool, default: False
            Should the image url in the export include a token enabling access without team
            membership or not?
        include_authorship : bool, default: False
            If set, include annotator and reviewer metadata for each annotation.
        version : Optional[str], default: None, enum: ["1.0", "2.0"]
            When used for V2 dataset, allows to force generation of either Darwin JSON 1.0 (Legacy) or newer 2.0.
            Omit this option to get your team's default.
        """
        str_version = str(version)
        if str_version == "2.0":
            format = "darwin_json_2"
        elif str_version == "1.0":
            format = "json"
        elif version is None:
            format = None
        else:
            raise UnknownExportVersion(version)

        filters = (
            None
            if not annotation_class_ids
            else {"annotation_class_ids": list(map(int, annotation_class_ids))}
        )

        self.client.api_v2.export_dataset(
            format=format,
            name=name,
            include_authorship=include_authorship,
            include_token=include_url_token,
            annotation_class_ids=None,
            filters=filters,
            dataset_slug=self.slug,
            team_slug=self.team,
        )

    def get_report(self, granularity: str = "day") -> str:
        """
        Returns a String representation of a CSV report for this ``RemoteDataset``.

        Parameters
        ----------
        granularity : str, default: "day"
            The granularity of the report, can be 'day', 'week' or 'month'.

        Returns
        -------
        str
            A CSV report.
        """
        response: Response = self.client.get_report(
            self.dataset_id, granularity, self.team
        )
        return response.text

    def workview_url_for_item(self, item: DatasetItem) -> str:
        """
        Returns the darwin URL for the given ``DatasetItem``.

        Parameters
        ----------
        item : DatasetItem
            The ``DatasetItem`` for which we want the url.

        Returns
        -------
        str
            The url.
        """
        return urljoin(
            self.client.base_url, f"/workview?dataset={self.dataset_id}&item={item.id}"
        )

    def post_comment(
        self,
        item: DatasetItem,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        slot_name: Optional[str] = None,
    ):
        """
        Adds a comment to an item in this dataset,
        Tries to infer slot_name if left out.
        """
        if not slot_name:
            if len(item.slots) != 1:
                raise ValueError(
                    f"Unable to infer slot for '{item.id}', has multiple slots: {','.join(item.slots)}"
                )
            slot_name = item.slots[0]["slot_name"]

        self.client.api_v2.post_comment(
            item.id, text, x, y, w, h, slot_name, team_slug=self.team
        )

    def import_annotation(self, item_id: ItemId, payload: Dict[str, Any]) -> None:
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

        self.client.api_v2.import_annotation(
            item_id, payload=payload, team_slug=self.team
        )

    def _fetch_stages(self, stage_type):
        detailed_dataset = self.client.api_v2.get_dataset(self.dataset_id)
        workflow_ids = detailed_dataset["workflow_ids"]
        if len(workflow_ids) == 0:
            raise ValueError("Dataset is not part of a workflow")
        # currently we can only be part of one workflow
        workflow_id = workflow_ids[0]
        workflow = self.client.api_v2.get_workflow(workflow_id, team_slug=self.team)
        return (
            workflow_id,
            [stage for stage in workflow["stages"] if stage["type"] == stage_type],
        )

    def _build_image_annotation(
        self, annotation_file: AnnotationFile, team_name: str
    ) -> Dict[str, Any]:
        return build_image_annotation(annotation_file, team_name)

    def register(
        self,
        object_store: ObjectStore,
        storage_keys: Union[List[str], Dict[str, List[str]]],
        fps: Optional[Union[str, float]] = None,
        multi_planar_view: bool = False,
        preserve_folders: bool = False,
        multi_slotted: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Register files from external storage in a Darwin dataset.

        Parameters
        ----------
        object_store : ObjectStore
            Object store to use for the registration.
        storage_keys : List[str] | Dict[str, List[str]]
            Either:
            - Single-slotted items: A list of storage keys
            - Multi-slotted items: A dictionary with keys as item names and values as lists of storage keys
        fps : Optional[str], default: None
            When the uploading file is a video, specify its framerate.
        multi_planar_view : bool, default: False
            Uses multiplanar view when uploading files.
        preserve_folders : bool, default: False
            Specify whether or not to preserve folder paths when uploading.
        multi_slotted : bool, default: False
            Specify whether the items are multi-slotted or not.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary with the list of registered files.

        Raises
        ------
        ValueError
            If the type of ``storage_keys``:
            - Isn't List[str] when ``multi_slotted`` is False.
            - Isn't Dict[str, List[str]] when ``multi_slotted`` is True.
        """
        if multi_slotted:
            try:
                StorageKeyDictModel(storage_keys=storage_keys)  # type: ignore
            except ValidationError as e:
                print(
                    f"Error validating storage keys: {e}\n\nPlease make sure your storage keys are a list of strings"
                )
                raise e
            results = self.register_multi_slotted(
                object_store,
                storage_keys,  # type: ignore
                fps,
                multi_planar_view,
                preserve_folders,
            )
            return results
        else:
            try:
                StorageKeyListModel(storage_keys=storage_keys)  # type: ignore
            except ValidationError as e:
                print(
                    f"Error validating storage keys: {e}\n\nPlease make sure your storage keys are a dictionary with keys as item names and values as lists of storage keys"
                )
                raise e
            results = self.register_single_slotted(
                object_store,
                storage_keys,  # type: ignore
                fps,
                multi_planar_view,
                preserve_folders,
            )
            return results

    def register_single_slotted(
        self,
        object_store: ObjectStore,
        storage_keys: List[str],
        fps: Optional[Union[str, float]] = None,
        multi_planar_view: bool = False,
        preserve_folders: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Register files in the dataset in a single slot.

        Parameters
        ----------
        object_store : ObjectStore
            Object store to use for the registration.
        storage_keys : List[str]
            List of storage keys to register.
        fps : Optional[str], default: None
            When the uploading file is a video, specify its framerate.
        multi_planar_view : bool, default: False
            Uses multiplanar view when uploading files.
        preserve_folders : bool, default: False
            Specify whether or not to preserve folder paths when uploading

        Returns
        -------
        Dict[str, List[str]]
            A dictionary with the list of registered files.

        Raises
        ------
        TypeError
            If the file type of any storage keyis not supported.
        """
        items = []
        for storage_key in storage_keys:
            file_type = get_external_file_type(storage_key)
            if not file_type:
                raise TypeError(
                    f"Unsupported file type for the following storage key: {storage_key}.\nPlease make sure your storage key ends with one of the supported extensions:\n{SUPPORTED_EXTENSIONS}"
                )
            item = {
                "path": parse_external_file_path(storage_key, preserve_folders),
                "type": file_type,
                "storage_key": storage_key,
                "name": (
                    storage_key.split("/")[-1] if "/" in storage_key else storage_key
                ),
            }
            if fps and file_type == "video":
                item["fps"] = fps
            if multi_planar_view and file_type == "dicom":
                item["extract_views"] = "true"
            items.append(item)

        # Do not register more than 10 items in a single request
        chunk_size = 10
        chunked_items = chunk_items(items, chunk_size)
        print(f"Registering {len(items)} items in chunks of {chunk_size} items...")
        results = {
            "registered": [],
            "blocked": [],
        }

        for chunk in chunked_items:
            payload = {
                "items": chunk,
                "dataset_slug": self.slug,
                "storage_slug": object_store.name,
            }
            print(f"Registering {len(chunk)} items...")
            response = self.client.api_v2.register_items(payload, team_slug=self.team)
            for item in json.loads(response.text)["items"]:
                item_info = f"Item {item['name']} registered with item ID {item['id']}"
                results["registered"].append(item_info)
            for item in json.loads(response.text)["blocked_items"]:
                item_info = f"Item {item['name']} was blocked for the reason: {item['slots'][0]['reason']}"
                results["blocked"].append(item_info)
        print(
            f"{len(results['registered'])} of {len(storage_keys)} items registered successfully"
        )
        if results["blocked"]:
            print("The following items were blocked:")
            for item in results["blocked"]:
                print(f"  - {item}")
        print(f"Reistration complete. Check your items in the dataset: {self.slug}")
        return results

    def register_multi_slotted(
        self,
        object_store: ObjectStore,
        storage_keys: Dict[str, List[str]],
        fps: Optional[Union[str, float]] = None,
        multi_planar_view: bool = False,
        preserve_folders: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Register files in the dataset in multiple slots.

        Parameters
        ----------
        object_store : ObjectStore
            Object store to use for the registration.
        storage_keys : Dict[str, List[str]
            Storage keys to register. The keys are the item names and the values are lists of storage keys.
        fps : Optional[str], default: None
            When the uploading file is a video, specify its framerate.
        multi_planar_view : bool, default: False
            Uses multiplanar view when uploading files.
        preserve_folders : bool, default: False
            Specify whether or not to preserve folder paths when uploading

        Returns
        -------
        Dict[str, List[str]]
            A dictionary with the list of registered files.

        Raises
        ------
        TypeError
            If the file type of any storage key is not supported.
        """
        items = []
        for item in storage_keys:
            slots = []
            for storage_key in storage_keys[item]:
                file_name = get_external_file_name(storage_key)
                file_type = get_external_file_type(storage_key)
                if not file_type:
                    raise TypeError(
                        f"Unsupported file type for the following storage key: {storage_key}.\nPlease make sure your storage key ends with one of the supported extensions:\n{SUPPORTED_EXTENSIONS}"
                    )
                slot = {
                    "slot_name": file_name,
                    "type": file_type,
                    "storage_key": storage_key,
                    "file_name": file_name,
                }
                if fps and file_type == "video":
                    slot["fps"] = fps
                if multi_planar_view and file_type == "dicom":
                    slot["extract_views"] = "true"
                slots.append(slot)
            items.append(
                {
                    "slots": slots,
                    "name": item,
                    "path": parse_external_file_path(
                        storage_keys[item][0], preserve_folders
                    ),
                }
            )

        # Do not register more than 10 items in a single request
        chunk_size = 10
        chunked_items = chunk_items(items, chunk_size)
        print(f"Registering {len(items)} items in chunks of {chunk_size} items...")
        results = {
            "registered": [],
            "blocked": [],
        }

        for chunk in chunked_items:
            payload = {
                "items": chunk,
                "dataset_slug": self.slug,
                "storage_slug": object_store.name,
            }
            print(f"Registering {len(chunk)} items...")
            response = self.client.api_v2.register_items(payload, team_slug=self.team)
            for item in json.loads(response.text)["items"]:
                item_info = f"Item {item['name']} registered with item ID {item['id']}"
                results["registered"].append(item_info)
            for item in json.loads(response.text)["blocked_items"]:
                item_info = f"Item {item['name']} was blocked for the reason: {item['slots'][0]['reason']}"
                results["blocked"].append(item_info)
        print(
            f"{len(results['registered'])} of {len(storage_keys)} items registered successfully"
        )
        if results["blocked"]:
            print("The following items were blocked:")
            for item in results["blocked"]:
                print(f"  - {item}")
        print(f"Reistration complete. Check your items in the dataset: {self.slug}")
        return results


def _find_files_to_upload_as_multi_file_items(
    search_files: List[PathLike],
    files_to_exclude: List[PathLike],
    fps: int,
    item_merge_mode: str,
) -> Tuple[List[LocalFile], List[MultiFileItem]]:
    """
    Finds files to upload according to the `item_merge_mode`.
    Does not search each directory recursively, only considers files in the first level of each directory.

    Parameters
    ----------
    search_files : List[PathLike]
        List of directories to search for files.
    files_to_exclude : List[PathLike]
        List of files to exclude from the file scan.
    fps : int
        When uploading video files, specify the framerate
    item_merge_mode : str
        Mode to merge the files in the folders. Valid options are: 'slots', 'series', 'channels'.

    Returns
    -------
    List[LocalFile]
        List of `LocalFile` objects contained within each `MultiFileItem`
    List[MultiFileItem]
        List of `MultiFileItem` objects to be uploaded
    """
    multi_file_items, local_files = [], []
    for directory in search_files:
        files_in_directory = list(
            find_files(
                [directory],
                files_to_exclude=files_to_exclude,
                recursive=False,
                sort=True,
            )
        )
        if not files_in_directory:
            print(
                f"Warning: There are no files in the first level of {directory}, skipping directory"
            )
            continue
        multi_file_item = MultiFileItem(
            Path(directory), files_in_directory, ItemMergeMode(item_merge_mode), fps
        )
        multi_file_items.append(multi_file_item)
        local_files.extend(multi_file_item.files)

    if not multi_file_items:
        raise ValueError(
            "No valid folders to upload after searching the passed directories for files"
        )
    return local_files, multi_file_items


def _find_files_to_upload_as_single_file_items(
    search_files: List[PathLike],
    files_to_upload: Optional[Sequence[Union[PathLike, LocalFile]]],
    files_to_exclude: List[PathLike],
    path: Optional[str],
    fps: int,
    as_frames: bool,
    extract_views: bool,
    preserve_folders: bool,
) -> List[LocalFile]:
    """
    Finds files to upload as single-slotted dataset items. Recursively searches the passed directories for files.

    Parameters
    ----------
    search_files : List[PathLike]
        List of directories to search for files.

    files_to_exclude : Optional[List[PathLike]]
        List of files to exclude from the file scan.
    files_to_upload : Optional[List[Union[PathLike, LocalFile]]]
        List of files to upload. These can be folders.
    path : Optional[str]
        Path to store the files in.
    fps: int
        When uploading video files, specify the framerate.
    as_frames: bool
        When uploading video files, specify whether to upload as a list of frames.
    extract_views: bool
        When uploading volume files, specify whether to split into orthogonal views.
    preserve_folders: bool
        Specify whether or not to preserve folder paths when uploading.

    Returns
    -------
    List[LocalFile]
        List of files to upload.
    """
    # Direct file paths
    uploading_files = [item for item in files_to_upload if isinstance(item, LocalFile)]

    generic_parameters_specified = (
        path is not None or fps != 0 or as_frames is not False
    )

    if (
        any(isinstance(item, LocalFile) for item in uploading_files)
        and generic_parameters_specified
    ):
        raise ValueError("Cannot specify a path when uploading a LocalFile object.")

    for found_file in find_files(search_files, files_to_exclude=files_to_exclude):
        local_path = path
        if preserve_folders:
            source_files = [
                source_file
                for source_file in search_files
                if is_relative_to(found_file, source_file)
            ]
            if source_files:
                local_path = str(
                    found_file.relative_to(source_files[0]).parent.as_posix()
                )
                if local_path == ".":
                    local_path = "/"
        uploading_files.append(
            LocalFile(
                found_file,
                fps=fps,
                as_frames=as_frames,
                extract_views=extract_views,
                path=local_path,
            )
        )

    if not uploading_files:
        raise ValueError(
            "No files to upload, check your path, exclusion filters and resume flag"
        )

    return uploading_files
