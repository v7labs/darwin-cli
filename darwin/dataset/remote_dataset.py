import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import orjson as json
from rich.console import Console

from darwin.dataset.download_manager import download_all_images_from_annotations
from darwin.dataset.identifier import DatasetIdentifier
from darwin.dataset.release import Release
from darwin.dataset.split_manager import split_dataset
from darwin.dataset.upload_manager import (
    FileUploadCallback,
    LocalFile,
    ProgressCallback,
    UploadHandler,
)
from darwin.dataset.utils import (
    exhaust_generator,
    get_annotations,
    get_classes,
    is_unix_like_os,
    make_class_lists,
)
from darwin.datatypes import AnnotationClass, AnnotationFile, ItemId, PathLike
from darwin.exceptions import MissingDependency, NotFound, UnsupportedExportFormat
from darwin.exporter.formats.darwin import build_image_annotation
from darwin.item import DatasetItem
from darwin.item_sorter import ItemSorter
from darwin.utils import parse_darwin_json, split_video_annotation, urljoin

if TYPE_CHECKING:
    from darwin.client import Client

from abc import ABC, abstractmethod


class RemoteDataset(ABC):
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
        version: int = 1,
        release: Optional[str] = None,
    ):
        self.team = team
        self.name = name
        self.slug = slug or name
        self.dataset_id = dataset_id
        self.item_count = item_count
        self.progress = progress
        self.client = client
        self.annotation_types: Optional[List[Dict[str, Any]]] = None
        self.console: Console = Console()
        self.version = version
        self.release = release

    @abstractmethod
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
        files_to_exclude: Optional[List[PathLike]] = None,
        path: Optional[str] = None,
        preserve_folders: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        file_upload_callback: Optional[FileUploadCallback] = None,
        item_merge_mode: Optional[str] = None,
    ) -> UploadHandler:
        pass

    def split_video_annotations(self, release_name: str = "latest") -> None:
        """
        Splits the video annotations from this ``RemoteDataset`` using the given release.

        Parameters
        ----------
        release_name : str, default: "latest"
            The name of the release to use.
        """
        release_dir: Path = self.local_path / "releases" / release_name
        annotations_path: Path = release_dir / "annotations"

        for count, annotation_file in enumerate(annotations_path.glob("*.json")):
            darwin_annotation: Optional[AnnotationFile] = parse_darwin_json(
                annotation_file, count
            )
            if not darwin_annotation or not darwin_annotation.is_video:
                continue

            frame_annotations = split_video_annotation(darwin_annotation)
            for frame_annotation in frame_annotations:
                annotation = self._build_image_annotation(frame_annotation, self.team)

                # When splitting into frames, we need to read each frame individually
                # Because we use the source name suffix, we need to adjust this to .png here
                current_stem = Path(
                    annotation["item"]["slots"][0]["source_files"][0].file_name
                ).stem
                annotation["item"]["slots"][0]["source_files"][0].file_name = (
                    current_stem + ".png"
                )
                # We also need to account for the folder that this function creates
                item_name = annotation["item"]["name"].split("/")[0]
                if annotation["item"]["path"] == "/":
                    annotation["item"]["path"] += item_name
                else:
                    annotation["item"]["path"] += "/" + item_name
                video_frame_annotations_path = annotations_path / annotation_file.stem
                video_frame_annotations_path.mkdir(exist_ok=True, parents=True)

                stem = Path(frame_annotation.filename).stem
                output_path = video_frame_annotations_path / f"{stem}.json"
                with output_path.open("w") as f:
                    op = json.dumps(annotation).decode("utf-8")
                    f.write(op)

            # Finally delete video annotations
            annotation_file.unlink()

        # Update class list, which is used when loading local annotations in a dataset
        make_class_lists(release_dir)

    def pull(
        self,
        *,
        release: Optional[Release] = None,
        blocking: bool = True,
        multi_processed: bool = True,
        only_annotations: bool = False,
        force_replace: bool = False,
        remove_extra: bool = False,
        subset_filter_annotations_function: Optional[Callable] = None,
        subset_folder_name: Optional[str] = None,
        use_folders: bool = True,
        video_frames: bool = False,
        force_slots: bool = False,
        ignore_slots: bool = False,
        retry: bool = False,
        retry_timeout: int = 600,
        retry_interval: int = 10,
    ) -> Tuple[Optional[Callable[[], Iterator[Any]]], int]:
        """
        Downloads a remote dataset (images and annotations) to the datasets directory.

        Parameters
        ----------
        release: Optional[Release], default: None
            The release to pull.
        blocking : bool, default: True
            If False, the dataset is not downloaded and a generator function is returned instead.
        multi_processed : bool, default: True
            Uses multiprocessing to download the dataset in parallel. If blocking is False this has no effect.
        only_annotations : bool, default: False
            Download only the annotations and no corresponding images.
        force_replace : bool, default: False
            Forces the re-download of an existing image.
        remove_extra : bool, default: False
            Removes local files that would not be overwritten by the release being pulled.
        subset_filter_annotations_function: Optional[Callable], default: None
            This function receives the directory where the annotations are downloaded and can
            perform any operation on them i.e. filtering them with custom rules or else.
            If it needs to receive other parameters is advised to use functools.partial() for it.
        subset_folder_name: Optional[str], default: None
            Name of the folder with the subset of the dataset. If not provided a timestamp is used.
        use_folders : bool, default: True
            Recreates folders from the dataset.
        video_frames : bool, default: False
            Pulls video frames images instead of video files.
        force_slots: bool
            Pulls all slots of items into deeper file structure ({prefix}/{item_name}/{slot_name}/{file_name})
        retry: bool
            If True, will repeatedly try to download the release if it is still processing up to a maximum of 5 minutes.

        Returns
        -------
        generator : function
            Generator for doing the actual downloads. This is None if blocking is ``True``.
        count : int
            The number of files.

        Raises
        ------
        UnsupportedExportFormat
            If the given ``release`` has an invalid format.
        ValueError
            If darwin in unable to get ``Team`` configuration.
        ValueError
            If the release is still processing after the maximum retry duration.
        """

        console = self.console or Console()

        if retry and retry_timeout < retry_interval:
            raise ValueError(
                f"The value of retry_timeout '{retry_timeout}' must be greater than or equal to the value of retry_interval '{retry_interval}'."
            )

        if release is None:
            release = self.get_release(include_unavailable=retry)

        if release.format != "json" and release.format != "darwin_json_2":
            raise UnsupportedExportFormat(release.format)

        if release.status.value == "pending":
            if retry:
                while release.status.value == "pending" and retry_timeout > 0:
                    console.print(
                        f"Release '{release.name}' for dataset '{self.name}' is still processing. Retrying in {retry_interval} seconds... {retry_timeout} seconds left before timeout."
                    )
                    time.sleep(retry_interval)
                    retry_timeout -= retry_interval
                    release = self.get_release(release.name, include_unavailable=retry)
                if release.status.value == "pending":
                    raise ValueError(
                        f"Release {release.name} for dataset '{self.name}' is still processing. Please try again later."
                    )
            else:
                raise ValueError(
                    f"Release '{release.name}' for dataset '{self.name}' is still processing. Please wait for it to be ready.\n\n If you would like to automatically retry, set the `retry` parameter to `True` with the SDK, or use the `--retry` flag with the CLI."
                )
        console.print(
            f"Release '{release.name}' for dataset '{self.name}' is ready for download. Starting download..."
        )

        release_dir = self.local_releases_path / release.name
        release_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            # Download the release from Darwin
            zip_file_path = release.download_zip(tmp_dir / "dataset.zip")
            with zipfile.ZipFile(zip_file_path) as z:
                # Extract annotations
                z.extractall(tmp_dir)
                # If a filtering function is provided, apply it
                if subset_filter_annotations_function is not None:
                    subset_filter_annotations_function(tmp_dir)
                    if subset_folder_name is None:
                        subset_folder_name = datetime.now().strftime(
                            "%m/%d/%Y_%H:%M:%S"
                        )
                annotations_dir: Path = (
                    release_dir / (subset_folder_name or "") / "annotations"
                )
                # Remove existing annotations if necessary
                if annotations_dir.exists():
                    try:
                        shutil.rmtree(annotations_dir)
                    except PermissionError:
                        print(
                            f"Could not remove dataset in {annotations_dir}. Permission denied."
                        )
                annotations_dir.mkdir(parents=True, exist_ok=False)
                stems: dict = {}

                # If properties were exported, move the metadata.json file to the annotations folder
                if (tmp_dir / ".v7").exists():
                    metadata_file = tmp_dir / ".v7" / "metadata.json"
                    metadata_dir = annotations_dir / ".v7"
                    metadata_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(metadata_file), str(metadata_dir / "metadata.json"))

                # Move the annotations into the right folder and rename them to have the image
                # original filename as contained in the json
                for annotation_path in tmp_dir.glob("*.json"):
                    annotation = parse_darwin_json(annotation_path, count=None)
                    if annotation is None:
                        continue

                    if video_frames and any(
                        not slot.frame_urls for slot in annotation.slots
                    ):
                        # will raise if not installed via pip install darwin-py[ocv]
                        try:
                            from cv2 import (  # pylint: disable=import-outside-toplevel # noqa F401
                                VideoCapture,
                            )
                        except ImportError as e:
                            raise MissingDependency(
                                "Missing Dependency: OpenCV required for Video Extraction. Install with `pip install darwin-py\[ocv]`"
                            ) from e
                    filename = Path(annotation.filename).stem
                    if filename in stems:
                        stems[filename] += 1
                        filename = f"{filename}_{stems[filename]}"
                    else:
                        stems[filename] = 1

                    destination_name = (
                        annotations_dir / f"{filename}{annotation_path.suffix}"
                    )
                    shutil.move(str(annotation_path), str(destination_name))

        # Extract the list of classes and create the text files
        make_class_lists(release_dir)

        if release.latest and is_unix_like_os():
            try:
                latest_dir: Path = self.local_releases_path / "latest"
                if latest_dir.is_symlink():
                    latest_dir.unlink()

                target_link: Path = self.local_releases_path / release_dir.name
                latest_dir.symlink_to(target_link)
            except OSError:
                self.console.log(
                    f"Could not mark release {release.name} as latest. Continuing..."
                )

        if only_annotations:
            # No images will be downloaded
            return None, 0

        # Create the generator with the download instructions
        progress, count = download_all_images_from_annotations(
            client=self.client,
            annotations_path=annotations_dir,
            images_path=self.local_images_path,
            force_replace=force_replace,
            remove_extra=remove_extra,
            use_folders=use_folders,
            video_frames=video_frames,
            force_slots=force_slots,
            ignore_slots=ignore_slots,
        )
        if count == 0:
            return None, count

        # If blocking is selected, download the dataset on the file system
        if blocking:
            max_workers = None
            env_max_workers = os.getenv("DARWIN_DOWNLOAD_FILES_CONCURRENCY")
            if env_max_workers and int(env_max_workers) > 0:
                max_workers = int(env_max_workers)

            console.print(
                f"Going to download {str(count)} files to {self.local_images_path.as_posix()} ."
            )
            successes, errors = exhaust_generator(
                progress=progress(),
                count=count,
                multi_processed=multi_processed,
                worker_count=max_workers,
            )
            if errors:
                self.console.print(
                    f"Encountered errors downloading {len(errors)} files"
                )
            for error in errors:
                self.console.print(f"\t - {error}")

            downloaded_file_count = len(
                [
                    f
                    for f in self.local_images_path.rglob("*")
                    if f.is_file() and not f.name.startswith(".")
                ]
            )

            console.print(
                f"Total file count after download completed {str(downloaded_file_count)}."
            )

            return None, count
        else:
            return progress, count

    def remove_remote(self) -> None:
        """Archives (soft-deletion) this ``RemoteDataset``."""
        self.client.archive_remote_dataset(self.dataset_id, self.team)

    @abstractmethod
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

    @abstractmethod
    def archive(self, items: Iterator[DatasetItem]) -> None:
        """
        Archives (soft-deletion) the given ``DatasetItem``\\s belonging to this ``RemoteDataset``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be archived.
        """

    @abstractmethod
    def restore_archived(self, items: Iterator[DatasetItem]) -> None:
        """
        Restores the archived ``DatasetItem``\\s that belong to this ``RemoteDataset``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be restored.
        """

    @abstractmethod
    def move_to_new(self, items: Iterator[DatasetItem]) -> None:
        """
        Changes the given ``DatasetItem``\\s status to ``new``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s whose status will change.
        """

    @abstractmethod
    def reset(self, items: Iterator[DatasetItem]) -> None:
        """
        Resets the given ``DatasetItem``\\s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be reset.
        """

    @abstractmethod
    def complete(self, items: Iterator[DatasetItem]) -> None:
        """
        Completes the given ``DatasetItem``\\s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be completed.
        """

    @abstractmethod
    def delete_items(self, items: Iterator[DatasetItem]) -> None:
        """
        Deletes the given ``DatasetItem``\\s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``\\s to be deleted.
        """

    def fetch_annotation_type_id_for_name(self, name: str) -> Optional[int]:
        """
        Fetches annotation type id for a annotation type name, such as ``bounding_box``.

        Parameters
        ----------
        name: str
            The name of the annotation we want the id for.


        Returns
        -------
        Optional[int]
            The id of the annotation type or ``None`` if it doesn't exist.
        """
        if not self.annotation_types:
            self.annotation_types = self.client.annotation_types()

        for annotation_type in self.annotation_types:
            if annotation_type["name"] == name:
                return annotation_type["id"]

        return None

    def create_annotation_class(
        self, name: str, type: str, subtypes: List[str] = []
    ) -> Dict[str, Any]:
        """
        Creates an annotation class for this ``RemoteDataset``.

        Parameters
        ----------
        name : str
            The name of the annotation class.
        type : str
            The type of the annotation class.
        subtypes : List[str], default: []
            Annotation class subtypes.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the server response.

        Raises
        ------
        ValueError
            If a given annotation type or subtype is unknown.
        """

        type_ids: List[int] = []
        for annotation_type in [type] + subtypes:
            type_id: Optional[int] = self.fetch_annotation_type_id_for_name(
                annotation_type
            )
            if not type_id and self.annotation_types is not None:
                list_of_annotation_types = ", ".join(
                    [type["name"] for type in self.annotation_types]
                )
                raise ValueError(
                    f"Unknown annotation type: '{annotation_type}', valid values: {list_of_annotation_types}"
                )

            if type_id is not None:
                type_ids.append(type_id)

        return self.client.create_annotation_class(self.dataset_id, type_ids, name)

    def add_annotation_class(
        self, annotation_class: Union[AnnotationClass, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Adds an annotation class to this ``RemoteDataset``.

        Parameters
        ----------
        annotation_class : Union[AnnotationClass, int]
            The annotation class to add or its id.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with the server response or ``None`` if the annotations class already
            exists.

        Raises
        ------
        ValueError
            If the given ``annotation_class`` does not exist in this ``RemoteDataset``'s team.
        """
        # Waiting for a better api for setting classes
        # in the meantime this will do
        all_classes: List[Dict[str, Any]] = self.fetch_remote_classes(True)

        if isinstance(annotation_class, int):
            match = [cls for cls in all_classes if cls["id"] == annotation_class]
            if not match:
                raise ValueError(
                    f"Annotation class id: `{annotation_class}` does not exist in Team."
                )
        else:
            annotation_class_type = (
                annotation_class.annotation_internal_type
                or annotation_class.annotation_type
            )
            match = [
                cls
                for cls in all_classes
                if cls["name"] == annotation_class.name
                and annotation_class_type in cls["annotation_types"]
            ]
            if not match:
                # We do not expect to reach here; as pervious logic divides annotation classes in imports
                # between `in team` and `new to platform`
                raise ValueError(
                    f"Annotation class name: `{annotation_class.name}`, type: `{annotation_class_type}`; does not exist in Team."
                )

        datasets = match[0]["datasets"]
        # check that we are not already part of the dataset
        for dataset in datasets:
            if dataset["id"] == self.dataset_id:
                return None
        datasets.append({"id": self.dataset_id})
        # we typecast to dictionary because we are not passing the raw=True parameter.
        class_id = match[0]["id"]
        payload = {"datasets": datasets, "id": class_id}
        return self.client.update_annotation_class(class_id, payload)

    def fetch_remote_classes(self, team_wide=False) -> List[Dict[str, Any]]:
        """
        Fetches all the Annotation Classes from this ``RemoteDataset``.

        Parameters
        ----------
        team_wide : bool, default: False
            If ``True`` will return all Annotation Classes that belong to the team. If ``False``
            will only return Annotation Classes which have been added to the dataset.

        Returns
        -------
        List[Dict[str, Any]]:
            List of Annotation Classes (can be empty).
        """
        all_classes: List[Dict[str, Any]] = self.client.fetch_remote_classes()

        classes_to_return = []
        for cls in all_classes:
            belongs_to_current_dataset = any(
                dataset["id"] == self.dataset_id for dataset in cls["datasets"]
            )
            cls["available"] = belongs_to_current_dataset
            if team_wide or belongs_to_current_dataset:
                classes_to_return.append(cls)
            elif cls["annotation_types"] == ["raster_layer"]:
                classes_to_return.append(cls)
        return classes_to_return

    def fetch_remote_attributes(self) -> List[Dict[str, Any]]:
        """
        Fetches all remote attributes on the remote dataset.

        Returns
        -------
        List[Dict[str, Any]]
            A List with the attributes, where each attribute is a dictionary.
        """
        return self.client.fetch_remote_attributes(self.dataset_id)

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    def get_release(
        self, name: str = "latest", include_unavailable: bool = True
    ) -> "Release":
        """
        Get a specific ``Release`` for this ``RemoteDataset``.

        Parameters
        ----------
        name : str, default: "latest"
            Name of the export.
        include_unavailable : bool, default: True
            If True, return all releases, including those that are not available.

        Returns
        -------
        Release
            The selected release.

        Raises
        ------
        NotFound
            The selected ``Release`` does not exist.
        """
        releases = self.get_releases(include_unavailable=include_unavailable)
        if not releases:
            raise NotFound(
                str(
                    f"No releases found for dataset '{self.name}'. Please create an export of this dataset first."
                )
            )

        # overwrite default name with stored dataset.release if supplied
        if self.release and name == "latest":
            name = self.release
        elif name == "latest":
            return (
                sorted(releases, key=lambda x: x.export_date, reverse=True)[0]
                if include_unavailable
                else next((release for release in releases if release.latest))
            )

        for release in releases:
            if str(release.name) == name:
                return release
        raise NotFound(
            str(
                f"Release name '{name}' not found in dataset '{self.name}'. Please check this release exists for this dataset."
            )
        )

    def split(
        self,
        val_percentage: float = 0.1,
        test_percentage: float = 0,
        split_seed: int = 0,
        make_default_split: bool = True,
        release_name: Optional[str] = None,
    ) -> None:
        """
        Creates lists of file names for each split for train, validation, and test.
        Note: This functions needs a local copy of the dataset.

        Parameters
        ----------
        val_percentage : float, default: 0.1
            Percentage of images used in the validation set.
        test_percentage : float, default: 0
            Percentage of images used in the test set.
        split_seed : int, default: 0
            Fix seed for random split creation.
        make_default_split: bool, default: True
            Makes this split the default split.
        release_name: Optional[str], default: None
            Version of the dataset.

        Raises
        ------
        NotFound
            If this ``RemoteDataset`` is not found locally.
        """
        if not self.local_path.exists():
            raise NotFound(
                "Local dataset not found: the split is performed on the local copy of the dataset. \
                           Pull the dataset from Darwin first using pull()"
            )
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        split_dataset(
            self.local_path,
            release_name=release_name,
            val_percentage=val_percentage,
            test_percentage=test_percentage,
            split_seed=split_seed,
            make_default_split=make_default_split,
        )

    def classes(
        self, annotation_type: str, release_name: Optional[str] = None
    ) -> List[str]:
        """
        Returns the list of ``class_type`` classes.

        Parameters
        ----------
        annotation_type : str
            The type of annotation classes, e.g. 'tag' or 'polygon'.
        release_name: Optional[str], default: None
            Version of the dataset.


        Returns
        -------
        classes: List[str]
            List of classes in the dataset of type ``class_type``.
        """
        assert self.local_path.exists()
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        return get_classes(
            self.local_path, release_name=release_name, annotation_type=annotation_type
        )

    def annotations(
        self,
        partition: str,
        split: str = "split",
        split_type: str = "stratified",
        annotation_type: str = "polygon",
        release_name: Optional[str] = None,
        annotation_format: Optional[str] = "darwin",
    ) -> Iterable[Dict[str, Any]]:
        """
        Returns all the annotations of a given split and partition in a single dictionary.

        Parameters
        ----------
        partition : str
            Selects one of the partitions [train, val, test].
        split : str, default: "split"
            Selects the split that defines the percentages used (use 'split' to select the default split.
        split_type : str, default: "stratified"
            Heuristic used to do the split [random, stratified].
        annotation_type : str, default: "polygon"
            The type of annotation classes [tag, polygon].
        release_name : Optional[str], default: None
            Version of the dataset.
        annotation_format : Optional[str], default: "darwin"
            Re-formatting of the annotation when loaded [coco, darwin].

        Yields
        -------
        Dict[str, Any]
            Dictionary representing an annotation from this ``RemoteDataset``.
        """
        assert self.local_path.exists()
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        for annotation in get_annotations(
            self.local_path,
            partition=partition,
            split=split,
            split_type=split_type,
            annotation_type=annotation_type,
            release_name=release_name,
            annotation_format=annotation_format,
        ):
            yield annotation

    @abstractmethod
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

    @abstractmethod
    def post_comment(
        self, item: DatasetItem, text: str, x: float, y: float, w: float, h: float
    ) -> None:
        """
        Adds a comment to an item in this dataset. The comment will be added with a bounding box.
        Creates the workflow for said item if necessary.

        Parameters
        ----------
        item : DatasetItem
            The ``DatasetItem`` which will receive the comment.
        text : str
            The text of the comment.
        x : float
            The x coordinate of the bounding box containing the comment.
        y : float
            The y coordinate of the bounding box containing the comment.
        w : float
            The width of the bounding box containing the comment.
        h : float
            The height of the bounding box containing the comment.
        """

    @abstractmethod
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
        ...

    @property
    def remote_path(self) -> Path:
        """Returns an URL specifying the location of the remote dataset."""
        return Path(urljoin(self.client.base_url, f"/datasets/{self.dataset_id}"))

    @property
    def local_path(self) -> Path:
        """Returns a Path to the local dataset."""
        datasets_dir: str = self.client.get_datasets_dir(self.team)

        if self.slug:
            return Path(datasets_dir) / self.team / self.slug
        else:
            return Path(datasets_dir) / self.team

    @property
    def local_releases_path(self) -> Path:
        """Returns a Path to the local dataset releases."""
        return self.local_path / "releases"

    @property
    def local_images_path(self) -> Path:
        """Returns a local Path to the images folder."""
        return self.local_path / "images"

    @property
    def identifier(self) -> DatasetIdentifier:
        """The ``DatasetIdentifier`` of this ``RemoteDataset``."""
        return DatasetIdentifier(team_slug=self.team, dataset_slug=self.slug)

    def _build_image_annotation(
        self, annotation_file: AnnotationFile, team_name: str
    ) -> Dict[str, Any]:
        return build_image_annotation(annotation_file, team_name)
