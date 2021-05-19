import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional
from urllib import parse

from darwin.dataset.download_manager import download_all_images_from_annotations
from darwin.dataset.identifier import DatasetIdentifier
from darwin.dataset.release import Release
from darwin.dataset.upload_manager import add_files_to_dataset
from darwin.dataset.utils import exhaust_generator, get_annotations, get_classes, make_class_lists, split_dataset
from darwin.exceptions import NotFound, UnsupportedExportFormat
from darwin.item import parse_dataset_item
from darwin.utils import find_files, urljoin
from darwin.validators import name_taken, validation_error

if TYPE_CHECKING:
    from darwin.client import Client


class RemoteDataset:
    def __init__(
        self,
        *,
        team: str,
        name: str,
        slug: Optional[str] = None,
        dataset_id: int,
        image_count: int = 0,
        progress: float = 0,
        client: "Client",
    ):
        """Inits a DarwinDataset.
        This class manages the remote and local versions of a dataset hosted on Darwin.
        It allows several dataset management operations such as syncing between
        remote and local, pulling a remote dataset, removing the local files, ...

        Parameters
        ----------
        name : str
            Name of the datasets as originally displayed on Darwin.
            It may contain white spaces, capital letters and special characters, e.g. `Bird Species!`
        slug : str
            This is the dataset name with everything lower-case, removed specials characters and
            spaces are replaced by dashes, e.g., `bird-species`. This string is unique within a team
        dataset_id : int
            Unique internal reference from the Darwin backend
        image_count : int
            Dataset size (number of images)
        progress : float
            How much of the dataset has been annotated 0.0 to 1.0 (1.0 == 100%)
        client : Client
            Client to use for interaction with the server
        """
        self.team = team
        self.name = name
        self.slug = slug or name
        self.dataset_id = dataset_id
        self.image_count = image_count
        self.progress = progress
        self.client = client

    def push(
        self,
        files_to_upload: List[str],
        blocking: bool = True,
        multi_threaded: bool = True,
        fps: int = 1,
        as_frames: bool = False,
        files_to_exclude: Optional[List[str]] = None,
        resume: bool = False,
        path: Optional[str] = None,
    ):
        """Uploads a local dataset (images ONLY) in the datasets directory.

        Parameters
        ----------
        files_to_upload : list[Path]
            List of files to upload. It can be a folder.
        blocking : bool
            If False, the dataset is not uploaded and a generator function is returned instead
        multi_threaded : bool
            Uses multiprocessing to upload the dataset in parallel.
            If blocking is False this has no effect.
        files_to_exclude : list[str]
            List of files to exclude from the file scan (which is done only if files is None)
        fps : int
            Number of file per seconds to upload
        as_frames: bool
            Annotate as video.
        resume : bool
            Flag for signalling the resuming of a push
        path: str
            Optional path to put the files into

        Returns
        -------
        generator : function
            Generator for doing the actual uploads. This is None if blocking is True
        count : int
            The files count
        """

        # paths needs to start with /
        if path and path[0] != "/":
            path = f"/{path}"

        # This is where the responses from the upload function will be saved/load for resume
        self.local_path.parent.mkdir(exist_ok=True)
        responses_path = self.local_path.parent / ".upload_responses.json"
        # Init optional parameters
        if files_to_exclude is None:
            files_to_exclude = []
        if files_to_upload is None:
            raise NotFound("Dataset location not found. Check your path.")

        if resume:
            if not responses_path.exists():
                raise NotFound("Dataset location not found. Check your path.")
            with responses_path.open() as f:
                logged_responses = json.load(f)
            files_to_exclude.extend(
                [
                    response["file_path"]
                    for response in logged_responses
                    if response["s3_response_status_code"].startswith("2")
                ]
            )

        files_to_upload = find_files(files=files_to_upload, recursive=True, files_to_exclude=files_to_exclude)

        if not files_to_upload:
            raise ValueError("No files to upload, check your path, exclusion filters and resume flag")

        progress, count = add_files_to_dataset(
            client=self.client,
            dataset_id=str(self.dataset_id),
            filenames=files_to_upload,
            fps=fps,
            as_frames=as_frames,
            team=self.team,
            path=path,
        )

        # If blocking is selected, upload the dataset remotely
        if blocking:
            responses = exhaust_generator(progress=progress, count=count, multi_threaded=multi_threaded)
            # Log responses to file
            if responses:
                responses = [{k: str(v) for k, v in response.items()} for response in responses]
                if resume:
                    responses.extend(logged_responses)
                with responses_path.open("w") as f:
                    json.dump(responses, f)
            return None, count
        else:
            return progress, count

    def pull(
        self,
        *,
        release: Optional[Release] = None,
        blocking: bool = True,
        multi_threaded: bool = True,
        only_annotations: bool = False,
        force_replace: bool = False,
        remove_extra: bool = False,
        subset_filter_annotations_function: Optional[Callable] = None,
        subset_folder_name: Optional[str] = None,
        use_folders: bool = False,
        video_frames: Optional[bool] = False,
    ):
        """Downloads a remote project (images and annotations) in the datasets directory.

        Parameters
        ----------
        release: Release
            The release to pull
        blocking : bool
            If False, the dataset is not downloaded and a generator function is returned instead
        multi_threaded : bool
            Uses multiprocessing to download the dataset in parallel. If blocking is False this has no effect.
        only_annotations: bool
            Download only the annotations and no corresponding images
        force_replace: bool
            Forces the re-download of an existing image
        remove_extra: bool
            Removes existing images for which there is not corresponding annotation
        subset_filter_annotations_function: Callable
            This function receives the directory where the annotations are downloaded and can
            perform any operation on them i.e. filtering them with custom rules or else.
            If it needs to receive other parameters is advised to use functools.partial() for it.
        subset_folder_name: str
            Name of the folder with the subset of the dataset. If not provided a timestamp is used.
        use_folders: bool
            Recreates folders from the dataset
        video_frames: bool
            Pulls video frames images instead of video files

        Returns
        -------
        generator : function
            Generator for doing the actual downloads. This is None if blocking is True
        count : int
            The files count
        """
        if release is None:
            release = self.get_release()

        if release.format != "json":
            raise UnsupportedExportFormat(release.format)

        release_dir = self.local_releases_path / release.name
        release_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            # Download the release from Darwin
            zip_file_path = release.download_zip(tmp_dir / "dataset.zip")
            with zipfile.ZipFile(zip_file_path) as z:
                # Extract annotations
                z.extractall(tmp_dir)
                # If a filtering function is provided, apply it
                if subset_filter_annotations_function is not None:
                    subset_filter_annotations_function(tmp_dir)
                    if subset_folder_name is None:
                        subset_folder_name = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
                annotations_dir = release_dir / (subset_folder_name or "") / "annotations"
                # Remove existing annotations if necessary
                if annotations_dir.exists():
                    try:
                        shutil.rmtree(annotations_dir)
                    except PermissionError:
                        print(f"Could not remove dataset in {annotations_dir}. Permission denied.")
                annotations_dir.mkdir(parents=True, exist_ok=False)
                # Move the annotations into the right folder and rename them to have the image
                # original filename as contained in the json
                for annotation_path in tmp_dir.glob("*.json"):
                    with annotation_path.open() as file:
                        annotation = json.load(file)
                    filename = Path(annotation["image"]["filename"]).stem
                    destination_name = annotations_dir / f"{filename}{annotation_path.suffix}"
                    shutil.move(str(annotation_path), str(destination_name))

        # Extract the list of classes and create the text files
        make_class_lists(release_dir)

        if release.latest:
            latest_dir = self.local_releases_path / "latest"
            if latest_dir.is_symlink():
                latest_dir.unlink()
            latest_dir.symlink_to(f"./{release_dir.name}")

        if only_annotations:
            # No images will be downloaded
            return None, 0

        team_config = self.client.config.get_team(self.team)
        api_key = team_config.get("api_key")

        # Create the generator with the download instructions
        progress, count = download_all_images_from_annotations(
            api_key=api_key,
            api_url=self.client.url,
            annotations_path=annotations_dir,
            images_path=self.local_images_path,
            force_replace=force_replace,
            remove_extra=remove_extra,
            use_folders=use_folders,
            video_frames=video_frames,
        )
        if count == 0:
            return None, count

        # If blocking is selected, download the dataset on the file system
        if blocking:
            exhaust_generator(progress=progress(), count=count, multi_threaded=multi_threaded)
            return None, count
        else:
            return progress, count

    def remove_remote(self):
        """Archives (soft-deletion) the remote dataset"""
        self.client.put(f"datasets/{self.dataset_id}/archive", payload={}, team=self.team)

    def fetch_remote_files(self, filters: Optional[dict] = None):
        """Fetch and lists all files on the remote dataset"""
        base_url = f"/datasets/{self.dataset_id}/items"
        parameters = {}
        if filters:
            for list_type in ["filenames", "statuses"]:
                if list_type in filters:
                    if type(filters[list_type]) is list:
                        parameters[list_type] = ",".join(filters[list_type])
                    else:
                        parameters[list_type] = filters[list_type]
            if "path" in filters:
                parameters["path"] = filters["path"]
            if "types" in filters:
                parameters["types"] = filters["types"]

        cursor = {"page[size]": 500}
        while True:
            response = self.client.post(f"{base_url}?{parse.urlencode(cursor)}", {"filter": parameters}, team=self.team)
            yield from [parse_dataset_item(item) for item in response["items"]]
            if response["metadata"]["next"]:
                cursor["page[from]"] = response["metadata"]["next"]
            else:
                return

    def archive(self, items):
        self.client.put(
            f"datasets/{self.dataset_id}/items/archive", {"filter": {"dataset_item_ids": [item.id for item in items]}}
        )

    def restore_archived(self, items):
        self.client.put(
            f"datasets/{self.dataset_id}/items/restore", {"filter": {"dataset_item_ids": [item.id for item in items]}}
        )

    def fetch_annotation_type_id_for_name(self, name: str):
        """Fetches annotation type id for a annotation type name, such as bounding_box"""
        annotation_types = self.client.get("/annotation_types")
        for annotation_type in annotation_types:
            if annotation_type["name"] == name:
                return annotation_type["id"]

    def create_annotation_class(self, name: str, type: str):
        type_id = self.fetch_annotation_type_id_for_name(type)
        return self.client.post(
            f"/annotation_classes",
            payload={
                "dataset_id": self.dataset_id,
                "name": name,
                "metadata": {"_color": "auto"},
                "annotation_type_ids": [type_id],
            },
            error_handlers=[name_taken, validation_error],
        )

    def fetch_remote_classes(self):
        """Fetches all remote classes on the remote dataset"""
        return self.client.get(f"/datasets/{self.dataset_id}/annotation_classes?include_tags=true")[
            "annotation_classes"
        ]

    def fetch_remote_attributes(self):
        """Fetches all remote attributes on the remote dataset"""
        return self.client.get(f"/datasets/{self.dataset_id}/attributes")

    def export(self, name: str, annotation_class_ids: Optional[List[str]] = None, include_url_token: bool = False):
        """Create a new release for the dataset

        Parameters
        ----------
        name: str
            Name of the release
        annotation_class_ids: List
            List of the classes to filter
        include_url_token: bool
            Should the image url in the export be include a token enabling access without team membership
        """
        if annotation_class_ids is None:
            annotation_class_ids = []
        payload = {
            "annotation_class_ids": annotation_class_ids,
            "name": name,
            "include_export_token": include_url_token,
        }
        self.client.post(
            f"/datasets/{self.dataset_id}/exports",
            payload=payload,
            team=self.team,
            error_handlers=[name_taken, validation_error],
        )

    def get_report(self, granularity="day"):
        return self.client.get(
            f"/reports/{self.team}/annotation?group_by=dataset,user&dataset_ids={self.dataset_id}&granularity={granularity}&format=csv&include=dataset.name,user.first_name,user.last_name,user.email",
            team=self.team,
            raw=True,
        ).text

    def get_releases(self):
        """Get a sorted list of releases with the most recent first

        Returns
        -------
        list(Release)
            Return a sorted list of releases with the most recent first
        Raises
        ------
        """
        try:
            releases_json = self.client.get(f"/datasets/{self.dataset_id}/exports", team=self.team)
        except NotFound:
            return []
        releases = [Release.parse_json(self.slug, self.team, payload) for payload in releases_json]
        return sorted(filter(lambda x: x.available, releases), key=lambda x: x.version, reverse=True)

    def get_release(self, name: str = "latest"):
        """Get a specific release for this dataset

        Parameters
        ----------
        name: str
            Name of the export

        Returns
        -------
        release: Release
            The selected release

        Raises
        ------
        NotFound
            The selected release does not exists
        """
        releases = self.get_releases()
        if not releases:
            raise NotFound(self.identifier)

        if name == "latest":
            return next((release for release in releases if release.latest))

        for release in releases:
            if str(release.name) == name:
                return release
        raise NotFound(self.identifier)

    def split(
        self,
        val_percentage: float = 0.1,
        test_percentage: float = 0,
        split_seed: int = 0,
        make_default_split: bool = True,
        release_name: Optional[str] = None,
    ):
        """
        Creates lists of file names for each split for train, validation, and test.
        Note: This functions needs a local copy of the dataset

        Parameters
        ----------
        val_percentage : float
            Percentage of images used in the validation set
        test_percentage : float
            Percentage of images used in the test set
        force_resplit : bool
            Discard previous split and create a new one
        split_seed : int
            Fix seed for random split creation
        make_default_split: bool
            Makes this split the default split
        release_name: str
            Version of the dataset
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

    def classes(self, annotation_type: str, release_name: Optional[str] = None):
        """
        Returns the list of `class_type` classes

        Parameters
        ----------
        annotation_type
            The type of annotation classes, e.g. 'tag' or 'polygon'
        release_name: str
            Version of the dataset


        Returns
        -------
        classes: list
            List of classes in the dataset of type `class_type`
        """
        assert self.local_path.exists()
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        return get_classes(self.local_path, release_name=release_name, annotation_type=annotation_type)

    def annotations(
        self,
        partition: str,
        split: str = "split",
        split_type: str = "stratified",
        annotation_type: str = "polygon",
        release_name: Optional[str] = None,
        annotation_format: Optional[str] = "darwin",
    ):
        """
        Returns all the annotations of a given split and partition in a single dictionary

        Parameters
        ----------
        partition
            Selects one of the partitions [train, val, test]
        split
            Selects the split that defines the percetages used (use 'split' to select the default split
        split_type
            Heuristic used to do the split [random, stratified]
        annotation_type
            The type of annotation classes [tag, polygon]
        release_name: str
            Version of the dataset
        annotation_format: str
            Re-formatting of the annotation when loaded [coco, darwin]

        Returns
        -------
        dict
            Dictionary containing all the annotations of the dataset
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

    def workview_url_for_item(self, item):
        return urljoin(self.client.base_url, f"/workview?dataset={self.dataset_id}&image={item.seq}")

    @property
    def remote_path(self) -> Path:
        """Returns an URL specifying the location of the remote dataset"""
        return Path(urljoin(self.client.base_url, f"/datasets/{self.dataset_id}"))

    @property
    def local_path(self) -> Path:
        """Returns a Path to the local dataset"""
        if self.slug is not None:
            return Path(self.client.get_datasets_dir(self.team)) / self.team / self.slug
        else:
            return Path(self.client.get_datasets_dir(self.team)) / self.team

    @property
    def local_releases_path(self) -> Path:
        """Returns a Path to the local dataset releases"""
        return self.local_path / "releases"

    @property
    def local_images_path(self) -> Path:
        """Returns a local Path to the images folder"""
        return self.local_path / "images"

    @property
    def identifier(self) -> DatasetIdentifier:
        return DatasetIdentifier(team_slug=self.team, dataset_slug=self.slug)
