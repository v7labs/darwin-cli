import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Callable
from datetime import datetime
from darwin.dataset.download_manager import download_all_images_from_annotations
from darwin.dataset.identifier import DatasetIdentifier
from darwin.dataset.release import Release
from darwin.dataset.upload_manager import add_files_to_dataset
from darwin.dataset.utils import exhaust_generator
from darwin.exceptions import NotFound
from darwin.utils import find_files, urljoin

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
        files_to_exclude: Optional[List[str]] = None,
        resume: bool = False,
    ):
        """Uploads a local project (images ONLY) in the projects directory.

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
        resume : bool
            Flag for signalling the resuming of a push

        Returns
        -------
        generator : function
            Generator for doing the actual uploads. This is None if blocking is True
        count : int
            The files count
        """

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

        files_to_upload = find_files(
            files=files_to_upload, recursive=True, files_to_exclude=files_to_exclude
        )

        if not files_to_upload:
            raise ValueError(
                "No files to upload, check your path, exclusion filters and resume flag"
            )

        progress, count = add_files_to_dataset(
            client=self.client,
            dataset_id=str(self.dataset_id),
            filenames=files_to_upload,
            fps=fps,
            team=self.team,
        )

        # If blocking is selected, upload the dataset remotely
        if blocking:
            responses = exhaust_generator(
                progress=progress, count=count, multi_threaded=multi_threaded
            )
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
        remove_extra: bool = True,
        subset_filter_annotations_function: Optional[Callable] = None,
        subset_folder_name: Optional[str] = None,
    ):
        """Downloads a remote project (images and annotations) in the projects directory.

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

        Returns
        -------
        generator : function
            Generator for doing the actual downloads. This is None if blocking is True
        count : int
            The files count
        """
        if release is None:
            release = self.get_release()

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
                annotations_dir = self.local_path / (subset_folder_name or "") / "annotations"
                # Remove existing annotations if necessary
                if annotations_dir.exists():
                    try:
                        shutil.rmtree(annotations_dir)
                    except PermissionError:
                        print(f"Could not remove dataset in {annotations_dir}. Permission denied.")
                annotations_dir.mkdir(parents=True, exist_ok=False)
                # Move the annotations into the right folder and rename them to have the image
                # original filename as contained in the json
                for annotation_path in tmp_dir.glob(f"*.json"):
                    annotation = json.load(annotation_path.open())
                    original_filename = Path(annotation['image']['original_filename'])
                    filename = Path(annotation['image']['filename']).stem
                    destination_name = annotations_dir / (filename + "_" + original_filename.stem + annotation_path.suffix)
                    shutil.move(str(annotation_path), str(destination_name))

        if only_annotations:
            # No images will be downloaded
            return None, 0

        # Create the generator with the download instructions
        images_dir = annotations_dir.parent / "images"
        progress, count = download_all_images_from_annotations(
            api_url=self.client.url,
            annotations_path=annotations_dir,
            images_path=images_dir,
            force_replace=force_replace,
            remove_extra=remove_extra,
        )
        if count == 0:
            print("Nothing to download")
            return None, count

        # If blocking is selected, download the dataset on the file system
        if blocking:
            exhaust_generator(
                progress=progress(), count=count, multi_threaded=multi_threaded
            )
            return None, count
        else:
            return progress, count

    def remove_remote(self):
        """Archives (soft-deletion) the remote dataset"""
        self.client.put(f"datasets/{self.dataset_id}/archive", payload={}, team=self.team)

    def get_report(self, granularity="day"):
        return self.client.get(
            f"/reports/{self.dataset_id}/annotation?group_by=dataset,user&dataset_ids={self.dataset_id}&granularity={granularity}&format=csv&include=dataset.name,user.first_name,user.last_name,user.email",
            team=self.team,
            raw=True,
        ).text

    def release(self, name: Optional[str] = None):
        """Create a new release for the dataset

        Parameters
        ----------
        name: str
            Name of the release

        Returns
        -------
        release: Release
            The release created right now
        """
        self.client.post(f"/datasets/{self.dataset_id}/exports", team=self.team)
        return self.get_release()

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
        return sorted(releases, key=lambda x: x.version, reverse=True)

    def get_release(self, version: str = "latest"):
        """Get a specific release for this dataset

        Parameters
        ----------
        version: str
            Name of the version

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

        if version == "latest":
            return releases[0]

        for release in releases:
            if str(release.version) == version:
                return release
        raise NotFound(self.identifier)

    @property
    def remote_path(self) -> Path:
        """Returns an URL specifying the location of the remote dataset"""
        return urljoin(self.client.base_url, f"/datasets/{self.dataset_id}")

    @property
    def local_path(self) -> Path:
        """Returns a Path to the local dataset"""
        if self.slug is not None:
            return Path(self.client.get_datasets_dir(self.team)) / self.slug
        else:
            return Path(self.client.get_datasets_dir(self.team))

    @property
    def identifier(self) -> DatasetIdentifier:
        return DatasetIdentifier(f"{self.team}/{self.slug}")
