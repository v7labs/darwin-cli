import argparse
import concurrent.futures
import datetime
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, NoReturn, Optional, Union, cast

import humanize
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.theme import Theme

from darwin.client import Client
from darwin.config import Config
from darwin.dataset import RemoteDataset
from darwin.dataset.identifier import DatasetIdentifier
from darwin.dataset.release import Release
from darwin.dataset.split_manager import split_dataset
from darwin.dataset.upload_manager import LocalFile
from darwin.dataset.utils import get_release_path
from darwin.datatypes import ExportParser, ImportParser, PathLike, Team
from darwin.exceptions import (
    InvalidLogin,
    MissingConfig,
    NameTaken,
    NotFound,
    Unauthenticated,
    UnsupportedExportFormat,
    UnsupportedFileType,
    ValidationError,
)
from darwin.exporter import ExporterNotFoundError, export_annotations, get_exporter
from darwin.exporter.formats import supported_formats as export_formats
from darwin.importer import ImporterNotFoundError, get_importer, import_annotations
from darwin.importer.formats import supported_formats as import_formats
from darwin.item import DatasetItem
from darwin.utils import (
    find_files,
    persist_client_configuration,
    prompt,
    secure_continue_request,
)


def validate_api_key(api_key: str) -> None:
    """
    Validates the given API key. Exits the application if it fails validation.

    Parameters
    ----------
    api_key: str
        The API key to be validated.
    """
    example_key = "DHMhAWr.BHucps-tKMAi6rWF1xieOpUvNe5WzrHP"

    if len(api_key) != 40:
        _error(f"Expected key to be 40 characters long\n(example: {example_key})")

    if "." not in api_key:
        _error(f"Expected key formatted as prefix . suffix\n(example: {example_key})")

    if len(api_key.split(".")[0]) != 7:
        _error(f"Expected key prefix to be 7 characters long\n(example: {example_key})")


def authenticate(api_key: str, default_team: Optional[bool] = None, datasets_dir: Optional[Path] = None) -> Config:
    """
    Authenticate the API key against the server and creates a configuration file for it.

    Parameters
    ----------
    api_key : str
        API key to use for the client login.
    default_team: Optional[bool]
        Flag to make the team the default one. Defaults to None.
    datasets_dir: Optional[Path]
        Dataset directory on the file system. Defaults to None.

    Returns
    -------
    Config
    A configuration object to handle YAML files.
    """
    # Resolve the home folder if the dataset_dir starts with ~ or ~user

    validate_api_key(api_key)

    try:
        client = Client.from_api_key(api_key=api_key)
        config_path = Path.home() / ".darwin" / "config.yaml"
        config_path.parent.mkdir(exist_ok=True)

        if default_team is None:
            default_team = input(f"Make {client.default_team} the default team? [y/N] ") in ["Y", "y"]
        if datasets_dir is None:
            datasets_dir = Path(prompt("Datasets directory", "~/.darwin/datasets"))

        datasets_dir = Path(datasets_dir).expanduser()
        Path(datasets_dir).mkdir(parents=True, exist_ok=True)

        client.set_datasets_dir(datasets_dir)

        default_team_name: Optional[str] = client.default_team if default_team else None
        return persist_client_configuration(client, default_team=default_team_name)

    except InvalidLogin:
        _error("Invalid API key")


def current_team() -> None:
    """Print the team currently authenticated against."""
    client: Client = _load_client()
    print(client.default_team)


def list_teams() -> None:
    """Print a table of teams to which the client belong to."""
    for team in _config().get_all_teams():
        if team.default:
            print(f"{team.slug} (default)")
        else:
            print(team.slug)


def set_team(team_slug: str) -> None:
    """
    Switches the client to the selected team and persist the change on the configuration file.

    Parameters
    ----------
    team_slug : str
        Slug of the team to switch to.
    """
    config = _config()
    config.set_default_team(team_slug)


def create_dataset(dataset_slug: str) -> None:
    """
    Creates a dataset remotely. Exits the application if the dataset's name is already taken or is
    not valid.

    Parameters
    ----------
    dataset_slug : str
        Slug of the new dataset.
    """
    identifier: DatasetIdentifier = DatasetIdentifier.parse(dataset_slug)
    client: Client = _load_client(team_slug=identifier.team_slug)
    try:
        dataset: RemoteDataset = client.create_dataset(name=identifier.dataset_slug)
        print(
            f"Dataset '{dataset.name}' ({dataset.team}/{dataset.slug}) has been created.\nAccess at {dataset.remote_path}"
        )
        print_new_version_info(client)
    except NameTaken:
        _error(f"Dataset name '{identifier.dataset_slug}' is already taken.")
    except ValidationError:
        _error(f"Dataset name '{identifier.dataset_slug}' is not valid.")


def local(team: Optional[str] = None) -> None:
    """
    Lists synced datasets, stored in the specified path.

    Parameters
    ----------
    team: Optional[str]
        The name of the team to list, or the defautl one if no team is given. Defaults to None.
    """
    table: Table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Image Count", justify="right")
    table.add_column("Sync Date", justify="right")
    table.add_column("Size", justify="right")

    client: Client = _load_client(offline=True)
    for dataset_path in client.list_local_datasets(team_slug=team):
        files_in_dataset_path = find_files([dataset_path])
        table.add_row(
            f"{dataset_path.parent.name}/{dataset_path.name}",
            str(len(files_in_dataset_path)),
            humanize.naturaldate(datetime.datetime.fromtimestamp(dataset_path.stat().st_mtime)),
            humanize.naturalsize(sum(p.stat().st_size for p in files_in_dataset_path)),
        )

    Console().print(table)


def path(dataset_slug: str) -> Path:
    """
    Returns the absolute path of the specified dataset.
    Exits the application if the dataset does not exist locally.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.

    Returns
    -------
    Path
        The absolute path of the dataset.
    """
    identifier: DatasetIdentifier = DatasetIdentifier.parse(dataset_slug)
    client: Client = _load_client(offline=True)

    for path in client.list_local_datasets(team_slug=identifier.team_slug):
        if identifier.dataset_slug == path.name:
            return path

    _error(
        f"Dataset '{identifier.dataset_slug}' does not exist locally. "
        f"Use 'darwin dataset remote' to see all the available datasets, "
        f"and 'darwin dataset pull' to pull them."
    )


def url(dataset_slug: str) -> None:
    """
    Prints the url of the specified dataset.
    Exits the application if no dataset was found.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    """
    client: Client = _load_client(offline=True)
    try:
        remote_dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        print(remote_dataset.remote_path)
    except NotFound as e:
        _error(f"Dataset '{e.name}' does not exist.")


def dataset_report(dataset_slug: str, granularity: str) -> None:
    """
    Prints a dataset's report.
    Exits the application if no dataset is found.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    granularity: str
        Granualarity of the report, can be 'day', 'week' or 'month'.
    """
    client: Client = _load_client(offline=True)
    try:
        remote_dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        report: str = remote_dataset.get_report(granularity)
        print(report)
    except NotFound:
        _error(f"Dataset '{dataset_slug}' does not exist.")


def export_dataset(
    dataset_slug: str, include_url_token: bool, name: str, annotation_class_ids: Optional[List[str]] = None
) -> None:
    """
    Create a new release for the dataset.

    Parameters
    ----------
    dataset_slug: str
        Slug of the dataset to which we perform the operation on.
    include_url_token: bool
        If True includes the url token, if False does not.
    name: str
        Name of the release.
    annotation_class_ids: Optional[List[str]]
        List of the classes to filter. Defautls to None.
    """
    client: Client = _load_client(offline=False)
    identifier: DatasetIdentifier = DatasetIdentifier.parse(dataset_slug)
    ds: RemoteDataset = client.get_remote_dataset(identifier)
    ds.export(annotation_class_ids=annotation_class_ids, name=name, include_url_token=include_url_token)
    identifier.version = name
    print(f"Dataset {dataset_slug} successfully exported to {identifier}")
    print_new_version_info(client)


def pull_dataset(
    dataset_slug: str, only_annotations: bool = False, folders: bool = False, video_frames: bool = False
) -> None:
    """
    Downloads a remote dataset (images and annotations) in the datasets directory.
    Exits the application if dataset is not found, the user is not authenticated, there are no
    releases or the export format for the latest release is not supported.

    Parameters
    ----------
    dataset_slug: str
        Slug of the dataset to which we perform the operation on.
    only_annotations: bool
        Download only the annotations and no corresponding images. Defaults to False.
    folders: bool
        Recreates the folders in the dataset. Defaults to False.
    video_frames: bool
        Pulls video frames images instead of video files. Defaults to False.
    """
    version: str = DatasetIdentifier.parse(dataset_slug).version or "latest"
    client: Client = _load_client(offline=False, maybe_guest=True)
    try:
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
    except NotFound:
        _error(
            f"Dataset '{dataset_slug}' does not exist, please check the spelling. "
            f"Use 'darwin remote' to list all the remote datasets."
        )
    except Unauthenticated:
        _error(f"please re-authenticate")

    try:
        release: Release = dataset.get_release(version)
        dataset.pull(release=release, only_annotations=only_annotations, use_folders=folders, video_frames=video_frames)
        print_new_version_info(client)
    except NotFound:
        _error(
            f"Version '{dataset.identifier}:{version}' does not exist "
            f"Use 'darwin dataset releases' to list all available versions."
        )
    except UnsupportedExportFormat as uef:
        _error(
            f"Version '{dataset.identifier}:{version}' is of format '{uef.format}', "
            f"only the darwin format ('json') is supported for `darwin dataset pull`"
        )

    print(f"Dataset {release.identifier} downloaded at {dataset.local_path}. ")


def split(dataset_slug: str, val_percentage: float, test_percentage: float, seed: int = 0) -> None:
    """
    Splits a local version of a dataset into train, validation, and test partitions.

    Parameters
    ----------
    dataset_slug: str
        Slug of the dataset to which we perform the operation on.
    val_percentage: float
        Percentage in the validation set.
    test_percentage: float
        Percentage in the test set.
    seed: int
        Random seed. Defaults to 0.
    """
    identifier: DatasetIdentifier = DatasetIdentifier.parse(dataset_slug)
    client: Client = _load_client(offline=True)

    for p in client.list_local_datasets(team_slug=identifier.team_slug):
        if identifier.dataset_slug == p.name:
            try:
                split_path = split_dataset(
                    dataset_path=p,
                    release_name=identifier.version,
                    val_percentage=val_percentage,
                    test_percentage=test_percentage,
                    split_seed=seed,
                )
                print(f"Partition lists saved at {split_path}")
                return
            except ImportError as e:
                _error(e.msg)
            except NotFound as e:
                _error(e.name)
            except ValueError as e:
                _error(e.args[0])

    _error(
        f"Dataset '{identifier.dataset_slug}' does not exist locally. "
        f"Use 'darwin dataset remote' to see all the available datasets, "
        f"and 'darwin dataset pull' to pull them."
    )


def list_remote_datasets(all_teams: bool, team: Optional[str] = None) -> None:
    """
    Lists remote datasets with its annotation progress.

    Parameters
    ----------
    all_teams: bool
        If True, lists remote datasets from all teams, if False, lists only datasets from the given
        Team.
    team: Optional[str]
        Name of the team with the datasets we want to see. Uses the default Team is non is given.
        Defaults to None.
    """
    # TODO: add listing open datasets

    table: Table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Item Count", justify="right")
    table.add_column("Complete Items", justify="right")

    datasets: List[RemoteDataset] = []
    client: Optional[Client] = None
    if all_teams:
        teams: List[Team] = _config().get_all_teams()
        for a_team in teams:
            client = _load_client(a_team.slug)
            datasets += list(client.list_remote_datasets())
    else:
        client = _load_client(team)
        datasets = list(client.list_remote_datasets())

    for dataset in datasets:
        table.add_row(f"{dataset.team}/{dataset.slug}", str(dataset.item_count), f"{dataset.progress * 100:.1f}%")
    if table.row_count == 0:
        print("No dataset available.")
    else:
        Console().print(table)

    print_new_version_info(client)


def remove_remote_dataset(dataset_slug: str) -> None:
    """
    Remove a remote dataset from the workview. The dataset gets archived.
    Exits the application if no dataset with the given slug were found.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    """
    client: Client = _load_client(offline=False)
    try:
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        print(f"About to delete {dataset.identifier} on darwin.")
        if not secure_continue_request():
            print("Cancelled.")
            return

        dataset.remove_remote()
        print_new_version_info(client)
    except NotFound:
        _error(f"No dataset with name '{dataset_slug}'")


def dataset_list_releases(dataset_slug: str) -> None:
    """
    Lists all the releases from the given dataset.
    Exits the application if no dataset with the given slug were found.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    """
    client: Client = _load_client(offline=False)
    try:
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        releases: List[Release] = dataset.get_releases()
        if len(releases) == 0:
            print("No available releases, export one first.")
            return

        table: Table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Item Count", justify="right")
        table.add_column("Class Count", justify="right")
        table.add_column("Export Date", justify="right")

        for release in releases:
            if not release.available:
                continue
            table.add_row(
                str(release.identifier), str(release.image_count), str(release.class_count), str(release.export_date)
            )

        Console().print(table)
        print_new_version_info(client)
    except NotFound:
        _error(f"No dataset with name '{dataset_slug}'")


def upload_data(
    dataset_identifier: str,
    files: Optional[List[Union[PathLike, LocalFile]]],
    files_to_exclude: Optional[List[PathLike]],
    fps: int,
    path: Optional[str],
    frames: bool,
    preserve_folders: bool = False,
    verbose: bool = False,
) -> None:
    """
    Uploads the provided files to the remote dataset.
    Exits the application if no dataset with the given name is found, the files in the given path
    have unsupported formats, or if there are no files found in the given Path.

    Parameters
    ----------
    dataset_identifier : str
        Slug of the dataset to retrieve.
    files : List[Union[PathLike, LocalFile]]
        List of files to upload. Can be None.
    files_to_exclude : List[PathLike]
        List of files to exclude from the file scan (which is done only if files is None).
    fps : int
        Frame rate to split videos in.
    path : Optional[str]
        If provided; files will be placed under this path in the v7 platform. If `preserve_folders`
        is `True` then it must be possible to draw a relative path from this folder to the one the
        files are in, otherwise an error will be raised.
    frames : bool
        Specify whether the files will be uploaded as a list of frames or not.
    preserve_folders : bool
        Specify whether or not to preserve folder paths when uploading.
    verbose : bool
        Specify whther to have full traces print when uploading files or not.
    """
    client: Client = _load_client()
    try:
        max_workers: int = concurrent.futures.ThreadPoolExecutor()._max_workers  # type: ignore

        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_identifier)

        sync_metadata: Progress = Progress(SpinnerColumn(), TextColumn("[bold blue]Syncing metadata"))

        overall_progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}"), BarColumn(), "{task.completed} of {task.total}"
        )

        file_progress = Progress(
            TextColumn("[bold green]{task.fields[filename]}", justify="right"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )

        progress_table: Table = Table.grid()
        progress_table.add_row(sync_metadata)
        progress_table.add_row(file_progress)
        progress_table.add_row(overall_progress)
        with Live(progress_table):
            sync_task: TaskID = sync_metadata.add_task("")
            file_tasks: Dict[str, TaskID] = {}
            overall_task = overall_progress.add_task(
                "[green]Total progress", filename="Total progress", total=0, visible=False
            )

            def progress_callback(total_file_count, file_advancement):
                sync_metadata.update(sync_task, visible=False)
                overall_progress.update(overall_task, total=total_file_count, advance=file_advancement, visible=True)

            def file_upload_callback(file_name, file_total_bytes, file_bytes_sent):
                if file_name not in file_tasks:
                    file_tasks[file_name] = file_progress.add_task(
                        f"[blue]{file_name}", filename=file_name, total=file_total_bytes
                    )

                # Rich has a concurrency issue, so sometimes updating progress
                # or removing a task fails. Wrapping this logic around a try/catch block
                # is a workaround, we should consider solving this properly (e.g.: using locks)
                try:
                    file_progress.update(file_tasks[file_name], completed=file_bytes_sent)

                    for task in file_progress.tasks:
                        if task.finished and len(file_progress.tasks) >= max_workers:
                            file_progress.remove_task(task.id)
                except Exception as e:
                    pass

            upload_manager = dataset.push(
                files,
                files_to_exclude=files_to_exclude,
                fps=fps,
                as_frames=frames,
                path=path,
                preserve_folders=preserve_folders,
                progress_callback=progress_callback,
                file_upload_callback=file_upload_callback,
            )
        console = Console(theme=_console_theme())

        console.print()

        if not upload_manager.blocked_count and not upload_manager.error_count:
            console.print(f"All {upload_manager.total_count} files have been successfully uploaded.\n", style="success")
            return

        already_existing_items = []
        other_skipped_items = []
        for item in upload_manager.blocked_items:
            if item.reason == "ALREADY_EXISTS":
                already_existing_items.append(item)
            else:
                other_skipped_items.append(item)

        if already_existing_items:
            console.print(
                f"Skipped {len(already_existing_items)} files already in the dataset.\n", style="warning",
            )

        if upload_manager.error_count or other_skipped_items:
            error_count = upload_manager.error_count + len(other_skipped_items)
            console.print(
                f"{error_count} files couldn't be uploaded because an error occurred.\n", style="error",
            )

        if not verbose and upload_manager.error_count:
            console.print('Re-run with "--verbose" for further details')
            return

        error_table: Table = Table(
            "Dataset Item ID", "Filename", "Remote Path", "Stage", "Reason", show_header=True, header_style="bold cyan"
        )

        for item in upload_manager.blocked_items:
            if item.reason != "ALREADY_EXISTS":
                error_table.add_row(str(item.dataset_item_id), item.filename, item.path, "UPLOAD_REQUEST", item.reason)

        for error in upload_manager.errors:
            for local_file in upload_manager.local_files:
                if local_file.local_path != error.file_path:
                    continue

                for pending_item in upload_manager.pending_items:
                    if pending_item.filename != local_file.data["filename"]:
                        continue

                    error_table.add_row(
                        str(pending_item.dataset_item_id),
                        pending_item.filename,
                        pending_item.path,
                        error.stage.name,
                        str(error.error),
                    )
                    break

        if error_table.row_count:
            console.print(error_table)
        print_new_version_info(client)
    except NotFound as e:
        _error(f"No dataset with name '{e.name}'")
    except UnsupportedFileType as e:
        _error(f"Unsupported file type {e.path.suffix} ({e.path.name})")
    except ValueError:
        _error(f"No files found")


def dataset_import(dataset_slug: str, format: str, files: List[PathLike], append: bool) -> None:
    """
    Imports annotation files to the given dataset.
    Exits the application if no dataset with the given slug is found.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    format: str
        Format of the export files.
    files: List[PathLike]
        List of where the files are.
    append: bool
        If True it appends the annotation from the files to the dataset, if False it will override
        the dataset's current annotations with the ones from the given files.
    """

    client: Client = _load_client(dataset_identifier=dataset_slug)

    try:
        parser: ImportParser = get_importer(format)
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        import_annotations(dataset, parser, files, append)
    except ImporterNotFoundError:
        _error(f"Unsupported import format: {format}, currently supported: {import_formats}")
    except AttributeError:
        _error(f"Unsupported import format: {format}, currently supported: {import_formats}")
    except NotFound as e:
        _error(f"No dataset with name '{e.name}'")


def list_files(
    dataset_slug: str,
    statuses: Optional[str],
    path: Optional[str],
    only_filenames: bool,
    sort_by: Optional[str] = "updated_at:desc",
) -> None:
    """
    List all file from the given dataset.
    Exits the application if it finds unknown file statuses, if no dataset with the given slug is
    found or if another general error occurred.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    statuses: Optional[str]
        Only list files with the given statuses. Valid statuses are: 'annotate', 'archived',
        'complete', 'new', 'review'.
    path: Optional[str]
        Only list files whose Path matches.
    only_filenames: bool
        If True, only prints the filenames, if False it prints the full file url.
    sort_by: Optional[str]
        Sort order for listing files. Defaults to 'updated_at:desc'.
    """
    client: Client = _load_client(dataset_identifier=dataset_slug)
    try:
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        filters: Dict[str, Any] = {}

        if statuses:
            for status in statuses.split(","):
                if not _has_valid_status(status):
                    _error(f"Invalid status '{status}', available statuses: annotate, archived, complete, new, review")
            filters["statuses"] = statuses
        else:
            filters["statuses"] = "new,annotate,review,complete"

        if path:
            filters["path"] = path

        if not sort_by:
            sort_by = "updated_at:desc"

        for file in dataset.fetch_remote_files(filters, sort_by):
            if only_filenames:
                print(file.filename)
            else:
                image_url = dataset.workview_url_for_item(file)
                print(f"{file.filename}\t{file.status if not file.archived else 'archived'}\t {image_url}")
    except NotFound as e:
        _error(f"No dataset with name '{e.name}'")
    except ValueError as e:
        _error(str(e))


def set_file_status(dataset_slug: str, status: str, files: List[str]) -> None:
    """
    Sets the status of the given files from the given dataset.
    Exits the application if the given status is unknown or if no dataset was found.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    status: str
        The new status for the files.
    files: List[str]
        Names of the files we want to update.
    """
    if status not in ["archived", "clear", "new", "restore-archived"]:
        _error(f"Invalid status '{status}', available statuses: archived, clear, new, restore-archived")

    client: Client = _load_client(dataset_identifier=dataset_slug)
    try:
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        items: Iterator[DatasetItem] = dataset.fetch_remote_files({"filenames": ",".join(files)})
        if status == "archived":
            dataset.archive(items)
        elif status == "clear":
            dataset.reset(items)
        elif status == "new":
            dataset.move_to_new(items)
        elif status == "restore-archived":
            dataset.restore_archived(items)
    except NotFound as e:
        _error(f"No dataset with name '{e.name}'")


def delete_files(dataset_slug: str, files: List[str], skip_user_confirmation: bool = False) -> None:
    """
    Deletes the files from the given dataset.
    Exits the application if no dataset with the given slug is found or a general error occurs.

    Parameters
    ----------
    dataset_slug: str
        The dataset's slug.
    files: List[str]
        The list of filenames to delete.
    skip_user_confirmation: bool
        If True, skips user confirmation, if False it will prompt the user. Defaults to False.
    """
    client: Client = _load_client(dataset_identifier=dataset_slug)
    try:
        console = Console()
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
        items: Iterator[DatasetItem] = dataset.fetch_remote_files({"filenames": ",".join(files)})
        if not skip_user_confirmation and not secure_continue_request():
            console.print("Cancelled.")
            return

        with console.status("[bold red]Deleting files..."):
            dataset.delete_items(items)
            console.print("[bold green]Files successfully deleted!")

    except NotFound as e:
        _error(f"No dataset with name '{e.name}'")
    except:
        _error(f"An error has occurred, please try again later.")


def dataset_convert(dataset_identifier: str, format: str, output_dir: Optional[PathLike] = None) -> None:
    """
    Converts the annotations from the given dataset to the given format.
    Exits the application if no dataset with the given slug exists or no releases for the dataset
    were previously pulled.

    Parameters
    ----------
    dataset_identifier: str
        The dataset identifier, normally in the "<team-slug>/<dataset-slug>:<version>" form.
    format: str
        The format we want to convert to.
    output_dir: Optional[PathLike]
        The folder where the exported annotation files will be. If None it will be the inside the
        annotations folder of the dataset under 'other_formats/{format}'. The Defaults to None.
    """
    identifier: DatasetIdentifier = DatasetIdentifier.parse(dataset_identifier)
    client: Client = _load_client(team_slug=identifier.team_slug)

    try:
        parser: ExportParser = get_exporter(format)
        dataset: RemoteDataset = client.get_remote_dataset(dataset_identifier=identifier)
        if not dataset.local_path.exists():
            _error(
                f"No annotations downloaded for dataset f{dataset}, first pull a release using "
                f"'darwin dataset pull {identifier}'"
            )

        release_path: Path = get_release_path(dataset.local_path, identifier.version)
        annotations_path: Path = release_path / "annotations"
        if output_dir is None:
            output_dir = release_path / "other_formats" / format
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_annotations(parser, [annotations_path], output_dir)
    except ExporterNotFoundError:
        _error(f"Unsupported export format: {format}, currently supported: {export_formats}")
    except AttributeError:
        _error(f"Unsupported export format: {format}, currently supported: {export_formats}")
    except NotFound as e:
        _error(f"No dataset with name '{e.name}'")


def convert(format: str, files: List[PathLike], output_dir: Path) -> None:
    """
    Converts the given files to the specified format.

    Parameters
    ----------
    format: str
        The target format to export to.
    files: List[PathLike]
        List of files to be converted.
    output_dir: Path
        Folder where the exported annotations will be placed.
    """
    try:
        parser: ExportParser = get_exporter(format)
    except ExporterNotFoundError:
        _error(f"Unsupported export format, currently supported: {export_formats}")
    except AttributeError:
        _error(f"Unsupported export format, currently supported: {export_formats}")

    export_annotations(parser, files, output_dir)


def post_comment(
    dataset_slug: str, filename: str, text: str, x: float = 1, y: float = 1, w: float = 1, h: float = 1
) -> None:
    """
    Creates a comment box with a comment for the given file in the given dataset.

    Parameters
    ----------
    dataset_slug: str
        The slug of the dataset the item belongs to.
    filename: str
        The filename to receive the commment.
    text: str
        The comment.
    x: float, default: 1
        X value of the top left coordinate for the comment box.
    y: float, default: 1
        Y value of the top left coordinate for the comment box.
    w: float, default: 1
        Width of the comment box.
    h: float, default: 1
        Height of the comment box.

    Raises
    ------
    NotFound
        If the Dataset was not found.
    """
    client: Client = _load_client(dataset_identifier=dataset_slug)
    console = Console()

    try:
        dataset = client.get_remote_dataset(dataset_identifier=dataset_slug)
    except NotFound:
        _error(f"unable to find dataset: {dataset_slug}")

    items: List[DatasetItem] = list(dataset.fetch_remote_files(filters={"filenames": [filename]}))

    if len(items) == 0:
        console.print(f"[bold yellow]No files matching '{filename}' found...")
        return

    item: DatasetItem = items.pop()
    maybe_workflow_id: Optional[int] = item.current_workflow_id

    if maybe_workflow_id is None:
        workflow_id: int = client.instantitate_item(item.id)
    else:
        workflow_id = maybe_workflow_id

    try:
        client.post_workflow_comment(workflow_id, text, x, y, w, h)
        console.print("[bold green]Comment added successfully!")
    except Exception:
        console.print("[bold red]There was an error posting your comment!\n")
        console.print(f"[red]{traceback.format_exc()}")


def help(parser: argparse.ArgumentParser, subparser: Optional[str] = None) -> None:
    """
    Prints the help text for the given command.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        The parser used to read input from the user.
    subparser: Optional[str]
        Actions from the parser to be processed. Defaults to None.
    """
    if subparser:
        parser = next(
            action.choices[subparser]
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction) and subparser in action.choices
        )

    actions = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]

    print(parser.description)
    print("\nCommands:")
    for action in actions:
        # get all subparsers and print help
        for choice in sorted(action._choices_actions, key=lambda x: x.dest):
            print("    {:<19} {}".format(choice.dest, choice.help))


def print_new_version_info(client: Optional[Client] = None) -> None:
    """
    Prints a message informing the user of a new darwin-py version.
    Does nothing if no new version is available or if no client is provided.

    Parameters
    ----------
    client: Optional[Client]
        The client containing information aboue the new verison. Defaults to None.
    """
    if not client or not client.newer_darwin_version:
        return

    (a, b, c) = tuple(client.newer_darwin_version)

    console = Console(theme=_console_theme(), stderr=True)
    console.print(
        f"A newer version of darwin-py ({a}.{b}.{c}) is available!",
        "Run the following command to install it:",
        "",
        f"    pip install darwin-py=={a}.{b}.{c}",
        "",
        sep="\n",
        style="warning",
    )


def _error(message: str) -> NoReturn:
    console = Console(theme=_console_theme())
    console.print(f"Error: {message}", style="error")
    sys.exit(1)


def _config() -> Config:
    return Config(Path.home() / ".darwin" / "config.yaml")


def _load_client(
    team_slug: Optional[str] = None,
    offline: bool = False,
    maybe_guest: bool = False,
    dataset_identifier: Optional[str] = None,
) -> Client:
    """Fetches a client, potentially offline

    Parameters
    ----------
    offline : bool
        Flag for using an offline client

    maybe_guest : bool
        Flag to make a guest client, if config is missing
    Returns
    -------
    Client
    The client requested
    """
    if not team_slug and dataset_identifier:
        team_slug = DatasetIdentifier.parse(dataset_identifier).team_slug
    try:
        api_key = os.getenv("DARWIN_API_KEY")
        if api_key:
            client = Client.from_api_key(api_key)
        else:
            config_dir = Path.home() / ".darwin" / "config.yaml"
            client = Client.from_config(config_dir, team_slug=team_slug)
        return client
    except MissingConfig:
        if maybe_guest:
            return Client.from_guest()
        else:
            _error("Authenticate first")
    except InvalidLogin:
        _error("Please re-authenticate")
    except Unauthenticated:
        _error("Please re-authenticate")


def _console_theme() -> Theme:
    return Theme({"success": "bold green", "warning": "bold yellow", "error": "bold red"})


def _has_valid_status(status: str) -> bool:
    return status in ["new", "annotate", "review", "complete", "archived"]
