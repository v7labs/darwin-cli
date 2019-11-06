from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from darwin.config import Config

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".jpg", ".PNG", ".JPEG", ".JPG"]
SUPPORTED_VIDEO_EXTENSIONS = [".bpm", ".mov", ".mp4", ".BPM", ".MOV", ".MP4"]


if TYPE_CHECKING:
    from darwin.client import Client


def urljoin(*parts: str) -> str:
    """Take as input an unpacked list of strings and joins them to form an URL"""
    return "/".join(part.strip("/") for part in parts)


def is_project_dir(project_path: Path) -> bool:
    """Verifies if the directory is a project from Darwin by inspecting its sturcture

    Parameters
    ----------
    project_path : Path
        Directory to examine

    Returns
    -------
    bool
    Is the directory is a project from Darwin?
    """
    return (project_path / "annotations").exists() and (project_path / "images").exists()


def prompt(msg: str, default: Optional[str] = None) -> str:
    """Prompt the user on a CLI to input a message

    Parameters
    ----------
    msg : str
        Message to print
    default : str
        Default values which is put between [] when the user is prompted

    Returns
    -------
    str
    The input from the user or the default value provided as parameter if user does not provide one
    """
    if default:
        msg = f"{msg} [{default}]: "
    else:
        msg = f"{msg}: "
    result = input(msg)
    if not result and default:
        return default
    return result


def find_files(
    root: Optional[Path] = None,
    files_list: Optional[List[str]] = None,
    recursive: Optional[bool] = True,
    exclude: Optional[List[str]] = None,
) -> List[Path]:
    """Retrieve a list of all files belonging to supported extensions. The exploration can be made
    recursive and a list of files can be excluded if desired.

    Parameters
    ----------
    root : Path
        Path to the root folder to explore
    files_list: list[str]
        List of files that will be filtered with the supported file extensions and returned
    recursive : bool
        Flag for recursive search
    exclude : list[str]
        List of files to exclude from the search

    Returns
    -------
    list[Path]
    List of all files belonging to supported extensions. Can't return None.
    """
    if files_list is None and root is None:
        raise ValueError(f"Invalid combined value for root (None) and files_list (None)")

    # Init the return value
    files = []
    if files_list is not None:
        files.extend([Path(f) for f in files_list
                      if Path(f).suffix in SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS])

    if root is not None:
        if not root.is_dir():
            raise ValueError(f"Root is not a directory ({root}).")
        # Scan for files at the chosen directory
        base = "**/*" if recursive else "*"
        pattern = [base + extension
                   for extension in SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS]
        files.extend([f for p in pattern for f in root.glob(p)])

    # Filter the list and return it
    if exclude is None:
        exclude = []
    return [f for f in files if f.name not in exclude and f not in exclude]


def secure_continue_request() -> bool:
    """Asks for explicit approval from the user. Empty string not accepted"""
    return input("Do you want to continue? [y/N] ") in ["Y", "y"]


def persist_client_configuration(client: "Client", config_path: Optional[Path] = None) -> Config:
    """Authenticate user against the server and creates a configuration file for it

    Parameters
    ----------
    client : Client
        Client to take the configurations from
    config_path : Path
        Optional path to specify where to save the configuration file

    Returns
    -------
    Config
    A configuration object to handle YAML files
    """
    if not config_path:
        config_path = Path.home() / ".darwin" / "config.yaml"
        config_path.parent.mkdir(exist_ok=True)

    default_config = {
        "token": client.token,
        "refresh_token": client.refresh_token,
        "api_endpoint": client.url,
        "base_url": client.base_url,
        "projects_dir": str(client.projects_dir),
    }

    return Config(config_path, default_config)
