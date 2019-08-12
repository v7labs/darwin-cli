from __future__ import annotations

import datetime
import shutil
from pathlib import Path

import darwin
from darwin.utils import SUPPORTED_IMAGE_EXTENSIONS


class LocalDataset:
    def __init__(self, project_path: Path, client: darwin.client.Client):
        self.project_path = project_path
        self.name = project_path.name
        # TODO is this intended? both name and slug get 'project_path.name'
        self.slug = project_path.name
        self._client = client

    @property
    def image_count(self) -> int:
        return sum(1 for p in (self.project_path / "images").glob("*") if p.suffix in SUPPORTED_IMAGE_EXTENSIONS)

    @property
    def disk_size(self) -> int:
        return sum(path.stat().st_size for path in self.project_path.glob("**"))

    @property
    def sync_date(self) -> datetime.datetime:
        timestamp = self.project_path.stat().st_mtime
        return datetime.datetime.fromtimestamp(timestamp)

    def remove(self):
        shutil.rmtree(self.project_path)
