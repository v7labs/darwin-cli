import argparse
import sys

import argcomplete


class Options(object):
    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description="Commandline tool to create/upload/download datasets on darwin."
        )

        subparsers = self.parser.add_subparsers(dest="command")
        parse_help = subparsers.add_parser("help", help="Show this help message and exit.")

        # AUTHENTICATE
        parser_authenticate = subparsers.add_parser("authenticate", help="Authenticate the user. ")

        # SELECT TEAM
        parser_create = subparsers.add_parser("team", help="List or pick teams. ")
        parser_create.add_argument("team_name", nargs="?", type=str, help="Team name to use. ")
        parser_create.add_argument(
            "-c",
            "--current",
            action="store_true",
            required=False,
            help="Shows only the current team. ",
        )
        # PROJECT REMOTE

        dataset = subparsers.add_parser(
            "dataset",
            help="Dataset related functions",
            description="Arguments to interact with datasets",
        )

        dataset_action = dataset.add_subparsers(dest="action")

        parser_remote = dataset_action.add_parser("remote", help="List remote datasets")
        parser_remote.add_argument("-t", "--team", help="Specify team")
        parser_remote.add_argument(
            "-a", "--all", action="store_true", help="List datasets for all teams"
        )

        parser_local = dataset_action.add_parser("local", help="List downloaded datasets")

        parser_create = dataset_action.add_parser("create", help="Creates a new dataset on darwin")
        parser_create.add_argument("dataset_name", type=str, help="Dataset name")
        parser_create.add_argument("-t", "--team", help="Specify team")

        parser_path = dataset_action.add_parser("path", help="Print local path to dataset")
        parser_path.add_argument("dataset", type=str, help="Dataset name")

        parser_url = dataset_action.add_parser("url", help="Print url to dataset on darwin")
        parser_url.add_argument("dataset", type=str, help="Dataset name")

        parse_help = dataset_action.add_parser("help", help="Show this help message and exit.")

        parser_push = dataset_action.add_parser(
            "push", help="Upload data to an existing (remote) dataset."
        )
        parser_push.add_argument(
            "dataset",
            type=str,
            help="[Remote] Dataset name: to list all the existing dataset, run 'darwin dataset remote'. ",
        )
        parser_push.add_argument("files", type=str, nargs="+", help="Files to upload")

        parser_push.add_argument(
            "-e",
            "--exclude",
            type=str,
            nargs="+",
            default="",
            help="Excludes the files with the specified extension/s if a data folder is provided as data path. ",
        )
        parser_push.add_argument(
            "-f",
            "--fps",
            type=int,
            default="1",
            help="Frames per second for video split (recommended: 1).",
        )

        parser_remove = dataset_action.add_parser(
            "remove", help="Remove a remote or remote and local dataset."
        )
        parser_remove.add_argument("dataset", type=str, help="Remote dataset name to delete.")

        parser_report = dataset_action.add_parser("report", help="Report about the annotators ")
        parser_report.add_argument("dataset", type=str, help="Remote dataset name to report on.")

        parser_report.add_argument(
            "-g",
            "--granularity",
            choices=["day", "week", "month", "total"],
            help="Granularity of the report",
        )

        parser_dataset_version = dataset_action.add_parser(
            "releases", help="Available version of a dataset."
        )
        parser_dataset_version.add_argument(
            "dataset", type=str, help="Remote dataset name to list."
        )

        parser_dataset_version = dataset_action.add_parser(
            "pull", help="Download a version of a dataset."
        )
        parser_dataset_version.add_argument(
            "dataset", type=str, help="Remote dataset name to download."
        )

        # parser_dataset_version.add_argument(
        #     "--only-zip", type=str, help="Only saves the zip file without expanding it."
        # )

        # VERSION
        parser_version = subparsers.add_parser(
            "version", help="Check current version of the repository. "
        )

        argcomplete.autocomplete(self.parser)

    def parse_args(self):
        args = self.parser.parse_args()
        if not args.command:
            self.parser.print_help()
            sys.exit()
        return args, self.parser
