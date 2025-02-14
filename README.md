# V7 Darwin Python SDK

[![Downloads](https://static.pepy.tech/personalized-badge/darwin-py?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/darwin-py) [![Downloads](https://static.pepy.tech/personalized-badge/darwin-py?period=month&units=international_system&left_color=black&right_color=blue&left_text=This%20month)](https://pepy.tech/project/darwin-py) [![GitHub Repo stars](https://img.shields.io/github/stars/v7labs/darwin-py?style=social)](https://github.com/v7labs/darwin-py/stargazers)
[![Twitter Follow](https://img.shields.io/twitter/follow/V7Labs?style=social)](https://twitter.com/V7Labs)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/v7labs/darwin-py/badge)](https://scorecard.dev/viewer/?uri=github.com/v7labs/darwin-py)

⚡️ Official library to annotate, manage datasets, and models on
[V7's Darwin Training Data Platform](https://darwin.v7labs.com). ⚡️

Darwin-py can both be used from the [command line](#usage-as-a-command-line-interface-cli) and as a [python library](#usage-as-a-python-library).

<hr/>

Main functions are (but not limited to):

-   Client authentication
-   Listing local and remote datasets
-   Create/remove datasets
-   Upload/download data to/from remote datasets
-   Direct integration with PyTorch dataloaders
-   Extracting video artifacts

Support tested for python 3.9 - 3.12

## 🏁 Installation

```
pip install darwin-py
```

You can now type `darwin` in your terminal and access the command line interface.

If you wish to use the PyTorch bindings, then you can use the `ml` flag to install all the additional requirements

```
pip install darwin-py[ml]
```

If you wish to use video frame extraction, then you can use the `ocv` flag to install all the additional requirements

```
pip install darwin-py[ocv]
```

If you wish to use video artifacts extraction, then you need to install [FFmpeg](https://www.ffmpeg.org/download.html)

To run test, first install the `test` extra package

```
pip install darwin-py[test]
```

### Configuration

#### Retry Configuration

The SDK includes a retry mechanism for handling API rate limits (429) and server errors (500, 502, 503, 504). You can configure the retry behavior using the following environment variables:

- `DARWIN_RETRY_INITIAL_WAIT`: Initial wait time in seconds between retries (default: 60)
- `DARWIN_RETRY_MAX_WAIT`: Maximum wait time in seconds between retries (default: 300)
- `DARWIN_RETRY_MAX_ATTEMPTS`: Maximum number of retry attempts (default: 10)

Example configuration:
```bash
# Configure shorter retry intervals and fewer attempts
export DARWIN_RETRY_INITIAL_WAIT=30
export DARWIN_RETRY_MAX_WAIT=120
export DARWIN_RETRY_MAX_ATTEMPTS=5
```

The retry mechanism will automatically handle:
- Rate limiting (HTTP 429)
- Server errors (HTTP 500, 502, 503, 504)

For each retry attempt, you'll see a message indicating the type of error and the wait time before the next attempt.

### Development

See our development and QA environment installation recommendations [here](docs/DEV.md)

---

## Usage as a Command Line Interface (CLI)

[Here you can find V7 labs doc on the CLI usage](https://docs.v7labs.com/docs/getting-started-1)

Once installed, `darwin` is accessible as a command line tool.
A useful way to navigate the CLI usage is through the help command `-h/--help` which will
provide additional information for each command available.

### Client Authentication

To perform remote operations on Darwin you first need to authenticate.
This requires a [team-specific API-key](https://darwin.v7labs.com/?settings=api-keys).
If you do not already have a Darwin account, you can [contact us](https://www.v7labs.com/contact) and we can set one up for you.

To start the authentication process:

```
$ darwin authenticate
API key:
Make example-team the default team? [y/N] y
Datasets directory [~/.darwin/datasets]:
Authentication succeeded.
```

You will be then prompted to enter your API-key, whether you want to set the corresponding team as
default and finally the desired location on the local file system for the datasets of that team.
This process will create a configuration file at `~/.darwin/config.yaml`.
This file will be updated with future authentications for different teams.

### Listing local and remote datasets

Lists a summary of local existing datasets

```
$ darwin dataset local
NAME            IMAGES     SYNC_DATE         SIZE
mydataset       112025     yesterday     159.2 GB
```

Lists a summary of remote datasets accessible by the current user.

```
$ darwin dataset remote
NAME                       IMAGES     PROGRESS
example-team/mydataset     112025        73.0%
```

### Create/remove a dataset

To create an empty dataset remotely:

```
$ darwin dataset create test
Dataset 'test' (example-team/test) has been created.
Access at https://darwin.v7labs.com/datasets/579
```

The dataset will be created in the team you're authenticated for.

To delete the project on the server:

```
$ darwin dataset remove test
About to delete example-team/test on darwin.
Do you want to continue? [y/N] y
```

### Upload/download data to/from a remote dataset

Uploads data to an existing remote project.
It takes the dataset name and a single image (or directory) with images/videos to upload as
parameters.

The `-e/--exclude` argument allows to indicate file extension/s to be ignored from the data_dir.
e.g.: `-e .jpg`

For videos, the frame rate extraction rate can be specified by adding `--fps <frame_rate>`

Supported extensions:

-   Video files: [`.mp4`, `.bpm`, `.mov` formats].
-   Image files [`.jpg`, `.jpeg`, `.png` formats].

```
$ darwin dataset push test /path/to/folder/with/images
100%|████████████████████████| 2/2 [00:01<00:00,  1.27it/s]
```

Before a dataset can be downloaded, a release needs to be generated:

```
$ darwin dataset export test 0.1
Dataset test successfully exported to example-team/test:0.1
```

This version is immutable, if new images / annotations have been added you will have to create a new release to included them.

To list all available releases

```
$ darwin dataset releases test
NAME                           IMAGES     CLASSES                   EXPORT_DATE
example-team/test:0.1               4           0     2019-12-07 11:37:35+00:00
```

And to finally download a release.

```
$ darwin dataset pull test:0.1
Dataset example-team/test:0.1 downloaded at /directory/choosen/at/authentication/time .
```

---

## Usage as a Python library

[Here you can find V7 labs doc on the usage as Python library](https://docs.v7labs.com/docs/install-update-the-darwin-sdk)

The framework is designed to be usable as a standalone python library.
Usage can be inferred from looking at the operations performed in `darwin/cli_functions.py`.
A minimal example to download a dataset is provided below and a more extensive one can be found in

[./darwin_demo.py](https://github.com/v7labs/darwin-py/blob/master/darwin_demo.py).

```python
from darwin.client import Client

client = Client.local() # use the configuration in ~/.darwin/config.yaml
dataset = client.get_remote_dataset("example-team/test")
dataset.pull() # downloads annotations and images for the latest exported version
```

Follow [this guide](https://docs.v7labs.com/docs/loading-a-dataset-in-python) for how to integrate darwin datasets directly in PyTorch.
