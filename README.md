# Darwin
Official library to manage datasets along with v7 Darwin annotation platform [https://darwin.v7labs.com](https://darwin.v7labs.com).

Support tested for python3.7.

## Installation

### Standard

```
pip install darwin-py
```
You can now type `darwin` in your terminal and access the command line interface.

### Development
After cloning the repository:

```
pip install --editable .
```

## Usage

Darwin can be used as a python library or as a command line tool.
Main functions are:

- Client authentication
- Listing local and remote datasets
- Create/remove a dataset 
- Upload/download data to/from a remote dataset

---

### As a library

Darwin can be used as a python library.

#### Client Authentication 
To access darwin you first need to authenticate, this can be done once by creating a configuration 
file, or every time by providing credentials at the login.

To create the configuration file first authenticate without ~/.darwin/config.yaml file 
(which can also be generated with the CLI):

```python
from darwin.client import Client

client = Client.login(email="simon@v7labs.com", password="*********")
```

Then, persist the configuration file with:

```python
from darwin.utils import persist_client_configuration

persist_client_configuration(client)   
```

Afterwards, calling the creation of Client with default parameters will load the configurations
previously saved:

```python
from darwin.client import Client

client = Client.local()
```


#### Listing local and remote datasets

Print a list of local existing projects

```python
from darwin.client import Client

client = Client.local()
for p in client.list_local_datasets():
    print(p.name)
```

Print a list of remote projects accessible by the current user.
Note: the list will include only those datasets which belong in the team where the user is currently 
authenticated. 

```python
from darwin.client import Client

client = Client.local()
for dataset in client.list_remote_datasets():
    print(dataset.slug, dataset.image_count)
```


#### Create/remove a dataset 

Dataset creation is handled by the client:

```python
from darwin.client import Client

client = Client.local()
client.create_dataset(name="This Is My New Dataset")
```

Whereas dataset removal is handled directly by the dataset itself:

```python
from darwin.client import Client

client = Client.local()
dataset = client.get_remote_dataset(slug="this-is-my-new-dataset")
dataset.remove_remote()
```


#### Upload/download data to/from a remote dataset

To upload data to an existing remote project there are several solutions.
A simple one is to update the local folder where the dataset is located and then upload the
files to the remote dataset. 

```python
from darwin.client import Client

client = Client.local()
dataset = client.get_remote_dataset(slug="example-dataset")
progress = dataset.push()
```

Note that `dataset.push()` takes an optional parameter `source_folder` with which is possible
to specify another location from which fetch the images to upload.
To download a remote project, images and annotations, in the projects directory 
(specified in the authentication process [default: ~/.darwin/projects]).

```python
from darwin.client import Client

client = Client.local()
dataset = client.get_remote_dataset(slug="example-dataset")
dataset.pull()
```


---

### Command line

`darwin` is also accessible as a command line tool.


#### Client Authentication 

A username (email address) and password is required to authenticate. 
If you do not already have a Darwin account, register for free at [https://darwin.v7labs.com](https://darwin.v7labs.com).
```
$ darwin authenticate
Username (email address): simon@v7labs.com
Password: *******
Project directory [~/.darwin/projects]: 
Projects directory created /Users/simon/.darwin/projects
Authentication succeeded.
```


#### Listing local and remote datasets 

Lists a summary of local existing projects
```
$ darwin local
NAME                IMAGES     SYNC DATE          SIZE
example-project          3         today      800.2 kB
```

Lists a summary of remote projects accessible by the current user.

```
$ darwin remote
NAME                 IMAGES     PROGRESS     ID
example-project           3         0.0%     89
```


#### Create/remove a dataset 

Creates an empty dataset remotely.

```
$ darwin create example-dataset
Dataset 'example-project' has been created.
Access at https://darwin.v7labs.com/datasets/example-project
``` 

To delete the project on the server add the `-r` /`--remote` flag
```
$ darwin remove example-project --remote
About to deleting example-project on darwin.
Do you want to continue? [y/N] y
```


#### Upload/download data to/from a remote dataset 

Uploads data to an existing remote project.
It takes the project name and a single image (or directory) with images/videos to upload as parameters. 

The `-e/--exclude` argument allows to indicate file extension/s to be ignored from the data_dir. E.g.: `-e .jpg`

For videos, the frame rate extraction rate can be specified by adding `--fps <frame_rate>`

Supported extensions:
-  Video files: [`.mp4`, `.bpm`, `.mov` formats].
-  Image files [`.jpg`, `.jpeg`, `.png` formats].

```
$ darwin push example-dataset -r path/to/images
Uploading: 100%|########################################################| 3/3 [00:01<00:00,  2.29it/s]
```

Downloads a remote project, images and annotations, in the projects directory 
(specified in the authentication process [default: `~/.darwin/projects`]).

```
$ darwin pull example-project
Pulling project example-project:latest
Downloading: 100%|########################################################| 3/3 [00:03<00:00,  4.11it/s]
```


## Table of Arguments

| parser          | parameter                | type               | required  |
| --------------- | ------------------------ | -----------------  | --------- |
| `authenticate`  |                          |                    |           |
| `team`          |                          |                    |           |
|                 | `team_name`              | str                | False     |
|                 | `-l`, `--list`           |                    | False     |
| `create`        | `project_name`           | str                | True      |
| `local`         |                          |                    |           |
| `path`          | `project_name`           | str/int            | True      |
| `pull`          | `project_name`           | str/int            | True      |
| `remote`        |                          | str                |           |
| `remove`        | `project_name`           | str                | True      |
|                 | `-r` `--remote`          | str                | True      |
| `url`           | `project_name`           | str                |           |
| `push`          | `project_name`           | str                | True      |
|                 | `files`                  | str                | True      |
|                 | `-e`, `--exclude`        | str                |           |
|                 | `--fps`                  | int                |           |