#!/bin/bash

set -e  # Exit on error

# Update and install dependencies
sudo apt update -y
sudo apt upgrade -y
sudo apt-get install -y python3-openssl
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init --path)"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile

# Apply changes
source ~/.profile

# Install Python 3.10 using pyenv
pyenv install 3.10.0
pyenv global 3.10.0
pyenv local 3.10

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Initialize Poetry environment
source /workspaces/darwin-py/.venv/bin/activate #poetry shell
poetry install

# Install additional Python dependencies
pip install darwin-py[test]
pip install darwin-py[ml]
pip install darwin-py[ocv]

pip install connected-components-3d
pip install nibabel

# FFmpeg
FFMPEG_VERSION="6.0"  # Ensure this is version 5 or higher
mkdir -p $HOME/.local/bin
echo "Downloading FFmpeg $FFMPEG_VERSION..."
cd $HOME/.local/bin
wget -O ffmpeg.tar.xz "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
tar -xf ffmpeg.tar.xz --strip-components=1
rm ffmpeg.tar.xz
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
cd -

#Run tests to verify all is good
pytest

echo "Setup complete!"