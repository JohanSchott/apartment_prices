#!/bin/bash -e

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Only for Debian, Ubuntu, and related Linux distributions
    sudo apt-get update -qq
    sudo apt-get install -qq -y --no-install-recommends $(cat requirements-ubuntu.txt)
fi

rm -rf ~/envMap
python3 -m venv ~/envMap
. ~/envMap/bin/activate
# Install required python libraries.
pip install --disable-pip-version-check -q -U uv==0.2.9
uv pip install -q -r requirements.in

echo "Installation successful!"