#!/bin/bash -ex

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

if [ "$(uname)" == "Darwin" ]; then
    echo "Install Mac OS X specific things here"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Only for Debian, Ubuntu, and related Linux distributions
    sudo apt-get install -y --no-install-recommends $(cat requirements-ubuntu.txt)
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    echo "Install 32 bits Windows NT specific things here"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    echo "Install 64 bits Windows NT specific things here"
fi

# Create virtual environment. But only if it does not already exist.
test -d ~/envMap || virtualenv -p python3 ~/envMap

## Activate virtual environment and append to PYTHONPATH.
source env.sh

## Install required python libraries.
pip install -U pip==20.0.2
pip install pip-compile-multi
pip-compile requirements.in
pip install -r requirements.txt

## Run unit-tests
pytest

