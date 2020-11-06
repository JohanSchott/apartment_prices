#!/bin/bash -ex

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# System libraries
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    brew install proj
    brew install geos
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Only for Debian, Ubuntu, and related Linux distributions
    sudo apt-get install -y --no-install-recommends $(cat requirements-ubuntu.txt)
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    echo "Install 32 bits Windows NT specific things here"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    echo "Install 64 bits Windows NT specific things here"
fi

# Create virtual environment
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    # Do not manage to install Python library cartopy on Python3.8
    conda create -y --name envMap python=3.7
else
    # Linux (and perhaps Windows)
    # Only create environment if it does not already exist.
    test -d ~/envMap || virtualenv -p python3.8 ~/envMap
fi

# Activate virtual environment and append to PYTHONPATH.
source env.sh

# Install Python libraries.
pip install -U pip==20.0.2
pip install pip-compile-multi
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    conda update -n base -c defaults conda
    pip-compile requirements-MAC-OS-pip.in
    pip install -r requirements-MAC-OS-pip.txt
    conda install -y --file requirements-MAC-OS-conda.txt
else
    # Linux (and perhaps Windows)
    pip-compile requirements.in
    pip install -r requirements.txt
fi

# Run unit-tests
pytest

