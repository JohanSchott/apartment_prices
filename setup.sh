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
    conda create -y --name envMap python=3.7
else
    # Linux (and perhaps Windows)
    # Only create environment if it does not already exist.
    test -d ~/envMap || virtualenv -p python3.8 ~/envMap
fi

# Activate virtual environment and append to PYTHONPATH.
source env.sh

# Install Python libraries.
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    conda update -n base -c defaults conda
    conda install -y --file requirements_MAC_OS.txt
    pip install pytest-pythonpath==0.7.3
    pip install geopy==2.0.0
else
    # Linux (and perhaps Windows)
    pip install -U pip==20.0.2
    pip install pip-compile-multi
    pip-compile requirements.in
    pip install -r requirements.txt
fi

# Run unit-tests
pytest

