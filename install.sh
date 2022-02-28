#!/bin/bash -e

# Change to script folder.
cd "$( dirname "${BASH_SOURCE[0]}" )"

# System libraries
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    brew update
    #brew upgrade
    brew install proj
    brew install geos
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Only for Debian, Ubuntu, and related Linux distributions
    sudo apt-get update -qq
    sudo apt-get install -qq -y --no-install-recommends $(cat requirements-ubuntu.txt)
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    echo "Install 32 bits Windows NT specific things here"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    echo "Install 64 bits Windows NT specific things here"
fi

test -d ~/envMap || python3.9 -m venv ~/envMap
. ~/envMap/bin/activate

pip install --disable-pip-version-check -q -U pip==22.0.3
pip install --disable-pip-version-check -q pip-tools==6.5.1
rm -f requirements.txt
if [ "$(uname)" == "Darwin" ]; then
    pip-compile -q requirements-OSX.in --output-file requirements.txt
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    pip-compile -q requirements.in
fi
pip install --disable-pip-version-check -q -r requirements.txt
