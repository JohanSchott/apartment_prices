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
    # Follow instructions from https://scitools.org.uk/cartopy/docs/latest/installing.html
    pip install --upgrade pyshp
    pip install shapely --no-binary shapely
    brew install pkg-config
    export PKG_CONFIG_PATH=/usr/local/bin/pkgconfig
    # On my laptop, Cartopy wants to use libproj.22.dylib (which does not exist).
    # I think Cartopy should use libproj.dylib.
    # As a work-around, I created a symbolic link.
    # This issue is discussed in:
    # https://github.com/SciTools/cartopy/issues/823
    # https://stackoverflow.com/questions/56466395/libproj-not-loaded-while-installing-sumo-on-macos#
    # This is the command I ran to create the symbolic link to make Cartopy work:
    # ln -s  /usr/local/opt/proj/lib/libproj.dylib /usr/local/opt/proj/lib/libproj.22.dylib

    pip-compile -q requirements-OSX.in --output-file requirements.txt
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    pip install --disable-pip-version-check -q $(cat requirements.in | grep numpy)
    pip-compile -q requirements.in
fi
pip install --disable-pip-version-check -q -r requirements.txt
