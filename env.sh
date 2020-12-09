#!/bin/bash

# Activate environment
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific things here
    conda activate envMap
    # Deactivate with: conda deactivate
else
    # Linux (and perhaps Windows)
    source ~/envMap/bin/activate
    # Decativate with: deactivate
fi

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$DIR

