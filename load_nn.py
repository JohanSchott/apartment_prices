#!/usr/bin/env python3

"""

load_nn
=======

This module contains a function to load trained neural networks from file.

"""

import numpy as np
import h5py

from .nn import Model


def load_nn_model_from_file(filename):
    """
    Load neural network model from a .h5 file.
    """
    h5f = h5py.File(filename,'r')
    # Convert data sets to a dictionary
    data = {}
    for key, value in h5f.items():
        data[key] = np.array(value)
    # Convert attributes to a dictionary
    attributes = {}
    for key, value in h5f.attrs.items():
        attributes[key] = value
    h5f.close()
    # Create Model object
    model = Model(data, attributes)
    return model


