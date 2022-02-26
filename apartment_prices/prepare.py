#!/usr/bin/env python3

"""

preparation
===========

This module contains functions for pre-processing example data.

"""

import numpy as np
import pandas as pd
import h5py

# Local stuff
from apartment_prices import time_stuff
from apartment_prices import location


def read_csv_and_save_processed_data_to_hdf5(csv_file):
    hdf5_file = csv_file[:-3] + "hdf5"
    x, y, normal_apartment_indices, y_label, features = load_and_setup_data(csv_file)
    write_to_hdf5(hdf5_file, x, y, features, y_label)


def load_apartment_data(filename_hdf5, wanted_features=None):
    """Load apartment data"""
    print("Open {} and extract data.".format(filename_hdf5))
    x, y, features, y_label = read_from_hdf5(filename_hdf5)
    if wanted_features is not None:
        assert np.all(wanted_features == features)
    return x, y, features, y_label


def write_to_hdf5(filename, x, y, features, y_label):
    asciiList = [n.encode("ascii", "ignore") for n in features]
    ascii_y_label_as_list = [y_label.encode("ascii", "ignore")]
    file_handle = h5py.File(filename, "w")
    file_handle.create_dataset("x", data=x)
    file_handle.create_dataset("y", data=y)
    file_handle.create_dataset("features", data=asciiList)
    file_handle.create_dataset("y_label", data=ascii_y_label_as_list)
    file_handle.close()


def read_from_hdf5(filename):
    file_handle = h5py.File(filename, "r")
    x = file_handle["x"][()]
    y = file_handle["y"][()]
    features = file_handle["features"][()]
    y_label = file_handle["y_label"][()]
    file_handle.close()
    features = features.astype(str)
    y_label = y_label.astype(str)[0]
    return x, y, features, y_label


def load_and_setup_data(csv_filename, features=None):
    """
    Return data needed for the machine learning model.

    Parameters
    ----------
    filename : str
    features : list

    """
    y_label = "soldPrice"
    if features is None:
        # Select features (among apartment property labels)
        features = np.array(
            [
                "soldDate",
                "livingArea",
                "rooms",
                "rent",
                "floor",
                "constructionYear",
                "latitude",
                "longitude",
                "distance2SthlmCenter",
            ]
        )
    # Read data apartment data from .csv file.
    df = pd.read_csv(csv_filename)
    # Apartment labels
    labels = np.array(df.columns[1:])
    # Apartment data
    apartments = df.values[:, 1:]
    print("All apartment properties:")
    for label in labels:
        print(label)
    print()
    # Get selected data
    x_raw, y_raw, normal_apartment_indices = setup_data(apartments, labels, features, y_label)
    return x_raw, y_raw, normal_apartment_indices, y_label, features


def setup_data(apartments, labels, features, y_label):
    """
    Return input and output data for final price predication.

    Replaces missing values in the input with the average
    of that feature.

    Parameters
    ----------
    apartments : array(M,N)
        M is the number of apartment examples.
        N is the number of apartment properties.
    labels : array(N)
        Descriptions of all apartment properties.
    features : array(K)
        Descriptions of all the features in the return matrix x.
    y_label : str
        Descriptions of the output property in the return vector y.

    Returns
    -------
    x : array(K,P)
        Feature matrix.
    y : array(P)
        Output vector.

    """
    # Feature data
    x = []
    normal_apartments = []
    k_lati = np.where(labels == "latitude")[0][0]
    k_long = np.where(labels == "longitude")[0][0]
    # Loop over all apartments
    for i in range(np.shape(apartments)[0]):
        # Resonable values: exclude apartments that are weird.
        # E.g. apartments that are either too small, too big,
        # too cheap, too expensive,
        # have too big price change from the starting price,
        # too high rent
        # or are strange is some other way.
        j = np.where(labels == "livingArea")[0][0]
        if float(apartments[i, j]) < 10:
            # Skip this apartment, too small
            continue
        if float(apartments[i, j]) > 150:
            # Skip this apartment, too big
            continue
        j = np.where(labels == "soldPrice")[0][0]
        if float(apartments[i, j]) < 0.5 * 10**6:
            # Skip this apartment, too cheap
            continue
        if float(apartments[i, j]) > 20 * 10**6:
            # Skip this apartment, too expensive
            continue
        j2 = np.where(labels == "listPrice")[0][0]
        if float(apartments[i, j2]) != 0:
            if float(apartments[i, j]) / float(apartments[i, j2]) < 0.6:
                # Skip this apartment, too big decrease
                continue
            if float(apartments[i, j]) / float(apartments[i, j2]) > 2.5:
                # Skip this apartment, too big increase
                continue
        else:
            # Skip this apartment, zero list price is a bit weird...
            continue
        j = np.where(labels == "rent")[0][0]
        if float(apartments[i, j]) > 20000:
            # Skip this apartment, too high rent
            continue
        j = np.where(labels == "rooms")[0][0]
        if float(apartments[i, j]) > 15:
            # Skip this apartment, too many rooms
            continue
        j = np.where(labels == "floor")[0][0]
        if float(apartments[i, j]) > 36:
            # Skip this apartment, too high floor.
            # According to wiki, the most number of floors in Stockholm
            # is at the moment (2019) 36 floors.
            continue
        apartment = np.zeros(len(features), dtype=np.float)
        for j, feature in enumerate(features):
            if feature in labels:
                k = np.where(labels == feature)[0][0]
                if feature == "soldDate":
                    year = int(apartments[i, k][:4])
                    month = int(apartments[i, k][5:7])
                    day = int(apartments[i, k][8:9])
                    apartment[j] = time_stuff.get_time_stamp(year, month, day)
                else:
                    apartment[j] = float(apartments[i, k])
            elif feature == "distance2SthlmCenter":
                apartment[j] = location.distance_2_sthlm_center(
                    float(apartments[i, k_lati]), float(apartments[i, k_long])
                )
            # elif feature == 'sizePerRoom':
            # x_raw, features = preparation.add_size_per_room_as_feature(x_raw, features)
            else:
                raise Exception("Feature " + feature + " does not exist...")
        # An apartment reaching this point is considered normal
        normal_apartments.append(i)
        x.append(apartment)

    normal_apartments = np.array(normal_apartments)
    x = np.array(x)
    # Output index
    y_label_index = np.where(labels == y_label)[0][0]
    # Output vector
    y = np.array(apartments[normal_apartments, y_label_index], dtype=np.float)
    print("{:d} apartments are un-normal and are excluded.".format(len(apartments) - len(normal_apartments)))

    # Transpose for later convinience
    x = x.T
    replace_missing_values(x)
    return x, y, normal_apartments


def replace_missing_values(x):
    """
    Update input parameter such that missing values are
    replaced with the feature average.
    """
    # Number of features and examples
    n, m = np.shape(x)
    s = 0
    for i in range(n):
        mask = np.isnan(x[i, :])
        s += np.sum(mask)
        x[i, mask] = np.mean(x[i, np.logical_not(mask)])
    if s > 0:
        print("{:d} NaN values are replaced with feature mean values".format(s))
