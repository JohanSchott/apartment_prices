"""
This module contains a neural network class.
"""


import h5py
import numpy as np

# Tensorflow libraries
import tensorflow as tf
from keras import regularizers
from keras.layers import Activation, Dense
from keras.models import Sequential, load_model

# Local libraries
from apartment_prices import nn


class Model_tf:
    """
    Basically a wrapper class around TensorFlow model.
    """

    def __init__(self, ai_name: str, model_design:dict|None=None, attributes:dict|None=None):
        """
        Create a new or load an existing AI model.
        """
        self.ai_name = ai_name
        if model_design is None and attributes is None:
            # Load an existing model
            self.load_existing_model()
        elif model_design is not None and attributes is not None:
            # Create new model
            self.create_new_model(model_design.copy(), attributes.copy())
        else:
            raise Exception("Wrong input parameters")
        # Sanity checks
        assert "activation_type" in self.model_design
        assert "layers" in self.model_design
        assert "predict_log_value" in self.model_design
        assert "features" in self.attributes
        assert len(self.attributes["features"]) == self.model_design["layers"][0]

    def create_new_model(self, model_design, attributes):
        self.model_design = model_design
        self.attributes = attributes
        # Will keep some performance result
        self.data = {}
        # Create Tensorflow model object
        model = Sequential()
        self.model = model
        for i, layer in enumerate(model_design["layers"][1:]):
            if i == 0:
                # First hidden layer
                model.add(
                    Dense(
                        layer,
                        kernel_regularizer=regularizers.l2(model_design["gamma"]),
                        input_dim=model_design["layers"][0],
                    )
                )
                model.add(Activation(model_design["activation_type"]))
            elif i == len(model_design["layers"][1:]) - 1:
                # Last layer
                assert layer == 1
                # No activation function on the last layer
                model.add(Dense(layer, kernel_regularizer=regularizers.l2(model_design["gamma"])))
            else:
                model.add(Dense(layer, kernel_regularizer=regularizers.l2(model_design["gamma"])))
                model.add(Activation(model_design["activation_type"]))
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            # For a mean squared error regression problem
            # self.model.compile(optimizer='rmsprop', loss='mse')
            model.compile(optimizer=optimizer, loss="mse")
            # self.model.compile(optimizer=optimizer, loss='mae')

    def load_existing_model(self):
        """
        Load model from disk.
        """
        filename_model_tf, filename_meta = self.get_filenames()
        # Load Tensorflow model
        self.model = load_model(filename_model_tf)
        # Load meta data
        self.model_design = {}
        self.attributes = {}
        self.data = {}
        # Read the meta-file
        with h5py.File(filename_meta, "r") as meta_file:
            group = meta_file["model_design"]
            for key, value in group.items():
                self.model_design[key] = value[()]
            group = meta_file["attributes"]
            for key, value in group.attrs.items():
                self.attributes[key] = value
            group = meta_file["data"]
            for key, value in group.items():
                self.data[key] = value[()]
        # Sanity checks
        if "mu_x" in self.data:
            assert self.data["mu_x"].shape[1] == self.model_design["layers"][0]
        if "std_x" in self.data:
            assert self.data["std_x"].shape[1] == self.model_design["layers"][0]

    def save_to_files(self, verbose):
        """
        Save model to disk.
        """
        filename_model_tf, filename_meta = self.get_filenames()
        if verbose:
            print("Save NN model to files:", filename_model_tf, filename_meta)
        # Save tensorflow model
        self.model.save(filename_model_tf)
        # Save meta data
        with h5py.File(filename_meta, "w") as meta_file:
            # Save model design parameters
            group = meta_file.create_group("model_design")
            for key, value in self.model_design.items():
                group.create_dataset(key, data=value)
            # Save attributes (each a list of strings)
            group = meta_file.create_group("attributes")
            for key, value in self.attributes.items():
                group.attrs.create(key, value, dtype=h5py.special_dtype(vlen=str))
            # Save some performance results and meta data.
            group = meta_file.create_group("data")
            for key, value in self.data.items():
                group.create_dataset(key, data=value)

    def get_filenames(self):
        filename_model_tf = "models/" + self.ai_name + "_tf.hdf5"
        filename_meta = "models/" + self.ai_name + "_meta.hdf5"
        return filename_model_tf, filename_meta

    def fit(self, x_train, y_train, x_cv, y_cv, batch_size=10000, epochs=1, verbose=1):
        """
        Train neural network model.

        Parameters
        ----------
        x_train : array(N,M)
        y_train : array(M)
        x_cv : array(N,K)
        y_cv : array(K)
        batch_size : int
            Number of examples in each batch
        epochs : int
            Number of epochs
        verbose : int
            0, 1 or 2.
        """
        # Sanity check
        assert x_train.shape[1] == len(y_train)
        assert x_cv.shape[1] == len(y_cv)
        # Convert to shapes expected by tensorflow
        x_train = x_train.T
        y_train = np.atleast_2d(y_train).T
        x_cv = x_cv.T
        y_cv = np.atleast_2d(y_cv).T

        # If predict_log_value is True, the cost function will be of the form:
        # (ln(m)-ln(y))^2 instead of (m-y)^2.
        # When m approx y, this can be Taylor expanded into ((m-y)/y)^2.
        # Hence the relative error is minimized instead of the absolute error.
        if self.model_design["predict_log_value"]:
            # Fit to logarithm of the output (instead of fitting to the output).
            y_train = np.log(y_train)
            y_cv = np.log(y_cv)
        # Print the number of neural network parameters
        r = nn.get_number_of_NN_parameters(self.model_design["layers"])
        print("{:d} parameters, {:d} training examples".format(r, y_train.shape[0]))
        # Normalization and scaling parameters
        mu_x, std_x = nn.get_norm_and_scale(x_train, axis=0)
        mu_y, std_y = nn.get_norm_and_scale(y_train, axis=0)
        # Set default values
        if "mu_x" not in self.data:
            self.data["mu_x"] = mu_x
        if "std_x" not in self.data:
            self.data["std_x"] = std_x
        if "mu_y" not in self.data:
            self.data["mu_y"] = mu_y
        if "std_y" not in self.data:
            self.data["std_y"] = std_y
        # Normalized and scaled data
        x_train = (x_train - self.data["mu_x"]) / self.data["std_x"]
        x_cv = (x_cv - self.data["mu_x"]) / self.data["std_x"]
        y_train = (y_train - self.data["mu_y"]) / self.data["std_y"]
        y_cv = (y_cv - self.data["mu_y"]) / self.data["std_y"]
        # Fit model to data
        history = self.model.fit(
            x_train, y_train, validation_data=(x_cv, y_cv), batch_size=batch_size, epochs=epochs, verbose=verbose
        )
        # Training and cross-validation cost values
        batch_size = min(10000, x_train.shape[0]) if x_train.ndim == 2 else None
        self.data["cost_train"] = self.model.evaluate(x_train, y_train, batch_size=batch_size)
        batch_size = min(10000, x_cv.shape[0]) if x_cv.ndim == 2 else None
        self.data["cost_cv"] = self.model.evaluate(x_cv, y_cv, batch_size=batch_size)
        return history

    def evaluate(self, x, y, batch_size=None):
        """
        Return cost function value.
        """
        # Sanity checks
        if not (
            "predict_log_value" in self.model_design
            and "mu_x" in self.data
            and "std_x" in self.data
            and "mu_y" in self.data
            and "std_y" in self.data
        ):
            raise Exception("Not all parameters are initioalized...")
        assert x.shape[1] == len(y)
        if batch_size is None and x.ndim == 2:
            batch_size = min(10000, x.shape[1])

        # Convert to shapes expected by tensorflow
        x = x.T
        y = np.atleast_2d(y).T
        if x.ndim == 1:
            x = np.atleast_2d(x)
        elif x.ndim > 2:
            raise Exception("Unexpected input dimension of apartment features")
        # If predict_log_value is True, the cost function will be of the form:
        # (ln(m)-ln(y))^2 instead of (m-y)^2.
        # When m approx y, this can be Taylor expanded into ((m-y)/y)^2.
        # Hence the relative error is minimized instead of the absolute error.
        if self.model_design["predict_log_value"]:
            # Fit to logarithm of the output (instead of fitting to the output).
            y = np.log(y)
        # Normalize and scale input
        x_norm = nn.norm_and_scale(x, self.data["mu_x"], self.data["std_x"])
        y_norm = nn.norm_and_scale(y, self.data["mu_y"], self.data["std_y"])
        return self.model.evaluate(x_norm, y_norm, batch_size=batch_size)

    def predict(self, x, batch_size=None):
        """
        Return prediction from neural network model.

        Parameters
        ----------
        x : real array(N,M) or array(N) or list(N)
            Number of features N.
            Number of data examples M.

        """
        # Convert to shapes expected by tensorflow
        x = np.array(x).T
        if x.ndim == 1:
            x = np.atleast_2d(x)
        elif x.ndim > 2:
            raise Exception("Unexpected input dimension of apartment features")
        if batch_size is None and x.ndim == 2:
            batch_size = min(10000, x.shape[1])
        # Sanity checks
        if not (
            "predict_log_value" in self.model_design
            and "mu_x" in self.data
            and "std_x" in self.data
            and "mu_y" in self.data
            and "std_y" in self.data
        ):
            raise Exception("Not all parameters are initioalized...")
        # Normalize and scale input
        x_norm = nn.norm_and_scale(x, self.data["mu_x"], self.data["std_x"])
        y_norm = self.model.predict(x_norm, batch_size=batch_size)
        # Go back to un-normalized and un-scaled output
        y = y_norm * self.data["std_y"] + self.data["mu_y"]
        if self.model_design["predict_log_value"]:
            y = np.exp(y)
        y = y.flatten()
        if y.size == 1:
            y = y[0]
        return y
