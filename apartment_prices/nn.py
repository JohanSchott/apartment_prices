"""
This module contains a neural network class and some useful neural network functions.
"""

from functools import partial

import h5py
import numpy as np
import scipy.special
from scipy.optimize import minimize

# Machine precision to use
dtype = np.float32
# dtype = np.float64


class Model:
    def __init__(self, data, attributes):
        """

        Parameters
        ----------
        data : dict
        attributes : dict

        """
        # Keep all data and attributes
        self.data = data.copy()
        self.attributes = attributes.copy()
        # Convert all model float arrays to the desired precision.
        for key, value in self.data.items():
            if value.dtype in {np.float64, np.float32}:
                self.data[key] = np.array(value, dtype=dtype)
        # Sanity checks
        assert "activation_type" in data
        assert "layers" in data
        assert "predict_log_value" in data
        assert "features" in attributes
        assert len(attributes["features"]) == data["layers"][0]
        if "mu_x" in data:
            assert len(data["mu_x"]) == data["layers"][0]
        if "std_x" in data:
            assert len(data["std_x"]) == data["layers"][0]

    def save_to_file(self, filename):
        """
        Save model to disk.
        """
        h5f = h5py.File(filename, "w")
        for key, value in self.data.items():
            h5f.create_dataset(key, data=value)
        # Save attributes (each a list of strings)
        for key, value in self.attributes.items():
            h5f.attrs.create(key, value, dtype=h5py.special_dtype(vlen=str))
        h5f.close()

    def train(
        self,
        x_train,
        y_train,
        x_cv,
        y_cv,
        gammas=(0.00001,),
        epochs=1,
        batchsize=30000,
        method="BFGS",
        gtol=1 * 1e-05,
        maxiter=10,
        renormalize=False,
    ):
        """
        Train neural network model.

        Parameters
        ----------
        x_train : array(N,M)
        y_train : array(M)
        x_cv : array(N,K)
        y_cv : array(K)
        gammas : list or array(P)
            Regularization parameters to try.
            Examples:
            gammas = np.array([1, 0.01, 0.0001])
            gammas = np.exp(np.linspace(np.log(1e-4), np.log(2), 5))[::-1]
        epochs : int
            Number of epochs
        batchsize : int
            Number of examples in each batch
        method : str
            Minimization method. Examples: 'CG', 'BFGS'.
        gtol : float
            Gradient convergence tolerence.
        maxiter : int
            Maximum number of minimization iterations per batch.
        renormalize : boolean
            If to renormalize input and output,
            or if use existing normalization values.
        """
        # Cast input parameters to desired precision
        x_train = np.array(x_train, dtype=dtype)
        y_train = np.array(y_train, dtype=dtype)
        x_cv = np.array(x_cv, dtype=dtype)
        y_cv = np.array(y_cv, dtype=dtype)
        gammas = np.array(gammas, dtype=dtype)

        # Cast to numpy array
        gammas = np.array(gammas)

        # If predict_log_value is True, the cost function will be of the form:
        # (ln(m)-ln(y))^2 instead of (m-y)^2.
        # When m approx y, this can be Taylor expanded into ((m-y)/y)^2.
        # Hence the relative error is minimized instead of the absolute error.
        if self.data["predict_log_value"]:
            # Fit to logarithm of the output (instead of fitting to the output).
            y_train = np.log(y_train)
            y_cv = np.log(y_cv)

        r = get_number_of_NN_parameters(self.data["layers"])

        print("{:d} parameters, {:d} training examples".format(r, len(y_train)))

        # Normalization and scaling parameters
        mu_x, std_x = get_norm_and_scale(x_train)
        mu_y, std_y = get_norm_and_scale(y_train)

        # Set default values
        if "p" not in self.data:
            # Randomly initialize parameters
            self.data["p"] = np.random.randn(r).astype(dtype)
        if "mu_x" not in self.data or renormalize:
            self.data["mu_x"] = mu_x
        if "std_x" not in self.data or renormalize:
            self.data["std_x"] = std_x
        if "mu_y" not in self.data or renormalize:
            self.data["mu_y"] = mu_y
        if "std_y" not in self.data or renormalize:
            self.data["std_y"] = std_y
        # Normalized and scaled data
        x_train = (x_train - self.data["mu_x"]) / self.data["std_x"]
        x_cv = (x_cv - self.data["mu_x"]) / self.data["std_x"]
        y_train = (y_train - self.data["mu_y"]) / self.data["std_y"]
        y_cv = (y_cv - self.data["mu_y"]) / self.data["std_y"]

        history = []
        cost_cv_best = np.inf
        for i, gamma in enumerate(gammas):
            print("Regularization parameter value:", gamma)
            p, hist = get_minimization_solution(
                self.data["p"],
                x_train,
                y_train,
                self.data["layers"],
                self.data["activation_type"],
                gamma=gamma,
                epochs=epochs,
                batchsize=batchsize,
                method=method,
                gtol=gtol,
                maxiter=maxiter,
            )
            cost_cv = cost_NN(p, x_cv, y_cv, self.data["layers"], self.data["activation_type"], output="value")
            hist["cost cv without regularization"] = cost_cv
            history.append(hist)
            if cost_cv <= cost_cv_best:
                i_best = i
                cost_cv_best = cost_cv
                p_best = p
        self.data["p"] = p_best
        print("Best regularization value: {:.6f}".format(gammas[i_best]))

        # Training and cross-validation errors,
        # without any regularization.
        self.data["cost_train"] = cost_NN(
            self.data["p"], x_train, y_train, self.data["layers"], self.data["activation_type"], output="value"
        )
        self.data["cost_cv"] = cost_NN(
            self.data["p"], x_cv, y_cv, self.data["layers"], self.data["activation_type"], output="value"
        )
        return history

    def get_cost(self, x, y, gamma=0.0, output="value"):
        """
        Return cost function value.
        """
        if not (
            "p" in self.data
            and "predict_log_value" in self.data
            and "mu_x" in self.data
            and "std_x" in self.data
            and "mu_y" in self.data
            and "std_y" in self.data
        ):
            raise Exception("Not all parameters are initioalized...")
        # Cast input parameters to desired precision
        x = np.array(x, dtype=dtype)
        y = np.array(y, dtype=dtype)
        gamma = np.array(gamma, dtype=dtype)
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        elif x.ndim > 2:
            raise Exception("Unexpected input dimension of apartment features")
        # Log value
        if self.data["predict_log_value"]:
            # Fit to logarithm of the output (instead of fitting to the output).
            y = np.log(y)
        # Normalize and scale input
        x_norm = norm_and_scale(x, self.data["mu_x"], self.data["std_x"])
        y_norm = norm_and_scale(y, self.data["mu_y"], self.data["std_y"])

        return cost_NN(
            self.data["p"],
            x_norm,
            y_norm,
            self.data["layers"],
            self.data["activation_type"],
            gamma=gamma,
            output=output,
        )

    def predict(self, x):
        """
        Return prediction from neural network model.

        Parameters
        ----------
        x : real array(N,M) or array(N) or list(N)
            Number of features N.
            Number of data examples M.

        """
        if not (
            "p" in self.data
            and "predict_log_value" in self.data
            and "mu_x" in self.data
            and "std_x" in self.data
            and "mu_y" in self.data
            and "std_y" in self.data
        ):
            raise Exception("Not all parameters are initioalized...")
        # Cast input parameters to desired precision
        x = np.array(x, dtype=dtype)
        if x.ndim == 1:
            x = np.atleast_2d(x).T
        elif x.ndim > 2:
            raise Exception("Unexpected input dimension of apartment features")
        # Normalize and scale input
        x_norm = norm_and_scale(x, self.data["mu_x"], self.data["std_x"])
        # Calculate output from neural network
        y_norm = hypothesis(x_norm, self.data["p"], self.data["layers"], self.data["activation_type"])
        # Go back to un-normalized and un-scaled output
        y = y_norm * self.data["std_y"] + self.data["mu_y"]
        if self.data["predict_log_value"]:
            y = np.exp(y)
        y = y.flatten()
        if y.size == 1:
            y = y[0]
        return y


def load_nn_model_from_file(filename):
    """
    Load neural network model from a .h5 file.
    """
    h5f = h5py.File(filename, "r")
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


def norm_and_scale(x, mu, std, axis=1):
    """
    Return normalized data.
    """
    if x.ndim in (1, 2):
        # Normalize and scale input
        x_ns = (x - mu) / std
    else:
        raise Exception("Not implemented yet...")
    return x_ns


def hypothesis(x, p, layers, activation_type="sigmoid", logistic_output=False):
    """
    Return the neural network prediction.

    Forward propagation.

    """
    if activation_type == "sigmoid":
        g = sigmoid
    elif activation_type == "ReLu":
        g = relu
    else:
        raise Exception("This activation function has not been implemented.")

    # Number of features and number of data examples
    n, m = np.shape(x)
    # Model parameters, unrolled
    w, b = unroll(p, layers)
    # Model predictions
    h = np.zeros(m, dtype=dtype)

    # Parallelized version, no loop over examples
    z = x
    for j, (wm, bv) in enumerate(zip(w, b)):
        z = np.dot(wm, z) + bv
        # Last layer is treated specially
        if j == len(layers) - 2:
            h = g(z) if logistic_output else z
        else:
            z = g(z)
    return h


def sigmoid(z):
    """
    Return the sigmoid function.
    """
    return scipy.special.expit(z)


def dsigmoid(z):
    """
    Return the derivative of the sigmoid function.
    """
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1 - sigmoid_value)


def relu(z):
    """
    Return the ReLu function.
    """
    return z * (z > 0)


def drelu(z):
    """
    Return the derivative of the ReLu function.
    """
    return 1.0 * (z > 0)


def unroll(p, layers):
    w = []
    b = []
    j, k = 0, 0
    for i in range(len(layers[:-1])):
        k = j + layers[i + 1] * layers[i]
        w.append(p[j:k].reshape((layers[i + 1], layers[i])))
        j = k
        k += layers[i + 1]
        b.append(p[j:k].reshape((layers[i + 1], 1)))
        j = k
    return w, b


def inroll(w, b):
    # Number of parameters
    r = 0
    for i in range(len(w)):
        r += w[i].size + b[i].size
    p = np.zeros(r, dtype=dtype)
    # Fill p
    j, k = 0, 0
    for i in range(len(w)):
        k = j + w[i].size
        p[j:k] = w[i].reshape(w[i].size)
        j = k
        k += b[i].size
        p[j:k] = b[i].reshape(b[i].size)
        j = k
    return p


def shuffle_and_divide_indices(m, p_train=0.6, p_cv=0.2):
    m_train = int(m * p_train)
    m_cv = int(m * p_cv)
    # Same division of data.
    # E.g. want to avoid testing on previous training data.
    p = np.random.RandomState(seed=42).permutation(m)
    indices = [p[:m_train], p[m_train : m_train + m_cv], p[m_train + m_cv :]]
    return indices


def shuffle_and_divide_up_data(x, y, p_train=0.6, p_cv=0.2):
    m = len(y)
    indices = shuffle_and_divide_indices(m, p_train, p_cv)
    x_train = x[:, indices[0]]
    y_train = y[indices[0]]
    x_cv = x[:, indices[1]]
    y_cv = y[indices[1]]
    x_test = x[:, indices[2]]
    y_test = y[indices[2]]
    return x_train, y_train, x_cv, y_cv, x_test, y_test


def get_norm_and_scale(x, axis=1):
    """
    Return norm, scaling and normalized data.
    """
    if x.ndim == 1:
        # Average and standard deviation
        mu = np.mean(x)
        std = np.std(x)
    elif x.ndim == 2:
        # Average and standard deviation
        mu = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
    else:
        raise Exception("Not implemented yet...")
    return mu, std


def hypothesis_linear_regression(p, x):
    """
    Return the prediction.
    """
    # Model parameters, unrolled
    w = p[:-1]
    b = p[-1]
    # Model predictions
    h = np.dot(w, x) + b
    return h


def cost_linear_regression(p, x, y, gamma=0, output="value and gradient"):
    """
    Return the linear regression cost function.

    """
    # Number of features and number of data examples
    n, m = np.shape(x)
    # Model parameters, unrolled
    w = p[:-1]
    # Model predictions
    h = hypothesis_linear_regression(p, x)
    # difference
    d = h - y
    # Loss values
    loss = d**2
    # Cost function value
    c = 1 / (2 * m) * np.sum(loss) + gamma / (2 * n) * np.sum(w**2)
    if output == "value":
        return c
    # Partial derivatives of the cost function
    dw = 1 / m * np.dot(x, d) + gamma / n * w
    db = 1 / m * np.sum(d)
    dp = np.hstack((dw, db))
    if output == "gradient":
        return dp
    elif output == "value and gradient":
        return c, dp
    else:
        raise Exception("Output option not possible.")


def gradient_descent(fun, jac, p_initial, alpha=0.05, maxiter=10**4, rel_df_tol=1e-5):
    """
    Minimize the function f.

    Parameters
    ----------
    fun : function
        f(p) should return the function value.
    jac : function
        jac(p) should return the gradient of f(p).
    p_initial: array(N)
        Initial parameter values.
    alpha : float
        Learning rate.
    maxiter : int
        Miximum number of gradient descent iterations.
    rel_df_tol : float
        Minimization stops if:
        abs(f(p_i) - f(p_{i-1}))/abs(f(p_i)) <  rel_df_tol

    Returns
    -------
    f_value: float
        Minimum function value.
    p : array(N)
        Parameter values which minimizes the function f.
    info : dict
        minimization information.

    """
    p = p_initial.copy()
    info = {}
    f_old_value = np.nan
    for i in range(maxiter):
        # Calculate function and its gradient
        f_new_value, f_new_gradient = fun(p), jac(p)
        if i > 0:
            rel_df = abs(f_new_value - f_old_value) / abs(f_new_value)
            if rel_df < rel_df_tol:
                info["status"] = "rel_df_tol satisfied"
                info["rel_df"] = rel_df
                info["iter_index"] = i
                break
        # Update parameters
        p += -alpha * f_new_gradient
        # Save the old function value
        f_old_value = f_new_value
    else:
        info["status"] = "maximum number of iterations reached"
        info["rel_df"] = rel_df
        info["iter_index"] = i
        print("Warning:" + info["status"])
    return fun(p), p, info


def train_linear_regression(x_train, y_train, x_cv, y_cv, gammas, alpha=0.05, maxiter=10**4, rel_df_tol=1e-5):
    """
    Return optimal model parameters.

    """
    # Number of parameters
    r = np.shape(x_train)[0] + 1
    # Initialize parameters
    p_initial = np.random.randn(r)
    # Loop over regularization values
    for i, gamma in enumerate(gammas):
        # Cost function, for linear regression
        fun = partial(cost_linear_regression, x=x_train, y=y_train, gamma=gamma, output="value")
        # Gradient of the cost function, for linear regression
        jac = partial(cost_linear_regression, x=x_train, y=y_train, gamma=gamma, output="gradient")
        # Train model by minimizing the cost function for the training data.
        # This gives the optimal parameters.
        _, p, _ = gradient_descent(fun, jac, p_initial, alpha, maxiter, rel_df_tol)
        cost_cv = cost_linear_regression(p, x_cv, y_cv, output="value")
        if i == 0:
            i_best = 0
            cost_cv_best = cost_cv
            p_best = p
        elif cost_cv < cost_cv_best:
            i_best = i
            cost_cv_best = cost_cv
            p_best = p
    return p_best, i_best


def train_NN(
    x_train,
    y_train,
    x_cv,
    y_cv,
    layers,
    gammas,
    activation_type="sigmoid",
    logistic_output=False,
    epochs=100,
    batchsize=None,
    method="BFGS",
    gtol=1e-05,
    maxiter=None,
):
    """
    Return optimal neural network model parameters.

    Parameters
    ----------
    x_train : array(N,M_train)
    y_train : array(M_train)
    x_cv : array(N,M_cv)
    y_cv : array(M_cv)
    layers : tuple
        Designs the neural network.
        The number of units in each layer.
        Note that the first value must be N.
        It's common with the last value being 1.
    gammas : array(K)
        Regularization parameter values.
    activation_type : str
        'sigmoid' or 'ReLu'.
    logistic_output : boolean
        If logistic regression or regression.
    epochs : int
        Number of epochs
    batchsize : int
        Size of one batch
    method : str
        Scipy minimization method
    gtol : float
        Gradient convergence tolerence
    maxiter : int
        Maximum number of minimization iterations for each batch.

    """
    # Number of parameters
    r = get_number_of_NN_parameters(layers)

    # Initialize parameters
    p0 = np.random.randn(r)

    # Loop over regularization values
    # Minimize validation error as a function of the
    # regularization parameter.
    for i, gamma in enumerate(gammas):
        print("Regularization parameter value:", gamma)
        p = get_minimization_solution(
            p0,
            x_train,
            y_train,
            layers,
            activation_type,
            logistic_output,
            gamma,
            epochs,
            batchsize,
            method,
            gtol,
            maxiter,
        )

        cost_cv = cost_NN(p, x_cv, y_cv, layers, activation_type, logistic_output, output="value")
        if i == 0:
            i_best = 0
            cost_cv_best = cost_cv
            p_best = p
        elif cost_cv < cost_cv_best:
            i_best = i
            cost_cv_best = cost_cv
            p_best = p
    return p_best, i_best


def get_minimization_solution(
    p0,
    x,
    y,
    layers,
    activation_type,
    logistic_output=False,
    gamma=0,
    epochs=100,
    batchsize=None,
    method="BFGS",
    gtol=1e-05,
    maxiter=None,
):
    """
    Return solution to minimization.
    """
    # Initial parameters
    p = p0
    # Number of examples
    m = len(y)
    assert x.shape[1] == m
    if batchsize is None:
        batchsize = m
    assert batchsize <= m
    # Number of batches
    batches = m // batchsize
    print("Start minimizing NN cost function")
    print("{:d} epochs, {:d} batches (each of size {:d}) \n".format(epochs, batches, batchsize))
    hist = {"cost": []}
    n_prints = 10
    counter = 0
    for epoch in range(epochs):
        # print('Epoch nr {:d}'.format(epoch))
        permutation = np.random.permutation(m)
        for batch in range(batches):
            # print('Batch nr {:d}'.format(batch))
            if batch == batches - 1:
                # Take all the remaining examples in the last batch
                indices = permutation[batch * batchsize :]
            else:
                indices = permutation[batch * batchsize : (batch + 1) * batchsize]
            # NN cost function and its gradient.
            fun = partial(
                cost_NN,
                x=x[:, indices],
                y=y[indices],
                layers=layers,
                activation_type=activation_type,
                logistic_output=logistic_output,
                gamma=gamma,
                output="value",
            )
            jac = partial(
                cost_NN,
                x=x[:, indices],
                y=y[indices],
                layers=layers,
                activation_type=activation_type,
                logistic_output=logistic_output,
                gamma=gamma,
                output="gradient",
            )
            # Train model by minimizing the cost function (for the training data).
            # This gives the optimal parameters.
            res = minimize(
                fun,
                p,
                jac=jac,
                method=method,
                options={"gtol": gtol, "maxiter": maxiter, "disp": False, "return_all": True},
            )  # 'norm': 'inf'

            if epoch % (epochs // n_prints if epochs >= n_prints else 1) == 0 and batch == 0:
                print("Report from epoch nr {:d} and batch nr {:d}".format(epoch, batch))
                print(res.message)
                print("batch minimization success:", res.success)
                print("termination status: {:d}".format(res.status))
                print("Cost (including regularization):", res.fun)
                print("{:d} iterations performed".format(res.nit))
                print("{:d} function evaluations".format(res.nfev))
                print("{:d} gradient evaluations".format(res.njev))
                print("")
            hist["cost"].append(res.fun)
            p = res.x
        if len(hist["cost"]) > 1 and epoch % (epochs // n_prints if epochs >= n_prints else 1) == 0:
            print("New cost values (including regularization):")
            print(hist["cost"][counter:])
            print()
            counter = len(hist["cost"])

    return p, hist


def get_number_of_NN_parameters(layers):
    r = 0
    for i in range(len(layers[:-1])):
        r += layers[i + 1] * (layers[i] + 1)
    return r


def cost_NN_only_value(p, x, y, layers, activation_type="sigmoid", logistic_output=False, gamma=0.0):
    """
    Return the neural network cost function.

    Parameters
    ----------
    p : array(K)
    x : array(N,M)
    y : array(M)
    layers : tuple
        Designs the neural network.
        The number of units in each layer.
        Note that the first value must be N.
        It's common with the last value being 1.
    activation_type : str
        'sigmoid' or 'ReLu'.
    logistic_output : boolean
        If logistic regression of regression.
    gamma : float
        Regularization parameter

    """
    # Number of features and number of data examples
    n, m = np.shape(x)
    # Model parameters, unrolled
    w, b = unroll(p, layers)
    h = hypothesis(p, x, layers, activation_type, logistic_output)
    # Loss values
    if logistic_output:
        # Cross entropy
        loss = -y * np.log(h) - (1 - y) * np.log(1 - h)
    else:
        # 1/2 times the squared difference
        loss = 1 / 2.0 * (h - y) ** 2
    # Cost function value, sum over training examples
    c = 1 / m * np.sum(loss)
    # Add regularization terms
    wsize = sum(wm.size for wm in w)
    for wm in w:
        c += gamma / (2 * wsize) * np.sum(np.sum(wm**2))
    return c


def cost_NN_numerical_gradient(
    p, x, y, layers, activation_type="sigmoid", logistic_output=False, gamma=0.0, delta=0.001
):
    """
    Return the neural network cost function and its numerical gradient.

    Parameters
    ----------
    p : array(K)
    x : array(N,M)
    y : array(M)
    layers : tuple
        Designs the neural network.
        The number of units in each layer.
        Note that the first value must be N.
        It's common with the last value being 1.
    activation_type : str
        'sigmoid' or 'ReLu'.
    logistic_output : boolean
        If logistic regression of regression.
    gamma : float
        Regularization parameter
    delta : float
        Step size in the estimation of the gradient.
    """
    f = partial(
        cost_NN_only_value,
        x=x,
        y=y,
        layers=layers,
        activation_type=activation_type,
        logistic_output=logistic_output,
        gamma=gamma,
    )
    c = f(p)
    dp = gradient_numerical(f, p, delta)
    return c, dp


def gradient_numerical(f, p, delta):
    """
    Return the numerical gradient of function f around point p.
    """
    # Numerical gradient
    dp = np.zeros_like(p)
    p_tmp = p.copy()
    for i in range(len(p)):
        p_tmp[i] += delta
        f_plus = f(p_tmp)
        p_tmp[i] -= 2 * delta
        f_minus = f(p_tmp)
        dp[i] = (f_plus - f_minus) / (2 * delta)
        # restore the value at index i in p_tmp
        p_tmp[i] = p[i]
    return dp


def get_g_and_dg(activation_type: str):
    if activation_type == "sigmoid":
        g = sigmoid
        dg = dsigmoid
    elif activation_type == "ReLu":
        g = relu
        dg = drelu
    else:
        raise Exception("This activation function has not been implemented.")
    return g, dg


def cost_NN(p, x, y, layers, activation_type="sigmoid", logistic_output=False, gamma=0.0, output="value and gradient"):
    """
    Return the neural network cost function and its gradient.

    Parameters
    ----------
    p : array(K)
    x : array(N,M)
    y : array(M)
    layers : tuple
        Designs the neural network.
        The number of units in each layer.
        Note that the first value must be N.
        It's common with the last value being 1.
    activation_type : str
        'sigmoid' or 'ReLu'.
    logistic_output : boolean
        If logistic regression of regression.
    gamma : float
        Regularization parameter
    output : str
        'value and gradient', 'value' or 'gradient'

    """
    g, dg = get_g_and_dg(activation_type)
    # Number of features and number of data examples
    n, m = np.shape(x)
    # Model parameters, unrolled
    w, b = unroll(p, layers)

    # Forward propagation
    # Parallelized version, no loop over examples
    z = []
    a = []
    for j, (wm, bv) in enumerate(zip(w, b)):
        # First to second layer is treated specially
        if j == 0:
            z.append(np.dot(wm, x) + bv)
        else:
            z.append(np.dot(wm, a[j - 1]) + bv)
        # Last layer is treated specially
        if not logistic_output and j == len(layers) - 2:
            a.append(z[j])
        else:
            a.append(g(z[j]))

    # Loss values
    if logistic_output:
        # Cross entropy
        loss = -y * np.log(a[-1]) - (1 - y) * np.log(1 - a[-1])
    else:
        # 1/2 times the squared difference
        loss = 1 / 2.0 * (a[-1] - y) ** 2
    # Cost function value, sum over training examples
    c = 1 / m * np.sum(loss)
    # Add regularization terms
    wsize = sum(wm.size for wm in w)
    for wm in w:
        c += gamma / (2 * wsize) * np.sum(np.sum(wm**2))

    if output == "value":
        # Return only the cost function value.
        return c

    # Partial derivatives of the cost function using
    # back-propagation.
    dw = []
    db = []
    # Parallelized version, no loop over examples
    # Initialize dz from the last layer.
    j = len(layers) - 2
    dz = a[j] - y
    for j in range(len(layers) - 1)[-1::-1]:
        # Last layer is treated specially
        if j != len(layers) - 2:
            # Element-wise multiplication between "da" and da/dz
            dz = np.dot(w[j + 1].T, dz) * dg(z[j])
        # First to second layer is treated specially
        if j == 0:
            dw.insert(0, 1 / m * np.dot(dz, x.T))
        else:
            dw.insert(0, 1 / m * np.dot(dz, a[j - 1].T))
        db.insert(0, 1 / m * np.sum(dz, axis=1, keepdims=True))

    # Add regularization terms
    for i, wm in enumerate(w):
        dw[i] += gamma / wsize * wm

    # Convert back to rank one array
    dp = inroll(dw, db)
    if output == "gradient":
        return dp
    elif output == "value and gradient":
        return c, dp
    else:
        raise Exception("Output option not possible.")
