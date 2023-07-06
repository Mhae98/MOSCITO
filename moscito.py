import numpy as np
from math import log

import pypardiso
import scipy.sparse.linalg
from scipy import sparse
import scipy
from typing import Literal
from scipy.spatial import distance


def _calculate_affinity(data):
    """
    Calculates affinity graph from coding matrix
    :param data: Coding matrix
    :return: Affinity matrix
    """
    vec_norm = sum(data ** 2)
    vec_norm = vec_norm.reshape(1, -1)
    return data.T @ data / (vec_norm.T @ vec_norm + 1e-6)


def _normalize_matrix(matrix):
    """Normalizes matrix to have column mean of 0 and unit variance (normalize like matlab)"""
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)


def _generate_gaussian_weight_vector(size: int, sigma: float = 1.0):
    lin_range = (-1, 1)
    lin_array = np.linspace(*lin_range, size + 1)[:, np.newaxis]  # expand (x,)-array to (x,1)-array
    sq_norm = -0.5 * distance.cdist(np.array([[1]]), lin_array, 'sqeuclidean')  # L2 distance (Squared Euclidian)
    gaussian: np.ndarray = np.exp(sq_norm / np.square(sigma))[0]  # gaussian
    gaussian_flipped = np.flip(gaussian)[1:]
    return np.concatenate((gaussian, gaussian_flipped))


def _generate_exponential_decay(size: int) -> np.ndarray:
    """
    Generates an array of values that exponentially decay around the center with the value 1
    :param size: size of the decay around the center
    :return: Numpy array of the exponential decay with size * 2 + 1 entries
    """
    from scipy.optimize import curve_fit
    exp_vals = np.linspace(0, min(size, 10), size + 1)
    right_half = np.array([np.exp(-x) for x in exp_vals])
    left_half = np.flip(right_half[1:])
    return np.concatenate([left_half, right_half])


def _generate_logarithmic_decay(size: int) -> np.ndarray:
    max_val = log(size + 1)
    left_half = np.array([log(i) / max_val for i in range(1, size + 1)])
    right_half = np.flip(left_half)
    return np.concatenate([left_half, [1], right_half])


class MOSCITO:
    def __init__(self,
                 d_size: int = 60,
                 seq_neighbors: int = 3,
                 weight_mode: Literal['binary', 'log', 'exp', 'gauss'] = 'binary',
                 lambda_1: float = 0.01,
                 lambda_2: float = 15,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 max_iterations: int = 12,
                 tolerance: float = 1e-4,
                 step_size: float = 0.1,
                 normalize: bool = True
                 ):
        """
        Implementation of the temporal subspace clustering algorithm
        :param d_size: Size of the dictionary used to approximate the data
        :param seq_neighbors: Size of sequential neighbourhood
        :param weight_mode: Choose different weighting of consecutive timeframes. Default: bin; Supported: bin, exp, log
        :param lambda_1: first tradeoff parameter - weights sparsity 
        :param lambda_2: second tradeoff parameter - weights temporal regularization
        :param alpha: Learning rate parameter
        :param beta: Learning rate parameter
        :param max_iterations: Maximal number of iterations the algorithm should run
        :param tolerance: Tolerance for the termination condition
        :param step_size: Tuning parameter
        :param normalize: Flag if input data should be normalized, default: True
        """
        self.d_size = d_size
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_size = step_size
        self.seq_neighbors = seq_neighbors
        self.weight_mode = weight_mode
        self.normalize = normalize

    def fit_predict(self, X: np.ndarray):
        """
        :param X: numpy array with shape (time-steps, ...)
        :return: affinity matrix
        """
        X = _normalize_matrix(X.T) if self.normalize else X.T

        time_steps, data_size = X.shape

        D = np.random.rand(time_steps, self.d_size)
        Z = np.random.rand(self.d_size, data_size)
        V = np.zeros((self.d_size, data_size))

        Y1 = np.zeros((time_steps, self.d_size))
        Y2 = np.zeros((self.d_size, data_size))

        W = np.zeros((data_size, data_size))

        weights_binary = [1 for _ in range(self.seq_neighbors * 2 + 1)]
        weights_exponential = _generate_exponential_decay(self.seq_neighbors)
        weights_log = [max([0, 1 - log(abs(i), 10)]) if abs(i) > 1 else 1 for i in range(-self.seq_neighbors, self.seq_neighbors + 1)]

        weight_vector = weights_binary
        if self.weight_mode == 'exp':
            weight_vector = weights_exponential
        elif self.weight_mode == 'log':
            weight_vector = weights_log
        elif self.weight_mode == 'gauss':
            weight_vector = _generate_gaussian_weight_vector(self.seq_neighbors)

        for i in range(data_size):
            start_w = max(0, i - self.seq_neighbors)
            end_w = min(data_size, i + self.seq_neighbors + 1)
            start_v = 0 if start_w != 0 else self.seq_neighbors - i
            end_v = self.seq_neighbors * 2 + 1 if end_w != data_size else self.seq_neighbors + (data_size - i)
            W[i, start_w:end_w] = weight_vector[start_v:end_v]
        for i in range(data_size):
            W[i, i] = 0

        # Construct laplacian
        DD = np.diag(np.sum(W, axis=0))
        laplacian = DD - W
        kron_R_Rt = self.lambda_2 * sparse.kron(laplacian, sparse.eye(self.d_size, self.d_size), format='csr')
        err = []

        solver = pypardiso.PyPardisoSolver()
        # Overview of all available parameter settings:
        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-iparm-parameter.html
        solver.set_iparm(1, 3)  # use the parallel (OpenMP) version of the nested dissection algorithm

        for i in range(self.max_iterations):
            # Construct Laplacian matrix
            f_old = np.linalg.norm(X - D @ Z, ord='fro')

            # Update U
            a = ((X @ V.T) - Y1) + (self.alpha * D)
            b = V @ V.T + self.alpha * np.eye(self.d_size)
            U = scipy.linalg.solve(b.T, a.T).T  # U = a / b <=> Ub = a

            # Update V
            kron_Xt_X = sparse.kron(sparse.eye(data_size), (U.T @ U), format='csr')
            left = kron_Xt_X + sparse.kron(sparse.eye(data_size),
                                           (self.lambda_1 + self.beta) * sparse.eye(self.d_size), format='csr') \
                             + kron_R_Rt
            right = U.T @ X - Y2 + self.beta * Z
            right = np.reshape(right, (self.d_size * data_size, 1), order='F')

            V_tmp = solver.solve(left, right)
            V = np.reshape(V_tmp, (self.d_size, data_size), order='F')

            # Update D
            D = U + Y1 / self.alpha
            D = D.clip(min=0)
            for index in range(D.shape[1]):
                norm = np.linalg.norm(D[:, index])
                if norm != 0:
                    D[:, index] = D[:, index] / norm

            # Update Z
            Z = V + Y2 / self.beta
            Z = Z.clip(min=0)

            # Calculating error and check if termination condition is met
            f_new = np.linalg.norm(X - D @ Z, ord='fro')
            error = abs(f_new - f_old) / max([1, abs(f_old)])
            if error < self.tolerance and np.count_nonzero(D) > 0:
                err.append(error)
                break
            else:
                err.append(error)
                Y1 = Y1 + self.step_size * self.alpha * (U - D)
                Y2 = Y2 + self.step_size * self.beta * (V - Z)
            pass

        self.D = D
        self.Z = Z
        self.err = err
        solver.free_memory(everything=True)

        affinity = _calculate_affinity(Z)
        self.affinity = affinity
        return affinity
