from sklearn.cluster import SpectralClustering
import pyemma
import numpy as np
from random import randint
from clustering.scar import SCAR


def _transform_data(data):
    vec_norm = sum(data ** 2)
    vec_norm = vec_norm.reshape(1, -1)
    return data.T @ data / (vec_norm.T @ vec_norm + 1e-6)


def spectral_clustering(data, k=10):
    """
    Use spectral clustering as a post-processing step after e.g. using TSC
    :param data: Data to be clustered
    :param k: Number of clusters
    :return: Discrete list of cluster assignments
    """
    sc = SpectralClustering(k, affinity='precomputed')
    sc.fit(_transform_data(data))
    return sc.labels_


def scar(data, k: int = 10, nn: int = 50, alpha: float = 0.5, theta: int = 100, laplacian: int = 2):
    """
    Use SCAR (Spectral Clustering Accelerated and Robustified) as a post-processing step after e.g. using TSC
    :param data: Data to be clustered
    :param k: Number of clusters
    :param nn: number of neighbours to consider for constructing the KNN graph (excluding the node itself)
    :param alpha: percentage of landmark points selected as subsample for the Nyström method from the original dataset
    :param theta: number of corrupted edges to remove
    :param laplacian: which graph Laplacian to use: 0: L, 1: L_rw, 2: L_sym
    :return: Discrete list of cluster assignments
    """
    return SCAR(k=k, nn=nn, alpha=alpha, theta=theta, laplacian=laplacian).fit_predict(_transform_data(data))


def pca_k_means(data, k: int):
    """
    Use PCA for dimensionality reduction and k-means for clustering
    :param data: Data to be clustered
    :param k: Number of clusters
    :return: Discrete list of cluster assignments
    """
    pca_out = pyemma.coordinates.pca(data).get_output()[0]
    labels = pyemma.coordinates.cluster_kmeans(pca_out, k=k, max_iter=50).get_output()[0].reshape(len(data))
    return labels


def spectral_pca(data, k: int):
    """
    Use PCA for dimensionality reduction and spectral clustering to get the cluster assignments
    :param data: Data to be clustered
    :param k: Number of clusters
    :return: Discrete list of cluster assignments
    """
    pca_out = pyemma.coordinates.pca(data, dim=k).get_output()[0]
    sc = SpectralClustering(k)
    sc.fit(pca_out)
    return sc.labels_


def spectral_tica(data, k: int, lag: int = 1):
    tica_out = pyemma.coordinates.tica(data, dim=k, lag=lag).get_output()[0]
    sc = SpectralClustering(k)
    sc.fit(tica_out)
    return sc.labels_


def spectral_raw(data, k: int):
    sc = SpectralClustering(k, affinity='precomputed')
    data = _transform_data(data.T)
    sc.fit(data)
    return sc.labels_


def tica_k_means(data, k: int, lag: int):
    """
    Use TICA for dimensionality reduction and k-means for clustering
    :param data: Data to be clustered
    :param k: Number of clusters
    :param lag: The seq_neighbors time, in multiples of the input time step
    :return: Discrete list of cluster assignments
    """
    tica_out = pyemma.coordinates.tica(data, lag=lag) .get_output()[0]
    labels = pyemma.coordinates.cluster_kmeans(tica_out, k=k, max_iter=50).get_output()[0].reshape(len(data))
    return labels


def vamp(data, k: int):
    vamp_out = pyemma.coordinates.vamp(data).get_output()[0]
    labels = pyemma.coordinates.cluster_kmeans(vamp_out, k=k, max_iter=50).get_output()[0].reshape(len(data))
    return labels


def random_clustering(data, k: int):
    """
    Randomly assign data points to clusters
    :param data: Data to be clustered
    :param k: Number of clusters
    :return: Discrete list of cluster assignments
    """
    return np.array([randint(0, k-1) for _ in range(len(data))])
