import numpy as np
from .Assignment1 import *


## PART 1 ###


def test_matrix_operations():
    A = np.array([[3, 4], [2, 1]])
    B = np.array([[1, 2], [0, -1]])
    ab_mult, ba_mult, a_b_add_transpose, a_b_dot_transpose, a_trace, ab_trace = (
        matrix_operations(A, B)
    )
    np.testing.assert_array_equal(ab_mult, np.array([[3, 2],[2, 3]]))
    np.testing.assert_array_equal(ba_mult, np.array([[7, 6],[-2, -1]]))
    np.testing.assert_array_equal(a_b_add_transpose, np.array([[4, 2],[6, 0]]))
    np.testing.assert_array_equal(a_b_dot_transpose, np.array([[11, -4],[4, -1]]))
    assert a_trace == 4
    assert ab_trace == 6


def test_eigensystem():
    M = np.array([[2,-1],[-1,2]])
    v = np.array([[1, 1], [1,  -1]])
    l = np.array([1., 3.])
    eigenvalues, eigenvectors = eigensystem(M)
    np.testing.assert_array_equal(M@eigenvalues, M@l)
    np.testing.assert_array_equal(eigenvalues*eigenvectors, l*v)


def test_matrix_inverse():
    C = np.array([[4,7],[2,6]])
    C_inverse_desired = np.array([[0.6,-0.7],[-0.2,0.4]])
    c_inverse_calc = matrix_inverse(C)
    np.testing.assert_allclose(c_inverse_calc, C_inverse_desired)
    np.testing.assert_allclose(C@c_inverse_calc, np.identity(2), atol=1e-12)


### PART 2 ###

def test_distribution_moments():
    c, mean, variance = distribution_moments(3.0)
    np.testing.assert_allclose(c, np.exp(-3))
    assert mean == 3
    assert variance == 3


def test_event_probabilities():
    p_a_intersect_b, p_b_given_a, p_a_union_b = event_probabilities()
    assert p_a_intersect_b == 0.18
    assert p_b_given_a == 0.36
    assert round(p_a_union_b, 2) == 0.62


def test_coin_estimates():
    mle, map_estimate = coin_estimates(3, 5)
    np.testing.assert_allclose(mle, 0.6)
    np.testing.assert_allclose(map_estimate, 2/3)


def test_coin_estimates_boundary():
    mle, map_estimate = coin_estimates(5, 5)
    assert mle == 1
    assert map_estimate == 1
    mle, map_estimate = coin_estimates(0, 5)
    assert mle == 0
    np.testing.assert_allclose(map_estimate, 1/6)


def test_gaussian_mle():
    mean_mle, variance_mle = gaussian_mle([1, 3, 5, 7])
    assert mean_mle == 4
    assert variance_mle == 5


def test_gaussian_map():
    mu_map = gaussian_map(5.0, 4.0, 0.0, 1.0)
    np.testing.assert_allclose(mu_map, 1.0)


### PART 3 ###

def test_generate_data():
    mean = np.array([2., -1.])
    covariance = np.array([[3, 1.2],[1.2, 1]])
    X = generate_data(500, 0, mean, covariance)
    assert X.shape == (500, 2)
    np.testing.assert_array_equal(X, generate_data(500, 0, mean, covariance))
    np.testing.assert_allclose(X.mean(axis=0), mean, atol=0.25)
    np.testing.assert_allclose(np.cov(X.T), covariance, atol=0.5)


def test_pca():
    X = np.array([[2, 0],[0, 1],[-2, 0],[0, -1]])
    X_copy = X.copy()
    result = pca(X)
    np.testing.assert_allclose(result["mean"], np.array([0, 0]))
    np.testing.assert_allclose(result["centered"], X)
    np.testing.assert_allclose(result["covariance"], np.array([[8/3, 0],[0, 2/3]]))
    np.testing.assert_allclose(result["eigenvalues"], np.array([8/3, 2/3]))
    # components are orthonormal, sign is free
    components = result["components"]
    np.testing.assert_allclose(components.T@components, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.abs(components), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(result["scores"], result["centered"]@components)
    np.testing.assert_array_equal(X, X_copy)


def test_pca_rank_deficient():
    X = np.array([[0, 0],[1, 1],[2, 2],[3, 3]])
    result = pca(X)
    eigenvalues = result["eigenvalues"]
    np.testing.assert_allclose(eigenvalues, np.array([10/3, 0]), atol=1e-12)
    assert np.all(eigenvalues >= 0)


def test_explained_variance():
    fractions = explained_variance(np.array([3., 1.]))
    np.testing.assert_allclose(fractions, np.array([0.75, 0.25]))
    np.testing.assert_allclose(fractions.sum(), 1)
