"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""

import matplotlib.pyplot as plt
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
#
# Explaination:
# Matrix Mulitplication is calculated via Dot Product
# AB=C  =>  C[i][j] = A[i][:] * B[:][j])
# Matrix Addition is calculated via adding each i,j element in A and B
# A+B=C  =>  C[i][j] = A[i][j] + B[i][j])
# Transpose is calculated via flipping the rows and columns of A
# C=A^T =>  C[i][j] = A[j][i]
# Trace is calculated via adding the numbers along the diagonal
# tr(A) = sum for i=0:n (A[i][i])
#
# Examples with default matricies:
# AB
# [[(3*1 + 4*0), (3*2 + -1*4)],[(2*1 + 1*0), (2*2 + -1*1)]]
# [[(3 + 0), (6 - 4)],[(2 + 0), (4 - 1)]]
# [[3, 2],[2, 3]]
#
# BA
# [[(1*3 + 2*2), (1*4 + 2*1)],[(0*3 + -1*2), (0*4 + -1*1)]]
# [[(3 + 4), (4 + 2)],[(0 - 2), (0 - 1)]]
# [[7, 6],[-2, -1]]
#
# (A+B)^T
# ([[(3 + 1), (4 + 2)],[(2 + 0), (1 + -1)]])^T
# ([[4, 6],[2, 0]])^T
# [[4, 2],[6, 0]]
#
# AB^T
# ([[3,4],[2,1]])@(([[1,2],[0,-1]])^T)
# [[3,4],[2,1]]@[[1,0],[2,-1]])
# [[(3*1 + 4*2), (3*0 + 4*-1)],[(2*1 + 1*2), (2*0 + 1*-1)]]
# [[(3 + 8), (0 - 4)], [(2 + 2), (0 - 1)]]
# [[11, -4],[4, -1]]
#
# tr(A)
# 3 + 1 = 4
#
# tr(AB)
# 3 + 3 = 6
def matrix_operations(
    A: np.ndarray | None = None, B: np.ndarray | None = None,
) -> tuple:
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    if A is None:
        A = np.array([[3, 4], [2, 1]])
    if B is None:
        B = np.array([[1, 2], [0, -1]])

    a_size = np.shape(A)
    b_size = np.shape(B)
    if a_size[0] != a_size[1] or b_size[0] != b_size[1] or a_size != b_size:
        raise ValueError("inputs must be square real matrices of matching size")

    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matmul.html
    ab_mult = np.linalg.matmul(A, B)
    ba_mult = np.linalg.matmul(B, A)

    # ref: https://numpy.org/devdocs/reference/generated/numpy.transpose.html
    a_b_add_transpose = np.transpose(A+B)
    a_b_dot_transpose = np.linalg.matmul(A, np.transpose(B))

    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.trace.html
    a_trace = np.linalg.trace(A)
    ab_trace = np.linalg.trace(ab_mult)

    return (ab_mult, ba_mult, a_b_add_transpose, a_b_dot_transpose, a_trace, ab_trace)


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
#
# Eigenvalues:
# Find lambda via the solution of det(A-lambdaI)=0
# Eigenvectors:
# For each eigenvalue lambda_i, solve ((A-lambda_i I)X = 0)
#
# Example: A = [[2,-1],[-1,2]]
# det([[2-lambda,-1],[-1,2-lambda]])=0
# (2-lambda)^2 - (-1)^2 = 0
# 4 - 4lambda + lambda^2 -1 = 0
# lambda = (4 +- sqrt(16-12))/2
# lambda = (4 +- 2)/2
# lambda = 1, 3
#
# lambda_1 = 1
# [[2-1, -1],[-1, 2-1]]x = 0
# [[1, -1],[-1, 1]]x = 0
# [[1, -1, 0],[-1, 1, 0]] => [[1, -1, 0],[0, 0, 0]] => v1 = [1, 1]
# lambda_2 = 3
# [[2-3, -1],[-1, 2-3]]x = 0
# [[-1, -1],[-1, -1]]x = 0
# [[-1, -1, 0],[-1, -1, 0]] => [[-1, -1, 0],[0, 0, 0]] => v2= [1, -1]
#
# Verify Mv = lambda*v
# Mv1 = [[2,-1],[-1,2]]@[1,1]  = [2-1, -1+2] = [1, 1]   = 1*[1, 1]   = lambda_1*v1
# Mv2 = [[2,-1],[-1,2]]@[1,-1] = [2+1, -1-2] = [3, -3]  = 3*[1, -1]  = lambda_2*v2
def eigensystem(M: np.ndarray|None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    if M is None:
        M = np.array([[2, -1], [-1, 2]])

    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
    eigenvalues, eigenvectors_unscaled = np.linalg.eigh(M)

    # numpy provides unscalled answer. fix order, sign, and nonzero scale
    mask = np.abs(eigenvectors_unscaled) > 1e-10           # True where entry is nonzero
    first_idx = mask.argmax(axis=0)                        # row of first nonzero in each column
    pivots = eigenvectors_unscaled[first_idx, np.arange(eigenvectors_unscaled.shape[1])]
    eigenvectors = eigenvectors_unscaled / pivots          # broadcasts across columns

    # verify Mv=lambda*v (allclose, since floating point rounding breaks exact equality)
    if not np.allclose(M@eigenvectors, eigenvalues*eigenvectors):
        raise Exception(f"Error in computing eigenpairs for {M}.")
    return eigenvalues, eigenvectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Hand-written proof submitted via canvas
# ref: https://en.wikipedia.org/wiki/Trace_(linear_algebra)#cite_note-5


# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
#
# In order for a matrix to be invertable, the determinant must be non-zero
# det([[4,7],[2,6]]) = 4*6 - 7*2 = 24 - 14 = 10
#
# To compute the inverse of C, you need to follow the Gauss-Jorden Elimination
# Augment C with the identity and row reduce until the left side is I: [C | I] => [I | C^-1]
#
# [[4, 7 | 1, 0], [2, 6 | 0, 1]]
#
# R1 = R1/4
# [[1, 7/4 | 1/4, 0],[2, 6   | 0,   1]]
#
# R2 = R2 - 2*R1
# [[1, 7/4 | 1/4,  0], [0, 6-7/2 | -1/2, 1]]
# [[1, 7/4 | 1/4,  0], [0, 5/2 | -1/2, 1]]
#
# R2 = R2 * 2/5
# [[1, 7/4 | 1/4,  0  ],[0, 1   | -1/5, 2/5]]
#
# R1 = R1 - 7/4*R2
# [[1, 0 | 1/4 + 7/20, 0 - 7/10],[0, 1 | -1/5, 2/5]]
# [[1, 0 | 3/5, -7/10],[0, 1 | -1/5, 2/5]]
#
# C^-1 = [[3/5, -7/10],[-1/5, 2/5]] = [[0.6, -0.7],[-0.2, 0.4]]
#
# Verify C@C^-1 = I
# [[4*0.6 + 7*-0.2, 4*-0.7 + 7*0.4],[2*0.6 + 6*-0.2, 2*-0.7 + 6*0.4]]
# [[2.4 - 1.4, -2.8 + 2.8],[1.2 - 1.2, -1.4 + 2.4]] = [[1, 0],[0, 1]]
def matrix_inverse(C: np.ndarray | None = None) -> np.ndarray:
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    if C is None:
        C = np.array([[4,7],[2,6]])

    if C.shape[0] != C.shape[1]:
        raise ValueError(f"Input must be a square matrix, got shape {C.shape}")

    # show invertable - determinant = 0
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    if np.isclose(np.linalg.det(C), 0, atol=1e-12):
        raise ValueError(f"det({C})==0. C is not invertable.")

    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    C_inverse = np.linalg.inv(C)
    return C_inverse


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
#
# P(X=k)=c*rate**k/k! is the Poisson Distribution
# sum_over_k=0_to_inf(P(X=k)) = 1
# sum_over_k=0_to_inf(c*3**k/k!) = 1
# exp**3 = sum_over_k=0_to_inf((3**k)/k!)                 Taylor series of e**x
# ce**3 = 1
# c = e**-3
#
# EXPECTATION
# E[X] = sum_over_k=0_to_inf(k * e**-3 * 3**k/k!)
#      = e**-3 * sum_over_k=1_to_inf(3**k/(k-1)!)          k=0 term is 0, k/k! = 1/(k-1)!
#      = 3e**-3 * sum_over_k=1_to_inf(3**(k-1)/(k-1)!)     pull out one factor of 3
#      = 3e**-3 * sum_over_j=0_to_inf(3**j/j!)             substitute j = k-1
#      = 3e**-3 * e**3
#      = 3
#
# VARIANCE
# Use Var(X) = E[X**2] - E[X]**2 (P2 Q6), with E[X**2] = E[X(X-1)] + E[X]
# E[X(X-1)] = sum_over_k=0_to_inf(k(k-1) * e**-3 * 3**k/k!)
#           = e**-3 * sum_over_k=2_to_inf(3**k/(k-2)!)     k=0,1 terms are 0
#           = 9e**-3 * sum_over_j=0_to_inf(3**j/j!)        substitute j = k-2
#           = 9e**-3 * e**3
#           = 9
# E[X**2] = 9 + 3 = 12
# Var(X) = 12 - 3**2 = 3
#
# In general for rate lambda: c = e**-lambda, E[X] = lambda, Var(X) = lambda
def distribution_moments(rate: float =3.0) -> tuple[np.float64, float, float]:
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    c = np.exp(-rate)
    mean = rate
    variance = rate
    return c, mean, variance


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
#
# INTERSECTION
# the intersection of two events is the event where both events occur
# solve for the intersection of A and B via the property of conditional probability
# P(A|B) = P(AnB)/P(B)
# P(AnB) = P(A|B)*P(B)
#
# UNION:
# the union of two events is the sum of events where either event occurs.
# the probability of A union be is defined by the properties of probability laws
# P(AuB) = P(A)+P(B)-P(AnB)
#
# CONDITIONAL
# Bayes rule allows us to calculate B|A based on the given probabulities
# P(B|A) = P(A|B)*P(B)/P(A) = P(AnB)/P(A)
#
# Example: p_a=0.5, p_b=0.3, p_a_given_b=0.6
# p_a_intersect_b = p_a_given_b * p_b = 0.6 * 0.3 = 0.18
# p_b_given_a = p_a_given_b * p_b / p_a = 0.6 * 0.3 / 0.5 = 0.36
# p_a_union_b = p_a + p_b - p_a_intersect_b = 0.5 + 0.3 - 0.18 = 0.62
def event_probabilities(
    p_a: float =0.5, p_b: float = 0.3, p_a_given_b: float=0.6
) -> tuple[float, float, float]:
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    p_a_intersect_b = p_a_given_b * p_b
    p_b_given_a = p_a_given_b * p_b / p_a
    p_a_union_b = p_a + p_b - p_a_intersect_b
    return p_a_intersect_b, p_b_given_a, p_a_union_b


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
#
# Each toss is an independent Bernoulli(theta). With h heads in n tosses (h=3, n=5)
#
# MLE
# L(theta) = P(data|theta) = theta**h * (1-theta)**(n-h) = theta**3 * (1-theta)**2
# log L(theta) = h*log(theta) + (n-h)*log(1-theta)]                     log is monotonic, same argmax
# d/dtheta log L = h/theta - (n-h)/(1-theta) = 0
# h(1-theta) = (n-h)theta
# h - h*theta = n*theta - h*theta
# theta_MLE = h/n = 3/5 = 0.6
# second derivative -h/theta**2 - (n-h)/(1-theta)**2 < 0, so this is a maximum
#
# MAP
# Prior P(theta) = 2*theta on [0,1]
# P(theta|data) proportional to P(data|theta)*P(theta)                  Bayes rule
#               = theta**h * (1-theta)**(n-h) * 2*theta
#               proportional to theta**(h+1) * (1-theta)**(n-h)         drop constant 2
# log posterior = (h+1)*log(theta) + (n-h)*log(1-theta) + const
# d/dtheta = (h+1)/theta - (n-h)/(1-theta) = 0
# (h+1)(1-theta) = (n-h)theta
# theta_MAP = (h+1)/(n+1) = 4/6 = 2/3
#
# Boundary maxima
# if h = n: posterior proportional to theta**(n+1), increasing on [0,1], max at theta = 1 = (n+1)/(n+1)
# if h = 0: MLE = 0 since L = (1-theta)**n is decreasing; the MAP posterior is theta*(1-theta)**n,
#           which is 0 at both ends, so the max is interior at 1/(n+1)
# so both closed forms hold on the boundaries too
def coin_estimates(heads: int =3, tosses: int =5) -> tuple[float, float]:
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    mle = heads / tosses
    map_estimate = (heads + 1) / (tosses + 1)
    return mle, map_estimate


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
#
# Likelihood of n i.i.d. samples from N(mu, sigma**2)
# L(mu, sigma**2) = prod_i (1/sqrt(2*pi*sigma**2)) * exp(-(x_i-mu)**2 / (2*sigma**2))
# log L = -n/2*log(2*pi) - n/2*log(sigma**2) - 1/(2*sigma**2) * sum_i (x_i-mu)**2
#
# MEAN
# d/dmu log L = 1/sigma**2 * sum_i (x_i-mu) = 0
# sum_i x_i - n*mu = 0
# mu_MLE = 1/n * sum_i x_i = (1+3+5+7)/4 = 16/4 = 4
#
# VARIANCE
# d/dsigma**2 log L = -n/(2*sigma**2) + 1/(2*sigma**4) * sum_i (x_i-mu)**2 = 0
# multiply by 2*sigma**4: -n*sigma**2 + sum_i (x_i-mu)**2 = 0
# sigma**2_MLE = 1/n * sum_i (x_i-mu_MLE)**2
#              = ((1-4)**2 + (3-4)**2 + (5-4)**2 + (7-4)**2)/4
#              = (9 + 1 + 1 + 9)/4 = 20/4 = 5
# note: the MLE divides by n (biased), not n-1
def gaussian_mle(data: list | None =None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    if data is None:
        data = [1, 3, 5, 7]
    data_array = np.asarray(data, dtype=float)

    mean_mle = np.mean(data_array)
    variance_mle = np.mean((data_array - mean_mle)**2)
    return mean_mle, variance_mle


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
#
# Likelihood x|mu ~ N(mu, sigma**2) with sigma**2 = 4
# prior mu ~ N(mu_0, tau**2) with mu_0 = 0, tau**2 = 1
# P(mu|x) proportional to P(x|mu)*P(mu)                                          Bayes rule
# log P(mu|x) = -(x-mu)**2/(2*sigma**2) - (mu-mu_0)**2/(2*tau**2) + const
# d/dmu = (x-mu)/sigma**2 - (mu-mu_0)/tau**2 = 0
# x/sigma**2 + mu_0/tau**2 = mu*(1/sigma**2 + 1/tau**2)
# mu_MAP = (x/sigma**2 + mu_0/tau**2) / (1/sigma**2 + 1/tau**2)
#        = (5/4 + 0/1) / (1/4 + 1/1)
#        = (5/4) / (5/4)
#        = 1
# second derivative -(1/sigma**2 + 1/tau**2) < 0, so this is a maximum
# the prior (variance 1) is more confident than the observation (variance 4),
# so the estimate is pulled strongly from x=5 toward the prior mean 0
def gaussian_map(
    x: float=5.0, observation_variance: float=4.0,
    prior_mean: float=0.0, prior_variance: float=1.0
) -> float:
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    observation_precision = 1 / observation_variance
    prior_precision = 1 / prior_variance
    mu_map = (x * observation_precision + prior_mean * prior_precision) / (
        observation_precision + prior_precision
    )
    return mu_map


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Var(X) = E[(X-E[X])**2]                       definition of variance
# Var(X) = E[X**2 - 2XE[X] + E[X]**2]           FOIL
# Var(X) = E[X**2] - 2E[X]E[X] + E[X]**2        linearity of expection
# Var(X) = E[X**2] - 2E[X]**2 + E[X]**2
# Var(X) = E[X**2] - E[X]**2
# Hand-written proof submitted via canvas


# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Hand-written proof submitted via canvas


# P3 Q2: Describe the Gaussian generator and observed correlation.
# Samples are drawn from a 2D multivariate normal N(mean, Sigma) with
# mean = [2, -1] and Sigma = [[3, 1.2],[1.2, 1]], using a seeded np.random.default_rng
# so the dataset is reproducible.
#
# The off-diagonal 1.2 is Cov(x1, x2), so the features are positively correlated:
# Cov(x1, x2) / sqrt(Var(x1)*Var(x2)) = 1.2 / sqrt(3*1) ~= 0.69
# The scatter plot shows this as a tilted ellipse: points with large x1 tend to have large x2.
# The cloud is wider along x1 (variance 3) than along x2 (variance 1).
# The sample correlation printed in main should be close to 0.69 (sampling noise for n=500).
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    if mean is None:
        mean = [2, -1]
    if covariance is None:
        covariance = [[3, 1.2], [1.2, 1]]

    # copy so the caller's arrays are never mutated
    mean = np.array(mean, dtype=float)
    covariance = np.array(covariance, dtype=float)

    # ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=n_samples)


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
#
# 1. Center: X_c = X - mean. Var(u^T X) = E[(u^T(X - E[X]))**2] from P3 Q1 is defined
#    about the mean, so the mean is subtracted first.
# 2. Covariance: Sigma = X_c^T X_c / (n-1). This is the sample estimate of
#    E[(X - E[X])(X - E[X])^T], so the variance of the data projected onto unit u is u^T Sigma u.
# 3. Eigendecomposition: maximize u^T Sigma u subject to u^T u = 1.
#    Lagrangian: u^T Sigma u - lambda(u^T u - 1); gradient 2 Sigma u - 2 lambda u = 0 => Sigma u = lambda u.
#    So the best directions are the eigenvectors of Sigma, and the variance along
#    each one is u^T Sigma u = lambda u^T u = lambda.
# 4. Select components: sort eigenvalues descending. The first column of `components` is the
#    direction of maximum variance (PC1); PC2 is orthogonal to it.
#    scores = X_c @ components are the coordinates of each point in the PC basis.
#
# Plot: the PC1 arrow points along the long axis of the tilted ellipse (up and to the
# right, ~25 degrees above the x1 axis for the true Sigma). PC2 is perpendicular along the short axis.
# Arrow lengths are scaled by sqrt(eigenvalue), i.e. one standard deviation along each direction.
# ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
def pca(X: np.ndarray) -> dict:
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    n = X.shape[0]

    # 1. center (creates a new array, X is not mutated)
    mean = X.mean(axis=0)
    centered = X - mean

    # 2. covariance with divisor n-1
    covariance = centered.T @ centered / (n - 1)

    # 3. eigendecomposition. eigh is for symmetric matrices and returns orthonormal
    # eigenvectors as columns, also for tied eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # rank-deficient data can give tiny negative eigenvalues from rounding; variance is >= 0
    eigenvalues = np.clip(eigenvalues, 0, None)

    # 4. eigh returns ascending order, so flip to descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]

    scores = centered @ components

    return {
        "mean": mean,
        "centered": centered,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
#
# The fraction of variance explained by PC_i is lambda_i / sum_j lambda_j, since the total
# variance is trace(Sigma) = sum of eigenvalues.
# For the true Sigma = [[3,1.2],[1.2,1]]: trace = 4, det = 3 - 1.44 = 1.56
# lambda = 2 +- sqrt(4 - 1.56) = 2 +- 1.562 => 3.562, 0.438
# fractions ~= 3.562/4 = 0.89 and 0.438/4 = 0.11 (sample values printed in main are close to this)
#
# So PC1 alone keeps about 89% of the variance. Projecting the 2D data onto PC1
# halves the dimensionality while losing only ~11% of the variance, because the two
# correlated features mostly carry the same information.
# Tradeoff: keeping fewer components gives a smaller, less noisy, computationally cheaper
# representation that is easier to visualize, but the variance in the dropped components is
# lost and the reconstruction error grows.
# In my personal opinion, since the general rule for statistically significant is 5-10%
# removing PC2 removes too much data and given that this is 2D data with only 500 samples
# there isn't any data visualization benefits and the memory improvements are minor for
# this specific data.
def explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
#
# Left panel (Q2/Q3): scatter of the samples with the mean marked and both principal directions
# drawn from the mean, each scaled to one standard deviation (sqrt(eigenvalue)).
# The tilted elliptical cloud shows the positive correlation. PC1 lies along the long axis
# and PC2 along the short axis, at right angles to each other.
# Right panel (Q4): bar chart of the explained variance fraction per component with the
# cumulative fraction shown as a line. PC1 is most of the bar total, which shows that a 1D
# projection keeps most of the structure.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    X = np.asarray(X, dtype=float)
    mean = result["mean"]
    eigenvalues = result["eigenvalues"]
    components = result["components"]
    fractions = explained_variance(eigenvalues)

    fig, (ax_data, ax_var) = plt.subplots(1, 2, figsize=(12, 5))

    # samples and principal directions
    ax_data.scatter(X[:, 0], X[:, 1], s=10, alpha=0.4, label="samples")
    ax_data.scatter(*mean, color="black", marker="x", s=80, label="mean")
    colors = ["tab:red", "tab:green"]
    for i in range(components.shape[1]):
        direction = components[:, i] * np.sqrt(eigenvalues[i])
        ax_data.annotate(
            "",
            xy=mean + direction,
            xytext=mean,
            arrowprops={"arrowstyle":"->", "color":colors[i], "lw":2.5}
        )
        # invisible line so the arrow gets a legend entry
        ax_data.plot([], [], color=colors[i], lw=2.5,
                     label=f"PC{i + 1} (lambda={eigenvalues[i]:.2f})")
    ax_data.set_xlabel("x1")
    ax_data.set_ylabel("x2")
    ax_data.set_title("Samples and principal directions (1 std. dev.)")
    ax_data.set_aspect("equal", adjustable="datalim")
    ax_data.legend()
    ax_data.grid(True, alpha=0.25)

    # explained variance fractions
    labels = [f"PC{i + 1}" for i in range(len(fractions))]
    bars = ax_var.bar(labels, fractions, color=colors[:len(fractions)], alpha=0.8,
                      label="per component")
    ax_var.bar_label(bars, labels=[f"{f:.1%}" for f in fractions])
    ax_var.plot(labels, np.cumsum(fractions), color="black", marker="o",
                label="cumulative")
    ax_var.set_xlabel("principal component")
    ax_var.set_ylabel("fraction of total variance explained")
    ax_var.set_ylim(0, 1.1)
    ax_var.set_title("Explained variance")
    ax_var.grid(True, axis="y", alpha=0.25)
    ax_var.legend()

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
#
# Tasks Completed:
# Part 1:
# - Calculated Linear Algebra transformations on default matrices in README
# - Code to implement Linear Algebra transformations on any matrix
# - Proof for trace(AB) = trace(BA)
#
# Part 2:
# - Calculated probabilities based on default probabilities in README
# - Code to calculate probabilties on any provided probability
# - Proof for VAR[X] = E[X**2] - (E[X])**2
#
# Part 3:
# - Code to generate data and calculate PCA on eigenvalues and eigenvectors
# - matplotlib plots for PCA graphs
# - Proof for Var[u.T*x]=u.T*sigma*u
#
# Misc:
# - Unit tests for functions using default matrix/prob & calculated solutions
#
# External Refrences:
# - CSE 242 Lecture 1, 2, & 3 slides
# - Modern Mathematical Statistics with Applications by Devore and Berk - pg 711
# - Numpy Documentation (exact urls linked as ref comment)
# - Claude AI Support (transcript uploaded to canvas)

if __name__ == "__main__":
    p1_q1 = matrix_operations(
        A=np.array([[3, 4], [2, 1]]),
        B=np.array([[1, 2], [0, -1]])
    )
    print(f"result of matrix_operations on A=[[3,4],[2,1]] B=[[1,2],[0,-1]]: {p1_q1}")
    p1_q2 = eigensystem(np.array([[2,-1],[-1,2]]))
    print(f"eigenpairs for ([[2,-1],[-1,2]]: {p1_q2}")
    p1_q4 = matrix_inverse(C=np.array([[4,7],[2,6]]))
    print(f"inverse of [[4,7],[2,6]]: {p1_q4}")

    p2_q1 = distribution_moments(rate=3.0)
    print(f"c, E[X], Var(X) for P(X=k)=c*3**k/k!: {p2_q1}")
    p2_q2 = event_probabilities(p_a=0.5, p_b=0.3, p_a_given_b=0.6)
    print(f"P(AnB), P(B|A), P(AuB) where P(A)=0.5, P(B)=0.3, P(A|B)=0.6: {p2_q2}")
    p2_q3 = coin_estimates(heads=3, tosses=5)
    print(f"MLE, MAP for H,H,T,H,T with prior 2*theta: {p2_q3}")
    p2_q4 = gaussian_mle(data=[1, 3, 5, 7])
    print(f"Gaussian mean MLE, variance MLE for [1,3,5,7]: {p2_q4}")
    p2_q5 = gaussian_map(x=5.0, observation_variance=4.0, prior_mean=0.0, prior_variance=1.0)
    print(f"Gaussian MAP for x=5, sigma**2=4, prior N(0,1): {p2_q5}")

    X = generate_data(n_samples=500, seed=0)
    print(f"sample correlation of generated data: {np.corrcoef(X.T)[0, 1]:.3f}")
    p3_q3 = pca(np.array(X))
    print(f"PCA eigenvalues: {p3_q3['eigenvalues']}")
    print(f"PCA components (columns): {p3_q3['components']}")
    p3_q4 = explained_variance(p3_q3["eigenvalues"])
    print(f"explained variance fractions: {p3_q4}")

    fig = make_plots(X, p3_q3)
    fig.savefig("Assignment1_figures.pdf", format="pdf")
    print("Saved P3 Q2/Q3/Q4 figures to Assignment1_figures.pdf")
