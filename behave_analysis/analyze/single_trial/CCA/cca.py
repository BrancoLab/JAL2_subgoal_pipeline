"""Implementation of Canonical Correlation Analysis (CCA) for aligning two datasets in a shared latent space.

Author: Laurence Freeman
Paper citation: Gallego et al 2020 Long-term stability of cortical population dynamics underlying consistent behavior

"""

import sys

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- Config ---------------


@dataclass
class CCAConfig:
    pca_components: int = 10


# ---------------- PCA  -----------------


def pca_transform(matrix1: np.array, matrix2: np.array, config: CCAConfig) -> tuple[np.array, np.array]:
    """Transform matrices using PCA. This creates an emperical latent space for each matrix.

    Args:
    -- matrix1: np.array, shape (nsamples, nfeatures)
    -- matrix2: np.array, shape (nsamples, nfeatures)

    Returns:
    -- A1: np.array, shape (ncomp, nsamples)
    -- A2: np.array, shape (ncomp, nsamples)
    """
    ncomp = config.pca_components
    pca = PCA(n_components=ncomp)
    A1 = pca.fit_transform(matrix1)
    pca = PCA(n_components=ncomp)
    A2 = pca.fit_transform(matrix2)
    assert A1.shape[1] == ncomp, "Second dimension is expected to be PC components"
    assert A2.shape[1] == ncomp, "Second dimension is expected to be PC components"
    A1, A2 = align_matrices_to_Gallego_notation(A1, A2)
    return A1, A2


# ---------------- CCA   ------------------


def qr_decomposition(A: np.array) -> tuple[np.array, np.array]:
    """Decompose a matrix A into an orthogonal matrix Q and an upper triangular matrix R.
    Where R is the transformation matrix that transforms the columns of Q to the original space of A.
    And the columns of Q form an ONB across the principal components.

    Args:
    -- A: np.array, shape (ncomp, nsamples)

    Returns:
    -- Q: np.array, shape (ncomp, ncomp)"""
    Q, R = np.linalg.qr(A.T)
    assert is_invertible(R), "R is not invertible."
    # assert is_ONB(Q), "Q is not an orthonormal basis."
    return Q, R


def inner_product(Q1: np.array, Q2: np.array, config: CCAConfig) -> np.array:
    """Find the inner product of two matrices Q1 and Q2 to match the similarity of the linear latent spaces.

    Args:
    -- Q1: np.array, shape (nsamples, ncomp)"""
    q1s = Q1.T.shape[1]
    q2s = Q2.shape[0]
    if not q1s == q2s:
        # We must subsample the matrix to match the dimensions
        if q1s > q2s:
            Q1 = Q1[:q2s, :]
        else:
            Q2 = Q2[:q1s, :]
    assert Q1.T.shape[1] == Q2.shape[0], "The dimensions of the matrices are not compatible for a matrix mult the resample failed."
    inner_product = Q1.T @ Q2
    assert inner_product.shape == (config.pca_components, config.pca_components), "The shape of the inner product is incorrect it must be square."
    return inner_product


def svd_inner_product(inner_product: np.array) -> tuple[np.array, np.array, np.array]:
    """Compute the singular value decomposition of the inner product."""
    U, S, V = np.linalg.svd(inner_product)
    V = V.T
    return U, S, V


def project_latent_space(A, R, RHS):
    """Project the latent space to the true manifold.

    Args:
    -- A: np.array, shape (nsamples, ncomp)
    -- R: np.array, shape (ncomp, ncomp)
    -- RHS: either U or V, np.array, shape (ncomp, ncomp)"""
    M_i = np.linalg.inv(R) @ RHS
    return A.T @ M_i


def cca_transform(A1: np.array, A2: np.array, config: CCAConfig) -> tuple[np.array, np.array]:
    """Transform two matrices to a shared axis using Canonical Correlation Analysis (CCA).

    Args:
    -- A1: np.array, shape (nsamples, ncomp)
    -- A2: np.array, shape (nsamples, ncomp)

    Returns:
    -- A1_mapped: np.array, shape (nsamples, ncomp)
    -- A2_mapped: np.array, shape (nsamples, ncomp)"""
    # PCA transform
    A1, A2 = pca_transform(A1, A2, config=config)  # (ncomp, nsamples)

    # QR decomposition
    Q1, R1 = qr_decomposition(A1)  # Q (samples, ncomp)
    Q2, R2 = qr_decomposition(A2)  # Q (samples, ncomp)

    print("Q1 shape: ", Q1.shape)
    print("Q2 shape: ", Q2.shape)

    # Inner product
    inner_prod = inner_product(Q1, Q2, config=config)
    # SVD of inner product
    U, S, V = svd_inner_product(inner_prod)
    # Project latent space
    A1_mapped = project_latent_space(A1, R1, U)
    A2_mapped = project_latent_space(A2, R2, V)
    
    assert A1_mapped.shape == A1.T.shape, "The shape of the mapped matrix is incorrect."
    assert A2_mapped.shape == A2.T.shape, "The shape of the mapped matrix is incorrect."
    
    return A1, A2, A1_mapped, A2_mapped


# ---------------- Utils ----------------


def align_matrices_to_Gallego_notation(A1: np.array, A2: np.array) -> tuple[np.array, np.array]:
    """Conduct transformation of matrices to align with Gallego notation."""
    return A1.T, A2.T


def is_invertible(A):
    """Check if a matrix is invertible.

    A high condition number indicates that the matrix is nearly singular (i.e., close to being non-invertible),
    while a low condition number indicates that the matrix is well-conditioned (i.e., far from being singular).

    sys.float_info.epsilon is the smallest representable positive number such that 1.0 + epsilon != 1.0.

    If the condition number is less than 1/epsilon, the matrix is considered to be invertible.
    """
    if np.linalg.cond(A) < 1 / sys.float_info.epsilon:
        return True
    else:
        return False


def is_ONB(Q):
    """Check if a matrix is an orthonormal basis."""
    if np.allclose(np.eye(Q.shape[0]), Q @ Q.T):
        return True
    else:
        return False


if __name__ == "__main__":
    # Run synthetic data to show the CCA is working

    def generate_smiley(nsamples=600):
        """Generate a smiley face dataset."""
        np.random.seed(42)
        # Distribute points evenly
        face_points = nsamples // 2
        eye_points = nsamples // 10
        smile_points = nsamples - (face_points + 2 * eye_points)

        # Generate circle for face outline
        theta = np.linspace(0, 2 * np.pi, face_points)
        x_face = np.cos(theta)
        y_face = np.sin(theta)

        # Generate eyes
        eye_theta = np.linspace(0, 2 * np.pi, eye_points)
        x_eye1 = -0.5 + 0.1 * np.cos(eye_theta)
        y_eye1 = 0.5 + 0.1 * np.sin(eye_theta)
        x_eye2 = 0.5 + 0.1 * np.cos(eye_theta)
        y_eye2 = 0.5 + 0.1 * np.sin(eye_theta)

        # Generate smile
        smile_theta = np.linspace(0, np.pi, smile_points)
        x_smile = 0.6 * np.cos(smile_theta)
        y_smile = -0.6 * np.sin(smile_theta) - 0.2

        # Combine all parts
        x = np.concatenate([x_face, x_eye1, x_eye2, x_smile])
        y = np.concatenate([y_face, y_eye1, y_eye2, y_smile])

        return np.stack((x, y), axis=1)

    # Generate synthetic smiley face data
    features = 100
    nsamples = 600
    nlatent = 10
    cfg = CCAConfig()

    # Synthetic data with smiley faces
    if 0:

        # Generate smiley face points
        smiley = generate_smiley(nsamples)

        # Function to create a transformation matrix that preserves structure
        def create_transformation(scale=1, rotation=0, translation=(0, 0)):
            cos_theta = np.cos(rotation)
            sin_theta = np.sin(rotation)
            tx, ty = translation

            transform = np.array([[scale * cos_theta, -scale * sin_theta, tx], [scale * sin_theta, scale * cos_theta, ty], [0, 0, 1]])

            return transform

        # Create two slightly different transformations
        transform1 = create_transformation(scale=2, rotation=np.pi / 6, translation=(1, 1))
        transform2 = create_transformation(scale=1.5, rotation=-np.pi / 4, translation=(-1, 2))

        # Apply transformations to smiley faces
        smiley_homogeneous = np.column_stack([smiley, np.ones(nsamples)])
        transformed_smiley1 = (smiley_homogeneous @ transform1.T)[:, :2]
        transformed_smiley2 = (smiley_homogeneous @ transform2.T)[:, :2]

        # Add some noise to create additional features
        noise_scale = 0.05
        additional_features1 = np.random.randn(nsamples, features - 2) * noise_scale
        additional_features2 = np.random.randn(nsamples, features - 2) * noise_scale

        # Combine transformed smileys with additional features
        matrix1 = np.hstack([transformed_smiley1, additional_features1])
        matrix2 = np.hstack([transformed_smiley2, additional_features2])

        A1, A2, A1_mapped, A2_mapped = cca_transform(matrix1, matrix2, config=cfg)

    # Synthetic data with sinusoidal patterns
    if 1:
        # Create a time vector
        t = np.linspace(0, 2 * np.pi, nsamples)

        # Generate latent features with sinusoidal patterns and noise
        latent_features = np.array([np.sin(t * (i + 1)) + 0.1 * np.random.randn(nsamples) for i in range(nlatent)]).T

        # Generate transformation matrices
        transformation1 = np.random.rand(nlatent, features)
        transformation2 = transformation1 + np.random.rand(nlatent, features) * 1  # Slightly shifted

        # Generate matrices by applying transformations
        matrix1 = latent_features @ transformation1
        matrix2 = latent_features @ transformation2

        A1, A2, A1_mapped, A2_mapped = cca_transform(matrix1, matrix2, config=cfg)

    # Plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.scatter(matrix1[:, 0], matrix1[:, 1], alpha=0.5)
    plt.title("Matrix 1")
    plt.axis("equal")

    plt.subplot(122)
    plt.scatter(matrix2[:, 0], matrix2[:, 1], alpha=0.5)
    plt.title("Matrix 2")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

    print("Shape of input matrix1:", matrix1.shape)
    print("Shape of input matrix2:", matrix2.shape)

    # Check the shapes
    print(f"A1_mapped shape: {A1_mapped.shape}")
    print(f"A2_mapped shape: {A2_mapped.shape}")
    print(f"A1 shape: {A1.shape}")
    print(f"A2 shape: {A2.shape}")

    # Plotting before alignment
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(A1[0, :], A1[1, :], alpha=0.5, label="Matrix1")  # Plot the first two components of A1 and all samples
    plt.scatter(A2[0, :], A2[1, :], alpha=0.5, label="Matrix2")  # Plot the first two components of A2 and all samples
    plt.title("Before Alignment")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()

    # Plotting after alignment
    plt.subplot(1, 2, 2)
    plt.scatter(A1_mapped[:, 0], A1_mapped[:, 1], alpha=0.5, label="Matrix1 Mapped")
    plt.scatter(A2_mapped[:, 0], A2_mapped[:, 1], alpha=0.5, label="Matrix2 Mapped")
    plt.title("After Alignment")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()

    plt.show()
