from numba import jit
import numpy as np

@jit(nopython=True)
def is_accepted(theta, phi, bounds):
    """Verifica se un dato angolo è dentro l'accettanza del rivelatore."""
    for i in range(bounds.shape[0]):
        phi_min, phi_max, theta_min, theta_max = bounds[i]
        if (phi_min <= phi <= phi_max) and (theta_min <= theta <= theta_max):
            return True
    return False

@jit(nopython=True)
def sample_isotropic_angles(num_samples):
    """Genera angoli isotropi (cos(theta) e phi)."""
    cos_theta = np.random.uniform(-1, 1, num_samples)
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    return cos_theta, phi

@jit(nopython=True)
def boost_lab_frame(vectors, beta):
    """Applica il boost di Lorentz ai quadrivettori."""
    gamma = 1 / np.sqrt(1 - beta**2)
    gamma = np.where(np.isinf(gamma), 1e10, gamma)

    boosted_vectors = np.empty_like(vectors)
    boosted_vectors[:, 0] = gamma * (vectors[:, 0] - beta * vectors[:, 3])
    boosted_vectors[:, 1] = vectors[:, 1]
    boosted_vectors[:, 2] = vectors[:, 2]
    boosted_vectors[:, 3] = gamma * (vectors[:, 3] - beta * vectors[:, 0])
    return boosted_vectors

@jit(nopython=True)
def calculate_invariant_mass(E1, E2, opening_angle):
    """Calcola la massa invariante."""
    if np.isnan(E1) or np.isnan(E2) or np.isnan(opening_angle):
        return np.nan
    if E1 <= 0 or E2 <= 0:
        return np.nan
    return 2.0 * np.sqrt(E1 * E2) * np.sin(opening_angle / 2.0)
