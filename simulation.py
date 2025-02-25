from numba import jit
import numpy as np
from utils import is_accepted, sample_isotropic_angles, boost_lab_frame, calculate_invariant_mass

M_PI0 = 134.9766

@jit(nopython=True)
def detector_energy_smearing(true_energy, a_params, a_prime_params, b0, resolution_params, adc_channels):
    measured_energy_preADC = some_function_to_calculate_energy(true_energy, a_params, a_prime_params, b0, resolution_params)

    if measured_energy_preADC <= 0:
        # Ensure that all branches return the same number of elements with the same types
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    reconstructed_energy, A_simulated, C_simulated = some_other_function(measured_energy_preADC, adc_channels)
    
    a_E = some_calculation_a_E(reconstructed_energy)
    a_prime_E = some_calculation_a_prime_E(reconstructed_energy)
    converter_fraction = some_calculation_converter_fraction(reconstructed_energy)
    energy_absorber = some_calculation_energy_absorber(reconstructed_energy)
    energy_converter = some_calculation_energy_converter(reconstructed_energy)
    expected_A = some_calculation_expected_A(A_simulated)
    expected_C = some_calculation_expected_C(C_simulated)

    return (reconstructed_energy, A_simulated, C_simulated, true_energy, a_E, a_prime_E, converter_fraction, energy_absorber, energy_converter, expected_A, expected_C)

# Example utility functions used within detector_energy_smearing (replace with actual implementations)
def some_function_to_calculate_energy(true_energy, a_params, a_prime_params, b0, resolution_params):
    # Placeholder implementation
    return true_energy * 0.9

def some_other_function(measured_energy_preADC, adc_channels):
    # Placeholder implementation
    return measured_energy_preADC * 0.95, 1.0, 1.0

def some_calculation_a_E(reconstructed_energy):
    # Placeholder implementation
    return reconstructed_energy * 0.1

def some_calculation_a_prime_E(reconstructed_energy):
    # Placeholder implementation
    return reconstructed_energy * 0.05

def some_calculation_converter_fraction(reconstructed_energy):
    # Placeholder implementation
    return reconstructed_energy * 0.8

def some_calculation_energy_absorber(reconstructed_energy):
    # Placeholder implementation
    return reconstructed_energy * 0.7

def some_calculation_energy_converter(reconstructed_energy):
    # Placeholder implementation
    return reconstructed_energy * 0.6

def some_calculation_expected_A(A_simulated):
    # Placeholder implementation
    return A_simulated * 1.1

def some_calculation_expected_C(C_simulated):
    # Placeholder implementation
    return C_simulated * 1.2
    # return reconstructed_energy, A_simulated, C_simulated, true_energy, a_E, a_prime_E, converter_fraction, energy_absorber, energy_converter, expected_A, expected_C
    

@jit(nopython=True)
def simulate_pi0_decay(E_pi0, num_samples):
    """Simula il decadimento del π⁰ nel sistema di riferimento del centro di massa."""
    cos_theta_cm, phi_cm = sample_isotropic_angles(num_samples)
    E_gamma_cms = M_PI0 / 2
    photon1_cms = np.empty((num_samples, 4))
    photon2_cms = np.empty((num_samples, 4))

    photon1_cms[:, 0] = E_gamma_cms
    photon1_cms[:, 1] = E_gamma_cms * np.sqrt(1 - cos_theta_cm**2) * np.cos(phi_cm)
    photon1_cms[:, 2] = E_gamma_cms * np.sqrt(1 - cos_theta_cm**2) * np.sin(phi_cm)
    photon1_cms[:, 3] = E_gamma_cms * cos_theta_cm

    photon2_cms[:, 0] = E_gamma_cms
    photon2_cms[:, 1] = -photon1_cms[:, 1]
    photon2_cms[:, 2] = -photon1_cms[:, 2]
    photon2_cms[:, 3] = -photon1_cms[:, 3]

    return photon1_cms, photon2_cms, E_gamma_cms

@jit(nopython=True)
def apply_detector_effects(photons_lab, theta_phi_bounds, E_gamma_min, a_params, a_prime_params, b0, resolution_params, adc_channels, apply_acceptance):
    """Applica gli effetti del rivelatore e restituisce i dati degli eventi accettati."""
    num_samples = photons_lab.shape[0] // 2
    accepted_indices = np.empty(num_samples, dtype=np.int64)
    accepted_energies = np.empty((num_samples, 2), dtype=np.float64)
    accepted_A = np.empty((num_samples, 2), dtype=np.int64)
    accepted_C = np.empty((num_samples, 2), dtype=np.int64)
    count = 0

    for i in range(num_samples):
        photon1_lab = photons_lab[2 * i]
        photon2_lab = photons_lab[2 * i + 1]

        photon_magnitudes1 = np.sqrt(np.sum(photon1_lab[1:] ** 2))
        photon_magnitudes2 = np.sqrt(np.sum(photon2_lab[1:] ** 2))
        theta_lab1 = np.arccos(photon1_lab[3] / photon_magnitudes1) if photon_magnitudes1 != 0 else 0.0
        theta_lab2 = np.arccos(photon2_lab[3] / photon_magnitudes2) if photon_magnitudes2 != 0 else 0.0
        phi_lab1 = np.arctan2(photon1_lab[2], photon1_lab[1]) % (2 * np.pi)
        phi_lab2 = np.arctan2(photon2_lab[2], photon2_lab[1]) % (2 * np.pi)

        accept1 = is_accepted(theta_lab1, phi_lab1, theta_phi_bounds) if apply_acceptance else True
        accept2 = is_accepted(theta_lab2, phi_lab2, theta_phi_bounds) if apply_acceptance else True

        if accept1:
            E_gamma_lab_res1, A1, C1, true_energy1, a_E1, a_prime_E1, converter_fraction1, energy_absorber1, energy_converter1, expected_A1, expected_C1 = detector_energy_smearing(photon1_lab[0], a_params, a_prime_params, b0, resolution_params, adc_channels)
        else:
            E_gamma_lab_res1, A1, C1 = None, None, None

        if accept2:
            E_gamma_lab_res2, A2, C2, true_energy2, a_E2, a_prime_E2, converter_fraction2, energy_absorber2, energy_converter2, expected_A2, expected_C2 = detector_energy_smearing(photon2_lab[0], a_params, a_prime_params, b0, resolution_params, adc_channels)
        else:
            E_gamma_lab_res2, A2, C2 = None, None, None

        if accept1 and accept2 and E_gamma_lab_res1 is not None and E_gamma_lab_res2 is not None and E_gamma_lab_res1 >= E_gamma_min and E_gamma_lab_res2 >= E_gamma_min:
            accepted_indices[count] = i
            accepted_energies[count, 0] = E_gamma_lab_res1
            accepted_energies[count, 1] = E_gamma_lab_res2
            accepted_A[count, 0] = A1
            accepted_A[count, 1] = A2
            accepted_C[count, 0] = C1
            accepted_C[count, 1] = C2
            count += 1

    return accepted_indices[:count], accepted_energies[:count], accepted_A[:count], accepted_C[:count]

@jit(nopython=True)
def run_simulation(T_pi0_samples, theta_phi_bounds, E_gamma_min, a_params, a_prime_params, b0, resolution_params, adc_channels, m_pi0, apply_acceptance):
    """Esegue l'intera simulazione."""
    num_samples = len(T_pi0_samples)
    E_pi0 = T_pi0_samples + m_pi0
    p_pi0 = np.sqrt(E_pi0**2 - m_pi0**2)
    beta = p_pi0 / E_pi0
    beta = np.clip(beta, -0.999999, 0.999999)

    photon1_cms, photon2_cms, E_gamma_cms = simulate_pi0_decay(E_pi0, num_samples)
    photon1_lab = boost_lab_frame(photon1_cms, beta)
    photon2_lab = boost_lab_frame(photon2_cms, beta)

    photons_lab = np.empty((2 * num_samples, 4))
    photons_lab[::2] = photon1_lab
    photons_lab[1::2] = photon2_lab

    pre_detector_energies = np.empty((num_samples, 2), dtype=np.float64)
    for i in range(num_samples):
        pre_detector_energies[i,0] = photons_lab[2*i,0]
        pre_detector_energies[i,1] = photons_lab[2*i + 1, 0]

    accepted_indices, accepted_calib_energies, accepted_A, accepted_C = apply_detector_effects(photons_lab, theta_phi_bounds, E_gamma_min, a_params, a_prime_params, b0, resolution_params, adc_channels, apply_acceptance)

    num_accepted = accepted_indices.shape[0]
    accepted_invariant_masses = np.empty(num_accepted, dtype=np.float64)
    accepted_opening_angles = np.empty(num_accepted, dtype=np.float64)
    accepted_photon_energies_lab = np.empty(2 * num_accepted, dtype=np.float64)

    for i in range(num_accepted):
        idx = accepted_indices[i]
        photon1_lab = photons_lab[2 * idx]
        photon2_lab = photons_lab[2 * idx + 1]
        E_gamma_lab_res1 = accepted_calib_energies[i, 0]
        E_gamma_lab_res2 = accepted_calib_energies[i, 1]

        photon_magnitudes1 = np.sqrt(np.sum(photon1_lab[1:] ** 2))
        photon_magnitudes2 = np.sqrt(np.sum(photon2_lab[1:] ** 2))
        cos_opening_angle = (photon1_lab[1] * photon2_lab[1] + photon1_lab[2] * photon2_lab[2] + photon1_lab[3] * photon2_lab[3]) / (photon_magnitudes1 * photon_magnitudes2)
        cos_opening_angle = max(-1.0, min(cos_opening_angle, 1.0))
        opening_angle = np.arccos(cos_opening_angle)

        accepted_invariant_masses[i] = calculate_invariant_mass(E_gamma_lab_res1, E_gamma_lab_res2, opening_angle)
        accepted_opening_angles[i] = opening_angle
        accepted_photon_energies_lab[2 * i] = E_gamma_lab_res1
        accepted_photon_energies_lab[2 * i + 1] = E_gamma_lab_res2

    return accepted_invariant_masses, accepted_opening_angles, accepted_photon_energies_lab, accepted_A, accepted_C, pre_detector_energies

@jit(nopython=True)
def generate_kinetic_energies(bins, counts, num_samples):
    """Campiona le energie cinetiche dei pioni da un istogramma fornito."""
    probabilities = counts / np.sum(counts)
    cumulative_prob = np.cumsum(probabilities)
    random_values = np.random.random(num_samples)
    sampled_bins = np.searchsorted(cumulative_prob, random_values)
    sampled_bins = np.clip(sampled_bins, 0, len(bins) - 2)
    bin_width = np.diff(bins)
    random_shift = np.random.uniform(0, 1, len(sampled_bins)) * bin_width[sampled_bins]
    T_pi0_samples = bins[sampled_bins] + random_shift
    return T_pi0_samples
