import numpy as np
from simulation import run_simulation, generate_kinetic_energies
from plotting import plot_results, estimate_background_from_sidebands, subtract_background, fit_signal, gaussian, lorentzian
import scipy.stats as stats
M_PI0 = 134.96 #MeV

def main():
    """Funzione principale per eseguire la simulazione."""
    num_samples = 100000
    E_gamma_min = 40.0  # Soglia minima di energia per i fotoni (MeV)
    sigma_theta = 0.08163  # Non più utilizzato direttamente
    
    # --- Parametri di Calibrazione (presi dal testo) e per la risoluzione ---
    a_params = (-5.8e-5, 0.154)       # Parametri per a(E)
    a_prime_params = (8.34e-5, 0.055)  # Parametri per a'(E)
    b0 = -0.3                         # Offset (MeV)
    k =  0                        # 1.6806  <----- Valori calibrati sperimentalmente
    c =  0               # -4.9133
    resolution_params = (k, c)
    adc_channels = (4096, 1024)      # Numero di canali ADC (Assorbitore, Convertitore)

    # --- Definizione dell'accettanza angolare (esempio) ---
    theta_phi_bounds_np = np.array([
        [0, 3.2705, 0.5079, 0.6091],
        [0, 6.3402, 0.9477, 1.1467],
        [0, 8.1301, 1.4434, 1.6982],
        [0, 8.1301, 1.9670, 2.2218],
        [0, 8.1301, 2.4906, 2.7454],
        [0, 8.1301, 3.7926, 3.5378],
        [0, 8.1301, 4.3162, 4.0614],
        [0, 8.1301, 4.8398, 4.5850],
        [0, 6.3402, 5.3355, 5.1365],
        [0, 3.2705, 5.7753, 5.6741],
    ])
    apply_acceptance = True  # Applica l'accettanza angolare

    seed = 42
    np.random.seed(seed)

    # --- Generazione delle energie cinetiche dei pioni (da un istogramma) ---
    pi0_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], dtype=np.float64)
    counts = np.array([5, 5, 12, 14.5, 18, 24, 22, 19, 19, 16.5, 16.5, 6], dtype=np.float64)
    T_pi0_samples = generate_kinetic_energies(pi0_bins, counts, num_samples)

    print("T_pi0_samples: min =", np.min(T_pi0_samples), "MeV")
    print("T_pi0_samples: max =", np.max(T_pi0_samples), "MeV")
    print("T_pi0_samples: mean =", np.mean(T_pi0_samples), "MeV")
    print("T_pi0_samples: std =", np.std(T_pi0_samples), "MeV")

    # --- Esegui la simulazione COMPLETA ---
    masses, angles, photon_energies_lab, accepted_A, accepted_C, pre_detector_energies = run_simulation(
        T_pi0_samples, theta_phi_bounds_np, E_gamma_min, a_params, a_prime_params, b0, resolution_params, adc_channels, M_PI0, apply_acceptance
    )

    # Stampa le energie (fuori da run_simulation, quindi fuori da Numba)
    for i in range(min(10, len(pre_detector_energies))):  # Stampa i primi 10 eventi
        print(f"Evento {i}: E_gamma1 (LAB) = {pre_detector_energies[i, 0]:.2f} MeV, E_gamma2 (LAB) = {pre_detector_energies[i, 1]:.2f} MeV")

    # --- Analisi e Plotting ---
    left_sideband = (50, 65)
    right_sideband = (210, 225)
    mass_bins = np.linspace(0, 300, 151) + 0.0001
    hist_masses, bin_edges_masses = np.histogram(masses, bins=mass_bins)
    m_data = (bin_edges_masses[:-1] + bin_edges_masses[1:]) / 2
    counts_original = hist_masses

    background_estimate = estimate_background_from_sidebands(m_data, counts_original, left_sideband, right_sideband)
    counts_subtracted = subtract_background(counts_original, background_estimate)

    model_function = gaussian
    initial_guess = [10000, 135, 10]
    fit_parameters, fit_errors, chi2_reduced, residuals, m_data_fit = fit_signal(m_data, counts_subtracted, model_function, initial_guess)

    if fit_parameters is not None:
        print(f"Fit Results ({model_function.__name__}):")
        print(f"  Parameters: {fit_parameters}")
        print(f"  Errors: {fit_errors}")
        print(f"  Reduced Chi-squared: {chi2_reduced:.2f}")
        if model_function.__name__ == "gaussian":
            fwhm = 2 * np.sqrt(2 * np.log(2)) * fit_parameters[2]
            fwhm_error = 2 * np.sqrt(2 * np.log(2)) * fit_errors[2]
            print(f"  FWHM: {fwhm:.2f} ± {fwhm_error:.2f} MeV/c^2")
            max_bin_index = np.argmax(hist_masses)
            max_bin_center = (bin_edges_masses[max_bin_index] + bin_edges_masses[max_bin_index + 1]) / 2
            error_max_mass = np.std(masses)
            print(f"Maximum of invariant mass spectrum: {max_bin_center:.2f} MeV/c^2 +/- {error_max_mass:.2f} MeV/c^2")
    else:
        print("Fit failed.")

    variance = np.var(masses)
    skewness = stats.skew(masses)
    print(f"Invariant mass spectrum variance: {variance:.2f} (MeV/c^2)^2")
    print(f"Invariant mass spectrum skewness: {skewness:.2f}")

    plot_results(masses, angles, photon_energies_lab, T_pi0_samples, pi0_bins, accepted_A, accepted_C, fit_parameters=fit_parameters, fit_errors=fit_errors, chi2_reduced=chi2_reduced, left_sideband=left_sideband, right_sideband=right_sideband, background_estimate=background_estimate, counts=counts_original, counts_subtracted=counts_subtracted, model_name=model_function.__name__, residuals=residuals, m_data_fit=m_data_fit)

if __name__ == "__main__":
    main()
