import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import numpy as np

def gaussian(m, A, m0, sigma):
    """Funzione gaussiana."""
    return A * np.exp(-(m - m0)**2 / (2 * sigma**2))

def lorentzian(m, A, m0, Gamma):
    """Funzione lorentziana."""
    return A * (Gamma / 2)**2 / ((m - m0)**2 + (Gamma / 2)**2)

def fit_signal(m_data, counts, model_function, initial_guess):
    """Esegue il fit dei dati usando una funzione modello specificata."""
    try:
        popt, pcov = curve_fit(model_function, m_data, counts, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        residuals = counts - model_function(m_data, *popt)
        chi_squared = np.sum(residuals**2 / counts, where=counts>0)
        dof = len(m_data) - len(initial_guess)
        chi2_reduced = chi_squared / dof if dof > 0 else np.inf
        return popt, perr, chi2_reduced, residuals, m_data
    except RuntimeError as e:
        print(f"Fit error: {e}")
        return None, None, None, None, None

def estimate_background_from_sidebands(m_data, counts, left_sideband, right_sideband):
    """Stima il fondo usando le bande laterali."""
    left_mask = (m_data >= left_sideband[0]) & (m_data <= left_sideband[1])
    right_mask = (m_data >= right_sideband[0]) & (m_data <= right_sideband[1])

    left_count = np.sum(counts[left_mask])
    right_count = np.sum(counts[right_mask])

    left_width = left_sideband[1] - left_sideband[0]
    right_width = right_sideband[1] - right_sideband[0]

    if left_width <= 0 or right_width <= 0:
        print("Warning: Sideband width is zero or negative.")
        return np.zeros_like(m_data)

    if left_count == 0 and right_count == 0:
        return np.zeros_like(m_data)

    if left_count > 0 and right_count > 0:
        left_mean = np.mean(m_data[left_mask])
        right_mean = np.mean(m_data[right_mask])
        if np.isnan(left_mean) or np.isnan(right_mean):
            return np.zeros_like(m_data)
        background_level = np.interp(m_data, [left_mean, right_mean], [left_count / left_width, right_count / right_width])
    elif left_count > 0:
        background_level = np.full_like(m_data, left_count / left_width)
    elif right_count > 0:
        background_level = np.full_like(m_data, right_count / right_width)
    else:
        return np.zeros_like(m_data)

    return background_level

def subtract_background(counts, background_estimate):
    """Sottrae il fondo stimato dai conteggi."""
    background_estimate = np.nan_to_num(background_estimate, nan=0.0)
    result = counts - background_estimate
    return np.maximum(0, result)

def plot_results(masses, angles, photon_energies_lab, T_pi0_samples, pi0_bins, accepted_A, accepted_C, fit_parameters=None, fit_errors=None, chi2_reduced=None, left_sideband=None, right_sideband=None, background_estimate=None, counts=None, counts_subtracted=None, model_name=None, residuals=None, m_data_fit=None):
    """Genera e visualizza i grafici dei risultati della simulazione."""
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    mass_bins = np.linspace(0, 300, 151)
    plt.hist(masses, bins=mass_bins, color='blue', alpha=0.7, label='Invariant Mass Spectrum')

    if fit_parameters is not None:
        m_fit = np.linspace(min(masses), max(masses), 500)
        if model_name == "gaussian":
            fit_curve = gaussian(m_fit, *fit_parameters)
        elif model_name == "lorentzian":
            fit_curve = lorentzian(m_fit, *fit_parameters)
        plt.plot(m_fit, fit_curve, 'r-', label=f'{model_name} Fit (χ²_rid = {chi2_reduced:.2f})')

    plt.xlabel('Invariant Mass (MeV/c^2)')
    plt.ylabel('Counts')
    plt.title('Invariant Mass Spectrum')
    plt.legend()
    ax = plt.gca()
    plt.xlim(0, 270)
