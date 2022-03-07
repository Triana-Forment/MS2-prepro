import sys
import numpy as np
from pyteomics import mgf

def mass_diff(mz1, mz2, mode_is_da):
    """
    Calculate the mass difference(s).
    Parameters
    ----------
    mz1
        First m/z value(s).
    mz2
        Second m/z value(s).
    mode_is_da : bool
        Mass difference in Dalton (True) or in ppm (False).
    Returns
    -------
        The mass difference(s) between the given m/z values.
    """
    return mz1 - mz2 if mode_is_da else (mz1 - mz2) / mz2 * 10 ** 6

def _get_non_precursor_peak_mask(
    mz: np.ndarray,
    pep_mass: float,
    max_charge: int,
    isotope: int,
    fragment_tol_mass: float,
    fragment_tol_mode: str,
) -> np.ndarray:
    """
    Get a mask to remove peaks that are close to the precursor mass peak (at
    different charges and isotopes).
    JIT helper function for `MsmsSpectrum.remove_precursor_peak`.
    Parameters
    ----------
    mz : np.ndarray
        The mass-to-charge ratios of the spectrum fragment peaks.
    pep_mass : float
        The mono-isotopic mass of the uncharged peptide.
    max_charge : int
        The maximum precursor loss charge.
    isotope : int
        The number of isotopic peaks to be checked.
    fragment_tol_mass : float
            Fragment mass tolerance around the precursor mass to remove the
            precursor peak.
    fragment_tol_mode : {'Da', 'ppm'}
            Fragment mass tolerance unit. Either 'Da' or 'ppm'.
    Returns
    -------
    np.ndarray
        Index mask specifying which peaks are retained after precursor peak
        filtering.
    """
    remove_mz = []
    for charge in range(max_charge, 0, -1):
        for iso in range(isotope + 1):
            remove_mz.append((pep_mass + iso) / charge + 1.0072766)

    mask = np.full_like(mz, True, np.bool_)
    mz_i = remove_i = 0
    while mz_i < len(mz) and remove_i < len(remove_mz):
        md = mass_diff(
            mz[mz_i], remove_mz[remove_i], fragment_tol_mode == "Da"
        )
        if md < -fragment_tol_mass:
            mz_i += 1
        elif md > fragment_tol_mass:
            remove_i += 1
        else:
            mask[mz_i] = False
            mz_i += 1

    return mask

def remove_precursor_peak(
        input_file,
        output_file,
        fragment_tol_mode: str,
	fragment_tol_mass: float,
        isotope: int = 0,
    ):
        """
        Remove fragment peak(s) close to the precursor m/z.
        Parameters
        ----------
        input_file : path to the mgf file to remove precursor peak
        output_file : name of the mgf file with the precursor peak removed
        fragment_tol_mass : float
            Fragment mass tolerance around the precursor mass to remove the
            precursor peak.
        fragment_tol_mode : {'Da', 'ppm'}
            Fragment mass tolerance unit. Either 'Da' or 'ppm'.
        isotope : int
            The number of precursor isotopic peaks to be checked (the default
            is 0 to check only the mono-isotopic peaks).
        Returns
        -------
        Nothing, it writes an mgf file with the name set in 'output_file'        
        """
        reader = mgf.read(input_file)
        for spectrum in reader:
            mz = spectrum["m/z array"]
            intensity = spectrum["intensity array"]
            precursor_mz = spectrum["params"]["pepmass"][0]
            precursor_charge = int(str(spectrum["params"]["charge"][0])[:1]) 
            neutral_mass = (precursor_mz - 1.0072766) * precursor_charge
            peak_mask = _get_non_precursor_peak_mask(mz, neutral_mass, precursor_charge, isotope, fragment_tol_mass,
            fragment_tol_mode)
            spectrum["m/z array"] = mz[peak_mask]
            spectrum["intensity array"] = intensity[peak_mask]
            mgf.write((spectrum,), output_file, file_mode='a')


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    remove_precursor_peak(*sys.argv[1:4], int(sys.argv[4]))

