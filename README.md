# MS2-prepro
## Python scripts and Jupyter notebooks for MS/MS spectra pre-processing

### Remove precursor peak 
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
        
Usage example: python remove_precursor_peak.py input.mgf output.mgf Da 10

### Rank TIC normalization
Normalizes spectra intensities by computing the sum of all raw intensities I(pk) in P that are smaller than I(pi) and dividing this sum by the total ion current (TIC) of P. So,the highest peak I(pj) equals 1. All other peaks pi are scaled relative to the total ion current of all peaks.
    
    Parameters
    ----------
        input_file : path to the mgf file to normalize
        output_file : name of the mgf file normalized
        
    Returns
    -------
        Nothing, it creates an mgf file with the intensities normalized in 'output_file'
        
Usage example: python rank_TIC_normalization.py input.mgf output.mgf
