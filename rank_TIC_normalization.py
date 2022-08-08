import sys
import numpy as np
from pyteomics import mgf

def rank_TIC_normalization(input_file, output_file):

    """
    Normalizes spectra intensities taking into account their rank and the total ion current.
    
    Parameters
    ----------
        input_file : path to the mgf file to normalize
        output_file : name of the mgf file normalized
        
    Returns
    -------
        Nothing, it writes an mgf file with the intensities normalized in the 'output_file'
    """

    reader = mgf.MGF(input_file)
    for spectrum in reader:
        spec_len = len(spectrum["intensity array"]) # Number of ions in MS2 spectrum
        TIC = np.sum(spectrum["intensity array"]) # Total ion current
        original_order = spectrum["intensity array"].argsort().argsort() # get index to restore order
        sorted_int = np.sort(spectrum["intensity array"])
        normalized = np.zeros([spec_len,])
        for index in range(spec_len):
            normalized[index] = np.sum(sorted_int[:index+1])/TIC
        spectrum["intensity array"] = normalized[original_order]
        mgf.write((spectrum,), output_file, file_mode='a')


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    rank_TIC_normalization(*sys.argv[1:])

